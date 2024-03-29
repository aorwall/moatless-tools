# pydata__xarray-3733

| **pydata/xarray** | `009aa66620b3437cf0de675013fa7d1ff231963c` |
| ---- | ---- |
| **No of patches** | 7 |
| **All found context length** | 12316 |
| **Any found context length** | 12316 |
| **Avg pos** | 51.42857142857143 |
| **Min pos** | 21 |
| **Max pos** | 111 |
| **Top file pos** | 1 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/__init__.py b/xarray/__init__.py
--- a/xarray/__init__.py
+++ b/xarray/__init__.py
@@ -17,7 +17,7 @@
 from .core.alignment import align, broadcast
 from .core.combine import auto_combine, combine_by_coords, combine_nested
 from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
-from .core.computation import apply_ufunc, dot, where
+from .core.computation import apply_ufunc, dot, polyval, where
 from .core.concat import concat
 from .core.dataarray import DataArray
 from .core.dataset import Dataset
@@ -65,6 +65,7 @@
     "open_mfdataset",
     "open_rasterio",
     "open_zarr",
+    "polyval",
     "register_dataarray_accessor",
     "register_dataset_accessor",
     "save_mfdataset",
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -1306,3 +1306,35 @@ def where(cond, x, y):
         dataset_join="exact",
         dask="allowed",
     )
+
+
+def polyval(coord, coeffs, degree_dim="degree"):
+    """Evaluate a polynomial at specific values
+
+    Parameters
+    ----------
+    coord : DataArray
+        The 1D coordinate along which to evaluate the polynomial.
+    coeffs : DataArray
+        Coefficients of the polynomials.
+    degree_dim : str, default "degree"
+        Name of the polynomial degree dimension in `coeffs`.
+
+    See also
+    --------
+    xarray.DataArray.polyfit
+    numpy.polyval
+    """
+    from .dataarray import DataArray
+    from .missing import get_clean_interp_index
+
+    x = get_clean_interp_index(coord, coord.name)
+
+    deg_coord = coeffs[degree_dim]
+
+    lhs = DataArray(
+        np.vander(x, int(deg_coord.max()) + 1),
+        dims=(coord.name, degree_dim),
+        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
+    )
+    return (lhs * coeffs).sum(degree_dim)
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -95,3 +95,30 @@ def func(x, window, axis=-1):
     # crop boundary.
     index = (slice(None),) * axis + (slice(drop_size, drop_size + orig_shape[axis]),)
     return out[index]
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    import dask.array as da
+
+    lhs_da = da.from_array(lhs, chunks=(rhs.chunks[0], lhs.shape[1]))
+    if skipna:
+        added_dim = rhs.ndim == 1
+        if added_dim:
+            rhs = rhs.reshape(rhs.shape[0], 1)
+        results = da.apply_along_axis(
+            nputils._nanpolyfit_1d,
+            0,
+            rhs,
+            lhs_da,
+            dtype=float,
+            shape=(lhs.shape[1] + 1,),
+            rcond=rcond,
+        )
+        coeffs = results[:-1, ...]
+        residuals = results[-1, ...]
+        if added_dim:
+            coeffs = coeffs.reshape(coeffs.shape[0])
+            residuals = residuals.reshape(residuals.shape[0])
+    else:
+        coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
+    return coeffs, residuals
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -3275,6 +3275,68 @@ def map_blocks(
 
         return map_blocks(func, self, args, kwargs)
 
+    def polyfit(
+        self,
+        dim: Hashable,
+        deg: int,
+        skipna: bool = None,
+        rcond: float = None,
+        w: Union[Hashable, Any] = None,
+        full: bool = False,
+        cov: bool = False,
+    ):
+        """
+        Least squares polynomial fit.
+
+        This replicates the behaviour of `numpy.polyfit` but differs by skipping
+        invalid values when `skipna = True`.
+
+        Parameters
+        ----------
+        dim : hashable
+            Coordinate along which to fit the polynomials.
+        deg : int
+            Degree of the fitting polynomial.
+        skipna : bool, optional
+            If True, removes all invalid values before fitting each 1D slices of the array.
+            Default is True if data is stored in a dask.array or if there is any
+            invalid values, False otherwise.
+        rcond : float, optional
+            Relative condition number to the fit.
+        w : Union[Hashable, Any], optional
+            Weights to apply to the y-coordinate of the sample points.
+            Can be an array-like object or the name of a coordinate in the dataset.
+        full : bool, optional
+            Whether to return the residuals, matrix rank and singular values in addition
+            to the coefficients.
+        cov : Union[bool, str], optional
+            Whether to return to the covariance matrix in addition to the coefficients.
+            The matrix is not scaled if `cov='unscaled'`.
+
+        Returns
+        -------
+        polyfit_results : Dataset
+            A single dataset which contains:
+
+            polyfit_coefficients
+                The coefficients of the best fit.
+            polyfit_residuals
+                The residuals of the least-square computation (only included if `full=True`)
+            [dim]_matrix_rank
+                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [dim]_singular_value
+                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            polyfit_covariance
+                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)
+
+        See also
+        --------
+        numpy.polyfit
+        """
+        return self._to_temp_dataset().polyfit(
+            dim, deg, skipna=skipna, rcond=rcond, w=w, full=full, cov=cov
+        )
+
     def pad(
         self,
         pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -76,6 +76,7 @@
     merge_coordinates_without_align,
     merge_data_and_coords,
 )
+from .missing import get_clean_interp_index
 from .options import OPTIONS, _get_keep_attrs
 from .pycompat import dask_array_type
 from .utils import (
@@ -5748,6 +5749,184 @@ def map_blocks(
 
         return map_blocks(func, self, args, kwargs)
 
+    def polyfit(
+        self,
+        dim: Hashable,
+        deg: int,
+        skipna: bool = None,
+        rcond: float = None,
+        w: Union[Hashable, Any] = None,
+        full: bool = False,
+        cov: Union[bool, str] = False,
+    ):
+        """
+        Least squares polynomial fit.
+
+        This replicates the behaviour of `numpy.polyfit` but differs by skipping
+        invalid values when `skipna = True`.
+
+        Parameters
+        ----------
+        dim : hashable
+            Coordinate along which to fit the polynomials.
+        deg : int
+            Degree of the fitting polynomial.
+        skipna : bool, optional
+            If True, removes all invalid values before fitting each 1D slices of the array.
+            Default is True if data is stored in a dask.array or if there is any
+            invalid values, False otherwise.
+        rcond : float, optional
+            Relative condition number to the fit.
+        w : Union[Hashable, Any], optional
+            Weights to apply to the y-coordinate of the sample points.
+            Can be an array-like object or the name of a coordinate in the dataset.
+        full : bool, optional
+            Whether to return the residuals, matrix rank and singular values in addition
+            to the coefficients.
+        cov : Union[bool, str], optional
+            Whether to return to the covariance matrix in addition to the coefficients.
+            The matrix is not scaled if `cov='unscaled'`.
+
+
+        Returns
+        -------
+        polyfit_results : Dataset
+            A single dataset which contains (for each "var" in the input dataset):
+
+            [var]_polyfit_coefficients
+                The coefficients of the best fit for each variable in this dataset.
+            [var]_polyfit_residuals
+                The residuals of the least-square computation for each variable (only included if `full=True`)
+            [dim]_matrix_rank
+                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [dim]_singular_values
+                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [var]_polyfit_covariance
+                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)
+
+        See also
+        --------
+        numpy.polyfit
+        """
+        variables = {}
+        skipna_da = skipna
+
+        x = get_clean_interp_index(self, dim)
+        xname = "{}_".format(self[dim].name)
+        order = int(deg) + 1
+        lhs = np.vander(x, order)
+
+        if rcond is None:
+            rcond = x.shape[0] * np.core.finfo(x.dtype).eps
+
+        # Weights:
+        if w is not None:
+            if isinstance(w, Hashable):
+                w = self.coords[w]
+            w = np.asarray(w)
+            if w.ndim != 1:
+                raise TypeError("Expected a 1-d array for weights.")
+            if w.shape[0] != lhs.shape[0]:
+                raise TypeError("Expected w and {} to have the same length".format(dim))
+            lhs *= w[:, np.newaxis]
+
+        # Scaling
+        scale = np.sqrt((lhs * lhs).sum(axis=0))
+        lhs /= scale
+
+        degree_dim = utils.get_temp_dimname(self.dims, "degree")
+
+        rank = np.linalg.matrix_rank(lhs)
+        if rank != order and not full:
+            warnings.warn(
+                "Polyfit may be poorly conditioned", np.RankWarning, stacklevel=4
+            )
+
+        if full:
+            rank = xr.DataArray(rank, name=xname + "matrix_rank")
+            variables[rank.name] = rank
+            sing = np.linalg.svd(lhs, compute_uv=False)
+            sing = xr.DataArray(
+                sing,
+                dims=(degree_dim,),
+                coords={degree_dim: np.arange(order)[::-1]},
+                name=xname + "singular_values",
+            )
+            variables[sing.name] = sing
+
+        for name, da in self.data_vars.items():
+            if dim not in da.dims:
+                continue
+
+            if skipna is None:
+                if isinstance(da.data, dask_array_type):
+                    skipna_da = True
+                else:
+                    skipna_da = np.any(da.isnull())
+
+            dims_to_stack = [dimname for dimname in da.dims if dimname != dim]
+            stacked_coords = {}
+            if dims_to_stack:
+                stacked_dim = utils.get_temp_dimname(dims_to_stack, "stacked")
+                rhs = da.transpose(dim, *dims_to_stack).stack(
+                    {stacked_dim: dims_to_stack}
+                )
+                stacked_coords = {stacked_dim: rhs[stacked_dim]}
+                scale_da = scale[:, np.newaxis]
+            else:
+                rhs = da
+                scale_da = scale
+
+            if w is not None:
+                rhs *= w[:, np.newaxis]
+
+            coeffs, residuals = duck_array_ops.least_squares(
+                lhs, rhs.data, rcond=rcond, skipna=skipna_da
+            )
+
+            if isinstance(name, str):
+                name = "{}_".format(name)
+            else:
+                # Thus a ReprObject => polyfit was called on a DataArray
+                name = ""
+
+            coeffs = xr.DataArray(
+                coeffs / scale_da,
+                dims=[degree_dim] + list(stacked_coords.keys()),
+                coords={degree_dim: np.arange(order)[::-1], **stacked_coords},
+                name=name + "polyfit_coefficients",
+            )
+            if dims_to_stack:
+                coeffs = coeffs.unstack(stacked_dim)
+            variables[coeffs.name] = coeffs
+
+            if full or (cov is True):
+                residuals = xr.DataArray(
+                    residuals if dims_to_stack else residuals.squeeze(),
+                    dims=list(stacked_coords.keys()),
+                    coords=stacked_coords,
+                    name=name + "polyfit_residuals",
+                )
+                if dims_to_stack:
+                    residuals = residuals.unstack(stacked_dim)
+                variables[residuals.name] = residuals
+
+            if cov:
+                Vbase = np.linalg.inv(np.dot(lhs.T, lhs))
+                Vbase /= np.outer(scale, scale)
+                if cov == "unscaled":
+                    fac = 1
+                else:
+                    if x.shape[0] <= order:
+                        raise ValueError(
+                            "The number of data points must exceed order to scale the covariance matrix."
+                        )
+                    fac = residuals / (x.shape[0] - order)
+                covariance = xr.DataArray(Vbase, dims=("cov_i", "cov_j"),) * fac
+                variables[name + "polyfit_covariance"] = covariance
+
+        return Dataset(data_vars=variables, attrs=self.attrs.copy())
+
     def pad(
         self,
         pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -597,3 +597,12 @@ def rolling_window(array, axis, window, center, fill_value):
         return dask_array_ops.rolling_window(array, axis, window, center, fill_value)
     else:  # np.ndarray
         return nputils.rolling_window(array, axis, window, center, fill_value)
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    """Return the coefficients and residuals of a least-squares fit.
+    """
+    if isinstance(rhs, dask_array_type):
+        return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
+    else:
+        return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
diff --git a/xarray/core/nputils.py b/xarray/core/nputils.py
--- a/xarray/core/nputils.py
+++ b/xarray/core/nputils.py
@@ -220,6 +220,39 @@ def f(values, axis=None, **kwargs):
     return f
 
 
+def _nanpolyfit_1d(arr, x, rcond=None):
+    out = np.full((x.shape[1] + 1,), np.nan)
+    mask = np.isnan(arr)
+    if not np.all(mask):
+        out[:-1], out[-1], _, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
+    return out
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    if skipna:
+        added_dim = rhs.ndim == 1
+        if added_dim:
+            rhs = rhs.reshape(rhs.shape[0], 1)
+        nan_cols = np.any(np.isnan(rhs), axis=0)
+        out = np.empty((lhs.shape[1] + 1, rhs.shape[1]))
+        if np.any(nan_cols):
+            out[:, nan_cols] = np.apply_along_axis(
+                _nanpolyfit_1d, 0, rhs[:, nan_cols], lhs
+            )
+        if np.any(~nan_cols):
+            out[:-1, ~nan_cols], out[-1, ~nan_cols], _, _ = np.linalg.lstsq(
+                lhs, rhs[:, ~nan_cols], rcond=rcond
+            )
+        coeffs = out[:-1, :]
+        residuals = out[-1, :]
+        if added_dim:
+            coeffs = coeffs.reshape(coeffs.shape[0])
+            residuals = residuals.reshape(residuals.shape[0])
+    else:
+        coeffs, residuals, _, _ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
+    return coeffs, residuals
+
+
 nanmin = _create_bottleneck_method("nanmin")
 nanmax = _create_bottleneck_method("nanmax")
 nanmean = _create_bottleneck_method("nanmean")

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/__init__.py | 20 | 20 | 86 | 31 | 43041
| xarray/__init__.py | 68 | 68 | 86 | 31 | 43041
| xarray/core/computation.py | 1309 | 1309 | - | 1 | -
| xarray/core/dask_array_ops.py | 98 | 98 | 21 | 10 | 12316
| xarray/core/dataarray.py | 3278 | 3278 | 111 | 9 | 52992
| xarray/core/dataset.py | 79 | 79 | 22 | 11 | 12938
| xarray/core/dataset.py | 5751 | 5751 | - | 11 | -
| xarray/core/duck_array_ops.py | 600 | 600 | 34 | 17 | 17411
| xarray/core/nputils.py | 223 | 223 | - | 30 | -


## Problem Statement

```
Implement polyfit?
Fitting a line (or curve) to data along a specified axis is a long-standing need of xarray users. There are many blog posts and SO questions about how to do it:
- http://atedstone.github.io/rate-of-change-maps/
- https://gist.github.com/luke-gregor/4bb5c483b2d111e52413b260311fbe43
- https://stackoverflow.com/questions/38960903/applying-numpy-polyfit-to-xarray-dataset
- https://stackoverflow.com/questions/52094320/with-xarray-how-to-parallelize-1d-operations-on-a-multidimensional-dataset
- https://stackoverflow.com/questions/36275052/applying-a-function-along-an-axis-of-a-dask-array

The main use case in my domain is finding the temporal trend on a 3D variable (e.g. temperature in time, lon, lat).

Yes, you can do it with apply_ufunc, but apply_ufunc is inaccessibly complex for many users. Much of our existing API could be removed and replaced with apply_ufunc calls, but that doesn't mean we should do it.

I am proposing we add a Dataarray method called `polyfit`. It would work like this:

\`\`\`python
x_ = np.linspace(0, 1, 10)
y_ = np.arange(5)
a_ = np.cos(y_)

x = xr.DataArray(x_, dims=['x'], coords={'x': x_})
a = xr.DataArray(a_, dims=['y'])
f = a*x
p = f.polyfit(dim='x', deg=1)

# equivalent numpy code
p_ = np.polyfit(x_, f.values.transpose(), 1)
np.testing.assert_allclose(p_[0], a_)
\`\`\`

Numpy's [polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit) function is already vectorized in the sense that it accepts 1D x and 2D y, performing the fit independently over each column of y. To extend this to ND, we would just need to reshape the data going in and out of the function. We do this already in [other packages](https://github.com/xgcm/xcape/blob/master/xcape/core.py#L16-L34). For dask, we could simply require that the dimension over which the fit is calculated be contiguous, and then call map_blocks.

Thoughts?




```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/computation.py** | 776 | 978| 2272 | 2272 | 10540 | 
| 2 | **1 xarray/core/computation.py** | 979 | 1069| 779 | 3051 | 10540 | 
| 3 | 2 xarray/core/common.py | 485 | 610| 1162 | 4213 | 23702 | 
| 4 | **2 xarray/core/computation.py** | 759 | 1069| 154 | 4367 | 23702 | 
| 5 | 3 xarray/core/parallel.py | 105 | 202| 1263 | 5630 | 27209 | 
| 6 | **3 xarray/core/computation.py** | 736 | 756| 206 | 5836 | 27209 | 
| 7 | 4 xarray/ufuncs.py | 1 | 36| 251 | 6087 | 28453 | 
| 8 | 5 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 6618 | 28984 | 
| 9 | **5 xarray/core/computation.py** | 542 | 649| 755 | 7373 | 28984 | 
| 10 | **5 xarray/core/computation.py** | 414 | 457| 387 | 7760 | 28984 | 
| 11 | **5 xarray/core/computation.py** | 652 | 733| 620 | 8380 | 28984 | 
| 12 | 6 xarray/core/arithmetic.py | 32 | 80| 448 | 8828 | 29747 | 
| 13 | 7 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 9012 | 30232 | 
| 14 | 8 xarray/core/npcompat.py | 31 | 50| 171 | 9183 | 31000 | 
| 15 | **9 xarray/core/dataarray.py** | 3221 | 3276| 496 | 9679 | 58900 | 
| 16 | **9 xarray/core/computation.py** | 213 | 244| 245 | 9924 | 58900 | 
| 17 | **9 xarray/core/dataarray.py** | 1 | 82| 433 | 10357 | 58900 | 
| 18 | **9 xarray/core/computation.py** | 1 | 40| 201 | 10558 | 58900 | 
| 19 | **9 xarray/core/dataarray.py** | 1317 | 1380| 478 | 11036 | 58900 | 
| 20 | **9 xarray/core/computation.py** | 1153 | 1215| 609 | 11645 | 58900 | 
| **-> 21 <-** | **10 xarray/core/dask_array_ops.py** | 30 | 98| 671 | 12316 | 59799 | 
| **-> 22 <-** | **11 xarray/core/dataset.py** | 1 | 132| 622 | 12938 | 108029 | 
| 23 | 12 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 13668 | 109441 | 
| 24 | 13 xarray/core/missing.py | 641 | 702| 438 | 14106 | 114507 | 
| 25 | 14 asv_bench/benchmarks/rolling.py | 1 | 17| 117 | 14223 | 115136 | 
| 26 | **14 xarray/core/dataarray.py** | 3153 | 3219| 536 | 14759 | 115136 | 
| 27 | **14 xarray/core/computation.py** | 349 | 411| 472 | 15231 | 115136 | 
| 28 | 15 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 15644 | 115549 | 
| 29 | 15 xarray/core/missing.py | 162 | 207| 275 | 15919 | 115549 | 
| 30 | **15 xarray/core/dataarray.py** | 1382 | 1435| 431 | 16350 | 115549 | 
| 31 | 16 xarray/plot/plot.py | 432 | 458| 206 | 16556 | 123682 | 
| 32 | **16 xarray/core/dataarray.py** | 2622 | 2641| 197 | 16753 | 123682 | 
| 33 | 16 asv_bench/benchmarks/interp.py | 41 | 56| 162 | 16915 | 123682 | 
| **-> 34 <-** | **17 xarray/core/duck_array_ops.py** | 539 | 600| 496 | 17411 | 128602 | 
| 35 | **17 xarray/core/duck_array_ops.py** | 108 | 132| 275 | 17686 | 128602 | 
| 36 | 17 xarray/ufuncs.py | 39 | 79| 370 | 18056 | 128602 | 
| 37 | 18 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 18215 | 128761 | 
| 38 | **18 xarray/core/dataarray.py** | 3294 | 3453| 1862 | 20077 | 128761 | 
| 39 | 19 xarray/plot/facetgrid.py | 215 | 284| 488 | 20565 | 133567 | 
| 40 | **19 xarray/core/computation.py** | 315 | 346| 240 | 20805 | 133567 | 
| 41 | 19 xarray/plot/plot.py | 295 | 329| 443 | 21248 | 133567 | 
| 42 | 19 xarray/ufuncs.py | 135 | 200| 321 | 21569 | 133567 | 
| 43 | 19 xarray/core/common.py | 1 | 35| 178 | 21747 | 133567 | 
| 44 | **19 xarray/core/duck_array_ops.py** | 24 | 54| 209 | 21956 | 133567 | 
| 45 | **19 xarray/core/dataarray.py** | 266 | 360| 853 | 22809 | 133567 | 
| 46 | **19 xarray/core/dataarray.py** | 1764 | 1817| 458 | 23267 | 133567 | 
| 47 | 19 xarray/plot/plot.py | 612 | 765| 1699 | 24966 | 133567 | 
| 48 | 20 xarray/coding/variables.py | 80 | 99| 149 | 25115 | 136110 | 
| 49 | 20 xarray/ufuncs.py | 118 | 132| 130 | 25245 | 136110 | 
| 50 | 20 xarray/core/parallel.py | 204 | 288| 743 | 25988 | 136110 | 
| 51 | 20 xarray/plot/plot.py | 1 | 28| 152 | 26140 | 136110 | 
| 52 | **20 xarray/core/duck_array_ops.py** | 1 | 21| 132 | 26272 | 136110 | 
| 53 | **20 xarray/core/computation.py** | 488 | 539| 401 | 26673 | 136110 | 
| 54 | 21 xarray/plot/utils.py | 1 | 54| 283 | 26956 | 142273 | 
| 55 | 21 xarray/plot/facetgrid.py | 552 | 590| 349 | 27305 | 142273 | 
| 56 | 22 xarray/backends/pydap_.py | 1 | 37| 271 | 27576 | 142902 | 
| 57 | **22 xarray/core/dataarray.py** | 216 | 264| 424 | 28000 | 142902 | 
| 58 | 22 xarray/core/parallel.py | 289 | 380| 799 | 28799 | 142902 | 
| 59 | 23 xarray/backends/pseudonetcdf_.py | 1 | 34| 261 | 29060 | 143517 | 
| 60 | 24 xarray/core/resample.py | 178 | 248| 608 | 29668 | 146122 | 
| 61 | **24 xarray/core/duck_array_ops.py** | 468 | 504| 300 | 29968 | 146122 | 
| 62 | **24 xarray/core/dataarray.py** | 362 | 403| 384 | 30352 | 146122 | 
| 63 | **24 xarray/core/computation.py** | 1072 | 1151| 727 | 31079 | 146122 | 
| 64 | **24 xarray/core/dataarray.py** | 2151 | 2173| 169 | 31248 | 146122 | 
| 65 | **24 xarray/core/dataarray.py** | 2063 | 2149| 879 | 32127 | 146122 | 
| 66 | 25 xarray/core/dask_array_compat.py | 1 | 100| 717 | 32844 | 147859 | 
| 67 | 25 xarray/core/npcompat.py | 78 | 97| 119 | 32963 | 147859 | 
| 68 | 26 doc/gallery/plot_lines_from_2d.py | 1 | 39| 258 | 33221 | 148117 | 
| 69 | **26 xarray/core/dataarray.py** | 760 | 795| 307 | 33528 | 148117 | 
| 70 | **26 xarray/core/dataarray.py** | 2217 | 2260| 422 | 33950 | 148117 | 
| 71 | **26 xarray/core/dataarray.py** | 2593 | 2620| 232 | 34182 | 148117 | 
| 72 | 27 xarray/core/groupby.py | 764 | 811| 444 | 34626 | 156046 | 
| 73 | 27 xarray/core/parallel.py | 1 | 43| 205 | 34831 | 156046 | 
| 74 | 27 xarray/core/parallel.py | 381 | 405| 242 | 35073 | 156046 | 
| 75 | 27 xarray/core/common.py | 1071 | 1120| 513 | 35586 | 156046 | 
| 76 | 28 xarray/core/dtypes.py | 1 | 42| 277 | 35863 | 157103 | 
| 77 | 29 xarray/core/alignment.py | 69 | 255| 1897 | 37760 | 163065 | 
| 78 | 29 xarray/plot/facetgrid.py | 76 | 213| 1058 | 38818 | 163065 | 
| 79 | 29 xarray/core/missing.py | 716 | 730| 182 | 39000 | 163065 | 
| 80 | **29 xarray/core/dataarray.py** | 2840 | 2899| 412 | 39412 | 163065 | 
| 81 | **30 xarray/core/nputils.py** | 115 | 132| 175 | 39587 | 165012 | 
| 82 | **30 xarray/core/nputils.py** | 1 | 33| 244 | 39831 | 165012 | 
| 83 | **30 xarray/core/dataset.py** | 2302 | 2498| 2123 | 41954 | 165012 | 
| 84 | 30 xarray/plot/facetgrid.py | 593 | 646| 315 | 42269 | 165012 | 
| 85 | **30 xarray/core/duck_array_ops.py** | 57 | 79| 195 | 42464 | 165012 | 
| **-> 86 <-** | **31 xarray/__init__.py** | 1 | 89| 577 | 43041 | 165589 | 
| 87 | **31 xarray/core/dataarray.py** | 2741 | 2788| 353 | 43394 | 165589 | 
| 88 | **31 xarray/core/duck_array_ops.py** | 324 | 348| 252 | 43646 | 165589 | 
| 89 | 31 xarray/core/groupby.py | 813 | 850| 331 | 43977 | 165589 | 
| 90 | 32 xarray/core/nanops.py | 76 | 116| 400 | 44377 | 167447 | 
| 91 | **32 xarray/core/dataset.py** | 4789 | 4896| 900 | 45277 | 167447 | 
| 92 | **32 xarray/core/dataarray.py** | 2790 | 2838| 381 | 45658 | 167447 | 
| 93 | 32 xarray/plot/plot.py | 31 | 113| 759 | 46417 | 167447 | 
| 94 | 32 xarray/core/missing.py | 362 | 378| 156 | 46573 | 167447 | 
| 95 | 32 xarray/plot/plot.py | 767 | 821| 318 | 46891 | 167447 | 
| 96 | **32 xarray/core/dataarray.py** | 3094 | 3151| 558 | 47449 | 167447 | 
| 97 | **32 xarray/core/dataarray.py** | 2175 | 2197| 168 | 47617 | 167447 | 
| 98 | 33 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 48332 | 171052 | 
| 99 | 34 xarray/core/rolling.py | 327 | 368| 364 | 48696 | 176297 | 
| 100 | 34 xarray/core/missing.py | 381 | 427| 287 | 48983 | 176297 | 
| 101 | **34 xarray/core/dask_array_ops.py** | 1 | 27| 227 | 49210 | 176297 | 
| 102 | **34 xarray/core/dataarray.py** | 1193 | 1251| 524 | 49734 | 176297 | 
| 103 | 35 xarray/core/concat.py | 116 | 144| 274 | 50008 | 180067 | 
| 104 | 36 xarray/core/combine.py | 716 | 771| 441 | 50449 | 188937 | 
| 105 | 36 xarray/core/rolling.py | 370 | 389| 201 | 50650 | 188937 | 
| 106 | 36 xarray/core/groupby.py | 1 | 36| 230 | 50880 | 188937 | 
| 107 | 36 asv_bench/benchmarks/rolling.py | 39 | 70| 337 | 51217 | 188937 | 
| 108 | 36 xarray/core/dask_array_compat.py | 149 | 220| 517 | 51734 | 188937 | 
| 109 | 36 xarray/plot/plot.py | 203 | 294| 774 | 52508 | 188937 | 
| 110 | 36 xarray/core/resample.py | 272 | 306| 323 | 52831 | 188937 | 
| **-> 111 <-** | **36 xarray/core/dataarray.py** | 3278 | 3444| 161 | 52992 | 188937 | 
| 112 | **36 xarray/core/duck_array_ops.py** | 240 | 281| 290 | 53282 | 188937 | 
| 113 | 36 xarray/plot/plot.py | 567 | 765| 266 | 53548 | 188937 | 
| 114 | 37 xarray/testing.py | 1 | 24| 145 | 53693 | 191462 | 
| 115 | **37 xarray/core/dataarray.py** | 1078 | 1093| 134 | 53827 | 191462 | 
| 116 | 38 xarray/conventions.py | 1 | 44| 297 | 54124 | 197191 | 
| 117 | **38 xarray/core/dataarray.py** | 1003 | 1038| 300 | 54424 | 197191 | 


### Hint

```
dask has `lstsq` https://docs.dask.org/en/latest/array-api.html#dask.array.linalg.lstsq . Would that avoid the dimension-must-have-one-chunk issue?

EDIT: I am in favour of adding this. It's a common use case like `differentiate` and `integrate`
I am in favour of adding this (and other common functionality), but I would comment that perhaps we should move forward with discussion about where to put extra functionality generally (the scipy to xarray's numpy if you will)? If only because otherwise the API could get to an unwieldy size? 

I can't remember where the relevant issue was, but for example this might go under an `xarray.utils` module?
I second @TomNicholas' point... functionality like this would be wonderful to have but where would be the best place for it to live?
The question of a standalone library has come up many times (see discussion in #1850). Everyone agrees it's a nice idea, but no one seems to have the bandwidth to take on ownership and maintenance of such a project.

Perhaps we need to put this issue on pause and figure out a general strategy here. The current Xarray API is far from a complete feature set, so more development is needed. But we should decide what belongs in xarray and what belongs elsewhere. #1850 is probably the best place to continue that conversation.
The quickest way to close this is probably to extend @fujiisoup's xr-scipy(https://github.com/fujiisoup/xr-scipy) to wrap `scipy.linalg.lstsq` and `dask.array.linalg.lstsq`. It is likely that all the necessary helper functions already exist.
Now that xarray itself has interpolate, gradient, and integrate, it seems like the only thing left in xr-scipy is fourier transforms, which is also what we provide in [xrft](https://github.com/xgcm/xrft)! 😆 
From a user perspective, I think people prefer to find stuff in one place.

From a maintainer perspective, as long as it's somewhat domain agnostic (e.g., "physical sciences" rather than "oceanography") and written to a reasonable level of code quality, I think it's fine to toss it into xarray. "Already exists in NumPy/SciPy" is probably a reasonable proxy for the former.

So I say: yes, let's toss in polyfit, along with fast fourier transforms.

If we're concerned about clutter, we can put stuff in a dedicated namespace, e.g., `xarray.wrappers`.
https://xscale.readthedocs.io/en/latest/generated/xscale.signal.fitting.polyfit.html#xscale.signal.fitting.polyfit
I'm getting deja-vu here... Xscale has a huge and impressive sounding API. But no code coverage and no commits since January. Like many of these projects, it seems to have bit off more than its maintainers could chew.

_Edit: I'd love for such a package to really achieve community uptake and become sustainable. I just don't quite know the roadmap for getting there._
It seems like these are getting reinvented often enough that it's worth
pulling some of these into xarray proper.

On Wed, Oct 2, 2019 at 9:04 AM Ryan Abernathey <notifications@github.com>
wrote:

> I'm getting deja-vu here... Xscale has a huge and impressive sounding API.
> But no code coverage and no commits since January. Like many of these
> projects, it seems to have bit off more than its maintainers could chew.
>
> —
> You are receiving this because you commented.
> Reply to this email directly, view it on GitHub
> <https://github.com/pydata/xarray/issues/3349?email_source=notifications&email_token=AAJJFVV5XANAGSPKHSF6KZLQMTA7HA5CNFSM4I3HCUUKYY3PNVWWK3TUL52HS4DFVREXG43VMVBW63LNMVXHJKTDN5WW2ZLOORPWSZGOEAFJCDI#issuecomment-537563405>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAJJFVREMUT5RKJJESG5NVLQMTA7HANCNFSM4I3HCUUA>
> .
>

xyzpy has a polyfit too : https://xyzpy.readthedocs.io/en/latest/manipulate.html
Started to work on this and facing some issues with the x-coordinate when its a datetime. For standard calendars, I can use `pd.to_numeric(da.time)`, but for non-standard calendars, it's not clear how to go ahead. If I use `xr.coding.times.encode_cf_datetime(coord)`, the coefficients we'll find will only make sense in the `polyval` function if we use the same time encoding. 


If I understand correctly, `pd.to_numeric` (and its inverse) works because it always uses 1970-01-01T00:00:00 as the reference date.  Could we do something similar when working with cftime dates?

Within xarray we typically convert dates to numeric values (e.g. when doing interpolation) using `xarray.core.duck_array_ops.datetime_to_numeric`, which takes an optional `offset` argument to control the reference date.  Would it work to always make sure to pass 1970-01-01T00:00:00 with the appropriate calendar type as the offset when constructing the ordinal x-coordinate for `polyfit`/`polyval`?
Thanks, it seems to work !
Excellent, looking forward to seeing it in a PR!
My current implementation is pretty naive. It's just calling numpy.polyfit using dask.array.apply_along_axis. Happy to put that in a PR as a starting point, but there are a couple of questions I had: 
* How to return the full output (residuals, rank, singular_values, rcond) ? A tuple of dataarrays or a dataset ?
* Do we want to use the dask least square functionality to allow for chunking within the x dimension ? Then it's not just a simple wrapper around polyfit. 
* Should we use np.polyfit or np.polynomial.polynomial.polyfit ?
[geocat.comp.ndpolyfit](https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.ndpolyfit.html#geocat.comp.ndpolyfit) extends `NumPy.polyfit` for multi-dimensional arrays and has support for Xarray and Dask. It does exactly what is requested here.

regards,

@andersy005 @clyne @matt-long @khallock
@maboualidev Nice ! I see you're storing the residuals in the DataArray attributes. From my perspective, it would be useful to have those directly as DataArrays. Thoughts ?

So it looks like there are multiple inspirations to draw from. Here is what I could gather. 

- `xscale.signal.fitting.polyfit(obj, deg=1, dim=None, coord=None)` supports chunking along the fitting dimension using `dask.array.linalg.lstsq`. No explicit missing data handling.
- `xyzpy.signal.xr_polyfit(obj, dim, ix=None, deg=0.5, poly='hermite')` applies `np.polynomial.polynomial.polyfit` using `xr.apply_ufunc` along dim with the help of `numba`. Also supports other types of polynomial (legendre, chebyshev, ...). Missing values are masked out 1D wise. 
 - `geocat.comp.ndpolyfit(x: Iterable, y: Iterable, deg: int, axis: int = 0, **kwargs) -> (xr.DataArray, da.Array)` reorders the array to apply `np.polyfit` along dim, returns the full outputs (residuals, rank, etc) as DataArray attributes.  Missing values are masked out in bulk if possible, 1D-wise otherwise. 

There does not seem to be matching `polyval` implementations for any of those nor support for indexing along a time dimension with a non-standard calendar. 

Hi @huard Thanks for the reply.

Regarding:

> There does not seem to be matching polyval implementations for any of those nor support for indexing along a time dimension with a non-standard calendar.

There is a pull request on GeoCAT-comp for [ndpolyval](https://github.com/NCAR/geocat-comp/pull/49). I think `polyval` and `polyfit` go hand-in-hand. If we have `ndpolyfit` there must be a also a `ndpolyval`. 

Regarding:

> I see you're storing the residuals in the DataArray attributes. From my perspective, it would be useful to have those directly as DataArrays. Thoughts ?

I see the point and agree with you. I think it is a good idea to be as similar to `NumPy.polyfit` as possible; even for the style of the output. I will see it through to have that changed in GeoCAT.


attn: @clyne and @khallock
@maboualidev Is your objective to integrate the GeoCat implementation into xarray or keep it standalone ? 

On my end, I'll submit a PR to add support for non-standard calendars to `xarray.core.missing.get_clean_interp`, which you'd then be able to use to get x values from coordinates. 
> @maboualidev Is your objective to integrate the GeoCat implementation into xarray or keep it standalone ?

[GeoCAT](https://geocat.ucar.edu) is the python version of [NCL](https://www.ncl.ucar.edu) and we are a team at [NCAR](https://ncar.ucar.edu) working on it. I know that the team decision is to make use of Xarray within GeoCAT as much as possible, though. 


Currently the plan is to keep GeoCAT as a standalone package that plays well with Xarray.

> On Dec 16, 2019, at 9:21 AM, Mohammad Abouali <notifications@github.com> wrote:
> 
> @maboualidev Is your objective to integrate the GeoCat implementation into xarray or keep it standalone ?
> 
> GeoCAT is the python version of NCL and we are a team at NCAR working on it. I know that the team decision is to make use of Xarray within GeoCAT as much as possible, though.
> 
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub, or unsubscribe.
> 




@clyne Let me rephrase my question: how do you feel about xarray providing a polyfit/polyval implementation essentially duplicating GeoCat's implementation ? 
GeoCAT is licensed under Apache 2.0. So if someone wants to incorporate it into Xarray they are welcome to it :-)
```

## Patch

```diff
diff --git a/xarray/__init__.py b/xarray/__init__.py
--- a/xarray/__init__.py
+++ b/xarray/__init__.py
@@ -17,7 +17,7 @@
 from .core.alignment import align, broadcast
 from .core.combine import auto_combine, combine_by_coords, combine_nested
 from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
-from .core.computation import apply_ufunc, dot, where
+from .core.computation import apply_ufunc, dot, polyval, where
 from .core.concat import concat
 from .core.dataarray import DataArray
 from .core.dataset import Dataset
@@ -65,6 +65,7 @@
     "open_mfdataset",
     "open_rasterio",
     "open_zarr",
+    "polyval",
     "register_dataarray_accessor",
     "register_dataset_accessor",
     "save_mfdataset",
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -1306,3 +1306,35 @@ def where(cond, x, y):
         dataset_join="exact",
         dask="allowed",
     )
+
+
+def polyval(coord, coeffs, degree_dim="degree"):
+    """Evaluate a polynomial at specific values
+
+    Parameters
+    ----------
+    coord : DataArray
+        The 1D coordinate along which to evaluate the polynomial.
+    coeffs : DataArray
+        Coefficients of the polynomials.
+    degree_dim : str, default "degree"
+        Name of the polynomial degree dimension in `coeffs`.
+
+    See also
+    --------
+    xarray.DataArray.polyfit
+    numpy.polyval
+    """
+    from .dataarray import DataArray
+    from .missing import get_clean_interp_index
+
+    x = get_clean_interp_index(coord, coord.name)
+
+    deg_coord = coeffs[degree_dim]
+
+    lhs = DataArray(
+        np.vander(x, int(deg_coord.max()) + 1),
+        dims=(coord.name, degree_dim),
+        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
+    )
+    return (lhs * coeffs).sum(degree_dim)
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -95,3 +95,30 @@ def func(x, window, axis=-1):
     # crop boundary.
     index = (slice(None),) * axis + (slice(drop_size, drop_size + orig_shape[axis]),)
     return out[index]
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    import dask.array as da
+
+    lhs_da = da.from_array(lhs, chunks=(rhs.chunks[0], lhs.shape[1]))
+    if skipna:
+        added_dim = rhs.ndim == 1
+        if added_dim:
+            rhs = rhs.reshape(rhs.shape[0], 1)
+        results = da.apply_along_axis(
+            nputils._nanpolyfit_1d,
+            0,
+            rhs,
+            lhs_da,
+            dtype=float,
+            shape=(lhs.shape[1] + 1,),
+            rcond=rcond,
+        )
+        coeffs = results[:-1, ...]
+        residuals = results[-1, ...]
+        if added_dim:
+            coeffs = coeffs.reshape(coeffs.shape[0])
+            residuals = residuals.reshape(residuals.shape[0])
+    else:
+        coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
+    return coeffs, residuals
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -3275,6 +3275,68 @@ def map_blocks(
 
         return map_blocks(func, self, args, kwargs)
 
+    def polyfit(
+        self,
+        dim: Hashable,
+        deg: int,
+        skipna: bool = None,
+        rcond: float = None,
+        w: Union[Hashable, Any] = None,
+        full: bool = False,
+        cov: bool = False,
+    ):
+        """
+        Least squares polynomial fit.
+
+        This replicates the behaviour of `numpy.polyfit` but differs by skipping
+        invalid values when `skipna = True`.
+
+        Parameters
+        ----------
+        dim : hashable
+            Coordinate along which to fit the polynomials.
+        deg : int
+            Degree of the fitting polynomial.
+        skipna : bool, optional
+            If True, removes all invalid values before fitting each 1D slices of the array.
+            Default is True if data is stored in a dask.array or if there is any
+            invalid values, False otherwise.
+        rcond : float, optional
+            Relative condition number to the fit.
+        w : Union[Hashable, Any], optional
+            Weights to apply to the y-coordinate of the sample points.
+            Can be an array-like object or the name of a coordinate in the dataset.
+        full : bool, optional
+            Whether to return the residuals, matrix rank and singular values in addition
+            to the coefficients.
+        cov : Union[bool, str], optional
+            Whether to return to the covariance matrix in addition to the coefficients.
+            The matrix is not scaled if `cov='unscaled'`.
+
+        Returns
+        -------
+        polyfit_results : Dataset
+            A single dataset which contains:
+
+            polyfit_coefficients
+                The coefficients of the best fit.
+            polyfit_residuals
+                The residuals of the least-square computation (only included if `full=True`)
+            [dim]_matrix_rank
+                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [dim]_singular_value
+                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            polyfit_covariance
+                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)
+
+        See also
+        --------
+        numpy.polyfit
+        """
+        return self._to_temp_dataset().polyfit(
+            dim, deg, skipna=skipna, rcond=rcond, w=w, full=full, cov=cov
+        )
+
     def pad(
         self,
         pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -76,6 +76,7 @@
     merge_coordinates_without_align,
     merge_data_and_coords,
 )
+from .missing import get_clean_interp_index
 from .options import OPTIONS, _get_keep_attrs
 from .pycompat import dask_array_type
 from .utils import (
@@ -5748,6 +5749,184 @@ def map_blocks(
 
         return map_blocks(func, self, args, kwargs)
 
+    def polyfit(
+        self,
+        dim: Hashable,
+        deg: int,
+        skipna: bool = None,
+        rcond: float = None,
+        w: Union[Hashable, Any] = None,
+        full: bool = False,
+        cov: Union[bool, str] = False,
+    ):
+        """
+        Least squares polynomial fit.
+
+        This replicates the behaviour of `numpy.polyfit` but differs by skipping
+        invalid values when `skipna = True`.
+
+        Parameters
+        ----------
+        dim : hashable
+            Coordinate along which to fit the polynomials.
+        deg : int
+            Degree of the fitting polynomial.
+        skipna : bool, optional
+            If True, removes all invalid values before fitting each 1D slices of the array.
+            Default is True if data is stored in a dask.array or if there is any
+            invalid values, False otherwise.
+        rcond : float, optional
+            Relative condition number to the fit.
+        w : Union[Hashable, Any], optional
+            Weights to apply to the y-coordinate of the sample points.
+            Can be an array-like object or the name of a coordinate in the dataset.
+        full : bool, optional
+            Whether to return the residuals, matrix rank and singular values in addition
+            to the coefficients.
+        cov : Union[bool, str], optional
+            Whether to return to the covariance matrix in addition to the coefficients.
+            The matrix is not scaled if `cov='unscaled'`.
+
+
+        Returns
+        -------
+        polyfit_results : Dataset
+            A single dataset which contains (for each "var" in the input dataset):
+
+            [var]_polyfit_coefficients
+                The coefficients of the best fit for each variable in this dataset.
+            [var]_polyfit_residuals
+                The residuals of the least-square computation for each variable (only included if `full=True`)
+            [dim]_matrix_rank
+                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [dim]_singular_values
+                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
+            [var]_polyfit_covariance
+                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)
+
+        See also
+        --------
+        numpy.polyfit
+        """
+        variables = {}
+        skipna_da = skipna
+
+        x = get_clean_interp_index(self, dim)
+        xname = "{}_".format(self[dim].name)
+        order = int(deg) + 1
+        lhs = np.vander(x, order)
+
+        if rcond is None:
+            rcond = x.shape[0] * np.core.finfo(x.dtype).eps
+
+        # Weights:
+        if w is not None:
+            if isinstance(w, Hashable):
+                w = self.coords[w]
+            w = np.asarray(w)
+            if w.ndim != 1:
+                raise TypeError("Expected a 1-d array for weights.")
+            if w.shape[0] != lhs.shape[0]:
+                raise TypeError("Expected w and {} to have the same length".format(dim))
+            lhs *= w[:, np.newaxis]
+
+        # Scaling
+        scale = np.sqrt((lhs * lhs).sum(axis=0))
+        lhs /= scale
+
+        degree_dim = utils.get_temp_dimname(self.dims, "degree")
+
+        rank = np.linalg.matrix_rank(lhs)
+        if rank != order and not full:
+            warnings.warn(
+                "Polyfit may be poorly conditioned", np.RankWarning, stacklevel=4
+            )
+
+        if full:
+            rank = xr.DataArray(rank, name=xname + "matrix_rank")
+            variables[rank.name] = rank
+            sing = np.linalg.svd(lhs, compute_uv=False)
+            sing = xr.DataArray(
+                sing,
+                dims=(degree_dim,),
+                coords={degree_dim: np.arange(order)[::-1]},
+                name=xname + "singular_values",
+            )
+            variables[sing.name] = sing
+
+        for name, da in self.data_vars.items():
+            if dim not in da.dims:
+                continue
+
+            if skipna is None:
+                if isinstance(da.data, dask_array_type):
+                    skipna_da = True
+                else:
+                    skipna_da = np.any(da.isnull())
+
+            dims_to_stack = [dimname for dimname in da.dims if dimname != dim]
+            stacked_coords = {}
+            if dims_to_stack:
+                stacked_dim = utils.get_temp_dimname(dims_to_stack, "stacked")
+                rhs = da.transpose(dim, *dims_to_stack).stack(
+                    {stacked_dim: dims_to_stack}
+                )
+                stacked_coords = {stacked_dim: rhs[stacked_dim]}
+                scale_da = scale[:, np.newaxis]
+            else:
+                rhs = da
+                scale_da = scale
+
+            if w is not None:
+                rhs *= w[:, np.newaxis]
+
+            coeffs, residuals = duck_array_ops.least_squares(
+                lhs, rhs.data, rcond=rcond, skipna=skipna_da
+            )
+
+            if isinstance(name, str):
+                name = "{}_".format(name)
+            else:
+                # Thus a ReprObject => polyfit was called on a DataArray
+                name = ""
+
+            coeffs = xr.DataArray(
+                coeffs / scale_da,
+                dims=[degree_dim] + list(stacked_coords.keys()),
+                coords={degree_dim: np.arange(order)[::-1], **stacked_coords},
+                name=name + "polyfit_coefficients",
+            )
+            if dims_to_stack:
+                coeffs = coeffs.unstack(stacked_dim)
+            variables[coeffs.name] = coeffs
+
+            if full or (cov is True):
+                residuals = xr.DataArray(
+                    residuals if dims_to_stack else residuals.squeeze(),
+                    dims=list(stacked_coords.keys()),
+                    coords=stacked_coords,
+                    name=name + "polyfit_residuals",
+                )
+                if dims_to_stack:
+                    residuals = residuals.unstack(stacked_dim)
+                variables[residuals.name] = residuals
+
+            if cov:
+                Vbase = np.linalg.inv(np.dot(lhs.T, lhs))
+                Vbase /= np.outer(scale, scale)
+                if cov == "unscaled":
+                    fac = 1
+                else:
+                    if x.shape[0] <= order:
+                        raise ValueError(
+                            "The number of data points must exceed order to scale the covariance matrix."
+                        )
+                    fac = residuals / (x.shape[0] - order)
+                covariance = xr.DataArray(Vbase, dims=("cov_i", "cov_j"),) * fac
+                variables[name + "polyfit_covariance"] = covariance
+
+        return Dataset(data_vars=variables, attrs=self.attrs.copy())
+
     def pad(
         self,
         pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -597,3 +597,12 @@ def rolling_window(array, axis, window, center, fill_value):
         return dask_array_ops.rolling_window(array, axis, window, center, fill_value)
     else:  # np.ndarray
         return nputils.rolling_window(array, axis, window, center, fill_value)
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    """Return the coefficients and residuals of a least-squares fit.
+    """
+    if isinstance(rhs, dask_array_type):
+        return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
+    else:
+        return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
diff --git a/xarray/core/nputils.py b/xarray/core/nputils.py
--- a/xarray/core/nputils.py
+++ b/xarray/core/nputils.py
@@ -220,6 +220,39 @@ def f(values, axis=None, **kwargs):
     return f
 
 
+def _nanpolyfit_1d(arr, x, rcond=None):
+    out = np.full((x.shape[1] + 1,), np.nan)
+    mask = np.isnan(arr)
+    if not np.all(mask):
+        out[:-1], out[-1], _, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
+    return out
+
+
+def least_squares(lhs, rhs, rcond=None, skipna=False):
+    if skipna:
+        added_dim = rhs.ndim == 1
+        if added_dim:
+            rhs = rhs.reshape(rhs.shape[0], 1)
+        nan_cols = np.any(np.isnan(rhs), axis=0)
+        out = np.empty((lhs.shape[1] + 1, rhs.shape[1]))
+        if np.any(nan_cols):
+            out[:, nan_cols] = np.apply_along_axis(
+                _nanpolyfit_1d, 0, rhs[:, nan_cols], lhs
+            )
+        if np.any(~nan_cols):
+            out[:-1, ~nan_cols], out[-1, ~nan_cols], _, _ = np.linalg.lstsq(
+                lhs, rhs[:, ~nan_cols], rcond=rcond
+            )
+        coeffs = out[:-1, :]
+        residuals = out[-1, :]
+        if added_dim:
+            coeffs = coeffs.reshape(coeffs.shape[0])
+            residuals = residuals.reshape(residuals.shape[0])
+    else:
+        coeffs, residuals, _, _ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
+    return coeffs, residuals
+
+
 nanmin = _create_bottleneck_method("nanmin")
 nanmax = _create_bottleneck_method("nanmax")
 nanmean = _create_bottleneck_method("nanmean")

```

## Test Patch

```diff
diff --git a/xarray/tests/test_computation.py b/xarray/tests/test_computation.py
--- a/xarray/tests/test_computation.py
+++ b/xarray/tests/test_computation.py
@@ -1120,3 +1120,35 @@ def test_where():
     actual = xr.where(cond, 1, 0)
     expected = xr.DataArray([1, 0], dims="x")
     assert_identical(expected, actual)
+
+
+@pytest.mark.parametrize("use_dask", [True, False])
+@pytest.mark.parametrize("use_datetime", [True, False])
+def test_polyval(use_dask, use_datetime):
+    if use_dask and not has_dask:
+        pytest.skip("requires dask")
+
+    if use_datetime:
+        xcoord = xr.DataArray(
+            pd.date_range("2000-01-01", freq="D", periods=10), dims=("x",), name="x"
+        )
+        x = xr.core.missing.get_clean_interp_index(xcoord, "x")
+    else:
+        xcoord = x = np.arange(10)
+
+    da = xr.DataArray(
+        np.stack((1.0 + x + 2.0 * x ** 2, 1.0 + 2.0 * x + 3.0 * x ** 2)),
+        dims=("d", "x"),
+        coords={"x": xcoord, "d": [0, 1]},
+    )
+    coeffs = xr.DataArray(
+        [[2, 1, 1], [3, 2, 1]],
+        dims=("d", "degree"),
+        coords={"d": [0, 1], "degree": [2, 1, 0]},
+    )
+    if use_dask:
+        coeffs = coeffs.chunk({"d": 2})
+
+    da_pv = xr.polyval(da.x, coeffs)
+
+    xr.testing.assert_allclose(da, da_pv.T)
diff --git a/xarray/tests/test_dataarray.py b/xarray/tests/test_dataarray.py
--- a/xarray/tests/test_dataarray.py
+++ b/xarray/tests/test_dataarray.py
@@ -23,6 +23,7 @@
     assert_array_equal,
     assert_equal,
     assert_identical,
+    has_dask,
     raises_regex,
     requires_bottleneck,
     requires_dask,
@@ -4191,6 +4192,55 @@ def test_rank(self):
         y = DataArray([0.75, 0.25, np.nan, 0.5, 1.0], dims=("z",))
         assert_equal(y.rank("z", pct=True), y)
 
+    @pytest.mark.parametrize("use_dask", [True, False])
+    @pytest.mark.parametrize("use_datetime", [True, False])
+    def test_polyfit(self, use_dask, use_datetime):
+        if use_dask and not has_dask:
+            pytest.skip("requires dask")
+        xcoord = xr.DataArray(
+            pd.date_range("1970-01-01", freq="D", periods=10), dims=("x",), name="x"
+        )
+        x = xr.core.missing.get_clean_interp_index(xcoord, "x")
+        if not use_datetime:
+            xcoord = x
+
+        da_raw = DataArray(
+            np.stack(
+                (10 + 1e-15 * x + 2e-28 * x ** 2, 30 + 2e-14 * x + 1e-29 * x ** 2)
+            ),
+            dims=("d", "x"),
+            coords={"x": xcoord, "d": [0, 1]},
+        )
+
+        if use_dask:
+            da = da_raw.chunk({"d": 1})
+        else:
+            da = da_raw
+
+        out = da.polyfit("x", 2)
+        expected = DataArray(
+            [[2e-28, 1e-15, 10], [1e-29, 2e-14, 30]],
+            dims=("d", "degree"),
+            coords={"degree": [2, 1, 0], "d": [0, 1]},
+        ).T
+        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)
+
+        # With NaN
+        da_raw[0, 1] = np.nan
+        if use_dask:
+            da = da_raw.chunk({"d": 1})
+        else:
+            da = da_raw
+        out = da.polyfit("x", 2, skipna=True, cov=True)
+        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)
+        assert "polyfit_covariance" in out
+
+        # Skipna + Full output
+        out = da.polyfit("x", 2, skipna=True, full=True)
+        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)
+        assert out.x_matrix_rank == 3
+        np.testing.assert_almost_equal(out.polyfit_residuals, [0, 0])
+
     def test_pad_constant(self):
         ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
         actual = ar.pad(dim_0=(1, 3))
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -5499,6 +5499,19 @@ def test_ipython_key_completion(self):
             ds.data_vars[item]  # should not raise
         assert sorted(actual) == sorted(expected)
 
+    def test_polyfit_output(self):
+        ds = create_test_data(seed=1)
+
+        out = ds.polyfit("dim2", 2, full=False)
+        assert "var1_polyfit_coefficients" in out
+
+        out = ds.polyfit("dim1", 2, full=True)
+        assert "var1_polyfit_coefficients" in out
+        assert "dim1_matrix_rank" in out
+
+        out = ds.polyfit("time", 2)
+        assert len(out.data_vars) == 0
+
     def test_pad(self):
         ds = create_test_data(seed=1)
         padded = ds.pad(dim2=(1, 1), constant_values=42)
diff --git a/xarray/tests/test_duck_array_ops.py b/xarray/tests/test_duck_array_ops.py
--- a/xarray/tests/test_duck_array_ops.py
+++ b/xarray/tests/test_duck_array_ops.py
@@ -16,6 +16,7 @@
     first,
     gradient,
     last,
+    least_squares,
     mean,
     np_timedelta64_to_float,
     pd_timedelta_to_float,
@@ -761,3 +762,20 @@ def test_timedelta_to_numeric(td):
     out = timedelta_to_numeric(td, "ns")
     np.testing.assert_allclose(out, 86400 * 1e9)
     assert isinstance(out, float)
+
+
+@pytest.mark.parametrize("use_dask", [True, False])
+@pytest.mark.parametrize("skipna", [True, False])
+def test_least_squares(use_dask, skipna):
+    if use_dask and not has_dask:
+        pytest.skip("requires dask")
+    lhs = np.array([[1, 2], [1, 2], [3, 2]])
+    rhs = DataArray(np.array([3, 5, 7]), dims=("y",))
+
+    if use_dask:
+        rhs = rhs.chunk({"y": 1})
+
+    coeffs, residuals = least_squares(lhs, rhs.data, skipna=skipna)
+
+    np.testing.assert_allclose(coeffs, [1.5, 1.25])
+    np.testing.assert_allclose(residuals, [2.0])

```


## Code snippets

### 1 - xarray/core/computation.py:

Start line: 776, End line: 978

```python
def apply_ufunc(
    func: Callable,
    *args: Any,
    input_core_dims: Sequence[Sequence] = None,
    output_core_dims: Optional[Sequence[Sequence]] = ((),),
    exclude_dims: AbstractSet = frozenset(),
    vectorize: bool = False,
    join: str = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = _NO_FILL_VALUE,
    keep_attrs: bool = False,
    kwargs: Mapping = None,
    dask: str = "forbidden",
    output_dtypes: Sequence = None,
    output_sizes: Mapping[Any, int] = None,
    meta: Any = None,
) -> Any:
    """Apply a vectorized function for unlabeled arrays on xarray objects.

    The function will be mapped over the data variable(s) of the input
    arguments using xarray's standard rules for labeled computation, including
    alignment, broadcasting, looping over GroupBy/Dataset variables, and
    merging of coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays
        (``.data``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs, you
        must set ``output_core_dims`` as well.
    *args : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    input_core_dims : Sequence[Sequence], optional
        List of the same length as ``args`` giving the list of core dimensions
        on each input argument that should not be broadcast. By default, we
        assume there are no core dimensions on any input arguments.

        For example, ``input_core_dims=[[], ['time']]`` indicates that all
        dimensions on the first argument and all dimensions other than 'time'
        on the second argument should be broadcast.

        Core dimensions are automatically moved to the last axes of input
        variables before applying ``func``, which facilitates using NumPy style
        generalized ufuncs [2]_.
    output_core_dims : List[tuple], optional
        List of the same length as the number of output arguments from
        ``func``, giving the list of core dimensions on each output that were
        not broadcast on the inputs. By default, we assume that ``func``
        outputs exactly one array, with axes corresponding to each broadcast
        dimension.

        Core dimensions are assumed to appear as the last dimensions of each
        output in the provided order.
    exclude_dims : set, optional
        Core dimensions on the inputs to exclude from alignment and
        broadcasting entirely. Any input coordinates along these dimensions
        will be dropped. Each excluded dimension must also appear in
        ``input_core_dims`` for at least one argument. Only dimensions listed
        here are allowed to change size between input and output objects.
    vectorize : bool, optional
        If True, then assume ``func`` only takes arrays defined over core
        dimensions as input and vectorize it automatically with
        :py:func:`numpy.vectorize`. This option exists for convenience, but is
        almost always slower than supplying a pre-vectorized function.
        Using this option requires NumPy version 1.12 or newer.
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        Method for joining the indexes of the passed objects along each
        dimension, and the variables of Dataset objects with mismatched
        data variables:

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        Method for joining variables of Dataset objects with mismatched
        data variables.

        - 'outer': take variables from both Dataset objects
        - 'inner': take only overlapped variables
        - 'left': take only variables from the first object
        - 'right': take only variables from the last object
        - 'exact': data variables on all Dataset objects must match exactly
    dataset_fill_value : optional
        Value used in place of missing variables on Dataset inputs when the
        datasets do not share the exact same ``data_vars``. Required if
        ``dataset_join not in {'inner', 'exact'}``, otherwise ignored.
    keep_attrs: boolean, Optional
        Whether to copy attributes from the first argument to the output.
    kwargs: dict, optional
        Optional keyword arguments passed directly on to call ``func``.
    dask: 'forbidden', 'allowed' or 'parallelized', optional
        How to handle applying to objects containing lazy data in the form of
        dask arrays:

        - 'forbidden' (default): raise an error if a dask array is encountered.
        - 'allowed': pass dask arrays directly on to ``func``.
        - 'parallelized': automatically parallelize ``func`` if any of the
          inputs are a dask array. If used, the ``output_dtypes`` argument must
          also be provided. Multiple output arguments are not yet supported.
    output_dtypes : list of dtypes, optional
        Optional list of output dtypes. Only used if dask='parallelized'.
    output_sizes : dict, optional
        Optional mapping from dimension names to sizes for outputs. Only used
        if dask='parallelized' and new dimensions (not found on inputs) appear
        on outputs.
    meta : optional
        Size-0 object representing the type of array wrapped by dask array. Passed on to
        ``dask.array.blockwise``.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    Examples
    --------

    Calculate the vector magnitude of two arguments:

    >>> def magnitude(a, b):
    ...     func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    ...     return xr.apply_ufunc(func, a, b)

    You can now apply ``magnitude()`` to ``xr.DataArray`` and ``xr.Dataset``
    objects, with automatically preserved dimensions and coordinates, e.g.,

    >>> array = xr.DataArray([1, 2, 3], coords=[("x", [0.1, 0.2, 0.3])])
    >>> magnitude(array, -array)
    <xarray.DataArray (x: 3)>
    array([1.414214, 2.828427, 4.242641])
    Coordinates:
      * x        (x) float64 0.1 0.2 0.3

    Plain scalars, numpy arrays and a mix of these with xarray objects is also
    supported:

    >>> magnitude(3, 4)
    5.0
    >>> magnitude(3, np.array([0, 4]))
    array([3., 5.])
    >>> magnitude(array, 0)
    <xarray.DataArray (x: 3)>
    array([1., 2., 3.])
    Coordinates:
      * x        (x) float64 0.1 0.2 0.3

    Other examples of how you could use ``apply_ufunc`` to write functions to
    (very nearly) replicate existing xarray functionality:

    Compute the mean (``.mean``) over one dimension::

        def mean(obj, dim):
            # note: apply always moves core dimensions to the end
            return apply_ufunc(np.mean, obj,
                               input_core_dims=[[dim]],
                               kwargs={'axis': -1})

    Inner product over a specific dimension (like ``xr.dot``)::

        def _inner(x, y):
            result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
            return result[..., 0, 0]

        def inner_product(a, b, dim):
            return apply_ufunc(_inner, a, b, input_core_dims=[[dim], [dim]])

    Stack objects along a new dimension (like ``xr.concat``)::

        def stack(objects, dim, new_coord):
            # note: this version does not stack coordinates
            func = lambda *x: np.stack(x, axis=-1)
            result = apply_ufunc(func, *objects,
                                 output_core_dims=[[dim]],
                                 join='outer',
                                 dataset_fill_value=np.nan)
            result[dim] = new_coord
            return result

    If your function is not vectorized but can be applied only to core
    dimensions, you can use ``vectorize=True`` to turn into a vectorized
    function. This wraps :py:func:`numpy.vectorize`, so the operation isn't
    terribly fast. Here we'll use it to calculate the distance between
    empirical samples from two probability distributions, using a scipy
    function that needs to be applied to vectors::

        import scipy.stats

        def earth_mover_distance(first_samples,
                                 second_samples,
                                 dim='ensemble'):
            return apply_ufunc(scipy.stats.wasserstein_distance,
                               first_samples, second_samples,
                               input_core_dims=[[dim], [dim]],
                               vectorize=True)

    Most of NumPy's builtin functions already broadcast their inputs
    appropriately for use in `apply`. You may find helper functions such as
    numpy.broadcast_arrays helpful in writing your function. `apply_ufunc` also
    works well with numba's vectorize and guvectorize. Further explanation with
    examples are provided in the xarray documentation [3]_.

    See also
    --------
    numpy.broadcast_arrays
    numba.vectorize
    numba.guvectorize

    References
    ----------
    .. [1] http://docs.scipy.org/doc/numpy/reference/ufuncs.html
    .. [2] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
    .. [3] http://xarray.pydata.org/en/stable/computation.html#wrapping-custom-computation
    """
    # ... other code
```
### 2 - xarray/core/computation.py:

Start line: 979, End line: 1069

```python
def apply_ufunc(
    func: Callable,
    *args: Any,
    input_core_dims: Sequence[Sequence] = None,
    output_core_dims: Optional[Sequence[Sequence]] = ((),),
    exclude_dims: AbstractSet = frozenset(),
    vectorize: bool = False,
    join: str = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = _NO_FILL_VALUE,
    keep_attrs: bool = False,
    kwargs: Mapping = None,
    dask: str = "forbidden",
    output_dtypes: Sequence = None,
    output_sizes: Mapping[Any, int] = None,
    meta: Any = None,
) -> Any:
    from .groupby import GroupBy
    from .dataarray import DataArray
    from .variable import Variable

    if input_core_dims is None:
        input_core_dims = ((),) * (len(args))
    elif len(input_core_dims) != len(args):
        raise ValueError(
            "input_core_dims must be None or a tuple with the length same to "
            "the number of arguments. Given input_core_dims: {}, "
            "number of args: {}.".format(input_core_dims, len(args))
        )

    if kwargs is None:
        kwargs = {}

    signature = _UFuncSignature(input_core_dims, output_core_dims)

    if exclude_dims and not exclude_dims <= signature.all_core_dims:
        raise ValueError(
            "each dimension in `exclude_dims` must also be a "
            "core dimension in the function signature"
        )

    if kwargs:
        func = functools.partial(func, **kwargs)

    if vectorize:
        if meta is None:
            # set meta=np.ndarray by default for numpy vectorized functions
            # work around dask bug computing meta with vectorized functions: GH5642
            meta = np.ndarray

        if signature.all_core_dims:
            func = np.vectorize(
                func, otypes=output_dtypes, signature=signature.to_gufunc_string()
            )
        else:
            func = np.vectorize(func, otypes=output_dtypes)

    variables_vfunc = functools.partial(
        apply_variable_ufunc,
        func,
        signature=signature,
        exclude_dims=exclude_dims,
        keep_attrs=keep_attrs,
        dask=dask,
        output_dtypes=output_dtypes,
        output_sizes=output_sizes,
        meta=meta,
    )

    if any(isinstance(a, GroupBy) for a in args):
        this_apply = functools.partial(
            apply_ufunc,
            func,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            exclude_dims=exclude_dims,
            join=join,
            dataset_join=dataset_join,
            dataset_fill_value=dataset_fill_value,
            keep_attrs=keep_attrs,
            dask=dask,
            meta=meta,
        )
        return apply_groupby_func(this_apply, *args)
    elif any(is_dict_like(a) for a in args):
        return apply_dataset_vfunc(
            variables_vfunc,
            *args,
            signature=signature,
            join=join,
            exclude_dims=exclude_dims,
            dataset_join=dataset_join,
            fill_value=dataset_fill_value,
            keep_attrs=keep_attrs,
        )
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray_vfunc(
            variables_vfunc,
            *args,
            signature=signature,
            join=join,
            exclude_dims=exclude_dims,
            keep_attrs=keep_attrs,
        )
    elif any(isinstance(a, Variable) for a in args):
        return variables_vfunc(*args)
    else:
        return apply_array_ufunc(func, *args, dask=dask)
```
### 3 - xarray/core/common.py:

Start line: 485, End line: 610

```python
class DataWithCoords(SupportsArithmetic, AttrAccessMixin):

    def pipe(
        self,
        func: Union[Callable[..., T], Tuple[Callable[..., T], str]],
        *args,
        **kwargs,
    ) -> T:
        """
        Apply ``func(self, *args, **kwargs)``

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : function
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        args : positional arguments passed into ``func``.
        kwargs : a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----

        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        >>> f(g(h(ds), arg1=a), arg2=b, arg3=c)

        You can write

        >>> (ds.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        >>> (ds.pipe(h).pipe(g, arg1=a).pipe((f, "arg2"), arg1=a, arg3=c))

        Examples
        --------

        >>> import numpy as np
        >>> import xarray as xr
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lat            (lat) int64 10 20
        * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 14.53 11.85 19.27 16.37
            precipitation  (lat, lon) float64 0.7315 0.7189 0.8481 0.4671

        >>> def adder(data, arg):
        ...     return data + arg
        ...
        >>> def div(data, arg):
        ...     return data / arg
        ...
        >>> def sub_mult(data, sub_arg, mult_arg):
        ...     return (data * mult_arg) - sub_arg
        ...
        >>> x.pipe(adder, 2)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lon            (lon) int64 150 160
        * lat            (lat) int64 10 20
        Data variables:
            temperature_c  (lat, lon) float64 16.53 13.85 21.27 18.37
            precipitation  (lat, lon) float64 2.731 2.719 2.848 2.467

        >>> x.pipe(adder, arg=2)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lon            (lon) int64 150 160
        * lat            (lat) int64 10 20
        Data variables:
            temperature_c  (lat, lon) float64 16.53 13.85 21.27 18.37
            precipitation  (lat, lon) float64 2.731 2.719 2.848 2.467

        >>> (
        ...     x.pipe(adder, arg=2)
        ...     .pipe(div, arg=2)
        ...     .pipe(sub_mult, sub_arg=2, mult_arg=2)
        ... )
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lon            (lon) int64 150 160
        * lat            (lat) int64 10 20
        Data variables:
            temperature_c  (lat, lon) float64 14.53 11.85 19.27 16.37
            precipitation  (lat, lon) float64 0.7315 0.7189 0.8481 0.4671

        See Also
        --------
        pandas.DataFrame.pipe
        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError(
                    "%s is both the pipe target and a keyword " "argument" % target
                )
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)
```
### 4 - xarray/core/computation.py:

Start line: 759, End line: 1069

```python
def apply_ufunc(
    func: Callable,
    *args: Any,
    input_core_dims: Sequence[Sequence] = None,
    output_core_dims: Optional[Sequence[Sequence]] = ((),),
    exclude_dims: AbstractSet = frozenset(),
    vectorize: bool = False,
    join: str = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = _NO_FILL_VALUE,
    keep_attrs: bool = False,
    kwargs: Mapping = None,
    dask: str = "forbidden",
    output_dtypes: Sequence = None,
    output_sizes: Mapping[Any, int] = None,
    meta: Any = None,
) -> Any:
    # ... other code
```
### 5 - xarray/core/parallel.py:

Start line: 105, End line: 202

```python
def map_blocks(
    func: Callable[..., T_DSorDA],
    obj: Union[DataArray, Dataset],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = None,
) -> T_DSorDA:
    """Apply a function to each chunk of a DataArray or Dataset. This function is
    experimental and its signature may change.

    Parameters
    ----------
    func: callable
        User-provided function that accepts a DataArray or Dataset as its first
        parameter. The function will receive a subset of 'obj' (see below),
        corresponding to one chunk along each chunked dimension. ``func`` will be
        executed as ``func(obj_subset, *args, **kwargs)``.

        The function will be first run on mocked-up data, that looks like 'obj' but
        has sizes 0, to determine properties of the returned object such as dtype,
        variable names, new dimensions and new indexes (if any).

        This function must return either a single DataArray or a single Dataset.

        This function cannot change size of existing dimensions, or add new chunked
        dimensions.
    obj: DataArray, Dataset
        Passed to the function as its first argument, one dask chunk at a time.
    args: Sequence
        Passed verbatim to func after unpacking, after the sliced obj. xarray objects,
        if any, will not be split by chunks. Passing dask collections is not allowed.
    kwargs: Mapping
        Passed verbatim to func after unpacking. xarray objects, if any, will not be
        split by chunks. Passing dask collections is not allowed.

    Returns
    -------
    A single DataArray or Dataset with dask backend, reassembled from the outputs of the
    function.

    Notes
    -----
    This function is designed for when one needs to manipulate a whole xarray object
    within each chunk. In the more common case where one can work on numpy arrays, it is
    recommended to use apply_ufunc.

    If none of the variables in obj is backed by dask, calling this function is
    equivalent to calling ``func(obj, *args, **kwargs)``.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks,
    xarray.DataArray.map_blocks

    Examples
    --------

    Calculate an anomaly from climatology using ``.groupby()``. Using
    ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
    its indices, and its methods like ``.groupby()``.

    >>> def calculate_anomaly(da, groupby_type="time.month"):
    ...     # Necessary workaround to xarray's check with zero dimensions
    ...     # https://github.com/pydata/xarray/issues/3575
    ...     if sum(da.shape) == 0:
    ...         return da
    ...     gb = da.groupby(groupby_type)
    ...     clim = gb.mean(dim="time")
    ...     return gb - clim
    >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")
    >>> np.random.seed(123)
    >>> array = xr.DataArray(
    ...     np.random.rand(len(time)), dims="time", coords=[time]
    ... ).chunk()
    >>> xr.map_blocks(calculate_anomaly, array).compute()
    <xarray.DataArray (time: 24)>
    array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,
            0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,
           -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,
            0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,
            0.07673453,  0.22865714,  0.19063865, -0.0590131 ])
    Coordinates:
      * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00

    Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
    to the function being applied in ``xr.map_blocks()``:

    >>> xr.map_blocks(
    ...     calculate_anomaly, array, kwargs={"groupby_type": "time.year"},
    ... )
    <xarray.DataArray (time: 24)>
    array([ 0.15361741, -0.25671244, -0.31600032,  0.008463  ,  0.1766172 ,
           -0.11974531,  0.43791243,  0.14197797, -0.06191987, -0.15073425,
           -0.19967375,  0.18619794, -0.05100474, -0.42989909, -0.09153273,
            0.24841842, -0.30708526, -0.31412523,  0.04197439,  0.0422506 ,
            0.14482397,  0.35985481,  0.23487834,  0.12144652])
    Coordinates:
        * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
    """
  # ... other code
```
### 6 - xarray/core/computation.py:

Start line: 736, End line: 756

```python
def apply_array_ufunc(func, *args, dask="forbidden"):
    """Apply a ndarray level function over ndarray objects."""
    if any(isinstance(arg, dask_array_type) for arg in args):
        if dask == "forbidden":
            raise ValueError(
                "apply_ufunc encountered a dask array on an "
                "argument, but handling for dask arrays has not "
                "been enabled. Either set the ``dask`` argument "
                "or load your data into memory first with "
                "``.load()`` or ``.compute()``"
            )
        elif dask == "parallelized":
            raise ValueError(
                "cannot use dask='parallelized' for apply_ufunc "
                "unless at least one input is an xarray object"
            )
        elif dask == "allowed":
            pass
        else:
            raise ValueError(f"unknown setting for dask array handling: {dask}")
    return func(*args)
```
### 7 - xarray/ufuncs.py:

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
from .core.duck_array_ops import _dask_or_eager_func
from .core.groupby import GroupBy as _GroupBy
from .core.pycompat import dask_array_type as _dask_array_type
from .core.variable import Variable as _Variable

_xarray_types = (_Variable, _DataArray, _Dataset, _GroupBy)
_dispatch_order = (_np.ndarray, _dask_array_type) + _xarray_types


def _dispatch_priority(obj):
    for priority, cls in enumerate(_dispatch_order):
        if isinstance(obj, cls):
            return priority
    return -1
```
### 8 - asv_bench/benchmarks/dataarray_missing.py:

Start line: 1, End line: 75

```python
import pandas as pd

import xarray as xr

from . import randn, requires_dask

try:
    import dask  # noqa: F401
except ImportError:
    pass


def make_bench_data(shape, frac_nan, chunks):
    vals = randn(shape, frac_nan)
    coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x", "y"), coords=coords)

    if chunks is not None:
        da = da.chunk(chunks)

    return da


def time_interpolate_na(shape, chunks, method, limit):
    if chunks is not None:
        requires_dask()
    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.interpolate_na(dim="time", method="linear", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_interpolate_na.param_names = ["shape", "chunks", "method", "limit"]
time_interpolate_na.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    ["linear", "spline", "quadratic", "cubic"],
    [None, 3],
)


def time_ffill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.ffill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_ffill.param_names = ["shape", "chunks", "limit"]
time_ffill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)


def time_bfill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.bfill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_bfill.param_names = ["shape", "chunks", "limit"]
time_bfill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)
```
### 9 - xarray/core/computation.py:

Start line: 542, End line: 649

```python
def apply_variable_ufunc(
    func,
    *args,
    signature,
    exclude_dims=frozenset(),
    dask="forbidden",
    output_dtypes=None,
    output_sizes=None,
    keep_attrs=False,
    meta=None,
):
    """Apply a ndarray level function over Variable and/or ndarray objects.
    """
    from .variable import Variable, as_compatible_data

    dim_sizes = unified_dim_sizes(
        (a for a in args if hasattr(a, "dims")), exclude_dims=exclude_dims
    )
    broadcast_dims = tuple(
        dim for dim in dim_sizes if dim not in signature.all_core_dims
    )
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]

    input_data = [
        broadcast_compat_data(arg, broadcast_dims, core_dims)
        if isinstance(arg, Variable)
        else arg
        for arg, core_dims in zip(args, signature.input_core_dims)
    ]

    if any(isinstance(array, dask_array_type) for array in input_data):
        if dask == "forbidden":
            raise ValueError(
                "apply_ufunc encountered a dask array on an "
                "argument, but handling for dask arrays has not "
                "been enabled. Either set the ``dask`` argument "
                "or load your data into memory first with "
                "``.load()`` or ``.compute()``"
            )
        elif dask == "parallelized":
            input_dims = [broadcast_dims + dims for dims in signature.input_core_dims]
            numpy_func = func

            def func(*arrays):
                return _apply_blockwise(
                    numpy_func,
                    arrays,
                    input_dims,
                    output_dims,
                    signature,
                    output_dtypes,
                    output_sizes,
                    meta,
                )

        elif dask == "allowed":
            pass
        else:
            raise ValueError(
                "unknown setting for dask array handling in "
                "apply_ufunc: {}".format(dask)
            )
    result_data = func(*input_data)

    if signature.num_outputs == 1:
        result_data = (result_data,)
    elif (
        not isinstance(result_data, tuple) or len(result_data) != signature.num_outputs
    ):
        raise ValueError(
            "applied function does not have the number of "
            "outputs specified in the ufunc signature. "
            "Result is not a tuple of {} elements: {!r}".format(
                signature.num_outputs, result_data
            )
        )

    output = []
    for dims, data in zip(output_dims, result_data):
        data = as_compatible_data(data)
        if data.ndim != len(dims):
            raise ValueError(
                "applied function returned data with unexpected "
                "number of dimensions: {} vs {}, for dimensions {}".format(
                    data.ndim, len(dims), dims
                )
            )

        var = Variable(dims, data, fastpath=True)
        for dim, new_size in var.sizes.items():
            if dim in dim_sizes and new_size != dim_sizes[dim]:
                raise ValueError(
                    "size of dimension {!r} on inputs was unexpectedly "
                    "changed by applied function from {} to {}. Only "
                    "dimensions specified in ``exclude_dims`` with "
                    "xarray.apply_ufunc are allowed to change size.".format(
                        dim, dim_sizes[dim], new_size
                    )
                )

        if keep_attrs and isinstance(args[0], Variable):
            var.attrs.update(args[0].attrs)
        output.append(var)

    if signature.num_outputs == 1:
        return output[0]
    else:
        return tuple(output)
```
### 10 - xarray/core/computation.py:

Start line: 414, End line: 457

```python
def apply_groupby_func(func, *args):
    """Apply a dataset or datarray level function over GroupBy, Dataset,
    DataArray, Variable and/or ndarray objects.
    """
    from .groupby import GroupBy, peek_at
    from .variable import Variable

    groupbys = [arg for arg in args if isinstance(arg, GroupBy)]
    assert groupbys, "must have at least one groupby to iterate over"
    first_groupby = groupbys[0]
    if any(not first_groupby._group.equals(gb._group) for gb in groupbys[1:]):
        raise ValueError(
            "apply_ufunc can only perform operations over "
            "multiple GroupBy objets at once if they are all "
            "grouped the same way"
        )

    grouped_dim = first_groupby._group.name
    unique_values = first_groupby._unique_coord.values

    iterators = []
    for arg in args:
        if isinstance(arg, GroupBy):
            iterator = (value for _, value in arg)
        elif hasattr(arg, "dims") and grouped_dim in arg.dims:
            if isinstance(arg, Variable):
                raise ValueError(
                    "groupby operations cannot be performed with "
                    "xarray.Variable objects that share a dimension with "
                    "the grouped dimension"
                )
            iterator = _iter_over_selections(arg, grouped_dim, unique_values)
        else:
            iterator = itertools.repeat(arg)
        iterators.append(iterator)

    applied = (func(*zipped_args) for zipped_args in zip(*iterators))
    applied_example, applied = peek_at(applied)
    combine = first_groupby._combine
    if isinstance(applied_example, tuple):
        combined = tuple(combine(output) for output in zip(*applied))
    else:
        combined = combine(applied)
    return combined
```
### 11 - xarray/core/computation.py:

Start line: 652, End line: 733

```python
def _apply_blockwise(
    func,
    args,
    input_dims,
    output_dims,
    signature,
    output_dtypes,
    output_sizes=None,
    meta=None,
):
    import dask.array

    if signature.num_outputs > 1:
        raise NotImplementedError(
            "multiple outputs from apply_ufunc not yet "
            "supported with dask='parallelized'"
        )

    if output_dtypes is None:
        raise ValueError(
            "output dtypes (output_dtypes) must be supplied to "
            "apply_func when using dask='parallelized'"
        )
    if not isinstance(output_dtypes, list):
        raise TypeError(
            "output_dtypes must be a list of objects coercible to "
            "numpy dtypes, got {}".format(output_dtypes)
        )
    if len(output_dtypes) != signature.num_outputs:
        raise ValueError(
            "apply_ufunc arguments output_dtypes and "
            "output_core_dims must have the same length: {} vs {}".format(
                len(output_dtypes), signature.num_outputs
            )
        )
    (dtype,) = output_dtypes

    if output_sizes is None:
        output_sizes = {}

    new_dims = signature.all_output_core_dims - signature.all_input_core_dims
    if any(dim not in output_sizes for dim in new_dims):
        raise ValueError(
            "when using dask='parallelized' with apply_ufunc, "
            "output core dimensions not found on inputs must "
            "have explicitly set sizes with ``output_sizes``: {}".format(new_dims)
        )

    for n, (data, core_dims) in enumerate(zip(args, signature.input_core_dims)):
        if isinstance(data, dask_array_type):
            # core dimensions cannot span multiple chunks
            for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                if len(data.chunks[axis]) != 1:
                    raise ValueError(
                        "dimension {!r} on {}th function argument to "
                        "apply_ufunc with dask='parallelized' consists of "
                        "multiple chunks, but is also a core dimension. To "
                        "fix, rechunk into a single dask array chunk along "
                        "this dimension, i.e., ``.chunk({})``, but beware "
                        "that this may significantly increase memory usage.".format(
                            dim, n, {dim: -1}
                        )
                    )

    (out_ind,) = output_dims

    blockwise_args = []
    for arg, dims in zip(args, input_dims):
        # skip leading dimensions that are implicitly added by broadcasting
        ndim = getattr(arg, "ndim", 0)
        trimmed_dims = dims[-ndim:] if ndim else ()
        blockwise_args.extend([arg, trimmed_dims])

    return dask.array.blockwise(
        func,
        out_ind,
        *blockwise_args,
        dtype=dtype,
        concatenate=True,
        new_axes=output_sizes,
        meta=meta,
    )
```
### 15 - xarray/core/dataarray.py:

Start line: 3221, End line: 3276

```python
class DataArray(AbstractArray, DataWithCoords):

    def map_blocks(
        self,
        func: "Callable[..., T_DSorDA]",
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = None,
    ) -> "T_DSorDA":
        """
        Apply a function to each chunk of this DataArray. This method is experimental
        and its signature may change.

        Parameters
        ----------
        func: callable
            User-provided function that accepts a DataArray as its first parameter. The
            function will receive a subset of this DataArray, corresponding to one chunk
            along each chunked dimension. ``func`` will be executed as
            ``func(obj_subset, *args, **kwargs)``.

            The function will be first run on mocked-up data, that looks like this array
            but has sizes 0, to determine properties of the returned object such as
            dtype, variable names, new dimensions and new indexes (if any).

            This function must return either a single DataArray or a single Dataset.

            This function cannot change size of existing dimensions, or add new chunked
            dimensions.
        args: Sequence
            Passed verbatim to func after unpacking, after the sliced DataArray. xarray
            objects, if any, will not be split by chunks. Passing dask collections is
            not allowed.
        kwargs: Mapping
            Passed verbatim to func after unpacking. xarray objects, if any, will not be
            split by chunks. Passing dask collections is not allowed.

        Returns
        -------
        A single DataArray or Dataset with dask backend, reassembled from the outputs of
        the function.

        Notes
        -----
        This method is designed for when one needs to manipulate a whole xarray object
        within each chunk. In the more common case where one can work on numpy arrays,
        it is recommended to use apply_ufunc.

        If none of the variables in this DataArray is backed by dask, calling this
        method is equivalent to calling ``func(self, *args, **kwargs)``.

        See Also
        --------
        dask.array.map_blocks, xarray.apply_ufunc, xarray.map_blocks,
        xarray.Dataset.map_blocks
        """
        from .parallel import map_blocks

        return map_blocks(func, self, args, kwargs)
```
### 16 - xarray/core/computation.py:

Start line: 213, End line: 244

```python
def apply_dataarray_vfunc(
    func, *args, signature, join="inner", exclude_dims=frozenset(), keep_attrs=False
):
    """Apply a variable level function over DataArray, Variable and/or ndarray
    objects.
    """
    from .dataarray import DataArray

    if len(args) > 1:
        args = deep_align(
            args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False
        )

    if keep_attrs and hasattr(args[0], "name"):
        name = args[0].name
    else:
        name = result_name(args)
    result_coords = build_output_coords(args, signature, exclude_dims)

    data_vars = [getattr(a, "variable", a) for a in args]
    result_var = func(*data_vars)

    if signature.num_outputs > 1:
        out = tuple(
            DataArray(variable, coords, name=name, fastpath=True)
            for variable, coords in zip(result_var, result_coords)
        )
    else:
        (coords,) = result_coords
        out = DataArray(result_var, coords, name=name, fastpath=True)

    return out
```
### 17 - xarray/core/dataarray.py:

Start line: 1, End line: 82

```python
import datetime
import functools
import warnings
from numbers import Number
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
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..plot.plot import _PlotMethods
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
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, _extract_indexes_from_coords
from .options import OPTIONS
from .utils import Default, ReprObject, _check_inplace, _default, either_dict_or_kwargs
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

if TYPE_CHECKING:
    T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)

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
```
### 18 - xarray/core/computation.py:

Start line: 1, End line: 40

```python
"""
Functions for applying functions that act on arrays to xarray's labeled data.
"""
import functools
import itertools
import operator
from collections import Counter
from typing import (
    TYPE_CHECKING,
    AbstractSet,
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
)

import numpy as np

from . import duck_array_ops, utils
from .alignment import deep_align
from .merge import merge_coordinates_without_align
from .options import OPTIONS
from .pycompat import dask_array_type
from .utils import is_dict_like
from .variable import Variable

if TYPE_CHECKING:
    from .coordinates import Coordinates  # noqa
    from .dataset import Dataset

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})
```
### 19 - xarray/core/dataarray.py:

Start line: 1317, End line: 1380

```python
class DataArray(AbstractArray, DataWithCoords):

    def interp(
        self,
        coords: Mapping[Hashable, Any] = None,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
        **coords_kwargs: Any,
    ) -> "DataArray":
        """ Multidimensional interpolation of variables.

        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            new coordinate can be an scalar, array-like or DataArray.
            If DataArrays are passed as new coordates, their dimensions are
            used for the broadcasting.
        method: {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array.
        assume_sorted: boolean, optional
            If False, values of x can be in any order and they are sorted
            first. If True, x has to be an array of monotonically increasing
            values.
        kwargs: dictionary
            Additional keyword passed to scipy's interpolator.
        ``**coords_kwargs`` : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated: xr.DataArray
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
        >>> da = xr.DataArray([1, 3], [("x", np.arange(2))])
        >>> da.interp(x=0.5)
        <xarray.DataArray ()>
        array(2.0)
        Coordinates:
            x        float64 0.5
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
```
### 20 - xarray/core/computation.py:

Start line: 1153, End line: 1215

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
        "..." + "".join([dim_map[d] for d in ds]) for ds in input_core_dims
    ]
    subscripts = ",".join(subscripts_list)
    subscripts += "->..." + "".join([dim_map[d] for d in output_core_dims[0]])

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
    return result.transpose(*[d for d in all_dims if d in result.dims])
```
### 21 - xarray/core/dask_array_ops.py:

Start line: 30, End line: 98

```python
def rolling_window(a, axis, window, center, fill_value):
    """Dask's equivalence to np.utils.rolling_window
    """
    import dask.array as da

    orig_shape = a.shape
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = int(window / 2)
    # For evenly sized window, we need to crop the first point of each block.
    offset = 1 if window % 2 == 0 else 0

    if depth[axis] > min(a.chunks[axis]):
        raise ValueError(
            "For window size %d, every chunk should be larger than %d, "
            "but the smallest chunk size is %d. Rechunk your array\n"
            "with a larger chunk size or a chunk size that\n"
            "more evenly divides the shape of your array."
            % (window, depth[axis], min(a.chunks[axis]))
        )

    # Although da.overlap pads values to boundaries of the array,
    # the size of the generated array is smaller than what we want
    # if center == False.
    if center:
        start = int(window / 2)  # 10 -> 5,  9 -> 4
        end = window - 1 - start
    else:
        start, end = window - 1, 0
    pad_size = max(start, end) + offset - depth[axis]
    drop_size = 0
    # pad_size becomes more than 0 when the overlapped array is smaller than
    # needed. In this case, we need to enlarge the original array by padding
    # before overlapping.
    if pad_size > 0:
        if pad_size < depth[axis]:
            # overlapping requires each chunk larger than depth. If pad_size is
            # smaller than the depth, we enlarge this and truncate it later.
            drop_size = depth[axis] - pad_size
            pad_size = depth[axis]
        shape = list(a.shape)
        shape[axis] = pad_size
        chunks = list(a.chunks)
        chunks[axis] = (pad_size,)
        fill_array = da.full(shape, fill_value, dtype=a.dtype, chunks=chunks)
        a = da.concatenate([fill_array, a], axis=axis)

    boundary = {d: fill_value for d in range(a.ndim)}

    # create overlap arrays
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)

    # apply rolling func
    def func(x, window, axis=-1):
        x = np.asarray(x)
        rolling = nputils._rolling_window(x, window, axis)
        return rolling[(slice(None),) * axis + (slice(offset, None),)]

    chunks = list(a.chunks)
    chunks.append(window)
    out = ag.map_blocks(
        func, dtype=a.dtype, new_axis=a.ndim, chunks=chunks, window=window, axis=axis
    )

    # crop boundary.
    index = (slice(None),) * axis + (slice(drop_size, drop_size + orig_shape[axis]),)
    return out[index]
```
### 22 - xarray/core/dataset.py:

Start line: 1, End line: 132

```python
import copy
import datetime
import functools
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
    TypeVar,
    Union,
    cast,
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
from .common import (
    DataWithCoords,
    ImplementsDatasetReduce,
    _contains_datetime_like_objects,
)
from .coordinates import (
    DatasetCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes,
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
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .utils import (
    Default,
    Frozen,
    SortedKeysDict,
    _check_inplace,
    _default,
    decode_numpy_dict_values,
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

    T_DSorDA = TypeVar("T_DSorDA", DataArray, "Dataset")

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
### 26 - xarray/core/dataarray.py:

Start line: 3153, End line: 3219

```python
class DataArray(AbstractArray, DataWithCoords):

    def integrate(
        self, dim: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
    ) -> "DataArray":
        """ integrate the array with the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. dim
            must be one dimensional.

        Parameters
        ----------
        dim: hashable, or a sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit: str, optional
            Can be used to specify the unit if datetime coordinate is used.
            One of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps',
            'fs', 'as'}

        Returns
        -------
        integrated: DataArray

        See also
        --------
        numpy.trapz: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.integrate("x")
        <xarray.DataArray (y: 3)>
        array([5.4, 6.6, 7.8])
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().integrate(dim, datetime_unit)
        return self._from_temp_dataset(ds)

    def unify_chunks(self) -> "DataArray":
        """ Unify chunk size along all chunked dimensions of this DataArray.

        Returns
        -------

        DataArray with consistent chunk sizes for all dask-array variables

        See Also
        --------

        dask.array.core.unify_chunks
        """
        ds = self._to_temp_dataset().unify_chunks()
        return self._from_temp_dataset(ds)
```
### 27 - xarray/core/computation.py:

Start line: 349, End line: 411

```python
def apply_dataset_vfunc(
    func,
    *args,
    signature,
    join="inner",
    dataset_join="exact",
    fill_value=_NO_FILL_VALUE,
    exclude_dims=frozenset(),
    keep_attrs=False,
):
    """Apply a variable level function over Dataset, dict of DataArray,
    DataArray, Variable and/or ndarray objects.
    """
    from .dataset import Dataset

    first_obj = args[0]  # we'll copy attrs from this in case keep_attrs=True

    if dataset_join not in _JOINS_WITHOUT_FILL_VALUES and fill_value is _NO_FILL_VALUE:
        raise TypeError(
            "to apply an operation to datasets with different "
            "data variables with apply_ufunc, you must supply the "
            "dataset_fill_value argument."
        )

    if len(args) > 1:
        args = deep_align(
            args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False
        )

    list_of_coords = build_output_coords(args, signature, exclude_dims)
    args = [getattr(arg, "data_vars", arg) for arg in args]

    result_vars = apply_dict_of_variables_vfunc(
        func, *args, signature=signature, join=dataset_join, fill_value=fill_value
    )

    if signature.num_outputs > 1:
        out = tuple(_fast_dataset(*args) for args in zip(result_vars, list_of_coords))
    else:
        (coord_vars,) = list_of_coords
        out = _fast_dataset(result_vars, coord_vars)

    if keep_attrs and isinstance(first_obj, Dataset):
        if isinstance(out, tuple):
            out = tuple(ds._copy_attrs_from(first_obj) for ds in out)
        else:
            out._copy_attrs_from(first_obj)
    return out


def _iter_over_selections(obj, dim, values):
    """Iterate over selections of an xarray object in the provided order."""
    from .groupby import _dummy_copy

    dummy = None
    for value in values:
        try:
            obj_sel = obj.sel(**{dim: value})
        except (KeyError, IndexError):
            if dummy is None:
                dummy = _dummy_copy(obj)
            obj_sel = dummy
        yield obj_sel
```
### 30 - xarray/core/dataarray.py:

Start line: 1382, End line: 1435

```python
class DataArray(AbstractArray, DataWithCoords):

    def interp_like(
        self,
        other: Union["DataArray", Dataset],
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
    ) -> "DataArray":
        """Interpolate this object onto the coordinates of another object,
        filling out of range values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset.
        method: string, optional.
            {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array. 'linear' is used by default.
        assume_sorted: boolean, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs: dictionary, optional
            Additional keyword passed to scipy's interpolator.

        Returns
        -------
        interpolated: xr.DataArray
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
```
### 32 - xarray/core/dataarray.py:

Start line: 2622, End line: 2641

```python
class DataArray(AbstractArray, DataWithCoords):

    @staticmethod
    def _inplace_binary_op(f: Callable) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a DataArray and "
                    "a grouped object are not permitted"
                )
            # n.b. we can't align other to self (with other.reindex_like(self))
            # because `other` may be converted into floats, which would cause
            # in-place arithmetic to fail unpredictably. Instead, we simply
            # don't support automatic alignment with in-place arithmetic.
            other_coords = getattr(other, "coords", None)
            other_variable = getattr(other, "variable", other)
            with self.coords._merge_inplace(other_coords):
                f(self.variable, other_variable)
            return self

        return func
```
### 34 - xarray/core/duck_array_ops.py:

Start line: 539, End line: 600

```python
mean.numeric_only = True  # type: ignore


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


_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg="%r with skipna=True is not yet implemented on dask arrays",
)


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanlast(values, axis)
    return take(values, -1, axis=axis)


def rolling_window(array, axis, window, center, fill_value):
    """
    Make an ndarray with a rolling window of axis-th dimension.
    The rolling dimension will be placed at the last dimension.
    """
    if isinstance(array, dask_array_type):
        return dask_array_ops.rolling_window(array, axis, window, center, fill_value)
    else:  # np.ndarray
        return nputils.rolling_window(array, axis, window, center, fill_value)
```
### 35 - xarray/core/duck_array_ops.py:

Start line: 108, End line: 132

```python
def notnull(data):
    return ~isnull(data)


transpose = _dask_or_eager_func("transpose")
_where = _dask_or_eager_func("where", array_args=slice(3))
isin = _dask_or_eager_func("isin", array_args=slice(2))
take = _dask_or_eager_func("take")
broadcast_to = _dask_or_eager_func("broadcast_to")
pad = _dask_or_eager_func("pad", dask_module=dask_array_compat)

_concatenate = _dask_or_eager_func("concatenate", list_of_args=True)
_stack = _dask_or_eager_func("stack", list_of_args=True)

array_all = _dask_or_eager_func("all")
array_any = _dask_or_eager_func("any")

tensordot = _dask_or_eager_func("tensordot", array_args=slice(2))
einsum = _dask_or_eager_func("einsum", array_args=slice(1, None))


def gradient(x, coord, axis, edge_order):
    if isinstance(x, dask_array_type):
        return dask_array.gradient(x, coord, axis=axis, edge_order=edge_order)
    return np.gradient(x, coord, axis=axis, edge_order=edge_order)
```
### 38 - xarray/core/dataarray.py:

Start line: 3294, End line: 3453

```python
class DataArray(AbstractArray, DataWithCoords):

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> "DataArray":
        """Pad this array along one or more dimensions.

        .. warning::
            This function is experimental and its behaviour is likely to change
            especially regarding padding of dimension coordinates (or IndexVariables).

        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),
        coordinates will be padded with the same mode, otherwise coordinates
        are padded using the "constant" mode with fill_value dtypes.NA.

        Parameters
        ----------
        pad_width : Mapping with the form of {dim: (pad_before, pad_after)}
            Number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : str
            One of the following string values (taken from numpy docs)

            'constant' (default)
                Pads with a constant value.
            'edge'
                Pads with the edge values of array.
            'linear_ramp'
                Pads with the linear ramp between end_value and the
                array edge value.
            'maximum'
                Pads with the maximum value of all or part of the
                vector along each axis.
            'mean'
                Pads with the mean value of all or part of the
                vector along each axis.
            'median'
                Pads with the median value of all or part of the
                vector along each axis.
            'minimum'
                Pads with the minimum value of all or part of the
                vector along each axis.
            'reflect'
                Pads with the reflection of the vector mirrored on
                the first and last values of the vector along each
                axis.
            'symmetric'
                Pads with the reflection of the vector mirrored
                along the edge of the array.
            'wrap'
                Pads with the wrap of the vector along the axis.
                The first values are used to pad the end and the
                end values are used to pad the beginning.
        stat_length : int, tuple or mapping of the form {dim: tuple}
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
            statistic lengths along each dimension.
            ((before, after),) yields same before and after statistic lengths
            for each dimension.
            (stat_length,) or int is a shortcut for before = after = statistic
            length for all axes.
            Default is ``None``, to use the entire axis.
        constant_values : scalar, tuple or mapping of the form {dim: tuple}
            Used in 'constant'.  The values to set the padded values for each
            axis.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            pad constants along each dimension.
            ``((before, after),)`` yields same before and after constants for each
            dimension.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all dimensions.
            Default is 0.
        end_values : scalar, tuple or mapping of the form {dim: tuple}
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            end values along each dimension.
            ``((before, after),)`` yields same before and after end values for each
            axis.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all axes.
            Default is 0.
        reflect_type : {'even', 'odd'}, optional
            Used in 'reflect', and 'symmetric'.  The 'even' style is the
            default with an unaltered reflection around the edge value.  For
            the 'odd' style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        **pad_width_kwargs:
            The keyword arguments form of ``pad_width``.
            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

        Returns
        -------
        padded : DataArray
            DataArray with the padded coordinates and data.

        See also
        --------
        DataArray.shift, DataArray.roll, DataArray.bfill, DataArray.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        By default when ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``. To avoid type promotion
        specify ``constant_values=np.nan``

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], coords=[("x", [0,1,2])])
        >>> arr.pad(x=(1,2), constant_values=0)
        <xarray.DataArray (x: 6)>
        array([0, 5, 6, 7, 0, 0])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 2.0 nan nan

        >>> da = xr.DataArray([[0,1,2,3], [10,11,12,13]],
                              dims=["x", "y"],
                              coords={"x": [0,1], "y": [10, 20 ,30, 40], "z": ("x", [100, 200])}
            )
        >>> da.pad(x=1)
        <xarray.DataArray (x: 4, y: 4)>
        array([[nan, nan, nan, nan],
               [ 0.,  1.,  2.,  3.],
               [10., 11., 12., 13.],
               [nan, nan, nan, nan]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan
        >>> da.pad(x=1, constant_values=np.nan)
        <xarray.DataArray (x: 4, y: 4)>
        array([[-9223372036854775808, -9223372036854775808, -9223372036854775808,
                -9223372036854775808],
               [                   0,                    1,                    2,
                                   3],
               [                  10,                   11,                   12,
                                  13],
               [-9223372036854775808, -9223372036854775808, -9223372036854775808,
                -9223372036854775808]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan
        """
        ds = self._to_temp_dataset().pad(
            pad_width=pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            **pad_width_kwargs,
        )
        return self._from_temp_dataset(ds)

    # this needs to be at the end, or mypy will confuse with `str`
    # https://mypy.readthedocs.io/en/latest/common_issues.html#dealing-with-conflicting-names
    str = property(StringAccessor)


# priority most be higher than Variable to properly work with binary ufuncs
ops.inject_all_ops_and_reduce_methods(DataArray, priority=60)
```
### 40 - xarray/core/computation.py:

Start line: 315, End line: 346

```python
def apply_dict_of_variables_vfunc(
    func, *args, signature, join="inner", fill_value=None
):
    """Apply a variable level function over dicts of DataArray, DataArray,
    Variable and ndarray objects.
    """
    args = [_as_variables_or_variable(arg) for arg in args]
    names = join_dict_keys(args, how=join)
    grouped_by_name = collect_dict_values(args, names, fill_value)

    result_vars = {}
    for name, variable_args in zip(names, grouped_by_name):
        result_vars[name] = func(*variable_args)

    if signature.num_outputs > 1:
        return _unpack_dict_tuples(result_vars, signature.num_outputs)
    else:
        return result_vars


def _fast_dataset(
    variables: Dict[Hashable, Variable], coord_variables: Mapping[Hashable, Variable]
) -> "Dataset":
    """Create a dataset as quickly as possible.

    Beware: the `variables` dict is modified INPLACE.
    """
    from .dataset import Dataset

    variables.update(coord_variables)
    coord_names = set(coord_variables)
    return Dataset._construct_direct(variables, coord_names)
```
### 44 - xarray/core/duck_array_ops.py:

Start line: 24, End line: 54

```python
def _dask_or_eager_func(
    name,
    eager_module=np,
    dask_module=dask_array,
    list_of_args=False,
    array_args=slice(1),
    requires_dask=None,
):
    """Create a function that dispatches to dask for dask array inputs."""
    if dask_module is not None:

        def f(*args, **kwargs):
            if list_of_args:
                dispatch_args = args[0]
            else:
                dispatch_args = args[array_args]
            if any(isinstance(a, dask_array_type) for a in dispatch_args):
                try:
                    wrapped = getattr(dask_module, name)
                except AttributeError as e:
                    raise AttributeError(f"{e}: requires dask >={requires_dask}")
            else:
                wrapped = getattr(eager_module, name)
            return wrapped(*args, **kwargs)

    else:

        def f(*args, **kwargs):
            return getattr(eager_module, name)(*args, **kwargs)

    return f
```
### 45 - xarray/core/dataarray.py:

Start line: 266, End line: 360

```python
class DataArray(AbstractArray, DataWithCoords):

    def __init__(
        self,
        data: Any = dtypes.NA,
        coords: Union[Sequence[Tuple], Mapping[Hashable, Any], None] = None,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        name: Hashable = None,
        attrs: Mapping = None,
        # internal parameters
        indexes: Dict[Hashable, pd.Index] = None,
        fastpath: bool = False,
    ):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xarray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            The following notations are accepted:

            - mapping {dimension name: array-like}
            - sequence of tuples that are valid arguments for xarray.Variable()
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

        dims : hashable or sequence of hashable, optional
            Name(s) of the data dimension(s). Must be either a hashable (only
            for 1D data) or a sequence of hashables with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            default to ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.
        """
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
        else:
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
                elif isinstance(data, pdcompat.Panel):
                    coords = [data.items, data.major_axis, data.minor_axis]

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
            indexes = dict(
                _extract_indexes_from_coords(coords)
            )  # needed for to_dataset

        # These fully describe a DataArray
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name

        # TODO(shoyer): document this argument, once it becomes part of the
        # public interface.
        self._indexes = indexes

        self._file_obj = None
```
### 46 - xarray/core/dataarray.py:

Start line: 1764, End line: 1817

```python
class DataArray(AbstractArray, DataWithCoords):

    def unstack(
        self,
        dim: Union[Hashable, Sequence[Hashable], None] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> "DataArray":
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : hashable or sequence of hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value: value to be filled. By default, np.nan
        sparse: use sparse-array if True

        Returns
        -------
        unstacked : DataArray
            Array with unstacked data.

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
          * x        (x) |S1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> stacked = arr.stack(z=("x", "y"))
        >>> stacked.indexes["z"]
        MultiIndex(levels=[['a', 'b'], [0, 1, 2]],
                   codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
                   names=['x', 'y'])
        >>> roundtripped = stacked.unstack()
        >>> arr.identical(roundtripped)
        True

        See Also
        --------
        DataArray.stack
        """
        ds = self._to_temp_dataset().unstack(dim, fill_value, sparse)
        return self._from_temp_dataset(ds)
```
### 52 - xarray/core/duck_array_ops.py:

Start line: 1, End line: 21

```python
"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
import contextlib
import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd

from . import dask_array_compat, dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import dask_array_type

try:
    import dask.array as dask_array
except ImportError:
    dask_array = None  # type: ignore
```
### 53 - xarray/core/computation.py:

Start line: 488, End line: 539

```python
def broadcast_compat_data(
    variable: Variable,
    broadcast_dims: Tuple[Hashable, ...],
    core_dims: Tuple[Hashable, ...],
) -> Any:
    data = variable.data

    old_dims = variable.dims
    new_dims = broadcast_dims + core_dims

    if new_dims == old_dims:
        # optimize for the typical case
        return data

    set_old_dims = set(old_dims)
    missing_core_dims = [d for d in core_dims if d not in set_old_dims]
    if missing_core_dims:
        raise ValueError(
            "operand to apply_ufunc has required core dimensions {}, but "
            "some of these dimensions are absent on an input variable: {}".format(
                list(core_dims), missing_core_dims
            )
        )

    set_new_dims = set(new_dims)
    unexpected_dims = [d for d in old_dims if d not in set_new_dims]
    if unexpected_dims:
        raise ValueError(
            "operand to apply_ufunc encountered unexpected "
            "dimensions %r on an input variable: these are core "
            "dimensions on other input or output variables" % unexpected_dims
        )

    # for consistency with numpy, keep broadcast dimensions to the left
    old_broadcast_dims = tuple(d for d in broadcast_dims if d in set_old_dims)
    reordered_dims = old_broadcast_dims + core_dims
    if reordered_dims != old_dims:
        order = tuple(old_dims.index(d) for d in reordered_dims)
        data = duck_array_ops.transpose(data, order)

    if new_dims != reordered_dims:
        key_parts = []
        for dim in new_dims:
            if dim in set_old_dims:
                key_parts.append(SLICE_NONE)
            elif key_parts:
                # no need to insert new axes at the beginning that are already
                # handled by broadcasting
                key_parts.append(np.newaxis)
        data = data[tuple(key_parts)]

    return data
```
### 57 - xarray/core/dataarray.py:

Start line: 216, End line: 264

```python
class DataArray(AbstractArray, DataWithCoords):
    """N-dimensional array with labeled coordinates and dimensions.

    DataArray provides a wrapper around numpy ndarrays that uses labeled
    dimensions and coordinates to support metadata aware operations. The API is
    similar to that for the pandas Series or DataFrame, but DataArray objects
    can have any number of dimensions, and their contents have fixed data
    types.

    Additional features over raw numpy arrays:

    - Apply operations over dimensions by name: ``x.sum('time')``.
    - Select or assign values by integer location (like numpy): ``x[:10]``
      or by label (like pandas): ``x.loc['2014-01-01']`` or
      ``x.sel(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across multiple
      dimensions (known in numpy as "broadcasting") based on dimension names,
      regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python dictionary:
      ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a DataArray
    always returns another DataArray.
    """

    _cache: Dict[str, Any]
    _coords: Dict[Any, Variable]
    _indexes: Optional[Dict[Hashable, pd.Index]]
    _name: Optional[Hashable]
    _variable: Variable

    __slots__ = (
        "_cache",
        "_coords",
        "_file_obj",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )

    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    _weighted_cls = weighted.DataArrayWeighted

    dt = property(CombinedDatetimelikeAccessor)
```
### 61 - xarray/core/duck_array_ops.py:

Start line: 468, End line: 504

```python
def _to_pytimedelta(array, unit="us"):
    index = pd.TimedeltaIndex(array.ravel(), unit=unit)
    return index.to_pytimedelta().reshape(array.shape)


def np_timedelta64_to_float(array, datetime_unit):
    """Convert numpy.timedelta64 to float.

    Notes
    -----
    The array is first converted to microseconds, which is less likely to
    cause overflow errors.
    """
    array = array.astype("timedelta64[ns]").astype(np.float64)
    conversion_factor = np.timedelta64(1, "ns") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array


def pd_timedelta_to_float(value, datetime_unit):
    """Convert pandas.Timedelta to float.

    Notes
    -----
    Built on the assumption that pandas timedelta values are in nanoseconds,
    which is also the numpy default resolution.
    """
    value = value.to_timedelta64()
    return np_timedelta64_to_float(value, datetime_unit)


def py_timedelta_to_float(array, datetime_unit):
    """Convert a timedelta object to a float, possibly at a loss of resolution.
    """
    array = np.asarray(array)
    array = np.reshape([a.total_seconds() for a in array.ravel()], array.shape) * 1e6
    conversion_factor = np.timedelta64(1, "us") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array
```
### 62 - xarray/core/dataarray.py:

Start line: 362, End line: 403

```python
class DataArray(AbstractArray, DataWithCoords):

    def _replace(
        self,
        variable: Variable = None,
        coords=None,
        name: Union[Hashable, None, Default] = _default,
        indexes=None,
    ) -> "DataArray":
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True, indexes=indexes)

    def _replace_maybe_drop_dims(
        self, variable: Variable, name: Union[Hashable, None, Default] = _default
    ) -> "DataArray":
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
            changed_dims = [
                k for k in variable.dims if variable.sizes[k] != self.sizes[k]
            ]
            indexes = propagate_indexes(self._indexes, exclude=changed_dims)
        else:
            allowed_dims = set(variable.dims)
            coords = {
                k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            }
            indexes = propagate_indexes(
                self._indexes, exclude=(set(self.dims) - allowed_dims)
            )
        return self._replace(variable, coords, name, indexes=indexes)
```
### 63 - xarray/core/computation.py:

Start line: 1072, End line: 1151

```python
def dot(*arrays, dims=None, **kwargs):
    """Generalized dot product for xarray objects. Like np.einsum, but
    provides a simpler interface based on array dimensions.

    Parameters
    ----------
    arrays: DataArray (or Variable) objects
        Arrays to compute.
    dims: '...', str or tuple of strings, optional
        Which dimensions to sum over. Ellipsis ('...') sums over all dimensions.
        If not specified, then all the common dimensions are summed over.
    **kwargs: dict
        Additional keyword arguments passed to numpy.einsum or
        dask.array.einsum

    Returns
    -------
    dot: DataArray

    Examples
    --------

    >>> import numpy as np
    >>> import xarray as xr
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
           [[ 4,  5],
            [ 6,  7]],
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
### 64 - xarray/core/dataarray.py:

Start line: 2151, End line: 2173

```python
class DataArray(AbstractArray, DataWithCoords):

    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)
```
### 65 - xarray/core/dataarray.py:

Start line: 2063, End line: 2149

```python
class DataArray(AbstractArray, DataWithCoords):

    def interpolate_na(
        self,
        dim: Hashable = None,
        method: str = "linear",
        limit: int = None,
        use_coordinate: Union[bool, str] = True,
        max_gap: Union[
            int, float, str, pd.Timedelta, np.timedelta64, datetime.timedelta
        ] = None,
        **kwargs: Any,
    ) -> "DataArray":
        """Fill in NaNs by interpolating according to different methods.

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to interpolate.
        method : str, optional
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation (Default). Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krog', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        use_coordinate : bool, str, default True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            eqaully-spaced along ``dim``. If True, the IndexVariable `dim` is
            used. If ``use_coordinate`` is a string, it specifies the name of a
            coordinate variariable to use as the index.
        limit : int, default None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit. This filling is done regardless of the size of
            the gap in the data. To only interpolate over gaps less than a given length,
            see ``max_gap``.
        max_gap: int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default None.
            Maximum size of gap, a continuous sequence of NaNs, that will be filled.
            Use None for no limit. When interpolating along a datetime64 dimension
            and ``use_coordinate=True``, ``max_gap`` can be one of the following:

            - a string that is valid input for pandas.to_timedelta
            - a :py:class:`numpy.timedelta64` object
            - a :py:class:`pandas.Timedelta` object
            - a :py:class:`datetime.timedelta` object

            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled
            dimensions has not been implemented yet. Gap length is defined as the difference
            between coordinate values at the first data point after a gap and the last value
            before a gap. For gaps at the beginning (end), gap length is defined as the difference
            between coordinate values at the first (last) valid data point and the first (last) NaN.
            For example, consider::

                <xarray.DataArray (x: 9)>
                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])
                Coordinates:
                  * x        (x) int64 0 1 2 3 4 5 6 7 8

            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively
        kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: DataArray
            Filled in DataArray.

        See also
        --------
        numpy.interp
        scipy.interpolate
        """
        from .missing import interp_na

        return interp_na(
            self,
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            **kwargs,
        )
```
### 69 - xarray/core/dataarray.py:

Start line: 760, End line: 795

```python
class DataArray(AbstractArray, DataWithCoords):

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
        return self._dask_finalize, (func, args, self.name)

    def __dask_postpersist__(self):
        func, args = self._to_temp_dataset().__dask_postpersist__()
        return self._dask_finalize, (func, args, self.name)

    @staticmethod
    def _dask_finalize(results, func, args, name):
        ds = func(results, *args)
        variable = ds._variables.pop(_THIS_ARRAY)
        coords = ds._variables
        return DataArray(variable, coords, name=name, fastpath=True)
```
### 70 - xarray/core/dataarray.py:

Start line: 2217, End line: 2260

```python
class DataArray(AbstractArray, DataWithCoords):

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "DataArray":
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : hashable or sequence of hashables, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dim' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            DataArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """

        var = self.variable.reduce(func, dim, axis, keep_attrs, keepdims, **kwargs)
        return self._replace_maybe_drop_dims(var)
```
### 71 - xarray/core/dataarray.py:

Start line: 2593, End line: 2620

```python
class DataArray(AbstractArray, DataWithCoords):

    @staticmethod
    def _binary_op(
        f: Callable[..., Any],
        reflexive: bool = False,
        join: str = None,  # see xarray.align
        **ignored_kwargs,
    ) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (Dataset, groupby.GroupBy)):
                return NotImplemented
            if isinstance(other, DataArray):
                align_type = OPTIONS["arithmetic_join"] if join is None else join
                self, other = align(self, other, join=align_type, copy=False)
            other_variable = getattr(other, "variable", other)
            other_coords = getattr(other, "coords", None)

            variable = (
                f(self.variable, other_variable)
                if not reflexive
                else f(other_variable, self.variable)
            )
            coords, indexes = self.coords._merge_raw(other_coords)
            name = self._result_name(other)

            return self._replace(variable, coords, name, indexes=indexes)

        return func
```
### 80 - xarray/core/dataarray.py:

Start line: 2840, End line: 2899

```python
class DataArray(AbstractArray, DataWithCoords):

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
        dims: '...', hashable or sequence of hashables, optional
            Which dimensions to sum over. Ellipsis ('...') sums over all dimensions.
            If not specified, then all the common dimensions are summed over.

        Returns
        -------
        result : DataArray
            Array resulting from the dot product over all shared dimensions.

        See also
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
        ('z')

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
### 81 - xarray/core/nputils.py:

Start line: 115, End line: 132

```python
class NumpyVIndexAdapter:
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        """Value must have dimensionality matching the key."""
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)
```
### 82 - xarray/core/nputils.py:

Start line: 1, End line: 33

```python
import warnings

import numpy as np
import pandas as pd
from numpy.core.multiarray import normalize_axis_index

try:
    import bottleneck as bn

    _USE_BOTTLENECK = True
except ImportError:
    # use numpy methods instead
    bn = np
    _USE_BOTTLENECK = False


def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def nanfirst(values, axis):
    axis = normalize_axis_index(axis, values.ndim)
    idx_first = np.argmax(~pd.isnull(values), axis=axis)
    return _select_along_axis(values, idx_first, axis)


def nanlast(values, axis):
    axis = normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    return _select_along_axis(values, idx_last, axis)
```
### 83 - xarray/core/dataset.py:

Start line: 2302, End line: 2498

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict. optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar, optional
            Value to use for newly missing values
        sparse: use sparse-array. By default, False
        **indexers_kwargs : {dim: indexer, ...}, optional
            Keyword arguments in the same form as ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer

        Examples
        --------

        Create a dataset with some fictional data.

        >>> import xarray as xr
        >>> import pandas as pd
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature": ("station", 20 * np.random.rand(4)),
        ...         "pressure": ("station", 500 * np.random.rand(4)),
        ...     },
        ...     coords={"station": ["boston", "nyc", "seattle", "denver"]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) <U7 'boston' 'nyc' 'seattle' 'denver'
        Data variables:
            temperature  (station) float64 18.84 14.59 19.22 17.16
            pressure     (station) float64 324.1 194.3 122.8 244.3
        >>> x.indexes
        station: Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ["boston", "austin", "seattle", "lincoln"]
        >>> x.reindex({"station": new_index})
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) object 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 18.84 nan 19.22 nan
            pressure     (station) float64 324.1 nan 122.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({"station": new_index}, fill_value=0)
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) object 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 18.84 0.0 19.22 0.0
            pressure     (station) float64 324.1 0.0 122.8 0.0

        Because the index is not monotonically increasing or decreasing, we cannot use arguments
        to the keyword method to fill the `NaN` values.

        >>> x.reindex({"station": new_index}, method="nearest")
        Traceback (most recent call last):
        ...
            raise ValueError('index must be monotonic increasing or decreasing')
        ValueError: index must be monotonic increasing or decreasing

        To further illustrate the filling functionality in reindex, we will create a
        dataset with a monotonically increasing index (for example, a sequence of dates).

        >>> x2 = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             "time",
        ...             [15.57, 12.77, np.nan, 0.3081, 16.59, 15.12],
        ...         ),
        ...         "pressure": ("time", 500 * np.random.rand(6)),
        ...     },
        ...     coords={"time": pd.date_range("01/01/2019", periods=6, freq="D")},
        ... )
        >>> x2
        <xarray.Dataset>
        Dimensions:      (time: 6)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-01 2019-01-02 ... 2019-01-06
        Data variables:
            temperature  (time) float64 15.57 12.77 nan 0.3081 16.59 15.12
            pressure     (time) float64 103.4 122.7 452.0 444.0 399.2 486.0

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range("12/29/2018", periods=10, freq="D")
        >>> x2.reindex({"time": time_index2})
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
        * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 nan nan nan 103.4 ... 444.0 399.2 486.0 nan

        The index entries that did not have a value in the original data frame (for example, `2018-12-29`)
        are by default filled with NaN. If desired, we can fill in the missing values using one of several options.

        For example, to back-propagate the last valid value to fill the `NaN` values,
        pass `bfill` as an argument to the `method` keyword.

        >>> x3 = x2.reindex({"time": time_index2}, method="bfill")
        >>> x3
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
        * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 15.57 15.57 15.57 15.57 ... 16.59 15.12 nan
            pressure     (time) float64 103.4 103.4 103.4 103.4 ... 399.2 486.0 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 1)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-03
        Data variables:
            temperature  (time) float64 nan
            pressure     (time) float64 452.0
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 2)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan
            pressure     (time) float64 452.0 nan

        This is because filling while reindexing does not look at dataset values, but only compares
        the original and desired indexes. If you do want to fill in the `NaN` values present in the
        original dataset, use the :py:meth:`~Dataset.fillna()` method.

        """
        return self._reindex(
            indexers,
            method,
            tolerance,
            copy,
            fill_value,
            sparse=False,
            **indexers_kwargs,
        )
```
### 85 - xarray/core/duck_array_ops.py:

Start line: 57, End line: 79

```python
def fail_on_dask_array_input(values, msg=None, func_name=None):
    if isinstance(values, dask_array_type):
        if msg is None:
            msg = "%r is not yet a valid method on dask arrays"
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)


# switch to use dask.array / __array_function__ version when dask supports it:
# https://github.com/dask/dask/pull/4822
moveaxis = npcompat.moveaxis

around = _dask_or_eager_func("around")
isclose = _dask_or_eager_func("isclose")


isnat = np.isnat
isnan = _dask_or_eager_func("isnan")
zeros_like = _dask_or_eager_func("zeros_like")


pandas_isnull = _dask_or_eager_func("isnull", eager_module=pd)
```
### 86 - xarray/__init__.py:

Start line: 1, End line: 89

```python
import pkg_resources

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
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast
from .core.combine import auto_combine, combine_by_coords, combine_nested
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import apply_ufunc, dot, where
from .core.concat import concat
from .core.dataarray import DataArray
from .core.dataset import Dataset
from .core.extensions import register_dataarray_accessor, register_dataset_accessor
from .core.merge import MergeError, merge
from .core.options import set_options
from .core.parallel import map_blocks
from .core.variable import Coordinate, IndexVariable, Variable, as_variable
from .util.print_versions import show_versions

try:
    __version__ = pkg_resources.get_distribution("xarray").version
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
    "auto_combine",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "decode_cf",
    "dot",
    "full_like",
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
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "save_mfdataset",
    "set_options",
    "show_versions",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
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
### 87 - xarray/core/dataarray.py:

Start line: 2741, End line: 2788

```python
class DataArray(AbstractArray, DataWithCoords):

    def shift(
        self,
        shifts: Mapping[Hashable, int] = None,
        fill_value: Any = dtypes.NA,
        **shifts_kwargs: int,
    ) -> "DataArray":
        """Shift this array by an offset along one or more dimensions.

        Only the data is moved; coordinates stay in place. Values shifted from
        beyond array bounds are replaced by NaN. This is consistent with the
        behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : DataArray
            DataArray with the same coordinates and attributes but shifted
            data.

        See also
        --------
        roll

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims="x")
        >>> arr.shift(x=1)
        <xarray.DataArray (x: 3)>
        array([ nan,   5.,   6.])
        Coordinates:
          * x        (x) int64 0 1 2
        """
        variable = self.variable.shift(
            shifts=shifts, fill_value=fill_value, **shifts_kwargs
        )
        return self._replace(variable=variable)
```
### 88 - xarray/core/duck_array_ops.py:

Start line: 324, End line: 348

```python
# Attributes `numeric_only`, `available_min_count` is used for docs.
# See ops.inject_reduce_methods
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True)
min = _create_nan_agg_method("min", coerce_strings=True)
sum = _create_nan_agg_method("sum")
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", dask_module=dask_array_compat)
median.numeric_only = True
prod = _create_nan_agg_method("prod")
prod.numeric_only = True
sum.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod")
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum")
cumsum_1d.numeric_only = True


_mean = _create_nan_agg_method("mean")
```
### 91 - xarray/core/dataset.py:

Start line: 4789, End line: 4896

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    @staticmethod
    def _unary_op(f, keep_attrs=False):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            variables = {}
            for k, v in self._variables.items():
                if k in self._coord_names:
                    variables[k] = v
                else:
                    variables[k] = f(v, *args, **kwargs)
            attrs = self._attrs if keep_attrs else None
            return self._replace_with_new_dims(variables, attrs=attrs)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join=None):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                return NotImplemented
            align_type = OPTIONS["arithmetic_join"] if join is None else join
            if isinstance(other, (DataArray, Dataset)):
                self, other = align(self, other, join=align_type, copy=False)
            g = f if not reflexive else lambda x, y: f(y, x)
            ds = self._calculate_binary_op(g, other, join=align_type)
            return ds

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a Dataset and "
                    "a grouped object are not permitted"
                )
            # we don't actually modify arrays in-place with in-place Dataset
            # arithmetic -- this lets us automatically align things
            if isinstance(other, (DataArray, Dataset)):
                other = other.reindex_like(self, copy=False)
            g = ops.inplace_to_noninplace_op(f)
            ds = self._calculate_binary_op(g, other, inplace=True)
            self._replace_with_new_dims(
                ds._variables,
                ds._coord_names,
                attrs=ds._attrs,
                indexes=ds._indexes,
                inplace=True,
            )
            return self

        return func

    def _calculate_binary_op(self, f, other, join="inner", inplace=False):
        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if inplace and set(lhs_data_vars) != set(rhs_data_vars):
                raise ValueError(
                    "datasets must have the same data variables "
                    "for in-place arithmetic operations: %s, %s"
                    % (list(lhs_data_vars), list(rhs_data_vars))
                )

            dest_vars = {}

            for k in lhs_data_vars:
                if k in rhs_data_vars:
                    dest_vars[k] = f(lhs_vars[k], rhs_vars[k])
                elif join in ["left", "outer"]:
                    dest_vars[k] = f(lhs_vars[k], np.nan)
            for k in rhs_data_vars:
                if k not in dest_vars and join in ["right", "outer"]:
                    dest_vars[k] = f(rhs_vars[k], np.nan)
            return dest_vars

        if utils.is_dict_like(other) and not isinstance(other, Dataset):
            # can't use our shortcut of doing the binary operation with
            # Variable objects, so apply over our data vars instead.
            new_data_vars = apply_over_both(
                self.data_vars, other, self.data_vars, other
            )
            return Dataset(new_data_vars)

        other_coords = getattr(other, "coords", None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(
                self.data_vars, other.data_vars, self.variables, other.variables
            )
        else:
            other_variable = getattr(other, "variable", other)
            new_vars = {k: f(self.variables[k], other_variable) for k in self.data_vars}
        ds._variables.update(new_vars)
        ds._dims = calculate_dimensions(ds._variables)
        return ds

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs
        for v in other.variables:
            if v in self.variables:
                self.variables[v].attrs = other.variables[v].attrs
```
### 92 - xarray/core/dataarray.py:

Start line: 2790, End line: 2838

```python
class DataArray(AbstractArray, DataWithCoords):

    def roll(
        self,
        shifts: Mapping[Hashable, int] = None,
        roll_coords: bool = None,
        **shifts_kwargs: int,
    ) -> "DataArray":
        """Roll this array by an offset along one or more dimensions.

        Unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to rotate each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        roll_coords : bool
            Indicates whether to  roll the coordinates by the offset
            The current default of roll_coords (None, equivalent to True) is
            deprecated and will change to False in a future version.
            Explicitly pass roll_coords to silence the warning.
        **shifts_kwargs : The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        rolled : DataArray
            DataArray with the same attributes but rolled data and coordinates.

        See also
        --------
        shift

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims="x")
        >>> arr.roll(x=1)
        <xarray.DataArray (x: 3)>
        array([7, 5, 6])
        Coordinates:
          * x        (x) int64 2 0 1
        """
        ds = self._to_temp_dataset().roll(
            shifts=shifts, roll_coords=roll_coords, **shifts_kwargs
        )
        return self._from_temp_dataset(ds)
```
### 96 - xarray/core/dataarray.py:

Start line: 3094, End line: 3151

```python
class DataArray(AbstractArray, DataWithCoords):

    def differentiate(
        self, coord: Hashable, edge_order: int = 1, datetime_unit: str = None
    ) -> "DataArray":
        """ Differentiate the array with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: hashable
            The coordinate to be used to compute the gradient.
        edge_order: 1 or 2. Default 1
            N-th order accurate differences at the boundaries.
        datetime_unit: None or any of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
            'us', 'ns', 'ps', 'fs', 'as'}
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: DataArray

        See also
        --------
        numpy.gradient: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.differentiate("x")
        <xarray.DataArray (x: 4, y: 3)>
        array([[30.      , 30.      , 30.      ],
               [27.545455, 27.545455, 27.545455],
               [27.545455, 27.545455, 27.545455],
               [30.      , 30.      , 30.      ]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().differentiate(coord, edge_order, datetime_unit)
        return self._from_temp_dataset(ds)
```
### 97 - xarray/core/dataarray.py:

Start line: 2175, End line: 2197

```python
class DataArray(AbstractArray, DataWithCoords):

    def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import bfill

        return bfill(self, dim, limit=limit)
```
### 101 - xarray/core/dask_array_ops.py:

Start line: 1, End line: 27

```python
import numpy as np

from . import dtypes, nputils


def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays
    """
    import dask.array as da

    dtype, fill_value = dtypes.maybe_promote(a.dtype)
    a = a.astype(dtype)
    # inputs for overlap
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = (window + 1) // 2
    boundary = {d: fill_value for d in range(a.ndim)}
    # Create overlap array.
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)
    # apply rolling func
    out = ag.map_blocks(
        moving_func, window, min_count=min_count, axis=axis, dtype=a.dtype
    )
    # trim array
    result = da.overlap.trim_internal(out, depth)
    return result
```
### 102 - xarray/core/dataarray.py:

Start line: 1193, End line: 1251

```python
class DataArray(AbstractArray, DataWithCoords):

    def reindex_like(
        self,
        other: Union["DataArray", Dataset],
        method: str = None,
        tolerance=None,
        copy: bool = True,
        fill_value=dtypes.NA,
    ) -> "DataArray":
        """Conform this object onto the indexes of another object, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to pandas.Index objects, which provides coordinates upon
            which to index the variables in this dataset. The indexes on this
            other object need not be the same as the indexes on this
            dataset. Any mis-matched index values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values from other not found on this
            data array:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar, optional
            Value to use for newly missing values

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but coordinates from
            the other object.

        See Also
        --------
        DataArray.reindex
        align
        """
        indexers = reindex_like_indexers(self, other)
        return self.reindex(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )
```
### 111 - xarray/core/dataarray.py:

Start line: 3278, End line: 3444

```python
class DataArray(AbstractArray, DataWithCoords):

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> "DataArray":
        # ... other code
```
### 112 - xarray/core/duck_array_ops.py:

Start line: 240, End line: 281

```python
def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)


def where(condition, x, y):
    """Three argument where() with better dtype promotion rules."""
    return _where(condition, *as_shared_dtype([x, y]))


def where_method(data, cond, other=dtypes.NA):
    if other is dtypes.NA:
        other = dtypes.get_fill_value(data.dtype)
    return where(cond, data, other)


def fillna(data, other):
    # we need to pass data first so pint has a chance of returning the
    # correct unit
    # TODO: revert after https://github.com/hgrecco/pint/issues/1019 is fixed
    return where(notnull(data), data, other)


def concatenate(arrays, axis=0):
    """concatenate() with better dtype promotion rules."""
    return _concatenate(as_shared_dtype(arrays), axis=axis)


def stack(arrays, axis=0):
    """stack() with better dtype promotion rules."""
    return _stack(as_shared_dtype(arrays), axis=axis)


@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        yield
```
### 115 - xarray/core/dataarray.py:

Start line: 1078, End line: 1093

```python
class DataArray(AbstractArray, DataWithCoords):

    def head(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by the the first `n`
        values along the specified dimension(s). Default `n` = 5

        See Also
        --------
        Dataset.head
        DataArray.tail
        DataArray.thin
        """
        ds = self._to_temp_dataset().head(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)
```
### 117 - xarray/core/dataarray.py:

Start line: 1003, End line: 1038

```python
class DataArray(AbstractArray, DataWithCoords):

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by integer indexing
        along the specified dimension(s).

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(indexers, drop=drop)
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers)

        coords = {}
        for coord_name, coord_value in self._coords.items():
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)
                if drop and coord_value.ndim == 0:
                    continue
            coords[coord_name] = coord_value

        return self._replace(variable=variable, coords=coords)
```
