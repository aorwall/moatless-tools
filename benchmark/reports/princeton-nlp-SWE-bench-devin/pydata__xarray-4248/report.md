# pydata__xarray-4248

| **pydata/xarray** | `98dc1f4ea18738492e074e9e51ddfed5cd30ab94` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -261,6 +261,8 @@ def inline_variable_array_repr(var, max_width):
         return inline_dask_repr(var.data)
     elif isinstance(var._data, sparse_array_type):
         return inline_sparse_repr(var.data)
+    elif hasattr(var._data, "_repr_inline_"):
+        return var._data._repr_inline_(max_width)
     elif hasattr(var._data, "__array_function__"):
         return maybe_truncate(repr(var._data).replace("\n", " "), max_width)
     else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/formatting.py | 264 | 264 | - | 1 | -


## Problem Statement

```
Feature request: show units in dataset overview
Here's a hypothetical dataset:

\`\`\`
<xarray.Dataset>
Dimensions:  (time: 3, x: 988, y: 822)
Coordinates:
  * x         (x) float64 ...
  * y         (y) float64 ...
  * time      (time) datetime64[ns] ...
Data variables:
    rainfall  (time, y, x) float32 ...
    max_temp  (time, y, x) float32 ...
\`\`\`

It would be really nice if the units of the coordinates and of the data variables were shown in the `Dataset` repr, for example as:

\`\`\`
<xarray.Dataset>
Dimensions:  (time: 3, x: 988, y: 822)
Coordinates:
  * x, in metres         (x)            float64 ...
  * y, in metres         (y)            float64 ...
  * time                 (time)         datetime64[ns] ...
Data variables:
    rainfall, in mm      (time, y, x)   float32 ...
    max_temp, in deg C   (time, y, x)   float32 ...
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/formatting.py** | 495 | 524| 206 | 206 | 5259 | 
| 2 | 2 xarray/plot/dataset_plot.py | 404 | 454| 390 | 596 | 8680 | 
| 3 | 3 xarray/core/dataset.py | 555 | 1390| 6285 | 6881 | 62448 | 
| 4 | 3 xarray/core/dataset.py | 375 | 425| 398 | 7279 | 62448 | 
| 5 | 3 xarray/plot/dataset_plot.py | 1 | 76| 518 | 7797 | 62448 | 
| 6 | 3 xarray/plot/dataset_plot.py | 167 | 243| 931 | 8728 | 62448 | 
| 7 | 4 xarray/conventions.py | 718 | 737| 110 | 8838 | 68302 | 
| 8 | 4 xarray/plot/dataset_plot.py | 110 | 164| 350 | 9188 | 68302 | 
| 9 | 5 xarray/core/coordinates.py | 221 | 244| 198 | 9386 | 71274 | 
| 10 | 5 xarray/core/dataset.py | 1 | 136| 647 | 10033 | 71274 | 
| 11 | 5 xarray/plot/dataset_plot.py | 79 | 107| 206 | 10239 | 71274 | 
| 12 | 5 xarray/plot/dataset_plot.py | 245 | 351| 765 | 11004 | 71274 | 
| 13 | 5 xarray/core/coordinates.py | 184 | 219| 252 | 11256 | 71274 | 
| 14 | 6 xarray/core/dataarray.py | 1 | 81| 433 | 11689 | 105226 | 
| 15 | 6 xarray/core/dataset.py | 2347 | 3028| 5902 | 17591 | 105226 | 
| 16 | 6 xarray/core/dataset.py | 5186 | 5853| 6177 | 23768 | 105226 | 
| 17 | 6 xarray/core/coordinates.py | 246 | 258| 116 | 23884 | 105226 | 
| 18 | 6 xarray/core/dataset.py | 187 | 211| 229 | 24113 | 105226 | 
| 19 | 7 xarray/core/combine.py | 516 | 712| 2442 | 26555 | 111939 | 
| 20 | **7 xarray/core/formatting.py** | 658 | 679| 158 | 26713 | 111939 | 
| 21 | 8 doc/gallery/plot_cartopy_facetgrid.py | 1 | 46| 327 | 27040 | 112266 | 
| 22 | 9 xarray/testing.py | 288 | 322| 413 | 27453 | 115252 | 
| 23 | 10 xarray/core/common.py | 94 | 110| 139 | 27592 | 128655 | 
| 24 | 10 xarray/core/coordinates.py | 304 | 324| 197 | 27789 | 128655 | 
| 25 | **10 xarray/core/formatting.py** | 271 | 291| 209 | 27998 | 128655 | 
| 26 | 10 xarray/core/dataset.py | 428 | 553| 1010 | 29008 | 128655 | 
| 27 | 10 xarray/core/dataset.py | 3030 | 3779| 6067 | 35075 | 128655 | 
| 28 | 11 xarray/core/concat.py | 398 | 426| 278 | 35353 | 132425 | 
| 29 | 11 xarray/core/dataset.py | 1392 | 1569| 1589 | 36942 | 132425 | 
| 30 | 11 xarray/conventions.py | 648 | 715| 614 | 37556 | 132425 | 
| 31 | 12 xarray/plot/facetgrid.py | 76 | 213| 1058 | 38614 | 137241 | 
| 32 | 12 xarray/core/dataset.py | 1571 | 2276| 5972 | 44586 | 137241 | 
| 33 | 13 xarray/core/formatting_html.py | 254 | 290| 260 | 44846 | 139438 | 
| 34 | 14 xarray/core/weighted.py | 183 | 224| 347 | 45193 | 141437 | 
| 35 | 15 xarray/core/merge.py | 635 | 840| 2707 | 47900 | 149260 | 
| 36 | 16 xarray/coding/times.py | 1 | 64| 451 | 48351 | 153114 | 
| 37 | 17 xarray/plot/plot.py | 617 | 776| 1741 | 50092 | 161324 | 
| 38 | 17 xarray/plot/dataset_plot.py | 353 | 401| 288 | 50380 | 161324 | 
| 39 | 17 xarray/core/coordinates.py | 33 | 77| 289 | 50669 | 161324 | 
| 40 | 17 xarray/core/dataset.py | 4401 | 5184| 6203 | 56872 | 161324 | 
| 41 | 18 xarray/plot/utils.py | 1 | 54| 283 | 57155 | 167852 | 
| 42 | 18 xarray/core/concat.py | 287 | 306| 159 | 57314 | 167852 | 
| 43 | 18 xarray/core/dataset.py | 3781 | 4399| 5357 | 62671 | 167852 | 
| 44 | 19 xarray/__init__.py | 1 | 93| 603 | 63274 | 168455 | 
| 45 | **19 xarray/core/formatting.py** | 597 | 621| 130 | 63404 | 168455 | 
| 46 | **19 xarray/core/formatting.py** | 312 | 324| 112 | 63516 | 168455 | 
| 47 | 20 xarray/core/variable.py | 407 | 502| 751 | 64267 | 189167 | 
| 48 | 20 xarray/core/formatting_html.py | 49 | 97| 369 | 64636 | 189167 | 
| 49 | 20 xarray/core/concat.py | 309 | 384| 609 | 65245 | 189167 | 


### Hint

```
I would love to see this.

What would we want the exact formatting to be? Square brackets to copy how units from `attrs['units']` are displayed on plots? e.g.
\`\`\`
<xarray.Dataset>
Dimensions:  (time: 3, x: 988, y: 822)
Coordinates:
  * x [m]             (x)            float64 ...
  * y [m]             (y)            float64 ...
  * time [s]          (time)         datetime64[ns] ...
Data variables:
    rainfall [mm]     (time, y, x)   float32 ...
    max_temp [deg C]  (time, y, x)   float32 ...
\`\`\`
The lack of vertical alignment is kind of ugly...

There are now two cases to discuss: units in `attrs`, and unit-aware arrays like pint. (If we do the latter we may not need the former though...)

from @keewis on #3616:

>At the moment, the formatting.diff_*_repr functions that provide the pretty-printing for assert_* use repr to format NEP-18 strings, truncating the result if it is too long. In the case of pint's quantities, this makes the pretty printing useless since only a few values are visible and the unit is in the truncated part.
>
> What should we about this? Does pint have to change its repr?

We could presumably just extract the units from pint's repr to display them separately. I don't know if that raises questions about generality of duck-typing arrays though @dcherian ? Is it fine to make units a special-case?
it was argued in pint that the unit is part of the data, so we should keep it as close to the data as possible. How about
\`\`\`
<xarray.Dataset>
Dimensions:  (time: 3, x: 988, y: 822)
Coordinates:
  * x             (x)          [m]     float64 ...
  * y             (y)          [m]     float64 ...
  * time          (time)       [s]     datetime64[ns] ...
Data variables:
    rainfall      (time, y, x) [mm]    float32 ...
    max_temp      (time, y, x) [deg C] float32 ...
\`\`\`
or
\`\`\`
<xarray.Dataset>
Dimensions:  (time: 3, x: 988, y: 822)
Coordinates:
  * x             (x)             float64 [m] ...
  * y             (y)             float64 [m] ...
  * time          (time)          datetime64[ns] [s] ...
Data variables:
    rainfall      (time, y, x)    float32 [mm] ...
    max_temp      (time, y, x)    float32 [deg C] ...
\`\`\`
The issue with the second example is that it is easy to confuse with numpy's dtype, though. Maybe we should use parentheses instead?

re special casing: I think would be fine for attributes since we already special case them for plotting, but I don't know about duck arrays. Even if we want to special case them, there are many unit libraries with different interfaces so we would either need to special case all of them or require a specific interface (or a function to retrieve the necessary data?).

Also, we should keep in mind is that using more horizontal space for the units results in less space for data. And we should not forget about https://github.com/dask/dask/issues/5329#issue-485927396, where a different kind of format was proposed, at least for the values of a `DataArray`.
Instead of trying to come up with our own formatting, how about supporting a `_repr_short_(self, length)` method on the duck array (with a fall back to the current behavior)? That way duck arrays have to explicitly define the format (or have a compatibility package like `pint-xarray` provide it for them) if they want something different from their normal repr and we don't have to add duck array specific code.

This won't help with displaying the `units` attributes (which we don't really need once we have support for pint arrays in indexes).
```

## Patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -261,6 +261,8 @@ def inline_variable_array_repr(var, max_width):
         return inline_dask_repr(var.data)
     elif isinstance(var._data, sparse_array_type):
         return inline_sparse_repr(var.data)
+    elif hasattr(var._data, "_repr_inline_"):
+        return var._data._repr_inline_(max_width)
     elif hasattr(var._data, "__array_function__"):
         return maybe_truncate(repr(var._data).replace("\n", " "), max_width)
     else:

```

## Test Patch

```diff
diff --git a/xarray/tests/test_formatting.py b/xarray/tests/test_formatting.py
--- a/xarray/tests/test_formatting.py
+++ b/xarray/tests/test_formatting.py
@@ -7,6 +7,7 @@
 
 import xarray as xr
 from xarray.core import formatting
+from xarray.core.npcompat import IS_NEP18_ACTIVE
 
 from . import raises_regex
 
@@ -391,6 +392,44 @@ def test_array_repr(self):
         assert actual == expected
 
 
+@pytest.mark.skipif(not IS_NEP18_ACTIVE, reason="requires __array_function__")
+def test_inline_variable_array_repr_custom_repr():
+    class CustomArray:
+        def __init__(self, value, attr):
+            self.value = value
+            self.attr = attr
+
+        def _repr_inline_(self, width):
+            formatted = f"({self.attr}) {self.value}"
+            if len(formatted) > width:
+                formatted = f"({self.attr}) ..."
+
+            return formatted
+
+        def __array_function__(self, *args, **kwargs):
+            return NotImplemented
+
+        @property
+        def shape(self):
+            return self.value.shape
+
+        @property
+        def dtype(self):
+            return self.value.dtype
+
+        @property
+        def ndim(self):
+            return self.value.ndim
+
+    value = CustomArray(np.array([20, 40]), "m")
+    variable = xr.Variable("x", value)
+
+    max_width = 10
+    actual = formatting.inline_variable_array_repr(variable, max_width=10)
+
+    assert actual == value._repr_inline_(max_width)
+
+
 def test_set_numpy_options():
     original_options = np.get_printoptions()
     with formatting.set_numpy_options(threshold=10):

```


## Code snippets

### 1 - xarray/core/formatting.py:

Start line: 495, End line: 524

```python
def dataset_repr(ds):
    summary = ["<xarray.{}>".format(type(ds).__name__)]

    col_width = _calculate_col_width(_get_col_items(ds.variables))

    dims_start = pretty_print("Dimensions:", col_width)
    summary.append("{}({})".format(dims_start, dim_summary(ds)))

    if ds.coords:
        summary.append(coords_repr(ds.coords, col_width=col_width))

    unindexed_dims_str = unindexed_dims_repr(ds.dims, ds.coords)
    if unindexed_dims_str:
        summary.append(unindexed_dims_str)

    summary.append(data_vars_repr(ds.data_vars, col_width=col_width))

    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return "\n".join(summary)


def diff_dim_summary(a, b):
    if a.dims != b.dims:
        return "Differing dimensions:\n    ({}) != ({})".format(
            dim_summary(a), dim_summary(b)
        )
    else:
        return ""
```
### 2 - xarray/plot/dataset_plot.py:

Start line: 404, End line: 454

```python
@_dsplot
def scatter(ds, x, y, ax, **kwargs):
    """
    Scatter Dataset data variables against each other.
    """

    if "add_colorbar" in kwargs or "add_legend" in kwargs:
        raise ValueError(
            "Dataset.plot.scatter does not accept "
            "'add_colorbar' or 'add_legend'. "
            "Use 'add_guide' instead."
        )

    cmap_params = kwargs.pop("cmap_params")
    hue = kwargs.pop("hue")
    hue_style = kwargs.pop("hue_style")
    markersize = kwargs.pop("markersize", None)
    size_norm = kwargs.pop("size_norm", None)
    size_mapping = kwargs.pop("size_mapping", None)  # set by facetgrid

    # need to infer size_mapping with full dataset
    data = _infer_scatter_data(ds, x, y, hue, markersize, size_norm, size_mapping)

    if hue_style == "discrete":
        primitive = []
        for label in np.unique(data["hue"].values):
            mask = data["hue"] == label
            if data["sizes"] is not None:
                kwargs.update(s=data["sizes"].where(mask, drop=True).values.flatten())

            primitive.append(
                ax.scatter(
                    data["x"].where(mask, drop=True).values.flatten(),
                    data["y"].where(mask, drop=True).values.flatten(),
                    label=label,
                    **kwargs,
                )
            )

    elif hue is None or hue_style == "continuous":
        if data["sizes"] is not None:
            kwargs.update(s=data["sizes"].values.ravel())
        if data["hue"] is not None:
            kwargs.update(c=data["hue"].values.ravel())

        primitive = ax.scatter(
            data["x"].values.ravel(), data["y"].values.ravel(), **cmap_params, **kwargs
        )

    return primitive
```
### 3 - xarray/core/dataset.py:

Start line: 555, End line: 1390

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    @classmethod
    def load_store(cls, store, decoder=None) -> "Dataset":
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables, attributes = store.load()
        if decoder:
            variables, attributes = decoder(variables, attributes)
        obj = cls(variables, attrs=attributes)
        obj._file_obj = store
        return obj

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)

    @property
    def attrs(self) -> Dict[Hashable, Any]:
        """Dictionary of global attributes on this dataset
        """
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self) -> Dict:
        """Dictionary of global encoding attributes on this dataset
        """
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See also
        --------
        DataArray.sizes
        """
        return self.dims

    def load(self, **kwargs) -> "Dataset":
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return this dataset.
        Unlike compute, the original dataset is modified and returned.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data
            for k, v in self.variables.items()
            if isinstance(v._data, dask_array_type)
        }
        if lazy_data:
            import dask.array as da

            # evaluate all the dask arrays simultaneously
            evaluated_data = da.compute(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        # load everything else sequentially
        for k, v in self.variables.items():
            if k not in lazy_data:
                v.load()

        return self

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token(
            (type(self), self._variables, self._coord_names, self._attrs)
        )

    def __dask_graph__(self):
        graphs = {k: v.__dask_graph__() for k, v in self.variables.items()}
        graphs = {k: v for k, v in graphs.items() if v is not None}
        if not graphs:
            return None
        else:
            try:
                from dask.highlevelgraph import HighLevelGraph

                return HighLevelGraph.merge(*graphs.values())
            except ImportError:
                from dask import sharedict

                return sharedict.merge(*graphs.values())

    def __dask_keys__(self):
        import dask

        return [
            v.__dask_keys__()
            for v in self.variables.values()
            if dask.is_dask_collection(v)
        ]

    def __dask_layers__(self):
        import dask

        return sum(
            [
                v.__dask_layers__()
                for v in self.variables.values()
                if dask.is_dask_collection(v)
            ],
            (),
        )

    @property
    def __dask_optimize__(self):
        import dask.array as da

        return da.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        import dask.array as da

        return da.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        import dask

        info = [
            (True, k, v.__dask_postcompute__())
            if dask.is_dask_collection(v)
            else (False, k, v)
            for k, v in self._variables.items()
        ]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postcompute, args

    def __dask_postpersist__(self):
        import dask

        info = [
            (True, k, v.__dask_postpersist__())
            if dask.is_dask_collection(v)
            else (False, k, v)
            for k, v in self._variables.items()
        ]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postpersist, args

    @staticmethod
    def _dask_postcompute(results, info, *args):
        variables = {}
        results2 = list(results[::-1])
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                r = results2.pop()
                result = func(r, *args2)
            else:
                result = v
            variables[k] = result

        final = Dataset._construct_direct(variables, *args)
        return final

    @staticmethod
    def _dask_postpersist(dsk, info, *args):
        variables = {}
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                result = func(dsk, *args2)
            else:
                result = v
            variables[k] = result

        return Dataset._construct_direct(variables, *args)

    def compute(self, **kwargs) -> "Dataset":
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return a new dataset.
        Unlike load, the original dataset is left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def _persist_inplace(self, **kwargs) -> "Dataset":
        """Persist all Dask arrays in memory
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data
            for k, v in self.variables.items()
            if isinstance(v._data, dask_array_type)
        }
        if lazy_data:
            import dask

            # evaluate all the dask arrays simultaneously
            evaluated_data = dask.persist(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        return self

    def persist(self, **kwargs) -> "Dataset":
        """ Trigger computation, keeping data as dask arrays

        This operation can be used to trigger computation on underlying dask
        arrays, similar to ``.compute()`` or ``.load()``.  However this
        operation keeps the data as dask arrays. This is particularly useful
        when using the dask.distributed scheduler and you want to load a large
        amount of data into distributed memory.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        new = self.copy(deep=False)
        return new._persist_inplace(**kwargs)

    @classmethod
    def _construct_direct(
        cls,
        variables,
        coord_names,
        dims=None,
        attrs=None,
        indexes=None,
        encoding=None,
        file_obj=None,
    ):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._file_obj = file_obj
        obj._encoding = encoding
        return obj

    def _replace(
        self,
        variables: Dict[Hashable, Variable] = None,
        coord_names: Set[Hashable] = None,
        dims: Dict[Any, int] = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Any, pd.Index], None, Default] = _default,
        encoding: Union[dict, None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Fastpath constructor for internal use.

        Returns an object with optionally with replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        dataset. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not _default:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is _default:
                indexes = copy.copy(self._indexes)
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj

    def _replace_with_new_dims(
        self,
        variables: Dict[Hashable, Variable],
        coord_names: set = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Hashable, pd.Index], None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )

    def _replace_vars_and_dims(
        self,
        variables: Dict[Hashable, Variable],
        coord_names: set = None,
        dims: Dict[Hashable, int] = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Deprecated version of _replace_with_new_dims().

        Unlike _replace_with_new_dims(), this method always recalculates
        indexes from variables.
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes=None, inplace=inplace
        )

    def _overwrite_indexes(self, indexes: Mapping[Any, pd.Index]) -> "Dataset":
        if not indexes:
            return self

        variables = self._variables.copy()
        new_indexes = dict(self.indexes)
        for name, idx in indexes.items():
            variables[name] = IndexVariable(name, idx)
            new_indexes[name] = idx
        obj = self._replace(variables, indexes=new_indexes)

        # switch from dimension to level names, if necessary
        dim_names: Dict[Hashable, str] = {}
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def copy(self, deep: bool = False, data: Mapping = None) -> "Dataset":
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------

        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])}, coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 -0.8079 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 0 1 2 3 4 5
            bar      (x) <U1 'a' 'b'

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            variables = {k: v.copy(deep=deep) for k, v in self._variables.items()}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")
        else:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    "Data must only contain variables in original "
                    "dataset. Extra variables: {}".format(keys_not_in_vars)
                )
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    "Data must contain all variables in original "
                    "dataset. Data is missing {}".format(keys_missing_from_data)
                )
            variables = {
                k: v.copy(deep=deep, data=data.get(k))
                for k, v in self._variables.items()
            }

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)

        return self._replace(variables, attrs=attrs)

    @property
    def _level_coords(self) -> Dict[str, Hashable]:
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords: Dict[str, Hashable] = {}
        for name, index in self.indexes.items():
            if isinstance(index, pd.MultiIndex):
                level_names = index.names
                (dim,) = self.variables[name].dims
                level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _copy_listed(self, names: Iterable[Hashable]) -> "Dataset":
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables: Dict[Hashable, Variable] = {}
        coord_names = set()
        indexes: Dict[Hashable, pd.Index] = {}

        for name in names:
            try:
                variables[name] = self._variables[name]
            except KeyError:
                ref_name, var_name, var = _get_virtual_variable(
                    self._variables, name, self._level_coords, self.dims
                )
                variables[var_name] = var
                if ref_name in self._coord_names or ref_name in self.dims:
                    coord_names.add(var_name)
                if (var_name,) == var.dims:
                    indexes[var_name] = var.to_index()

        needed_dims: Set[Hashable] = set()
        for v in variables.values():
            needed_dims.update(v.dims)

        dims = {k: self.dims[k] for k in needed_dims}

        for k in self._coord_names:
            if set(self.variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
                if k in self.indexes:
                    indexes[k] = self.indexes[k]

        return self._replace(variables, coord_names, dims, indexes=indexes)

    def _construct_dataarray(self, name: Hashable) -> "DataArray":
        """Construct a DataArray by indexing this dataset
        """
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims
            )

        needed_dims = set(variable.dims)

        coords: Dict[Hashable, Variable] = {}
        for k in self.coords:
            if set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = {k: v for k, v in self._indexes.items() if k in coords}

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)

    def __copy__(self) -> "Dataset":
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> "Dataset":
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    @property
    def _attr_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for attribute-style access
        """
        return self._item_sources + [self.attrs]

    @property
    def _item_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for key-completion
        """
        return [
            self.data_vars,
            self.coords,
            {d: self[d] for d in self.dims},
            LevelCoordinatesSource(self),
        ]

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._variables

    def __len__(self) -> int:
        return len(self.data_vars)

    def __bool__(self) -> bool:
        return bool(self.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)

    def __array__(self, dtype=None):
        raise TypeError(
            "cannot directly convert an xarray.Dataset into a "
            "numpy array. Instead, create an xarray.DataArray "
            "first, either with indexing on the Dataset or by "
            "invoking the `to_array()` method."
        )

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.variables.values())

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        return _LocIndexer(self)

    # FIXME https://github.com/python/mypy/issues/7328
    @overload
    def __getitem__(self, key: Mapping) -> "Dataset":  # type: ignore
        ...

    @overload
    def __getitem__(self, key: Hashable) -> "DataArray":  # type: ignore
        ...

    @overload
    def __getitem__(self, key: Any) -> "Dataset":
        ...

    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))

    def __setitem__(self, key: Hashable, value) -> None:
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError(
                "cannot yet use a dictionary as a key " "to set Dataset values"
            )

        self.update({key: value})

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)
        if key in self.indexes:
            assert self._indexes is not None
            del self._indexes[key]
        self._dims = calculate_dimensions(self._variables)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    def _all_compat(self, other: "Dataset", compat_str: str) -> bool:
        """Helper function for equals and identical
        """

        # some stores (e.g., scipy) do not seem to preserve order, so don't
        # require matching order for equality
        def compat(x: Variable, y: Variable) -> bool:
            return getattr(x, compat_str)(y)

        return self._coord_names == other._coord_names and utils.dict_equiv(
            self._variables, other._variables, compat=compat
        )

    def broadcast_equals(self, other: "Dataset") -> bool:
        """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        See Also
        --------
        Dataset.equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, "broadcast_equals")
        except (TypeError, AttributeError):
            return False

    def equals(self, other: "Dataset") -> bool:
        """Two Datasets are equal if they have matching variables and
        coordinates, all of which are equal.

        Datasets can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``Dataset``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, "equals")
        except (TypeError, AttributeError):
            return False

    def identical(self, other: "Dataset") -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return utils.dict_equiv(self.attrs, other.attrs) and self._all_compat(
                other, "identical"
            )
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing
        """
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables
        """
        return DataVariables(self)
```
### 4 - xarray/core/dataset.py:

Start line: 375, End line: 425

```python
class DataVariables(Mapping[Hashable, "DataArray"]):
    __slots__ = ("_dataset",)

    def __init__(self, dataset: "Dataset"):
        self._dataset = dataset

    def __iter__(self) -> Iterator[Hashable]:
        return (
            key
            for key in self._dataset._variables
            if key not in self._dataset._coord_names
        )

    def __len__(self) -> int:
        return len(self._dataset._variables) - len(self._dataset._coord_names)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._dataset._variables and key not in self._dataset._coord_names

    def __getitem__(self, key: Hashable) -> "DataArray":
        if key not in self._dataset._coord_names:
            return cast("DataArray", self._dataset[key])
        raise KeyError(key)

    def __repr__(self) -> str:
        return formatting.data_vars_repr(self)

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        all_variables = self._dataset.variables
        return Frozen({k: all_variables[k] for k in self})

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython. """
        return [
            key
            for key in self._dataset._ipython_key_completions_()
            if key not in self._dataset._coord_names
        ]


class _LocIndexer:
    __slots__ = ("dataset",)

    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Hashable, Any]) -> "Dataset":
        if not utils.is_dict_like(key):
            raise TypeError("can only lookup dictionaries from Dataset.loc")
        return self.dataset.sel(key)
```
### 5 - xarray/plot/dataset_plot.py:

Start line: 1, End line: 76

```python
import functools

import numpy as np
import pandas as pd

from ..core.alignment import broadcast
from .facetgrid import _easy_facetgrid
from .utils import (
    _add_colorbar,
    _is_numeric,
    _process_cmap_cbar_kwargs,
    get_axis,
    label_from_attrs,
)

# copied from seaborn
_MARKERSIZE_RANGE = np.array([18.0, 72.0])


def _infer_meta_data(ds, x, y, hue, hue_style, add_guide):
    dvars = set(ds.variables.keys())
    error_msg = " must be one of ({:s})".format(", ".join(dvars))

    if x not in dvars:
        raise ValueError("x" + error_msg)

    if y not in dvars:
        raise ValueError("y" + error_msg)

    if hue is not None and hue not in dvars:
        raise ValueError("hue" + error_msg)

    if hue:
        hue_is_numeric = _is_numeric(ds[hue].values)

        if hue_style is None:
            hue_style = "continuous" if hue_is_numeric else "discrete"

        if not hue_is_numeric and (hue_style == "continuous"):
            raise ValueError(
                "Cannot create a colorbar for a non numeric" " coordinate: " + hue
            )

        if add_guide is None or add_guide is True:
            add_colorbar = True if hue_style == "continuous" else False
            add_legend = True if hue_style == "discrete" else False
        else:
            add_colorbar = False
            add_legend = False
    else:
        if add_guide is True:
            raise ValueError("Cannot set add_guide when hue is None.")
        add_legend = False
        add_colorbar = False

    if hue_style is not None and hue_style not in ["discrete", "continuous"]:
        raise ValueError(
            "hue_style must be either None, 'discrete' " "or 'continuous'."
        )

    if hue:
        hue_label = label_from_attrs(ds[hue])
        hue = ds[hue]
    else:
        hue_label = None
        hue = None

    return {
        "add_colorbar": add_colorbar,
        "add_legend": add_legend,
        "hue_label": hue_label,
        "hue_style": hue_style,
        "xlabel": label_from_attrs(ds[x]),
        "ylabel": label_from_attrs(ds[y]),
        "hue": hue,
    }
```
### 6 - xarray/plot/dataset_plot.py:

Start line: 167, End line: 243

```python
def _dsplot(plotfunc):
    commondoc = """
    Parameters
    ----------

    ds : Dataset
    x, y : string
        Variable names for x, y axis.
    hue: str, optional
        Variable by which to color scattered points
    hue_style: str, optional
        Can be either 'discrete' (legend) or 'continuous' (color bar).
    markersize: str, optional (scatter only)
        Variably by which to vary size of scattered points
    size_norm: optional
        Either None or 'Norm' instance to normalize the 'markersize' variable.
    add_guide: bool, optional
        Add a guide that depends on hue_style
            - for "discrete", build a legend.
              This is the default for non-numeric `hue` variables.
            - for "continuous",  build a colorbar
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : integer, optional
        Use together with ``col`` to wrap faceted plots
    ax : matplotlib axes, optional
        If None, uses the current axis. Not applicable when using facets.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only applies
        to FacetGrid plotting.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space. If not provided, this
        will be either be ``viridis`` (if the function infers a sequential
        dataset) or ``RdBu_r`` (if the function infers a diverging dataset).
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. If ``cmap`` is seaborn color palette and the plot type
        is not ``contour`` or ``contourf``, ``levels`` must also be specified.
    colors : discrete colors to plot, optional
        A single color or a list of colors. If the plot type is not ``contour``
        or ``contourf``, the ``levels`` argument is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    **kwargs : optional
        Additional keyword arguments to matplotlib
    """
    # ... other code
```
### 7 - xarray/conventions.py:

Start line: 718, End line: 737

```python
def encode_dataset_coordinates(dataset):
    """Encode coordinates on the given dataset object into variable specific
    and global attributes.

    When possible, this is done according to CF conventions.

    Parameters
    ----------
    dataset : Dataset
        Object to encode.

    Returns
    -------
    variables : dict
    attrs : dict
    """
    non_dim_coord_names = set(dataset.coords) - set(dataset.dims)
    return _encode_coordinates(
        dataset._variables, dataset.attrs, non_dim_coord_names=non_dim_coord_names
    )
```
### 8 - xarray/plot/dataset_plot.py:

Start line: 110, End line: 164

```python
# copied from seaborn
def _parse_size(data, norm):

    import matplotlib as mpl

    if data is None:
        return None

    data = data.values.flatten()

    if not _is_numeric(data):
        levels = np.unique(data)
        numbers = np.arange(1, 1 + len(levels))[::-1]
    else:
        levels = numbers = np.sort(np.unique(data))

    min_width, max_width = _MARKERSIZE_RANGE
    # width_range = min_width, max_width

    if norm is None:
        norm = mpl.colors.Normalize()
    elif isinstance(norm, tuple):
        norm = mpl.colors.Normalize(*norm)
    elif not isinstance(norm, mpl.colors.Normalize):
        err = "``size_norm`` must be None, tuple, " "or Normalize object."
        raise ValueError(err)

    norm.clip = True
    if not norm.scaled():
        norm(np.asarray(numbers))
    # limits = norm.vmin, norm.vmax

    scl = norm(numbers)
    widths = np.asarray(min_width + scl * (max_width - min_width))
    if scl.mask.any():
        widths[scl.mask] = 0
    sizes = dict(zip(levels, widths))

    return pd.Series(sizes)


class _Dataset_PlotMethods:
    """
    Enables use of xarray.plot functions as attributes on a Dataset.
    For example, Dataset.plot.scatter
    """

    def __init__(self, dataset):
        self._ds = dataset

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Dataset.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ds.plot.scatter(...)"
        )
```
### 9 - xarray/core/coordinates.py:

Start line: 221, End line: 244

```python
class DatasetCoordinates(Coordinates):

    def _update_coords(
        self, coords: Dict[Hashable, Variable], indexes: Mapping[Hashable, pd.Index]
    ) -> None:
        from .dataset import calculate_dimensions

        variables = self._data._variables.copy()
        variables.update(coords)

        # check for inconsistent state *before* modifying anything in-place
        dims = calculate_dimensions(variables)
        new_coord_names = set(coords)
        for dim, size in dims.items():
            if dim in variables:
                new_coord_names.add(dim)

        self._data._variables = variables
        self._data._coord_names.update(new_coord_names)
        self._data._dims = dims

        # TODO(shoyer): once ._indexes is always populated by a dict, modify
        # it to update inplace instead.
        original_indexes = dict(self._data.indexes)
        original_indexes.update(indexes)
        self._data._indexes = original_indexes
```
### 10 - xarray/core/dataset.py:

Start line: 1, End line: 136

```python
import copy
import datetime
import functools
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
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
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .utils import (
    Default,
    Frozen,
    SortedKeysDict,
    _check_inplace,
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
### 20 - xarray/core/formatting.py:

Start line: 658, End line: 679

```python
def diff_dataset_repr(a, b, compat):
    summary = [
        "Left and right {} objects are not {}".format(
            type(a).__name__, _compat_to_str(compat)
        )
    ]

    col_width = _calculate_col_width(
        set(_get_col_items(a.variables) + _get_col_items(b.variables))
    )

    summary.append(diff_dim_summary(a, b))
    summary.append(diff_coords_repr(a.coords, b.coords, compat, col_width=col_width))
    summary.append(
        diff_data_vars_repr(a.data_vars, b.data_vars, compat, col_width=col_width)
    )

    if compat == "identical":
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))

    return "\n".join(summary)
```
### 25 - xarray/core/formatting.py:

Start line: 271, End line: 291

```python
def summarize_variable(
    name: Hashable, var, col_width: int, marker: str = " ", max_width: int = None
):
    """Summarize a variable in one line, e.g., for the Dataset.__repr__."""
    if max_width is None:
        max_width_options = OPTIONS["display_width"]
        if not isinstance(max_width_options, int):
            raise TypeError(f"`max_width` value of `{max_width}` is not a valid int")
        else:
            max_width = max_width_options
    first_col = pretty_print(f"  {marker} {name} ", col_width)
    if var.dims:
        dims_str = "({}) ".format(", ".join(map(str, var.dims)))
    else:
        dims_str = ""
    front_str = f"{first_col}{dims_str}{var.dtype} "

    values_width = max_width - len(front_str)
    values_str = inline_variable_array_repr(var, values_width)

    return front_str + values_str
```
### 45 - xarray/core/formatting.py:

Start line: 597, End line: 621

```python
diff_coords_repr = functools.partial(
    _diff_mapping_repr, title="Coordinates", summarizer=summarize_coord
)


diff_data_vars_repr = functools.partial(
    _diff_mapping_repr, title="Data variables", summarizer=summarize_datavar
)


diff_attrs_repr = functools.partial(
    _diff_mapping_repr, title="Attributes", summarizer=summarize_attr
)


def _compat_to_str(compat):
    if callable(compat):
        compat = compat.__name__

    if compat == "equals":
        return "equal"
    elif compat == "allclose":
        return "close"
    else:
        return compat
```
### 46 - xarray/core/formatting.py:

Start line: 312, End line: 324

```python
def summarize_coord(name: Hashable, var, col_width: int):
    is_index = name in var.dims
    marker = "*" if is_index else " "
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            return "\n".join(
                [
                    _summarize_coord_multiindex(coord, col_width, marker),
                    _summarize_coord_levels(coord, col_width),
                ]
            )
    return summarize_variable(name, var.variable, col_width, marker)
```
