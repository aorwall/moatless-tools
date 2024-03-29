# pydata__xarray-4819

| **pydata/xarray** | `a2b1712afd957deaf189c9b1a04e469596d853c9` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 2581 |
| **Any found context length** | 2581 |
| **Avg pos** | 14.5 |
| **Min pos** | 6 |
| **Max pos** | 23 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2247,6 +2247,28 @@ def drop_sel(
         ds = self._to_temp_dataset().drop_sel(labels, errors=errors)
         return self._from_temp_dataset(ds)
 
+    def drop_isel(self, indexers=None, **indexers_kwargs):
+        """Drop index positions from this DataArray.
+
+        Parameters
+        ----------
+        indexers : mapping of hashable to Any
+            Index locations to drop
+        **indexers_kwargs : {dim: position, ...}, optional
+            The keyword arguments form of ``dim`` and ``positions``
+
+        Returns
+        -------
+        dropped : DataArray
+
+        Raises
+        ------
+        IndexError
+        """
+        dataset = self._to_temp_dataset()
+        dataset = dataset.drop_isel(indexers=indexers, **indexers_kwargs)
+        return self._from_temp_dataset(dataset)
+
     def dropna(
         self, dim: Hashable, how: str = "any", thresh: int = None
     ) -> "DataArray":
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4053,13 +4053,78 @@ def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
                 labels_for_dim = [labels_for_dim]
             labels_for_dim = np.asarray(labels_for_dim)
             try:
-                index = self.indexes[dim]
+                index = self.get_index(dim)
             except KeyError:
                 raise ValueError("dimension %r does not have coordinate labels" % dim)
             new_index = index.drop(labels_for_dim, errors=errors)
             ds = ds.loc[{dim: new_index}]
         return ds
 
+    def drop_isel(self, indexers=None, **indexers_kwargs):
+        """Drop index positions from this Dataset.
+
+        Parameters
+        ----------
+        indexers : mapping of hashable to Any
+            Index locations to drop
+        **indexers_kwargs : {dim: position, ...}, optional
+            The keyword arguments form of ``dim`` and ``positions``
+
+        Returns
+        -------
+        dropped : Dataset
+
+        Raises
+        ------
+        IndexError
+
+        Examples
+        --------
+        >>> data = np.arange(6).reshape(2, 3)
+        >>> labels = ["a", "b", "c"]
+        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
+        >>> ds
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 3)
+        Coordinates:
+          * y        (y) <U1 'a' 'b' 'c'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 0 1 2 3 4 5
+        >>> ds.drop_isel(y=[0, 2])
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 1)
+        Coordinates:
+          * y        (y) <U1 'b'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 1 4
+        >>> ds.drop_isel(y=1)
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 2)
+        Coordinates:
+          * y        (y) <U1 'a' 'c'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 0 2 3 5
+        """
+
+        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "drop")
+
+        ds = self
+        dimension_index = {}
+        for dim, pos_for_dim in indexers.items():
+            # Don't cast to set, as it would harm performance when labels
+            # is a large numpy array
+            if utils.is_scalar(pos_for_dim):
+                pos_for_dim = [pos_for_dim]
+            pos_for_dim = np.asarray(pos_for_dim)
+            index = self.get_index(dim)
+            new_index = index.delete(pos_for_dim)
+            dimension_index[dim] = new_index
+        ds = ds.loc[dimension_index]
+        return ds
+
     def drop_dims(
         self, drop_dims: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
     ) -> "Dataset":

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/dataarray.py | 2250 | 2250 | 6 | 1 | 2581
| xarray/core/dataset.py | 4056 | 4056 | 23 | 3 | 23048


## Problem Statement

```
drop_sel indices in dimension that doesn't have coordinates?
<!-- Please do a quick search of existing issues to make sure that this has not been asked before. -->

**Is your feature request related to a problem? Please describe.**

I am trying to drop particular indices from a dimension that doesn't have coordinates.

Following: [drop_sel() documentation](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.drop_sel.html#xarray.Dataset.drop_sel),
but leaving out the coordinate labels:
\`\`\`python
data = np.random.randn(2, 3)
ds = xr.Dataset({"A": (["x", "y"], data)})
ds.drop_sel(y=[1])
\`\`\`
gives me an error.

**Describe the solution you'd like**

I would think `drop_isel` should exist and work in analogy to `drop_sel` as `isel` does to `sel`.

**Describe alternatives you've considered**

As far as I know, I could either create coordinates especially to in order to drop, or rebuild a new dataset. Both are not congenial. (I'd be grateful to know if there is actually a straightforward way to do this I've overlooked.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/dataarray.py** | 2219 | 2248| 238 | 238 | 36496 | 
| 2 | 2 xarray/core/utils.py | 809 | 854| 290 | 528 | 42703 | 
| 3 | **2 xarray/core/dataarray.py** | 2199 | 2217| 133 | 661 | 42703 | 
| 4 | 2 xarray/core/utils.py | 857 | 896| 270 | 931 | 42703 | 
| 5 | **3 xarray/core/dataset.py** | 4147 | 4424| 1453 | 2384 | 100189 | 
| **-> 6 <-** | **3 xarray/core/dataarray.py** | 2250 | 2272| 197 | 2581 | 100189 | 
| 7 | **3 xarray/core/dataarray.py** | 1125 | 1213| 811 | 3392 | 100189 | 
| 8 | **3 xarray/core/dataset.py** | 2036 | 2689| 6104 | 9496 | 100189 | 
| 9 | 4 xarray/core/coordinates.py | 247 | 259| 116 | 9612 | 103026 | 
| 10 | 5 xarray/core/common.py | 370 | 383| 149 | 9761 | 118369 | 
| 11 | 6 xarray/core/indexes.py | 70 | 87| 121 | 9882 | 119363 | 
| 12 | 6 xarray/core/common.py | 1095 | 1152| 580 | 10462 | 119363 | 
| 13 | 6 xarray/core/coordinates.py | 222 | 245| 198 | 10660 | 119363 | 
| 14 | **6 xarray/core/dataarray.py** | 1062 | 1123| 544 | 11204 | 119363 | 
| 15 | 6 xarray/core/common.py | 1154 | 1230| 845 | 12049 | 119363 | 
| 16 | 6 xarray/core/common.py | 1231 | 1270| 328 | 12377 | 119363 | 
| 17 | **6 xarray/core/dataarray.py** | 1864 | 1893| 218 | 12595 | 119363 | 
| 18 | 7 xarray/core/combine.py | 552 | 750| 2476 | 15071 | 126602 | 
| 19 | 7 xarray/core/coordinates.py | 305 | 325| 197 | 15268 | 126602 | 
| 20 | **7 xarray/core/dataarray.py** | 3858 | 3953| 934 | 16202 | 126602 | 
| 21 | 8 xarray/core/indexing.py | 1132 | 1152| 146 | 16348 | 138523 | 
| 22 | **8 xarray/core/dataset.py** | 1 | 135| 643 | 16991 | 138523 | 
| **-> 23 <-** | **8 xarray/core/dataset.py** | 3386 | 4145| 6057 | 23048 | 138523 | 
| 24 | 9 xarray/core/groupby.py | 681 | 707| 265 | 23313 | 146387 | 
| 25 | 10 xarray/core/alignment.py | 82 | 271| 1952 | 25265 | 152448 | 
| 26 | 10 xarray/core/coordinates.py | 285 | 303| 174 | 25439 | 152448 | 
| 27 | 11 xarray/core/duck_array_ops.py | 117 | 141| 275 | 25714 | 157680 | 
| 28 | 11 xarray/core/coordinates.py | 79 | 111| 301 | 26015 | 157680 | 
| 29 | 11 xarray/core/indexing.py | 108 | 203| 875 | 26890 | 157680 | 
| 30 | 12 xarray/core/variable.py | 1123 | 1156| 322 | 27212 | 179677 | 
| 31 | 12 xarray/core/coordinates.py | 33 | 77| 289 | 27501 | 179677 | 
| 32 | 13 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 28234 | 181237 | 
| 33 | 14 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 28647 | 181650 | 
| 34 | 14 xarray/core/utils.py | 763 | 788| 253 | 28900 | 181650 | 
| 35 | **14 xarray/core/dataarray.py** | 2173 | 2197| 208 | 29108 | 181650 | 
| 36 | 14 xarray/core/common.py | 985 | 1094| 1167 | 30275 | 181650 | 
| 37 | 15 xarray/plot/utils.py | 749 | 777| 384 | 30659 | 188288 | 
| 38 | 15 xarray/core/coordinates.py | 146 | 181| 262 | 30921 | 188288 | 
| 39 | 15 xarray/core/combine.py | 47 | 115| 536 | 31457 | 188288 | 
| 40 | 16 xarray/plot/dataset_plot.py | 1 | 74| 510 | 31967 | 191737 | 
| 41 | 16 xarray/core/coordinates.py | 348 | 387| 371 | 32338 | 191737 | 
| 42 | **16 xarray/core/dataarray.py** | 1 | 87| 442 | 32780 | 191737 | 
| 43 | 17 xarray/core/concat.py | 242 | 263| 163 | 32943 | 196582 | 
| 44 | 17 xarray/core/indexes.py | 1 | 36| 317 | 33260 | 196582 | 
| 45 | 17 xarray/core/coordinates.py | 184 | 220| 268 | 33528 | 196582 | 
| 46 | **17 xarray/core/dataarray.py** | 4052 | 4153| 1061 | 34589 | 196582 | 
| 47 | 17 asv_bench/benchmarks/indexing.py | 145 | 162| 145 | 34734 | 196582 | 
| 48 | **17 xarray/core/dataset.py** | 298 | 353| 452 | 35186 | 196582 | 


### Hint

```
I don't know of an easy way (which does not mean that there is none). `drop_sel` could be adjusted to work with _dimensions without coordinates_ by replacing

https://github.com/pydata/xarray/blob/ff6b1f542e52dc330e294fd367f846e02c2955a2/xarray/core/dataset.py#L4038

by `index = self.get_index(dim)`. That would then be analog to `sel`. I think `drop_isel` would also be a welcome addition.
Can I work on this?
Sure. PRs are always welcome! 
```

## Patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2247,6 +2247,28 @@ def drop_sel(
         ds = self._to_temp_dataset().drop_sel(labels, errors=errors)
         return self._from_temp_dataset(ds)
 
+    def drop_isel(self, indexers=None, **indexers_kwargs):
+        """Drop index positions from this DataArray.
+
+        Parameters
+        ----------
+        indexers : mapping of hashable to Any
+            Index locations to drop
+        **indexers_kwargs : {dim: position, ...}, optional
+            The keyword arguments form of ``dim`` and ``positions``
+
+        Returns
+        -------
+        dropped : DataArray
+
+        Raises
+        ------
+        IndexError
+        """
+        dataset = self._to_temp_dataset()
+        dataset = dataset.drop_isel(indexers=indexers, **indexers_kwargs)
+        return self._from_temp_dataset(dataset)
+
     def dropna(
         self, dim: Hashable, how: str = "any", thresh: int = None
     ) -> "DataArray":
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4053,13 +4053,78 @@ def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
                 labels_for_dim = [labels_for_dim]
             labels_for_dim = np.asarray(labels_for_dim)
             try:
-                index = self.indexes[dim]
+                index = self.get_index(dim)
             except KeyError:
                 raise ValueError("dimension %r does not have coordinate labels" % dim)
             new_index = index.drop(labels_for_dim, errors=errors)
             ds = ds.loc[{dim: new_index}]
         return ds
 
+    def drop_isel(self, indexers=None, **indexers_kwargs):
+        """Drop index positions from this Dataset.
+
+        Parameters
+        ----------
+        indexers : mapping of hashable to Any
+            Index locations to drop
+        **indexers_kwargs : {dim: position, ...}, optional
+            The keyword arguments form of ``dim`` and ``positions``
+
+        Returns
+        -------
+        dropped : Dataset
+
+        Raises
+        ------
+        IndexError
+
+        Examples
+        --------
+        >>> data = np.arange(6).reshape(2, 3)
+        >>> labels = ["a", "b", "c"]
+        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
+        >>> ds
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 3)
+        Coordinates:
+          * y        (y) <U1 'a' 'b' 'c'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 0 1 2 3 4 5
+        >>> ds.drop_isel(y=[0, 2])
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 1)
+        Coordinates:
+          * y        (y) <U1 'b'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 1 4
+        >>> ds.drop_isel(y=1)
+        <xarray.Dataset>
+        Dimensions:  (x: 2, y: 2)
+        Coordinates:
+          * y        (y) <U1 'a' 'c'
+        Dimensions without coordinates: x
+        Data variables:
+            A        (x, y) int64 0 2 3 5
+        """
+
+        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "drop")
+
+        ds = self
+        dimension_index = {}
+        for dim, pos_for_dim in indexers.items():
+            # Don't cast to set, as it would harm performance when labels
+            # is a large numpy array
+            if utils.is_scalar(pos_for_dim):
+                pos_for_dim = [pos_for_dim]
+            pos_for_dim = np.asarray(pos_for_dim)
+            index = self.get_index(dim)
+            new_index = index.delete(pos_for_dim)
+            dimension_index[dim] = new_index
+        ds = ds.loc[dimension_index]
+        return ds
+
     def drop_dims(
         self, drop_dims: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
     ) -> "Dataset":

```

## Test Patch

```diff
diff --git a/xarray/tests/test_dataarray.py b/xarray/tests/test_dataarray.py
--- a/xarray/tests/test_dataarray.py
+++ b/xarray/tests/test_dataarray.py
@@ -2327,6 +2327,12 @@ def test_drop_index_labels(self):
         with pytest.warns(DeprecationWarning):
             arr.drop([0, 1, 3], dim="y", errors="ignore")
 
+    def test_drop_index_positions(self):
+        arr = DataArray(np.random.randn(2, 3), dims=["x", "y"])
+        actual = arr.drop_sel(y=[0, 1])
+        expected = arr[:, 2:]
+        assert_identical(actual, expected)
+
     def test_dropna(self):
         x = np.random.randn(4, 4)
         x[::2, 0] = np.nan
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -2371,8 +2371,12 @@ def test_drop_index_labels(self):
             data.drop(DataArray(["a", "b", "c"]), dim="x", errors="ignore")
         assert_identical(expected, actual)
 
-        with raises_regex(ValueError, "does not have coordinate labels"):
-            data.drop_sel(y=1)
+        actual = data.drop_sel(y=[1])
+        expected = data.isel(y=[0, 2])
+        assert_identical(expected, actual)
+
+        with raises_regex(KeyError, "not found in axis"):
+            data.drop_sel(x=0)
 
     def test_drop_labels_by_keyword(self):
         data = Dataset(
@@ -2410,6 +2414,34 @@ def test_drop_labels_by_keyword(self):
         with pytest.raises(ValueError):
             data.drop(dim="x", x="a")
 
+    def test_drop_labels_by_position(self):
+        data = Dataset(
+            {"A": (["x", "y"], np.random.randn(2, 6)), "x": ["a", "b"], "y": range(6)}
+        )
+        # Basic functionality.
+        assert len(data.coords["x"]) == 2
+
+        actual = data.drop_isel(x=0)
+        expected = data.drop_sel(x="a")
+        assert_identical(expected, actual)
+
+        actual = data.drop_isel(x=[0])
+        expected = data.drop_sel(x=["a"])
+        assert_identical(expected, actual)
+
+        actual = data.drop_isel(x=[0, 1])
+        expected = data.drop_sel(x=["a", "b"])
+        assert_identical(expected, actual)
+        assert actual.coords["x"].size == 0
+
+        actual = data.drop_isel(x=[0, 1], y=range(0, 6, 2))
+        expected = data.drop_sel(x=["a", "b"], y=range(0, 6, 2))
+        assert_identical(expected, actual)
+        assert actual.coords["x"].size == 0
+
+        with pytest.raises(KeyError):
+            data.drop_isel(z=1)
+
     def test_drop_dims(self):
         data = xr.Dataset(
             {

```


## Code snippets

### 1 - xarray/core/dataarray.py:

Start line: 2219, End line: 2248

```python
class DataArray(AbstractArray, DataWithCoords):

    def drop_sel(
        self,
        labels: Mapping[Hashable, Any] = None,
        *,
        errors: str = "raise",
        **labels_kwargs,
    ) -> "DataArray":
        """Drop index labels from this DataArray.

        Parameters
        ----------
        labels : mapping of hashable to Any
            Index labels to drop
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if
            any of the index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.
        **labels_kwargs : {dim: label, ...}, optional
            The keyword arguments form of ``dim`` and ``labels``

        Returns
        -------
        dropped : DataArray
        """
        if labels_kwargs or isinstance(labels, dict):
            labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        ds = self._to_temp_dataset().drop_sel(labels, errors=errors)
        return self._from_temp_dataset(ds)
```
### 2 - xarray/core/utils.py:

Start line: 809, End line: 854

```python
def drop_dims_from_indexers(
    indexers: Mapping[Hashable, Any],
    dims: Union[list, Mapping[Hashable, int]],
    missing_dims: str,
) -> Mapping[Hashable, Any]:
    """Depending on the setting of missing_dims, drop any dimensions from indexers that
    are not present in dims.

    Parameters
    ----------
    indexers : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return indexers

    elif missing_dims == "warn":

        # don't modify input
        indexers = dict(indexers)

        invalid = indexers.keys() - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )
        for key in invalid:
            indexers.pop(key)

        return indexers

    elif missing_dims == "ignore":
        return {key: val for key, val in indexers.items() if key in dims}

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
```
### 3 - xarray/core/dataarray.py:

Start line: 2199, End line: 2217

```python
class DataArray(AbstractArray, DataWithCoords):

    def drop(
        self,
        labels: Mapping = None,
        dim: Hashable = None,
        *,
        errors: str = "raise",
        **labels_kwargs,
    ) -> "DataArray":
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        DataArray.drop_vars
        DataArray.drop_sel
        """
        ds = self._to_temp_dataset().drop(labels, dim, errors=errors)
        return self._from_temp_dataset(ds)
```
### 4 - xarray/core/utils.py:

Start line: 857, End line: 896

```python
def drop_missing_dims(
    supplied_dims: Collection, dims: Collection, missing_dims: str
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
        supplied_dims_set = set(val for val in supplied_dims if val is not ...)
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
### 5 - xarray/core/dataset.py:

Start line: 4147, End line: 4424

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def dropna(
        self,
        dim: Hashable,
        how: str = "any",
        thresh: int = None,
        subset: Iterable[Hashable] = None,
    ):
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, default: "any"
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default: None
            If supplied, require this many non-NA values.
        subset : iterable of hashable, optional
            Which variables to check for missing values. By default, all
            variables in the dataset are checked.

        Returns
        -------
        Dataset
        """
        # TODO: consider supporting multiple dimensions? Or not, given that
        # there are some ugly edge cases, e.g., pandas's dropna differs
        # depending on the order of the supplied axes.

        if dim not in self.dims:
            raise ValueError("%s must be a single dataset dimension" % dim)

        if subset is None:
            subset = iter(self.data_vars)

        count = np.zeros(self.dims[dim], dtype=np.int64)
        size = 0

        for k in subset:
            array = self._variables[k]
            if dim in array.dims:
                dims = [d for d in array.dims if d != dim]
                count += np.asarray(array.count(dims))  # type: ignore
                size += np.prod([self.dims[d] for d in dims])

        if thresh is not None:
            mask = count >= thresh
        elif how == "any":
            mask = count == size
        elif how == "all":
            mask = count > 0
        elif how is not None:
            raise ValueError("invalid how option: %s" % how)
        else:
            raise TypeError("must specify how or thresh")

        return self.isel({dim: mask})

    def fillna(self, value: Any) -> "Dataset":
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray, DataArray, dict or Dataset
            Used to fill all matching missing values in this dataset's data
            variables. Scalars, ndarrays or DataArrays arguments are used to
            fill all data with aligned coordinates (for DataArrays).
            Dictionaries or datasets match data variables and then align
            coordinates if necessary.

        Returns
        -------
        Dataset

        Examples
        --------

        >>> import numpy as np
        >>> import xarray as xr
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5]),
        ...         "D": ("x", [np.nan, 3, np.nan, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3]},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 nan 2.0 nan 0.0
            B        (x) float64 3.0 4.0 nan 1.0
            C        (x) float64 nan nan nan 5.0
            D        (x) float64 nan 3.0 nan 4.0

        Replace all `NaN` values with 0s.

        >>> ds.fillna(0)
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 0.0 2.0 0.0 0.0
            B        (x) float64 3.0 4.0 0.0 1.0
            C        (x) float64 0.0 0.0 0.0 5.0
            D        (x) float64 0.0 3.0 0.0 4.0

        Replace all `NaN` elements in column ‘A’, ‘B’, ‘C’, and ‘D’, with 0, 1, 2, and 3 respectively.

        >>> values = {"A": 0, "B": 1, "C": 2, "D": 3}
        >>> ds.fillna(value=values)
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 0.0 2.0 0.0 0.0
            B        (x) float64 3.0 4.0 1.0 1.0
            C        (x) float64 2.0 2.0 2.0 5.0
            D        (x) float64 3.0 3.0 3.0 4.0
        """
        if utils.is_dict_like(value):
            value_keys = getattr(value, "data_vars", value).keys()
            if not set(value_keys) <= set(self.data_vars.keys()):
                raise ValueError(
                    "all variables in the argument to `fillna` "
                    "must be contained in the original dataset"
                )
        out = ops.fillna(self, value)
        return out

    def interpolate_na(
        self,
        dim: Hashable = None,
        method: str = "linear",
        limit: int = None,
        use_coordinate: Union[bool, Hashable] = True,
        max_gap: Union[
            int, float, str, pd.Timedelta, np.timedelta64, datetime.timedelta
        ] = None,
        **kwargs: Any,
    ) -> "Dataset":
        # ... other code
```
### 6 - xarray/core/dataarray.py:

Start line: 2250, End line: 2272

```python
class DataArray(AbstractArray, DataWithCoords):

    def dropna(
        self, dim: Hashable, how: str = "any", thresh: int = None
    ) -> "DataArray":
        """Returns a new array with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, optional
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default: None
            If supplied, require this many non-NA values.

        Returns
        -------
        DataArray
        """
        ds = self._to_temp_dataset().dropna(dim, how=how, thresh=thresh)
        return self._from_temp_dataset(ds)
```
### 7 - xarray/core/dataarray.py:

Start line: 1125, End line: 1213

```python
class DataArray(AbstractArray, DataWithCoords):

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance=None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by selecting index
        labels along the specified dimension(s).

        In contrast to `DataArray.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        .. warning::

          Do not try to assign values when using any of the indexing methods
          ``isel`` or ``sel``::

            da = xr.DataArray([0, 1, 2, 3], dims=['x'])
            # DO NOT do this
            da.isel(x=[0, 1, 2])[1] = -1

          Assigning values with the chained indexing using ``.sel`` or
          ``.isel`` fails silently.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            A new DataArray with the same contents as this DataArray, except the
            data and each dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this DataArray, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel

        """
        ds = self._to_temp_dataset().sel(
            indexers=indexers,
            drop=drop,
            method=method,
            tolerance=tolerance,
            **indexers_kwargs,
        )
        return self._from_temp_dataset(ds)
```
### 8 - xarray/core/dataset.py:

Start line: 2036, End line: 2689

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed along the specified
        dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            return self._isel_fancy(indexers, drop=drop, missing_dims=missing_dims)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's
        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        variables = {}
        dims: Dict[Hashable, Tuple[int, ...]] = {}
        coord_names = self._coord_names.copy()
        indexes = self._indexes.copy() if self._indexes is not None else None

        for var_name, var_value in self._variables.items():
            var_indexers = {k: v for k, v in indexers.items() if k in var_value.dims}
            if var_indexers:
                var_value = var_value.isel(var_indexers)
                if drop and var_value.ndim == 0 and var_name in coord_names:
                    coord_names.remove(var_name)
                    if indexes:
                        indexes.pop(var_name, None)
                    continue
                if indexes and var_name in indexes:
                    if var_value.ndim == 1:
                        indexes[var_name] = var_value.to_index()
                    else:
                        del indexes[var_name]
            variables[var_name] = var_value
            dims.update(zip(var_value.dims, var_value.shape))

        return self._construct_direct(
            variables=variables,
            coord_names=coord_names,
            dims=dims,
            attrs=self._attrs,
            indexes=indexes,
            encoding=self._encoding,
            file_obj=self._file_obj,
        )

    def _isel_fancy(
        self,
        indexers: Mapping[Hashable, Any],
        *,
        drop: bool,
        missing_dims: str = "raise",
    ) -> "Dataset":
        # Note: we need to preserve the original indexers variable in order to merge the
        # coords below
        indexers_list = list(self._validate_indexers(indexers, missing_dims))

        variables: Dict[Hashable, Variable] = {}
        indexes: Dict[Hashable, pd.Index] = {}

        for name, var in self.variables.items():
            var_indexers = {k: v for k, v in indexers_list if k in var.dims}
            if drop and name in var_indexers:
                continue  # drop this variable

            if name in self.indexes:
                new_var, new_index = isel_variable_and_index(
                    name, var, self.indexes[name], var_indexers
                )
                if new_index is not None:
                    indexes[name] = new_index
            elif var_indexers:
                new_var = var.isel(indexers=var_indexers)
            else:
                new_var = var.copy(deep=False)

            variables[name] = new_var

        coord_names = self._coord_names & variables.keys()
        selected = self._replace_with_new_dims(variables, coord_names, indexes)

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(indexers)
        variables.update(coord_vars)
        indexes.update(new_indexes)
        coord_names = self._coord_names & variables.keys() | coord_vars.keys()
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.


        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        pos_indexers, new_indexes = remap_label_indexers(
            self, indexers=indexers, method=method, tolerance=tolerance
        )
        result = self.isel(indexers=pos_indexers, drop=drop)
        return result._overwrite_indexes(new_indexes)

    def head(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with the first `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.


        See Also
        --------
        Dataset.tail
        Dataset.thin
        DataArray.head
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "head")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
        indexers_slices = {k: slice(val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def tail(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with the last `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.


        See Also
        --------
        Dataset.head
        Dataset.thin
        DataArray.tail
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "tail")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
        indexers_slices = {
            k: slice(-val, None) if val != 0 else slice(val)
            for k, val in indexers.items()
        }
        return self.isel(indexers_slices)

    def thin(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed along every `n`-th
        value for the specified dimension(s)

        Parameters
        ----------
        indexers : dict or int
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.


        See Also
        --------
        Dataset.head
        Dataset.tail
        DataArray.thin
        """
        if (
            not indexers_kwargs
            and not isinstance(indexers, int)
            and not is_dict_like(indexers)
        ):
            raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "thin")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
            elif v == 0:
                raise ValueError("step cannot be zero")
        indexers_slices = {k: slice(None, None, val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def broadcast_like(
        self, other: Union["Dataset", "DataArray"], exclude: Iterable[Hashable] = None
    ) -> "Dataset":
        """Broadcast this DataArray against another Dataset or DataArray.
        This is equivalent to xr.broadcast(other, self)[1]

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

        """
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        args = align(other, self, join="outer", copy=False, exclude=exclude)

        dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)

        return _broadcast_helper(args[1], exclude, dims_map, common_coords)

    def reindex_like(
        self,
        other: Union["Dataset", "DataArray"],
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
    ) -> "Dataset":
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
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for filling index values from other not found in this
            dataset:

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
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like maps
            variable names to fill values.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but coordinates from the
            other object.

        See Also
        --------
        Dataset.reindex
        align
        """
        indexers = alignment.reindex_like_indexers(self, other)
        return self.reindex(
            indexers=indexers,
            method=method,
            copy=copy,
            fill_value=fill_value,
            tolerance=tolerance,
        )

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
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
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
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like,
            maps variable names (including coordinates) to fill values.
        sparse : bool, default: False
            use sparse-array.
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
            temperature  (station) float64 10.98 14.3 12.06 10.9
            pressure     (station) float64 211.8 322.9 218.8 445.9
        >>> x.indexes
        station: Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ["boston", "austin", "seattle", "lincoln"]
        >>> x.reindex({"station": new_index})
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 nan 12.06 nan
            pressure     (station) float64 211.8 nan 218.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({"station": new_index}, fill_value=0)
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 0.0 218.8 0.0

        We can also use different fill values for each variable.

        >>> x.reindex(
        ...     {"station": new_index}, fill_value={"temperature": 0, "pressure": 100}
        ... )
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 100.0 218.8 100.0

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
            pressure     (time) float64 481.8 191.7 395.9 264.4 284.0 462.8

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range("12/29/2018", periods=10, freq="D")
        >>> x2.reindex({"time": time_index2})
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 nan nan nan 481.8 ... 264.4 284.0 462.8 nan

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
            pressure     (time) float64 481.8 481.8 481.8 481.8 ... 284.0 462.8 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 1)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03
        Data variables:
            temperature  (time) float64 nan
            pressure     (time) float64 395.9
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 2)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan
            pressure     (time) float64 395.9 nan

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
### 9 - xarray/core/coordinates.py:

Start line: 247, End line: 259

```python
class DatasetCoordinates(Coordinates):

    def __delitem__(self, key: Hashable) -> None:
        if key in self:
            del self._data[key]
        else:
            raise KeyError(f"{key!r} is not a coordinate variable.")

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython. """
        return [
            key
            for key in self._data._ipython_key_completions_()
            if key not in self._data.data_vars
        ]
```
### 10 - xarray/core/common.py:

Start line: 370, End line: 383

```python
class DataWithCoords(SupportsArithmetic, AttrAccessMixin):

    def get_index(self, key: Hashable) -> pd.Index:
        """Get an index for a dimension, with fall-back to a default RangeIndex"""
        if key not in self.dims:
            raise KeyError(key)

        try:
            return self.indexes[key]
        except KeyError:
            return pd.Index(range(self.sizes[key]), name=key)

    def _calc_assign_results(
        self: C, kwargs: Mapping[Hashable, Union[T, Callable[[C], T]]]
    ) -> Dict[Hashable, T]:
        return {k: v(self) if callable(v) else v for k, v in kwargs.items()}
```
### 14 - xarray/core/dataarray.py:

Start line: 1062, End line: 1123

```python
class DataArray(AbstractArray, DataWithCoords):

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by integer indexing
        along the specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(
                indexers, drop=drop, missing_dims=missing_dims
            )
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)

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
### 17 - xarray/core/dataarray.py:

Start line: 1864, End line: 1893

```python
class DataArray(AbstractArray, DataWithCoords):

    def reset_index(
        self,
        dims_or_levels: Union[Hashable, Sequence[Hashable]],
        drop: bool = False,
    ) -> Optional["DataArray"]:
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : hashable or sequence of hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
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
        """
        coords, _ = split_indexes(
            dims_or_levels, self._coords, set(), self._level_coords, drop=drop
        )
        return self._replace(coords=coords)
```
### 20 - xarray/core/dataarray.py:

Start line: 3858, End line: 3953

```python
class DataArray(AbstractArray, DataWithCoords):

    def idxmin(
        self,
        dim: Hashable = None,
        skipna: bool = None,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool = None,
    ) -> "DataArray":
        """Return the coordinate label of the minimum value along a dimension.

        Returns a new `DataArray` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to minimum
        values along that dimension.

        In comparison to :py:meth:`~DataArray.argmin`, this returns the
        coordinate label while :py:meth:`~DataArray.argmin` returns the index.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to apply `idxmin`.  This is optional for 1D
            arrays, but required for arrays with 2 or more dimensions.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for ``float``, ``complex``, and ``object``
            dtypes; other dtypes either do not have a sentinel missing value
            (``int``) or ``skipna=True`` has not been implemented
            (``datetime64`` or ``timedelta64``).
        fill_value : Any, default: NaN
            Value to be filled in case all of the values along a dimension are
            null.  By default this is NaN.  The fill value and result are
            automatically converted to a compatible dtype if possible.
            Ignored if ``skipna`` is False.
        keep_attrs : bool, default: False
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one.  If False (default), the new object
            will be returned without attributes.

        Returns
        -------
        reduced : DataArray
            New `DataArray` object with `idxmin` applied to its data and the
            indicated dimension removed.

        See also
        --------
        Dataset.idxmin, DataArray.idxmax, DataArray.min, DataArray.argmin

        Examples
        --------

        >>> array = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array.min()
        <xarray.DataArray ()>
        array(-2)
        >>> array.argmin()
        <xarray.DataArray ()>
        array(4)
        >>> array.idxmin()
        <xarray.DataArray 'x' ()>
        array('e', dtype='<U1')

        >>> array = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.NaN, 2.0, np.NaN, -2.0],
        ...         [np.NaN, np.NaN, 1.0, np.NaN, np.NaN],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": np.arange(5.0) ** 2},
        ... )
        >>> array.min(dim="x")
        <xarray.DataArray (y: 3)>
        array([-2., -4.,  1.])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.argmin(dim="x")
        <xarray.DataArray (y: 3)>
        array([4, 0, 2])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.idxmin(dim="x")
        <xarray.DataArray 'x' (y: 3)>
        array([16.,  0.,  4.])
        Coordinates:
          * y        (y) int64 -1 0 1
        """
        return computation._calc_idxminmax(
            array=self,
            func=lambda x, *args, **kwargs: x.argmin(*args, **kwargs),
            dim=dim,
            skipna=skipna,
            fill_value=fill_value,
            keep_attrs=keep_attrs,
        )
```
### 22 - xarray/core/dataset.py:

Start line: 1, End line: 135

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
from .pycompat import is_duck_dask_array
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    SortedKeysDict,
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
### 23 - xarray/core/dataset.py:

Start line: 3386, End line: 4145

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def set_index(
        self,
        indexes: Mapping[Hashable, Union[Hashable, Sequence[Hashable]]] = None,
        append: bool = False,
        **indexes_kwargs: Union[Hashable, Sequence[Hashable]],
    ) -> "Dataset":
        """Set Dataset (multi-)indexes using one or more existing coordinates
        or variables.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> ds = xr.Dataset({"v": arr})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0
        >>> ds.set_index(x="a")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0

        See Also
        --------
        Dataset.reset_index
        Dataset.swap_dims
        """
        indexes = either_dict_or_kwargs(indexes, indexes_kwargs, "set_index")
        variables, coord_names = merge_indexes(
            indexes, self._variables, self._coord_names, append=append
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reset_index(
        self,
        dims_or_levels: Union[Hashable, Sequence[Hashable]],
        drop: bool = False,
    ) -> "Dataset":
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : str or list
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        """
        variables, coord_names = split_indexes(
            dims_or_levels,
            self._variables,
            self._coord_names,
            cast(Mapping[Hashable, Hashable], self._level_coords),
            drop=drop,
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reorder_levels(
        self,
        dim_order: Mapping[Hashable, Sequence[int]] = None,
        **dim_order_kwargs: Sequence[int],
    ) -> "Dataset":
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs : optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs, "reorder_levels")
        variables = self._variables.copy()
        indexes = dict(self.indexes)
        for dim, order in dim_order.items():
            coord = self._variables[dim]
            index = self.indexes[dim]
            if not isinstance(index, pd.MultiIndex):
                raise ValueError(f"coordinate {dim} has no MultiIndex")
            new_index = index.reorder_levels(order)
            variables[dim] = IndexVariable(coord.dims, new_index)
            indexes[dim] = new_index

        return self._replace(variables, indexes=indexes)

    def _stack_once(self, dims, new_dim):
        if ... in dims:
            dims = list(infix_dims(dims, self.dims))
        variables = {}
        for name, var in self.variables.items():
            if name not in dims:
                if any(d in var.dims for d in dims):
                    add_dims = [d for d in dims if d not in var.dims]
                    vdims = list(var.dims) + add_dims
                    shape = [self.dims[d] for d in vdims]
                    exp_var = var.set_dims(vdims, shape)
                    stacked_var = exp_var.stack(**{new_dim: dims})
                    variables[name] = stacked_var
                else:
                    variables[name] = var.copy(deep=False)

        # consider dropping levels that are unused?
        levels = [self.get_index(dim) for dim in dims]
        idx = utils.multiindex_from_product_levels(levels, names=dims)
        variables[new_dim] = IndexVariable(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | {new_dim}

        indexes = {k: v for k, v in self.indexes.items() if k not in dims}
        indexes[new_dim] = idx

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def stack(
        self,
        dimensions: Mapping[Hashable, Sequence[Hashable]] = None,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> "Dataset":
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : mapping of hashable to sequence of hashable
            Mapping of the form `new_name=(dim1, dim2, ...)`. Names of new
            dimensions, and the existing dimensions that they replace. An
            ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See also
        --------
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def to_stacked_array(
        self,
        new_dim: Hashable,
        sample_dims: Sequence[Hashable],
        variable_dim: str = "variable",
        name: Hashable = None,
    ) -> "DataArray":
        """Combine variables of differing dimensionality into a DataArray
        without broadcasting.

        This method is similar to Dataset.to_array but does not broadcast the
        variables.

        Parameters
        ----------
        new_dim : hashable
            Name of the new stacked coordinate
        sample_dims : sequence of hashable
            Dimensions that **will not** be stacked. Each array in the dataset
            must share these dimensions. For machine learning applications,
            these define the dimensions over which samples are drawn.
        variable_dim : str, optional
            Name of the level in the stacked coordinate which corresponds to
            the variables.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        stacked : DataArray
            DataArray with the specified dimensions and data variables
            stacked together. The stacked coordinate is named ``new_dim``
            and represented by a MultiIndex object with a level containing the
            data variable names. The name of this level is controlled using
            the ``variable_dim`` argument.

        See Also
        --------
        Dataset.to_array
        Dataset.stack
        DataArray.to_unstacked_dataset

        Examples
        --------
        >>> data = xr.Dataset(
        ...     data_vars={
        ...         "a": (("x", "y"), [[0, 1, 2], [3, 4, 5]]),
        ...         "b": ("x", [6, 7]),
        ...     },
        ...     coords={"y": ["u", "v", "w"]},
        ... )

        >>> data
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 'u' 'v' 'w'
        Dimensions without coordinates: x
        Data variables:
            a        (x, y) int64 0 1 2 3 4 5
            b        (x) int64 6 7

        >>> data.to_stacked_array("z", sample_dims=["x"])
        <xarray.DataArray 'a' (x: 2, z: 4)>
        array([[0, 1, 2, 6],
               [3, 4, 5, 7]])
        Coordinates:
          * z         (z) MultiIndex
          - variable  (z) object 'a' 'a' 'a' 'b'
          - y         (z) object 'u' 'v' 'w' nan
        Dimensions without coordinates: x

        """
        stacking_dims = tuple(dim for dim in self.dims if dim not in sample_dims)

        for variable in self:
            dims = self[variable].dims
            dims_include_sample_dims = set(sample_dims) <= set(dims)
            if not dims_include_sample_dims:
                raise ValueError(
                    "All variables in the dataset must contain the "
                    "dimensions {}.".format(dims)
                )

        def ensure_stackable(val):
            assign_coords = {variable_dim: val.name}
            for dim in stacking_dims:
                if dim not in val.dims:
                    assign_coords[dim] = None

            expand_dims = set(stacking_dims).difference(set(val.dims))
            expand_dims.add(variable_dim)
            # must be list for .expand_dims
            expand_dims = list(expand_dims)

            return (
                val.assign_coords(**assign_coords)
                .expand_dims(expand_dims)
                .stack({new_dim: (variable_dim,) + stacking_dims})
            )

        # concatenate the arrays
        stackable_vars = [ensure_stackable(self[key]) for key in self.data_vars]
        data_array = xr.concat(stackable_vars, dim=new_dim)

        # coerce the levels of the MultiIndex to have the same type as the
        # input dimensions. This code is messy, so it might be better to just
        # input a dummy value for the singleton dimension.
        idx = data_array.indexes[new_dim]
        levels = [idx.levels[0]] + [
            level.astype(self[level.name].dtype) for level in idx.levels[1:]
        ]
        new_idx = idx.set_levels(levels)
        data_array[new_dim] = IndexVariable(new_dim, new_idx)

        if name is not None:
            data_array.name = name

        return data_array

    def _unstack_once(self, dim: Hashable, fill_value, sparse) -> "Dataset":
        index = self.get_index(dim)
        index = remove_unused_levels_categories(index)
        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)

        # take a shortcut in case the MultiIndex was not modified.
        if index.equals(full_idx):
            obj = self
        else:
            obj = self._reindex(
                {dim: full_idx}, copy=False, fill_value=fill_value, sparse=sparse
            )

        new_dim_names = index.names
        new_dim_sizes = [lev.size for lev in index.levels]

        variables: Dict[Hashable, Variable] = {}
        indexes = {k: v for k, v in self.indexes.items() if k != dim}

        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = dict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack({dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            variables[name] = IndexVariable(name, lev)
            indexes[name] = lev

        coord_names = set(self._coord_names) - {dim} | set(new_dim_names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def unstack(
        self,
        dim: Union[Hashable, Iterable[Hashable]] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> "Dataset":
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value : scalar or dict-like, default: nan
            value to be filled. If a dict-like, maps variable names to
            fill values. If not provided or if the dict-like does not
            contain all variables, the dtype's NA value will be used.
        sparse : bool, default: False
            use sparse-array if True

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See also
        --------
        Dataset.stack
        """
        if dim is None:
            dims = [
                d for d in self.dims if isinstance(self.get_index(d), pd.MultiIndex)
            ]
        else:
            if isinstance(dim, str) or not isinstance(dim, Iterable):
                dims = [dim]
            else:
                dims = list(dim)

            missing_dims = [d for d in dims if d not in self.dims]
            if missing_dims:
                raise ValueError(
                    "Dataset does not contain the dimensions: %s" % missing_dims
                )

            non_multi_dims = [
                d for d in dims if not isinstance(self.get_index(d), pd.MultiIndex)
            ]
            if non_multi_dims:
                raise ValueError(
                    "cannot unstack dimensions that do not "
                    "have a MultiIndex: %s" % non_multi_dims
                )

        result = self.copy(deep=False)
        for dim in dims:
            result = result._unstack_once(dim, fill_value, sparse)
        return result

    def update(self, other: "CoercibleMapping") -> "Dataset":
        """Update this dataset's variables with those from another dataset.

        Parameters
        ----------
        other : Dataset or mapping
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}


        Returns
        -------
        updated : Dataset
            Updated dataset.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.
        """
        merge_result = dataset_update_method(self, other)
        return self._replace(inplace=True, **merge_result._asdict())

    def merge(
        self,
        other: Union["CoercibleMapping", "DataArray"],
        overwrite_vars: Union[Hashable, Iterable[Hashable]] = frozenset(),
        compat: str = "no_conflicts",
        join: str = "outer",
        fill_value: Any = dtypes.NA,
    ) -> "Dataset":
        """Merge the arrays of two datasets into a single dataset.

        This method generally does not allow for overriding data, with the
        exception of attributes, which are ignored on the second dataset.
        Variables with the same name are checked for conflicts via the equals
        or identical methods.

        Parameters
        ----------
        other : Dataset or mapping
            Dataset or variables to merge with this dataset.
        overwrite_vars : hashable or iterable of hashable, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {"broadcast_equals", "equals", "identical", \
                  "no_conflicts"}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
            - 'no_conflicts': only values which are not null in both datasets
              must be equal. The returned dataset then contains the combination
              of all non-null values.

        join : {"outer", "inner", "left", "right", "exact"}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).
        """
        other = other.to_dataset() if isinstance(other, xr.DataArray) else other
        merge_result = dataset_merge_method(
            self,
            other,
            overwrite_vars=overwrite_vars,
            compat=compat,
            join=join,
            fill_value=fill_value,
        )
        return self._replace(**merge_result._asdict())

    def _assert_all_in_dataset(
        self, names: Iterable[Hashable], virtual_okay: bool = False
    ) -> None:
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError(
                "One or more of the specified variables "
                "cannot be found in this dataset"
            )

    def drop_vars(
        self, names: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
    ) -> "Dataset":
        """Drop variables from this dataset.

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables to drop.
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            dataset are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset

        """
        # the Iterable check is required for mypy
        if is_scalar(names) or not isinstance(names, Iterable):
            names = {names}
        else:
            names = set(names)
        if errors == "raise":
            self._assert_all_in_dataset(names)

        variables = {k: v for k, v in self._variables.items() if k not in names}
        coord_names = {k for k in self._coord_names if k in variables}
        indexes = {k: v for k, v in self.indexes.items() if k not in names}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def drop(self, labels=None, dim=None, *, errors="raise", **labels_kwargs):
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        Dataset.drop_vars
        Dataset.drop_sel
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if is_dict_like(labels) and not isinstance(labels, dict):
            warnings.warn(
                "dropping coordinates using `drop` is be deprecated; use drop_vars.",
                FutureWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)

        if labels_kwargs or isinstance(labels, dict):
            if dim is not None:
                raise ValueError("cannot specify dim and dict-like arguments.")
            labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        if dim is None and (is_scalar(labels) or isinstance(labels, Iterable)):
            warnings.warn(
                "dropping variables using `drop` will be deprecated; using drop_vars is encouraged.",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)
        if dim is not None:
            warnings.warn(
                "dropping labels using list-like labels is deprecated; using "
                "dict-like arguments with `drop_sel`, e.g. `ds.drop_sel(dim=[labels]).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.drop_sel({dim: labels}, errors=errors, **labels_kwargs)

        warnings.warn(
            "dropping labels using `drop` will be deprecated; using drop_sel is encouraged.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.drop_sel(labels, errors=errors)

    def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
        """Drop index labels from this dataset.

        Parameters
        ----------
        labels : mapping of hashable to Any
            Index labels to drop
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if
            any of the index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.
        **labels_kwargs : {dim: label, ...}, optional
            The keyword arguments form of ``dim`` and ``labels``

        Returns
        -------
        dropped : Dataset

        Examples
        --------
        >>> data = np.random.randn(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds.drop_sel(y=["a", "c"])
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) float64 0.4002 1.868
        >>> ds.drop_sel(y="b")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) float64 1.764 0.9787 2.241 -0.9773
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        ds = self
        for dim, labels_for_dim in labels.items():
            # Don't cast to set, as it would harm performance when labels
            # is a large numpy array
            if utils.is_scalar(labels_for_dim):
                labels_for_dim = [labels_for_dim]
            labels_for_dim = np.asarray(labels_for_dim)
            try:
                index = self.indexes[dim]
            except KeyError:
                raise ValueError("dimension %r does not have coordinate labels" % dim)
            new_index = index.drop(labels_for_dim, errors=errors)
            ds = ds.loc[{dim: new_index}]
        return ds

    def drop_dims(
        self, drop_dims: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
    ) -> "Dataset":
        """Drop dimensions and associated variables from this dataset.

        Parameters
        ----------
        drop_dims : hashable or iterable of hashable
            Dimension or dimensions to drop.
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if any of the
            dimensions passed are not in the dataset. If 'ignore', any given
            labels that are in the dataset are dropped and no error is raised.

        Returns
        -------
        obj : Dataset
            The dataset without the given dimensions (or any variables
            containing those dimensions)
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if
            any of the dimensions passed are not
            in the dataset. If 'ignore', any given dimensions that are in the
            dataset are dropped and no error is raised.
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if isinstance(drop_dims, str) or not isinstance(drop_dims, Iterable):
            drop_dims = {drop_dims}
        else:
            drop_dims = set(drop_dims)

        if errors == "raise":
            missing_dims = drop_dims - set(self.dims)
            if missing_dims:
                raise ValueError(
                    "Dataset does not contain the dimensions: %s" % missing_dims
                )

        drop_vars = {k for k, v in self._variables.items() if set(v.dims) & drop_dims}
        return self.drop_vars(drop_vars)

    def transpose(self, *dims: Hashable) -> "Dataset":
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dims : hashable, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.

        Returns
        -------
        transposed : Dataset
            Each array in the dataset (including) coordinates will be
            transposed to the given order.

        Notes
        -----
        This operation returns a view of each array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded into memory.

        See Also
        --------
        numpy.transpose
        DataArray.transpose
        """
        if dims:
            if set(dims) ^ set(self.dims) and ... not in dims:
                raise ValueError(
                    "arguments to transpose (%s) must be "
                    "permuted dataset dimensions (%s)" % (dims, tuple(self.dims))
                )
        ds = self.copy()
        for name, var in self._variables.items():
            var_dims = tuple(dim for dim in dims if dim in (var.dims + (...,)))
            ds._variables[name] = var.transpose(*var_dims)
        return ds
```
### 35 - xarray/core/dataarray.py:

Start line: 2173, End line: 2197

```python
class DataArray(AbstractArray, DataWithCoords):

    @property
    def T(self) -> "DataArray":
        return self.transpose()

    def drop_vars(
        self, names: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
    ) -> "DataArray":
        """Returns an array with dropped variables.

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables to drop.
        errors: {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            DataArray are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset
            New Dataset copied from `self` with variables removed.
        """
        ds = self._to_temp_dataset().drop_vars(names, errors=errors)
        return self._from_temp_dataset(ds)
```
### 42 - xarray/core/dataarray.py:

Start line: 1, End line: 87

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
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
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
### 46 - xarray/core/dataarray.py:

Start line: 4052, End line: 4153

```python
class DataArray(AbstractArray, DataWithCoords):

    def argmin(
        self,
        dim: Union[Hashable, Sequence[Hashable]] = None,
        axis: int = None,
        keep_attrs: bool = None,
        skipna: bool = None,
    ) -> Union["DataArray", Dict[Hashable, "DataArray"]]:
        """Index or indices of the minimum of the DataArray over one or more dimensions.

        If a sequence is passed to 'dim', then result returned as dict of DataArrays,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a DataArray with dtype int.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : hashable, sequence of hashable or ..., optional
            The dimensions over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : DataArray or dict of DataArray

        See also
        --------
        Variable.argmin, DataArray.idxmin

        Examples
        --------
        >>> array = xr.DataArray([0, 2, -1, 3], dims="x")
        >>> array.min()
        <xarray.DataArray ()>
        array(-1)
        >>> array.argmin()
        <xarray.DataArray ()>
        array(2)
        >>> array.argmin(...)
        {'x': <xarray.DataArray ()>
        array(2)}
        >>> array.isel(array.argmin(...))
        <xarray.DataArray ()>
        array(-1)

        >>> array = xr.DataArray(
        ...     [[[3, 2, 1], [3, 1, 2], [2, 1, 3]], [[1, 3, 2], [2, -5, 1], [2, 3, 1]]],
        ...     dims=("x", "y", "z"),
        ... )
        >>> array.min(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[ 1,  2,  1],
               [ 2, -5,  1],
               [ 2,  1,  1]])
        Dimensions without coordinates: y, z
        >>> array.argmin(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[1, 0, 0],
               [1, 1, 1],
               [0, 0, 1]])
        Dimensions without coordinates: y, z
        >>> array.argmin(dim=["x"])
        {'x': <xarray.DataArray (y: 3, z: 3)>
        array([[1, 0, 0],
               [1, 1, 1],
               [0, 0, 1]])
        Dimensions without coordinates: y, z}
        >>> array.min(dim=("x", "z"))
        <xarray.DataArray (y: 3)>
        array([ 1, -5,  1])
        Dimensions without coordinates: y
        >>> array.argmin(dim=["x", "z"])
        {'x': <xarray.DataArray (y: 3)>
        array([0, 1, 0])
        Dimensions without coordinates: y, 'z': <xarray.DataArray (y: 3)>
        array([2, 1, 1])
        Dimensions without coordinates: y}
        >>> array.isel(array.argmin(dim=["x", "z"]))
        <xarray.DataArray (y: 3)>
        array([ 1, -5,  1])
        Dimensions without coordinates: y
        """
        result = self.variable.argmin(dim, axis, keep_attrs, skipna)
        if isinstance(result, dict):
            return {k: self._replace_maybe_drop_dims(v) for k, v in result.items()}
        else:
            return self._replace_maybe_drop_dims(result)
```
### 48 - xarray/core/dataset.py:

Start line: 298, End line: 353

```python
def split_indexes(
    dims_or_levels: Union[Hashable, Sequence[Hashable]],
    variables: Mapping[Hashable, Variable],
    coord_names: Set[Hashable],
    level_coords: Mapping[Hashable, Hashable],
    drop: bool = False,
) -> Tuple[Dict[Hashable, Variable], Set[Hashable]]:
    """Extract (multi-)indexes (levels) as variables.

    Not public API. Used in Dataset and DataArray reset_index
    methods.
    """
    if isinstance(dims_or_levels, str) or not isinstance(dims_or_levels, Sequence):
        dims_or_levels = [dims_or_levels]

    dim_levels: DefaultDict[Any, List[Hashable]] = defaultdict(list)
    dims = []
    for k in dims_or_levels:
        if k in level_coords:
            dim_levels[level_coords[k]].append(k)
        else:
            dims.append(k)

    vars_to_replace = {}
    vars_to_create: Dict[Hashable, Variable] = {}
    vars_to_remove = []

    for d in dims:
        index = variables[d].to_index()
        if isinstance(index, pd.MultiIndex):
            dim_levels[d] = index.names
        else:
            vars_to_remove.append(d)
            if not drop:
                vars_to_create[str(d) + "_"] = Variable(d, index, variables[d].attrs)

    for d, levs in dim_levels.items():
        index = variables[d].to_index()
        if len(levs) == index.nlevels:
            vars_to_remove.append(d)
        else:
            vars_to_replace[d] = IndexVariable(d, index.droplevel(levs))

        if not drop:
            for lev in levs:
                idx = index.get_level_values(lev)
                vars_to_create[idx.name] = Variable(d, idx, variables[d].attrs)

    new_variables = dict(variables)
    for v in set(vars_to_remove):
        del new_variables[v]
    new_variables.update(vars_to_replace)
    new_variables.update(vars_to_create)
    new_coord_names = (coord_names | set(vars_to_create)) - set(vars_to_remove)

    return new_variables, new_coord_names
```
