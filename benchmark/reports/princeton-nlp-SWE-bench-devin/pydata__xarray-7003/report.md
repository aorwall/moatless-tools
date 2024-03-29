# pydata__xarray-7003

| **pydata/xarray** | `5bec4662a7dd4330eca6412c477ca3f238323ed2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2765 |
| **Any found context length** | 984 |
| **Avg pos** | 11.0 |
| **Min pos** | 2 |
| **Max pos** | 9 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1092,12 +1092,13 @@ def get_unique(self) -> list[T_PandasOrXarrayIndex]:
         """Return a list of unique indexes, preserving order."""
 
         unique_indexes: list[T_PandasOrXarrayIndex] = []
-        seen: set[T_PandasOrXarrayIndex] = set()
+        seen: set[int] = set()
 
         for index in self._indexes.values():
-            if index not in seen:
+            index_id = id(index)
+            if index_id not in seen:
                 unique_indexes.append(index)
-                seen.add(index)
+                seen.add(index_id)
 
         return unique_indexes
 
@@ -1201,9 +1202,24 @@ def copy_indexes(
         """
         new_indexes = {}
         new_index_vars = {}
+
         for idx, coords in self.group_by_index():
+            if isinstance(idx, pd.Index):
+                convert_new_idx = True
+                dim = next(iter(coords.values())).dims[0]
+                if isinstance(idx, pd.MultiIndex):
+                    idx = PandasMultiIndex(idx, dim)
+                else:
+                    idx = PandasIndex(idx, dim)
+            else:
+                convert_new_idx = False
+
             new_idx = idx.copy(deep=deep)
             idx_vars = idx.create_variables(coords)
+
+            if convert_new_idx:
+                new_idx = cast(PandasIndex, new_idx).index
+
             new_indexes.update({k: new_idx for k in coords})
             new_index_vars.update(idx_vars)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/indexes.py | 1095 | 1100 | 2 | 1 | 984
| xarray/core/indexes.py | 1204 | 1204 | 9 | 1 | 2765


## Problem Statement

```
Indexes.get_unique() TypeError with pandas indexes
@benbovy I also just tested the `get_unique()` method that you mentioned and maybe noticed a related issue here, which I'm not sure is wanted / expected.

Taking the above dataset `ds`, accessing this function results in an error:

\`\`\`python
> ds.indexes.get_unique()

TypeError: unhashable type: 'MultiIndex'
\`\`\`

However, for `xindexes` it works:
\`\`\`python
> ds.xindexes.get_unique()

[<xarray.core.indexes.PandasMultiIndex at 0x7f105bf1df20>]
\`\`\`

_Originally posted by @lukasbindreiter in https://github.com/pydata/xarray/issues/6752#issuecomment-1236717180_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/indexes.py** | 1164 | 1193| 242 | 242 | 10615 | 
| **-> 2 <-** | **1 xarray/core/indexes.py** | 1008 | 1108| 742 | 984 | 10615 | 
| 3 | 2 xarray/core/indexing.py | 120 | 152| 304 | 1288 | 23227 | 
| 4 | **2 xarray/core/indexes.py** | 877 | 916| 314 | 1602 | 23227 | 
| 5 | **2 xarray/core/indexes.py** | 343 | 362| 156 | 1758 | 23227 | 
| 6 | **2 xarray/core/indexes.py** | 430 | 484| 454 | 2212 | 23227 | 
| 7 | **2 xarray/core/indexes.py** | 1 | 31| 155 | 2367 | 23227 | 
| 8 | **2 xarray/core/indexes.py** | 918 | 934| 147 | 2514 | 23227 | 
| **-> 9 <-** | **2 xarray/core/indexes.py** | 1195 | 1225| 251 | 2765 | 23227 | 
| 10 | **2 xarray/core/indexes.py** | 249 | 282| 287 | 3052 | 23227 | 
| 11 | **2 xarray/core/indexes.py** | 936 | 950| 157 | 3209 | 23227 | 
| 12 | **2 xarray/core/indexes.py** | 586 | 604| 156 | 3365 | 23227 | 
| 13 | 3 xarray/core/alignment.py | 287 | 326| 416 | 3781 | 31739 | 
| 14 | **3 xarray/core/indexes.py** | 771 | 875| 898 | 4679 | 31739 | 
| 15 | 4 asv_bench/benchmarks/indexing.py | 133 | 150| 145 | 4824 | 33240 | 
| 16 | **4 xarray/core/indexes.py** | 572 | 584| 122 | 4946 | 33240 | 
| 17 | **4 xarray/core/indexes.py** | 364 | 428| 525 | 5471 | 33240 | 
| 18 | 5 conftest.py | 1 | 42| 278 | 5749 | 33518 | 
| 19 | **5 xarray/core/indexes.py** | 284 | 304| 178 | 5927 | 33518 | 
| 20 | **5 xarray/core/indexes.py** | 735 | 769| 204 | 6131 | 33518 | 
| 21 | **5 xarray/core/indexes.py** | 1329 | 1350| 222 | 6353 | 33518 | 
| 22 | **5 xarray/core/indexes.py** | 34 | 124| 662 | 7015 | 33518 | 
| 23 | 5 asv_bench/benchmarks/indexing.py | 113 | 130| 147 | 7162 | 33518 | 
| 24 | **5 xarray/core/indexes.py** | 1353 | 1364| 103 | 7265 | 33518 | 
| 25 | 5 asv_bench/benchmarks/indexing.py | 79 | 90| 118 | 7383 | 33518 | 
| 26 | **5 xarray/core/indexes.py** | 202 | 214| 127 | 7510 | 33518 | 
| 27 | 6 xarray/core/groupby.py | 249 | 279| 194 | 7704 | 44448 | 
| 28 | 6 xarray/core/groupby.py | 67 | 95| 236 | 7940 | 44448 | 
| 29 | **6 xarray/core/indexes.py** | 306 | 341| 263 | 8203 | 44448 | 
| 30 | **6 xarray/core/indexes.py** | 1296 | 1326| 216 | 8419 | 44448 | 
| 31 | **6 xarray/core/indexes.py** | 1257 | 1293| 210 | 8629 | 44448 | 
| 32 | 6 xarray/core/indexing.py | 1463 | 1486| 192 | 8821 | 44448 | 
| 33 | 7 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 8988 | 44615 | 
| 34 | **7 xarray/core/indexes.py** | 510 | 534| 264 | 9252 | 44615 | 
| 35 | 7 xarray/core/indexing.py | 1 | 33| 201 | 9453 | 44615 | 
| 36 | **7 xarray/core/indexes.py** | 217 | 247| 236 | 9689 | 44615 | 
| 37 | **7 xarray/core/indexes.py** | 127 | 163| 269 | 9958 | 44615 | 
| 38 | **7 xarray/core/indexes.py** | 705 | 733| 253 | 10211 | 44615 | 
| 39 | 7 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 10949 | 44615 | 
| 40 | 7 xarray/core/indexing.py | 1506 | 1584| 599 | 11548 | 44615 | 
| 41 | 8 xarray/testing.py | 256 | 304| 484 | 12032 | 48117 | 
| 42 | 8 xarray/core/indexing.py | 1443 | 1461| 225 | 12257 | 48117 | 
| 43 | **8 xarray/core/indexes.py** | 537 | 570| 281 | 12538 | 48117 | 
| 44 | **8 xarray/core/indexes.py** | 1141 | 1162| 164 | 12702 | 48117 | 
| 45 | 8 xarray/testing.py | 345 | 381| 430 | 13132 | 48117 | 
| 46 | **8 xarray/core/indexes.py** | 1391 | 1409| 189 | 13321 | 48117 | 
| 47 | **8 xarray/core/indexes.py** | 1110 | 1139| 242 | 13563 | 48117 | 
| 48 | **8 xarray/core/indexes.py** | 487 | 507| 217 | 13780 | 48117 | 
| 49 | 9 xarray/core/dataset.py | 406 | 8884| 196 | 13976 | 124433 | 
| 50 | 9 xarray/core/indexing.py | 1383 | 1409| 258 | 14234 | 124433 | 
| 51 | 10 xarray/core/utils.py | 115 | 140| 232 | 14466 | 131087 | 
| 52 | 10 xarray/core/groupby.py | 539 | 551| 133 | 14599 | 131087 | 
| 53 | 10 asv_bench/benchmarks/indexing.py | 93 | 110| 201 | 14800 | 131087 | 
| 54 | 11 xarray/coding/cftimeindex.py | 487 | 513| 226 | 15026 | 138042 | 
| 55 | 12 xarray/core/missing.py | 1 | 25| 184 | 15210 | 144289 | 
| 56 | 13 xarray/core/types.py | 1 | 79| 771 | 15981 | 145274 | 
| 57 | 14 xarray/core/nputils.py | 96 | 115| 212 | 16193 | 147143 | 
| 58 | 14 xarray/core/utils.py | 77 | 112| 235 | 16428 | 147143 | 
| 59 | 15 xarray/coding/frequencies.py | 224 | 241| 120 | 16548 | 149317 | 
| 60 | 15 xarray/core/indexing.py | 1488 | 1503| 205 | 16753 | 149317 | 
| 61 | 16 xarray/core/common.py | 453 | 606| 192 | 16945 | 164223 | 
| 62 | 16 xarray/core/indexing.py | 512 | 549| 335 | 17280 | 164223 | 
| 63 | 16 xarray/core/nputils.py | 1 | 37| 264 | 17544 | 164223 | 
| 64 | 16 xarray/core/indexing.py | 155 | 197| 342 | 17886 | 164223 | 
| 65 | 16 xarray/core/alignment.py | 328 | 340| 160 | 18046 | 164223 | 
| 66 | 17 asv_bench/benchmarks/reindexing.py | 1 | 53| 415 | 18461 | 164638 | 
| 67 | 18 asv_bench/benchmarks/groupby.py | 23 | 47| 266 | 18727 | 165971 | 
| 68 | **18 xarray/core/indexes.py** | 606 | 648| 401 | 19128 | 165971 | 
| 69 | 18 xarray/core/groupby.py | 1 | 64| 387 | 19515 | 165971 | 
| 70 | 18 asv_bench/benchmarks/indexing.py | 62 | 76| 151 | 19666 | 165971 | 
| 71 | 18 xarray/core/indexing.py | 666 | 685| 157 | 19823 | 165971 | 
| 72 | 18 xarray/coding/cftimeindex.py | 561 | 594| 294 | 20117 | 165971 | 
| 73 | 18 asv_bench/benchmarks/groupby.py | 63 | 78| 120 | 20237 | 165971 | 
| 74 | 18 asv_bench/benchmarks/groupby.py | 1 | 21| 174 | 20411 | 165971 | 
| 75 | 19 xarray/core/variable.py | 2894 | 2910| 162 | 20573 | 190478 | 
| 76 | 19 asv_bench/benchmarks/groupby.py | 50 | 60| 167 | 20740 | 190478 | 
| 77 | 20 xarray/coding/times.py | 336 | 355| 251 | 20991 | 196532 | 
| 78 | 20 xarray/coding/cftimeindex.py | 428 | 452| 272 | 21263 | 196532 | 
| 79 | 20 xarray/core/indexing.py | 498 | 510| 142 | 21405 | 196532 | 
| 80 | 20 xarray/coding/cftimeindex.py | 281 | 323| 395 | 21800 | 196532 | 
| 81 | 20 xarray/core/groupby.py | 223 | 246| 276 | 22076 | 196532 | 
| 82 | 20 xarray/core/indexing.py | 1214 | 1224| 112 | 22188 | 196532 | 
| 83 | **20 xarray/core/indexes.py** | 1367 | 1388| 159 | 22347 | 196532 | 
| 84 | **20 xarray/core/indexes.py** | 166 | 187| 145 | 22492 | 196532 | 


## Patch

```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1092,12 +1092,13 @@ def get_unique(self) -> list[T_PandasOrXarrayIndex]:
         """Return a list of unique indexes, preserving order."""
 
         unique_indexes: list[T_PandasOrXarrayIndex] = []
-        seen: set[T_PandasOrXarrayIndex] = set()
+        seen: set[int] = set()
 
         for index in self._indexes.values():
-            if index not in seen:
+            index_id = id(index)
+            if index_id not in seen:
                 unique_indexes.append(index)
-                seen.add(index)
+                seen.add(index_id)
 
         return unique_indexes
 
@@ -1201,9 +1202,24 @@ def copy_indexes(
         """
         new_indexes = {}
         new_index_vars = {}
+
         for idx, coords in self.group_by_index():
+            if isinstance(idx, pd.Index):
+                convert_new_idx = True
+                dim = next(iter(coords.values())).dims[0]
+                if isinstance(idx, pd.MultiIndex):
+                    idx = PandasMultiIndex(idx, dim)
+                else:
+                    idx = PandasIndex(idx, dim)
+            else:
+                convert_new_idx = False
+
             new_idx = idx.copy(deep=deep)
             idx_vars = idx.create_variables(coords)
+
+            if convert_new_idx:
+                new_idx = cast(PandasIndex, new_idx).index
+
             new_indexes.update({k: new_idx for k in coords})
             new_index_vars.update(idx_vars)
 

```

## Test Patch

```diff
diff --git a/xarray/tests/test_indexes.py b/xarray/tests/test_indexes.py
--- a/xarray/tests/test_indexes.py
+++ b/xarray/tests/test_indexes.py
@@ -9,6 +9,7 @@
 
 import xarray as xr
 from xarray.core.indexes import (
+    Hashable,
     Index,
     Indexes,
     PandasIndex,
@@ -535,7 +536,7 @@ def test_copy(self) -> None:
 
 class TestIndexes:
     @pytest.fixture
-    def unique_indexes(self) -> list[PandasIndex]:
+    def indexes_and_vars(self) -> tuple[list[PandasIndex], dict[Hashable, Variable]]:
         x_idx = PandasIndex(pd.Index([1, 2, 3], name="x"), "x")
         y_idx = PandasIndex(pd.Index([4, 5, 6], name="y"), "y")
         z_pd_midx = pd.MultiIndex.from_product(
@@ -543,10 +544,29 @@ def unique_indexes(self) -> list[PandasIndex]:
         )
         z_midx = PandasMultiIndex(z_pd_midx, "z")
 
-        return [x_idx, y_idx, z_midx]
+        indexes = [x_idx, y_idx, z_midx]
+
+        variables = {}
+        for idx in indexes:
+            variables.update(idx.create_variables())
+
+        return indexes, variables
+
+    @pytest.fixture(params=["pd_index", "xr_index"])
+    def unique_indexes(
+        self, request, indexes_and_vars
+    ) -> list[PandasIndex] | list[pd.Index]:
+        xr_indexes, _ = indexes_and_vars
+
+        if request.param == "pd_index":
+            return [idx.index for idx in xr_indexes]
+        else:
+            return xr_indexes
 
     @pytest.fixture
-    def indexes(self, unique_indexes) -> Indexes[Index]:
+    def indexes(
+        self, unique_indexes, indexes_and_vars
+    ) -> Indexes[Index] | Indexes[pd.Index]:
         x_idx, y_idx, z_midx = unique_indexes
         indexes: dict[Any, Index] = {
             "x": x_idx,
@@ -555,9 +575,8 @@ def indexes(self, unique_indexes) -> Indexes[Index]:
             "one": z_midx,
             "two": z_midx,
         }
-        variables: dict[Any, Variable] = {}
-        for idx in unique_indexes:
-            variables.update(idx.create_variables())
+
+        _, variables = indexes_and_vars
 
         return Indexes(indexes, variables)
 

```


## Code snippets

### 1 - xarray/core/indexes.py:

Start line: 1164, End line: 1193

```python
class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):

    def group_by_index(
        self,
    ) -> list[tuple[T_PandasOrXarrayIndex, dict[Hashable, Variable]]]:
        """Returns a list of unique indexes and their corresponding coordinates."""

        index_coords = []

        for i in self._id_index:
            index = self._id_index[i]
            coords = {k: self._variables[k] for k in self._id_coord_names[i]}
            index_coords.append((index, coords))

        return index_coords

    def to_pandas_indexes(self) -> Indexes[pd.Index]:
        """Returns an immutable proxy for Dataset or DataArrary pandas indexes.

        Raises an error if this proxy contains indexes that cannot be coerced to
        pandas.Index objects.

        """
        indexes: dict[Hashable, pd.Index] = {}

        for k, idx in self._indexes.items():
            if isinstance(idx, pd.Index):
                indexes[k] = idx
            elif isinstance(idx, Index):
                indexes[k] = idx.to_pandas_index()

        return Indexes(indexes, self._variables)
```
### 2 - xarray/core/indexes.py:

Start line: 1008, End line: 1108

```python
class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
    """Immutable proxy for Dataset or DataArrary indexes.

    Keys are coordinate names and values may correspond to either pandas or
    xarray indexes.

    Also provides some utility methods.

    """

    _indexes: dict[Any, T_PandasOrXarrayIndex]
    _variables: dict[Any, Variable]

    __slots__ = (
        "_indexes",
        "_variables",
        "_dims",
        "__coord_name_id",
        "__id_index",
        "__id_coord_names",
    )

    def __init__(
        self,
        indexes: dict[Any, T_PandasOrXarrayIndex],
        variables: dict[Any, Variable],
    ):
        """Constructor not for public consumption.

        Parameters
        ----------
        indexes : dict
            Indexes held by this object.
        variables : dict
            Indexed coordinate variables in this object.

        """
        self._indexes = indexes
        self._variables = variables

        self._dims: Mapping[Hashable, int] | None = None
        self.__coord_name_id: dict[Any, int] | None = None
        self.__id_index: dict[int, T_PandasOrXarrayIndex] | None = None
        self.__id_coord_names: dict[int, tuple[Hashable, ...]] | None = None

    @property
    def _coord_name_id(self) -> dict[Any, int]:
        if self.__coord_name_id is None:
            self.__coord_name_id = {k: id(idx) for k, idx in self._indexes.items()}
        return self.__coord_name_id

    @property
    def _id_index(self) -> dict[int, T_PandasOrXarrayIndex]:
        if self.__id_index is None:
            self.__id_index = {id(idx): idx for idx in self.get_unique()}
        return self.__id_index

    @property
    def _id_coord_names(self) -> dict[int, tuple[Hashable, ...]]:
        if self.__id_coord_names is None:
            id_coord_names: Mapping[int, list[Hashable]] = defaultdict(list)
            for k, v in self._coord_name_id.items():
                id_coord_names[v].append(k)
            self.__id_coord_names = {k: tuple(v) for k, v in id_coord_names.items()}

        return self.__id_coord_names

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return Frozen(self._variables)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        from .variable import calculate_dimensions

        if self._dims is None:
            self._dims = calculate_dimensions(self._variables)

        return Frozen(self._dims)

    def copy(self):
        return type(self)(dict(self._indexes), dict(self._variables))

    def get_unique(self) -> list[T_PandasOrXarrayIndex]:
        """Return a list of unique indexes, preserving order."""

        unique_indexes: list[T_PandasOrXarrayIndex] = []
        seen: set[T_PandasOrXarrayIndex] = set()

        for index in self._indexes.values():
            if index not in seen:
                unique_indexes.append(index)
                seen.add(index)

        return unique_indexes

    def is_multi(self, key: Hashable) -> bool:
        """Return True if ``key`` maps to a multi-coordinate index,
        False otherwise.
        """
        return len(self._id_coord_names[self._coord_name_id[key]]) > 1
```
### 3 - xarray/core/indexing.py:

Start line: 120, End line: 152

```python
def group_indexers_by_index(
    obj: T_Xarray,
    indexers: Mapping[Any, Any],
    options: Mapping[str, Any],
) -> list[tuple[Index, dict[Any, Any]]]:
    """Returns a list of unique indexes and their corresponding indexers."""
    unique_indexes = {}
    grouped_indexers: Mapping[int | None, dict] = defaultdict(dict)

    for key, label in indexers.items():
        index: Index = obj.xindexes.get(key, None)

        if index is not None:
            index_id = id(index)
            unique_indexes[index_id] = index
            grouped_indexers[index_id][key] = label
        elif key in obj.coords:
            raise KeyError(f"no index found for coordinate {key!r}")
        elif key not in obj.dims:
            raise KeyError(f"{key!r} is not a valid dimension or coordinate")
        elif len(options):
            raise ValueError(
                f"cannot supply selection options {options!r} for dimension {key!r}"
                "that has no associated coordinate or index"
            )
        else:
            # key is a dimension without a "dimension-coordinate"
            # failback to location-based selection
            # TODO: depreciate this implicit behavior and suggest using isel instead?
            unique_indexes[None] = None
            grouped_indexers[None][key] = label

    return [(unique_indexes[k], grouped_indexers[k]) for k in unique_indexes]
```
### 4 - xarray/core/indexes.py:

Start line: 877, End line: 916

```python
class PandasMultiIndex(PandasIndex):

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        # ... other code

        if new_index is not None:
            if isinstance(new_index, pd.MultiIndex):
                level_coords_dtype = {
                    k: self.level_coords_dtype[k] for k in new_index.names
                }
                new_index = self._replace(
                    new_index, level_coords_dtype=level_coords_dtype
                )
                dims_dict = {}
                drop_coords = []
            else:
                new_index = PandasIndex(
                    new_index,
                    new_index.name,
                    coord_dtype=self.level_coords_dtype[new_index.name],
                )
                dims_dict = {self.dim: new_index.index.name}
                drop_coords = [self.dim]

            # variable(s) attrs and encoding metadata are propagated
            # when replacing the indexes in the resulting xarray object
            new_vars = new_index.create_variables()
            indexes = cast(Dict[Any, Index], {k: new_index for k in new_vars})

            # add scalar variable for each dropped level
            variables = new_vars
            for name, val in scalar_coord_values.items():
                variables[name] = Variable([], val)

            return IndexSelResult(
                {self.dim: indexer},
                indexes=indexes,
                variables=variables,
                drop_indexes=list(scalar_coord_values),
                drop_coords=drop_coords,
                rename_dims=dims_dict,
            )

        else:
            return IndexSelResult({self.dim: indexer})
```
### 5 - xarray/core/indexes.py:

Start line: 343, End line: 362

```python
class PandasIndex(Index):

    def to_pandas_index(self) -> pd.Index:
        return self.index

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> PandasIndex | None:
        from .variable import Variable

        indxr = indexers[self.dim]
        if isinstance(indxr, Variable):
            if indxr.dims != (self.dim,):
                # can't preserve a index if result has new dimensions
                return None
            else:
                indxr = indxr.data
        if not isinstance(indxr, slice) and is_scalar(indxr):
            # scalar indexer: drop index
            return None

        return self._replace(self.index[indxr])
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
### 7 - xarray/core/indexes.py:

Start line: 1, End line: 31

```python
from __future__ import annotations

import collections.abc
import copy
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd

from . import formatting, nputils, utils
from .indexing import IndexSelResult, PandasIndexingAdapter, PandasMultiIndexingAdapter
from .utils import Frozen, get_valid_numpy_dtype, is_dict_like, is_scalar

if TYPE_CHECKING:
    from .types import ErrorOptions, T_Index
    from .variable import Variable

IndexVars = Dict[Any, "Variable"]
```
### 8 - xarray/core/indexes.py:

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
### 9 - xarray/core/indexes.py:

Start line: 1195, End line: 1225

```python
class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):

    def copy_indexes(
        self, deep: bool = True
    ) -> tuple[dict[Hashable, T_PandasOrXarrayIndex], dict[Hashable, Variable]]:
        """Return a new dictionary with copies of indexes, preserving
        unique indexes.

        """
        new_indexes = {}
        new_index_vars = {}
        for idx, coords in self.group_by_index():
            new_idx = idx.copy(deep=deep)
            idx_vars = idx.create_variables(coords)
            new_indexes.update({k: new_idx for k in coords})
            new_index_vars.update(idx_vars)

        return new_indexes, new_index_vars

    def __iter__(self) -> Iterator[T_PandasOrXarrayIndex]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __contains__(self, key) -> bool:
        return key in self._indexes

    def __getitem__(self, key) -> T_PandasOrXarrayIndex:
        return self._indexes[key]

    def __repr__(self):
        return formatting.indexes_repr(self)
```
### 10 - xarray/core/indexes.py:

Start line: 249, End line: 282

```python
class PandasIndex(Index):

    @classmethod
    def from_variables(cls, variables: Mapping[Any, Variable]) -> PandasIndex:
        if len(variables) != 1:
            raise ValueError(
                f"PandasIndex only accepts one variable, found {len(variables)} variables"
            )

        name, var = next(iter(variables.items()))

        if var.ndim != 1:
            raise ValueError(
                "PandasIndex only accepts a 1-dimensional variable, "
                f"variable {name!r} has {var.ndim} dimensions"
            )

        dim = var.dims[0]

        # TODO: (benbovy - explicit indexes): add __index__ to ExplicitlyIndexesNDArrayMixin?
        # this could be eventually used by Variable.to_index() and would remove the need to perform
        # the checks below.

        # preserve wrapped pd.Index (if any)
        data = getattr(var._data, "array", var.data)
        # multi-index level variable: get level index
        if isinstance(var._data, PandasMultiIndexingAdapter):
            level = var._data.level
            if level is not None:
                data = var._data.array.get_level_values(level)

        obj = cls(data, dim, coord_dtype=var.dtype)
        assert not isinstance(obj.index, pd.MultiIndex)
        obj.index.name = name

        return obj
```
### 11 - xarray/core/indexes.py:

Start line: 936, End line: 950

```python
class PandasMultiIndex(PandasIndex):

    def rename(self, name_dict, dims_dict):
        if not set(self.index.names) & set(name_dict) and self.dim not in dims_dict:
            return self

        # pandas 1.3.0: could simply do `self.index.rename(names_dict)`
        new_names = [name_dict.get(k, k) for k in self.index.names]
        index = self.index.rename(new_names)

        new_dim = dims_dict.get(self.dim, self.dim)
        new_level_coords_dtype = {
            k: v for k, v in zip(new_names, self.level_coords_dtype.values())
        }
        return self._replace(
            index, dim=new_dim, level_coords_dtype=new_level_coords_dtype
        )
```
### 12 - xarray/core/indexes.py:

Start line: 586, End line: 604

```python
class PandasMultiIndex(PandasIndex):

    @classmethod
    def concat(  # type: ignore[override]
        cls,
        indexes: Sequence[PandasMultiIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] = None,
    ) -> PandasMultiIndex:
        new_pd_index = cls._concat_indexes(indexes, dim, positions)

        if not indexes:
            level_coords_dtype = None
        else:
            level_coords_dtype = {}
            for name in indexes[0].level_coords_dtype:
                level_coords_dtype[name] = np.result_type(
                    *[idx.level_coords_dtype[name] for idx in indexes]
                )

        return cls(new_pd_index, dim=dim, level_coords_dtype=level_coords_dtype)
```
### 14 - xarray/core/indexes.py:

Start line: 771, End line: 875

```python
class PandasMultiIndex(PandasIndex):

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        from .dataarray import DataArray
        from .variable import Variable

        if method is not None or tolerance is not None:
            raise ValueError(
                "multi-index does not support ``method`` and ``tolerance``"
            )

        new_index = None
        scalar_coord_values = {}

        # label(s) given for multi-index level(s)
        if all([lbl in self.index.names for lbl in labels]):
            label_values = {}
            for k, v in labels.items():
                label_array = normalize_label(v, dtype=self.level_coords_dtype[k])
                try:
                    label_values[k] = as_scalar(label_array)
                except ValueError:
                    # label should be an item not an array-like
                    raise ValueError(
                        "Vectorized selection is not "
                        f"available along coordinate {k!r} (multi-index level)"
                    )

            has_slice = any([isinstance(v, slice) for v in label_values.values()])

            if len(label_values) == self.index.nlevels and not has_slice:
                indexer = self.index.get_loc(
                    tuple(label_values[k] for k in self.index.names)
                )
            else:
                indexer, new_index = self.index.get_loc_level(
                    tuple(label_values.values()), level=tuple(label_values.keys())
                )
                scalar_coord_values.update(label_values)
                # GH2619. Raise a KeyError if nothing is chosen
                if indexer.dtype.kind == "b" and indexer.sum() == 0:
                    raise KeyError(f"{labels} not found")

        # assume one label value given for the multi-index "array" (dimension)
        else:
            if len(labels) > 1:
                coord_name = next(iter(set(labels) - set(self.index.names)))
                raise ValueError(
                    f"cannot provide labels for both coordinate {coord_name!r} (multi-index array) "
                    f"and one or more coordinates among {self.index.names!r} (multi-index levels)"
                )

            coord_name, label = next(iter(labels.items()))

            if is_dict_like(label):
                invalid_levels = [
                    name for name in label if name not in self.index.names
                ]
                if invalid_levels:
                    raise ValueError(
                        f"invalid multi-index level names {invalid_levels}"
                    )
                return self.sel(label)

            elif isinstance(label, slice):
                indexer = _query_slice(self.index, label, coord_name)

            elif isinstance(label, tuple):
                if _is_nested_tuple(label):
                    indexer = self.index.get_locs(label)
                elif len(label) == self.index.nlevels:
                    indexer = self.index.get_loc(label)
                else:
                    levels = [self.index.names[i] for i in range(len(label))]
                    indexer, new_index = self.index.get_loc_level(label, level=levels)
                    scalar_coord_values.update({k: v for k, v in zip(levels, label)})

            else:
                label_array = normalize_label(label)
                if label_array.ndim == 0:
                    label_value = as_scalar(label_array)
                    indexer, new_index = self.index.get_loc_level(label_value, level=0)
                    scalar_coord_values[self.index.names[0]] = label_value
                elif label_array.dtype.kind == "b":
                    indexer = label_array
                else:
                    if label_array.ndim > 1:
                        raise ValueError(
                            "Vectorized selection is not available along "
                            f"coordinate {coord_name!r} with a multi-index"
                        )
                    indexer = get_indexer_nd(self.index, label_array)
                    if np.any(indexer < 0):
                        raise KeyError(f"not all values found in index {coord_name!r}")

                # attach dimension names and/or coordinates to positional indexer
                if isinstance(label, Variable):
                    indexer = Variable(label.dims, indexer)
                elif isinstance(label, DataArray):
                    # do not include label-indexer DataArray coordinates that conflict
                    # with the level names of this index
                    coords = {
                        k: v
                        for k, v in label._coords.items()
                        if k not in self.index.names
                    }
                    indexer = DataArray(indexer, coords=coords, dims=label.dims)
        # ... other code
```
### 16 - xarray/core/indexes.py:

Start line: 572, End line: 584

```python
class PandasMultiIndex(PandasIndex):

    @classmethod
    def from_variables(cls, variables: Mapping[Any, Variable]) -> PandasMultiIndex:
        _check_dim_compat(variables)
        dim = next(iter(variables.values())).dims[0]

        index = pd.MultiIndex.from_arrays(
            [var.values for var in variables.values()], names=variables.keys()
        )
        index.name = dim
        level_coords_dtype = {name: var.dtype for name, var in variables.items()}
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)

        return obj
```
### 17 - xarray/core/indexes.py:

Start line: 364, End line: 428

```python
class PandasIndex(Index):

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        from .dataarray import DataArray
        from .variable import Variable

        if method is not None and not isinstance(method, str):
            raise TypeError("``method`` must be a string")

        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            indexer = _query_slice(self.index, label, coord_name, method, tolerance)
        elif is_dict_like(label):
            raise ValueError(
                "cannot use a dict-like object for selection on "
                "a dimension that does not have a MultiIndex"
            )
        else:
            label_array = normalize_label(label, dtype=self.coord_dtype)
            if label_array.ndim == 0:
                label_value = as_scalar(label_array)
                if isinstance(self.index, pd.CategoricalIndex):
                    if method is not None:
                        raise ValueError(
                            "'method' is not supported when indexing using a CategoricalIndex."
                        )
                    if tolerance is not None:
                        raise ValueError(
                            "'tolerance' is not supported when indexing using a CategoricalIndex."
                        )
                    indexer = self.index.get_loc(label_value)
                else:
                    if method is not None:
                        indexer = get_indexer_nd(
                            self.index, label_array, method, tolerance
                        )
                        if np.any(indexer < 0):
                            raise KeyError(
                                f"not all values found in index {coord_name!r}"
                            )
                    else:
                        try:
                            indexer = self.index.get_loc(label_value)
                        except KeyError as e:
                            raise KeyError(
                                f"not all values found in index {coord_name!r}. "
                                "Try setting the `method` keyword argument (example: method='nearest')."
                            ) from e

            elif label_array.dtype.kind == "b":
                indexer = label_array
            else:
                indexer = get_indexer_nd(self.index, label_array, method, tolerance)
                if np.any(indexer < 0):
                    raise KeyError(f"not all values found in index {coord_name!r}")

            # attach dimension names and/or coordinates to positional indexer
            if isinstance(label, Variable):
                indexer = Variable(label.dims, indexer)
            elif isinstance(label, DataArray):
                indexer = DataArray(indexer, coords=label._coords, dims=label.dims)

        return IndexSelResult({self.dim: indexer})
```
### 19 - xarray/core/indexes.py:

Start line: 284, End line: 304

```python
class PandasIndex(Index):

    @staticmethod
    def _concat_indexes(indexes, dim, positions=None) -> pd.Index:
        new_pd_index: pd.Index

        if not indexes:
            new_pd_index = pd.Index([])
        else:
            if not all(idx.dim == dim for idx in indexes):
                dims = ",".join({f"{idx.dim!r}" for idx in indexes})
                raise ValueError(
                    f"Cannot concatenate along dimension {dim!r} indexes with "
                    f"dimensions: {dims}"
                )
            pd_indexes = [idx.index for idx in indexes]
            new_pd_index = pd_indexes[0].append(pd_indexes[1:])

            if positions is not None:
                indices = nputils.inverse_permutation(np.concatenate(positions))
                new_pd_index = new_pd_index.take(indices)

        return new_pd_index
```
### 20 - xarray/core/indexes.py:

Start line: 735, End line: 769

```python
class PandasMultiIndex(PandasIndex):

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        from .variable import IndexVariable

        if variables is None:
            variables = {}

        index_vars: IndexVars = {}
        for name in (self.dim,) + self.index.names:
            if name == self.dim:
                level = None
                dtype = None
            else:
                level = name
                dtype = self.level_coords_dtype[name]

            var = variables.get(name, None)
            if var is not None:
                attrs = var.attrs
                encoding = var.encoding
            else:
                attrs = {}
                encoding = {}

            data = PandasMultiIndexingAdapter(self.index, dtype=dtype, level=level)
            index_vars[name] = IndexVariable(
                self.dim,
                data,
                attrs=attrs,
                encoding=encoding,
                fastpath=True,
            )

        return index_vars
```
### 21 - xarray/core/indexes.py:

Start line: 1329, End line: 1350

```python
def _apply_indexes(
    indexes: Indexes[Index],
    args: Mapping[Any, Any],
    func: str,
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    new_indexes: dict[Hashable, Index] = {k: v for k, v in indexes.items()}
    new_index_variables: dict[Hashable, Variable] = {}

    for index, index_vars in indexes.group_by_index():
        index_dims = {d for var in index_vars.values() for d in var.dims}
        index_args = {k: v for k, v in args.items() if k in index_dims}
        if index_args:
            new_index = getattr(index, func)(index_args)
            if new_index is not None:
                new_indexes.update({k: new_index for k in index_vars})
                new_index_vars = new_index.create_variables(index_vars)
                new_index_variables.update(new_index_vars)
            else:
                for k in index_vars:
                    new_indexes.pop(k, None)

    return new_indexes, new_index_variables
```
### 22 - xarray/core/indexes.py:

Start line: 34, End line: 124

```python
class Index:
    """Base class inherited by all xarray-compatible indexes."""

    @classmethod
    def from_variables(cls, variables: Mapping[Any, Variable]) -> Index:
        raise NotImplementedError()

    @classmethod
    def concat(
        cls: type[T_Index],
        indexes: Sequence[T_Index],
        dim: Hashable,
        positions: Iterable[Iterable[int]] = None,
    ) -> T_Index:
        raise NotImplementedError()

    @classmethod
    def stack(cls, variables: Mapping[Any, Variable], dim: Hashable) -> Index:
        raise NotImplementedError(
            f"{cls!r} cannot be used for creating an index of stacked coordinates"
        )

    def unstack(self) -> tuple[dict[Hashable, Index], pd.MultiIndex]:
        raise NotImplementedError()

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        if variables is not None:
            # pass through
            return dict(**variables)
        else:
            return {}

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a TypeError
        if this is not supported.

        This method is used by all xarray operations that expect/require a
        pandas.Index object.

        """
        raise TypeError(f"{self!r} cannot be cast to a pandas.Index object")

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Index | None:
        return None

    def sel(self, labels: dict[Any, Any]) -> IndexSelResult:
        raise NotImplementedError(f"{self!r} doesn't support label-based selection")

    def join(self: T_Index, other: T_Index, how: str = "inner") -> T_Index:
        raise NotImplementedError(
            f"{self!r} doesn't support alignment with inner/outer join method"
        )

    def reindex_like(self: T_Index, other: T_Index) -> dict[Hashable, Any]:
        raise NotImplementedError(f"{self!r} doesn't support re-indexing labels")

    def equals(self, other):  # pragma: no cover
        raise NotImplementedError()

    def roll(self, shifts: Mapping[Any, int]) -> Index | None:
        return None

    def rename(
        self, name_dict: Mapping[Any, Hashable], dims_dict: Mapping[Any, Hashable]
    ) -> Index:
        return self

    def __copy__(self) -> Index:
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> Index:
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    def copy(self, deep: bool = True) -> Index:
        cls = self.__class__
        copied = cls.__new__(cls)
        if deep:
            for k, v in self.__dict__.items():
                setattr(copied, k, copy.deepcopy(v))
        else:
            copied.__dict__.update(self.__dict__)
        return copied

    def __getitem__(self, indexer: Any):
        raise NotImplementedError()
```
### 24 - xarray/core/indexes.py:

Start line: 1353, End line: 1364

```python
def isel_indexes(
    indexes: Indexes[Index],
    indexers: Mapping[Any, Any],
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    return _apply_indexes(indexes, indexers, "isel")


def roll_indexes(
    indexes: Indexes[Index],
    shifts: Mapping[Any, int],
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    return _apply_indexes(indexes, shifts, "roll")
```
### 26 - xarray/core/indexes.py:

Start line: 202, End line: 214

```python
def as_scalar(value: np.ndarray):
    # see https://github.com/pydata/xarray/pull/4292 for details
    return value[()] if value.dtype.kind in "mM" else value.item()


def get_indexer_nd(index, labels, method=None, tolerance=None):
    """Wrapper around :meth:`pandas.Index.get_indexer` supporting n-dimensional
    labels
    """
    flat_labels = np.ravel(labels)
    flat_indexer = index.get_indexer(flat_labels, method=method, tolerance=tolerance)
    indexer = flat_indexer.reshape(labels.shape)
    return indexer
```
### 29 - xarray/core/indexes.py:

Start line: 306, End line: 341

```python
class PandasIndex(Index):

    @classmethod
    def concat(
        cls,
        indexes: Sequence[PandasIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] = None,
    ) -> PandasIndex:
        new_pd_index = cls._concat_indexes(indexes, dim, positions)

        if not indexes:
            coord_dtype = None
        else:
            coord_dtype = np.result_type(*[idx.coord_dtype for idx in indexes])

        return cls(new_pd_index, dim=dim, coord_dtype=coord_dtype)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        from .variable import IndexVariable

        name = self.index.name
        attrs: Mapping[Hashable, Any] | None
        encoding: Mapping[Hashable, Any] | None

        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        data = PandasIndexingAdapter(self.index, dtype=self.coord_dtype)
        var = IndexVariable(self.dim, data, attrs=attrs, encoding=encoding)
        return {name: var}
```
### 30 - xarray/core/indexes.py:

Start line: 1296, End line: 1326

```python
def indexes_all_equal(
    elements: Sequence[tuple[Index, dict[Hashable, Variable]]]
) -> bool:
    """Check if indexes are all equal.

    If they are not of the same type or they do not implement this check, check
    if their coordinate variables are all equal instead.

    """

    def check_variables():
        variables = [e[1] for e in elements]
        return any(
            not variables[0][k].equals(other_vars[k])
            for other_vars in variables[1:]
            for k in variables[0]
        )

    indexes = [e[0] for e in elements]
    same_type = all(type(indexes[0]) is type(other_idx) for other_idx in indexes[1:])
    if same_type:
        try:
            not_equal = any(
                not indexes[0].equals(other_idx) for other_idx in indexes[1:]
            )
        except NotImplementedError:
            not_equal = check_variables()
    else:
        not_equal = check_variables()

    return not not_equal
```
### 31 - xarray/core/indexes.py:

Start line: 1257, End line: 1293

```python
def indexes_equal(
    index: Index,
    other_index: Index,
    variable: Variable,
    other_variable: Variable,
    cache: dict[tuple[int, int], bool | None] = None,
) -> bool:
    """Check if two indexes are equal, possibly with cached results.

    If the two indexes are not of the same type or they do not implement
    equality, fallback to coordinate labels equality check.

    """
    if cache is None:
        # dummy cache
        cache = {}

    key = (id(index), id(other_index))
    equal: bool | None = None

    if key not in cache:
        if type(index) is type(other_index):
            try:
                equal = index.equals(other_index)
            except NotImplementedError:
                equal = None
            else:
                cache[key] = equal
        else:
            equal = None
    else:
        equal = cache[key]

    if equal is None:
        equal = variable.equals(other_variable)

    return cast(bool, equal)
```
### 34 - xarray/core/indexes.py:

Start line: 510, End line: 534

```python
def remove_unused_levels_categories(index: pd.Index) -> pd.Index:
    """
    Remove unused levels from MultiIndex and unused categories from CategoricalIndex
    """
    if isinstance(index, pd.MultiIndex):
        index = index.remove_unused_levels()
        # if it contains CategoricalIndex, we need to remove unused categories
        # manually. See https://github.com/pandas-dev/pandas/issues/30846
        if any(isinstance(lev, pd.CategoricalIndex) for lev in index.levels):
            levels = []
            for i, level in enumerate(index.levels):
                if isinstance(level, pd.CategoricalIndex):
                    level = level[index.codes[i]].remove_unused_categories()
                else:
                    level = level[index.codes[i]]
                levels.append(level)
            # TODO: calling from_array() reorders MultiIndex levels. It would
            # be best to avoid this, if possible, e.g., by using
            # MultiIndex.remove_unused_levels() (which does not reorder) on the
            # part of the MultiIndex that is not categorical, or by fixing this
            # upstream in pandas.
            index = pd.MultiIndex.from_arrays(levels, names=index.names)
    elif isinstance(index, pd.CategoricalIndex):
        index = index.remove_unused_categories()
    return index
```
### 36 - xarray/core/indexes.py:

Start line: 217, End line: 247

```python
class PandasIndex(Index):
    """Wrap a pandas.Index as an xarray compatible index."""

    index: pd.Index
    dim: Hashable
    coord_dtype: Any

    __slots__ = ("index", "dim", "coord_dtype")

    def __init__(self, array: Any, dim: Hashable, coord_dtype: Any = None):
        # make a shallow copy: cheap and because the index name may be updated
        # here or in other constructors (cannot use pd.Index.rename as this
        # constructor is also called from PandasMultiIndex)
        index = utils.safe_cast_to_index(array).copy()

        if index.name is None:
            index.name = dim

        self.index = index
        self.dim = dim

        if coord_dtype is None:
            coord_dtype = get_valid_numpy_dtype(index)
        self.coord_dtype = coord_dtype

    def _replace(self, index, dim=None, coord_dtype=None):
        if dim is None:
            dim = self.dim
        if coord_dtype is None:
            coord_dtype = self.coord_dtype
        return type(self)(index, dim, coord_dtype)
```
### 37 - xarray/core/indexes.py:

Start line: 127, End line: 163

```python
def _sanitize_slice_element(x):
    from .dataarray import DataArray
    from .variable import Variable

    if not isinstance(x, tuple) and len(np.shape(x)) != 0:
        raise ValueError(
            f"cannot use non-scalar arrays in a slice for xarray indexing: {x}"
        )

    if isinstance(x, (Variable, DataArray)):
        x = x.values

    if isinstance(x, np.ndarray):
        x = x[()]

    return x


def _query_slice(index, label, coord_name="", method=None, tolerance=None):
    if method is not None or tolerance is not None:
        raise NotImplementedError(
            "cannot use ``method`` argument if any indexers are slice objects"
        )
    indexer = index.slice_indexer(
        _sanitize_slice_element(label.start),
        _sanitize_slice_element(label.stop),
        _sanitize_slice_element(label.step),
    )
    if not isinstance(indexer, slice):
        # unlike pandas, in xarray we never want to silently convert a
        # slice indexer into an array indexer
        raise KeyError(
            "cannot represent labeled-based slice indexer for coordinate "
            f"{coord_name!r} with a slice over integer positions; the index is "
            "unsorted or non-unique"
        )
    return indexer
```
### 38 - xarray/core/indexes.py:

Start line: 705, End line: 733

```python
class PandasMultiIndex(PandasIndex):

    def keep_levels(
        self, level_variables: Mapping[Any, Variable]
    ) -> PandasMultiIndex | PandasIndex:
        """Keep only the provided levels and return a new multi-index with its
        corresponding coordinates.

        """
        index = self.index.droplevel(
            [k for k in self.index.names if k not in level_variables]
        )

        if isinstance(index, pd.MultiIndex):
            level_coords_dtype = {k: self.level_coords_dtype[k] for k in index.names}
            return self._replace(index, level_coords_dtype=level_coords_dtype)
        else:
            return PandasIndex(
                index, self.dim, coord_dtype=self.level_coords_dtype[index.name]
            )

    def reorder_levels(
        self, level_variables: Mapping[Any, Variable]
    ) -> PandasMultiIndex:
        """Re-arrange index levels using input order and return a new multi-index with
        its corresponding coordinates.

        """
        index = self.index.reorder_levels(level_variables.keys())
        level_coords_dtype = {k: self.level_coords_dtype[k] for k in index.names}
        return self._replace(index, level_coords_dtype=level_coords_dtype)
```
### 43 - xarray/core/indexes.py:

Start line: 537, End line: 570

```python
class PandasMultiIndex(PandasIndex):
    """Wrap a pandas.MultiIndex as an xarray compatible index."""

    level_coords_dtype: dict[str, Any]

    __slots__ = ("index", "dim", "coord_dtype", "level_coords_dtype")

    def __init__(self, array: Any, dim: Hashable, level_coords_dtype: Any = None):
        super().__init__(array, dim)

        # default index level names
        names = []
        for i, idx in enumerate(self.index.levels):
            name = idx.name or f"{dim}_level_{i}"
            if name == dim:
                raise ValueError(
                    f"conflicting multi-index level name {name!r} with dimension {dim!r}"
                )
            names.append(name)
        self.index.names = names

        if level_coords_dtype is None:
            level_coords_dtype = {
                idx.name: get_valid_numpy_dtype(idx) for idx in self.index.levels
            }
        self.level_coords_dtype = level_coords_dtype

    def _replace(self, index, dim=None, level_coords_dtype=None) -> PandasMultiIndex:
        if dim is None:
            dim = self.dim
        index.name = dim
        if level_coords_dtype is None:
            level_coords_dtype = self.level_coords_dtype
        return type(self)(index, dim, level_coords_dtype)
```
### 44 - xarray/core/indexes.py:

Start line: 1141, End line: 1162

```python
class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):

    def get_all_dims(
        self, key: Hashable, errors: ErrorOptions = "raise"
    ) -> Mapping[Hashable, int]:
        """Return all dimensions shared by an index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        dims : dict
            A dictionary of all dimensions shared by an index.

        """
        from .variable import calculate_dimensions

        return calculate_dimensions(self.get_all_coords(key, errors=errors))
```
### 46 - xarray/core/indexes.py:

Start line: 1391, End line: 1409

```python
def assert_no_index_corrupted(
    indexes: Indexes[Index],
    coord_names: set[Hashable],
) -> None:
    """Assert removing coordinates will not corrupt indexes."""

    # An index may be corrupted when the set of its corresponding coordinate name(s)
    # partially overlaps the set of coordinate names to remove
    for index, index_coords in indexes.group_by_index():
        common_names = set(index_coords) & coord_names
        if common_names and len(common_names) != len(index_coords):
            common_names_str = ", ".join(f"{k!r}" for k in common_names)
            index_names_str = ", ".join(f"{k!r}" for k in index_coords)
            raise ValueError(
                f"cannot remove coordinate(s) {common_names_str}, which would corrupt "
                f"the following index built from coordinates {index_names_str}:\n"
                f"{index}"
            )
```
### 47 - xarray/core/indexes.py:

Start line: 1110, End line: 1139

```python
class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):

    def get_all_coords(
        self, key: Hashable, errors: ErrorOptions = "raise"
    ) -> dict[Hashable, Variable]:
        """Return all coordinates having the same index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        coords : dict
            A dictionary of all coordinate variables having the same index.

        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if key not in self._indexes:
            if errors == "raise":
                raise ValueError(f"no index found for {key!r} coordinate")
            else:
                return {}

        all_coord_names = self._id_coord_names[self._coord_name_id[key]]
        return {k: self._variables[k] for k in all_coord_names}
```
### 48 - xarray/core/indexes.py:

Start line: 487, End line: 507

```python
def _check_dim_compat(variables: Mapping[Any, Variable], all_dims: str = "equal"):
    """Check that all multi-index variable candidates are 1-dimensional and
    either share the same (single) dimension or each have a different dimension.

    """
    if any([var.ndim != 1 for var in variables.values()]):
        raise ValueError("PandasMultiIndex only accepts 1-dimensional variables")

    dims = {var.dims for var in variables.values()}

    if all_dims == "equal" and len(dims) > 1:
        raise ValueError(
            "unmatched dimensions for multi-index variables "
            + ", ".join([f"{k!r} {v.dims}" for k, v in variables.items()])
        )

    if all_dims == "different" and len(dims) < len(variables):
        raise ValueError(
            "conflicting dimensions for multi-index product variables "
            + ", ".join([f"{k!r} {v.dims}" for k, v in variables.items()])
        )
```
### 68 - xarray/core/indexes.py:

Start line: 606, End line: 648

```python
class PandasMultiIndex(PandasIndex):

    @classmethod
    def stack(
        cls, variables: Mapping[Any, Variable], dim: Hashable
    ) -> PandasMultiIndex:
        """Create a new Pandas MultiIndex from the product of 1-d variables (levels) along a
        new dimension.

        Level variables must have a dimension distinct from each other.

        Keeps levels the same (doesn't refactorize them) so that it gives back the original
        labels after a stack/unstack roundtrip.

        """
        _check_dim_compat(variables, all_dims="different")

        level_indexes = [utils.safe_cast_to_index(var) for var in variables.values()]
        for name, idx in zip(variables, level_indexes):
            if isinstance(idx, pd.MultiIndex):
                raise ValueError(
                    f"cannot create a multi-index along stacked dimension {dim!r} "
                    f"from variable {name!r} that wraps a multi-index"
                )

        split_labels, levels = zip(*[lev.factorize() for lev in level_indexes])
        labels_mesh = np.meshgrid(*split_labels, indexing="ij")
        labels = [x.ravel() for x in labels_mesh]

        index = pd.MultiIndex(levels, labels, sortorder=0, names=variables.keys())
        level_coords_dtype = {k: var.dtype for k, var in variables.items()}

        return cls(index, dim, level_coords_dtype=level_coords_dtype)

    def unstack(self) -> tuple[dict[Hashable, Index], pd.MultiIndex]:
        clean_index = remove_unused_levels_categories(self.index)

        new_indexes: dict[Hashable, Index] = {}
        for name, lev in zip(clean_index.names, clean_index.levels):
            idx = PandasIndex(
                lev.copy(), name, coord_dtype=self.level_coords_dtype[name]
            )
            new_indexes[name] = idx

        return new_indexes, clean_index
```
### 83 - xarray/core/indexes.py:

Start line: 1367, End line: 1388

```python
def filter_indexes_from_coords(
    indexes: Mapping[Any, Index],
    filtered_coord_names: set,
) -> dict[Hashable, Index]:
    """Filter index items given a (sub)set of coordinate names.

    Drop all multi-coordinate related index items for any key missing in the set
    of coordinate names.

    """
    filtered_indexes: dict[Any, Index] = dict(**indexes)

    index_coord_names: dict[Hashable, set[Hashable]] = defaultdict(set)
    for name, idx in indexes.items():
        index_coord_names[id(idx)].add(name)

    for idx_coord_names in index_coord_names.values():
        if not idx_coord_names <= filtered_coord_names:
            for k in idx_coord_names:
                del filtered_indexes[k]

    return filtered_indexes
```
### 84 - xarray/core/indexes.py:

Start line: 166, End line: 187

```python
def _asarray_tuplesafe(values):
    """
    Convert values into a numpy array of at most 1-dimension, while preserving
    tuples.

    Adapted from pandas.core.common._asarray_tuplesafe
    """
    if isinstance(values, tuple):
        result = utils.to_0d_object_array(values)
    else:
        result = np.asarray(values)
        if result.ndim == 2:
            result = np.empty(len(values), dtype=object)
            result[:] = values

    return result


def _is_nested_tuple(possible_tuple):
    return isinstance(possible_tuple, tuple) and any(
        isinstance(value, (tuple, list, slice)) for value in possible_tuple
    )
```
