# pydata__xarray-3239

| **pydata/xarray** | `e90e8bc06cf8e7c97c7dc4c0e8ff1bf87c49faf6` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 1328 |
| **Any found context length** | 1328 |
| **Avg pos** | 247.0 |
| **Min pos** | 4 |
| **Max pos** | 138 |
| **Top file pos** | 1 |
| **Missing snippets** | 32 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -761,7 +761,7 @@ def open_mfdataset(
         `xarray.auto_combine` is used, but in the future this behavior will 
         switch to use `xarray.combine_by_coords` by default.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts when merging:
          * 'broadcast_equals': all values must be equal when variables are
@@ -772,6 +772,7 @@ def open_mfdataset(
          * 'no_conflicts': only values which are not null in both datasets
            must be equal. The returned dataset then contains the combination
            of all non-null values.
+         * 'override': skip comparing and pick variable from first dataset
     preprocess : callable, optional
         If provided, call this function on each dataset prior to concatenation.
         You can find the file-name from which each dataset was loaded in
diff --git a/xarray/core/combine.py b/xarray/core/combine.py
--- a/xarray/core/combine.py
+++ b/xarray/core/combine.py
@@ -243,6 +243,7 @@ def _combine_1d(
                 dim=concat_dim,
                 data_vars=data_vars,
                 coords=coords,
+                compat=compat,
                 fill_value=fill_value,
                 join=join,
             )
@@ -351,7 +352,7 @@ def combine_nested(
         Must be the same length as the depth of the list passed to
         ``datasets``.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential merge conflicts:
 
@@ -363,6 +364,7 @@ def combine_nested(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' or list of str}, optional
@@ -504,7 +506,7 @@ def combine_by_coords(
     datasets : sequence of xarray.Dataset
         Dataset objects to combine.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
 
@@ -516,6 +518,7 @@ def combine_by_coords(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' or list of str}, optional
@@ -598,6 +601,7 @@ def combine_by_coords(
             concat_dims=concat_dims,
             data_vars=data_vars,
             coords=coords,
+            compat=compat,
             fill_value=fill_value,
             join=join,
         )
@@ -667,7 +671,7 @@ def auto_combine(
         component files. Set ``concat_dim=None`` explicitly to disable
         concatenation.
     compat : {'identical', 'equals', 'broadcast_equals',
-             'no_conflicts'}, optional
+             'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
         - 'broadcast_equals': all values must be equal when variables are
@@ -678,6 +682,7 @@ def auto_combine(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' o list of str}, optional
@@ -832,6 +837,7 @@ def _old_auto_combine(
                 dim=dim,
                 data_vars=data_vars,
                 coords=coords,
+                compat=compat,
                 fill_value=fill_value,
                 join=join,
             )
@@ -850,6 +856,7 @@ def _auto_concat(
     coords="different",
     fill_value=dtypes.NA,
     join="outer",
+    compat="no_conflicts",
 ):
     if len(datasets) == 1 and dim is None:
         # There is nothing more to combine, so kick out early.
@@ -876,5 +883,10 @@ def _auto_concat(
                 )
             dim, = concat_dims
         return concat(
-            datasets, dim=dim, data_vars=data_vars, coords=coords, fill_value=fill_value
+            datasets,
+            dim=dim,
+            data_vars=data_vars,
+            coords=coords,
+            fill_value=fill_value,
+            compat=compat,
         )
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -4,6 +4,7 @@
 
 from . import dtypes, utils
 from .alignment import align
+from .merge import unique_variable, _VALID_COMPAT
 from .variable import IndexVariable, Variable, as_variable
 from .variable import concat as concat_vars
 
@@ -59,12 +60,19 @@ def concat(
             those corresponding to other dimensions.
           * list of str: The listed coordinate variables will be concatenated,
             in addition to the 'minimal' coordinates.
-    compat : {'equals', 'identical'}, optional
-        String indicating how to compare non-concatenated variables and
-        dataset global attributes for potential conflicts. 'equals' means
-        that all variable values and dimensions must be the same;
-        'identical' means that variable attributes and global attributes
-        must also be equal.
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
+        String indicating how to compare non-concatenated variables of the same name for
+        potential conflicts. This is passed down to merge.
+
+        - 'broadcast_equals': all values must be equal when variables are
+          broadcast against each other to ensure common dimensions.
+        - 'equals': all values and dimensions must be the same.
+        - 'identical': all values, dimensions and attributes must be the
+          same.
+        - 'no_conflicts': only values which are not null in both datasets
+          must be equal. The returned dataset then contains the combination
+          of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     positions : None or list of integer arrays, optional
         List of integer arrays which specifies the integer positions to which
         to assign each dataset along the concatenated dimension. If not
@@ -107,6 +115,12 @@ def concat(
     except StopIteration:
         raise ValueError("must supply at least one object to concatenate")
 
+    if compat not in _VALID_COMPAT:
+        raise ValueError(
+            "compat=%r invalid: must be 'broadcast_equals', 'equals', 'identical', 'no_conflicts' or 'override'"
+            % compat
+        )
+
     if isinstance(first_obj, DataArray):
         f = _dataarray_concat
     elif isinstance(first_obj, Dataset):
@@ -143,23 +157,39 @@ def _calc_concat_dim_coord(dim):
     return dim, coord
 
 
-def _calc_concat_over(datasets, dim, data_vars, coords):
+def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
     """
     Determine which dataset variables need to be concatenated in the result,
-    and which can simply be taken from the first dataset.
     """
     # Return values
     concat_over = set()
     equals = {}
 
-    if dim in datasets[0]:
+    if dim in dim_names:
+        concat_over_existing_dim = True
         concat_over.add(dim)
+    else:
+        concat_over_existing_dim = False
+
+    concat_dim_lengths = []
     for ds in datasets:
+        if concat_over_existing_dim:
+            if dim not in ds.dims:
+                if dim in ds:
+                    ds = ds.set_coords(dim)
+                else:
+                    raise ValueError("%r is not present in all datasets" % dim)
         concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
+        concat_dim_lengths.append(ds.dims.get(dim, 1))
 
     def process_subset_opt(opt, subset):
         if isinstance(opt, str):
             if opt == "different":
+                if compat == "override":
+                    raise ValueError(
+                        "Cannot specify both %s='different' and compat='override'."
+                        % subset
+                    )
                 # all nonindexes that are not the same in each dataset
                 for k in getattr(datasets[0], subset):
                     if k not in concat_over:
@@ -173,7 +203,7 @@ def process_subset_opt(opt, subset):
                         for ds_rhs in datasets[1:]:
                             v_rhs = ds_rhs.variables[k].compute()
                             computed.append(v_rhs)
-                            if not v_lhs.equals(v_rhs):
+                            if not getattr(v_lhs, compat)(v_rhs):
                                 concat_over.add(k)
                                 equals[k] = False
                                 # computed variables are not to be re-computed
@@ -209,7 +239,29 @@ def process_subset_opt(opt, subset):
 
     process_subset_opt(data_vars, "data_vars")
     process_subset_opt(coords, "coords")
-    return concat_over, equals
+    return concat_over, equals, concat_dim_lengths
+
+
+# determine dimensional coordinate names and a dict mapping name to DataArray
+def _parse_datasets(datasets):
+
+    dims = set()
+    all_coord_names = set()
+    data_vars = set()  # list of data_vars
+    dim_coords = dict()  # maps dim name to variable
+    dims_sizes = {}  # shared dimension sizes to expand variables
+
+    for ds in datasets:
+        dims_sizes.update(ds.dims)
+        all_coord_names.update(ds.coords)
+        data_vars.update(ds.data_vars)
+
+        for dim in set(ds.dims) - dims:
+            if dim not in dim_coords:
+                dim_coords[dim] = ds.coords[dim].variable
+        dims = dims | set(ds.dims)
+
+    return dim_coords, dims_sizes, all_coord_names, data_vars
 
 
 def _dataset_concat(
@@ -227,11 +279,6 @@ def _dataset_concat(
     """
     from .dataset import Dataset
 
-    if compat not in ["equals", "identical"]:
-        raise ValueError(
-            "compat=%r invalid: must be 'equals' " "or 'identical'" % compat
-        )
-
     dim, coord = _calc_concat_dim_coord(dim)
     # Make sure we're working on a copy (we'll be loading variables)
     datasets = [ds.copy() for ds in datasets]
@@ -239,62 +286,65 @@ def _dataset_concat(
         *datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value
     )
 
-    concat_over, equals = _calc_concat_over(datasets, dim, data_vars, coords)
+    dim_coords, dims_sizes, coord_names, data_names = _parse_datasets(datasets)
+    dim_names = set(dim_coords)
+    unlabeled_dims = dim_names - coord_names
+
+    both_data_and_coords = coord_names & data_names
+    if both_data_and_coords:
+        raise ValueError(
+            "%r is a coordinate in some datasets but not others." % both_data_and_coords
+        )
+    # we don't want the concat dimension in the result dataset yet
+    dim_coords.pop(dim, None)
+    dims_sizes.pop(dim, None)
+
+    # case where concat dimension is a coordinate or data_var but not a dimension
+    if (dim in coord_names or dim in data_names) and dim not in dim_names:
+        datasets = [ds.expand_dims(dim) for ds in datasets]
+
+    # determine which variables to concatentate
+    concat_over, equals, concat_dim_lengths = _calc_concat_over(
+        datasets, dim, dim_names, data_vars, coords, compat
+    )
+
+    # determine which variables to merge, and then merge them according to compat
+    variables_to_merge = (coord_names | data_names) - concat_over - dim_names
+
+    result_vars = {}
+    if variables_to_merge:
+        to_merge = {var: [] for var in variables_to_merge}
+
+        for ds in datasets:
+            absent_merge_vars = variables_to_merge - set(ds.variables)
+            if absent_merge_vars:
+                raise ValueError(
+                    "variables %r are present in some datasets but not others. "
+                    % absent_merge_vars
+                )
 
-    def insert_result_variable(k, v):
-        assert isinstance(v, Variable)
-        if k in datasets[0].coords:
-            result_coord_names.add(k)
-        result_vars[k] = v
+            for var in variables_to_merge:
+                to_merge[var].append(ds.variables[var])
 
-    # create the new dataset and add constant variables
-    result_vars = OrderedDict()
-    result_coord_names = set(datasets[0].coords)
+        for var in variables_to_merge:
+            result_vars[var] = unique_variable(
+                var, to_merge[var], compat=compat, equals=equals.get(var, None)
+            )
+    else:
+        result_vars = OrderedDict()
+    result_vars.update(dim_coords)
+
+    # assign attrs and encoding from first dataset
     result_attrs = datasets[0].attrs
     result_encoding = datasets[0].encoding
 
-    for k, v in datasets[0].variables.items():
-        if k not in concat_over:
-            insert_result_variable(k, v)
-
-    # check that global attributes and non-concatenated variables are fixed
-    # across all datasets
+    # check that global attributes are fixed across all datasets if necessary
     for ds in datasets[1:]:
         if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
-            raise ValueError("dataset global attributes not equal")
-        for k, v in ds.variables.items():
-            if k not in result_vars and k not in concat_over:
-                raise ValueError("encountered unexpected variable %r" % k)
-            elif (k in result_coord_names) != (k in ds.coords):
-                raise ValueError(
-                    "%r is a coordinate in some datasets but not " "others" % k
-                )
-            elif k in result_vars and k != dim:
-                # Don't use Variable.identical as it internally invokes
-                # Variable.equals, and we may already know the answer
-                if compat == "identical" and not utils.dict_equiv(
-                    v.attrs, result_vars[k].attrs
-                ):
-                    raise ValueError("variable %s not identical across datasets" % k)
-
-                # Proceed with equals()
-                try:
-                    # May be populated when using the "different" method
-                    is_equal = equals[k]
-                except KeyError:
-                    result_vars[k].load()
-                    is_equal = v.equals(result_vars[k])
-                if not is_equal:
-                    raise ValueError("variable %s not equal across datasets" % k)
+            raise ValueError("Dataset global attributes not equal.")
 
     # we've already verified everything is consistent; now, calculate
     # shared dimension sizes so we can expand the necessary variables
-    dim_lengths = [ds.dims.get(dim, 1) for ds in datasets]
-    non_concat_dims = {}
-    for ds in datasets:
-        non_concat_dims.update(ds.dims)
-    non_concat_dims.pop(dim, None)
-
     def ensure_common_dims(vars):
         # ensure each variable with the given name shares the same
         # dimensions and the same shape for all of them except along the
@@ -302,25 +352,27 @@ def ensure_common_dims(vars):
         common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
         if dim not in common_dims:
             common_dims = (dim,) + common_dims
-        for var, dim_len in zip(vars, dim_lengths):
+        for var, dim_len in zip(vars, concat_dim_lengths):
             if var.dims != common_dims:
-                common_shape = tuple(
-                    non_concat_dims.get(d, dim_len) for d in common_dims
-                )
+                common_shape = tuple(dims_sizes.get(d, dim_len) for d in common_dims)
                 var = var.set_dims(common_dims, common_shape)
             yield var
 
     # stack up each variable to fill-out the dataset (in order)
+    # n.b. this loop preserves variable order, needed for groupby.
     for k in datasets[0].variables:
         if k in concat_over:
             vars = ensure_common_dims([ds.variables[k] for ds in datasets])
             combined = concat_vars(vars, dim, positions)
-            insert_result_variable(k, combined)
+            assert isinstance(combined, Variable)
+            result_vars[k] = combined
 
     result = Dataset(result_vars, attrs=result_attrs)
-    result = result.set_coords(result_coord_names)
+    result = result.set_coords(coord_names)
     result.encoding = result_encoding
 
+    result = result.drop(unlabeled_dims, errors="ignore")
+
     if coord is not None:
         # add concat dimension last to ensure that its in the final Dataset
         result[coord.name] = coord
@@ -342,7 +394,7 @@ def _dataarray_concat(
 
     if data_vars != "all":
         raise ValueError(
-            "data_vars is not a valid argument when " "concatenating DataArray objects"
+            "data_vars is not a valid argument when concatenating DataArray objects"
         )
 
     datasets = []
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -1549,8 +1549,8 @@ def set_index(
         obj : DataArray
             Another DataArray, with this data but replaced coordinates.
 
-        Example
-        -------
+        Examples
+        --------
         >>> arr = xr.DataArray(data=np.ones((2, 3)),
         ...                    dims=['x', 'y'],
         ...                    coords={'x':
diff --git a/xarray/core/merge.py b/xarray/core/merge.py
--- a/xarray/core/merge.py
+++ b/xarray/core/merge.py
@@ -44,6 +44,7 @@
         "broadcast_equals": 2,
         "minimal": 3,
         "no_conflicts": 4,
+        "override": 5,
     }
 )
 
@@ -70,8 +71,8 @@ class MergeError(ValueError):
     # TODO: move this to an xarray.exceptions module?
 
 
-def unique_variable(name, variables, compat="broadcast_equals"):
-    # type: (Any, List[Variable], str) -> Variable
+def unique_variable(name, variables, compat="broadcast_equals", equals=None):
+    # type: (Any, List[Variable], str, bool) -> Variable
     """Return the unique variable from a list of variables or raise MergeError.
 
     Parameters
@@ -81,8 +82,10 @@ def unique_variable(name, variables, compat="broadcast_equals"):
     variables : list of xarray.Variable
         List of Variable objects, all of which go by the same name in different
         inputs.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         Type of equality check to use.
+    equals: None or bool,
+        corresponding to result of compat test
 
     Returns
     -------
@@ -93,30 +96,38 @@ def unique_variable(name, variables, compat="broadcast_equals"):
     MergeError: if any of the variables are not equal.
     """  # noqa
     out = variables[0]
-    if len(variables) > 1:
-        combine_method = None
 
-        if compat == "minimal":
-            compat = "broadcast_equals"
+    if len(variables) == 1 or compat == "override":
+        return out
+
+    combine_method = None
+
+    if compat == "minimal":
+        compat = "broadcast_equals"
+
+    if compat == "broadcast_equals":
+        dim_lengths = broadcast_dimension_size(variables)
+        out = out.set_dims(dim_lengths)
+
+    if compat == "no_conflicts":
+        combine_method = "fillna"
 
-        if compat == "broadcast_equals":
-            dim_lengths = broadcast_dimension_size(variables)
-            out = out.set_dims(dim_lengths)
+    if equals is None:
+        out = out.compute()
+        for var in variables[1:]:
+            equals = getattr(out, compat)(var)
+            if not equals:
+                break
 
-        if compat == "no_conflicts":
-            combine_method = "fillna"
+    if not equals:
+        raise MergeError(
+            "conflicting values for variable %r on objects to be combined. You can skip this check by specifying compat='override'."
+            % (name)
+        )
 
+    if combine_method:
         for var in variables[1:]:
-            if not getattr(out, compat)(var):
-                raise MergeError(
-                    "conflicting values for variable %r on "
-                    "objects to be combined:\n"
-                    "first value: %r\nsecond value: %r" % (name, out, var)
-                )
-            if combine_method:
-                # TODO: add preservation of attrs into fillna
-                out = getattr(out, combine_method)(var)
-                out.attrs = var.attrs
+            out = getattr(out, combine_method)(var)
 
     return out
 
@@ -152,7 +163,7 @@ def merge_variables(
     priority_vars : mapping with Variable or None values, optional
         If provided, variables are always taken from this dict in preference to
         the input variable dictionaries, without checking for conflicts.
-    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts', 'override'}, optional
         Type of equality check to use when checking for conflicts.
 
     Returns
@@ -449,7 +460,7 @@ def merge_core(
     ----------
     objs : list of mappings
         All values must be convertable to labeled arrays.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         Compatibility checks to use when merging variables.
     join : {'outer', 'inner', 'left', 'right'}, optional
         How to combine objects with different indexes.
@@ -519,7 +530,7 @@ def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
     objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
         Merge together all variables from these objects. If any of them are
         DataArray objects, they must have a name.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
 
@@ -531,6 +542,7 @@ def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
         String indicating how to combine differing indexes in objects.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/backends/api.py | 764 | 764 | 15 | 1 | 5856
| xarray/backends/api.py | 775 | 775 | 15 | 1 | 5856
| xarray/core/combine.py | 246 | 246 | 138 | 8 | 49572
| xarray/core/combine.py | 354 | 354 | - | 8 | -
| xarray/core/combine.py | 366 | 366 | - | 8 | -
| xarray/core/combine.py | 507 | 507 | 98 | 8 | 36724
| xarray/core/combine.py | 519 | 519 | 98 | 8 | 36724
| xarray/core/combine.py | 601 | 601 | 32 | 8 | 13897
| xarray/core/combine.py | 670 | 670 | 127 | 8 | 45896
| xarray/core/combine.py | 681 | 681 | 127 | 8 | 45896
| xarray/core/combine.py | 835 | 835 | 67 | 8 | 24473
| xarray/core/combine.py | 853 | 853 | 69 | 8 | 24902
| xarray/core/combine.py | 879 | 879 | 69 | 8 | 24902
| xarray/core/concat.py | 7 | 7 | - | 14 | -
| xarray/core/concat.py | 62 | 67 | - | 14 | -
| xarray/core/concat.py | 110 | 110 | 84 | 14 | 30859
| xarray/core/concat.py | 146 | 155 | 70 | 14 | 25443
| xarray/core/concat.py | 176 | 176 | 70 | 14 | 25443
| xarray/core/concat.py | 212 | 212 | 70 | 14 | 25443
| xarray/core/concat.py | 230 | 234 | 78 | 14 | 28909
| xarray/core/concat.py | 242 | 297 | - | 14 | -
| xarray/core/concat.py | 305 | 321 | - | 14 | -
| xarray/core/concat.py | 345 | 345 | - | 14 | -
| xarray/core/dataarray.py | 1552 | 1553 | - | 26 | -
| xarray/core/merge.py | 47 | 47 | - | 22 | -
| xarray/core/merge.py | 73 | 74 | - | 22 | -
| xarray/core/merge.py | 84 | 84 | - | 22 | -
| xarray/core/merge.py | 96 | 119 | - | 22 | -
| xarray/core/merge.py | 155 | 155 | - | 22 | -
| xarray/core/merge.py | 452 | 452 | - | 22 | -
| xarray/core/merge.py | 522 | 522 | - | 22 | -
| xarray/core/merge.py | 534 | 534 | - | 22 | -


## Problem Statement

```
We need a fast path for open_mfdataset
It would be great to have a "fast path" option for `open_mfdataset`, in which all alignment / coordinate checking is bypassed. This would be used in cases where the user knows that many netCDF files all share the same coordinates (e.g. model output, satellite records from the same product, etc.). The coordinates would just be taken from the first file, and only the data variables would be read from all subsequent files. The only checking would be that the data variables have the correct shape.

Implementing this would require some refactoring. @jbusecke mentioned that he had developed a solution for this (related to #1704), so maybe he could be the one to add this feature to xarray.

This is also related to #1385.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/backends/api.py** | 911 | 972| 521 | 521 | 10748 | 
| 2 | **1 xarray/backends/api.py** | 856 | 910| 560 | 1081 | 10748 | 
| 3 | 2 asv_bench/benchmarks/dataset_io.py | 315 | 328| 114 | 1195 | 14353 | 
| **-> 4 <-** | **2 xarray/backends/api.py** | 696 | 965| 133 | 1328 | 14353 | 
| 5 | 2 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 1770 | 14353 | 
| 6 | 2 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 2039 | 14353 | 
| 7 | 2 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 2151 | 14353 | 
| 8 | 2 asv_bench/benchmarks/dataset_io.py | 222 | 296| 613 | 2764 | 14353 | 
| 9 | 2 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 3113 | 14353 | 
| 10 | **2 xarray/backends/api.py** | 1192 | 1230| 287 | 3400 | 14353 | 
| 11 | 2 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 3664 | 14353 | 
| 12 | 2 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 3827 | 14353 | 
| 13 | 2 asv_bench/benchmarks/dataset_io.py | 299 | 312| 113 | 3940 | 14353 | 
| 14 | 3 xarray/core/common.py | 1059 | 1092| 302 | 4242 | 24928 | 
| **-> 15 <-** | **3 xarray/backends/api.py** | 723 | 856| 1614 | 5856 | 24928 | 
| 16 | 3 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 6121 | 24928 | 
| 17 | 3 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 6836 | 24928 | 
| 18 | 3 asv_bench/benchmarks/dataset_io.py | 435 | 470| 186 | 7022 | 24928 | 
| 19 | 4 xarray/core/coordinates.py | 221 | 251| 238 | 7260 | 27713 | 
| 20 | **4 xarray/backends/api.py** | 442 | 484| 354 | 7614 | 27713 | 
| 21 | 5 xarray/core/dataset.py | 4928 | 5086| 1656 | 9270 | 67965 | 
| 22 | 6 xarray/backends/netCDF4_.py | 341 | 389| 393 | 9663 | 71914 | 
| 23 | 6 xarray/backends/netCDF4_.py | 391 | 427| 356 | 10019 | 71914 | 
| 24 | **6 xarray/backends/api.py** | 972 | 1068| 756 | 10775 | 71914 | 
| 25 | **6 xarray/backends/api.py** | 486 | 542| 594 | 11369 | 71914 | 
| 26 | **6 xarray/backends/api.py** | 655 | 693| 334 | 11703 | 71914 | 
| 27 | **6 xarray/backends/api.py** | 1069 | 1094| 260 | 11963 | 71914 | 
| 28 | 7 xarray/backends/file_manager.py | 295 | 331| 206 | 12169 | 74455 | 
| 29 | 7 xarray/backends/netCDF4_.py | 429 | 465| 300 | 12469 | 74455 | 
| 30 | 7 xarray/backends/netCDF4_.py | 467 | 523| 450 | 12919 | 74455 | 
| 31 | **8 xarray/core/combine.py** | 711 | 776| 613 | 13532 | 81502 | 
| **-> 32 <-** | **8 xarray/core/combine.py** | 581 | 627| 365 | 13897 | 81502 | 
| 33 | **8 xarray/backends/api.py** | 400 | 440| 345 | 14242 | 81502 | 
| 34 | 8 xarray/core/dataset.py | 347 | 5082| 414 | 14656 | 81502 | 
| 35 | 9 xarray/backends/pynio_.py | 1 | 13| 128 | 14784 | 82109 | 
| 36 | 10 xarray/backends/h5netcdf_.py | 1 | 18| 115 | 14899 | 83967 | 
| 37 | 10 xarray/backends/netCDF4_.py | 1 | 29| 215 | 15114 | 83967 | 
| 38 | 11 xarray/backends/scipy_.py | 64 | 101| 310 | 15424 | 85704 | 
| 39 | 11 xarray/core/coordinates.py | 133 | 148| 136 | 15560 | 85704 | 
| 40 | 12 xarray/core/missing.py | 442 | 454| 142 | 15702 | 89978 | 
| 41 | 12 xarray/core/coordinates.py | 37 | 81| 288 | 15990 | 89978 | 
| 42 | 12 xarray/backends/h5netcdf_.py | 186 | 266| 607 | 16597 | 89978 | 
| 43 | 13 setup.py | 32 | 110| 736 | 17333 | 90987 | 
| 44 | 13 xarray/backends/netCDF4_.py | 151 | 177| 242 | 17575 | 90987 | 
| 45 | **13 xarray/core/combine.py** | 204 | 222| 195 | 17770 | 90987 | 
| 46 | **14 xarray/core/concat.py** | 298 | 311| 191 | 17961 | 93909 | 
| 47 | 15 xarray/conventions.py | 637 | 692| 478 | 18439 | 99508 | 
| 48 | 15 xarray/core/coordinates.py | 117 | 131| 139 | 18578 | 99508 | 
| 49 | 15 xarray/core/coordinates.py | 150 | 179| 240 | 18818 | 99508 | 
| 50 | 16 xarray/core/alignment.py | 150 | 207| 488 | 19306 | 104006 | 
| 51 | 17 xarray/tutorial.py | 28 | 97| 483 | 19789 | 104931 | 
| 52 | **17 xarray/core/concat.py** | 313 | 328| 168 | 19957 | 104931 | 
| 53 | 18 xarray/core/nputils.py | 231 | 242| 155 | 20112 | 106934 | 
| 54 | 18 xarray/core/alignment.py | 457 | 503| 352 | 20464 | 106934 | 
| 55 | **18 xarray/core/combine.py** | 779 | 811| 266 | 20730 | 106934 | 
| 56 | 18 xarray/backends/netCDF4_.py | 32 | 58| 207 | 20937 | 106934 | 
| 57 | 19 xarray/backends/pseudonetcdf_.py | 39 | 91| 356 | 21293 | 107555 | 
| 58 | 19 xarray/backends/h5netcdf_.py | 118 | 153| 315 | 21608 | 107555 | 
| 59 | 19 xarray/backends/h5netcdf_.py | 155 | 184| 206 | 21814 | 107555 | 
| 60 | 20 xarray/backends/common.py | 381 | 395| 119 | 21933 | 109986 | 
| 61 | 20 xarray/backends/common.py | 78 | 127| 305 | 22238 | 109986 | 
| 62 | 20 xarray/backends/netCDF4_.py | 134 | 148| 120 | 22358 | 109986 | 
| 63 | 20 xarray/backends/scipy_.py | 104 | 209| 736 | 23094 | 109986 | 
| 64 | **20 xarray/backends/api.py** | 1122 | 1190| 725 | 23819 | 109986 | 
| 65 | 20 xarray/backends/netCDF4_.py | 61 | 76| 131 | 23950 | 109986 | 
| 66 | 21 asv_bench/benchmarks/combine.py | 1 | 39| 332 | 24282 | 110318 | 
| **-> 67 <-** | **21 xarray/core/combine.py** | 814 | 843| 191 | 24473 | 110318 | 
| 68 | **22 xarray/core/merge.py** | 402 | 417| 134 | 24607 | 115160 | 
| **-> 69 <-** | **22 xarray/core/combine.py** | 846 | 881| 295 | 24902 | 115160 | 
| **-> 70 <-** | **22 xarray/core/concat.py** | 146 | 212| 541 | 25443 | 115160 | 
| 71 | 23 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 528 | 25971 | 115688 | 
| 72 | 24 xarray/backends/rasterio_.py | 324 | 370| 392 | 26363 | 118896 | 
| 73 | 24 xarray/core/alignment.py | 1 | 17| 122 | 26485 | 118896 | 
| 74 | 24 xarray/backends/pseudonetcdf_.py | 1 | 36| 265 | 26750 | 118896 | 
| 75 | 24 xarray/backends/common.py | 1 | 41| 221 | 26971 | 118896 | 
| 76 | 24 xarray/conventions.py | 695 | 714| 110 | 27081 | 118896 | 
| 77 | **24 xarray/backends/api.py** | 545 | 653| 1147 | 28228 | 118896 | 
| **-> 78 <-** | **24 xarray/core/concat.py** | 215 | 296| 681 | 28909 | 118896 | 
| 79 | 24 xarray/backends/file_manager.py | 1 | 18| 120 | 29029 | 118896 | 
| 80 | 24 xarray/backends/file_manager.py | 220 | 250| 322 | 29351 | 118896 | 
| 81 | 24 xarray/backends/rasterio_.py | 242 | 322| 858 | 30209 | 118896 | 
| 82 | 24 xarray/core/nputils.py | 1 | 41| 303 | 30512 | 118896 | 
| 83 | 25 xarray/backends/pydap_.py | 40 | 56| 133 | 30645 | 119526 | 
| **-> 84 <-** | **25 xarray/core/concat.py** | 99 | 119| 214 | 30859 | 119526 | 
| 85 | **25 xarray/backends/api.py** | 219 | 258| 290 | 31149 | 119526 | 
| 86 | **26 xarray/core/dataarray.py** | 743 | 773| 269 | 31418 | 143562 | 
| 87 | 27 xarray/core/computation.py | 1025 | 1062| 417 | 31835 | 153202 | 
| 88 | 27 xarray/core/alignment.py | 387 | 454| 642 | 32477 | 153202 | 
| 89 | 28 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 32890 | 153615 | 
| 90 | **28 xarray/core/dataarray.py** | 422 | 446| 205 | 33095 | 153615 | 
| 91 | **28 xarray/core/dataarray.py** | 2220 | 2247| 300 | 33395 | 153615 | 
| 92 | 29 xarray/backends/cfgrib_.py | 1 | 29| 210 | 33605 | 154105 | 
| 93 | **29 xarray/backends/api.py** | 287 | 399| 1198 | 34803 | 154105 | 
| 94 | 29 xarray/backends/netCDF4_.py | 298 | 339| 276 | 35079 | 154105 | 
| 95 | 29 xarray/backends/netCDF4_.py | 78 | 98| 195 | 35274 | 154105 | 
| 96 | **29 xarray/core/dataarray.py** | 448 | 472| 225 | 35499 | 154105 | 
| 97 | **29 xarray/core/dataarray.py** | 2473 | 2500| 226 | 35725 | 154105 | 
| **-> 98 <-** | **29 xarray/core/combine.py** | 466 | 579| 999 | 36724 | 154105 | 
| 99 | **29 xarray/backends/api.py** | 1 | 58| 303 | 37027 | 154105 | 
| 100 | **29 xarray/core/merge.py** | 601 | 640| 317 | 37344 | 154105 | 
| 101 | 30 xarray/core/options.py | 1 | 71| 487 | 37831 | 155263 | 
| 102 | **30 xarray/core/merge.py** | 643 | 669| 224 | 38055 | 155263 | 
| 103 | 30 xarray/backends/common.py | 129 | 168| 245 | 38300 | 155263 | 
| 104 | 31 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 38459 | 155422 | 
| 105 | 31 xarray/core/dataset.py | 1 | 112| 550 | 39009 | 155422 | 
| 106 | 32 asv_bench/benchmarks/indexing.py | 60 | 74| 151 | 39160 | 156738 | 
| 107 | 32 xarray/core/nputils.py | 205 | 228| 187 | 39347 | 156738 | 
| 108 | **32 xarray/core/dataarray.py** | 371 | 403| 298 | 39645 | 156738 | 
| 109 | 32 xarray/backends/h5netcdf_.py | 21 | 38| 169 | 39814 | 156738 | 
| 110 | 32 xarray/backends/scipy_.py | 211 | 241| 222 | 40036 | 156738 | 
| 111 | 32 xarray/backends/pydap_.py | 59 | 95| 225 | 40261 | 156738 | 
| 112 | 32 xarray/core/common.py | 77 | 101| 145 | 40406 | 156738 | 
| 113 | **32 xarray/core/merge.py** | 364 | 399| 255 | 40661 | 156738 | 
| 114 | 32 xarray/backends/h5netcdf_.py | 41 | 66| 176 | 40837 | 156738 | 
| 115 | **32 xarray/core/dataarray.py** | 2502 | 2521| 197 | 41034 | 156738 | 
| 116 | 32 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 41764 | 156738 | 
| 117 | 32 xarray/tutorial.py | 100 | 138| 309 | 42073 | 156738 | 
| 118 | 33 xarray/__init__.py | 1 | 44| 286 | 42359 | 157025 | 
| 119 | 33 xarray/backends/netCDF4_.py | 180 | 200| 215 | 42574 | 157025 | 
| 120 | **33 xarray/core/dataarray.py** | 1 | 78| 379 | 42953 | 157025 | 
| 121 | 34 xarray/core/npcompat.py | 136 | 220| 724 | 43677 | 160435 | 
| 122 | **34 xarray/core/dataarray.py** | 166 | 181| 132 | 43809 | 160435 | 
| 123 | **34 xarray/core/combine.py** | 266 | 308| 255 | 44064 | 160435 | 
| 124 | 34 xarray/core/computation.py | 359 | 421| 471 | 44535 | 160435 | 
| 125 | 34 xarray/core/alignment.py | 210 | 280| 445 | 44980 | 160435 | 
| 126 | 34 xarray/core/computation.py | 756 | 1062| 147 | 45127 | 160435 | 
| **-> 127 <-** | **34 xarray/core/combine.py** | 630 | 709| 769 | 45896 | 160435 | 
| 128 | 34 xarray/backends/h5netcdf_.py | 69 | 116| 306 | 46202 | 160435 | 
| 129 | 35 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 46386 | 160920 | 
| 130 | 36 xarray/core/groupby.py | 924 | 961| 225 | 46611 | 168063 | 
| 131 | 36 xarray/core/coordinates.py | 1 | 34| 168 | 46779 | 168063 | 
| 132 | 37 xarray/plot/dataset_plot.py | 245 | 351| 770 | 47549 | 171490 | 
| 133 | 37 xarray/plot/dataset_plot.py | 404 | 454| 390 | 47939 | 171490 | 
| 134 | 37 xarray/core/alignment.py | 40 | 58| 152 | 48091 | 171490 | 
| 135 | 37 xarray/core/computation.py | 325 | 356| 242 | 48333 | 171490 | 
| 136 | 37 xarray/core/alignment.py | 506 | 589| 708 | 49041 | 171490 | 
| 137 | 38 xarray/coding/cftime_offsets.py | 307 | 346| 298 | 49339 | 179821 | 
| **-> 138 <-** | **38 xarray/core/combine.py** | 225 | 263| 233 | 49572 | 179821 | 
| 139 | 39 xarray/core/variable.py | 2108 | 2131| 168 | 49740 | 197361 | 
| 140 | 40 xarray/ufuncs.py | 1 | 35| 247 | 49987 | 198245 | 
| 141 | 41 xarray/backends/netcdf3.py | 1 | 34| 225 | 50212 | 199223 | 
| 142 | 41 xarray/core/computation.py | 972 | 1023| 525 | 50737 | 199223 | 


### Hint

```
@rabernat - Depending on the structure of the dataset, another possibility that would speed up some `open_mfdataset` tasks substantially is to implement the step of opening each file and getting its metadata in in some parallel way (dask/joblib/etc.) and either returning the just dataset schema or a picklable version of the dataset itself.  I think this will only be able to work with `autoclose=True` but it could be quite useful when working with many files. 
I did not really find an elegant solution. What I did was just specify all dims and coords as `drop_variables` and then update those from a master file with 
\`\`\`
ds.update(ds_master)
\`\`\` 
Perhaps this could be generalized in a sense, by reading all coords and dims just from the first file. 
Would these two options be necessarily mutually exclusive?

I think parallelizing the read in sounds amazing.  

But isnt there some merit in skipping some of the checks all together, if the user is sure about the structure of the data contained in the many files?

I am often working with the aforementioned type of data (many files either contain a new timestep or a different variable, but most of the dimensions/coordinates are the same). 

In some cases I am finding that reading the data "lazily" consumes a significant amount of the time in my workflow. I am unsure how hard this would be to achieve, and perhaps it is not worth it after all.

Just putting out a few ideas, while I wait for my `xr.open_mfdataset` to finish :-)
@jbusecke - No. These options are not mutually exclusive. The parallel open is, in my opinion, the lowest hanging fruit so that's why I started there. There are other improvements that we can tackle incrementally. 
Awesome, thanks for the clarification.
I just looked at #1981 and it seems indeed very elegant (in fact I just now used this approach to parallelize printing of movie frames!) Thanks for that!

I am currently motivated to fix this.

1. Over in https://github.com/pydata/xarray/pull/1413#issuecomment-302843502 @rabernat mentioned
> allowing the user to pass join='exact' via open_mfdataset. A related optimization would be to allow the user to pass coords='minimal' (or other concat coords options) via open_mfdataset.

2. @shoyer suggested calling decode_cf later here though perhaps this wont help too much: https://github.com/pydata/xarray/issues/1385#issuecomment-439263419

Is this all that we can do on the xarray side?
@dcherian I'm sorry, I'm very interested in this but after reading the issues I'm still not clear on what's being proposed:

What exactly is the bottleneck? Is it reading the coords from all the files? Is it loading the coord values into memory? Is it performing the alignment checks on those coords once they're in memory? Is it performing alignment checks on the dimensions? Is this suggestion relevant to datasets that don't have any coords?

Which of these steps would a `join='exact'` option omit?

> A related optimization would be to allow the user to pass coords='minimal' (or other concat coords options) via open_mfdataset.

But this is already an option to `open_mfdataset`?
The original issue of this thread is that you sometimes might want to *disable* alignment checks for coordinates other than the `concat_dim` and only check for same dimensions and dimension shapes. 

When you `xr.merge` with `join='exact'`, it still checks for alignment (see https://github.com/pydata/xarray/pull/1330#issuecomment-302711852), but does not join the coordinates if they are not aligned. This behavior (not joining) is also included in what @rabernat envisioned here, but his suggestion goes beyond that: you don't even load coordinate values from all but the first dataset and just blindly trust that they are aligned.

So `xr.open_mfdataset(join='exact', coords='minimal')` does not fix this issue here, I think.
So I think it is quite important to consider this issue together with #2697. An xml specification called [NCML](https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/ncml/) already exists which tells software how to put together multiple netCDF files into a single virtual netcdf. We should leverage this existing spec as much as possible.

A realistic use case for me is that I have, say 1000 files of high-res model output, each with large coordinate variables, all generated from the same model run. If we want to  for for which we *know a priori* that certain coordinates (dimension coordinates or otherwise) are identical, we could save a lot of disk reads (the slow part of `open_mfdataset`) by never reading those coordinates at all. Enabling this would require a pretty low-level change in xarray. For example, we couldn't even rely on `open_dataset` in its current form to open files, because `open_dataset` eagerly loads all dimension coordinates into indexes. One way forward might be to create a new Store class.

For a catalog of tricks I use to optimize opening these sorts of big, complex, multi-file datasets (e.g. CMIP), check out
https://github.com/pangeo-data/esgf2xarray/blob/master/esgf2zarr/aggregate.py

One common use-case is files with large numbers of `concat_dim`-invariant non-dimensional co-ordinates.  This is easy to speed up by dropping those variables from all but the first file. 

e.g.
https://github.com/pangeo-data/esgf2xarray/blob/6a5e4df0d329c2f23b403cbfbb65f0f1dfa98d52/esgf2zarr/aggregate.py#L107-L110
\`\`\` python
    # keep only coordinates from first ensemble member to simplify merge
    first = member_dsets_aligned[0]
    rest = [mds.reset_coords(drop=True) for mds in member_dsets_aligned[1:]]
    objs_to_concat = [first] + rest
\`\`\`

Similarly https://github.com/NCAR/intake-esm/blob/e86a8e8a80ce0fd4198665dbef3ba46af264b5ea/intake_esm/aggregate.py#L53-L57

\`\`\` python
def merge_vars_two_datasets(ds1, ds2):
    """
    Merge two datasets, dropping all variables from
    second dataset that already exist in the first dataset's coordinates.
    """
\`\`\`

See also #2039 (second code block)

One way to do this might be to add a `master_file` kwarg to `open_mfdataset`. This would imply `coords='minimal', join='exact'` (I think; `prealigned=True` in some other proposals) and would drop non-dimensional coordinates from all but the first file and then call concat. 

As bonus it would assign attributes from the `master_file` to the merged dataset (for which I think there are open issues) : this functionality exists in `netCDF4.MFDataset` so that's a plus.

EDIT: #2039 (third code block) is also a possibility. This might look like
\`\`\` python
xr.open_mfdataset('files*.nc', master_file='first', concat_dim='time')
\`\`\`
in which case the first file is read; all coords that are not `concat_dim` become `drop_variables` for an `open_dataset` call that reads the remaining files. We then merge with the first dataset and assign attrs.

EDIT2: `master_file` combines two different functionalities here: specifying a "template file" and a file to choose attributes from. So maybe we need two kwargs: `template_file` and `attrs_from`?
```

## Patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -761,7 +761,7 @@ def open_mfdataset(
         `xarray.auto_combine` is used, but in the future this behavior will 
         switch to use `xarray.combine_by_coords` by default.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts when merging:
          * 'broadcast_equals': all values must be equal when variables are
@@ -772,6 +772,7 @@ def open_mfdataset(
          * 'no_conflicts': only values which are not null in both datasets
            must be equal. The returned dataset then contains the combination
            of all non-null values.
+         * 'override': skip comparing and pick variable from first dataset
     preprocess : callable, optional
         If provided, call this function on each dataset prior to concatenation.
         You can find the file-name from which each dataset was loaded in
diff --git a/xarray/core/combine.py b/xarray/core/combine.py
--- a/xarray/core/combine.py
+++ b/xarray/core/combine.py
@@ -243,6 +243,7 @@ def _combine_1d(
                 dim=concat_dim,
                 data_vars=data_vars,
                 coords=coords,
+                compat=compat,
                 fill_value=fill_value,
                 join=join,
             )
@@ -351,7 +352,7 @@ def combine_nested(
         Must be the same length as the depth of the list passed to
         ``datasets``.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential merge conflicts:
 
@@ -363,6 +364,7 @@ def combine_nested(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' or list of str}, optional
@@ -504,7 +506,7 @@ def combine_by_coords(
     datasets : sequence of xarray.Dataset
         Dataset objects to combine.
     compat : {'identical', 'equals', 'broadcast_equals',
-              'no_conflicts'}, optional
+              'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
 
@@ -516,6 +518,7 @@ def combine_by_coords(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' or list of str}, optional
@@ -598,6 +601,7 @@ def combine_by_coords(
             concat_dims=concat_dims,
             data_vars=data_vars,
             coords=coords,
+            compat=compat,
             fill_value=fill_value,
             join=join,
         )
@@ -667,7 +671,7 @@ def auto_combine(
         component files. Set ``concat_dim=None`` explicitly to disable
         concatenation.
     compat : {'identical', 'equals', 'broadcast_equals',
-             'no_conflicts'}, optional
+             'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
         - 'broadcast_equals': all values must be equal when variables are
@@ -678,6 +682,7 @@ def auto_combine(
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     data_vars : {'minimal', 'different', 'all' or list of str}, optional
         Details are in the documentation of concat
     coords : {'minimal', 'different', 'all' o list of str}, optional
@@ -832,6 +837,7 @@ def _old_auto_combine(
                 dim=dim,
                 data_vars=data_vars,
                 coords=coords,
+                compat=compat,
                 fill_value=fill_value,
                 join=join,
             )
@@ -850,6 +856,7 @@ def _auto_concat(
     coords="different",
     fill_value=dtypes.NA,
     join="outer",
+    compat="no_conflicts",
 ):
     if len(datasets) == 1 and dim is None:
         # There is nothing more to combine, so kick out early.
@@ -876,5 +883,10 @@ def _auto_concat(
                 )
             dim, = concat_dims
         return concat(
-            datasets, dim=dim, data_vars=data_vars, coords=coords, fill_value=fill_value
+            datasets,
+            dim=dim,
+            data_vars=data_vars,
+            coords=coords,
+            fill_value=fill_value,
+            compat=compat,
         )
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -4,6 +4,7 @@
 
 from . import dtypes, utils
 from .alignment import align
+from .merge import unique_variable, _VALID_COMPAT
 from .variable import IndexVariable, Variable, as_variable
 from .variable import concat as concat_vars
 
@@ -59,12 +60,19 @@ def concat(
             those corresponding to other dimensions.
           * list of str: The listed coordinate variables will be concatenated,
             in addition to the 'minimal' coordinates.
-    compat : {'equals', 'identical'}, optional
-        String indicating how to compare non-concatenated variables and
-        dataset global attributes for potential conflicts. 'equals' means
-        that all variable values and dimensions must be the same;
-        'identical' means that variable attributes and global attributes
-        must also be equal.
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
+        String indicating how to compare non-concatenated variables of the same name for
+        potential conflicts. This is passed down to merge.
+
+        - 'broadcast_equals': all values must be equal when variables are
+          broadcast against each other to ensure common dimensions.
+        - 'equals': all values and dimensions must be the same.
+        - 'identical': all values, dimensions and attributes must be the
+          same.
+        - 'no_conflicts': only values which are not null in both datasets
+          must be equal. The returned dataset then contains the combination
+          of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     positions : None or list of integer arrays, optional
         List of integer arrays which specifies the integer positions to which
         to assign each dataset along the concatenated dimension. If not
@@ -107,6 +115,12 @@ def concat(
     except StopIteration:
         raise ValueError("must supply at least one object to concatenate")
 
+    if compat not in _VALID_COMPAT:
+        raise ValueError(
+            "compat=%r invalid: must be 'broadcast_equals', 'equals', 'identical', 'no_conflicts' or 'override'"
+            % compat
+        )
+
     if isinstance(first_obj, DataArray):
         f = _dataarray_concat
     elif isinstance(first_obj, Dataset):
@@ -143,23 +157,39 @@ def _calc_concat_dim_coord(dim):
     return dim, coord
 
 
-def _calc_concat_over(datasets, dim, data_vars, coords):
+def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
     """
     Determine which dataset variables need to be concatenated in the result,
-    and which can simply be taken from the first dataset.
     """
     # Return values
     concat_over = set()
     equals = {}
 
-    if dim in datasets[0]:
+    if dim in dim_names:
+        concat_over_existing_dim = True
         concat_over.add(dim)
+    else:
+        concat_over_existing_dim = False
+
+    concat_dim_lengths = []
     for ds in datasets:
+        if concat_over_existing_dim:
+            if dim not in ds.dims:
+                if dim in ds:
+                    ds = ds.set_coords(dim)
+                else:
+                    raise ValueError("%r is not present in all datasets" % dim)
         concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
+        concat_dim_lengths.append(ds.dims.get(dim, 1))
 
     def process_subset_opt(opt, subset):
         if isinstance(opt, str):
             if opt == "different":
+                if compat == "override":
+                    raise ValueError(
+                        "Cannot specify both %s='different' and compat='override'."
+                        % subset
+                    )
                 # all nonindexes that are not the same in each dataset
                 for k in getattr(datasets[0], subset):
                     if k not in concat_over:
@@ -173,7 +203,7 @@ def process_subset_opt(opt, subset):
                         for ds_rhs in datasets[1:]:
                             v_rhs = ds_rhs.variables[k].compute()
                             computed.append(v_rhs)
-                            if not v_lhs.equals(v_rhs):
+                            if not getattr(v_lhs, compat)(v_rhs):
                                 concat_over.add(k)
                                 equals[k] = False
                                 # computed variables are not to be re-computed
@@ -209,7 +239,29 @@ def process_subset_opt(opt, subset):
 
     process_subset_opt(data_vars, "data_vars")
     process_subset_opt(coords, "coords")
-    return concat_over, equals
+    return concat_over, equals, concat_dim_lengths
+
+
+# determine dimensional coordinate names and a dict mapping name to DataArray
+def _parse_datasets(datasets):
+
+    dims = set()
+    all_coord_names = set()
+    data_vars = set()  # list of data_vars
+    dim_coords = dict()  # maps dim name to variable
+    dims_sizes = {}  # shared dimension sizes to expand variables
+
+    for ds in datasets:
+        dims_sizes.update(ds.dims)
+        all_coord_names.update(ds.coords)
+        data_vars.update(ds.data_vars)
+
+        for dim in set(ds.dims) - dims:
+            if dim not in dim_coords:
+                dim_coords[dim] = ds.coords[dim].variable
+        dims = dims | set(ds.dims)
+
+    return dim_coords, dims_sizes, all_coord_names, data_vars
 
 
 def _dataset_concat(
@@ -227,11 +279,6 @@ def _dataset_concat(
     """
     from .dataset import Dataset
 
-    if compat not in ["equals", "identical"]:
-        raise ValueError(
-            "compat=%r invalid: must be 'equals' " "or 'identical'" % compat
-        )
-
     dim, coord = _calc_concat_dim_coord(dim)
     # Make sure we're working on a copy (we'll be loading variables)
     datasets = [ds.copy() for ds in datasets]
@@ -239,62 +286,65 @@ def _dataset_concat(
         *datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value
     )
 
-    concat_over, equals = _calc_concat_over(datasets, dim, data_vars, coords)
+    dim_coords, dims_sizes, coord_names, data_names = _parse_datasets(datasets)
+    dim_names = set(dim_coords)
+    unlabeled_dims = dim_names - coord_names
+
+    both_data_and_coords = coord_names & data_names
+    if both_data_and_coords:
+        raise ValueError(
+            "%r is a coordinate in some datasets but not others." % both_data_and_coords
+        )
+    # we don't want the concat dimension in the result dataset yet
+    dim_coords.pop(dim, None)
+    dims_sizes.pop(dim, None)
+
+    # case where concat dimension is a coordinate or data_var but not a dimension
+    if (dim in coord_names or dim in data_names) and dim not in dim_names:
+        datasets = [ds.expand_dims(dim) for ds in datasets]
+
+    # determine which variables to concatentate
+    concat_over, equals, concat_dim_lengths = _calc_concat_over(
+        datasets, dim, dim_names, data_vars, coords, compat
+    )
+
+    # determine which variables to merge, and then merge them according to compat
+    variables_to_merge = (coord_names | data_names) - concat_over - dim_names
+
+    result_vars = {}
+    if variables_to_merge:
+        to_merge = {var: [] for var in variables_to_merge}
+
+        for ds in datasets:
+            absent_merge_vars = variables_to_merge - set(ds.variables)
+            if absent_merge_vars:
+                raise ValueError(
+                    "variables %r are present in some datasets but not others. "
+                    % absent_merge_vars
+                )
 
-    def insert_result_variable(k, v):
-        assert isinstance(v, Variable)
-        if k in datasets[0].coords:
-            result_coord_names.add(k)
-        result_vars[k] = v
+            for var in variables_to_merge:
+                to_merge[var].append(ds.variables[var])
 
-    # create the new dataset and add constant variables
-    result_vars = OrderedDict()
-    result_coord_names = set(datasets[0].coords)
+        for var in variables_to_merge:
+            result_vars[var] = unique_variable(
+                var, to_merge[var], compat=compat, equals=equals.get(var, None)
+            )
+    else:
+        result_vars = OrderedDict()
+    result_vars.update(dim_coords)
+
+    # assign attrs and encoding from first dataset
     result_attrs = datasets[0].attrs
     result_encoding = datasets[0].encoding
 
-    for k, v in datasets[0].variables.items():
-        if k not in concat_over:
-            insert_result_variable(k, v)
-
-    # check that global attributes and non-concatenated variables are fixed
-    # across all datasets
+    # check that global attributes are fixed across all datasets if necessary
     for ds in datasets[1:]:
         if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
-            raise ValueError("dataset global attributes not equal")
-        for k, v in ds.variables.items():
-            if k not in result_vars and k not in concat_over:
-                raise ValueError("encountered unexpected variable %r" % k)
-            elif (k in result_coord_names) != (k in ds.coords):
-                raise ValueError(
-                    "%r is a coordinate in some datasets but not " "others" % k
-                )
-            elif k in result_vars and k != dim:
-                # Don't use Variable.identical as it internally invokes
-                # Variable.equals, and we may already know the answer
-                if compat == "identical" and not utils.dict_equiv(
-                    v.attrs, result_vars[k].attrs
-                ):
-                    raise ValueError("variable %s not identical across datasets" % k)
-
-                # Proceed with equals()
-                try:
-                    # May be populated when using the "different" method
-                    is_equal = equals[k]
-                except KeyError:
-                    result_vars[k].load()
-                    is_equal = v.equals(result_vars[k])
-                if not is_equal:
-                    raise ValueError("variable %s not equal across datasets" % k)
+            raise ValueError("Dataset global attributes not equal.")
 
     # we've already verified everything is consistent; now, calculate
     # shared dimension sizes so we can expand the necessary variables
-    dim_lengths = [ds.dims.get(dim, 1) for ds in datasets]
-    non_concat_dims = {}
-    for ds in datasets:
-        non_concat_dims.update(ds.dims)
-    non_concat_dims.pop(dim, None)
-
     def ensure_common_dims(vars):
         # ensure each variable with the given name shares the same
         # dimensions and the same shape for all of them except along the
@@ -302,25 +352,27 @@ def ensure_common_dims(vars):
         common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
         if dim not in common_dims:
             common_dims = (dim,) + common_dims
-        for var, dim_len in zip(vars, dim_lengths):
+        for var, dim_len in zip(vars, concat_dim_lengths):
             if var.dims != common_dims:
-                common_shape = tuple(
-                    non_concat_dims.get(d, dim_len) for d in common_dims
-                )
+                common_shape = tuple(dims_sizes.get(d, dim_len) for d in common_dims)
                 var = var.set_dims(common_dims, common_shape)
             yield var
 
     # stack up each variable to fill-out the dataset (in order)
+    # n.b. this loop preserves variable order, needed for groupby.
     for k in datasets[0].variables:
         if k in concat_over:
             vars = ensure_common_dims([ds.variables[k] for ds in datasets])
             combined = concat_vars(vars, dim, positions)
-            insert_result_variable(k, combined)
+            assert isinstance(combined, Variable)
+            result_vars[k] = combined
 
     result = Dataset(result_vars, attrs=result_attrs)
-    result = result.set_coords(result_coord_names)
+    result = result.set_coords(coord_names)
     result.encoding = result_encoding
 
+    result = result.drop(unlabeled_dims, errors="ignore")
+
     if coord is not None:
         # add concat dimension last to ensure that its in the final Dataset
         result[coord.name] = coord
@@ -342,7 +394,7 @@ def _dataarray_concat(
 
     if data_vars != "all":
         raise ValueError(
-            "data_vars is not a valid argument when " "concatenating DataArray objects"
+            "data_vars is not a valid argument when concatenating DataArray objects"
         )
 
     datasets = []
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -1549,8 +1549,8 @@ def set_index(
         obj : DataArray
             Another DataArray, with this data but replaced coordinates.
 
-        Example
-        -------
+        Examples
+        --------
         >>> arr = xr.DataArray(data=np.ones((2, 3)),
         ...                    dims=['x', 'y'],
         ...                    coords={'x':
diff --git a/xarray/core/merge.py b/xarray/core/merge.py
--- a/xarray/core/merge.py
+++ b/xarray/core/merge.py
@@ -44,6 +44,7 @@
         "broadcast_equals": 2,
         "minimal": 3,
         "no_conflicts": 4,
+        "override": 5,
     }
 )
 
@@ -70,8 +71,8 @@ class MergeError(ValueError):
     # TODO: move this to an xarray.exceptions module?
 
 
-def unique_variable(name, variables, compat="broadcast_equals"):
-    # type: (Any, List[Variable], str) -> Variable
+def unique_variable(name, variables, compat="broadcast_equals", equals=None):
+    # type: (Any, List[Variable], str, bool) -> Variable
     """Return the unique variable from a list of variables or raise MergeError.
 
     Parameters
@@ -81,8 +82,10 @@ def unique_variable(name, variables, compat="broadcast_equals"):
     variables : list of xarray.Variable
         List of Variable objects, all of which go by the same name in different
         inputs.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         Type of equality check to use.
+    equals: None or bool,
+        corresponding to result of compat test
 
     Returns
     -------
@@ -93,30 +96,38 @@ def unique_variable(name, variables, compat="broadcast_equals"):
     MergeError: if any of the variables are not equal.
     """  # noqa
     out = variables[0]
-    if len(variables) > 1:
-        combine_method = None
 
-        if compat == "minimal":
-            compat = "broadcast_equals"
+    if len(variables) == 1 or compat == "override":
+        return out
+
+    combine_method = None
+
+    if compat == "minimal":
+        compat = "broadcast_equals"
+
+    if compat == "broadcast_equals":
+        dim_lengths = broadcast_dimension_size(variables)
+        out = out.set_dims(dim_lengths)
+
+    if compat == "no_conflicts":
+        combine_method = "fillna"
 
-        if compat == "broadcast_equals":
-            dim_lengths = broadcast_dimension_size(variables)
-            out = out.set_dims(dim_lengths)
+    if equals is None:
+        out = out.compute()
+        for var in variables[1:]:
+            equals = getattr(out, compat)(var)
+            if not equals:
+                break
 
-        if compat == "no_conflicts":
-            combine_method = "fillna"
+    if not equals:
+        raise MergeError(
+            "conflicting values for variable %r on objects to be combined. You can skip this check by specifying compat='override'."
+            % (name)
+        )
 
+    if combine_method:
         for var in variables[1:]:
-            if not getattr(out, compat)(var):
-                raise MergeError(
-                    "conflicting values for variable %r on "
-                    "objects to be combined:\n"
-                    "first value: %r\nsecond value: %r" % (name, out, var)
-                )
-            if combine_method:
-                # TODO: add preservation of attrs into fillna
-                out = getattr(out, combine_method)(var)
-                out.attrs = var.attrs
+            out = getattr(out, combine_method)(var)
 
     return out
 
@@ -152,7 +163,7 @@ def merge_variables(
     priority_vars : mapping with Variable or None values, optional
         If provided, variables are always taken from this dict in preference to
         the input variable dictionaries, without checking for conflicts.
-    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts', 'override'}, optional
         Type of equality check to use when checking for conflicts.
 
     Returns
@@ -449,7 +460,7 @@ def merge_core(
     ----------
     objs : list of mappings
         All values must be convertable to labeled arrays.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         Compatibility checks to use when merging variables.
     join : {'outer', 'inner', 'left', 'right'}, optional
         How to combine objects with different indexes.
@@ -519,7 +530,7 @@ def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
     objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
         Merge together all variables from these objects. If any of them are
         DataArray objects, they must have a name.
-    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
+    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
         String indicating how to compare variables of the same name for
         potential conflicts:
 
@@ -531,6 +542,7 @@ def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
         - 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
+        - 'override': skip comparing and pick variable from first dataset
     join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
         String indicating how to combine differing indexes in objects.
 

```

## Test Patch

```diff
diff --git a/xarray/tests/test_combine.py b/xarray/tests/test_combine.py
--- a/xarray/tests/test_combine.py
+++ b/xarray/tests/test_combine.py
@@ -327,13 +327,13 @@ class TestCheckShapeTileIDs:
     def test_check_depths(self):
         ds = create_test_data(0)
         combined_tile_ids = {(0,): ds, (0, 1): ds}
-        with raises_regex(ValueError, "sub-lists do not have " "consistent depths"):
+        with raises_regex(ValueError, "sub-lists do not have consistent depths"):
             _check_shape_tile_ids(combined_tile_ids)
 
     def test_check_lengths(self):
         ds = create_test_data(0)
         combined_tile_ids = {(0, 0): ds, (0, 1): ds, (0, 2): ds, (1, 0): ds, (1, 1): ds}
-        with raises_regex(ValueError, "sub-lists do not have " "consistent lengths"):
+        with raises_regex(ValueError, "sub-lists do not have consistent lengths"):
             _check_shape_tile_ids(combined_tile_ids)
 
 
@@ -565,11 +565,6 @@ def test_combine_concat_over_redundant_nesting(self):
         expected = Dataset({"x": [0]})
         assert_identical(expected, actual)
 
-    def test_combine_nested_but_need_auto_combine(self):
-        objs = [Dataset({"x": [0, 1]}), Dataset({"x": [2], "wall": [0]})]
-        with raises_regex(ValueError, "cannot be combined"):
-            combine_nested(objs, concat_dim="x")
-
     @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0])
     def test_combine_nested_fill_value(self, fill_value):
         datasets = [
@@ -618,7 +613,7 @@ def test_combine_by_coords(self):
         assert_equal(actual, expected)
 
         objs = [Dataset({"x": 0}), Dataset({"x": 1})]
-        with raises_regex(ValueError, "Could not find any dimension " "coordinates"):
+        with raises_regex(ValueError, "Could not find any dimension coordinates"):
             combine_by_coords(objs)
 
         objs = [Dataset({"x": [0], "y": [0]}), Dataset({"x": [0]})]
@@ -761,7 +756,7 @@ def test_auto_combine(self):
             auto_combine(objs)
 
         objs = [Dataset({"x": [0], "y": [0]}), Dataset({"x": [0]})]
-        with pytest.raises(KeyError):
+        with raises_regex(ValueError, "'y' is not present in all datasets"):
             auto_combine(objs)
 
     def test_auto_combine_previously_failed(self):
diff --git a/xarray/tests/test_concat.py b/xarray/tests/test_concat.py
--- a/xarray/tests/test_concat.py
+++ b/xarray/tests/test_concat.py
@@ -5,8 +5,7 @@
 import pytest
 
 from xarray import DataArray, Dataset, Variable, concat
-from xarray.core import dtypes
-
+from xarray.core import dtypes, merge
 from . import (
     InaccessibleArray,
     assert_array_equal,
@@ -18,6 +17,34 @@
 from .test_dataset import create_test_data
 
 
+def test_concat_compat():
+    ds1 = Dataset(
+        {
+            "has_x_y": (("y", "x"), [[1, 2]]),
+            "has_x": ("x", [1, 2]),
+            "no_x_y": ("z", [1, 2]),
+        },
+        coords={"x": [0, 1], "y": [0], "z": [-1, -2]},
+    )
+    ds2 = Dataset(
+        {
+            "has_x_y": (("y", "x"), [[3, 4]]),
+            "has_x": ("x", [1, 2]),
+            "no_x_y": (("q", "z"), [[1, 2]]),
+        },
+        coords={"x": [0, 1], "y": [1], "z": [-1, -2], "q": [0]},
+    )
+
+    result = concat([ds1, ds2], dim="y", data_vars="minimal", compat="broadcast_equals")
+    assert_equal(ds2.no_x_y, result.no_x_y.transpose())
+
+    for var in ["has_x", "no_x_y"]:
+        assert "y" not in result[var]
+
+    with raises_regex(ValueError, "'q' is not present in all datasets"):
+        concat([ds1, ds2], dim="q", data_vars="all", compat="broadcast_equals")
+
+
 class TestConcatDataset:
     @pytest.fixture
     def data(self):
@@ -92,7 +119,7 @@ def test_concat_coords(self):
             actual = concat(objs, dim="x", coords=coords)
             assert_identical(expected, actual)
         for coords in ["minimal", []]:
-            with raises_regex(ValueError, "not equal across"):
+            with raises_regex(merge.MergeError, "conflicting values"):
                 concat(objs, dim="x", coords=coords)
 
     def test_concat_constant_index(self):
@@ -103,8 +130,10 @@ def test_concat_constant_index(self):
         for mode in ["different", "all", ["foo"]]:
             actual = concat([ds1, ds2], "y", data_vars=mode)
             assert_identical(expected, actual)
-        with raises_regex(ValueError, "not equal across datasets"):
-            concat([ds1, ds2], "y", data_vars="minimal")
+        with raises_regex(merge.MergeError, "conflicting values"):
+            # previously dim="y", and raised error which makes no sense.
+            # "foo" has dimension "y" so minimal should concatenate it?
+            concat([ds1, ds2], "new_dim", data_vars="minimal")
 
     def test_concat_size0(self):
         data = create_test_data()
@@ -134,6 +163,14 @@ def test_concat_errors(self):
         data = create_test_data()
         split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
 
+        with raises_regex(ValueError, "must supply at least one"):
+            concat([], "dim1")
+
+        with raises_regex(ValueError, "Cannot specify both .*='different'"):
+            concat(
+                [data, data], dim="concat_dim", data_vars="different", compat="override"
+            )
+
         with raises_regex(ValueError, "must supply at least one"):
             concat([], "dim1")
 
@@ -146,7 +183,7 @@ def test_concat_errors(self):
             concat([data0, data1], "dim1", compat="identical")
         assert_identical(data, concat([data0, data1], "dim1", compat="equals"))
 
-        with raises_regex(ValueError, "encountered unexpected"):
+        with raises_regex(ValueError, "present in some datasets"):
             data0, data1 = deepcopy(split_data)
             data1["foo"] = ("bar", np.random.randn(10))
             concat([data0, data1], "dim1")
diff --git a/xarray/tests/test_dask.py b/xarray/tests/test_dask.py
--- a/xarray/tests/test_dask.py
+++ b/xarray/tests/test_dask.py
@@ -825,7 +825,6 @@ def kernel(name):
     """Dask kernel to test pickling/unpickling and __repr__.
     Must be global to make it pickleable.
     """
-    print("kernel(%s)" % name)
     global kernel_call_count
     kernel_call_count += 1
     return np.ones(1, dtype=np.int64)
diff --git a/xarray/tests/test_merge.py b/xarray/tests/test_merge.py
--- a/xarray/tests/test_merge.py
+++ b/xarray/tests/test_merge.py
@@ -196,6 +196,8 @@ def test_merge_compat(self):
         with raises_regex(ValueError, "compat=.* invalid"):
             ds1.merge(ds2, compat="foobar")
 
+        assert ds1.identical(ds1.merge(ds2, compat="override"))
+
     def test_merge_auto_align(self):
         ds1 = xr.Dataset({"a": ("x", [1, 2]), "x": [0, 1]})
         ds2 = xr.Dataset({"b": ("x", [3, 4]), "x": [1, 2]})

```


## Code snippets

### 1 - xarray/backends/api.py:

Start line: 911, End line: 972

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="different",
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    **kwargs
):
    # ... other code
    try:
        if combine == "_old_auto":
            # Use the old auto_combine for now
            # Remove this after deprecation cycle from #2616 is complete
            basic_msg = dedent(
                """\
            In xarray version 0.13 the default behaviour of `open_mfdataset`
            will change. To retain the existing behavior, pass
            combine='nested'. To use future default behavior, pass
            combine='by_coords'. See
            http://xarray.pydata.org/en/stable/combining.html#combining-multi
            """
            )
            warnings.warn(basic_msg, FutureWarning, stacklevel=2)

            combined = auto_combine(
                datasets,
                concat_dim=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                from_openmfds=True,
            )
        elif combine == "nested":
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
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(
                datasets, compat=compat, data_vars=data_vars, coords=coords, join=join
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

    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs
    return combined


WRITEABLE_STORES = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "h5netcdf": backends.H5NetCDFStore,
}
```
### 2 - xarray/backends/api.py:

Start line: 856, End line: 910

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="different",
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    **kwargs
):  # noqa
    if isinstance(paths, str):
        if is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        paths = sorted(glob(paths))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    # If combine='by_coords' then this is unnecessary, but quick.
    # If combine='nested' then this creates a flat list which is easier to
    # iterate over, while saving the originally-supplied structure as "ids"
    if combine == "nested":
        if str(concat_dim) == "_not_supplied":
            raise ValueError("Must supply concat_dim when using " "combine='nested'")
        else:
            if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
                concat_dim = [concat_dim]
    combined_ids_paths = _infer_concat_order_from_positions(paths)
    ids, paths = (list(combined_ids_paths.keys()), list(combined_ids_paths.values()))

    open_kwargs = dict(
        engine=engine, chunks=chunks or {}, lock=lock, autoclose=autoclose, **kwargs
    )

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
    file_objs = [getattr_(ds, "_file_obj") for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, file_objs = dask.compute(datasets, file_objs)

    # Combine all datasets, closing them in case of a ValueError
    # ... other code
```
### 3 - asv_bench/benchmarks/dataset_io.py:

Start line: 315, End line: 328

```python
class IOReadMultipleNetCDF4(IOMultipleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF4"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_netcdf4(self):
        xr.open_mfdataset(self.filenames_list, engine="netcdf4").load()

    def time_open_dataset_netcdf4(self):
        xr.open_mfdataset(self.filenames_list, engine="netcdf4")
```
### 4 - xarray/backends/api.py:

Start line: 696, End line: 965

```python
class _MultiFileCloser:
    __slots__ = ("file_objs",)

    def __init__(self, file_objs):
        self.file_objs = file_objs

    def close(self):
        for f in self.file_objs:
            f.close()


def open_mfdataset(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="different",
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    **kwargs
):
    # ... other code
```
### 5 - asv_bench/benchmarks/dataset_io.py:

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
### 6 - asv_bench/benchmarks/dataset_io.py:

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
### 7 - asv_bench/benchmarks/dataset_io.py:

Start line: 331, End line: 344

```python
class IOReadMultipleNetCDF3(IOReadMultipleNetCDF4):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF3_64BIT"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_scipy(self):
        xr.open_mfdataset(self.filenames_list, engine="scipy").load()

    def time_open_dataset_scipy(self):
        xr.open_mfdataset(self.filenames_list, engine="scipy")
```
### 8 - asv_bench/benchmarks/dataset_io.py:

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
                encoding=None,
                attrs={"units": "foo units", "description": "a description"},
            )
            ds["bar"] = xr.DataArray(
                randn((nt, self.nx, self.ny), frac_nan=0.2),
                coords={"lon": lons, "lat": lats, "time": times},
                dims=("time", "lon", "lat"),
                name="bar",
                encoding=None,
                attrs={"units": "bar units", "description": "a description"},
            )
            ds["baz"] = xr.DataArray(
                randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
                coords={"lon": lons, "lat": lats},
                dims=("lon", "lat"),
                name="baz",
                encoding=None,
                attrs={"units": "baz units", "description": "a description"},
            )

            ds.attrs = {"history": "created for xarray benchmarking"}

            self.ds_list.append(ds)
            self.filenames_list.append("test_netcdf_%i.nc" % i)
```
### 9 - asv_bench/benchmarks/dataset_io.py:

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
### 10 - xarray/backends/api.py:

Start line: 1192, End line: 1230

```python
def save_mfdataset(
    datasets, paths, mode="w", format=None, groups=None, engine=None, compute=True
):
    # ... other code

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
### 15 - xarray/backends/api.py:

Start line: 723, End line: 856

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="different",
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    **kwargs
):
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to 
    combine the datasets into one before returning the result, and if 
    combine='nested' then ``combine_nested`` is used. The filepaths must be 
    structured according to which combining function is used, the details of 
    which are given in the documentation for ``combine_by_coords`` and 
    ``combine_nested``. By default the old (now deprecated) ``auto_combine`` 
    will be used, please specify either ``combine='by_coords'`` or 
    ``combine='nested'`` in future. Requires dask to be installed. See 
    documentation for details on dask [1]. Attributes from the first dataset 
    file are used for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open. Paths can be given as strings or as pathlib
        Paths. If concatenation along more than one dimension is desired, then
        ``paths`` must be a nested list-of-lists (see ``manual_combine`` for
        details). (A string glob will be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk
        sizes. In general, these should divide the dimensions of each dataset.
        If int, chunk each dimension by ``chunks``.
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see the
        full documentation for more details [2].
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only
        need to provide this argument if any of the dimensions along which you
        want to concatenate is not a dimension in the original datasets, e.g.,
        if you want to stack a collection of 2D arrays along a third dimension.
        Set ``concat_dim=[..., None, ...]`` explicitly to
        disable concatenation along a particular dimension.
    combine : {'by_coords', 'nested'}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is 
        used to combine all the data. If this argument is not provided, 
        `xarray.auto_combine` is used, but in the future this behavior will 
        switch to use `xarray.combine_by_coords` by default.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:
         * 'broadcast_equals': all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * 'equals': all values and dimensions must be the same.
         * 'identical': all values, dimensions and attributes must be the
           same.
         * 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding['source']``.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib'}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:
          * 'minimal': Only data variables in which the dimension already
            appears are included.
          * 'different': Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * 'all': All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the 'minimal' data variables.
    coords : {'minimal', 'different', 'all' or list of str}, optional
        These coordinate variables will be concatenated together:
         * 'minimal': Only coordinates in which the dimension already appears
           are included.
         * 'different': Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * 'all': All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the 'minimal' coordinates.
    parallel : bool, optional
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {'outer', 'inner', 'left', 'right', 'exact, 'override'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
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
    auto_combine
    open_dataset

    References
    ----------

    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    # ... other code
```
### 20 - xarray/backends/api.py:

Start line: 442, End line: 484

```python
def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    # ... other code

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
        )

        _protect_dataset_variables_inplace(ds, cache)

        if chunks is not None:
            from dask.base import tokenize

            # if passed an actual file path, augment the token with
            # the file modification time
            if isinstance(filename_or_obj, str) and not is_remote_uri(filename_or_obj):
                mtime = os.path.getmtime(filename_or_obj)
            else:
                mtime = None
            token = tokenize(
                filename_or_obj,
                mtime,
                group,
                decode_cf,
                mask_and_scale,
                decode_times,
                concat_characters,
                decode_coords,
                engine,
                chunks,
                drop_variables,
                use_cftime,
            )
            name_prefix = "open_dataset-%s" % token
            ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token)
            ds2._file_obj = ds._file_obj
        else:
            ds2 = ds

        return ds2
    # ... other code
```
### 24 - xarray/backends/api.py:

Start line: 972, End line: 1068

```python
# type: Dict[str, Callable]


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
    _validate_attrs(dataset)

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
### 25 - xarray/backends/api.py:

Start line: 486, End line: 542

```python
def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    # ... other code

    if isinstance(filename_or_obj, Path):
        filename_or_obj = str(filename_or_obj)

    if isinstance(filename_or_obj, AbstractDataStore):
        store = filename_or_obj

    elif isinstance(filename_or_obj, str):
        filename_or_obj = _normalize_path(filename_or_obj)

        if engine is None:
            engine = _get_default_engine(filename_or_obj, allow_remote=True)
        if engine == "netcdf4":
            store = backends.NetCDF4DataStore.open(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )
        elif engine == "scipy":
            store = backends.ScipyDataStore(filename_or_obj, **backend_kwargs)
        elif engine == "pydap":
            store = backends.PydapDataStore.open(filename_or_obj, **backend_kwargs)
        elif engine == "h5netcdf":
            store = backends.H5NetCDFStore(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )
        elif engine == "pynio":
            store = backends.NioDataStore(filename_or_obj, lock=lock, **backend_kwargs)
        elif engine == "pseudonetcdf":
            store = backends.PseudoNetCDFDataStore.open(
                filename_or_obj, lock=lock, **backend_kwargs
            )
        elif engine == "cfgrib":
            store = backends.CfGribDataStore(
                filename_or_obj, lock=lock, **backend_kwargs
            )

    else:
        if engine not in [None, "scipy", "h5netcdf"]:
            raise ValueError(
                "can only read bytes or file-like objects "
                "with engine='scipy' or 'h5netcdf'"
            )
        engine = _get_engine_from_magic_number(filename_or_obj)
        if engine == "scipy":
            store = backends.ScipyDataStore(filename_or_obj, **backend_kwargs)
        elif engine == "h5netcdf":
            store = backends.H5NetCDFStore(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )

    with close_on_error(store):
        ds = maybe_decode_store(store)

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds
```
### 26 - xarray/backends/api.py:

Start line: 655, End line: 693

```python
def open_dataarray(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):

    dataset = open_dataset(
        filename_or_obj,
        group=group,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        autoclose=autoclose,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        lock=lock,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        use_cftime=use_cftime,
    )

    if len(dataset.data_vars) != 1:
        raise ValueError(
            "Given file dataset contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        data_array, = dataset.data_vars.values()

    data_array._file_obj = dataset._file_obj

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in dataset.attrs:
        data_array.name = dataset.attrs[DATAARRAY_NAME]
        del dataset.attrs[DATAARRAY_NAME]

    if data_array.name == DATAARRAY_VARIABLE:
        data_array.name = None

    return data_array
```
### 27 - xarray/backends/api.py:

Start line: 1069, End line: 1094

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
    # ... other code
    try:
        # TODO: allow this work (setting up the file for writing array data)
        # to be parallelized with dask
        dump_to_store(
            dataset, store, writer, encoding=encoding, unlimited_dims=unlimited_dims
        )
        if autoclose:
            store.close()

        if multifile:
            return writer, store

        writes = writer.sync(compute=compute)

        if path_or_file is None:
            store.sync()
            return target.getvalue()
    finally:
        if not multifile and compute:
            store.close()

    if not compute:
        import dask

        return dask.delayed(_finalize_store)(writes, store)
    return None
```
### 31 - xarray/core/combine.py:

Start line: 711, End line: 776

```python
def auto_combine(
    datasets,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    from_openmfds=False,
):

    if not from_openmfds:
        basic_msg = dedent(
            """\
        In xarray version 0.13 `auto_combine` will be deprecated. See
        http://xarray.pydata.org/en/stable/combining.html#combining-multi"""
        )
        warnings.warn(basic_msg, FutureWarning, stacklevel=2)

    if concat_dim == "_not_supplied":
        concat_dim = _CONCAT_DIM_DEFAULT
        message = ""
    else:
        message = dedent(
            """\
        Also `open_mfdataset` will no longer accept a `concat_dim` argument.
        To get equivalent behaviour from now on please use the new
        `combine_nested` function instead (or the `combine='nested'` option to
        `open_mfdataset`)."""
        )

    if _dimension_coords_exist(datasets):
        message += dedent(
            """\
        The datasets supplied have global dimension coordinates. You may want
        to use the new `combine_by_coords` function (or the
        `combine='by_coords'` option to `open_mfdataset`) to order the datasets
        before concatenation. Alternatively, to continue concatenating based
        on the order the datasets are supplied in future, please use the new
        `combine_nested` function (or the `combine='nested'` option to
        open_mfdataset)."""
        )
    else:
        message += dedent(
            """\
        The datasets supplied do not have global dimension coordinates. In
        future, to continue concatenating without supplying dimension
        coordinates, please use the new `combine_nested` function (or the
        `combine='nested'` option to open_mfdataset."""
        )

    if _requires_concat_and_merge(datasets):
        manual_dims = [concat_dim].append(None)
        message += dedent(
            """\
        The datasets supplied require both concatenation and merging. From
        xarray version 0.13 this will operation will require either using the
        new `combine_nested` function (or the `combine='nested'` option to
        open_mfdataset), with a nested list structure such that you can combine
        along the dimensions {}. Alternatively if your datasets have global
        dimension coordinates then you can use the new `combine_by_coords`
        function.""".format(
                manual_dims
            )
        )

    warnings.warn(message, FutureWarning, stacklevel=2)

    return _old_auto_combine(
        datasets,
        concat_dim=concat_dim,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        fill_value=fill_value,
        join=join,
    )
```
### 32 - xarray/core/combine.py:

Start line: 581, End line: 627

```python
def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars)
        )

        _check_shape_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(
            combined_ids,
            concat_dims=concat_dims,
            data_vars=data_vars,
            coords=coords,
            fill_value=fill_value,
            join=join,
        )

        # Check the overall coordinates are monotonically increasing
        for dim in concat_dims:
            indexes = concatenated.indexes.get(dim)
            if not (indexes.is_monotonic_increasing or indexes.is_monotonic_decreasing):
                raise ValueError(
                    "Resulting object does not have monotonic"
                    " global indexes along dimension {}".format(dim)
                )
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(
        concatenated_grouped_by_data_vars,
        compat=compat,
        fill_value=fill_value,
        join=join,
    )


# Everything beyond here is only needed until the deprecation cycle in #2616
# is completed


_CONCAT_DIM_DEFAULT = "__infer_concat_dim__"
```
### 33 - xarray/backends/api.py:

Start line: 400, End line: 440

```python
def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    engines = [
        None,
        "netcdf4",
        "scipy",
        "pydap",
        "h5netcdf",
        "pynio",
        "cfgrib",
        "pseudonetcdf",
    ]
    if engine not in engines:
        raise ValueError(
            "unrecognized engine for open_dataset: {}\n"
            "must be one of: {}".format(engine, engines)
        )

    if autoclose is not None:
        warnings.warn(
            "The autoclose argument is no longer used by "
            "xarray.open_dataset() and is now ignored; it will be removed in "
            "a future version of xarray. If necessary, you can control the "
            "maximum number of simultaneous open files with "
            "xarray.set_options(file_cache_maxsize=...).",
            FutureWarning,
            stacklevel=2,
        )

    if mask_and_scale is None:
        mask_and_scale = not engine == "pseudonetcdf"

    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    if cache is None:
        cache = chunks is None

    if backend_kwargs is None:
        backend_kwargs = {}
    # ... other code
```
### 45 - xarray/core/combine.py:

Start line: 204, End line: 222

```python
def _combine_all_along_first_dim(
    combined_ids, dim, data_vars, coords, compat, fill_value=dtypes.NA, join="outer"
):

    # Group into lines of datasets which must be combined along dim
    # need to sort by _new_tile_id first for groupby to work
    # TODO remove all these sorted OrderedDicts once python >= 3.6 only
    combined_ids = OrderedDict(sorted(combined_ids.items(), key=_new_tile_id))
    grouped = itertools.groupby(combined_ids.items(), key=_new_tile_id)

    # Combine all of these datasets along dim
    new_combined_ids = {}
    for new_id, group in grouped:
        combined_ids = OrderedDict(sorted(group))
        datasets = combined_ids.values()
        new_combined_ids[new_id] = _combine_1d(
            datasets, dim, compat, data_vars, coords, fill_value, join
        )
    return new_combined_ids
```
### 46 - xarray/core/concat.py:

Start line: 298, End line: 311

```python
def _dataset_concat(
    datasets,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):
    # ... other code

    def ensure_common_dims(vars):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(
                    non_concat_dims.get(d, dim_len) for d in common_dims
                )
                var = var.set_dims(common_dims, common_shape)
            yield var
    # ... other code
```
### 52 - xarray/core/concat.py:

Start line: 313, End line: 328

```python
def _dataset_concat(
    datasets,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):

    # stack up each variable to fill-out the dataset (in order)
    for k in datasets[0].variables:
        if k in concat_over:
            vars = ensure_common_dims([ds.variables[k] for ds in datasets])
            combined = concat_vars(vars, dim, positions)
            insert_result_variable(k, combined)

    result = Dataset(result_vars, attrs=result_attrs)
    result = result.set_coords(result_coord_names)
    result.encoding = result_encoding

    if coord is not None:
        # add concat dimension last to ensure that its in the final Dataset
        result[coord.name] = coord

    return result
```
### 55 - xarray/core/combine.py:

Start line: 779, End line: 811

```python
def _dimension_coords_exist(datasets):
    """
    Check if the datasets have consistent global dimension coordinates
    which would in future be used by `auto_combine` for concatenation ordering.
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Simulates performing the multidimensional combine on each group of data
    # variables before merging back together
    try:
        for vars, datasets_with_same_vars in grouped_by_vars:
            _infer_concat_order_from_coords(list(datasets_with_same_vars))
        return True
    except ValueError:
        # ValueError means datasets don't have global dimension coordinates
        # Or something else went wrong in trying to determine them
        return False


def _requires_concat_and_merge(datasets):
    """
    Check if the datasets require the use of both xarray.concat and
    xarray.merge, which in future might require the user to use
    `manual_combine` instead.
    """
    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    return len(list(grouped_by_vars)) > 1
```
### 64 - xarray/backends/api.py:

Start line: 1122, End line: 1190

```python
def save_mfdataset(
    datasets, paths, mode="w", format=None, groups=None, engine=None, compute=True
):
    """Write multiple datasets to disk as netCDF files simultaneously.

    This function is intended for use with datasets consisting of dask.array
    objects, in which case it can write the multiple datasets to disk
    simultaneously using a shared thread pool.

    When not using dask, it is no different than calling ``to_netcdf``
    repeatedly.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to save.
    paths : list of str or list of Paths
        List of paths to which to save each corresponding dataset.
    mode : {'w', 'a'}, optional
        Write ('w') or append ('a') mode. If mode='w', any existing file at
        these locations will be overwritten.
    format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
              'NETCDF3_CLASSIC'}, optional

        File format for the resulting netCDF file:

        * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
          features.
        * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
          netCDF 3 compatible API features.
        * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
          which fully supports 2+ GB files, but is only compatible with
          clients linked against netCDF version 3.6.0 or later.
        * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
          handle 2+ GB files very well.

        All formats are supported by the netCDF4-python library.
        scipy.io.netcdf only supports the last two formats.

        The default format is NETCDF4 if you are saving a file to disk and
        have the netCDF4-python library available. Otherwise, xarray falls
        back to using scipy to write netCDF files and defaults to the
        NETCDF3_64BIT format (scipy does not support netCDF4).
    groups : list of str, optional
        Paths to the netCDF4 group in each corresponding file to which to save
        datasets (only works for format='NETCDF4'). The groups will be created
        if necessary.
    engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, with a
        preference for 'netcdf4' if writing to a file on disk.
        See `Dataset.to_netcdf` for additional information.
    compute: boolean
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.

    Examples
    --------

    Save a dataset into one netCDF per year of data:

    >>> years, datasets = zip(*ds.groupby('time.year'))
    >>> paths = ['%s.nc' % y for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    if mode == "w" and len(set(paths)) < len(paths):
        raise ValueError(
            "cannot use mode='w' when writing multiple " "datasets to the same path"
        )
    # ... other code
```
### 67 - xarray/core/combine.py:

Start line: 814, End line: 843

```python
def _old_auto_combine(
    datasets,
    concat_dim=_CONCAT_DIM_DEFAULT,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    if concat_dim is not None:
        dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim

        sorted_datasets = sorted(datasets, key=vars_as_keys)
        grouped = itertools.groupby(sorted_datasets, key=vars_as_keys)

        concatenated = [
            _auto_concat(
                list(datasets),
                dim=dim,
                data_vars=data_vars,
                coords=coords,
                fill_value=fill_value,
                join=join,
            )
            for vars, datasets in grouped
        ]
    else:
        concatenated = datasets
    merged = merge(concatenated, compat=compat, fill_value=fill_value, join=join)
    return merged
```
### 68 - xarray/core/merge.py:

Start line: 402, End line: 417

```python
def merge_data_and_coords(data, coords, compat="broadcast_equals", join="outer"):
    """Used in Dataset.__init__."""
    objs = [data, coords]
    explicit_coords = coords.keys()
    indexes = dict(extract_indexes(coords))
    return merge_core(
        objs, compat, join, explicit_coords=explicit_coords, indexes=indexes
    )


def extract_indexes(coords):
    """Yields the name & index of valid indexes from a mapping of coords"""
    for name, variable in coords.items():
        variable = as_variable(variable, name=name)
        if variable.dims == (name,):
            yield name, variable.to_index()
```
### 69 - xarray/core/combine.py:

Start line: 846, End line: 881

```python
def _auto_concat(
    datasets,
    dim=None,
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    if len(datasets) == 1 and dim is None:
        # There is nothing more to combine, so kick out early.
        return datasets[0]
    else:
        if dim is None:
            ds0 = datasets[0]
            ds1 = datasets[1]
            concat_dims = set(ds0.dims)
            if ds0.dims != ds1.dims:
                dim_tuples = set(ds0.dims.items()) - set(ds1.dims.items())
                concat_dims = {i for i, _ in dim_tuples}
            if len(concat_dims) > 1:
                concat_dims = {d for d in concat_dims if not ds0[d].equals(ds1[d])}
            if len(concat_dims) > 1:
                raise ValueError(
                    "too many different dimensions to " "concatenate: %s" % concat_dims
                )
            elif len(concat_dims) == 0:
                raise ValueError(
                    "cannot infer dimension to concatenate: "
                    "supply the ``concat_dim`` argument "
                    "explicitly"
                )
            dim, = concat_dims
        return concat(
            datasets, dim=dim, data_vars=data_vars, coords=coords, fill_value=fill_value
        )
```
### 70 - xarray/core/concat.py:

Start line: 146, End line: 212

```python
def _calc_concat_over(datasets, dim, data_vars, coords):
    """
    Determine which dataset variables need to be concatenated in the result,
    and which can simply be taken from the first dataset.
    """
    # Return values
    concat_over = set()
    equals = {}

    if dim in datasets[0]:
        concat_over.add(dim)
    for ds in datasets:
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == "different":
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        # Compare the variable of all datasets vs. the one
                        # of the first dataset. Perform the minimum amount of
                        # loads in order to avoid multiple loads from disk
                        # while keeping the RAM footprint low.
                        v_lhs = datasets[0].variables[k].load()
                        # We'll need to know later on if variables are equal.
                        computed = []
                        for ds_rhs in datasets[1:]:
                            v_rhs = ds_rhs.variables[k].compute()
                            computed.append(v_rhs)
                            if not v_lhs.equals(v_rhs):
                                concat_over.add(k)
                                equals[k] = False
                                # computed variables are not to be re-computed
                                # again in the future
                                for ds, v in zip(datasets[1:], computed):
                                    ds.variables[k].data = v.data
                                break
                        else:
                            equals[k] = True

            elif opt == "all":
                concat_over.update(
                    set(getattr(datasets[0], subset)) - set(datasets[0].dims)
                )
            elif opt == "minimal":
                pass
            else:
                raise ValueError("unexpected value for %s: %s" % (subset, opt))
        else:
            invalid_vars = [k for k in opt if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == "coords":
                    raise ValueError(
                        "some variables in coords are not coordinates on "
                        "the first dataset: %s" % (invalid_vars,)
                    )
                else:
                    raise ValueError(
                        "some variables in data_vars are not data variables "
                        "on the first dataset: %s" % (invalid_vars,)
                    )
            concat_over.update(opt)

    process_subset_opt(data_vars, "data_vars")
    process_subset_opt(coords, "coords")
    return concat_over, equals
```
### 77 - xarray/backends/api.py:

Start line: 545, End line: 653

```python
def open_dataarray(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    """Open an DataArray from a file or file-like object containing a single
    data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Paths are interpreted as a path to a netCDF file or an
        OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
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
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib'}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. 'gregorian', 'proleptic_gregorian', 'standard', or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

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
### 78 - xarray/core/concat.py:

Start line: 215, End line: 296

```python
def _dataset_concat(
    datasets,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from .dataset import Dataset

    if compat not in ["equals", "identical"]:
        raise ValueError(
            "compat=%r invalid: must be 'equals' " "or 'identical'" % compat
        )

    dim, coord = _calc_concat_dim_coord(dim)
    # Make sure we're working on a copy (we'll be loading variables)
    datasets = [ds.copy() for ds in datasets]
    datasets = align(
        *datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value
    )

    concat_over, equals = _calc_concat_over(datasets, dim, data_vars, coords)

    def insert_result_variable(k, v):
        assert isinstance(v, Variable)
        if k in datasets[0].coords:
            result_coord_names.add(k)
        result_vars[k] = v

    # create the new dataset and add constant variables
    result_vars = OrderedDict()
    result_coord_names = set(datasets[0].coords)
    result_attrs = datasets[0].attrs
    result_encoding = datasets[0].encoding

    for k, v in datasets[0].variables.items():
        if k not in concat_over:
            insert_result_variable(k, v)

    # check that global attributes and non-concatenated variables are fixed
    # across all datasets
    for ds in datasets[1:]:
        if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
            raise ValueError("dataset global attributes not equal")
        for k, v in ds.variables.items():
            if k not in result_vars and k not in concat_over:
                raise ValueError("encountered unexpected variable %r" % k)
            elif (k in result_coord_names) != (k in ds.coords):
                raise ValueError(
                    "%r is a coordinate in some datasets but not " "others" % k
                )
            elif k in result_vars and k != dim:
                # Don't use Variable.identical as it internally invokes
                # Variable.equals, and we may already know the answer
                if compat == "identical" and not utils.dict_equiv(
                    v.attrs, result_vars[k].attrs
                ):
                    raise ValueError("variable %s not identical across datasets" % k)

                # Proceed with equals()
                try:
                    # May be populated when using the "different" method
                    is_equal = equals[k]
                except KeyError:
                    result_vars[k].load()
                    is_equal = v.equals(result_vars[k])
                if not is_equal:
                    raise ValueError("variable %s not equal across datasets" % k)

    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    dim_lengths = [ds.dims.get(dim, 1) for ds in datasets]
    non_concat_dims = {}
    for ds in datasets:
        non_concat_dims.update(ds.dims)
    non_concat_dims.pop(dim, None)
    # ... other code
```
### 84 - xarray/core/concat.py:

Start line: 99, End line: 119

```python
def concat(
    objs,
    dim,
    data_vars="all",
    coords="different",
    compat="equals",
    positions=None,
    fill_value=dtypes.NA,
    join="outer",
):
    # TODO: add ignore_index arguments copied from pandas.concat
    # TODO: support concatenating scalar coordinates even if the concatenated
    # dimension already exists
    from .dataset import Dataset
    from .dataarray import DataArray

    try:
        first_obj, objs = utils.peek_at(objs)
    except StopIteration:
        raise ValueError("must supply at least one object to concatenate")

    if isinstance(first_obj, DataArray):
        f = _dataarray_concat
    elif isinstance(first_obj, Dataset):
        f = _dataset_concat
    else:
        raise TypeError(
            "can only concatenate xarray Dataset and DataArray "
            "objects, got %s" % type(first_obj)
        )
    return f(objs, dim, data_vars, coords, compat, positions, fill_value, join)
```
### 85 - xarray/backends/api.py:

Start line: 219, End line: 258

```python
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


def load_dataset(filename_or_obj, **kwargs):
    """Open, load into memory, and close a Dataset from a file or file-like
    object.

    This is a thin wrapper around :py:meth:`~xarray.open_dataset`. It differs
    from `open_dataset` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataset` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataset`. See that documentation for further details.

    Returns
    -------
    dataset : Dataset
        The newly created Dataset.

    See Also
    --------
    open_dataset
    """
    if "cache" in kwargs:
        raise TypeError("cache has no effect in this context")

    with open_dataset(filename_or_obj, **kwargs) as ds:
        return ds.load()
```
### 86 - xarray/core/dataarray.py:

Start line: 743, End line: 773

```python
class DataArray(AbstractArray, DataWithCoords):

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
### 90 - xarray/core/dataarray.py:

Start line: 422, End line: 446

```python
class DataArray(AbstractArray, DataWithCoords):

    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)

    def _from_temp_dataset(
        self, dataset: Dataset, name: Hashable = __default
    ) -> "DataArray":
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        return self._replace(variable, coords, name)

    def _to_dataset_split(self, dim: Hashable) -> Dataset:
        def subset(dim, label):
            array = self.loc[{dim: label}]
            if dim in array.coords:
                del array.coords[dim]
            array.attrs = {}
            return array

        variables = OrderedDict(
            [(label, subset(dim, label)) for label in self.get_index(dim)]
        )
        coords = self.coords.to_dataset()
        if dim in coords:
            del coords[dim]
        return Dataset(variables, coords, self.attrs)
```
### 91 - xarray/core/dataarray.py:

Start line: 2220, End line: 2247

```python
class DataArray(AbstractArray, DataWithCoords):

    def to_netcdf(self, *args, **kwargs) -> Optional["Delayed"]:
        """Write DataArray contents to a netCDF file.

        All parameters are passed directly to `xarray.Dataset.to_netcdf`.

        Notes
        -----
        Only xarray.Dataset objects can be written to netCDF files, so
        the xarray.DataArray is converted to a xarray.Dataset object
        containing a single variable. If the DataArray has no name, or if the
        name is the same as a co-ordinate name, then it is given the name
        '__xarray_dataarray_variable__'.
        """
        from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

        if self.name is None:
            # If no name is set then use a generic xarray name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            # The name is the same as one of the coords names, which netCDF
            # doesn't support, so rename it but keep track of the old name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            # No problems with the name - so we're fine!
            dataset = self.to_dataset()

        return dataset.to_netcdf(*args, **kwargs)
```
### 93 - xarray/backends/api.py:

Start line: 287, End line: 399

```python
def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
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
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib', \
        'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. 'gregorian', 'proleptic_gregorian', 'standard', or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

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
### 96 - xarray/core/dataarray.py:

Start line: 448, End line: 472

```python
class DataArray(AbstractArray, DataWithCoords):

    def _to_dataset_whole(
        self, name: Hashable = None, shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "unable to convert unnamed DataArray to a "
                "Dataset without providing an explicit name"
            )
        if name in self.coords:
            raise ValueError(
                "cannot create a Dataset from a DataArray with "
                "the same name as one of its coordinates"
            )
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        coord_names = set(self._coords)
        dataset = Dataset._from_vars_and_coord_names(variables, coord_names)
        return dataset
```
### 97 - xarray/core/dataarray.py:

Start line: 2473, End line: 2500

```python
class DataArray(AbstractArray, DataWithCoords):

    @staticmethod
    def _binary_op(
        f: Callable[..., Any],
        reflexive: bool = False,
        join: str = None,  # see xarray.align
        **ignored_kwargs
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
            coords = self.coords._merge_raw(other_coords)
            name = self._result_name(other)

            return self._replace(variable, coords, name)

        return func
```
### 98 - xarray/core/combine.py:

Start line: 466, End line: 579

```python
def vars_as_keys(ds):
    return tuple(sorted(ds))


def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Attempt to auto-magically combine the given datasets into one by using
    dimension coordinates.

    This method attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `manual_combine`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to combine.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
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
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> x1
    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   0 1 2
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...

    >>> x2
    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   3 4 5
    Data variables:
        temperature     (x) float64 6.97 8.13 7.42 ...

    >>> combined = xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:         (x: 6)
    Coords:
      * position        (x) int64   0 1 2 3 4 5
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...
    """
    # ... other code
```
### 99 - xarray/backends/api.py:

Start line: 1, End line: 58

```python
import os.path
import warnings
from glob import glob
from io import BytesIO
from numbers import Number
from pathlib import Path
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Tuple,
    Union,
)

import numpy as np

from .. import DataArray, Dataset, auto_combine, backends, coding, conventions
from ..core import indexing
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from ..core.utils import close_on_error, is_grib_path, is_remote_uri
from .common import AbstractDataStore, ArrayWriter
from .locks import _get_scheduler

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


DATAARRAY_NAME = "__xarray_dataarray_name__"
DATAARRAY_VARIABLE = "__xarray_dataarray_variable__"


def _get_default_engine_remote_uri():
    try:
        import netCDF4  # noqa

        engine = "netcdf4"
    except ImportError:  # pragma: no cover
        try:
            import pydap  # noqa

            engine = "pydap"
        except ImportError:
            raise ValueError(
                "netCDF4 or pydap is required for accessing "
                "remote datasets via OPeNDAP"
            )
    return engine
```
### 100 - xarray/core/merge.py:

Start line: 601, End line: 640

```python
def dataset_merge_method(
    dataset: "Dataset",
    other: "DatasetLike",
    overwrite_vars: Union[Hashable, Iterable[Hashable]],
    compat: str,
    join: str,
    fill_value: Any,
) -> Tuple["OrderedDict[Hashable, Variable]", Set[Hashable], Dict[Hashable, int]]:
    """Guts of the Dataset.merge method.
    """

    # we are locked into supporting overwrite_vars for the Dataset.merge
    # method due for backwards compatibility
    # TODO: consider deprecating it?

    if isinstance(overwrite_vars, Iterable) and not isinstance(overwrite_vars, str):
        overwrite_vars = set(overwrite_vars)
    else:
        overwrite_vars = {overwrite_vars}

    if not overwrite_vars:
        objs = [dataset, other]
        priority_arg = None
    elif overwrite_vars == set(other):
        objs = [dataset, other]
        priority_arg = 1
    else:
        other_overwrite = OrderedDict()  # type: MutableDatasetLike
        other_no_overwrite = OrderedDict()  # type: MutableDatasetLike
        for k, v in other.items():
            if k in overwrite_vars:
                other_overwrite[k] = v
            else:
                other_no_overwrite[k] = v
        objs = [dataset, other_no_overwrite, other_overwrite]
        priority_arg = 2

    return merge_core(
        objs, compat, join, priority_arg=priority_arg, fill_value=fill_value
    )
```
### 102 - xarray/core/merge.py:

Start line: 643, End line: 669

```python
def dataset_update_method(
    dataset: "Dataset", other: "DatasetLike"
) -> Tuple["OrderedDict[Hashable, Variable]", Set[Hashable], Dict[Hashable, int]]:
    """Guts of the Dataset.update method.

    This drops a duplicated coordinates from `other` if `other` is not an
    `xarray.Dataset`, e.g., if it's a dict with DataArray values (GH2068,
    GH2180).
    """
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    if not isinstance(other, Dataset):
        other = OrderedDict(other)
        for key, value in other.items():
            if isinstance(value, DataArray):
                # drop conflicting coordinates
                coord_names = [
                    c
                    for c in value.coords
                    if c not in value.dims and c in dataset.coords
                ]
                if coord_names:
                    other[key] = value.drop(coord_names)

    return merge_core([dataset, other], priority_arg=1, indexes=dataset.indexes)
```
### 108 - xarray/core/dataarray.py:

Start line: 371, End line: 403

```python
class DataArray(AbstractArray, DataWithCoords):

    def _replace(
        self,
        variable: Variable = None,
        coords=None,
        name: Optional[Hashable] = __default,
    ) -> "DataArray":
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is self.__default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True)

    def _replace_maybe_drop_dims(
        self, variable: Variable, name: Optional[Hashable] = __default
    ) -> "DataArray":
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
        elif variable.dims == self.dims:
            # Shape has changed (e.g. from reduce(..., keepdims=True)
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = OrderedDict(
                (k, v)
                for k, v in self._coords.items()
                if v.shape == tuple(new_sizes[d] for d in v.dims)
            )
        else:
            allowed_dims = set(variable.dims)
            coords = OrderedDict(
                (k, v) for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            )
        return self._replace(variable, coords, name)
```
### 113 - xarray/core/merge.py:

Start line: 364, End line: 399

```python
def expand_and_merge_variables(objs, priority_arg=None):
    """Merge coordinate variables without worrying about alignment.

    This function is used for merging variables in computation.py.
    """
    expanded = expand_variable_dicts(objs)
    priority_vars = _get_priority_vars(objs, priority_arg)
    variables = merge_variables(expanded, priority_vars)
    return variables


def merge_coords(
    objs,
    compat="minimal",
    join="outer",
    priority_arg=None,
    indexes=None,
    fill_value=dtypes.NA,
):
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objs)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    expanded = expand_variable_dicts(aligned)
    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)
    assert_unique_multiindex_level_names(variables)

    return variables
```
### 115 - xarray/core/dataarray.py:

Start line: 2502, End line: 2521

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
### 120 - xarray/core/dataarray.py:

Start line: 1, End line: 78

```python
import functools
import sys
import warnings
from collections import OrderedDict
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
    Union,
    cast,
    overload,
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
)
from .accessor_dt import DatetimeAccessor
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
from .dataset import Dataset, merge_indexes, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes
from .options import OPTIONS
from .utils import ReprObject, _check_inplace, either_dict_or_kwargs
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
```
### 122 - xarray/core/dataarray.py:

Start line: 166, End line: 181

```python
def _check_data_shape(data, coords, dims):
    if data is dtypes.NA:
        data = np.nan
    if coords is not None and utils.is_scalar(data, include_0d=False):
        if utils.is_dict_like(coords):
            if dims is None:
                return data
            else:
                data_shape = tuple(
                    as_variable(coords[k], k).size if k in coords.keys() else 1
                    for k in dims
                )
        else:
            data_shape = tuple(as_variable(coord, "foo").size for coord in coords)
        data = np.full(data_shape, data)
    return data
```
### 123 - xarray/core/combine.py:

Start line: 266, End line: 308

```python
def _new_tile_id(single_id_ds_pair):
    tile_id, ds = single_id_ds_pair
    return tile_id[1:]


def _nested_combine(
    datasets,
    concat_dims,
    compat,
    data_vars,
    coords,
    ids,
    fill_value=dtypes.NA,
    join="outer",
):

    if len(datasets) == 0:
        return Dataset()

    # Arrange datasets for concatenation
    # Use information from the shape of the user input
    if not ids:
        # Determine tile_IDs by structure of input in N-D
        # (i.e. ordering in list-of-lists)
        combined_ids = _infer_concat_order_from_positions(datasets)
    else:
        # Already sorted so just use the ids already passed
        combined_ids = OrderedDict(zip(ids, datasets))

    # Check that the inferred shape is combinable
    _check_shape_tile_ids(combined_ids)

    # Apply series of concatenate or merge operations along each dimension
    combined = _combine_nd(
        combined_ids,
        concat_dims,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        fill_value=fill_value,
        join=join,
    )
    return combined
```
### 127 - xarray/core/combine.py:

Start line: 630, End line: 709

```python
def auto_combine(
    datasets,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    from_openmfds=False,
):
    """
    Attempt to auto-magically combine the given datasets into one.

    This entire function is deprecated in favour of ``combine_nested`` and
    ``combine_by_coords``.

    This method attempts to combine a list of datasets into a single entity by
    inspecting metadata and using a combination of concat and merge.
    It does not concatenate along more than one dimension or sort data under
    any circumstances. It does align coordinates, but different variables on
    datasets can cause it to fail under some scenarios. In complex cases, you
    may need to clean up your data and use ``concat``/``merge`` explicitly.
    ``auto_combine`` works well if you have N years of data and M data
    variables, and each combination of a distinct time period and set of data
    variables is saved its own dataset.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to merge.
    concat_dim : str or DataArray or Index, optional
        Dimension along which to concatenate variables, as used by
        :py:func:`xarray.concat`. You only need to provide this argument if
        the dimension along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
    compat : {'identical', 'equals', 'broadcast_equals',
             'no_conflicts'}, optional
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
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' o list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """
    # ... other code
```
### 138 - xarray/core/combine.py:

Start line: 225, End line: 263

```python
def _combine_1d(
    datasets,
    concat_dim,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Applies either concat or merge to 1D list of datasets depending on value
    of concat_dim
    """

    if concat_dim is not None:
        try:
            combined = concat(
                datasets,
                dim=concat_dim,
                data_vars=data_vars,
                coords=coords,
                fill_value=fill_value,
                join=join,
            )
        except ValueError as err:
            if "encountered unexpected variable" in str(err):
                raise ValueError(
                    "These objects cannot be combined using only "
                    "xarray.combine_nested, instead either use "
                    "xarray.combine_by_coords, or do it manually "
                    "with xarray.concat, xarray.merge and "
                    "xarray.align"
                )
            else:
                raise
    else:
        combined = merge(datasets, compat=compat, fill_value=fill_value, join=join)

    return combined
```
