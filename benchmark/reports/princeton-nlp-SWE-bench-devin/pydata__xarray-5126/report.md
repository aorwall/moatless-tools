# pydata__xarray-5126

| **pydata/xarray** | `6bfbaede69eb73810cb63672a8161bd1fc147594` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 4601 |
| **Any found context length** | 746 |
| **Avg pos** | 71.33333333333333 |
| **Min pos** | 3 |
| **Max pos** | 41 |
| **Top file pos** | 1 |
| **Missing snippets** | 18 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -11,7 +11,7 @@
 from pandas.errors import OutOfBoundsDatetime
 
 from .duck_array_ops import array_equiv
-from .options import OPTIONS
+from .options import OPTIONS, _get_boolean_with_default
 from .pycompat import dask_array_type, sparse_array_type
 from .utils import is_duck_array
 
@@ -371,7 +371,9 @@ def _calculate_col_width(col_items):
     return col_width
 
 
-def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
+def _mapping_repr(
+    mapping, title, summarizer, expand_option_name, col_width=None, max_rows=None
+):
     if col_width is None:
         col_width = _calculate_col_width(mapping)
     if max_rows is None:
@@ -379,7 +381,9 @@ def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
     summary = [f"{title}:"]
     if mapping:
         len_mapping = len(mapping)
-        if len_mapping > max_rows:
+        if not _get_boolean_with_default(expand_option_name, default=True):
+            summary = [f"{summary[0]} ({len_mapping})"]
+        elif len_mapping > max_rows:
             summary = [f"{summary[0]} ({max_rows}/{len_mapping})"]
             first_rows = max_rows // 2 + max_rows % 2
             items = list(mapping.items())
@@ -396,12 +400,18 @@ def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
 
 
 data_vars_repr = functools.partial(
-    _mapping_repr, title="Data variables", summarizer=summarize_datavar
+    _mapping_repr,
+    title="Data variables",
+    summarizer=summarize_datavar,
+    expand_option_name="display_expand_data_vars",
 )
 
 
 attrs_repr = functools.partial(
-    _mapping_repr, title="Attributes", summarizer=summarize_attr
+    _mapping_repr,
+    title="Attributes",
+    summarizer=summarize_attr,
+    expand_option_name="display_expand_attrs",
 )
 
 
@@ -409,7 +419,11 @@ def coords_repr(coords, col_width=None):
     if col_width is None:
         col_width = _calculate_col_width(_get_col_items(coords))
     return _mapping_repr(
-        coords, title="Coordinates", summarizer=summarize_coord, col_width=col_width
+        coords,
+        title="Coordinates",
+        summarizer=summarize_coord,
+        expand_option_name="display_expand_coords",
+        col_width=col_width,
     )
 
 
@@ -493,9 +507,14 @@ def array_repr(arr):
     else:
         name_str = ""
 
+    if _get_boolean_with_default("display_expand_data", default=True):
+        data_repr = short_data_repr(arr)
+    else:
+        data_repr = inline_variable_array_repr(arr, OPTIONS["display_width"])
+
     summary = [
         "<xarray.{} {}({})>".format(type(arr).__name__, name_str, dim_summary(arr)),
-        short_data_repr(arr),
+        data_repr,
     ]
 
     if hasattr(arr, "coords"):
diff --git a/xarray/core/formatting_html.py b/xarray/core/formatting_html.py
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -6,6 +6,7 @@
 import pkg_resources
 
 from .formatting import inline_variable_array_repr, short_data_repr
+from .options import _get_boolean_with_default
 
 STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")
 
@@ -164,9 +165,14 @@ def collapsible_section(
     )
 
 
-def _mapping_section(mapping, name, details_func, max_items_collapse, enabled=True):
+def _mapping_section(
+    mapping, name, details_func, max_items_collapse, expand_option_name, enabled=True
+):
     n_items = len(mapping)
-    collapsed = n_items >= max_items_collapse
+    expanded = _get_boolean_with_default(
+        expand_option_name, n_items < max_items_collapse
+    )
+    collapsed = not expanded
 
     return collapsible_section(
         name,
@@ -188,7 +194,11 @@ def dim_section(obj):
 def array_section(obj):
     # "unique" id to expand/collapse the section
     data_id = "section-" + str(uuid.uuid4())
-    collapsed = "checked"
+    collapsed = (
+        "checked"
+        if _get_boolean_with_default("display_expand_data", default=True)
+        else ""
+    )
     variable = getattr(obj, "variable", obj)
     preview = escape(inline_variable_array_repr(variable, max_width=70))
     data_repr = short_data_repr_html(obj)
@@ -209,6 +219,7 @@ def array_section(obj):
     name="Coordinates",
     details_func=summarize_coords,
     max_items_collapse=25,
+    expand_option_name="display_expand_coords",
 )
 
 
@@ -217,6 +228,7 @@ def array_section(obj):
     name="Data variables",
     details_func=summarize_vars,
     max_items_collapse=15,
+    expand_option_name="display_expand_data_vars",
 )
 
 
@@ -225,6 +237,7 @@ def array_section(obj):
     name="Attributes",
     details_func=summarize_attrs,
     max_items_collapse=10,
+    expand_option_name="display_expand_attrs",
 )
 
 
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -6,6 +6,10 @@
 DISPLAY_MAX_ROWS = "display_max_rows"
 DISPLAY_STYLE = "display_style"
 DISPLAY_WIDTH = "display_width"
+DISPLAY_EXPAND_ATTRS = "display_expand_attrs"
+DISPLAY_EXPAND_COORDS = "display_expand_coords"
+DISPLAY_EXPAND_DATA_VARS = "display_expand_data_vars"
+DISPLAY_EXPAND_DATA = "display_expand_data"
 ENABLE_CFTIMEINDEX = "enable_cftimeindex"
 FILE_CACHE_MAXSIZE = "file_cache_maxsize"
 KEEP_ATTRS = "keep_attrs"
@@ -19,6 +23,10 @@
     DISPLAY_MAX_ROWS: 12,
     DISPLAY_STYLE: "html",
     DISPLAY_WIDTH: 80,
+    DISPLAY_EXPAND_ATTRS: "default",
+    DISPLAY_EXPAND_COORDS: "default",
+    DISPLAY_EXPAND_DATA_VARS: "default",
+    DISPLAY_EXPAND_DATA: "default",
     ENABLE_CFTIMEINDEX: True,
     FILE_CACHE_MAXSIZE: 128,
     KEEP_ATTRS: "default",
@@ -38,6 +46,10 @@ def _positive_integer(value):
     DISPLAY_MAX_ROWS: _positive_integer,
     DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
     DISPLAY_WIDTH: _positive_integer,
+    DISPLAY_EXPAND_ATTRS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_COORDS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_DATA_VARS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_DATA: lambda choice: choice in [True, False, "default"],
     ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
     FILE_CACHE_MAXSIZE: _positive_integer,
     KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
@@ -65,8 +77,8 @@ def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
 }
 
 
-def _get_keep_attrs(default):
-    global_choice = OPTIONS["keep_attrs"]
+def _get_boolean_with_default(option, default):
+    global_choice = OPTIONS[option]
 
     if global_choice == "default":
         return default
@@ -74,10 +86,14 @@ def _get_keep_attrs(default):
         return global_choice
     else:
         raise ValueError(
-            "The global option keep_attrs must be one of True, False or 'default'."
+            f"The global option {option} must be one of True, False or 'default'."
         )
 
 
+def _get_keep_attrs(default):
+    return _get_boolean_with_default("keep_attrs", default)
+
+
 class set_options:
     """Set options for xarray in a controlled context.
 
@@ -108,6 +124,22 @@ class set_options:
       Default: ``'default'``.
     - ``display_style``: display style to use in jupyter for xarray objects.
       Default: ``'text'``. Other options are ``'html'``.
+    - ``display_expand_attrs``: whether to expand the attributes section for
+      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
+    - ``display_expand_coords``: whether to expand the coordinates section for
+      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
+    - ``display_expand_data``: whether to expand the data section for display
+      of ``DataArray`` objects. Can be ``True`` to always expand, ``False`` to
+      always collapse, or ``default`` to expand unless over a pre-defined limit.
+      Default: ``default``.
+    - ``display_expand_data_vars``: whether to expand the data variables section
+      for display of ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
 
 
     You can use ``set_options`` either as a context manager:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/formatting.py | 14 | 14 | 20 | 2 | 7150
| xarray/core/formatting.py | 374 | 374 | 21 | 2 | 7401
| xarray/core/formatting.py | 382 | 382 | 21 | 2 | 7401
| xarray/core/formatting.py | 399 | 404 | 5 | 2 | 1537
| xarray/core/formatting.py | 412 | 412 | 5 | 2 | 1537
| xarray/core/formatting.py | 496 | 496 | 41 | 2 | 14924
| xarray/core/formatting_html.py | 9 | 9 | 19 | 1 | 6879
| xarray/core/formatting_html.py | 167 | 169 | 7 | 1 | 2355
| xarray/core/formatting_html.py | 191 | 191 | 3 | 1 | 746
| xarray/core/formatting_html.py | 212 | 212 | 3 | 1 | 746
| xarray/core/formatting_html.py | 220 | 220 | 3 | 1 | 746
| xarray/core/formatting_html.py | 228 | 228 | 3 | 1 | 746
| xarray/core/options.py | 9 | 9 | 11 | 4 | 4601
| xarray/core/options.py | 22 | 22 | 11 | 4 | 4601
| xarray/core/options.py | 41 | 41 | 11 | 4 | 4601
| xarray/core/options.py | 68 | 69 | 11 | 4 | 4601
| xarray/core/options.py | 77 | 77 | 11 | 4 | 4601
| xarray/core/options.py | 111 | 111 | 8 | 4 | 3165


## Problem Statement

```
FR: Provide option for collapsing the HTML display in notebooks
# Issue description
The overly long output of the text repr of xarray always bugged so I was very happy that the recently implemented html repr collapsed the data part, and equally sad to see that 0.16.0 reverted that, IMHO, correct design implementation back, presumably to align it with the text repr.

# Suggested solution
As the opinions will vary on what a good repr should do, similar to existing xarray.set_options I would like to have an option that let's me control if the data part (and maybe other parts?) appear in a collapsed fashion for the html repr.

# Additional questions
* Is it worth considering this as well for the text repr? Or is that harder to implement?

Any guidance on 
  * which files need to change
  * potential pitfalls

would be welcome. I'm happy to work on this, as I seem to be the only one not liking the current implementation.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/formatting_html.py** | 136 | 164| 284 | 284 | 2192 | 
| 2 | **1 xarray/core/formatting_html.py** | 231 | 250| 180 | 464 | 2192 | 
| **-> 3 <-** | **1 xarray/core/formatting_html.py** | 188 | 228| 282 | 746 | 2192 | 
| 4 | **1 xarray/core/formatting_html.py** | 99 | 133| 404 | 1150 | 2192 | 
| **-> 5 <-** | **2 xarray/core/formatting.py** | 398 | 455| 387 | 1537 | 7669 | 
| 6 | 3 doc/conf.py | 269 | 323| 704 | 2241 | 11015 | 
| **-> 7 <-** | **3 xarray/core/formatting_html.py** | 167 | 185| 114 | 2355 | 11015 | 
| **-> 8 <-** | **4 xarray/core/options.py** | 81 | 164| 810 | 3165 | 12372 | 
| 9 | **4 xarray/core/formatting_html.py** | 253 | 289| 257 | 3422 | 12372 | 
| 10 | 4 doc/conf.py | 200 | 267| 632 | 4054 | 12372 | 
| **-> 11 <-** | **4 xarray/core/options.py** | 1 | 78| 547 | 4601 | 12372 | 
| 12 | **4 xarray/core/formatting_html.py** | 48 | 96| 369 | 4970 | 12372 | 
| 13 | 4 doc/conf.py | 102 | 120| 197 | 5167 | 12372 | 
| 14 | 4 doc/conf.py | 1 | 101| 670 | 5837 | 12372 | 
| 15 | **4 xarray/core/formatting.py** | 458 | 472| 127 | 5964 | 12372 | 
| 16 | **4 xarray/core/formatting.py** | 219 | 245| 203 | 6167 | 12372 | 
| 17 | 5 xarray/core/common.py | 1 | 45| 240 | 6407 | 28213 | 
| 18 | 6 xarray/plot/utils.py | 1 | 54| 283 | 6690 | 34952 | 
| **-> 19 <-** | **6 xarray/core/formatting_html.py** | 1 | 29| 189 | 6879 | 34952 | 
| **-> 20 <-** | **6 xarray/core/formatting.py** | 1 | 39| 271 | 7150 | 34952 | 
| **-> 21 <-** | **6 xarray/core/formatting.py** | 374 | 395| 251 | 7401 | 34952 | 
| 22 | **6 xarray/core/formatting.py** | 637 | 668| 250 | 7651 | 34952 | 
| 23 | **6 xarray/core/formatting_html.py** | 32 | 45| 110 | 7761 | 34952 | 
| 24 | 7 xarray/plot/plot.py | 636 | 786| 1643 | 9404 | 43210 | 
| 25 | **7 xarray/core/formatting.py** | 336 | 345| 149 | 9553 | 43210 | 
| 26 | **7 xarray/core/formatting.py** | 248 | 270| 221 | 9774 | 43210 | 
| 27 | **7 xarray/core/formatting.py** | 610 | 634| 130 | 9904 | 43210 | 
| 28 | 8 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 10317 | 43623 | 
| 29 | 9 xarray/core/rolling.py | 522 | 565| 350 | 10667 | 51155 | 
| 30 | 10 xarray/core/dask_array_compat.py | 155 | 225| 658 | 11325 | 52974 | 
| 31 | **10 xarray/core/formatting.py** | 515 | 544| 206 | 11531 | 52974 | 
| 32 | 10 xarray/core/rolling.py | 477 | 520| 416 | 11947 | 52974 | 
| 33 | **10 xarray/core/formatting.py** | 671 | 692| 158 | 12105 | 52974 | 
| 34 | **10 xarray/core/formatting.py** | 547 | 607| 473 | 12578 | 52974 | 
| 35 | 10 doc/conf.py | 121 | 199| 785 | 13363 | 52974 | 
| 36 | 11 asv_bench/benchmarks/repr.py | 1 | 19| 0 | 13363 | 53078 | 
| 37 | 12 xarray/core/weighted.py | 192 | 233| 342 | 13705 | 55141 | 
| 38 | 13 xarray/backends/api.py | 914 | 968| 420 | 14125 | 67197 | 
| 39 | 13 xarray/plot/plot.py | 294 | 330| 463 | 14588 | 67197 | 
| 40 | 14 xarray/testing.py | 177 | 201| 175 | 14763 | 70291 | 
| **-> 41 <-** | **14 xarray/core/formatting.py** | 489 | 512| 161 | 14924 | 70291 | 
| 42 | **14 xarray/core/formatting.py** | 475 | 486| 118 | 15042 | 70291 | 
| 43 | 14 xarray/core/rolling.py | 376 | 453| 710 | 15752 | 70291 | 
| 44 | 14 xarray/core/common.py | 104 | 120| 139 | 15891 | 70291 | 
| 45 | 15 xarray/core/dataset.py | 2097 | 2745| 6100 | 21991 | 131947 | 
| 46 | 16 doc/gallery/plot_control_colorbar.py | 1 | 33| 247 | 22238 | 132194 | 
| 47 | 16 xarray/core/rolling.py | 665 | 687| 146 | 22384 | 132194 | 
| 48 | 17 xarray/core/ops.py | 277 | 333| 349 | 22733 | 134576 | 
| 49 | 18 xarray/plot/dataset_plot.py | 277 | 389| 828 | 23561 | 139262 | 
| 50 | 19 xarray/__init__.py | 1 | 93| 603 | 24164 | 139865 | 
| 51 | 19 xarray/backends/api.py | 652 | 961| 459 | 24623 | 139865 | 
| 52 | 20 xarray/core/parallel.py | 482 | 551| 730 | 25353 | 144911 | 
| 53 | 20 doc/conf.py | 326 | 384| 356 | 25709 | 144911 | 
| 54 | 20 xarray/core/common.py | 85 | 102| 130 | 25839 | 144911 | 
| 55 | 21 xarray/core/groupby.py | 1 | 34| 223 | 26062 | 152786 | 
| 56 | 21 xarray/core/common.py | 1126 | 1183| 575 | 26637 | 152786 | 
| 57 | 22 doc/gallery/plot_cartopy_facetgrid.py | 1 | 46| 327 | 26964 | 153113 | 
| 58 | 23 xarray/conventions.py | 1 | 29| 168 | 27132 | 159514 | 
| 59 | 23 xarray/core/common.py | 65 | 82| 175 | 27307 | 159514 | 
| 60 | 23 xarray/core/rolling.py | 170 | 201| 245 | 27552 | 159514 | 
| 61 | 24 xarray/plot/__init__.py | 1 | 17| 0 | 27552 | 159608 | 
| 62 | 24 xarray/core/dataset.py | 415 | 440| 207 | 27759 | 159608 | 
| 63 | 24 xarray/core/common.py | 48 | 63| 124 | 27883 | 159608 | 
| 64 | 25 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 28616 | 161168 | 
| 65 | 25 xarray/core/dataset.py | 1 | 133| 655 | 29271 | 161168 | 
| 66 | 25 xarray/core/rolling.py | 1 | 36| 226 | 29497 | 161168 | 
| 67 | 25 xarray/plot/utils.py | 418 | 448| 254 | 29751 | 161168 | 
| 68 | 26 xarray/core/indexing.py | 1030 | 1097| 746 | 30497 | 173129 | 
| 69 | 26 xarray/core/rolling.py | 637 | 663| 212 | 30709 | 173129 | 
| 70 | 27 xarray/plot/facetgrid.py | 77 | 218| 1075 | 31784 | 178296 | 
| 71 | 27 xarray/core/common.py | 1260 | 1293| 279 | 32063 | 178296 | 
| 72 | 28 xarray/core/duck_array_ops.py | 361 | 388| 321 | 32384 | 183629 | 
| 73 | 28 xarray/plot/facetgrid.py | 385 | 419| 303 | 32687 | 183629 | 
| 74 | 28 xarray/plot/utils.py | 267 | 305| 388 | 33075 | 183629 | 
| 75 | 28 xarray/plot/dataset_plot.py | 1 | 101| 728 | 33803 | 183629 | 
| 76 | 28 xarray/core/rolling.py | 143 | 168| 281 | 34084 | 183629 | 
| 77 | 28 xarray/core/groupby.py | 844 | 886| 365 | 34449 | 183629 | 
| 78 | 28 xarray/plot/dataset_plot.py | 192 | 275| 977 | 35426 | 183629 | 
| 79 | 28 xarray/core/rolling.py | 619 | 635| 161 | 35587 | 183629 | 
| 80 | 28 xarray/plot/utils.py | 627 | 647| 156 | 35743 | 183629 | 
| 81 | 29 xarray/core/utils.py | 609 | 631| 151 | 35894 | 189830 | 
| 82 | 30 xarray/core/alignment.py | 82 | 270| 1951 | 37845 | 195889 | 
| 83 | 31 asv_bench/benchmarks/unstacking.py | 1 | 30| 194 | 38039 | 196083 | 
| 84 | 31 xarray/core/rolling.py | 958 | 979| 162 | 38201 | 196083 | 
| 85 | 32 xarray/util/generate_ops.py | 79 | 142| 749 | 38950 | 198582 | 


### Hint

```
Related: #4182
```

## Patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -11,7 +11,7 @@
 from pandas.errors import OutOfBoundsDatetime
 
 from .duck_array_ops import array_equiv
-from .options import OPTIONS
+from .options import OPTIONS, _get_boolean_with_default
 from .pycompat import dask_array_type, sparse_array_type
 from .utils import is_duck_array
 
@@ -371,7 +371,9 @@ def _calculate_col_width(col_items):
     return col_width
 
 
-def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
+def _mapping_repr(
+    mapping, title, summarizer, expand_option_name, col_width=None, max_rows=None
+):
     if col_width is None:
         col_width = _calculate_col_width(mapping)
     if max_rows is None:
@@ -379,7 +381,9 @@ def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
     summary = [f"{title}:"]
     if mapping:
         len_mapping = len(mapping)
-        if len_mapping > max_rows:
+        if not _get_boolean_with_default(expand_option_name, default=True):
+            summary = [f"{summary[0]} ({len_mapping})"]
+        elif len_mapping > max_rows:
             summary = [f"{summary[0]} ({max_rows}/{len_mapping})"]
             first_rows = max_rows // 2 + max_rows % 2
             items = list(mapping.items())
@@ -396,12 +400,18 @@ def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
 
 
 data_vars_repr = functools.partial(
-    _mapping_repr, title="Data variables", summarizer=summarize_datavar
+    _mapping_repr,
+    title="Data variables",
+    summarizer=summarize_datavar,
+    expand_option_name="display_expand_data_vars",
 )
 
 
 attrs_repr = functools.partial(
-    _mapping_repr, title="Attributes", summarizer=summarize_attr
+    _mapping_repr,
+    title="Attributes",
+    summarizer=summarize_attr,
+    expand_option_name="display_expand_attrs",
 )
 
 
@@ -409,7 +419,11 @@ def coords_repr(coords, col_width=None):
     if col_width is None:
         col_width = _calculate_col_width(_get_col_items(coords))
     return _mapping_repr(
-        coords, title="Coordinates", summarizer=summarize_coord, col_width=col_width
+        coords,
+        title="Coordinates",
+        summarizer=summarize_coord,
+        expand_option_name="display_expand_coords",
+        col_width=col_width,
     )
 
 
@@ -493,9 +507,14 @@ def array_repr(arr):
     else:
         name_str = ""
 
+    if _get_boolean_with_default("display_expand_data", default=True):
+        data_repr = short_data_repr(arr)
+    else:
+        data_repr = inline_variable_array_repr(arr, OPTIONS["display_width"])
+
     summary = [
         "<xarray.{} {}({})>".format(type(arr).__name__, name_str, dim_summary(arr)),
-        short_data_repr(arr),
+        data_repr,
     ]
 
     if hasattr(arr, "coords"):
diff --git a/xarray/core/formatting_html.py b/xarray/core/formatting_html.py
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -6,6 +6,7 @@
 import pkg_resources
 
 from .formatting import inline_variable_array_repr, short_data_repr
+from .options import _get_boolean_with_default
 
 STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")
 
@@ -164,9 +165,14 @@ def collapsible_section(
     )
 
 
-def _mapping_section(mapping, name, details_func, max_items_collapse, enabled=True):
+def _mapping_section(
+    mapping, name, details_func, max_items_collapse, expand_option_name, enabled=True
+):
     n_items = len(mapping)
-    collapsed = n_items >= max_items_collapse
+    expanded = _get_boolean_with_default(
+        expand_option_name, n_items < max_items_collapse
+    )
+    collapsed = not expanded
 
     return collapsible_section(
         name,
@@ -188,7 +194,11 @@ def dim_section(obj):
 def array_section(obj):
     # "unique" id to expand/collapse the section
     data_id = "section-" + str(uuid.uuid4())
-    collapsed = "checked"
+    collapsed = (
+        "checked"
+        if _get_boolean_with_default("display_expand_data", default=True)
+        else ""
+    )
     variable = getattr(obj, "variable", obj)
     preview = escape(inline_variable_array_repr(variable, max_width=70))
     data_repr = short_data_repr_html(obj)
@@ -209,6 +219,7 @@ def array_section(obj):
     name="Coordinates",
     details_func=summarize_coords,
     max_items_collapse=25,
+    expand_option_name="display_expand_coords",
 )
 
 
@@ -217,6 +228,7 @@ def array_section(obj):
     name="Data variables",
     details_func=summarize_vars,
     max_items_collapse=15,
+    expand_option_name="display_expand_data_vars",
 )
 
 
@@ -225,6 +237,7 @@ def array_section(obj):
     name="Attributes",
     details_func=summarize_attrs,
     max_items_collapse=10,
+    expand_option_name="display_expand_attrs",
 )
 
 
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -6,6 +6,10 @@
 DISPLAY_MAX_ROWS = "display_max_rows"
 DISPLAY_STYLE = "display_style"
 DISPLAY_WIDTH = "display_width"
+DISPLAY_EXPAND_ATTRS = "display_expand_attrs"
+DISPLAY_EXPAND_COORDS = "display_expand_coords"
+DISPLAY_EXPAND_DATA_VARS = "display_expand_data_vars"
+DISPLAY_EXPAND_DATA = "display_expand_data"
 ENABLE_CFTIMEINDEX = "enable_cftimeindex"
 FILE_CACHE_MAXSIZE = "file_cache_maxsize"
 KEEP_ATTRS = "keep_attrs"
@@ -19,6 +23,10 @@
     DISPLAY_MAX_ROWS: 12,
     DISPLAY_STYLE: "html",
     DISPLAY_WIDTH: 80,
+    DISPLAY_EXPAND_ATTRS: "default",
+    DISPLAY_EXPAND_COORDS: "default",
+    DISPLAY_EXPAND_DATA_VARS: "default",
+    DISPLAY_EXPAND_DATA: "default",
     ENABLE_CFTIMEINDEX: True,
     FILE_CACHE_MAXSIZE: 128,
     KEEP_ATTRS: "default",
@@ -38,6 +46,10 @@ def _positive_integer(value):
     DISPLAY_MAX_ROWS: _positive_integer,
     DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
     DISPLAY_WIDTH: _positive_integer,
+    DISPLAY_EXPAND_ATTRS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_COORDS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_DATA_VARS: lambda choice: choice in [True, False, "default"],
+    DISPLAY_EXPAND_DATA: lambda choice: choice in [True, False, "default"],
     ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
     FILE_CACHE_MAXSIZE: _positive_integer,
     KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
@@ -65,8 +77,8 @@ def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
 }
 
 
-def _get_keep_attrs(default):
-    global_choice = OPTIONS["keep_attrs"]
+def _get_boolean_with_default(option, default):
+    global_choice = OPTIONS[option]
 
     if global_choice == "default":
         return default
@@ -74,10 +86,14 @@ def _get_keep_attrs(default):
         return global_choice
     else:
         raise ValueError(
-            "The global option keep_attrs must be one of True, False or 'default'."
+            f"The global option {option} must be one of True, False or 'default'."
         )
 
 
+def _get_keep_attrs(default):
+    return _get_boolean_with_default("keep_attrs", default)
+
+
 class set_options:
     """Set options for xarray in a controlled context.
 
@@ -108,6 +124,22 @@ class set_options:
       Default: ``'default'``.
     - ``display_style``: display style to use in jupyter for xarray objects.
       Default: ``'text'``. Other options are ``'html'``.
+    - ``display_expand_attrs``: whether to expand the attributes section for
+      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
+    - ``display_expand_coords``: whether to expand the coordinates section for
+      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
+    - ``display_expand_data``: whether to expand the data section for display
+      of ``DataArray`` objects. Can be ``True`` to always expand, ``False`` to
+      always collapse, or ``default`` to expand unless over a pre-defined limit.
+      Default: ``default``.
+    - ``display_expand_data_vars``: whether to expand the data variables section
+      for display of ``Dataset`` objects. Can be ``True`` to always
+      expand, ``False`` to always collapse, or ``default`` to expand unless over
+      a pre-defined limit. Default: ``default``.
 
 
     You can use ``set_options`` either as a context manager:

```

## Test Patch

```diff
diff --git a/xarray/tests/test_formatting.py b/xarray/tests/test_formatting.py
--- a/xarray/tests/test_formatting.py
+++ b/xarray/tests/test_formatting.py
@@ -391,6 +391,17 @@ def test_array_repr(self):
 
         assert actual == expected
 
+        with xr.set_options(display_expand_data=False):
+            actual = formatting.array_repr(ds[(1, 2)])
+            expected = dedent(
+                """\
+            <xarray.DataArray (1, 2) (test: 1)>
+            0
+            Dimensions without coordinates: test"""
+            )
+
+            assert actual == expected
+
 
 def test_inline_variable_array_repr_custom_repr():
     class CustomArray:
@@ -492,3 +503,19 @@ def test__mapping_repr(display_max_rows, n_vars, n_attr):
         len_summary = len(summary)
         data_vars_print_size = min(display_max_rows, len_summary)
         assert len_summary == data_vars_print_size
+
+    with xr.set_options(
+        display_expand_coords=False,
+        display_expand_data_vars=False,
+        display_expand_attrs=False,
+    ):
+        actual = formatting.dataset_repr(ds)
+        expected = dedent(
+            f"""\
+            <xarray.Dataset>
+            Dimensions:      (time: 2)
+            Coordinates: (1)
+            Data variables: ({n_vars})
+            Attributes: ({n_attr})"""
+        )
+        assert actual == expected
diff --git a/xarray/tests/test_formatting_html.py b/xarray/tests/test_formatting_html.py
--- a/xarray/tests/test_formatting_html.py
+++ b/xarray/tests/test_formatting_html.py
@@ -115,6 +115,17 @@ def test_repr_of_dataarray(dataarray):
         formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 2
     )
 
+    with xr.set_options(display_expand_data=False):
+        formatted = fh.array_repr(dataarray)
+        assert "dim_0" in formatted
+        # has an expanded data section
+        assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 0
+        # coords and attrs don't have an items so they'll be be disabled and collapsed
+        assert (
+            formatted.count("class='xr-section-summary-in' type='checkbox' disabled >")
+            == 2
+        )
+
 
 def test_summary_of_multiindex_coord(multiindex):
     idx = multiindex.x.variable.to_index_variable()
@@ -138,6 +149,20 @@ def test_repr_of_dataset(dataset):
     assert "&lt;U4" in formatted or "&gt;U4" in formatted
     assert "&lt;IA&gt;" in formatted
 
+    with xr.set_options(
+        display_expand_coords=False,
+        display_expand_data_vars=False,
+        display_expand_attrs=False,
+    ):
+        formatted = fh.dataset_repr(dataset)
+        # coords, attrs, and data_vars are collapsed
+        assert (
+            formatted.count("class='xr-section-summary-in' type='checkbox'  checked>")
+            == 0
+        )
+        assert "&lt;U4" in formatted or "&gt;U4" in formatted
+        assert "&lt;IA&gt;" in formatted
+
 
 def test_repr_text_fallback(dataset):
     formatted = fh.dataset_repr(dataset)

```


## Code snippets

### 1 - xarray/core/formatting_html.py:

Start line: 136, End line: 164

```python
def summarize_vars(variables):
    vars_li = "".join(
        f"<li class='xr-var-item'>{summarize_variable(k, v)}</li>"
        for k, v in variables.items()
    )

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


def collapsible_section(
    name, inline_details="", details="", n_items=None, enabled=True, collapsed=False
):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
    enabled = "" if enabled and has_items else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    return (
        f"<input id='{data_id}' class='xr-section-summary-in' "
        f"type='checkbox' {enabled} {collapsed}>"
        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='xr-section-inline-details'>{inline_details}</div>"
        f"<div class='xr-section-details'>{details}</div>"
    )
```
### 2 - xarray/core/formatting_html.py:

Start line: 231, End line: 250

```python
def _obj_repr(obj, header_components, sections):
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    """
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    icons_svg, css_style = _load_static_files()
    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' hidden>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )
```
### 3 - xarray/core/formatting_html.py:

Start line: 188, End line: 228

```python
def array_section(obj):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    collapsed = "checked"
    variable = getattr(obj, "variable", obj)
    preview = escape(inline_variable_array_repr(variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon("icon-database")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<div class='xr-array-data'>{data_repr}</div>"
        "</div>"
    )


coord_section = partial(
    _mapping_section,
    name="Coordinates",
    details_func=summarize_coords,
    max_items_collapse=25,
)


datavar_section = partial(
    _mapping_section,
    name="Data variables",
    details_func=summarize_vars,
    max_items_collapse=15,
)


attr_section = partial(
    _mapping_section,
    name="Attributes",
    details_func=summarize_attrs,
    max_items_collapse=10,
)
```
### 4 - xarray/core/formatting_html.py:

Start line: 99, End line: 133

```python
def summarize_variable(name, var, is_index=False, dtype=None, preview=None):
    variable = var.variable if hasattr(var, "variable") else var

    cssclass_idx = " class='xr-has-index'" if is_index else ""
    dims_str = f"({', '.join(escape(dim) for dim in var.dims)})"
    name = escape(str(name))
    dtype = dtype or escape(str(var.dtype))

    # "unique" ids required to expand/collapse subsections
    attrs_id = "attrs-" + str(uuid.uuid4())
    data_id = "data-" + str(uuid.uuid4())
    disabled = "" if len(var.attrs) else "disabled"

    preview = preview or escape(inline_variable_array_repr(variable, 35))
    attrs_ul = summarize_attrs(var.attrs)
    data_repr = short_data_repr_html(variable)

    attrs_icon = _icon("icon-file-text2")
    data_icon = _icon("icon-database")

    return (
        f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        f"<div class='xr-var-dims'>{dims_str}</div>"
        f"<div class='xr-var-dtype'>{dtype}</div>"
        f"<div class='xr-var-preview xr-preview'>{preview}</div>"
        f"<input id='{attrs_id}' class='xr-var-attrs-in' "
        f"type='checkbox' {disabled}>"
        f"<label for='{attrs_id}' title='Show/Hide attributes'>"
        f"{attrs_icon}</label>"
        f"<input id='{data_id}' class='xr-var-data-in' type='checkbox'>"
        f"<label for='{data_id}' title='Show/Hide data repr'>"
        f"{data_icon}</label>"
        f"<div class='xr-var-attrs'>{attrs_ul}</div>"
        f"<div class='xr-var-data'>{data_repr}</div>"
    )
```
### 5 - xarray/core/formatting.py:

Start line: 398, End line: 455

```python
data_vars_repr = functools.partial(
    _mapping_repr, title="Data variables", summarizer=summarize_datavar
)


attrs_repr = functools.partial(
    _mapping_repr, title="Attributes", summarizer=summarize_attr
)


def coords_repr(coords, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(_get_col_items(coords))
    return _mapping_repr(
        coords, title="Coordinates", summarizer=summarize_coord, col_width=col_width
    )


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), f"{k}: "))
    return "\n".join(summary)


def dim_summary(obj):
    elements = [f"{k}: {v}" for k, v in obj.sizes.items()]
    return ", ".join(elements)


def unindexed_dims_repr(dims, coords):
    unindexed_dims = [d for d in dims if d not in coords]
    if unindexed_dims:
        dims_str = ", ".join(f"{d}" for d in unindexed_dims)
        return "Dimensions without coordinates: " + dims_str
    else:
        return None


@contextlib.contextmanager
def set_numpy_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def limit_lines(string: str, *, limit: int):
    """
    If the string is more lines than the limit,
    this returns the middle lines replaced by an ellipsis
    """
    lines = string.splitlines()
    if len(lines) > limit:
        string = "\n".join(chain(lines[: limit // 2], ["..."], lines[-limit // 2 :]))
    return string
```
### 6 - doc/conf.py:

Start line: 269, End line: 323

```python
rediraffe_redirects = {
    "terminology.rst": "user-guide/terminology.rst",
    "data-structures.rst": "user-guide/data-structures.rst",
    "indexing.rst": "user-guide/indexing.rst",
    "interpolation.rst": "user-guide/interpolation.rst",
    "computation.rst": "user-guide/computation.rst",
    "groupby.rst": "user-guide/groupby.rst",
    "reshaping.rst": "user-guide/reshaping.rst",
    "combining.rst": "user-guide/combining.rst",
    "time-series.rst": "user-guide/time-series.rst",
    "weather-climate.rst": "user-guide/weather-climate.rst",
    "pandas.rst": "user-guide/pandas.rst",
    "io.rst": "user-guide/io.rst",
    "dask.rst": "user-guide/dask.rst",
    "plotting.rst": "user-guide/plotting.rst",
    "duckarrays.rst": "user-guide/duckarrays.rst",
    "related-projects.rst": "ecosystem.rst",
    "faq.rst": "getting-started-guide/faq.rst",
    "why-xarray.rst": "getting-started-guide/why-xarray.rst",
    "installing.rst": "getting-started-guide/installing.rst",
    "quick-overview.rst": "getting-started-guide/quick-overview.rst",
}

# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
ipython_savefig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_build", "html", "_static"
)
if not os.path.exists(ipython_savefig_dir):
    os.makedirs(ipython_savefig_dir)


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt

# Output file base name for HTML help builder.
htmlhelp_basename = "xarraydoc"


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "iris": ("https://scitools-iris.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "cftime": ("https://unidata.github.io/cftime", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
}
```
### 7 - xarray/core/formatting_html.py:

Start line: 167, End line: 185

```python
def _mapping_section(mapping, name, details_func, max_items_collapse, enabled=True):
    n_items = len(mapping)
    collapsed = n_items >= max_items_collapse

    return collapsible_section(
        name,
        details=details_func(mapping),
        n_items=n_items,
        enabled=enabled,
        collapsed=collapsed,
    )


def dim_section(obj):
    dim_list = format_dims(obj.dims, list(obj.coords))

    return collapsible_section(
        "Dimensions", inline_details=dim_list, enabled=False, collapsed=True
    )
```
### 8 - xarray/core/options.py:

Start line: 81, End line: 164

```python
class set_options:
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
    - ``display_max_rows``: maximum display rows. Default: ``12``.
    - ``arithmetic_join``: DataArray/Dataset alignment in binary operations.
      Default: ``'inner'``.
    - ``file_cache_maxsize``: maximum number of open files to hold in xarray's
      global least-recently-usage cached. This should be smaller than your
      system's per-process file descriptor limit, e.g., ``ulimit -n`` on Linux.
      Default: 128.
    - ``warn_for_unclosed_files``: whether or not to issue a warning when
      unclosed files are deallocated (default False). This is mostly useful
      for debugging.
    - ``cmap_sequential``: colormap to use for nondivergent data plots.
      Default: ``viridis``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)
    - ``cmap_divergent``: colormap to use for divergent data plots.
      Default: ``RdBu_r``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)
    - ``keep_attrs``: rule for whether to keep attributes on xarray
      Datasets/dataarrays after operations. Either ``True`` to always keep
      attrs, ``False`` to always discard them, or ``'default'`` to use original
      logic that attrs should only be kept in unambiguous circumstances.
      Default: ``'default'``.
    - ``display_style``: display style to use in jupyter for xarray objects.
      Default: ``'text'``. Other options are ``'html'``.


    You can use ``set_options`` either as a context manager:

    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    ...
    <xarray.Dataset>
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 0 1 2 ... 998 999
    Data variables:
        *empty*

    Or to set global options:

    >>> xr.set_options(display_width=80)  # doctest: +ELLIPSIS
    <xarray.core.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                if k == ARITHMETIC_JOIN:
                    expected = f"Expected one of {_JOIN_OPTIONS!r}"
                elif k == DISPLAY_STYLE:
                    expected = f"Expected one of {_DISPLAY_OPTIONS!r}"
                else:
                    expected = ""
                raise ValueError(
                    f"option {k!r} given an invalid value: {v!r}. " + expected
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
```
### 9 - xarray/core/formatting_html.py:

Start line: 253, End line: 289

```python
def array_repr(arr):
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))

    obj_type = "xarray.{}".format(type(arr).__name__)
    arr_name = f"'{arr.name}'" if getattr(arr, "name", None) else ""
    coord_names = list(arr.coords) if hasattr(arr, "coords") else []

    header_components = [
        f"<div class='xr-obj-type'>{obj_type}</div>",
        f"<div class='xr-array-name'>{arr_name}</div>",
        format_dims(dims, coord_names),
    ]

    sections = [array_section(arr)]

    if hasattr(arr, "coords"):
        sections.append(coord_section(arr.coords))

    sections.append(attr_section(arr.attrs))

    return _obj_repr(arr, header_components, sections)


def dataset_repr(ds):
    obj_type = "xarray.{}".format(type(ds).__name__)

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        attr_section(ds.attrs),
    ]

    return _obj_repr(ds, header_components, sections)
```
### 10 - doc/conf.py:

Start line: 200, End line: 267

```python
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output ----------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_title = ""

html_context = {
    "github_user": "pydata",
    "github_repo": "xarray",
    "github_version": "master",
    "doc_path": "doc",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = dict(
    # analytics_id=''  this is configured in rtfd.io
    # canonical_url="",
    repository_url="https://github.com/pydata/xarray",
    repository_branch="master",
    path_to_docs="doc",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
    extra_navbar="",
    navbar_footer_text="",
    extra_footer="""<p>Xarray is a fiscally sponsored project of <a href="https://numfocus.org">NumFOCUS</a>,
    a nonprofit dedicated to supporting the open-source scientific computing community.<br>
    Theme by the <a href="https://ebp.jupyterbook.org">Executable Book Project</a></p>""",
    twitter_url="https://twitter.com/xarray_devs",
)


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/dataset-diagram-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["style.css"]


# configuration for sphinxext.opengraph
ogp_site_url = "https://xarray.pydata.org/en/latest/"
ogp_image = "https://xarray.pydata.org/en/stable/_static/dataset-diagram-logo.png"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image" />',
    '<meta property="twitter:site" content="@xarray_dev />',
    '<meta name="image" property="og:image" content="https://xarray.pydata.org/en/stable/_static/dataset-diagram-logo.png">',
]

# Redirects for pages that were moved to new locations
```
### 11 - xarray/core/options.py:

Start line: 1, End line: 78

```python
import warnings

ARITHMETIC_JOIN = "arithmetic_join"
CMAP_DIVERGENT = "cmap_divergent"
CMAP_SEQUENTIAL = "cmap_sequential"
DISPLAY_MAX_ROWS = "display_max_rows"
DISPLAY_STYLE = "display_style"
DISPLAY_WIDTH = "display_width"
ENABLE_CFTIMEINDEX = "enable_cftimeindex"
FILE_CACHE_MAXSIZE = "file_cache_maxsize"
KEEP_ATTRS = "keep_attrs"
WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"


OPTIONS = {
    ARITHMETIC_JOIN: "inner",
    CMAP_DIVERGENT: "RdBu_r",
    CMAP_SEQUENTIAL: "viridis",
    DISPLAY_MAX_ROWS: 12,
    DISPLAY_STYLE: "html",
    DISPLAY_WIDTH: 80,
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    KEEP_ATTRS: "default",
    WARN_FOR_UNCLOSED_FILES: False,
}

_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    DISPLAY_MAX_ROWS: _positive_integer,
    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
    DISPLAY_WIDTH: _positive_integer,
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
    KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
}


def _set_file_cache_maxsize(value):
    from ..backends.file_manager import FILE_CACHE

    FILE_CACHE.maxsize = value


def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
    warnings.warn(
        "The enable_cftimeindex option is now a no-op "
        "and will be removed in a future version of xarray.",
        FutureWarning,
    )


_SETTERS = {
    ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
}


def _get_keep_attrs(default):
    global_choice = OPTIONS["keep_attrs"]

    if global_choice == "default":
        return default
    elif global_choice in [True, False]:
        return global_choice
    else:
        raise ValueError(
            "The global option keep_attrs must be one of True, False or 'default'."
        )
```
### 12 - xarray/core/formatting_html.py:

Start line: 48, End line: 96

```python
def summarize_attrs(attrs):
    attrs_dl = "".join(
        f"<dt><span>{escape(k)} :</span></dt>" f"<dd>{escape(str(v))}</dd>"
        for k, v in attrs.items()
    )

    return f"<dl class='xr-attrs'>{attrs_dl}</dl>"


def _icon(icon_name):
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        "<svg class='icon xr-{0}'>"
        "<use xlink:href='#{0}'>"
        "</use>"
        "</svg>".format(icon_name)
    )


def _summarize_coord_multiindex(name, coord):
    preview = f"({', '.join(escape(l) for l in coord.level_names)})"
    return summarize_variable(
        name, coord, is_index=True, dtype="MultiIndex", preview=preview
    )


def summarize_coord(name, var):
    is_index = name in var.dims
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            coords = {}
            coords[name] = _summarize_coord_multiindex(name, coord)
            for lname in coord.level_names:
                var = coord.get_level_variable(lname)
                coords[lname] = summarize_variable(lname, var)
            return coords

    return {name: summarize_variable(name, var, is_index)}


def summarize_coords(variables):
    coords = {}
    for k, v in variables.items():
        coords.update(**summarize_coord(k, v))

    vars_li = "".join(f"<li class='xr-var-item'>{v}</li>" for v in coords.values())

    return f"<ul class='xr-var-list'>{vars_li}</ul>"
```
### 15 - xarray/core/formatting.py:

Start line: 458, End line: 472

```python
def short_numpy_repr(array):
    array = np.asarray(array)

    # default to lower precision so a full (abbreviated) line can fit on
    # one line with the default display_width
    options = {"precision": 6, "linewidth": OPTIONS["display_width"], "threshold": 200}
    if array.ndim < 3:
        edgeitems = 3
    elif array.ndim == 3:
        edgeitems = 2
    else:
        edgeitems = 1
    options["edgeitems"] = edgeitems
    with set_numpy_options(**options):
        return repr(array)
```
### 16 - xarray/core/formatting.py:

Start line: 219, End line: 245

```python
_KNOWN_TYPE_REPRS = {np.ndarray: "np.ndarray"}
with contextlib.suppress(ImportError):
    import sparse

    _KNOWN_TYPE_REPRS[sparse.COO] = "sparse.COO"


def inline_dask_repr(array):
    """Similar to dask.array.DataArray.__repr__, but without
    redundant information that's already printed by the repr
    function of the xarray wrapper.
    """
    assert isinstance(array, dask_array_type), array

    chunksize = tuple(c[0] for c in array.chunks)

    if hasattr(array, "_meta"):
        meta = array._meta
        if type(meta) in _KNOWN_TYPE_REPRS:
            meta_repr = _KNOWN_TYPE_REPRS[type(meta)]
        else:
            meta_repr = type(meta).__name__
        meta_string = f", meta={meta_repr}"
    else:
        meta_string = ""

    return f"dask.array<chunksize={chunksize}{meta_string}>"
```
### 19 - xarray/core/formatting_html.py:

Start line: 1, End line: 29

```python
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape

import pkg_resources

from .formatting import inline_variable_array_repr, short_data_repr

STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [
        pkg_resources.resource_string("xarray", fname).decode("utf8")
        for fname in STATIC_FILES
    ]


def short_data_repr_html(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()
    else:
        text = escape(short_data_repr(array))
        return f"<pre>{text}</pre>"
```
### 20 - xarray/core/formatting.py:

Start line: 1, End line: 39

```python
"""String formatting routines for __repr__.
"""
import contextlib
import functools
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from typing import Hashable

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from .duck_array_ops import array_equiv
from .options import OPTIONS
from .pycompat import dask_array_type, sparse_array_type
from .utils import is_duck_array


def pretty_print(x, numchars: int):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = maybe_truncate(x, numchars)
    return s + " " * max(numchars - len(s), 0)


def maybe_truncate(obj, maxlen=500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[: (maxlen - 3)] + "..."
    return s


def wrap_indent(text, start="", length=None):
    if length is None:
        length = len(start)
    indent = "\n" + " " * length
    return start + indent.join(x for x in text.splitlines())
```
### 21 - xarray/core/formatting.py:

Start line: 374, End line: 395

```python
def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    if max_rows is None:
        max_rows = OPTIONS["display_max_rows"]
    summary = [f"{title}:"]
    if mapping:
        len_mapping = len(mapping)
        if len_mapping > max_rows:
            summary = [f"{summary[0]} ({max_rows}/{len_mapping})"]
            first_rows = max_rows // 2 + max_rows % 2
            items = list(mapping.items())
            summary += [summarizer(k, v, col_width) for k, v in items[:first_rows]]
            if max_rows > 1:
                last_rows = max_rows // 2
                summary += [pretty_print("    ...", col_width) + " ..."]
                summary += [summarizer(k, v, col_width) for k, v in items[-last_rows:]]
        else:
            summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return "\n".join(summary)
```
### 22 - xarray/core/formatting.py:

Start line: 637, End line: 668

```python
def diff_array_repr(a, b, compat):
    # used for DataArray, Variable and IndexVariable
    summary = [
        "Left and right {} objects are not {}".format(
            type(a).__name__, _compat_to_str(compat)
        )
    ]

    summary.append(diff_dim_summary(a, b))
    if callable(compat):
        equiv = compat
    else:
        equiv = array_equiv

    if not equiv(a.data, b.data):
        temp = [wrap_indent(short_numpy_repr(obj), start="    ") for obj in (a, b)]
        diff_data_repr = [
            ab_side + "\n" + ab_data_repr
            for ab_side, ab_data_repr in zip(("L", "R"), temp)
        ]
        summary += ["Differing values:"] + diff_data_repr

    if hasattr(a, "coords"):
        col_width = _calculate_col_width(set(a.coords) | set(b.coords))
        summary.append(
            diff_coords_repr(a.coords, b.coords, compat, col_width=col_width)
        )

    if compat == "identical":
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))

    return "\n".join(summary)
```
### 23 - xarray/core/formatting_html.py:

Start line: 32, End line: 45

```python
def format_dims(dims, coord_names):
    if not dims:
        return ""

    dim_css_map = {
        k: " class='xr-has-index'" if k in coord_names else "" for k, v in dims.items()
    }

    dims_li = "".join(
        f"<li><span{dim_css_map[dim]}>" f"{escape(dim)}</span>: {size}</li>"
        for dim, size in dims.items()
    )

    return f"<ul class='xr-dim-list'>{dims_li}</ul>"
```
### 25 - xarray/core/formatting.py:

Start line: 336, End line: 345

```python
def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    # Indent key and add ':', then right-pad if col_width is not None
    k_str = f"    {key}:"
    if col_width is not None:
        k_str = pretty_print(k_str, col_width)
    # Replace tabs and newlines, so we print on one line in known width
    v_str = str(value).replace("\t", "\\t").replace("\n", "\\n")
    # Finally, truncate to the desired display width
    return maybe_truncate(f"{k_str} {v_str}", OPTIONS["display_width"])
```
### 26 - xarray/core/formatting.py:

Start line: 248, End line: 270

```python
def inline_sparse_repr(array):
    """Similar to sparse.COO.__repr__, but without the redundant shape/dtype."""
    assert isinstance(array, sparse_array_type), array
    return "<{}: nnz={:d}, fill_value={!s}>".format(
        type(array).__name__, array.nnz, array.fill_value
    )


def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    if var._in_memory:
        return format_array_flat(var, max_width)
    elif isinstance(var._data, dask_array_type):
        return inline_dask_repr(var.data)
    elif isinstance(var._data, sparse_array_type):
        return inline_sparse_repr(var.data)
    elif hasattr(var._data, "_repr_inline_"):
        return var._data._repr_inline_(max_width)
    elif hasattr(var._data, "__array_function__"):
        return maybe_truncate(repr(var._data).replace("\n", " "), max_width)
    else:
        # internal xarray array type
        return "..."
```
### 27 - xarray/core/formatting.py:

Start line: 610, End line: 634

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
### 31 - xarray/core/formatting.py:

Start line: 515, End line: 544

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
### 33 - xarray/core/formatting.py:

Start line: 671, End line: 692

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
### 34 - xarray/core/formatting.py:

Start line: 547, End line: 607

```python
def _diff_mapping_repr(a_mapping, b_mapping, compat, title, summarizer, col_width=None):
    def extra_items_repr(extra_keys, mapping, ab_side):
        extra_repr = [summarizer(k, mapping[k], col_width) for k in extra_keys]
        if extra_repr:
            header = f"{title} only on the {ab_side} object:"
            return [header] + extra_repr
        else:
            return []

    a_keys = set(a_mapping)
    b_keys = set(b_mapping)

    summary = []

    diff_items = []

    for k in a_keys & b_keys:
        try:
            # compare xarray variable
            if not callable(compat):
                compatible = getattr(a_mapping[k], compat)(b_mapping[k])
            else:
                compatible = compat(a_mapping[k], b_mapping[k])
            is_variable = True
        except AttributeError:
            # compare attribute value
            if is_duck_array(a_mapping[k]) or is_duck_array(b_mapping[k]):
                compatible = array_equiv(a_mapping[k], b_mapping[k])
            else:
                compatible = a_mapping[k] == b_mapping[k]

            is_variable = False

        if not compatible:
            temp = [
                summarizer(k, vars[k], col_width) for vars in (a_mapping, b_mapping)
            ]

            if compat == "identical" and is_variable:
                attrs_summary = []

                for m in (a_mapping, b_mapping):
                    attr_s = "\n".join(
                        summarize_attr(ak, av) for ak, av in m[k].attrs.items()
                    )
                    attrs_summary.append(attr_s)

                temp = [
                    "\n".join([var_s, attr_s]) if attr_s else var_s
                    for var_s, attr_s in zip(temp, attrs_summary)
                ]

            diff_items += [ab_side + s[1:] for ab_side, s in zip(("L", "R"), temp)]

    if diff_items:
        summary += [f"Differing {title.lower()}:"] + diff_items

    summary += extra_items_repr(a_keys - b_keys, a_mapping, "left")
    summary += extra_items_repr(b_keys - a_keys, b_mapping, "right")

    return "\n".join(summary)
```
### 41 - xarray/core/formatting.py:

Start line: 489, End line: 512

```python
def array_repr(arr):
    # used for DataArray, Variable and IndexVariable
    if hasattr(arr, "name") and arr.name is not None:
        name_str = f"{arr.name!r} "
    else:
        name_str = ""

    summary = [
        "<xarray.{} {}({})>".format(type(arr).__name__, name_str, dim_summary(arr)),
        short_data_repr(arr),
    ]

    if hasattr(arr, "coords"):
        if arr.coords:
            summary.append(repr(arr.coords))

        unindexed_dims_str = unindexed_dims_repr(arr.dims, arr.coords)
        if unindexed_dims_str:
            summary.append(unindexed_dims_str)

    if arr.attrs:
        summary.append(attrs_repr(arr.attrs))

    return "\n".join(summary)
```
### 42 - xarray/core/formatting.py:

Start line: 475, End line: 486

```python
def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if isinstance(array, np.ndarray):
        return short_numpy_repr(array)
    elif is_duck_array(internal_data):
        return limit_lines(repr(array.data), limit=40)
    elif array._in_memory or array.size < 1e5:
        return short_numpy_repr(array)
    else:
        # internal xarray array type
        return f"[{array.size} values with dtype={array.dtype}]"
```
