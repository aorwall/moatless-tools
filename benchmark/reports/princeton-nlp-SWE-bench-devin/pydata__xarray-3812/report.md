# pydata__xarray-3812

| **pydata/xarray** | `8512b7bf498c0c300f146447c0b05545842e9404` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3607 |
| **Any found context length** | 3607 |
| **Avg pos** | 9.0 |
| **Min pos** | 9 |
| **Max pos** | 9 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -20,7 +20,7 @@
     CMAP_SEQUENTIAL: "viridis",
     CMAP_DIVERGENT: "RdBu_r",
     KEEP_ATTRS: "default",
-    DISPLAY_STYLE: "text",
+    DISPLAY_STYLE: "html",
 }
 
 _JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/options.py | 23 | 23 | 9 | 4 | 3607


## Problem Statement

```
Turn on _repr_html_ by default?
I just wanted to open this to discuss turning the _repr_html_ on by default. This PR https://github.com/pydata/xarray/pull/3425 added it as a style option, but I suspect that more people will use if it is on by default. Does that seem like a reasonable change?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/conf.py | 215 | 350| 839 | 839 | 2825 | 
| 2 | 2 xarray/core/formatting_html.py | 200 | 236| 213 | 1052 | 4926 | 
| 3 | 2 doc/conf.py | 111 | 214| 820 | 1872 | 4926 | 
| 4 | 2 xarray/core/formatting_html.py | 239 | 275| 256 | 2128 | 4926 | 
| 5 | 2 xarray/core/formatting_html.py | 182 | 197| 172 | 2300 | 4926 | 
| 6 | 2 xarray/core/formatting_html.py | 1 | 23| 178 | 2478 | 4926 | 
| 7 | 2 xarray/core/formatting_html.py | 93 | 127| 403 | 2881 | 4926 | 
| 8 | 3 xarray/core/formatting.py | 209 | 235| 203 | 3084 | 9872 | 
| **-> 9 <-** | **4 xarray/core/options.py** | 1 | 75| 523 | 3607 | 11103 | 
| 10 | 4 doc/conf.py | 1 | 110| 777 | 4384 | 11103 | 
| 11 | 4 xarray/core/formatting_html.py | 130 | 158| 284 | 4668 | 11103 | 
| 12 | 5 xarray/plot/plot.py | 432 | 458| 206 | 4874 | 19236 | 
| 13 | 6 xarray/core/common.py | 1 | 35| 178 | 5052 | 31827 | 
| 14 | 6 xarray/core/formatting.py | 411 | 425| 127 | 5179 | 31827 | 
| 15 | 6 xarray/core/formatting.py | 1 | 37| 252 | 5431 | 31827 | 
| 16 | **6 xarray/core/options.py** | 78 | 150| 708 | 6139 | 31827 | 
| 17 | 6 xarray/core/formatting_html.py | 42 | 90| 369 | 6508 | 31827 | 
| 18 | 7 xarray/plot/__init__.py | 1 | 17| 0 | 6508 | 31921 | 
| 19 | 7 xarray/core/formatting.py | 569 | 588| 104 | 6612 | 31921 | 
| 20 | 7 xarray/core/formatting_html.py | 161 | 179| 114 | 6726 | 31921 | 
| 21 | 7 xarray/plot/plot.py | 116 | 200| 585 | 7311 | 31921 | 
| 22 | 7 xarray/core/formatting.py | 444 | 467| 161 | 7472 | 31921 | 
| 23 | 7 xarray/core/formatting_html.py | 26 | 39| 110 | 7582 | 31921 | 
| 24 | 7 xarray/core/formatting.py | 238 | 258| 197 | 7779 | 31921 | 
| 25 | 7 xarray/plot/plot.py | 612 | 765| 1699 | 9478 | 31921 | 
| 26 | 7 xarray/core/formatting.py | 313 | 322| 149 | 9627 | 31921 | 
| 27 | 8 xarray/plot/utils.py | 1 | 54| 283 | 9910 | 38084 | 
| 28 | 8 xarray/plot/plot.py | 1 | 28| 152 | 10062 | 38084 | 
| 29 | 8 xarray/plot/plot.py | 295 | 329| 443 | 10505 | 38084 | 
| 30 | 9 xarray/conventions.py | 106 | 127| 169 | 10674 | 43813 | 
| 31 | 9 xarray/core/formatting.py | 428 | 441| 127 | 10801 | 43813 | 
| 32 | 9 xarray/plot/utils.py | 570 | 585| 125 | 10926 | 43813 | 
| 33 | 10 xarray/core/dataarray.py | 1 | 81| 430 | 11356 | 69748 | 
| 34 | 10 xarray/plot/plot.py | 567 | 765| 266 | 11622 | 69748 | 
| 35 | 10 xarray/core/formatting.py | 591 | 618| 231 | 11853 | 69748 | 
| 36 | 11 xarray/plot/dataset_plot.py | 245 | 351| 765 | 12618 | 73169 | 
| 37 | 11 xarray/core/formatting.py | 325 | 408| 566 | 13184 | 73169 | 
| 38 | 12 xarray/backends/api.py | 919 | 988| 588 | 13772 | 84102 | 
| 39 | 12 xarray/plot/plot.py | 461 | 565| 1237 | 15009 | 84102 | 
| 40 | 12 xarray/core/formatting.py | 470 | 499| 206 | 15215 | 84102 | 
| 41 | 12 xarray/core/dataarray.py | 361 | 402| 384 | 15599 | 84102 | 
| 42 | 12 doc/conf.py | 351 | 362| 165 | 15764 | 84102 | 
| 43 | 12 xarray/core/dataarray.py | 404 | 430| 251 | 16015 | 84102 | 
| 44 | 12 xarray/plot/dataset_plot.py | 110 | 164| 350 | 16365 | 84102 | 
| 45 | 12 xarray/core/formatting.py | 502 | 566| 488 | 16853 | 84102 | 
| 46 | 13 xarray/core/pycompat.py | 1 | 20| 0 | 16853 | 84200 | 
| 47 | 13 xarray/plot/plot.py | 31 | 113| 759 | 17612 | 84200 | 
| 48 | 14 xarray/__init__.py | 1 | 89| 577 | 18189 | 84777 | 
| 49 | 15 xarray/core/utils.py | 532 | 555| 153 | 18342 | 89659 | 
| 50 | 16 setup.py | 1 | 5| 0 | 18342 | 89677 | 
| 51 | 16 xarray/plot/utils.py | 140 | 265| 861 | 19203 | 89677 | 
| 52 | 16 xarray/backends/api.py | 658 | 696| 335 | 19538 | 89677 | 
| 53 | 17 xarray/core/dataset.py | 1 | 130| 615 | 20153 | 136088 | 
| 54 | 18 properties/conftest.py | 1 | 9| 0 | 20153 | 136141 | 
| 55 | 18 xarray/backends/api.py | 64 | 81| 146 | 20299 | 136141 | 
| 56 | 19 xarray/core/duck_array_ops.py | 324 | 348| 252 | 20551 | 141053 | 
| 57 | 20 xarray/core/npcompat.py | 78 | 97| 119 | 20670 | 141821 | 
| 58 | 20 xarray/core/dataarray.py | 187 | 212| 236 | 20906 | 141821 | 
| 59 | 21 doc/gallery/plot_rasterio_rgb.py | 1 | 33| 217 | 21123 | 142038 | 
| 60 | 21 xarray/plot/plot.py | 767 | 821| 318 | 21441 | 142038 | 
| 61 | 21 xarray/core/formatting.py | 621 | 642| 158 | 21599 | 142038 | 
| 62 | 21 xarray/core/common.py | 1021 | 1070| 513 | 22112 | 142038 | 
| 63 | 21 xarray/core/common.py | 55 | 72| 175 | 22287 | 142038 | 
| 64 | 21 xarray/core/dataarray.py | 2647 | 2674| 177 | 22464 | 142038 | 
| 65 | 22 doc/gallery/plot_control_colorbar.py | 1 | 33| 247 | 22711 | 142285 | 
| 66 | 23 xarray/plot/facetgrid.py | 76 | 213| 1058 | 23769 | 147091 | 
| 67 | 24 doc/gallery/plot_colorbar_center.py | 1 | 44| 323 | 24092 | 147414 | 
| 68 | 24 xarray/core/dataarray.py | 2575 | 2602| 232 | 24324 | 147414 | 
| 69 | 24 xarray/backends/api.py | 84 | 109| 157 | 24481 | 147414 | 
| 70 | 25 xarray/core/dtypes.py | 145 | 170| 175 | 24656 | 148471 | 
| 71 | 26 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 25069 | 148884 | 
| 72 | 26 xarray/core/dataarray.py | 2625 | 2645| 140 | 25209 | 148884 | 
| 73 | 27 xarray/core/rolling.py | 124 | 140| 151 | 25360 | 153813 | 
| 74 | 28 xarray/core/groupby.py | 1 | 36| 230 | 25590 | 161650 | 
| 75 | 28 xarray/core/dtypes.py | 1 | 42| 277 | 25867 | 161650 | 
| 76 | 28 xarray/plot/dataset_plot.py | 1 | 76| 518 | 26385 | 161650 | 
| 77 | 28 xarray/core/common.py | 1132 | 1165| 302 | 26687 | 161650 | 
| 78 | 28 xarray/core/dataarray.py | 746 | 781| 307 | 26994 | 161650 | 
| 79 | 28 xarray/plot/facetgrid.py | 286 | 318| 236 | 27230 | 161650 | 
| 80 | 28 xarray/backends/api.py | 1 | 61| 321 | 27551 | 161650 | 
| 81 | 28 xarray/conventions.py | 82 | 103| 204 | 27755 | 161650 | 
| 82 | 29 xarray/backends/file_manager.py | 1 | 21| 139 | 27894 | 164210 | 
| 83 | 29 xarray/plot/facetgrid.py | 457 | 518| 467 | 28361 | 164210 | 
| 84 | 30 xarray/core/coordinates.py | 284 | 302| 176 | 28537 | 167122 | 
| 85 | 30 xarray/conventions.py | 176 | 218| 400 | 28937 | 167122 | 
| 86 | 30 xarray/core/common.py | 94 | 110| 139 | 29076 | 167122 | 
| 87 | 31 xarray/core/nputils.py | 56 | 67| 125 | 29201 | 169067 | 
| 88 | 31 xarray/plot/facetgrid.py | 1 | 32| 186 | 29387 | 169067 | 
| 89 | 32 xarray/core/indexing.py | 1444 | 1462| 209 | 29596 | 180730 | 
| 90 | 33 xarray/backends/zarr.py | 544 | 612| 536 | 30132 | 185913 | 
| 91 | 33 xarray/plot/facetgrid.py | 441 | 455| 140 | 30272 | 185913 | 
| 92 | 34 asv_bench/benchmarks/indexing.py | 123 | 141| 147 | 30419 | 187325 | 
| 93 | 34 xarray/plot/utils.py | 266 | 287| 267 | 30686 | 187325 | 
| 94 | 34 xarray/plot/utils.py | 375 | 399| 205 | 30891 | 187325 | 
| 95 | 35 xarray/core/dask_array_compat.py | 1 | 99| 714 | 31605 | 188558 | 
| 96 | 35 xarray/plot/utils.py | 621 | 667| 320 | 31925 | 188558 | 


### Hint

```
Yes from me! 
I still think it's worth keeping the option though
+1! I'm too often too lazy to turn it on, what a shame! And also +1 for keeping the option.
+1
```

## Patch

```diff
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -20,7 +20,7 @@
     CMAP_SEQUENTIAL: "viridis",
     CMAP_DIVERGENT: "RdBu_r",
     KEEP_ATTRS: "default",
-    DISPLAY_STYLE: "text",
+    DISPLAY_STYLE: "html",
 }
 
 _JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])

```

## Test Patch

```diff
diff --git a/xarray/tests/test_options.py b/xarray/tests/test_options.py
--- a/xarray/tests/test_options.py
+++ b/xarray/tests/test_options.py
@@ -68,12 +68,12 @@ def test_nested_options():
 
 
 def test_display_style():
-    original = "text"
+    original = "html"
     assert OPTIONS["display_style"] == original
     with pytest.raises(ValueError):
         xarray.set_options(display_style="invalid_str")
-    with xarray.set_options(display_style="html"):
-        assert OPTIONS["display_style"] == "html"
+    with xarray.set_options(display_style="text"):
+        assert OPTIONS["display_style"] == "text"
     assert OPTIONS["display_style"] == original
 
 
@@ -177,10 +177,11 @@ def test_merge_attr_retention(self):
 
     def test_display_style_text(self):
         ds = create_test_dataset_attrs()
-        text = ds._repr_html_()
-        assert text.startswith("<pre>")
-        assert "&#x27;nested&#x27;" in text
-        assert "&lt;xarray.Dataset&gt;" in text
+        with xarray.set_options(display_style="text"):
+            text = ds._repr_html_()
+            assert text.startswith("<pre>")
+            assert "&#x27;nested&#x27;" in text
+            assert "&lt;xarray.Dataset&gt;" in text
 
     def test_display_style_html(self):
         ds = create_test_dataset_attrs()
@@ -191,9 +192,10 @@ def test_display_style_html(self):
 
     def test_display_dataarray_style_text(self):
         da = create_test_dataarray_attrs()
-        text = da._repr_html_()
-        assert text.startswith("<pre>")
-        assert "&lt;xarray.DataArray &#x27;var1&#x27;" in text
+        with xarray.set_options(display_style="text"):
+            text = da._repr_html_()
+            assert text.startswith("<pre>")
+            assert "&lt;xarray.DataArray &#x27;var1&#x27;" in text
 
     def test_display_dataarray_style_html(self):
         da = create_test_dataarray_attrs()

```


## Code snippets

### 1 - doc/conf.py:

Start line: 215, End line: 350

```python
ipython_savefig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_build", "html", "_static"
)
if not os.path.exists(ipython_savefig_dir):
    os.makedirs(ipython_savefig_dir)

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True
htmlhelp_basename = "xarraydoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
latex_documents = [
    ("index", "xarray.tex", "xarray Documentation", "xarray Developers", "manual")
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
man_pages = [("index", "xarray", "xarray Documentation", ["xarray Developers"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
texinfo_documents = [
    (
        "index",
        "xarray",
        "xarray Documentation",
        "xarray Developers",
        "xarray",
        "N-D labeled arrays and datasets in Python.",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# Example configuration for intersphinx: refer to the Python standard library.
```
### 2 - xarray/core/formatting_html.py:

Start line: 200, End line: 236

```python
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


def _obj_repr(header_components, sections):
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    return (
        "<div>"
        f"{ICONS_SVG}<style>{CSS_STYLE}</style>"
        "<div class='xr-wrap'>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )
```
### 3 - doc/conf.py:

Start line: 111, End line: 214

```python
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "xarray"
copyright = "2014-%s, xarray Developers" % datetime.datetime.now().year

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = xarray.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = xarray.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"logo_only": True}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

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

# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
```
### 4 - xarray/core/formatting_html.py:

Start line: 239, End line: 275

```python
def array_repr(arr):
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))

    obj_type = "xarray.{}".format(type(arr).__name__)
    arr_name = "'{}'".format(arr.name) if getattr(arr, "name", None) else ""
    coord_names = list(arr.coords) if hasattr(arr, "coords") else []

    header_components = [
        "<div class='xr-obj-type'>{}</div>".format(obj_type),
        "<div class='xr-array-name'>{}</div>".format(arr_name),
        format_dims(dims, coord_names),
    ]

    sections = [array_section(arr)]

    if hasattr(arr, "coords"):
        sections.append(coord_section(arr.coords))

    sections.append(attr_section(arr.attrs))

    return _obj_repr(header_components, sections)


def dataset_repr(ds):
    obj_type = "xarray.{}".format(type(ds).__name__)

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        attr_section(ds.attrs),
    ]

    return _obj_repr(header_components, sections)
```
### 5 - xarray/core/formatting_html.py:

Start line: 182, End line: 197

```python
def array_section(obj):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    collapsed = ""
    preview = escape(inline_variable_array_repr(obj.variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon("icon-database")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<pre class='xr-array-data'>{data_repr}</pre>"
        "</div>"
    )
```
### 6 - xarray/core/formatting_html.py:

Start line: 1, End line: 23

```python
import uuid
from collections import OrderedDict
from functools import partial
from html import escape

import pkg_resources

from .formatting import inline_variable_array_repr, short_data_repr

CSS_FILE_PATH = "/".join(("static", "css", "style.css"))
CSS_STYLE = pkg_resources.resource_string("xarray", CSS_FILE_PATH).decode("utf8")


ICONS_SVG_PATH = "/".join(("static", "html", "icons-svg-inline.html"))
ICONS_SVG = pkg_resources.resource_string("xarray", ICONS_SVG_PATH).decode("utf8")


def short_data_repr_html(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()
    return escape(short_data_repr(array))
```
### 7 - xarray/core/formatting_html.py:

Start line: 93, End line: 127

```python
def summarize_variable(name, var, is_index=False, dtype=None, preview=None):
    variable = var.variable if hasattr(var, "variable") else var

    cssclass_idx = " class='xr-has-index'" if is_index else ""
    dims_str = f"({', '.join(escape(dim) for dim in var.dims)})"
    name = escape(name)
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
        f"<pre class='xr-var-data'>{data_repr}</pre>"
    )
```
### 8 - xarray/core/formatting.py:

Start line: 209, End line: 235

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
### 9 - xarray/core/options.py:

Start line: 1, End line: 75

```python
import warnings

DISPLAY_WIDTH = "display_width"
ARITHMETIC_JOIN = "arithmetic_join"
ENABLE_CFTIMEINDEX = "enable_cftimeindex"
FILE_CACHE_MAXSIZE = "file_cache_maxsize"
WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
CMAP_SEQUENTIAL = "cmap_sequential"
CMAP_DIVERGENT = "cmap_divergent"
KEEP_ATTRS = "keep_attrs"
DISPLAY_STYLE = "display_style"


OPTIONS = {
    DISPLAY_WIDTH: 80,
    ARITHMETIC_JOIN: "inner",
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    WARN_FOR_UNCLOSED_FILES: False,
    CMAP_SEQUENTIAL: "viridis",
    CMAP_DIVERGENT: "RdBu_r",
    KEEP_ATTRS: "default",
    DISPLAY_STYLE: "text",
}

_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    DISPLAY_WIDTH: _positive_integer,
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
    KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
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
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
    ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
}


def _get_keep_attrs(default):
    global_choice = OPTIONS["keep_attrs"]

    if global_choice == "default":
        return default
    elif global_choice in [True, False]:
        return global_choice
    else:
        raise ValueError(
            "The global option keep_attrs must be one of" " True, False or 'default'."
        )
```
### 10 - doc/conf.py:

Start line: 1, End line: 110

```python
# -*- coding: utf-8 -*-
#
# xarray documentation build configuration file, created by
# sphinx-quickstart on Thu Feb  6 18:57:54 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


import datetime
import os
import pathlib
import subprocess
import sys
from contextlib import suppress

# make sure the source version is preferred (#3567)
root = pathlib.Path(__file__).absolute().parent.parent
os.environ["PYTHONPATH"] = str(root)
sys.path.insert(0, str(root))

import xarray  # isort:skip

allowed_failures = set()

print("python exec:", sys.executable)
print("sys.path:", sys.path)

if "conda" in sys.executable:
    print("conda environment:")
    subprocess.run(["conda", "list"])
else:
    print("pip environment:")
    subprocess.run(["pip", "list"])

print("xarray: %s, %s" % (xarray.__version__, xarray.__file__))

with suppress(ImportError):
    import matplotlib

    matplotlib.use("Agg")

try:
    import rasterio
except ImportError:
    allowed_failures.update(
        ["gallery/plot_rasterio_rgb.py", "gallery/plot_rasterio.py"]
    )

try:
    import cartopy
except ImportError:
    allowed_failures.update(
        [
            "gallery/plot_cartopy_facetgrid.py",
            "gallery/plot_rasterio_rgb.py",
            "gallery/plot_rasterio.py",
        ]
    )

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]

extlinks = {
    "issue": ("https://github.com/pydata/xarray/issues/%s", "GH"),
    "pull": ("https://github.com/pydata/xarray/pull/%s", "PR"),
}

nbsphinx_timeout = 600
nbsphinx_execute = "always"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

You can run this notebook in a `live session <https://mybinder.org/v2/gh/pydata/xarray/doc/examples/master?urlpath=lab/tree/doc/{{ docname }}>`_ |Binder| or view it `on Github <https://github.com/pydata/xarray/blob/master/doc/{{ docname }}>`_.

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pydata/xarray/master?urlpath=lab/tree/doc/{{ docname }}
"""

autosummary_generate = True
autodoc_typehints = "none"

napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_class_members_toctree = True
```
### 16 - xarray/core/options.py:

Start line: 78, End line: 150

```python
class set_options:
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
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

    >>> ds = xr.Dataset({'x': np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    <xarray.Dataset>
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 0 1 2 3 4 5 6 ...
    Data variables:
        *empty*

    Or to set global options:

    >>> xr.set_options(display_width=80)
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
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
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
