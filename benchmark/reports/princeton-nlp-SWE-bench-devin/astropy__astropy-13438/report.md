# astropy__astropy-13438

| **astropy/astropy** | `4bd88be61fdf4185b9c198f7e689a40041e392ee` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 9532 |
| **Any found context length** | 8582 |
| **Avg pos** | 41.0 |
| **Min pos** | 19 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/table/jsviewer.py b/astropy/table/jsviewer.py
--- a/astropy/table/jsviewer.py
+++ b/astropy/table/jsviewer.py
@@ -15,7 +15,7 @@ class Conf(_config.ConfigNamespace):
     """
 
     jquery_url = _config.ConfigItem(
-        'https://code.jquery.com/jquery-3.1.1.min.js',
+        'https://code.jquery.com/jquery-3.6.0.min.js',
         'The URL to the jquery library.')
 
     datatables_url = _config.ConfigItem(
@@ -134,7 +134,7 @@ def __init__(self, use_local_files=False, display_length=50):
     @property
     def jquery_urls(self):
         if self._use_local_files:
-            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.1.1.min.js'),
+            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.6.0.min.js'),
                     'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min.js')]
         else:
             return [conf.jquery_url, conf.datatables_url]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/table/jsviewer.py | 18 | 18 | 22 | 1 | 9532
| astropy/table/jsviewer.py | 137 | 137 | 19 | 1 | 8582


## Problem Statement

```
[Security] Jquery 3.1.1 is vulnerable to untrusted code execution
<!-- This comments are hidden when you submit the issue,
so you do not need to remove them! -->

<!-- Please be sure to check out our contributing guidelines,
https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md .
Please be sure to check out our code of conduct,
https://github.com/astropy/astropy/blob/main/CODE_OF_CONDUCT.md . -->

<!-- Please have a search on our GitHub repository to see if a similar
issue has already been posted.
If a similar issue is closed, have a quick look to see if you are satisfied
by the resolution.
If not please go ahead and open an issue! -->

<!-- Please check that the development version still produces the same bug.
You can install development version with
pip install git+https://github.com/astropy/astropy
command. -->

### Description
<!-- Provide a general description of the bug. -->
Passing HTML from untrusted sources - even after sanitizing it - to one of jQuery's DOM manipulation methods (i.e. .html(), .append(), and others) may execute untrusted code (see [CVE-2020-11022](https://nvd.nist.gov/vuln/detail/cve-2020-11022) and [CVE-2020-11023](https://nvd.nist.gov/vuln/detail/cve-2020-11023))

### Expected behavior
<!-- What did you expect to happen. -->
Update jquery to the version 3.5 or newer in https://github.com/astropy/astropy/tree/main/astropy/extern/jquery/data/js

### Actual behavior
<!-- What actually happened. -->
<!-- Was the output confusing or poorly described? -->
 jquery version 3.1.1 is distributed with the latest astropy release

<!-- ### Steps to Reproduce 
<!-- Ideally a code example could be provided so we can run it ourselves. -->
<!-- If you are pasting code, use triple backticks (\`\`\`) around
your code snippet. -->
<!-- If necessary, sanitize your screen output to be pasted so you do not
reveal secrets like tokens and passwords. -->
<!--
1. [First Step]
2. [Second Step]
3. [and so on...]

\`\`\`python
# Put your Python code snippet here.
\`\`\`
-->
<!--### System Details
<!-- Even if you do not think this is necessary, it is useful information for the maintainers.
Please run the following snippet and paste the output below:
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("Numpy", numpy.__version__)
import erfa; print("pyerfa", erfa.__version__)
import astropy; print("astropy", astropy.__version__)
import scipy; print("Scipy", scipy.__version__)
import matplotlib; print("Matplotlib", matplotlib.__version__)
-->

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/table/jsviewer.py** | 30 | 105| 623 | 623 | 1610 | 
| 2 | 2 docs/conf.py | 261 | 346| 779 | 1402 | 5926 | 
| 3 | 3 astropy/utils/iers/iers.py | 12 | 78| 762 | 2164 | 17539 | 
| 4 | 4 astropy/utils/compat/numpycompat.py | 7 | 24| 269 | 2433 | 17849 | 
| 5 | 4 docs/conf.py | 27 | 100| 786 | 3219 | 17849 | 
| 6 | 5 astropy/version.py | 1 | 40| 232 | 3451 | 18082 | 
| 7 | 6 astropy/io/votable/validator/html.py | 4 | 83| 486 | 3937 | 20378 | 
| 8 | 7 astropy/table/pandas.py | 10 | 32| 126 | 4063 | 21191 | 
| 9 | 8 astropy/io/misc/pandas/connect.py | 33 | 52| 128 | 4191 | 22127 | 
| 10 | 9 astropy/utils/data.py | 5 | 81| 483 | 4674 | 39943 | 
| 11 | 10 astropy/io/votable/exceptions.py | 630 | 639| 116 | 4790 | 53267 | 
| 12 | 11 astropy/constants/astropyconst40.py | 7 | 50| 483 | 5273 | 53805 | 
| 13 | 12 astropy/constants/astropyconst20.py | 7 | 50| 483 | 5756 | 54343 | 
| 14 | 13 examples/template/example-template.py | 24 | 99| 553 | 6309 | 55024 | 
| 15 | 14 astropy/constants/constant.py | 33 | 74| 379 | 6688 | 56935 | 
| 16 | 14 docs/conf.py | 164 | 260| 897 | 7585 | 56935 | 
| 17 | 15 astropy/utils/compat/optional_deps.py | 5 | 22| 223 | 7808 | 57357 | 
| 18 | 16 astropy/__init__.py | 9 | 41| 237 | 8045 | 59115 | 
| **-> 19 <-** | **16 astropy/table/jsviewer.py** | 108 | 169| 537 | 8582 | 59115 | 
| 20 | 16 docs/conf.py | 101 | 163| 672 | 9254 | 59115 | 
| 21 | 17 astropy/io/misc/asdf/deprecation.py | 1 | 10| 95 | 9349 | 59210 | 
| **-> 22 <-** | **17 astropy/table/jsviewer.py** | 3 | 27| 183 | 9532 | 59210 | 
| 23 | 18 astropy/coordinates/sites.py | 100 | 118| 190 | 9722 | 60299 | 
| 24 | 18 astropy/utils/data.py | 1806 | 1845| 396 | 10118 | 60299 | 
| 25 | 19 astropy/io/votable/validator/main.py | 76 | 161| 749 | 10867 | 61484 | 
| 26 | 20 astropy/io/ascii/html.py | 372 | 456| 775 | 11642 | 65019 | 
| 27 | 20 astropy/io/votable/validator/main.py | 8 | 43| 183 | 11825 | 65019 | 
| 28 | 21 astropy/extern/configobj/validate.py | 356 | 443| 679 | 12504 | 77651 | 
| 29 | 21 astropy/extern/configobj/validate.py | 1420 | 1456| 348 | 12852 | 77651 | 
| 30 | 22 astropy/io/fits/util.py | 3 | 30| 125 | 12977 | 84688 | 
| 31 | 23 astropy/wcs/wcs.py | 31 | 70| 319 | 13296 | 115547 | 
| 32 | 24 astropy/stats/jackknife.py | 145 | 177| 287 | 13583 | 117246 | 
| 33 | 24 astropy/utils/data.py | 1132 | 1222| 759 | 14342 | 117246 | 
| 34 | 24 astropy/extern/configobj/validate.py | 1 | 217| 679 | 15021 | 117246 | 
| 35 | 25 astropy/conftest.py | 7 | 61| 375 | 15396 | 118409 | 
| 36 | 25 astropy/io/votable/exceptions.py | 412 | 441| 214 | 15610 | 118409 | 
| 37 | 26 astropy/wcs/wcsapi/sliced_low_level_wcs.py | 1 | 11| 97 | 15707 | 118506 | 
| 38 | 26 astropy/io/votable/validator/main.py | 46 | 73| 202 | 15909 | 118506 | 
| 39 | 26 docs/conf.py | 415 | 436| 178 | 16087 | 118506 | 
| 40 | 26 astropy/utils/data.py | 1089 | 1129| 467 | 16554 | 118506 | 
| 41 | 27 astropy/nddata/nduncertainty.py | 3 | 28| 178 | 16732 | 126403 | 
| 42 | 28 astropy/utils/compat/__init__.py | 16 | 20| 33 | 16765 | 126552 | 
| 43 | 29 astropy/table/__init__.py | 3 | 14| 188 | 16953 | 127397 | 
| 44 | 29 astropy/io/votable/validator/html.py | 86 | 101| 167 | 17120 | 127397 | 
| 45 | 29 astropy/__init__.py | 205 | 230| 291 | 17411 | 127397 | 
| 46 | 30 astropy/io/ascii/__init__.py | 7 | 48| 302 | 17713 | 127735 | 
| 47 | 31 astropy/convolution/utils.py | 2 | 31| 143 | 17856 | 130672 | 
| 48 | 31 astropy/wcs/wcs.py | 3489 | 3541| 341 | 18197 | 130672 | 
| 49 | 32 astropy/io/ascii/ui.py | 12 | 48| 193 | 18390 | 138571 | 
| 50 | 33 astropy/io/votable/table.py | 317 | 336| 151 | 18541 | 141686 | 
| 51 | 34 astropy/constants/iau2015.py | 73 | 98| 356 | 18897 | 142792 | 
| 52 | 35 astropy/utils/decorators.py | 161 | 204| 350 | 19247 | 151313 | 
| 53 | 35 docs/conf.py | 347 | 365| 163 | 19410 | 151313 | 
| 54 | 35 astropy/extern/configobj/validate.py | 628 | 657| 281 | 19691 | 151313 | 
| 55 | 36 astropy/constants/__init__.py | 16 | 57| 259 | 19950 | 151692 | 
| 56 | 37 astropy/config/configuration.py | 12 | 81| 497 | 20447 | 158099 | 
| 57 | 37 astropy/io/votable/table.py | 212 | 316| 757 | 21204 | 158099 | 
| 58 | **37 astropy/table/jsviewer.py** | 172 | 201| 250 | 21454 | 158099 | 
| 59 | 38 astropy/coordinates/errors.py | 6 | 23| 114 | 21568 | 159163 | 
| 60 | 38 astropy/io/votable/validator/html.py | 104 | 114| 138 | 21706 | 159163 | 
| 61 | 39 astropy/utils/misc.py | 861 | 891| 215 | 21921 | 165938 | 
| 62 | 40 conftest.py | 6 | 65| 598 | 22519 | 166582 | 
| 63 | 41 astropy/time/formats.py | 1919 | 1933| 158 | 22677 | 185828 | 
| 64 | 41 astropy/extern/configobj/validate.py | 1459 | 1473| 116 | 22793 | 185828 | 
| 65 | 42 astropy/visualization/mpl_style.py | 7 | 84| 607 | 23400 | 186496 | 
| 66 | 42 astropy/extern/configobj/validate.py | 446 | 457| 106 | 23506 | 186496 | 
| 67 | 42 astropy/io/ascii/ui.py | 231 | 250| 236 | 23742 | 186496 | 
| 68 | 43 astropy/io/ascii/ecsv.py | 7 | 31| 187 | 23929 | 190693 | 
| 69 | 43 astropy/utils/compat/optional_deps.py | 25 | 43| 146 | 24075 | 190693 | 
| 70 | 43 astropy/nddata/nduncertainty.py | 204 | 235| 348 | 24423 | 190693 | 
| 71 | 43 astropy/io/votable/validator/html.py | 117 | 159| 330 | 24753 | 190693 | 
| 72 | 44 astropy/visualization/scripts/__init__.py | 1 | 2| 16 | 24769 | 190710 | 
| 73 | 45 astropy/io/votable/validator/__init__.py | 1 | 7| 40 | 24809 | 190751 | 
| 74 | 45 astropy/utils/misc.py | 7 | 83| 440 | 25249 | 190751 | 
| 75 | 45 astropy/table/__init__.py | 47 | 78| 338 | 25587 | 190751 | 
| 76 | 46 astropy/nddata/__init__.py | 11 | 29| 102 | 25689 | 191106 | 
| 77 | 46 astropy/io/votable/exceptions.py | 657 | 700| 339 | 26028 | 191106 | 
| 78 | 46 astropy/conftest.py | 126 | 147| 190 | 26218 | 191106 | 
| 79 | 47 astropy/stats/__init__.py | 14 | 46| 280 | 26498 | 191499 | 
| 80 | 47 astropy/io/votable/exceptions.py | 728 | 740| 156 | 26654 | 191499 | 
| 81 | 48 astropy/units/_typing.py | 6 | 21| 84 | 26738 | 191623 | 
| 82 | 49 astropy/visualization/wcsaxes/patches.py | 4 | 26| 242 | 26980 | 193303 | 
| 83 | 50 astropy/units/astrophys.py | 144 | 162| 140 | 27120 | 194814 | 
| 84 | 50 astropy/io/votable/exceptions.py | 480 | 497| 227 | 27347 | 194814 | 
| 85 | 50 astropy/utils/iers/iers.py | 786 | 832| 525 | 27872 | 194814 | 
| 86 | 51 astropy/coordinates/funcs.py | 11 | 27| 137 | 28009 | 198050 | 
| 87 | 52 astropy/_dev/scm_version.py | 1 | 10| 80 | 28089 | 198130 | 


### Hint

```
Welcome to Astropy 👋 and thank you for your first issue!

A project member will respond to you as soon as possible; in the meantime, please double-check the [guidelines for submitting issues](https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md#reporting-issues) and make sure you've provided the requested details.

GitHub issues in the Astropy repository are used to track bug reports and feature requests; If your issue poses a question about how to use Astropy, please instead raise your question in the [Astropy Discourse user forum](https://community.openastronomy.org/c/astropy/8) and close this issue.

If you feel that this issue has not been responded to in a timely manner, please leave a comment mentioning our software support engineer @embray, or send a message directly to the [development mailing list](http://groups.google.com/group/astropy-dev).  If the issue is urgent or sensitive in nature (e.g., a security vulnerability) please send an e-mail directly to the private e-mail feedback@astropy.org.
Besides the jquery files  in [astropy/extern/jquery/data/js/](https://github.com/astropy/astropy/tree/main/astropy/extern/jquery/data/js), the jquery version number appears in [astropy/table/jsviewer.py](https://github.com/astropy/astropy/blob/main/astropy/table/jsviewer.py) twice, and in [table/tests/test_jsviewer.py](https://github.com/astropy/astropy/blob/main/astropy/table/tests/test_jsviewer.py) four times. This might be a good time to introduce a constant for the jquery version, and use that ~across the codebase. Or at least~ across the tests.

@skukhtichev Maybe we could speed up the fix by creating a PR?
As Python does not have built-in support for defining constants, I think it's better to keep the hard-coded strings in [astropy/table/jsviewer.py](https://github.com/astropy/astropy/blob/main/astropy/table/jsviewer.py). Don't want to introduce another security problem by allowing attackers to downgrade the jquery version at runtime. Still, a variable for the tests would simplify future updates.
> Maybe we could speed up the fix by creating a PR?

That would definitely help! 😸 

We discussed this in Astropy Slack (https://www.astropy.org/help.html) and had a few ideas, the latest being download the updated files from https://cdn.datatables.net/ but no one has the time to actually do anything yet.

We usually do not modify the bundled code (unless there is no choice) but rather just copy them over. This is because your changes will get lost in the next upgrade unless we have a patch file on hand with instructions (though that can easily break too if upstream has changed too much).
I'll see what I can do about a PR tomorrow :-)
I'd get the jquery update from https://releases.jquery.com/jquery/, latest version is 3.6.0.
```

## Patch

```diff
diff --git a/astropy/table/jsviewer.py b/astropy/table/jsviewer.py
--- a/astropy/table/jsviewer.py
+++ b/astropy/table/jsviewer.py
@@ -15,7 +15,7 @@ class Conf(_config.ConfigNamespace):
     """
 
     jquery_url = _config.ConfigItem(
-        'https://code.jquery.com/jquery-3.1.1.min.js',
+        'https://code.jquery.com/jquery-3.6.0.min.js',
         'The URL to the jquery library.')
 
     datatables_url = _config.ConfigItem(
@@ -134,7 +134,7 @@ def __init__(self, use_local_files=False, display_length=50):
     @property
     def jquery_urls(self):
         if self._use_local_files:
-            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.1.1.min.js'),
+            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.6.0.min.js'),
                     'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min.js')]
         else:
             return [conf.jquery_url, conf.datatables_url]

```

## Test Patch

```diff
diff --git a/astropy/table/tests/test_jsviewer.py b/astropy/table/tests/test_jsviewer.py
--- a/astropy/table/tests/test_jsviewer.py
+++ b/astropy/table/tests/test_jsviewer.py
@@ -13,6 +13,8 @@
 from astropy.utils.misc import _NOT_OVERWRITING_MSG_MATCH
 
 EXTERN_DIR = abspath(join(dirname(extern.__file__), 'jquery', 'data'))
+JQUERY_MIN_JS = 'jquery-3.6.0.min.js'
+
 
 REFERENCE = """
 <html>
@@ -101,7 +103,7 @@ def test_write_jsviewer_default(tmpdir):
         display_length='10, 25, 50, 100, 500, 1000',
         datatables_css_url='https://cdn.datatables.net/1.10.12/css/jquery.dataTables.css',
         datatables_js_url='https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js',
-        jquery_url='https://code.jquery.com/jquery-3.1.1.min.js'
+        jquery_url='https://code.jquery.com/' + JQUERY_MIN_JS
     )
     with open(tmpfile) as f:
         assert f.read().strip() == ref.strip()
@@ -144,7 +146,7 @@ def test_write_jsviewer_mixin(tmpdir, mixin):
         display_length='10, 25, 50, 100, 500, 1000',
         datatables_css_url='https://cdn.datatables.net/1.10.12/css/jquery.dataTables.css',
         datatables_js_url='https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js',
-        jquery_url='https://code.jquery.com/jquery-3.1.1.min.js'
+        jquery_url='https://code.jquery.com/' + JQUERY_MIN_JS
     )
     with open(tmpfile) as f:
         assert f.read().strip() == ref.strip()
@@ -170,7 +172,7 @@ def test_write_jsviewer_options(tmpdir):
         display_length='5, 10, 25, 50, 100, 500, 1000',
         datatables_css_url='https://cdn.datatables.net/1.10.12/css/jquery.dataTables.css',
         datatables_js_url='https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js',
-        jquery_url='https://code.jquery.com/jquery-3.1.1.min.js'
+        jquery_url='https://code.jquery.com/' + JQUERY_MIN_JS
     )
     with open(tmpfile) as f:
         assert f.read().strip() == ref.strip()
@@ -194,7 +196,7 @@ def test_write_jsviewer_local(tmpdir):
         display_length='10, 25, 50, 100, 500, 1000',
         datatables_css_url='file://' + join(EXTERN_DIR, 'css', 'jquery.dataTables.css'),
         datatables_js_url='file://' + join(EXTERN_DIR, 'js', 'jquery.dataTables.min.js'),
-        jquery_url='file://' + join(EXTERN_DIR, 'js', 'jquery-3.1.1.min.js')
+        jquery_url='file://' + join(EXTERN_DIR, 'js', JQUERY_MIN_JS)
     )
     with open(tmpfile) as f:
         assert f.read().strip() == ref.strip()

```


## Code snippets

### 1 - astropy/table/jsviewer.py:

Start line: 30, End line: 105

```python
conf = Conf()


EXTERN_JS_DIR = abspath(join(dirname(extern.__file__), 'jquery', 'data', 'js'))
EXTERN_CSS_DIR = abspath(join(dirname(extern.__file__), 'jquery', 'data', 'css'))

_SORTING_SCRIPT_PART_1 = """
var astropy_sort_num = function(a, b) {{
    var a_num = parseFloat(a);
    var b_num = parseFloat(b);

    if (isNaN(a_num) && isNaN(b_num))
        return ((a < b) ? -1 : ((a > b) ? 1 : 0));
    else if (!isNaN(a_num) && !isNaN(b_num))
        return ((a_num < b_num) ? -1 : ((a_num > b_num) ? 1 : 0));
    else
        return isNaN(a_num) ? -1 : 1;
}}
"""

_SORTING_SCRIPT_PART_2 = """
jQuery.extend( jQuery.fn.dataTableExt.oSort, {{
    "optionalnum-asc": astropy_sort_num,
    "optionalnum-desc": function (a,b) {{ return -astropy_sort_num(a, b); }}
}});
"""

IPYNB_JS_SCRIPT = """
<script>
%(sorting_script1)s
require.config({{paths: {{
    datatables: '{datatables_url}'
}}}});
require(["datatables"], function(){{
    console.log("$('#{tid}').dataTable()");
    %(sorting_script2)s
    $('#{tid}').dataTable({{
        order: [],
        pageLength: {display_length},
        lengthMenu: {display_length_menu},
        pagingType: "full_numbers",
        columnDefs: [{{targets: {sort_columns}, type: "optionalnum"}}]
    }});
}});
</script>
""" % dict(sorting_script1=_SORTING_SCRIPT_PART_1,
           sorting_script2=_SORTING_SCRIPT_PART_2)

HTML_JS_SCRIPT = _SORTING_SCRIPT_PART_1 + _SORTING_SCRIPT_PART_2 + """
$(document).ready(function() {{
    $('#{tid}').dataTable({{
        order: [],
        pageLength: {display_length},
        lengthMenu: {display_length_menu},
        pagingType: "full_numbers",
        columnDefs: [{{targets: {sort_columns}, type: "optionalnum"}}]
    }});
}} );
"""


# Default CSS for the JSViewer writer
DEFAULT_CSS = """\
body {font-family: sans-serif;}
table.dataTable {width: auto !important; margin: 0 !important;}
.dataTables_filter, .dataTables_paginate {float: left !important; margin-left:1em}
"""


# Default CSS used when rendering a table in the IPython notebook
DEFAULT_CSS_NB = """\
table.dataTable {clear: both; width: auto !important; margin: 0 !important;}
.dataTables_info, .dataTables_length, .dataTables_filter, .dataTables_paginate{
display: inline-block; margin-right: 1em; }
.paginate_button { margin-right: 5px; }
"""
```
### 2 - docs/conf.py:

Start line: 261, End line: 346

```python
htmlhelp_basename = project + 'doc'

# A dictionary of values to pass into the template engine’s context for all pages.
html_context = {
    'to_be_indexed': ['stable', 'latest'],
    'is_development': dev
}

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
latex_documents = [('index', project + '.tex', project + ' Documentation',
                    author, 'manual')]

latex_logo = '_static/astropy_logo.pdf'


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
man_pages = [('index', project.lower(), project + ' Documentation',
              [author], 1)]

# Setting this URL is requited by sphinx-astropy
github_issues_url = 'https://github.com/astropy/astropy/issues/'
edit_on_github_branch = 'main'

# Enable nitpicky mode - which ensures that all references in the docs
# resolve.

nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []

for line in open('nitpick-exceptions'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

# -- Options for the Sphinx gallery -------------------------------------------

try:
    import warnings

    import sphinx_gallery  # noqa: F401
    extensions += ["sphinx_gallery.gen_gallery"]  # noqa: F405

    sphinx_gallery_conf = {
        'backreferences_dir': 'generated/modules',  # path to store the module using example template  # noqa: E501
        'filename_pattern': '^((?!skip_).)*$',  # execute all examples except those that start with "skip_"  # noqa: E501
        'examples_dirs': f'..{os.sep}examples',  # path to the examples scripts
        'gallery_dirs': 'generated/examples',  # path to save gallery generated examples
        'reference_url': {
            'astropy': None,
            'matplotlib': 'https://matplotlib.org/stable/',
            'numpy': 'https://numpy.org/doc/stable/',
        },
        'abort_on_example_error': True
    }

    # Filter out backend-related warnings as described in
    # https://github.com/sphinx-gallery/sphinx-gallery/pull/564
    warnings.filterwarnings("ignore", category=UserWarning,
                            message='Matplotlib is currently using agg, which is a'
                                    ' non-GUI backend, so cannot show the figure.')

except ImportError:
    sphinx_gallery = None


# -- Options for linkcheck output -------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = ['https://journals.aas.org/manuscript-preparation/',
                    'https://maia.usno.navy.mil/',
                    'https://www.usno.navy.mil/USNO/time/gps/usno-gps-time-transfer',
                    'https://aa.usno.navy.mil/publications/docs/Circular_179.php',
                    'http://data.astropy.org',
                    'https://doi.org/10.1017/S0251107X00002406',  # internal server error
                    'https://doi.org/10.1017/pasa.2013.31',  # internal server error
                    'https://pyfits.readthedocs.io/en/v3.2.1/',  # defunct page in CHANGES.rst
                    r'https://github\.com/astropy/astropy/(?:issues|pull)/\d+']
linkcheck_timeout = 180
```
### 3 - astropy/utils/iers/iers.py:

Start line: 12, End line: 78

```python
import re
from datetime import datetime
from warnings import warn
from urllib.parse import urlparse

import numpy as np
import erfa

from astropy.time import Time, TimeDelta
from astropy import config as _config
from astropy import units as u
from astropy.table import QTable, MaskedColumn
from astropy.utils.data import (get_pkg_data_filename, clear_download_cache,
                                is_url_in_cache, get_readable_fileobj)
from astropy.utils.state import ScienceState
from astropy import utils
from astropy.utils.exceptions import AstropyWarning

__all__ = ['Conf', 'conf', 'earth_orientation_table',
           'IERS', 'IERS_B', 'IERS_A', 'IERS_Auto',
           'FROM_IERS_B', 'FROM_IERS_A', 'FROM_IERS_A_PREDICTION',
           'TIME_BEFORE_IERS_RANGE', 'TIME_BEYOND_IERS_RANGE',
           'IERS_A_FILE', 'IERS_A_URL', 'IERS_A_URL_MIRROR', 'IERS_A_README',
           'IERS_B_FILE', 'IERS_B_URL', 'IERS_B_README',
           'IERSRangeError', 'IERSStaleWarning', 'IERSWarning',
           'IERSDegradedAccuracyWarning',
           'LeapSeconds', 'IERS_LEAP_SECOND_FILE', 'IERS_LEAP_SECOND_URL',
           'IETF_LEAP_SECOND_URL']

# IERS-A default file name, URL, and ReadMe with content description
IERS_A_FILE = 'finals2000A.all'
IERS_A_URL = 'https://datacenter.iers.org/data/9/finals2000A.all'
IERS_A_URL_MIRROR = 'https://maia.usno.navy.mil/ser7/finals2000A.all'
IERS_A_README = get_pkg_data_filename('data/ReadMe.finals2000A')

# IERS-B default file name, URL, and ReadMe with content description
IERS_B_FILE = get_pkg_data_filename('data/eopc04_IAU2000.62-now')
IERS_B_URL = 'http://hpiers.obspm.fr/iers/eop/eopc04/eopc04_IAU2000.62-now'
IERS_B_README = get_pkg_data_filename('data/ReadMe.eopc04_IAU2000')

# LEAP SECONDS default file name, URL, and alternative format/URL
IERS_LEAP_SECOND_FILE = get_pkg_data_filename('data/Leap_Second.dat')
IERS_LEAP_SECOND_URL = 'https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat'
IETF_LEAP_SECOND_URL = 'https://www.ietf.org/timezones/data/leap-seconds.list'

# Status/source values returned by IERS.ut1_utc
FROM_IERS_B = 0
FROM_IERS_A = 1
FROM_IERS_A_PREDICTION = 2
TIME_BEFORE_IERS_RANGE = -1
TIME_BEYOND_IERS_RANGE = -2

MJD_ZERO = 2400000.5

INTERPOLATE_ERROR = """\
interpolating from IERS_Auto using predictive values that are more
than {0} days old.

Normally you should not see this error because this class
automatically downloads the latest IERS-A table.  Perhaps you are
offline?  If you understand what you are doing then this error can be
suppressed by setting the auto_max_age configuration variable to
``None``:

  from astropy.utils.iers import conf
  conf.auto_max_age = None
"""
```
### 4 - astropy/utils/compat/numpycompat.py:

Start line: 7, End line: 24

```python
import numpy as np
from astropy.utils import minversion

__all__ = ['NUMPY_LT_1_19', 'NUMPY_LT_1_19_1', 'NUMPY_LT_1_20',
           'NUMPY_LT_1_21_1', 'NUMPY_LT_1_22', 'NUMPY_LT_1_22_1',
           'NUMPY_LT_1_23']

# TODO: It might also be nice to have aliases to these named for specific
# features/bugs we're checking for (ex:
# astropy.table.table._BROKEN_UNICODE_TABLE_SORT)
NUMPY_LT_1_19 = not minversion(np, '1.19')
NUMPY_LT_1_19_1 = not minversion(np, '1.19.1')
NUMPY_LT_1_20 = not minversion(np, '1.20')
NUMPY_LT_1_21_1 = not minversion(np, '1.21.1')
NUMPY_LT_1_22 = not minversion(np, '1.22')
NUMPY_LT_1_22_1 = not minversion(np, '1.22.1')
NUMPY_LT_1_23 = not minversion(np, '1.23dev0')
```
### 5 - docs/conf.py:

Start line: 27, End line: 100

```python
import os
import sys
import configparser
from datetime import datetime
from importlib import metadata

import doctest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

# -- Check for missing dependencies -------------------------------------------
missing_requirements = {}
for line in metadata.requires('astropy'):
    if 'extra == "docs"' in line:
        req = Requirement(line.split(';')[0])
        req_package = req.name.lower()
        req_specifier = str(req.specifier)

        try:
            version = metadata.version(req_package)
        except metadata.PackageNotFoundError:
            missing_requirements[req_package] = req_specifier

        if version not in SpecifierSet(req_specifier, prereleases=True):
            missing_requirements[req_package] = req_specifier

if missing_requirements:
    print('The following packages could not be found and are required to '
          'build the documentation:')
    for key, val in missing_requirements.items():
        print(f'    * {key} {val}')
    print('Please install the "docs" requirements.')
    sys.exit(1)

from sphinx_astropy.conf.v1 import *  # noqa

# -- Plot configuration -------------------------------------------------------
plot_rcparams = {}
plot_rcparams['figure.figsize'] = (6, 6)
plot_rcparams['savefig.facecolor'] = 'none'
plot_rcparams['savefig.bbox'] = 'tight'
plot_rcparams['axes.labelsize'] = 'large'
plot_rcparams['figure.subplot.hspace'] = 0.5

plot_apply_rcparams = True
plot_html_show_source_link = False
plot_formats = ['png', 'svg', 'pdf']
# Don't use the default - which includes a numpy and matplotlib import
plot_pre_code = ""

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.7'

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("X.Y.Z")` here.
check_sphinx_version("1.2.1")  # noqa: F405

# The intersphinx_mapping in sphinx_astropy.sphinx refers to astropy for
# the benefit of other packages who want to refer to objects in the
# astropy core.  However, we don't want to cyclically reference astropy in its
# own build so we remove it here.
del intersphinx_mapping['astropy']  # noqa: F405

# add any custom intersphinx for astropy
intersphinx_mapping['astropy-dev'] = ('https://docs.astropy.org/en/latest/', None)  # noqa: F405
intersphinx_mapping['pyerfa'] = ('https://pyerfa.readthedocs.io/en/stable/', None)  # noqa: F405
intersphinx_mapping['pytest'] = ('https://docs.pytest.org/en/stable/', None)  # noqa: F405
intersphinx_mapping['ipython'] = ('https://ipython.readthedocs.io/en/stable/', None)  # noqa: F405
intersphinx_mapping['pandas'] = ('https://pandas.pydata.org/pandas-docs/stable/', None)  # noqa: F405, E501
intersphinx_mapping['sphinx_automodapi'] = ('https://sphinx-automodapi.readthedocs.io/en/stable/', None)  # noqa: F405, E501
intersphinx_mapping['packagetemplate'] = ('https://docs.astropy.org/projects/package-template/en/latest/', None)  # noqa: F405, E501
intersphinx_mapping['h5py'] = ('https://docs.h5py.org/en/stable/', None)  # noqa: F405
```
### 6 - astropy/version.py:

Start line: 1, End line: 40

```python
# NOTE: First try _dev.scm_version if it exists and setuptools_scm is installed
# This file is not included in astropy wheels/tarballs, so otherwise it will
# fall back on the generated _version module.
try:
    try:
        from ._dev.scm_version import version
    except ImportError:
        from ._version import version
except Exception:
    import warnings
    warnings.warn(
        f'could not determine {__name__.split(".")[0]} package version; '
        f'this indicates a broken installation')
    del warnings

    version = '0.0.0'


# We use Version to define major, minor, micro, but ignore any suffixes.
def split_version(version):
    pieces = [0, 0, 0]

    try:
        from packaging.version import Version

        v = Version(version)
        pieces = [v.major, v.minor, v.micro]

    except Exception:
        pass

    return pieces


major, minor, bugfix = split_version(version)

del split_version  # clean up namespace.

release = 'dev' not in version
```
### 7 - astropy/io/votable/validator/html.py:

Start line: 4, End line: 83

```python
import contextlib
from math import ceil
import os
import re

# ASTROPY
from astropy.utils.xml.writer import XMLWriter, xml_escape
from astropy import online_docs_root

# VO
from astropy.io.votable import exceptions

html_header = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html
        PUBLIC "-//W3C//DTD XHTML Basic 1.0//EN"
        "http://www.w3.org/TR/xhtml-basic/xhtml-basic10.dtd">
"""

default_style = """
body {
font-family: sans-serif
}
a {
text-decoration: none
}
.highlight {
color: red;
font-weight: bold;
text-decoration: underline;
}
.green { background-color: #ddffdd }
.red   { background-color: #ffdddd }
.yellow { background-color: #ffffdd }
tr:hover { background-color: #dddddd }
table {
        border-width: 1px;
        border-spacing: 0px;
        border-style: solid;
        border-color: gray;
        border-collapse: collapse;
        background-color: white;
        padding: 5px;
}
table th {
        border-width: 1px;
        padding: 5px;
        border-style: solid;
        border-color: gray;
}
table td {
        border-width: 1px;
        padding: 5px;
        border-style: solid;
        border-color: gray;
}
"""


@contextlib.contextmanager
def make_html_header(w):
    w.write(html_header)
    with w.tag('html', xmlns="http://www.w3.org/1999/xhtml", lang="en-US"):
        with w.tag('head'):
            w.element('title', 'VO Validation results')
            w.element('style', default_style)

            with w.tag('body'):
                yield


def write_source_line(w, line, nchar=0):
    part1 = xml_escape(line[:nchar].decode('utf-8'))
    char = xml_escape(line[nchar:nchar+1].decode('utf-8'))
    part2 = xml_escape(line[nchar+1:].decode('utf-8'))

    w.write('  ')
    w.write(part1)
    w.write(f'<span class="highlight">{char}</span>')
    w.write(part2)
    w.write('\n\n')
```
### 8 - astropy/table/pandas.py:

Start line: 10, End line: 32

```python
try:
    from IPython import display

    html = display.Image(url=url)._repr_html_()

    class HTMLWithBackup(display.HTML):
        def __init__(self, data, backup_text):
            super().__init__(data)
            self.backup_text = backup_text

        def __repr__(self):
            if self.backup_text is None:
                return super().__repr__()
            else:
                return self.backup_text

    dhtml = HTMLWithBackup(html, ascii_uncoded)
    display.display(dhtml)
except ImportError:
    print(ascii_uncoded)
except (UnicodeEncodeError, SyntaxError):
    pass
```
### 9 - astropy/io/misc/pandas/connect.py:

Start line: 33, End line: 52

```python
def import_html_libs():
    """Try importing dependencies for reading HTML.

    This is copied from pandas.io.html
    """
    # import things we need
    # but make this done on a first use basis

    global _IMPORTS
    if _IMPORTS:
        return

    global _HAS_BS4, _HAS_LXML, _HAS_HTML5LIB

    from astropy.utils.compat.optional_deps import (
        HAS_BS4 as _HAS_BS4,
        HAS_LXML as _HAS_LXML,
        HAS_HTML5LIB as _HAS_HTML5LIB
    )
    _IMPORTS = True
```
### 10 - astropy/utils/data.py:

Start line: 5, End line: 81

```python
import atexit
import contextlib
import errno
import fnmatch
import functools
import hashlib
import os
import io
import re
import shutil
# import ssl moved inside functions using ssl to avoid import failure
# when running in pyodide/Emscripten
import sys
import urllib.request
import urllib.error
import urllib.parse
import zipfile
import ftplib

from tempfile import NamedTemporaryFile, gettempdir, TemporaryDirectory, mkdtemp
from warnings import warn

try:
    import certifi
except ImportError:
    # certifi support is optional; when available it will be used for TLS/SSL
    # downloads
    certifi = None

import astropy.config.paths
from astropy import config as _config
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.introspection import find_current_module, resolve_name


# Order here determines order in the autosummary
__all__ = [
    'Conf', 'conf',
    'download_file', 'download_files_in_parallel',
    'get_readable_fileobj',
    'get_pkg_data_fileobj', 'get_pkg_data_filename',
    'get_pkg_data_contents', 'get_pkg_data_fileobjs',
    'get_pkg_data_filenames', 'get_pkg_data_path',
    'is_url', 'is_url_in_cache', 'get_cached_urls',
    'cache_total_size', 'cache_contents',
    'export_download_cache', 'import_download_cache', 'import_file_to_cache',
    'check_download_cache',
    'clear_download_cache',
    'compute_hash',
    'get_free_space_in_dir',
    'check_free_space_in_dir',
    'get_file_contents',
    'CacheMissingWarning',
    "CacheDamaged"
]

_dataurls_to_alias = {}


class _NonClosingBufferedReader(io.BufferedReader):
    def __del__(self):
        try:
            # NOTE: self.raw will not be closed, but left in the state
            # it was in at detactment
            self.detach()
        except Exception:
            pass


class _NonClosingTextIOWrapper(io.TextIOWrapper):
    def __del__(self):
        try:
            # NOTE: self.stream will not be closed, but left in the state
            # it was in at detactment
            self.detach()
        except Exception:
            pass
```
### 19 - astropy/table/jsviewer.py:

Start line: 108, End line: 169

```python
class JSViewer:
    """Provides an interactive HTML export of a Table.

    This class provides an interface to the `DataTables
    <https://datatables.net/>`_ library, which allow to visualize interactively
    an HTML table. It is used by the `~astropy.table.Table.show_in_browser`
    method.

    Parameters
    ----------
    use_local_files : bool, optional
        Use local files or a CDN for JavaScript libraries. Default False.
    display_length : int, optional
        Number or rows to show. Default to 50.

    """

    def __init__(self, use_local_files=False, display_length=50):
        self._use_local_files = use_local_files
        self.display_length_menu = [[10, 25, 50, 100, 500, 1000, -1],
                                    [10, 25, 50, 100, 500, 1000, "All"]]
        self.display_length = display_length
        for L in self.display_length_menu:
            if display_length not in L:
                L.insert(0, display_length)

    @property
    def jquery_urls(self):
        if self._use_local_files:
            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.1.1.min.js'),
                    'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min.js')]
        else:
            return [conf.jquery_url, conf.datatables_url]

    @property
    def css_urls(self):
        if self._use_local_files:
            return ['file://' + join(EXTERN_CSS_DIR,
                                     'jquery.dataTables.css')]
        else:
            return conf.css_urls

    def _jstable_file(self):
        if self._use_local_files:
            return 'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min')
        else:
            return conf.datatables_url[:-3]

    def ipynb(self, table_id, css=None, sort_columns='[]'):
        html = f'<style>{css if css is not None else DEFAULT_CSS_NB}</style>'
        html += IPYNB_JS_SCRIPT.format(
            display_length=self.display_length,
            display_length_menu=self.display_length_menu,
            datatables_url=self._jstable_file(),
            tid=table_id, sort_columns=sort_columns)
        return html

    def html_js(self, table_id='table0', sort_columns='[]'):
        return HTML_JS_SCRIPT.format(
            display_length=self.display_length,
            display_length_menu=self.display_length_menu,
            tid=table_id, sort_columns=sort_columns).strip()
```
### 22 - astropy/table/jsviewer.py:

Start line: 3, End line: 27

```python
from os.path import abspath, dirname, join

from .table import Table

import astropy.io.registry as io_registry
import astropy.config as _config
from astropy import extern


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.table.jsviewer`.
    """

    jquery_url = _config.ConfigItem(
        'https://code.jquery.com/jquery-3.1.1.min.js',
        'The URL to the jquery library.')

    datatables_url = _config.ConfigItem(
        'https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js',
        'The URL to the jquery datatables library.')

    css_urls = _config.ConfigItem(
        ['https://cdn.datatables.net/1.10.12/css/jquery.dataTables.css'],
        'The URLs to the css file(s) to include.', cfgtype='string_list')
```
### 58 - astropy/table/jsviewer.py:

Start line: 172, End line: 201

```python
def write_table_jsviewer(table, filename, table_id=None, max_lines=5000,
                         table_class="display compact", jskwargs=None,
                         css=DEFAULT_CSS, htmldict=None, overwrite=False):
    if table_id is None:
        table_id = f'table{id(table)}'

    jskwargs = jskwargs or {}
    jsv = JSViewer(**jskwargs)

    sortable_columns = [i for i, col in enumerate(table.columns.values())
                        if col.info.dtype.kind in 'iufc']
    html_options = {
        'table_id': table_id,
        'table_class': table_class,
        'css': css,
        'cssfiles': jsv.css_urls,
        'jsfiles': jsv.jquery_urls,
        'js': jsv.html_js(table_id=table_id, sort_columns=sortable_columns)
    }
    if htmldict:
        html_options.update(htmldict)

    if max_lines < len(table):
        table = table[:max_lines]
    table.write(filename, format='html', htmldict=html_options,
                overwrite=overwrite)


io_registry.register_writer('jsviewer', Table, write_table_jsviewer)
```
