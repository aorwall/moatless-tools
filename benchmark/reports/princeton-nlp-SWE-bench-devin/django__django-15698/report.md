# django__django-15698

| **django/django** | `1a78ef2b85467a18ea6d7eaa4b27f67d11c87b9e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11211 |
| **Any found context length** | 11211 |
| **Avg pos** | 31.0 |
| **Min pos** | 31 |
| **Max pos** | 31 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -913,15 +913,18 @@ def _resolve_lookup(self, context):
                         try:  # method call (assuming no args required)
                             current = current()
                         except TypeError:
-                            signature = inspect.signature(current)
                             try:
-                                signature.bind()
-                            except TypeError:  # arguments *were* required
-                                current = (
-                                    context.template.engine.string_if_invalid
-                                )  # invalid method call
+                                signature = inspect.signature(current)
+                            except ValueError:  # No signature found.
+                                current = context.template.engine.string_if_invalid
                             else:
-                                raise
+                                try:
+                                    signature.bind()
+                                except TypeError:  # Arguments *were* required.
+                                    # Invalid method call.
+                                    current = context.template.engine.string_if_invalid
+                                else:
+                                    raise
         except Exception as e:
             template_name = getattr(context, "template_name", None) or "unknown"
             logger.debug(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/template/base.py | 916 | 924 | 31 | 2 | 11211


## Problem Statement

```
Templates crash when calling methods for built-in types.
Description
	 
		(last modified by Daniel)
	 
Found during a 2.2 -> 3.2 upgrade:
Given a template:
{{ foo }}
where foo is non-existant, it returns nothing, empty. (That's good)
{{ foo.count }}
also empty (Also good)
{% include 'second_template.html' with bar=foo %}
and then in second_template.html having:
{{ bar.count }}
results in
 File "/Users/daniel/src/django-bug-test/.v/lib/python3.8/site-packages/django/template/base.py", line 861, in _resolve_lookup
	signature = inspect.signature(current)
 File "/Users/daniel/.pyenv/versions/3.8.3/lib/python3.8/inspect.py", line 3093, in signature
	return Signature.from_callable(obj, follow_wrapped=follow_wrapped)
 File "/Users/daniel/.pyenv/versions/3.8.3/lib/python3.8/inspect.py", line 2842, in from_callable
	return _signature_from_callable(obj, sigcls=cls,
 File "/Users/daniel/.pyenv/versions/3.8.3/lib/python3.8/inspect.py", line 2296, in _signature_from_callable
	return _signature_from_builtin(sigcls, obj,
 File "/Users/daniel/.pyenv/versions/3.8.3/lib/python3.8/inspect.py", line 2107, in _signature_from_builtin
	raise ValueError("no signature found for builtin {!r}".format(func))
Exception Type: ValueError at /
Exception Value: no signature found for builtin <built-in method count of str object at 0x1100ff2f0>
On django 2.2, this would not crash, but resulted in empty (as I expected).
this seems to fix it for me:
diff --git a/django/template/base.py b/django/template/base.py
index a1ab437eca..f95aec5a90 100644
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -913,15 +913,19 @@ def _resolve_lookup(self, context):
						 try: # method call (assuming no args required)
							 current = current()
						 except TypeError:
-							signature = inspect.signature(current)
							 try:
-								signature.bind()
-							except TypeError: # arguments *were* required
-								current = (
-									context.template.engine.string_if_invalid
-								) # invalid method call
+								signature = inspect.signature(current)
+							except ValueError: # python builtins might not have signature
+								current = context.template.engine.string_if_invalid
							 else:
-								raise
+								try:
+									signature.bind()
+								except TypeError: # arguments *were* required
+									current = (
+										context.template.engine.string_if_invalid
+									) # invalid method call
+								else:
+									raise
		 except Exception as e:
			 template_name = getattr(context, "template_name", None) or "unknown"
			 logger.debug(

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/template/backends/dummy.py | 1 | 53| 327 | 327 | 327 | 
| 2 | **2 django/template/base.py** | 1 | 91| 754 | 1081 | 8578 | 
| 3 | 3 django/template/engine.py | 112 | 179| 459 | 1540 | 10076 | 
| 4 | 4 django/template/backends/django.py | 80 | 107| 176 | 1716 | 10980 | 
| 5 | 5 django/template/utils.py | 1 | 65| 407 | 2123 | 11696 | 
| 6 | 6 django/core/checks/templates.py | 1 | 47| 311 | 2434 | 12175 | 
| 7 | 6 django/template/engine.py | 1 | 63| 397 | 2831 | 12175 | 
| 8 | **6 django/template/base.py** | 138 | 177| 296 | 3127 | 12175 | 
| 9 | 6 django/core/checks/templates.py | 50 | 76| 166 | 3293 | 12175 | 
| 10 | 7 django/template/response.py | 1 | 51| 398 | 3691 | 13288 | 
| 11 | 8 django/template/__init__.py | 1 | 76| 394 | 4085 | 13682 | 
| 12 | 9 django/template/backends/jinja2.py | 1 | 52| 344 | 4429 | 14498 | 
| 13 | 10 django/views/debug.py | 1 | 56| 351 | 4780 | 19244 | 
| 14 | 10 django/template/engine.py | 65 | 83| 180 | 4960 | 19244 | 
| 15 | 11 django/template/backends/base.py | 1 | 82| 516 | 5476 | 19761 | 
| 16 | 12 django/template/context.py | 133 | 168| 288 | 5764 | 21653 | 
| 17 | 12 django/template/backends/django.py | 48 | 77| 214 | 5978 | 21653 | 
| 18 | 13 django/core/management/templates.py | 153 | 227| 609 | 6587 | 24655 | 
| 19 | 13 django/core/management/templates.py | 256 | 289| 240 | 6827 | 24655 | 
| 20 | 14 django/template/smartif.py | 117 | 151| 188 | 7015 | 26183 | 
| 21 | 15 django/forms/utils.py | 48 | 55| 113 | 7128 | 27902 | 
| 22 | 15 django/core/management/templates.py | 82 | 151| 579 | 7707 | 27902 | 
| 23 | 15 django/template/backends/django.py | 1 | 45| 303 | 8010 | 27902 | 
| 24 | 15 django/template/utils.py | 67 | 94| 197 | 8207 | 27902 | 
| 25 | **15 django/template/base.py** | 609 | 644| 361 | 8568 | 27902 | 
| 26 | 16 django/template/exceptions.py | 1 | 45| 262 | 8830 | 28165 | 
| 27 | 17 django/utils/translation/template.py | 1 | 36| 384 | 9214 | 30153 | 
| 28 | 18 django/forms/renderers.py | 32 | 103| 443 | 9657 | 30802 | 
| 29 | 19 django/views/i18n.py | 88 | 191| 702 | 10359 | 33305 | 
| 30 | 20 django/template/loader.py | 1 | 49| 299 | 10658 | 33721 | 
| **-> 31 <-** | **20 django/template/base.py** | 867 | 939| 553 | 11211 | 33721 | 
| 32 | 21 django/views/csrf.py | 15 | 100| 839 | 12050 | 35274 | 
| 33 | **21 django/template/base.py** | 712 | 747| 270 | 12320 | 35274 | 
| 34 | 22 django/template/backends/utils.py | 1 | 16| 0 | 12320 | 35365 | 
| 35 | **22 django/template/base.py** | 206 | 282| 502 | 12822 | 35365 | 
| 36 | **22 django/template/base.py** | 94 | 135| 228 | 13050 | 35365 | 
| 37 | **22 django/template/base.py** | 776 | 841| 558 | 13608 | 35365 | 
| 38 | 23 django/template/loader_tags.py | 301 | 353| 402 | 14010 | 38077 | 
| 39 | 23 django/template/backends/jinja2.py | 55 | 91| 233 | 14243 | 38077 | 
| 40 | 23 django/template/response.py | 53 | 67| 120 | 14363 | 38077 | 
| 41 | 23 django/views/debug.py | 459 | 496| 301 | 14664 | 38077 | 
| 42 | 23 django/template/context.py | 244 | 270| 197 | 14861 | 38077 | 
| 43 | 24 django/template/defaulttags.py | 869 | 889| 133 | 14994 | 48850 | 
| 44 | 24 django/utils/translation/template.py | 65 | 247| 1459 | 16453 | 48850 | 
| 45 | 24 django/views/debug.py | 498 | 553| 446 | 16899 | 48850 | 
| 46 | 24 django/template/response.py | 69 | 165| 606 | 17505 | 48850 | 
| 47 | 25 django/utils/deprecation.py | 42 | 84| 339 | 17844 | 49921 | 
| 48 | 26 django/views/defaults.py | 1 | 26| 151 | 17995 | 50904 | 
| 49 | **26 django/template/base.py** | 543 | 572| 245 | 18240 | 50904 | 
| 50 | **26 django/template/base.py** | 749 | 773| 210 | 18450 | 50904 | 
| 51 | 27 django/template/loaders/base.py | 1 | 52| 286 | 18736 | 51191 | 
| 52 | 28 django/forms/forms.py | 319 | 365| 340 | 19076 | 55272 | 
| 53 | 29 django/db/models/base.py | 477 | 590| 953 | 20029 | 73817 | 
| 54 | 29 django/template/context.py | 1 | 24| 128 | 20157 | 73817 | 
| 55 | 29 django/utils/translation/template.py | 39 | 63| 165 | 20322 | 73817 | 
| 56 | 29 django/template/backends/jinja2.py | 94 | 128| 238 | 20560 | 73817 | 
| 57 | **29 django/template/base.py** | 285 | 338| 359 | 20919 | 73817 | 
| 58 | 29 django/template/defaulttags.py | 892 | 980| 694 | 21613 | 73817 | 
| 59 | 30 django/utils/baseconv.py | 1 | 55| 147 | 21760 | 74665 | 
| 60 | 31 django/template/autoreload.py | 33 | 55| 149 | 21909 | 75022 | 
| 61 | 31 django/views/csrf.py | 1 | 13| 132 | 22041 | 75022 | 
| 62 | 31 django/db/models/base.py | 1523 | 1558| 273 | 22314 | 75022 | 
| 63 | 32 django/utils/autoreload.py | 59 | 87| 156 | 22470 | 80241 | 
| 64 | 32 django/template/loader_tags.py | 1 | 39| 207 | 22677 | 80241 | 
| 65 | 33 django/core/checks/model_checks.py | 1 | 90| 671 | 23348 | 82051 | 
| 66 | **33 django/template/base.py** | 341 | 365| 169 | 23517 | 82051 | 
| 67 | 33 django/template/engine.py | 85 | 110| 204 | 23721 | 82051 | 
| 68 | 33 django/forms/utils.py | 58 | 78| 150 | 23871 | 82051 | 
| 69 | 33 django/core/checks/model_checks.py | 161 | 185| 267 | 24138 | 82051 | 
| 70 | 33 django/core/checks/model_checks.py | 187 | 228| 345 | 24483 | 82051 | 
| 71 | 33 django/views/csrf.py | 101 | 161| 581 | 25064 | 82051 | 
| 72 | **33 django/template/base.py** | 1034 | 1047| 127 | 25191 | 82051 | 
| 73 | 34 django/utils/datetime_safe.py | 1 | 81| 501 | 25692 | 82867 | 
| 74 | 35 django/template/loaders/cached.py | 1 | 66| 501 | 26193 | 83592 | 
| 75 | **35 django/template/base.py** | 179 | 204| 154 | 26347 | 83592 | 
| 76 | 35 django/forms/utils.py | 197 | 204| 133 | 26480 | 83592 | 
| 77 | 35 django/template/defaulttags.py | 665 | 716| 339 | 26819 | 83592 | 
| 78 | 36 django/views/static.py | 56 | 79| 211 | 27030 | 84578 | 
| 79 | 36 django/template/engine.py | 181 | 195| 138 | 27168 | 84578 | 
| 80 | 37 django/core/checks/security/base.py | 1 | 79| 691 | 27859 | 86767 | 
| 81 | 37 django/template/defaulttags.py | 1289 | 1323| 293 | 28152 | 86767 | 
| 82 | 38 django/template/library.py | 281 | 366| 648 | 28800 | 89323 | 
| 83 | 39 django/templatetags/cache.py | 1 | 55| 426 | 29226 | 90065 | 
| 84 | 39 django/template/loader_tags.py | 277 | 298| 243 | 29469 | 90065 | 
| 85 | 39 django/core/management/templates.py | 381 | 403| 200 | 29669 | 90065 | 
| 86 | 39 django/template/backends/django.py | 110 | 138| 208 | 29877 | 90065 | 
| 87 | 39 django/template/loader.py | 52 | 67| 117 | 29994 | 90065 | 
| 88 | 40 django/db/backends/ddl_references.py | 184 | 220| 253 | 30247 | 91694 | 
| 89 | 40 django/forms/renderers.py | 1 | 29| 206 | 30453 | 91694 | 
| 90 | 40 django/template/engine.py | 197 | 213| 134 | 30587 | 91694 | 
| 91 | 40 django/forms/forms.py | 226 | 317| 702 | 31289 | 91694 | 
| 92 | 40 django/db/models/base.py | 1785 | 1807| 171 | 31460 | 91694 | 
| 93 | 40 django/views/debug.py | 280 | 312| 236 | 31696 | 91694 | 
| 94 | 40 django/template/defaulttags.py | 1232 | 1255| 176 | 31872 | 91694 | 
| 95 | 40 django/core/management/templates.py | 229 | 254| 212 | 32084 | 91694 | 
| 96 | 40 django/template/loader_tags.py | 243 | 274| 272 | 32356 | 91694 | 
| 97 | 41 django/template/loaders/locmem.py | 1 | 27| 127 | 32483 | 91821 | 
| 98 | 41 django/template/defaulttags.py | 92 | 114| 175 | 32658 | 91821 | 
| 99 | 41 django/template/smartif.py | 154 | 214| 426 | 33084 | 91821 | 
| 100 | 42 django/contrib/auth/checks.py | 1 | 104| 728 | 33812 | 93337 | 
| 101 | 43 django/core/checks/messages.py | 25 | 38| 154 | 33966 | 93917 | 
| 102 | 43 django/template/context.py | 27 | 130| 655 | 34621 | 93917 | 
| 103 | 43 django/db/models/base.py | 1 | 64| 346 | 34967 | 93917 | 
| 104 | 44 django/db/models/fields/__init__.py | 1 | 110| 663 | 35630 | 112640 | 
| 105 | 45 django/template/defaultfilters.py | 869 | 914| 379 | 36009 | 119120 | 
| 106 | 45 django/template/defaulttags.py | 545 | 569| 191 | 36200 | 119120 | 
| 107 | 46 django/shortcuts.py | 1 | 25| 161 | 36361 | 120244 | 
| 108 | 46 django/template/smartif.py | 1 | 41| 299 | 36660 | 120244 | 
| 109 | 46 django/template/defaulttags.py | 1 | 64| 338 | 36998 | 120244 | 
| 110 | 46 django/core/checks/messages.py | 59 | 82| 161 | 37159 | 120244 | 
| 111 | 47 django/views/generic/__init__.py | 1 | 40| 204 | 37363 | 120449 | 
| 112 | 48 docs/_ext/djangodocs.py | 383 | 402| 204 | 37567 | 123673 | 
| 113 | 49 django/contrib/gis/gdal/prototypes/generation.py | 144 | 178| 237 | 37804 | 124851 | 
| 114 | 50 django/contrib/contenttypes/checks.py | 28 | 47| 129 | 37933 | 125112 | 
| 115 | 50 django/template/library.py | 246 | 278| 304 | 38237 | 125112 | 
| 116 | 50 django/template/loader_tags.py | 160 | 208| 366 | 38603 | 125112 | 
| 117 | 51 django/contrib/admindocs/views.py | 65 | 99| 297 | 38900 | 128594 | 
| 118 | 51 django/utils/deprecation.py | 1 | 39| 228 | 39128 | 128594 | 
| 119 | 51 django/db/models/base.py | 248 | 365| 874 | 40002 | 128594 | 
| 120 | 51 django/views/defaults.py | 124 | 150| 190 | 40192 | 128594 | 
| 121 | **51 django/template/base.py** | 843 | 865| 190 | 40382 | 128594 | 
| 122 | 51 django/template/library.py | 1 | 54| 336 | 40718 | 128594 | 
| 123 | 52 django/core/checks/registry.py | 1 | 25| 130 | 40848 | 129272 | 
| 124 | 53 django/db/models/functions/datetime.py | 219 | 254| 304 | 41152 | 132051 | 
| 125 | 53 django/template/autoreload.py | 1 | 30| 206 | 41358 | 132051 | 
| 126 | 53 django/template/library.py | 99 | 154| 321 | 41679 | 132051 | 
| 127 | 53 django/db/models/base.py | 1395 | 1425| 216 | 41895 | 132051 | 
| 128 | 54 django/core/management/commands/inspectdb.py | 54 | 236| 1426 | 43321 | 134887 | 
| 129 | **54 django/template/base.py** | 367 | 399| 372 | 43693 | 134887 | 
| 130 | 55 django/template/context_processors.py | 36 | 55| 132 | 43825 | 135382 | 
| 131 | 56 django/db/models/functions/text.py | 1 | 39| 266 | 44091 | 137751 | 
| 132 | 56 django/db/models/base.py | 1809 | 1842| 232 | 44323 | 137751 | 
| 133 | 56 django/contrib/admindocs/views.py | 394 | 427| 211 | 44534 | 137751 | 
| 134 | 56 django/core/checks/model_checks.py | 135 | 159| 268 | 44802 | 137751 | 
| 135 | 56 django/db/models/base.py | 1760 | 1783| 176 | 44978 | 137751 | 
| 136 | 56 django/db/models/functions/datetime.py | 256 | 311| 471 | 45449 | 137751 | 
| 137 | 56 django/template/defaulttags.py | 248 | 278| 263 | 45712 | 137751 | 
| 138 | 56 django/forms/utils.py | 138 | 195| 348 | 46060 | 137751 | 
| 139 | 57 django/template/loaders/filesystem.py | 1 | 46| 287 | 46347 | 138038 | 
| 140 | 58 django/utils/itercompat.py | 1 | 9| 0 | 46347 | 138078 | 
| 141 | 58 django/template/library.py | 228 | 243| 126 | 46473 | 138078 | 
| 142 | 59 django/core/serializers/xml_serializer.py | 438 | 456| 126 | 46599 | 141674 | 
| 143 | 60 django/views/generic/detail.py | 113 | 181| 518 | 47117 | 143004 | 
| 144 | 60 django/utils/datetime_safe.py | 84 | 119| 313 | 47430 | 143004 | 
| 145 | 60 django/db/models/base.py | 1592 | 1617| 184 | 47614 | 143004 | 
| 146 | 60 django/template/defaultfilters.py | 365 | 452| 504 | 48118 | 143004 | 
| 147 | 61 django/templatetags/static.py | 95 | 131| 242 | 48360 | 144023 | 
| 148 | 62 django/contrib/admin/templatetags/base.py | 32 | 46| 138 | 48498 | 144338 | 
| 149 | 62 django/views/debug.py | 225 | 277| 471 | 48969 | 144338 | 
| 150 | 62 django/template/defaulttags.py | 183 | 245| 532 | 49501 | 144338 | 
| 151 | 62 django/views/debug.py | 75 | 102| 178 | 49679 | 144338 | 
| 152 | 63 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 49679 | 144414 | 
| 153 | 63 django/views/debug.py | 211 | 223| 143 | 49822 | 144414 | 
| 154 | 64 django/forms/boundfield.py | 55 | 82| 188 | 50010 | 146878 | 
| 155 | 65 django/core/signing.py | 1 | 95| 795 | 50805 | 149065 | 
| 156 | 66 django/db/models/fields/related.py | 302 | 339| 296 | 51101 | 163677 | 
| 157 | 67 django/core/checks/security/csrf.py | 1 | 42| 305 | 51406 | 164142 | 
| 158 | 67 django/template/library.py | 369 | 386| 110 | 51516 | 164142 | 
| 159 | 68 django/core/exceptions.py | 1 | 121| 436 | 51952 | 165335 | 
| 160 | 68 django/db/models/fields/related.py | 1449 | 1578| 984 | 52936 | 165335 | 
| 161 | 68 django/contrib/contenttypes/checks.py | 1 | 25| 130 | 53066 | 165335 | 
| 162 | 69 django/core/handlers/exception.py | 63 | 158| 600 | 53666 | 166453 | 
| 163 | 69 django/template/defaulttags.py | 280 | 292| 147 | 53813 | 166453 | 
| 164 | 70 django/utils/datastructures.py | 220 | 254| 226 | 54039 | 168758 | 
| 165 | 70 django/template/defaulttags.py | 418 | 434| 121 | 54160 | 168758 | 
| 166 | 71 django/views/generic/base.py | 172 | 207| 243 | 54403 | 170631 | 
| 167 | 71 django/db/models/base.py | 1378 | 1393| 138 | 54541 | 170631 | 
| 168 | 71 django/db/models/fields/__init__.py | 1454 | 1490| 276 | 54817 | 170631 | 
| 169 | 72 django/db/backends/sqlite3/base.py | 343 | 365| 183 | 55000 | 173662 | 
| 170 | 72 django/template/defaulttags.py | 572 | 662| 812 | 55812 | 173662 | 
| 171 | **72 django/template/base.py** | 436 | 456| 156 | 55968 | 173662 | 
| 172 | 73 django/db/models/fields/related_lookups.py | 1 | 39| 214 | 56182 | 175265 | 
| 173 | 74 django/db/models/functions/__init__.py | 94 | 191| 461 | 56643 | 176087 | 
| 174 | 74 django/core/management/templates.py | 42 | 80| 265 | 56908 | 176087 | 
| 175 | 74 django/template/defaulttags.py | 376 | 415| 218 | 57126 | 176087 | 
| 176 | 75 django/db/backends/base/creation.py | 327 | 351| 282 | 57408 | 179028 | 
| 177 | 76 django/db/migrations/serializer.py | 290 | 311| 166 | 57574 | 181715 | 
| 178 | 76 django/views/defaults.py | 102 | 121| 144 | 57718 | 181715 | 
| 179 | 77 django/core/checks/urls.py | 76 | 118| 266 | 57984 | 182421 | 
| 180 | 78 django/utils/safestring.py | 47 | 73| 166 | 58150 | 182834 | 
| 181 | 78 django/db/models/base.py | 195 | 247| 458 | 58608 | 182834 | 
| 182 | **78 django/template/base.py** | 574 | 606| 207 | 58815 | 182834 | 
| 183 | 78 django/db/models/functions/datetime.py | 335 | 416| 443 | 59258 | 182834 | 
| 184 | 78 django/db/models/base.py | 2255 | 2446| 1302 | 60560 | 182834 | 
| 185 | 79 django/db/models/functions/comparison.py | 104 | 119| 178 | 60738 | 184589 | 
| 186 | 79 django/contrib/admindocs/views.py | 1 | 36| 256 | 60994 | 184589 | 
| 187 | 79 django/db/models/base.py | 1298 | 1345| 409 | 61403 | 184589 | 
| 188 | 79 django/db/models/base.py | 2049 | 2154| 736 | 62139 | 184589 | 
| 189 | 80 django/forms/models.py | 889 | 912| 198 | 62337 | 196833 | 
| 190 | 81 django/core/serializers/base.py | 36 | 68| 175 | 62512 | 199509 | 


### Hint

```
Isn't a bug, that the template resolver tries to step into ''.<some_builtin_method> here? Maybe the for bit in self.lookups: descent could exit as soon as current drops to an empty string?
Thanks for the report. foo may exist, the same error is raised for each method call for bultin types, e.g. def myview(request): return render(request, "myview.html", {"foo": "X"}) myview.html: {{foo.count}} Regression in 09341856ed9008875c1cc883dc0c287670131458.
​https://github.com/django/django/pull/15698
Hi Mariusz, Just saw the flag Patch needs improvement is checked. How exactly should I improve the patch, besides that I need to add tests (or only tests need to be added)? I'm a first-time Django contributor any help is appreciated! Cheng
```

## Patch

```diff
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -913,15 +913,18 @@ def _resolve_lookup(self, context):
                         try:  # method call (assuming no args required)
                             current = current()
                         except TypeError:
-                            signature = inspect.signature(current)
                             try:
-                                signature.bind()
-                            except TypeError:  # arguments *were* required
-                                current = (
-                                    context.template.engine.string_if_invalid
-                                )  # invalid method call
+                                signature = inspect.signature(current)
+                            except ValueError:  # No signature found.
+                                current = context.template.engine.string_if_invalid
                             else:
-                                raise
+                                try:
+                                    signature.bind()
+                                except TypeError:  # Arguments *were* required.
+                                    # Invalid method call.
+                                    current = context.template.engine.string_if_invalid
+                                else:
+                                    raise
         except Exception as e:
             template_name = getattr(context, "template_name", None) or "unknown"
             logger.debug(

```

## Test Patch

```diff
diff --git a/tests/template_tests/tests.py b/tests/template_tests/tests.py
--- a/tests/template_tests/tests.py
+++ b/tests/template_tests/tests.py
@@ -183,6 +183,14 @@ def test_node_origin(self):
         for node in template.nodelist:
             self.assertEqual(node.origin, template.origin)
 
+    def test_render_built_in_type_method(self):
+        """
+        Templates should not crash when rendering methods for built-in types
+        without required arguments.
+        """
+        template = self._engine().from_string("{{ description.count }}")
+        self.assertEqual(template.render(Context({"description": "test"})), "")
+
 
 class TemplateTests(TemplateTestMixin, SimpleTestCase):
     debug_engine = False

```


## Code snippets

### 1 - django/template/backends/dummy.py:

Start line: 1, End line: 53

```python
import string

from django.core.exceptions import ImproperlyConfigured
from django.template import Origin, TemplateDoesNotExist
from django.utils.html import conditional_escape

from .base import BaseEngine
from .utils import csrf_input_lazy, csrf_token_lazy


class TemplateStrings(BaseEngine):

    app_dirname = "template_strings"

    def __init__(self, params):
        params = params.copy()
        options = params.pop("OPTIONS").copy()
        if options:
            raise ImproperlyConfigured("Unknown options: {}".format(", ".join(options)))
        super().__init__(params)

    def from_string(self, template_code):
        return Template(template_code)

    def get_template(self, template_name):
        tried = []
        for template_file in self.iter_template_filenames(template_name):
            try:
                with open(template_file, encoding="utf-8") as fp:
                    template_code = fp.read()
            except FileNotFoundError:
                tried.append(
                    (
                        Origin(template_file, template_name, self),
                        "Source does not exist",
                    )
                )
            else:
                return Template(template_code)
        raise TemplateDoesNotExist(template_name, tried=tried, backend=self)


class Template(string.Template):
    def render(self, context=None, request=None):
        if context is None:
            context = {}
        else:
            context = {k: conditional_escape(v) for k, v in context.items()}
        if request is not None:
            context["csrf_input"] = csrf_input_lazy(request)
            context["csrf_token"] = csrf_token_lazy(request)
        return self.safe_substitute(context)
```
### 2 - django/template/base.py:

Start line: 1, End line: 91

```python
"""
This is the Django template system.

How it works:

The Lexer.tokenize() method converts a template string (i.e., a string
containing markup with custom template tags) to tokens, which can be either
plain text (TokenType.TEXT), variables (TokenType.VAR), or block statements
(TokenType.BLOCK).

The Parser() class takes a list of tokens in its constructor, and its parse()
method returns a compiled template -- which is, under the hood, a list of
Node objects.

Each Node is responsible for creating some sort of output -- e.g. simple text
(TextNode), variable values in a given context (VariableNode), results of basic
logic (IfNode), results of looping (ForNode), or anything else. The core Node
types are TextNode, VariableNode, IfNode and ForNode, but plugin modules can
define their own custom node types.

Each Node has a render() method, which takes a Context and returns a string of
the rendered node. For example, the render() method of a Variable Node returns
the variable's value as a string. The render() method of a ForNode returns the
rendered output of whatever was inside the loop, recursively.

The Template class is a convenient wrapper that takes care of template
compilation and rendering.

Usage:

The only thing you should ever use directly in this file is the Template class.
Create a compiled template object with a template_string, then call render()
with a context. In the compilation stage, the TemplateSyntaxError exception
will be raised if the template doesn't have proper syntax.

Sample code:

>>> from django import template
>>> s = '<html>{% if test %}<h1>{{ varvalue }}</h1>{% endif %}</html>'
>>> t = template.Template(s)

(t is now a compiled template, and its render() method can be called multiple
times with multiple contexts)

>>> c = template.Context({'test':True, 'varvalue': 'Hello'})
>>> t.render(c)
'<html><h1>Hello</h1></html>'
>>> c = template.Context({'test':False, 'varvalue': 'Hello'})
>>> t.render(c)
'<html></html>'
"""

import inspect
import logging
import re
from enum import Enum

from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy

from .exceptions import TemplateSyntaxError

# template syntax constants
FILTER_SEPARATOR = "|"
FILTER_ARGUMENT_SEPARATOR = ":"
VARIABLE_ATTRIBUTE_SEPARATOR = "."
BLOCK_TAG_START = "{%"
BLOCK_TAG_END = "%}"
VARIABLE_TAG_START = "{{"
VARIABLE_TAG_END = "}}"
COMMENT_TAG_START = "{#"
COMMENT_TAG_END = "#}"
SINGLE_BRACE_START = "{"
SINGLE_BRACE_END = "}"

# what to report as the origin for templates that come from non-loader sources
# (e.g. strings)
UNKNOWN_SOURCE = "<unknown source>"

# Match BLOCK_TAG_*, VARIABLE_TAG_*, and COMMENT_TAG_* tags and capture the
# entire tag, including start/end delimiters. Using re.compile() is faster
# than instantiating SimpleLazyObject with _lazy_re_compile().
tag_re = re.compile(r"({%.*?%}|{{.*?}}|{#.*?#})")

logger = logging.getLogger("django.template")
```
### 3 - django/template/engine.py:

Start line: 112, End line: 179

```python
class Engine:

    @cached_property
    def template_context_processors(self):
        context_processors = _builtin_context_processors
        context_processors += tuple(self.context_processors)
        return tuple(import_string(path) for path in context_processors)

    def get_template_builtins(self, builtins):
        return [import_library(x) for x in builtins]

    def get_template_libraries(self, libraries):
        loaded = {}
        for name, path in libraries.items():
            loaded[name] = import_library(path)
        return loaded

    @cached_property
    def template_loaders(self):
        return self.get_template_loaders(self.loaders)

    def get_template_loaders(self, template_loaders):
        loaders = []
        for template_loader in template_loaders:
            loader = self.find_template_loader(template_loader)
            if loader is not None:
                loaders.append(loader)
        return loaders

    def find_template_loader(self, loader):
        if isinstance(loader, (tuple, list)):
            loader, *args = loader
        else:
            args = []

        if isinstance(loader, str):
            loader_class = import_string(loader)
            return loader_class(self, *args)
        else:
            raise ImproperlyConfigured(
                "Invalid value in template loaders configuration: %r" % loader
            )

    def find_template(self, name, dirs=None, skip=None):
        tried = []
        for loader in self.template_loaders:
            try:
                template = loader.get_template(name, skip=skip)
                return template, template.origin
            except TemplateDoesNotExist as e:
                tried.extend(e.tried)
        raise TemplateDoesNotExist(name, tried=tried)

    def from_string(self, template_code):
        """
        Return a compiled Template object for the given template code,
        handling template inheritance recursively.
        """
        return Template(template_code, engine=self)

    def get_template(self, template_name):
        """
        Return a compiled Template object for the given template name,
        handling template inheritance recursively.
        """
        template, origin = self.find_template(template_name)
        if not hasattr(template, "render"):
            # template needs to be compiled
            template = Template(template, origin, template_name, engine=self)
        return template
```
### 4 - django/template/backends/django.py:

Start line: 80, End line: 107

```python
def reraise(exc, backend):
    """
    Reraise TemplateDoesNotExist while maintaining template debug information.
    """
    new = copy_exception(exc, backend)
    raise new from exc


def get_template_tag_modules():
    """
    Yield (module_name, module_path) pairs for all installed template tag
    libraries.
    """
    candidates = ["django.templatetags"]
    candidates.extend(
        f"{app_config.name}.templatetags" for app_config in apps.get_app_configs()
    )

    for candidate in candidates:
        try:
            pkg = import_module(candidate)
        except ImportError:
            # No templatetags package defined. This is safe to ignore.
            continue

        if hasattr(pkg, "__path__"):
            for name in get_package_libraries(pkg):
                yield name[len(candidate) + 1 :], name
```
### 5 - django/template/utils.py:

Start line: 1, End line: 65

```python
import functools
from collections import Counter
from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string


class InvalidTemplateEngineError(ImproperlyConfigured):
    pass


class EngineHandler:
    def __init__(self, templates=None):
        """
        templates is an optional list of template engine definitions
        (structured like settings.TEMPLATES).
        """
        self._templates = templates
        self._engines = {}

    @cached_property
    def templates(self):
        if self._templates is None:
            self._templates = settings.TEMPLATES

        templates = {}
        backend_names = []
        for tpl in self._templates:
            try:
                # This will raise an exception if 'BACKEND' doesn't exist or
                # isn't a string containing at least one dot.
                default_name = tpl["BACKEND"].rsplit(".", 2)[-2]
            except Exception:
                invalid_backend = tpl.get("BACKEND", "<not defined>")
                raise ImproperlyConfigured(
                    "Invalid BACKEND for a template engine: {}. Check "
                    "your TEMPLATES setting.".format(invalid_backend)
                )

            tpl = {
                "NAME": default_name,
                "DIRS": [],
                "APP_DIRS": False,
                "OPTIONS": {},
                **tpl,
            }

            templates[tpl["NAME"]] = tpl
            backend_names.append(tpl["NAME"])

        counts = Counter(backend_names)
        duplicates = [alias for alias, count in counts.most_common() if count > 1]
        if duplicates:
            raise ImproperlyConfigured(
                "Template engine aliases aren't unique, duplicates: {}. "
                "Set a unique NAME for each engine in settings.TEMPLATES.".format(
                    ", ".join(duplicates)
                )
            )

        return templates
```
### 6 - django/core/checks/templates.py:

Start line: 1, End line: 47

```python
import copy
from collections import defaultdict

from django.conf import settings
from django.template.backends.django import get_template_tag_modules

from . import Error, Tags, register

E001 = Error(
    "You have 'APP_DIRS': True in your TEMPLATES but also specify 'loaders' "
    "in OPTIONS. Either remove APP_DIRS or remove the 'loaders' option.",
    id="templates.E001",
)
E002 = Error(
    "'string_if_invalid' in TEMPLATES OPTIONS must be a string but got: {} ({}).",
    id="templates.E002",
)
E003 = Error(
    "{} is used for multiple template tag modules: {}",
    id="templates.E003",
)


@register(Tags.templates)
def check_setting_app_dirs_loaders(app_configs, **kwargs):
    return (
        [E001]
        if any(
            conf.get("APP_DIRS") and "loaders" in conf.get("OPTIONS", {})
            for conf in settings.TEMPLATES
        )
        else []
    )


@register(Tags.templates)
def check_string_if_invalid_is_string(app_configs, **kwargs):
    errors = []
    for conf in settings.TEMPLATES:
        string_if_invalid = conf.get("OPTIONS", {}).get("string_if_invalid", "")
        if not isinstance(string_if_invalid, str):
            error = copy.copy(E002)
            error.msg = error.msg.format(
                string_if_invalid, type(string_if_invalid).__name__
            )
            errors.append(error)
    return errors
```
### 7 - django/template/engine.py:

Start line: 1, End line: 63

```python
import functools

from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string

from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library


class Engine:
    default_builtins = [
        "django.template.defaulttags",
        "django.template.defaultfilters",
        "django.template.loader_tags",
    ]

    def __init__(
        self,
        dirs=None,
        app_dirs=False,
        context_processors=None,
        debug=False,
        loaders=None,
        string_if_invalid="",
        file_charset="utf-8",
        libraries=None,
        builtins=None,
        autoescape=True,
    ):
        if dirs is None:
            dirs = []
        if context_processors is None:
            context_processors = []
        if loaders is None:
            loaders = ["django.template.loaders.filesystem.Loader"]
            if app_dirs:
                loaders += ["django.template.loaders.app_directories.Loader"]
            loaders = [("django.template.loaders.cached.Loader", loaders)]
        else:
            if app_dirs:
                raise ImproperlyConfigured(
                    "app_dirs must not be set when loaders is defined."
                )
        if libraries is None:
            libraries = {}
        if builtins is None:
            builtins = []

        self.dirs = dirs
        self.app_dirs = app_dirs
        self.autoescape = autoescape
        self.context_processors = context_processors
        self.debug = debug
        self.loaders = loaders
        self.string_if_invalid = string_if_invalid
        self.file_charset = file_charset
        self.libraries = libraries
        self.template_libraries = self.get_template_libraries(libraries)
        self.builtins = self.default_builtins + builtins
        self.template_builtins = self.get_template_builtins(self.builtins)
```
### 8 - django/template/base.py:

Start line: 138, End line: 177

```python
class Template:
    def __init__(self, template_string, origin=None, name=None, engine=None):
        # If Template is instantiated directly rather than from an Engine and
        # exactly one Django template engine is configured, use that engine.
        # This is required to preserve backwards-compatibility for direct use
        # e.g. Template('...').render(Context({...}))
        if engine is None:
            from .engine import Engine

            engine = Engine.get_default()
        if origin is None:
            origin = Origin(UNKNOWN_SOURCE)
        self.name = name
        self.origin = origin
        self.engine = engine
        self.source = str(template_string)  # May be lazy.
        self.nodelist = self.compile_nodelist()

    def __iter__(self):
        for node in self.nodelist:
            yield from node

    def __repr__(self):
        return '<%s template_string="%s...">' % (
            self.__class__.__qualname__,
            self.source[:20].replace("\n", ""),
        )

    def _render(self, context):
        return self.nodelist.render(context)

    def render(self, context):
        "Display stage -- can be called many times"
        with context.render_context.push_state(self):
            if context.template is None:
                with context.bind_template(self):
                    context.template_name = self.name
                    return self._render(context)
            else:
                return self._render(context)
```
### 9 - django/core/checks/templates.py:

Start line: 50, End line: 76

```python
@register(Tags.templates)
def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
    errors = []
    libraries = defaultdict(list)

    for conf in settings.TEMPLATES:
        custom_libraries = conf.get("OPTIONS", {}).get("libraries", {})
        for module_name, module_path in custom_libraries.items():
            libraries[module_name].append(module_path)

    for module_name, module_path in get_template_tag_modules():
        libraries[module_name].append(module_path)

    for library_name, items in libraries.items():
        if len(items) > 1:
            errors.append(
                Error(
                    E003.msg.format(
                        repr(library_name),
                        ", ".join(repr(item) for item in items),
                    ),
                    id=E003.id,
                )
            )

    return errors
```
### 10 - django/template/response.py:

Start line: 1, End line: 51

```python
from django.http import HttpResponse

from .loader import get_template, select_template


class ContentNotRenderedError(Exception):
    pass


class SimpleTemplateResponse(HttpResponse):
    rendering_attrs = ["template_name", "context_data", "_post_render_callbacks"]

    def __init__(
        self,
        template,
        context=None,
        content_type=None,
        status=None,
        charset=None,
        using=None,
        headers=None,
    ):
        # It would seem obvious to call these next two members 'template' and
        # 'context', but those names are reserved as part of the test Client
        # API. To avoid the name collision, we use different names.
        self.template_name = template
        self.context_data = context

        self.using = using

        self._post_render_callbacks = []

        # _request stores the current request object in subclasses that know
        # about requests, like TemplateResponse. It's defined in the base class
        # to minimize code duplication.
        # It's called self._request because self.request gets overwritten by
        # django.test.client.Client. Unlike template_name and context_data,
        # _request should not be considered part of the public API.
        self._request = None

        # content argument doesn't make sense here because it will be replaced
        # with rendered template so we always pass empty string in order to
        # prevent errors and provide shorter signature.
        super().__init__("", content_type, status, charset=charset, headers=headers)

        # _is_rendered tracks whether the template and context has been baked
        # into a final response.
        # Super __init__ doesn't know any better than to set self.content to
        # the empty string we just gave it, which wrongly sets _is_rendered
        # True, so we initialize it to False after the call to super __init__.
        self._is_rendered = False
```
### 25 - django/template/base.py:

Start line: 609, End line: 644

```python
# This only matches constant *strings* (things in quotes or marked for
# translation). Numbers are treated as variables for implementation reasons
# (so that they retain their type when passed to filters).
constant_string = r"""
(?:%(i18n_open)s%(strdq)s%(i18n_close)s|
%(i18n_open)s%(strsq)s%(i18n_close)s|
%(strdq)s|
%(strsq)s)
""" % {
    "strdq": r'"[^"\\]*(?:\\.[^"\\]*)*"',  # double-quoted string
    "strsq": r"'[^'\\]*(?:\\.[^'\\]*)*'",  # single-quoted string
    "i18n_open": re.escape("_("),
    "i18n_close": re.escape(")"),
}
constant_string = constant_string.replace("\n", "")

filter_raw_string = r"""
^(?P<constant>%(constant)s)|
^(?P<var>[%(var_chars)s]+|%(num)s)|
 (?:\s*%(filter_sep)s\s*
     (?P<filter_name>\w+)
         (?:%(arg_sep)s
             (?:
              (?P<constant_arg>%(constant)s)|
              (?P<var_arg>[%(var_chars)s]+|%(num)s)
             )
         )?
 )""" % {
    "constant": constant_string,
    "num": r"[-+\.]?\d[\d\.e]*",
    "var_chars": r"\w\.",
    "filter_sep": re.escape(FILTER_SEPARATOR),
    "arg_sep": re.escape(FILTER_ARGUMENT_SEPARATOR),
}

filter_re = _lazy_re_compile(filter_raw_string, re.VERBOSE)
```
### 31 - django/template/base.py:

Start line: 867, End line: 939

```python
class Variable:

    def _resolve_lookup(self, context):
        """
        Perform resolution of a real variable (i.e. not a literal) against the
        given context.

        As indicated by the method's name, this method is an implementation
        detail and shouldn't be called by external code. Use Variable.resolve()
        instead.
        """
        current = context
        try:  # catch-all for silent variable failures
            for bit in self.lookups:
                try:  # dictionary lookup
                    current = current[bit]
                    # ValueError/IndexError are for numpy.array lookup on
                    # numpy < 1.9 and 1.9+ respectively
                except (TypeError, AttributeError, KeyError, ValueError, IndexError):
                    try:  # attribute lookup
                        # Don't return class attributes if the class is the context:
                        if isinstance(current, BaseContext) and getattr(
                            type(current), bit
                        ):
                            raise AttributeError
                        current = getattr(current, bit)
                    except (TypeError, AttributeError):
                        # Reraise if the exception was raised by a @property
                        if not isinstance(current, BaseContext) and bit in dir(current):
                            raise
                        try:  # list-index lookup
                            current = current[int(bit)]
                        except (
                            IndexError,  # list index out of range
                            ValueError,  # invalid literal for int()
                            KeyError,  # current is a dict without `int(bit)` key
                            TypeError,
                        ):  # unsubscriptable object
                            raise VariableDoesNotExist(
                                "Failed lookup for key [%s] in %r",
                                (bit, current),
                            )  # missing attribute
                if callable(current):
                    if getattr(current, "do_not_call_in_templates", False):
                        pass
                    elif getattr(current, "alters_data", False):
                        current = context.template.engine.string_if_invalid
                    else:
                        try:  # method call (assuming no args required)
                            current = current()
                        except TypeError:
                            signature = inspect.signature(current)
                            try:
                                signature.bind()
                            except TypeError:  # arguments *were* required
                                current = (
                                    context.template.engine.string_if_invalid
                                )  # invalid method call
                            else:
                                raise
        except Exception as e:
            template_name = getattr(context, "template_name", None) or "unknown"
            logger.debug(
                "Exception while resolving variable '%s' in template '%s'.",
                bit,
                template_name,
                exc_info=True,
            )

            if getattr(e, "silent_variable_failure", False):
                current = context.template.engine.string_if_invalid
            else:
                raise

        return current
```
### 33 - django/template/base.py:

Start line: 712, End line: 747

```python
class FilterExpression:

    def resolve(self, context, ignore_failures=False):
        if self.is_var:
            try:
                obj = self.var.resolve(context)
            except VariableDoesNotExist:
                if ignore_failures:
                    obj = None
                else:
                    string_if_invalid = context.template.engine.string_if_invalid
                    if string_if_invalid:
                        if "%s" in string_if_invalid:
                            return string_if_invalid % self.var
                        else:
                            return string_if_invalid
                    else:
                        obj = string_if_invalid
        else:
            obj = self.var
        for func, args in self.filters:
            arg_vals = []
            for lookup, arg in args:
                if not lookup:
                    arg_vals.append(mark_safe(arg))
                else:
                    arg_vals.append(arg.resolve(context))
            if getattr(func, "expects_localtime", False):
                obj = template_localtime(obj, context.use_tz)
            if getattr(func, "needs_autoescape", False):
                new_obj = func(obj, autoescape=context.autoescape, *arg_vals)
            else:
                new_obj = func(obj, *arg_vals)
            if getattr(func, "is_safe", False) and isinstance(obj, SafeData):
                obj = mark_safe(new_obj)
            else:
                obj = new_obj
        return obj
```
### 35 - django/template/base.py:

Start line: 206, End line: 282

```python
class Template:

    def get_exception_info(self, exception, token):
        """
        Return a dictionary containing contextual line information of where
        the exception occurred in the template. The following information is
        provided:

        message
            The message of the exception raised.

        source_lines
            The lines before, after, and including the line the exception
            occurred on.

        line
            The line number the exception occurred on.

        before, during, after
            The line the exception occurred on split into three parts:
            1. The content before the token that raised the error.
            2. The token that raised the error.
            3. The content after the token that raised the error.

        total
            The number of lines in source_lines.

        top
            The line number where source_lines starts.

        bottom
            The line number where source_lines ends.

        start
            The start position of the token in the template source.

        end
            The end position of the token in the template source.
        """
        start, end = token.position
        context_lines = 10
        line = 0
        upto = 0
        source_lines = []
        before = during = after = ""
        for num, next in enumerate(linebreak_iter(self.source)):
            if start >= upto and end <= next:
                line = num
                before = escape(self.source[upto:start])
                during = escape(self.source[start:end])
                after = escape(self.source[end:next])
            source_lines.append((num, escape(self.source[upto:next])))
            upto = next
        total = len(source_lines)

        top = max(1, line - context_lines)
        bottom = min(total, line + 1 + context_lines)

        # In some rare cases exc_value.args can be empty or an invalid
        # string.
        try:
            message = str(exception.args[0])
        except (IndexError, UnicodeDecodeError):
            message = "(Could not get exception message)"

        return {
            "message": message,
            "source_lines": source_lines[top:bottom],
            "before": before,
            "during": during,
            "after": after,
            "top": top,
            "bottom": bottom,
            "total": total,
            "line": line,
            "name": self.origin.name,
            "start": start,
            "end": end,
        }
```
### 36 - django/template/base.py:

Start line: 94, End line: 135

```python
class TokenType(Enum):
    TEXT = 0
    VAR = 1
    BLOCK = 2
    COMMENT = 3


class VariableDoesNotExist(Exception):
    def __init__(self, msg, params=()):
        self.msg = msg
        self.params = params

    def __str__(self):
        return self.msg % self.params


class Origin:
    def __init__(self, name, template_name=None, loader=None):
        self.name = name
        self.template_name = template_name
        self.loader = loader

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<%s name=%r>" % (self.__class__.__qualname__, self.name)

    def __eq__(self, other):
        return (
            isinstance(other, Origin)
            and self.name == other.name
            and self.loader == other.loader
        )

    @property
    def loader_name(self):
        if self.loader:
            return "%s.%s" % (
                self.loader.__module__,
                self.loader.__class__.__name__,
            )
```
### 37 - django/template/base.py:

Start line: 776, End line: 841

```python
class Variable:
    """
    A template variable, resolvable against a given context. The variable may
    be a hard-coded string (if it begins and ends with single or double quote
    marks)::

        >>> c = {'article': {'section':'News'}}
        >>> Variable('article.section').resolve(c)
        'News'
        >>> Variable('article').resolve(c)
        {'section': 'News'}
        >>> class AClass: pass
        >>> c = AClass()
        >>> c.article = AClass()
        >>> c.article.section = 'News'

    (The example assumes VARIABLE_ATTRIBUTE_SEPARATOR is '.')
    """

    __slots__ = ("var", "literal", "lookups", "translate", "message_context")

    def __init__(self, var):
        self.var = var
        self.literal = None
        self.lookups = None
        self.translate = False
        self.message_context = None

        if not isinstance(var, str):
            raise TypeError("Variable must be a string or number, got %s" % type(var))
        try:
            # First try to treat this variable as a number.
            #
            # Note that this could cause an OverflowError here that we're not
            # catching. Since this should only happen at compile time, that's
            # probably OK.

            # Try to interpret values containing a period or an 'e'/'E'
            # (possibly scientific notation) as a float;  otherwise, try int.
            if "." in var or "e" in var.lower():
                self.literal = float(var)
                # "2." is invalid
                if var[-1] == ".":
                    raise ValueError
            else:
                self.literal = int(var)
        except ValueError:
            # A ValueError means that the variable isn't a number.
            if var[0:2] == "_(" and var[-1] == ")":
                # The result of the lookup should be translated at rendering
                # time.
                self.translate = True
                var = var[2:-1]
            # If it's wrapped with quotes (single or double), then
            # we're also dealing with a literal.
            try:
                self.literal = mark_safe(unescape_string_literal(var))
            except ValueError:
                # Otherwise we'll set self.lookups so that resolve() knows we're
                # dealing with a bonafide variable
                if VARIABLE_ATTRIBUTE_SEPARATOR + "_" in var or var[0] == "_":
                    raise TemplateSyntaxError(
                        "Variables and attributes may "
                        "not begin with underscores: '%s'" % var
                    )
                self.lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))
```
### 49 - django/template/base.py:

Start line: 543, End line: 572

```python
class Parser:

    def error(self, token, e):
        """
        Return an exception annotated with the originating token. Since the
        parser can be called recursively, check if a token is already set. This
        ensures the innermost token is highlighted if an exception occurs,
        e.g. a compile error within the body of an if statement.
        """
        if not isinstance(e, Exception):
            e = TemplateSyntaxError(e)
        if not hasattr(e, "token"):
            e.token = token
        return e

    def invalid_block_tag(self, token, command, parse_until=None):
        if parse_until:
            raise self.error(
                token,
                "Invalid block tag on line %d: '%s', expected %s. Did you "
                "forget to register or load this tag?"
                % (
                    token.lineno,
                    command,
                    get_text_list(["'%s'" % p for p in parse_until], "or"),
                ),
            )
        raise self.error(
            token,
            "Invalid block tag on line %d: '%s'. Did you forget to register "
            "or load this tag?" % (token.lineno, command),
        )
```
### 50 - django/template/base.py:

Start line: 749, End line: 773

```python
class FilterExpression:

    def args_check(name, func, provided):
        provided = list(provided)
        # First argument, filter input, is implied.
        plen = len(provided) + 1
        # Check to see if a decorator is providing the real function.
        func = inspect.unwrap(func)

        args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
        alen = len(args)
        dlen = len(defaults or [])
        # Not enough OR Too many
        if plen < (alen - dlen) or plen > alen:
            raise TemplateSyntaxError(
                "%s requires %d arguments, %d provided" % (name, alen - dlen, plen)
            )

        return True

    args_check = staticmethod(args_check)

    def __str__(self):
        return self.token

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__qualname__, self.token)
```
### 57 - django/template/base.py:

Start line: 285, End line: 338

```python
def linebreak_iter(template_source):
    yield 0
    p = template_source.find("\n")
    while p >= 0:
        yield p + 1
        p = template_source.find("\n", p + 1)
    yield len(template_source) + 1


class Token:
    def __init__(self, token_type, contents, position=None, lineno=None):
        """
        A token representing a string from the template.

        token_type
            A TokenType, either .TEXT, .VAR, .BLOCK, or .COMMENT.

        contents
            The token source string.

        position
            An optional tuple containing the start and end index of the token
            in the template source. This is used for traceback information
            when debug is on.

        lineno
            The line number the token appears on in the template source.
            This is used for traceback information and gettext files.
        """
        self.token_type, self.contents = token_type, contents
        self.lineno = lineno
        self.position = position

    def __repr__(self):
        token_name = self.token_type.name.capitalize()
        return '<%s token: "%s...">' % (
            token_name,
            self.contents[:20].replace("\n", ""),
        )

    def split_contents(self):
        split = []
        bits = smart_split(self.contents)
        for bit in bits:
            # Handle translation-marked template pieces
            if bit.startswith(('_("', "_('")):
                sentinel = bit[2] + ")"
                trans_bit = [bit]
                while not bit.endswith(sentinel):
                    bit = next(bits)
                    trans_bit.append(bit)
                bit = " ".join(trans_bit)
            split.append(bit)
        return split
```
### 66 - django/template/base.py:

Start line: 341, End line: 365

```python
class Lexer:
    def __init__(self, template_string):
        self.template_string = template_string
        self.verbatim = False

    def __repr__(self):
        return '<%s template_string="%s...", verbatim=%s>' % (
            self.__class__.__qualname__,
            self.template_string[:20].replace("\n", ""),
            self.verbatim,
        )

    def tokenize(self):
        """
        Return a list of tokens from a given template_string.
        """
        in_tag = False
        lineno = 1
        result = []
        for token_string in tag_re.split(self.template_string):
            if token_string:
                result.append(self.create_token(token_string, None, lineno, in_tag))
                lineno += token_string.count("\n")
            in_tag = not in_tag
        return result
```
### 72 - django/template/base.py:

Start line: 1034, End line: 1047

```python
def render_value_in_context(value, context):
    """
    Convert any value to a string to become part of a rendered template. This
    means escaping, if required, and conversion to a string. If value is a
    string, it's expected to already be translated.
    """
    value = template_localtime(value, use_tz=context.use_tz)
    value = localize(value, use_l10n=context.use_l10n)
    if context.autoescape:
        if not issubclass(type(value), str):
            value = str(value)
        return conditional_escape(value)
    else:
        return str(value)
```
### 75 - django/template/base.py:

Start line: 179, End line: 204

```python
class Template:

    def compile_nodelist(self):
        """
        Parse and compile the template source into a nodelist. If debug
        is True and an exception occurs during parsing, the exception is
        annotated with contextual line information where it occurred in the
        template source.
        """
        if self.engine.debug:
            lexer = DebugLexer(self.source)
        else:
            lexer = Lexer(self.source)

        tokens = lexer.tokenize()
        parser = Parser(
            tokens,
            self.engine.template_libraries,
            self.engine.template_builtins,
            self.origin,
        )

        try:
            return parser.parse()
        except Exception as e:
            if self.engine.debug:
                e.template_debug = self.get_exception_info(e, e.token)
            raise
```
### 121 - django/template/base.py:

Start line: 843, End line: 865

```python
class Variable:

    def resolve(self, context):
        """Resolve this variable against a given context."""
        if self.lookups is not None:
            # We're dealing with a variable that needs to be resolved
            value = self._resolve_lookup(context)
        else:
            # We're dealing with a literal, so it's already been "resolved"
            value = self.literal
        if self.translate:
            is_safe = isinstance(value, SafeData)
            msgid = value.replace("%", "%%")
            msgid = mark_safe(msgid) if is_safe else msgid
            if self.message_context:
                return pgettext_lazy(self.message_context, msgid)
            else:
                return gettext_lazy(msgid)
        return value

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self.var)

    def __str__(self):
        return self.var
```
### 129 - django/template/base.py:

Start line: 367, End line: 399

```python
class Lexer:

    def create_token(self, token_string, position, lineno, in_tag):
        """
        Convert the given token string into a new Token object and return it.
        If in_tag is True, we are processing something that matched a tag,
        otherwise it should be treated as a literal string.
        """
        if in_tag:
            # The [0:2] and [2:-2] ranges below strip off *_TAG_START and
            # *_TAG_END. The 2's are hard-coded for performance. Using
            # len(BLOCK_TAG_START) would permit BLOCK_TAG_START to be
            # different, but it's not likely that the TAG_START values will
            # change anytime soon.
            token_start = token_string[0:2]
            if token_start == BLOCK_TAG_START:
                content = token_string[2:-2].strip()
                if self.verbatim:
                    # Then a verbatim block is being processed.
                    if content != self.verbatim:
                        return Token(TokenType.TEXT, token_string, position, lineno)
                    # Otherwise, the current verbatim block is ending.
                    self.verbatim = False
                elif content[:9] in ("verbatim", "verbatim "):
                    # Then a verbatim block is starting.
                    self.verbatim = "end%s" % content
                return Token(TokenType.BLOCK, content, position, lineno)
            if not self.verbatim:
                content = token_string[2:-2].strip()
                if token_start == VARIABLE_TAG_START:
                    return Token(TokenType.VAR, content, position, lineno)
                # BLOCK_TAG_START was handled above.
                assert token_start == COMMENT_TAG_START
                return Token(TokenType.COMMENT, content, position, lineno)
        return Token(TokenType.TEXT, token_string, position, lineno)
```
### 171 - django/template/base.py:

Start line: 436, End line: 456

```python
class Parser:
    def __init__(self, tokens, libraries=None, builtins=None, origin=None):
        # Reverse the tokens so delete_first_token(), prepend_token(), and
        # next_token() can operate at the end of the list in constant time.
        self.tokens = list(reversed(tokens))
        self.tags = {}
        self.filters = {}
        self.command_stack = []

        if libraries is None:
            libraries = {}
        if builtins is None:
            builtins = []

        self.libraries = libraries
        for builtin in builtins:
            self.add_library(builtin)
        self.origin = origin

    def __repr__(self):
        return "<%s tokens=%r>" % (self.__class__.__qualname__, self.tokens)
```
### 182 - django/template/base.py:

Start line: 574, End line: 606

```python
class Parser:

    def unclosed_block_tag(self, parse_until):
        command, token = self.command_stack.pop()
        msg = "Unclosed tag on line %d: '%s'. Looking for one of: %s." % (
            token.lineno,
            command,
            ", ".join(parse_until),
        )
        raise self.error(token, msg)

    def next_token(self):
        return self.tokens.pop()

    def prepend_token(self, token):
        self.tokens.append(token)

    def delete_first_token(self):
        del self.tokens[-1]

    def add_library(self, lib):
        self.tags.update(lib.tags)
        self.filters.update(lib.filters)

    def compile_filter(self, token):
        """
        Convenient wrapper for FilterExpression
        """
        return FilterExpression(token, self)

    def find_filter(self, filter_name):
        if filter_name in self.filters:
            return self.filters[filter_name]
        else:
            raise TemplateSyntaxError("Invalid filter: '%s'" % filter_name)
```
