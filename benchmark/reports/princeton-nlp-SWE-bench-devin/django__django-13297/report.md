# django__django-13297

| **django/django** | `8954f255bbf5f4ee997fd6de62cb50fc9b5dd697` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2537 |
| **Any found context length** | 1339 |
| **Avg pos** | 26.0 |
| **Min pos** | 6 |
| **Max pos** | 10 |
| **Top file pos** | 4 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -11,7 +11,7 @@
 from django.urls import reverse
 from django.utils.decorators import classonlymethod
 from django.utils.deprecation import RemovedInDjango40Warning
-from django.utils.functional import SimpleLazyObject
+from django.utils.functional import lazy
 
 logger = logging.getLogger('django.request')
 
@@ -169,7 +169,6 @@ def _wrap_url_kwargs_with_deprecation_warning(url_kwargs):
     context_kwargs = {}
     for key, value in url_kwargs.items():
         # Bind into function closure.
-        @SimpleLazyObject
         def access_value(key=key, value=value):
             warnings.warn(
                 'TemplateView passing URL kwargs to the context is '
@@ -178,7 +177,7 @@ def access_value(key=key, value=value):
                 RemovedInDjango40Warning, stacklevel=2,
             )
             return value
-        context_kwargs[key] = access_value
+        context_kwargs[key] = lazy(access_value, type(value))()
     return context_kwargs
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/generic/base.py | 14 | 14 | 6 | 4 | 1339
| django/views/generic/base.py | 172 | 172 | 10 | 4 | 2537
| django/views/generic/base.py | 181 | 181 | 10 | 4 | 2537


## Problem Statement

```
TemplateView.get_context_data()'s kwargs returns SimpleLazyObjects that causes a crash when filtering.
Description
	
Example Code that works in 3.0, but not in 3.1:
class OfferView(TemplateView):
	template_name = "offers/offer.html"
	def get_context_data(self, **kwargs):
		offer_slug = kwargs.get("offer_slug", "")
		offer = get_object_or_404(Account, slug=offer_slug)
		return {"offer": offer, "offer_slug": offer_slug}
In order to make this work in 3.1, you have to explicitly convert the result of kwargs.get() to a string to get the SimpleLazyObject to resolve:
class OfferView(TemplateView):
	template_name = "offers/offer.html"
	def get_context_data(self, **kwargs):
		offer_slug = kwargs.get("offer_slug", "")
		offer = get_object_or_404(Account, slug=str(offer_slug))
		return {"offer": offer, "offer_slug": offer_slug}
The error generated if you don't is:
Error binding parameter 0 - probably unsupported type
from django/db/backends/sqlite3/operations.py, line 144, in _quote_params_for_last_executed_query
In both cases, the urls.py looks like:
path(
		"/offers/<slug:offer_slug>/",
		OfferView.as_view(),
		name="offer_view",
	),
When debugging, I found that offer_slug (coming in from kwargs.get) was of type 'SimpleLazyObject' in Django 3.1, and when I explicitly converted it to a string, get_object_or_404 behaved as expected.
This is using Python 3.7.8 with SQLite.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admindocs/views.py | 87 | 115| 285 | 285 | 3296 | 
| 2 | 1 django/contrib/admindocs/views.py | 318 | 347| 201 | 486 | 3296 | 
| 3 | 2 django/views/generic/list.py | 113 | 136| 205 | 691 | 4868 | 
| 4 | 3 django/views/generic/detail.py | 78 | 108| 241 | 932 | 6183 | 
| 5 | 3 django/contrib/admindocs/views.py | 156 | 180| 234 | 1166 | 6183 | 
| **-> 6 <-** | **4 django/views/generic/base.py** | 1 | 30| 173 | 1339 | 7964 | 
| 7 | 4 django/contrib/admindocs/views.py | 56 | 84| 285 | 1624 | 7964 | 
| 8 | 4 django/views/generic/detail.py | 111 | 171| 504 | 2128 | 7964 | 
| 9 | 5 django/views/generic/__init__.py | 1 | 23| 189 | 2317 | 8154 | 
| **-> 10 <-** | **5 django/views/generic/base.py** | 157 | 182| 220 | 2537 | 8154 | 
| 11 | 5 django/views/generic/detail.py | 1 | 56| 429 | 2966 | 8154 | 
| 12 | 5 django/contrib/admindocs/views.py | 183 | 249| 584 | 3550 | 8154 | 
| 13 | 5 django/contrib/admindocs/views.py | 250 | 315| 573 | 4123 | 8154 | 
| 14 | 5 django/contrib/admindocs/views.py | 118 | 133| 154 | 4277 | 8154 | 
| 15 | 6 django/views/defaults.py | 1 | 24| 149 | 4426 | 9196 | 
| 16 | 6 django/views/generic/list.py | 139 | 158| 196 | 4622 | 9196 | 
| 17 | 7 django/views/generic/edit.py | 152 | 199| 340 | 4962 | 10912 | 
| 18 | 8 django/views/generic/dates.py | 575 | 607| 296 | 5258 | 16353 | 
| 19 | 8 django/views/generic/dates.py | 318 | 342| 229 | 5487 | 16353 | 
| 20 | 9 django/template/backends/dummy.py | 1 | 53| 325 | 5812 | 16678 | 
| 21 | 9 django/views/generic/edit.py | 1 | 67| 479 | 6291 | 16678 | 
| 22 | 9 django/views/generic/dates.py | 610 | 631| 214 | 6505 | 16678 | 
| 23 | 10 django/views/csrf.py | 15 | 100| 835 | 7340 | 18222 | 
| 24 | 10 django/contrib/admindocs/views.py | 136 | 154| 187 | 7527 | 18222 | 
| 25 | 10 django/views/csrf.py | 101 | 155| 577 | 8104 | 18222 | 
| 26 | 11 django/contrib/admin/views/main.py | 496 | 527| 224 | 8328 | 22618 | 
| 27 | 12 django/template/base.py | 668 | 703| 272 | 8600 | 30496 | 
| 28 | 12 django/contrib/admindocs/views.py | 33 | 53| 159 | 8759 | 30496 | 
| 29 | 12 django/views/generic/detail.py | 58 | 76| 154 | 8913 | 30496 | 
| 30 | 13 django/template/response.py | 60 | 145| 587 | 9500 | 31575 | 
| 31 | 14 django/contrib/admin/options.py | 1910 | 1948| 330 | 9830 | 50144 | 
| 32 | 14 django/views/generic/list.py | 1 | 48| 336 | 10166 | 50144 | 
| 33 | 15 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 10462 | 53993 | 
| 34 | 15 django/views/defaults.py | 122 | 149| 214 | 10676 | 53993 | 
| 35 | 16 django/template/context.py | 133 | 167| 288 | 10964 | 55874 | 
| 36 | **16 django/views/generic/base.py** | 210 | 241| 247 | 11211 | 55874 | 
| 37 | 17 django/contrib/auth/views.py | 286 | 327| 314 | 11525 | 58538 | 
| 38 | 17 django/contrib/admin/options.py | 1685 | 1756| 653 | 12178 | 58538 | 
| 39 | 17 django/template/response.py | 1 | 43| 383 | 12561 | 58538 | 
| 40 | 17 django/template/context.py | 233 | 260| 199 | 12760 | 58538 | 
| 41 | 18 django/template/loader.py | 52 | 67| 117 | 12877 | 58954 | 
| 42 | 18 django/contrib/admin/options.py | 1627 | 1651| 279 | 13156 | 58954 | 
| 43 | 18 django/views/generic/dates.py | 293 | 316| 203 | 13359 | 58954 | 
| 44 | 19 django/shortcuts.py | 57 | 78| 216 | 13575 | 60052 | 
| 45 | 20 django/template/engine.py | 81 | 147| 457 | 14032 | 61362 | 
| 46 | 20 django/contrib/admin/views/main.py | 442 | 494| 440 | 14472 | 61362 | 
| 47 | **20 django/views/generic/base.py** | 33 | 49| 136 | 14608 | 61362 | 
| 48 | 20 django/views/generic/edit.py | 202 | 242| 263 | 14871 | 61362 | 
| 49 | 20 django/views/generic/list.py | 50 | 75| 244 | 15115 | 61362 | 
| 50 | 21 django/db/backends/sqlite3/base.py | 399 | 419| 183 | 15298 | 67298 | 
| 51 | 21 django/shortcuts.py | 81 | 99| 200 | 15498 | 67298 | 
| 52 | 22 django/contrib/flatpages/views.py | 1 | 45| 399 | 15897 | 67888 | 
| 53 | 22 django/views/csrf.py | 1 | 13| 132 | 16029 | 67888 | 
| 54 | 23 django/views/debug.py | 1 | 47| 296 | 16325 | 72350 | 
| 55 | 23 django/contrib/admin/options.py | 1757 | 1838| 744 | 17069 | 72350 | 
| 56 | 23 django/template/engine.py | 149 | 163| 138 | 17207 | 72350 | 
| 57 | 24 django/contrib/gis/views.py | 1 | 21| 155 | 17362 | 72505 | 
| 58 | 24 django/views/generic/list.py | 77 | 111| 270 | 17632 | 72505 | 
| 59 | 24 django/contrib/admin/options.py | 1540 | 1626| 760 | 18392 | 72505 | 
| 60 | 24 django/contrib/admindocs/views.py | 1 | 30| 223 | 18615 | 72505 | 
| 61 | 25 django/db/models/query.py | 578 | 618| 331 | 18946 | 89684 | 
| 62 | 25 django/db/models/query.py | 804 | 843| 322 | 19268 | 89684 | 
| 63 | 26 django/template/context_processors.py | 35 | 50| 126 | 19394 | 90173 | 
| 64 | 27 django/core/management/templates.py | 58 | 118| 526 | 19920 | 92848 | 
| 65 | 27 django/contrib/flatpages/views.py | 48 | 70| 191 | 20111 | 92848 | 
| 66 | 27 django/contrib/admin/options.py | 1653 | 1667| 173 | 20284 | 92848 | 
| 67 | 27 django/contrib/admin/options.py | 1 | 97| 767 | 21051 | 92848 | 
| 68 | 28 django/contrib/admin/filters.py | 20 | 59| 295 | 21346 | 96941 | 
| 69 | 29 django/db/models/sql/query.py | 1467 | 1552| 801 | 22147 | 119388 | 
| 70 | 29 django/template/response.py | 45 | 58| 120 | 22267 | 119388 | 
| 71 | 30 django/urls/utils.py | 1 | 63| 460 | 22727 | 119848 | 
| 72 | 30 django/contrib/auth/views.py | 208 | 222| 133 | 22860 | 119848 | 
| 73 | 30 django/contrib/admin/views/main.py | 214 | 263| 420 | 23280 | 119848 | 
| 74 | **30 django/views/generic/base.py** | 121 | 154| 241 | 23521 | 119848 | 
| 75 | 30 django/views/generic/dates.py | 438 | 475| 296 | 23817 | 119848 | 
| 76 | 30 django/template/base.py | 1 | 94| 779 | 24596 | 119848 | 
| 77 | 30 django/contrib/auth/views.py | 330 | 362| 239 | 24835 | 119848 | 
| 78 | 30 django/views/defaults.py | 100 | 119| 149 | 24984 | 119848 | 
| 79 | 30 django/db/models/query.py | 561 | 576| 146 | 25130 | 119848 | 
| 80 | 30 django/views/generic/dates.py | 557 | 572| 123 | 25253 | 119848 | 
| 81 | 31 django/views/i18n.py | 286 | 303| 158 | 25411 | 122385 | 
| 82 | 31 django/template/base.py | 572 | 607| 361 | 25772 | 122385 | 
| 83 | 32 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 26024 | 122682 | 
| 84 | 33 django/template/loader_tags.py | 276 | 322| 392 | 26416 | 125261 | 
| 85 | 34 django/contrib/auth/urls.py | 1 | 21| 224 | 26640 | 125485 | 
| 86 | 34 django/views/debug.py | 180 | 192| 143 | 26783 | 125485 | 
| 87 | 34 django/views/generic/list.py | 161 | 199| 340 | 27123 | 125485 | 
| 88 | 34 django/template/base.py | 140 | 172| 255 | 27378 | 125485 | 
| 89 | 35 django/template/defaulttags.py | 1327 | 1391| 504 | 27882 | 136628 | 
| 90 | 35 django/views/generic/dates.py | 478 | 518| 358 | 28240 | 136628 | 
| 91 | 35 django/views/generic/edit.py | 70 | 101| 269 | 28509 | 136628 | 
| 92 | 35 django/db/models/query.py | 1651 | 1757| 1063 | 29572 | 136628 | 
| 93 | 35 django/db/models/query.py | 184 | 241| 453 | 30025 | 136628 | 
| 94 | 35 django/template/loader_tags.py | 153 | 192| 320 | 30345 | 136628 | 
| 95 | 35 django/template/base.py | 816 | 881| 540 | 30885 | 136628 | 
| 96 | 35 django/contrib/admin/options.py | 1127 | 1172| 482 | 31367 | 136628 | 
| 97 | 35 django/views/generic/dates.py | 521 | 554| 278 | 31645 | 136628 | 
| 98 | 36 django/db/models/aggregates.py | 70 | 96| 266 | 31911 | 137929 | 
| 99 | 36 django/contrib/admin/filters.py | 62 | 115| 411 | 32322 | 137929 | 
| 100 | 37 django/db/models/sql/compiler.py | 885 | 977| 839 | 33161 | 152173 | 
| 101 | 38 django/contrib/syndication/views.py | 168 | 221| 475 | 33636 | 153912 | 
| 102 | 38 django/contrib/admin/views/main.py | 1 | 45| 324 | 33960 | 153912 | 
| 103 | 38 django/views/generic/dates.py | 397 | 435| 293 | 34253 | 153912 | 
| 104 | 39 django/template/utils.py | 64 | 90| 195 | 34448 | 154620 | 
| 105 | 40 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 34899 | 156073 | 
| 106 | 40 django/contrib/auth/views.py | 224 | 244| 163 | 35062 | 156073 | 
| 107 | 40 django/contrib/auth/views.py | 247 | 284| 348 | 35410 | 156073 | 
| 108 | 40 django/db/backends/sqlite3/base.py | 174 | 202| 297 | 35707 | 156073 | 
| 109 | 41 django/template/backends/django.py | 48 | 76| 210 | 35917 | 156929 | 
| 110 | 42 django/db/models/base.py | 404 | 505| 871 | 36788 | 173573 | 
| 111 | 42 django/views/generic/dates.py | 1 | 65| 420 | 37208 | 173573 | 
| 112 | 42 django/views/generic/edit.py | 129 | 149| 182 | 37390 | 173573 | 
| 113 | 42 django/template/base.py | 199 | 275| 503 | 37893 | 173573 | 
| 114 | 42 django/db/models/base.py | 961 | 975| 212 | 38105 | 173573 | 
| 115 | 42 django/db/models/query.py | 1097 | 1138| 323 | 38428 | 173573 | 
| 116 | 42 django/db/models/sql/query.py | 697 | 732| 389 | 38817 | 173573 | 
| 117 | 42 django/db/backends/sqlite3/introspection.py | 57 | 78| 218 | 39035 | 173573 | 
| 118 | 42 django/db/models/sql/query.py | 1288 | 1353| 772 | 39807 | 173573 | 
| 119 | 42 django/contrib/syndication/views.py | 29 | 48| 195 | 40002 | 173573 | 
| 120 | 42 django/db/models/sql/query.py | 1109 | 1141| 338 | 40340 | 173573 | 
| 121 | 42 django/db/models/sql/query.py | 118 | 133| 145 | 40485 | 173573 | 
| 122 | 43 django/contrib/admin/sites.py | 221 | 240| 221 | 40706 | 177783 | 
| 123 | 44 django/template/backends/jinja2.py | 1 | 51| 341 | 41047 | 178605 | 
| 124 | 44 django/contrib/admin/views/main.py | 123 | 212| 861 | 41908 | 178605 | 
| 125 | 45 django/contrib/admin/widgets.py | 161 | 192| 243 | 42151 | 182399 | 
| 126 | 45 django/views/generic/edit.py | 103 | 126| 194 | 42345 | 182399 | 
| 127 | 46 django/contrib/admindocs/urls.py | 1 | 51| 307 | 42652 | 182706 | 
| 128 | 47 django/db/backends/sqlite3/features.py | 1 | 80| 725 | 43377 | 183431 | 
| 129 | 48 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 43620 | 187547 | 
| 130 | 48 django/template/defaulttags.py | 423 | 457| 270 | 43890 | 187547 | 
| 131 | 48 django/template/engine.py | 1 | 53| 388 | 44278 | 187547 | 
| 132 | 48 django/template/context_processors.py | 1 | 32| 218 | 44496 | 187547 | 
| 133 | 49 django/db/models/query_utils.py | 25 | 54| 185 | 44681 | 190253 | 
| 134 | 49 django/contrib/admin/options.py | 377 | 429| 504 | 45185 | 190253 | 
| 135 | 50 django/template/library.py | 201 | 234| 304 | 45489 | 192790 | 
| 136 | 50 django/views/defaults.py | 27 | 76| 401 | 45890 | 192790 | 
| 137 | 50 django/db/models/sql/query.py | 1696 | 1735| 439 | 46329 | 192790 | 
| 138 | 50 django/db/backends/sqlite3/base.py | 1 | 77| 532 | 46861 | 192790 | 
| 139 | 51 django/template/smartif.py | 114 | 147| 188 | 47049 | 194316 | 
| 140 | 51 django/contrib/admin/options.py | 1251 | 1324| 659 | 47708 | 194316 | 
| 141 | 52 django/views/decorators/debug.py | 77 | 92| 132 | 47840 | 194905 | 
| 142 | 52 django/contrib/admin/options.py | 611 | 633| 280 | 48120 | 194905 | 
| 143 | 53 django/contrib/admin/views/autocomplete.py | 1 | 35| 246 | 48366 | 195297 | 
| 144 | 53 django/db/models/query.py | 413 | 453| 343 | 48709 | 195297 | 


### Hint

```
Thanks for the report. get_object_or_404() and QuerySet.filter() with SimpleLazyObject throw the same exception in Django 2.2 or 3.0. TemplateView.get_context_data()'s kwargs returns SimpleLazyObjects in Django 3.1 which causes a crash. Passing URL kwargs into context is deprecated (see #19878) but should still work in Django 3.1 and 3.2. Regression in 4ed534758cb6a11df9f49baddecca5a6cdda9311. Reproduced at 60626162f76f26d32a38d18151700cb041201fb3.
```

## Patch

```diff
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -11,7 +11,7 @@
 from django.urls import reverse
 from django.utils.decorators import classonlymethod
 from django.utils.deprecation import RemovedInDjango40Warning
-from django.utils.functional import SimpleLazyObject
+from django.utils.functional import lazy
 
 logger = logging.getLogger('django.request')
 
@@ -169,7 +169,6 @@ def _wrap_url_kwargs_with_deprecation_warning(url_kwargs):
     context_kwargs = {}
     for key, value in url_kwargs.items():
         # Bind into function closure.
-        @SimpleLazyObject
         def access_value(key=key, value=value):
             warnings.warn(
                 'TemplateView passing URL kwargs to the context is '
@@ -178,7 +177,7 @@ def access_value(key=key, value=value):
                 RemovedInDjango40Warning, stacklevel=2,
             )
             return value
-        context_kwargs[key] = access_value
+        context_kwargs[key] = lazy(access_value, type(value))()
     return context_kwargs
 
 

```

## Test Patch

```diff
diff --git a/tests/generic_views/test_base.py b/tests/generic_views/test_base.py
--- a/tests/generic_views/test_base.py
+++ b/tests/generic_views/test_base.py
@@ -3,7 +3,8 @@
 from django.core.exceptions import ImproperlyConfigured
 from django.http import HttpResponse
 from django.test import (
-    RequestFactory, SimpleTestCase, ignore_warnings, override_settings,
+    RequestFactory, SimpleTestCase, TestCase, ignore_warnings,
+    override_settings,
 )
 from django.test.utils import require_jinja2
 from django.urls import resolve
@@ -11,6 +12,7 @@
 from django.views.generic import RedirectView, TemplateView, View
 
 from . import views
+from .models import Artist
 
 
 class SimpleView(View):
@@ -571,7 +573,9 @@ def test_template_mixin_without_template(self):
 
 
 @override_settings(ROOT_URLCONF='generic_views.urls')
-class DeprecationTests(SimpleTestCase):
+class DeprecationTests(TestCase):
+    rf = RequestFactory()
+
     @ignore_warnings(category=RemovedInDjango40Warning)
     def test_template_params(self):
         """A generic template view passes kwargs as context."""
@@ -603,3 +607,17 @@ def test_template_params_warning(self):
             str(response.context['foo2'])
         self.assertEqual(response.context['key'], 'value')
         self.assertIsInstance(response.context['view'], View)
+
+    @ignore_warnings(category=RemovedInDjango40Warning)
+    def test_template_params_filtering(self):
+        class ArtistView(TemplateView):
+            template_name = 'generic_views/about.html'
+
+            def get_context_data(self, *, artist_name, **kwargs):
+                context = super().get_context_data(**kwargs)
+                artist = Artist.objects.get(name=artist_name)
+                return {**context, 'artist': artist}
+
+        artist = Artist.objects.create(name='Rene Magritte')
+        response = ArtistView.as_view()(self.rf.get('/'), artist_name=artist.name)
+        self.assertEqual(response.context_data['artist'], artist)

```


## Code snippets

### 1 - django/contrib/admindocs/views.py:

Start line: 87, End line: 115

```python
class TemplateFilterIndexView(BaseAdminDocsView):
    template_name = 'admin_doc/template_filter_index.html'

    def get_context_data(self, **kwargs):
        filters = []
        try:
            engine = Engine.get_default()
        except ImproperlyConfigured:
            # Non-trivial TEMPLATES settings aren't supported (#24125).
            pass
        else:
            app_libs = sorted(engine.template_libraries.items())
            builtin_libs = [('', lib) for lib in engine.template_builtins]
            for module_name, library in builtin_libs + app_libs:
                for filter_name, filter_func in library.filters.items():
                    title, body, metadata = utils.parse_docstring(filter_func.__doc__)
                    title = title and utils.parse_rst(title, 'filter', _('filter:') + filter_name)
                    body = body and utils.parse_rst(body, 'filter', _('filter:') + filter_name)
                    for key in metadata:
                        metadata[key] = utils.parse_rst(metadata[key], 'filter', _('filter:') + filter_name)
                    tag_library = module_name.split('.')[-1]
                    filters.append({
                        'name': filter_name,
                        'title': title,
                        'body': body,
                        'meta': metadata,
                        'library': tag_library,
                    })
        return super().get_context_data(**{**kwargs, 'filters': filters})
```
### 2 - django/contrib/admindocs/views.py:

Start line: 318, End line: 347

```python
class TemplateDetailView(BaseAdminDocsView):
    template_name = 'admin_doc/template_detail.html'

    def get_context_data(self, **kwargs):
        template = self.kwargs['template']
        templates = []
        try:
            default_engine = Engine.get_default()
        except ImproperlyConfigured:
            # Non-trivial TEMPLATES settings aren't supported (#24125).
            pass
        else:
            # This doesn't account for template loaders (#24128).
            for index, directory in enumerate(default_engine.dirs):
                template_file = Path(directory) / template
                if template_file.exists():
                    template_contents = template_file.read_text()
                else:
                    template_contents = ''
                templates.append({
                    'file': template_file,
                    'exists': template_file.exists(),
                    'contents': template_contents,
                    'order': index,
                })
        return super().get_context_data(**{
            **kwargs,
            'name': template,
            'templates': templates,
        })
```
### 3 - django/views/generic/list.py:

Start line: 113, End line: 136

```python
class MultipleObjectMixin(ContextMixin):

    def get_context_data(self, *, object_list=None, **kwargs):
        """Get the context for this view."""
        queryset = object_list if object_list is not None else self.object_list
        page_size = self.get_paginate_by(queryset)
        context_object_name = self.get_context_object_name(queryset)
        if page_size:
            paginator, page, queryset, is_paginated = self.paginate_queryset(queryset, page_size)
            context = {
                'paginator': paginator,
                'page_obj': page,
                'is_paginated': is_paginated,
                'object_list': queryset
            }
        else:
            context = {
                'paginator': None,
                'page_obj': None,
                'is_paginated': False,
                'object_list': queryset
            }
        if context_object_name is not None:
            context[context_object_name] = queryset
        context.update(kwargs)
        return super().get_context_data(**context)
```
### 4 - django/views/generic/detail.py:

Start line: 78, End line: 108

```python
class SingleObjectMixin(ContextMixin):

    def get_slug_field(self):
        """Get the name of a slug field to be used to look up by slug."""
        return self.slug_field

    def get_context_object_name(self, obj):
        """Get the name to use for the object."""
        if self.context_object_name:
            return self.context_object_name
        elif isinstance(obj, models.Model):
            return obj._meta.model_name
        else:
            return None

    def get_context_data(self, **kwargs):
        """Insert the single object into the context dict."""
        context = {}
        if self.object:
            context['object'] = self.object
            context_object_name = self.get_context_object_name(self.object)
            if context_object_name:
                context[context_object_name] = self.object
        context.update(kwargs)
        return super().get_context_data(**context)


class BaseDetailView(SingleObjectMixin, View):
    """A base view for displaying a single object."""
    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        context = self.get_context_data(object=self.object)
        return self.render_to_response(context)
```
### 5 - django/contrib/admindocs/views.py:

Start line: 156, End line: 180

```python
class ViewDetailView(BaseAdminDocsView):

    def get_context_data(self, **kwargs):
        view = self.kwargs['view']
        view_func = self._get_view_func(view)
        if view_func is None:
            raise Http404
        title, body, metadata = utils.parse_docstring(view_func.__doc__)
        title = title and utils.parse_rst(title, 'view', _('view:') + view)
        body = body and utils.parse_rst(body, 'view', _('view:') + view)
        for key in metadata:
            metadata[key] = utils.parse_rst(metadata[key], 'model', _('view:') + view)
        return super().get_context_data(**{
            **kwargs,
            'name': view,
            'summary': title,
            'body': body,
            'meta': metadata,
        })


class ModelIndexView(BaseAdminDocsView):
    template_name = 'admin_doc/model_index.html'

    def get_context_data(self, **kwargs):
        m_list = [m._meta for m in apps.get_models()]
        return super().get_context_data(**{**kwargs, 'models': m_list})
```
### 6 - django/views/generic/base.py:

Start line: 1, End line: 30

```python
import logging
import warnings
from functools import update_wrapper

from django.core.exceptions import ImproperlyConfigured
from django.http import (
    HttpResponse, HttpResponseGone, HttpResponseNotAllowed,
    HttpResponsePermanentRedirect, HttpResponseRedirect,
)
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.decorators import classonlymethod
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import SimpleLazyObject

logger = logging.getLogger('django.request')


class ContextMixin:
    """
    A default context mixin that passes the keyword arguments received by
    get_context_data() as the template context.
    """
    extra_context = None

    def get_context_data(self, **kwargs):
        kwargs.setdefault('view', self)
        if self.extra_context is not None:
            kwargs.update(self.extra_context)
        return kwargs
```
### 7 - django/contrib/admindocs/views.py:

Start line: 56, End line: 84

```python
class TemplateTagIndexView(BaseAdminDocsView):
    template_name = 'admin_doc/template_tag_index.html'

    def get_context_data(self, **kwargs):
        tags = []
        try:
            engine = Engine.get_default()
        except ImproperlyConfigured:
            # Non-trivial TEMPLATES settings aren't supported (#24125).
            pass
        else:
            app_libs = sorted(engine.template_libraries.items())
            builtin_libs = [('', lib) for lib in engine.template_builtins]
            for module_name, library in builtin_libs + app_libs:
                for tag_name, tag_func in library.tags.items():
                    title, body, metadata = utils.parse_docstring(tag_func.__doc__)
                    title = title and utils.parse_rst(title, 'tag', _('tag:') + tag_name)
                    body = body and utils.parse_rst(body, 'tag', _('tag:') + tag_name)
                    for key in metadata:
                        metadata[key] = utils.parse_rst(metadata[key], 'tag', _('tag:') + tag_name)
                    tag_library = module_name.split('.')[-1]
                    tags.append({
                        'name': tag_name,
                        'title': title,
                        'body': body,
                        'meta': metadata,
                        'library': tag_library,
                    })
        return super().get_context_data(**{**kwargs, 'tags': tags})
```
### 8 - django/views/generic/detail.py:

Start line: 111, End line: 171

```python
class SingleObjectTemplateResponseMixin(TemplateResponseMixin):
    template_name_field = None
    template_name_suffix = '_detail'

    def get_template_names(self):
        """
        Return a list of template names to be used for the request. May not be
        called if render_to_response() is overridden. Return the following list:

        * the value of ``template_name`` on the view (if provided)
        * the contents of the ``template_name_field`` field on the
          object instance that the view is operating upon (if available)
        * ``<app_label>/<model_name><template_name_suffix>.html``
        """
        try:
            names = super().get_template_names()
        except ImproperlyConfigured:
            # If template_name isn't specified, it's not a problem --
            # we just start with an empty list.
            names = []

            # If self.template_name_field is set, grab the value of the field
            # of that name from the object; this is the most specific template
            # name, if given.
            if self.object and self.template_name_field:
                name = getattr(self.object, self.template_name_field, None)
                if name:
                    names.insert(0, name)

            # The least-specific option is the default <app>/<model>_detail.html;
            # only use this if the object in question is a model.
            if isinstance(self.object, models.Model):
                object_meta = self.object._meta
                names.append("%s/%s%s.html" % (
                    object_meta.app_label,
                    object_meta.model_name,
                    self.template_name_suffix
                ))
            elif getattr(self, 'model', None) is not None and issubclass(self.model, models.Model):
                names.append("%s/%s%s.html" % (
                    self.model._meta.app_label,
                    self.model._meta.model_name,
                    self.template_name_suffix
                ))

            # If we still haven't managed to find any template names, we should
            # re-raise the ImproperlyConfigured to alert the user.
            if not names:
                raise

        return names


class DetailView(SingleObjectTemplateResponseMixin, BaseDetailView):
    """
    Render a "detail" view of an object.

    By default this is a model instance looked up from `self.queryset`, but the
    view will support display of *any* object by overriding `self.get_object()`.
    """
```
### 9 - django/views/generic/__init__.py:

Start line: 1, End line: 23

```python
from django.views.generic.base import RedirectView, TemplateView, View
from django.views.generic.dates import (
    ArchiveIndexView, DateDetailView, DayArchiveView, MonthArchiveView,
    TodayArchiveView, WeekArchiveView, YearArchiveView,
)
from django.views.generic.detail import DetailView
from django.views.generic.edit import (
    CreateView, DeleteView, FormView, UpdateView,
)
from django.views.generic.list import ListView

__all__ = [
    'View', 'TemplateView', 'RedirectView', 'ArchiveIndexView',
    'YearArchiveView', 'MonthArchiveView', 'WeekArchiveView', 'DayArchiveView',
    'TodayArchiveView', 'DateDetailView', 'DetailView', 'FormView',
    'CreateView', 'UpdateView', 'DeleteView', 'ListView', 'GenericViewError',
]


class GenericViewError(Exception):
    """A problem in a generic view."""
    pass
```
### 10 - django/views/generic/base.py:

Start line: 157, End line: 182

```python
class TemplateView(TemplateResponseMixin, ContextMixin, View):
    """Render a template."""
    def get(self, request, *args, **kwargs):
        # RemovedInDjango40Warning: when the deprecation ends, replace with:
        #   context = self.get_context_data()
        context_kwargs = _wrap_url_kwargs_with_deprecation_warning(kwargs)
        context = self.get_context_data(**context_kwargs)
        return self.render_to_response(context)


# RemovedInDjango40Warning
def _wrap_url_kwargs_with_deprecation_warning(url_kwargs):
    context_kwargs = {}
    for key, value in url_kwargs.items():
        # Bind into function closure.
        @SimpleLazyObject
        def access_value(key=key, value=value):
            warnings.warn(
                'TemplateView passing URL kwargs to the context is '
                'deprecated. Reference %s in your template through '
                'view.kwargs instead.' % key,
                RemovedInDjango40Warning, stacklevel=2,
            )
            return value
        context_kwargs[key] = access_value
    return context_kwargs
```
### 36 - django/views/generic/base.py:

Start line: 210, End line: 241

```python
class RedirectView(View):

    def get(self, request, *args, **kwargs):
        url = self.get_redirect_url(*args, **kwargs)
        if url:
            if self.permanent:
                return HttpResponsePermanentRedirect(url)
            else:
                return HttpResponseRedirect(url)
        else:
            logger.warning(
                'Gone: %s', request.path,
                extra={'status_code': 410, 'request': request}
            )
            return HttpResponseGone()

    def head(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def options(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)
```
### 47 - django/views/generic/base.py:

Start line: 33, End line: 49

```python
class View:
    """
    Intentionally simple parent class for all views. Only implements
    dispatch-by-method and simple sanity checking.
    """

    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def __init__(self, **kwargs):
        """
        Constructor. Called in the URLconf; can contain helpful extra
        keyword arguments, and other things.
        """
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        for key, value in kwargs.items():
            setattr(self, key, value)
```
### 74 - django/views/generic/base.py:

Start line: 121, End line: 154

```python
class TemplateResponseMixin:
    """A mixin that can be used to render a template."""
    template_name = None
    template_engine = None
    response_class = TemplateResponse
    content_type = None

    def render_to_response(self, context, **response_kwargs):
        """
        Return a response, using the `response_class` for this view, with a
        template rendered with the given context.

        Pass response_kwargs to the constructor of the response class.
        """
        response_kwargs.setdefault('content_type', self.content_type)
        return self.response_class(
            request=self.request,
            template=self.get_template_names(),
            context=context,
            using=self.template_engine,
            **response_kwargs
        )

    def get_template_names(self):
        """
        Return a list of template names to be used for the request. Must return
        a list. May not be called if render_to_response() is overridden.
        """
        if self.template_name is None:
            raise ImproperlyConfigured(
                "TemplateResponseMixin requires either a definition of "
                "'template_name' or an implementation of 'get_template_names()'")
        else:
            return [self.template_name]
```
