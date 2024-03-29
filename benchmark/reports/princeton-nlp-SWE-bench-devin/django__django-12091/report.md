# django__django-12091

| **django/django** | `5d654e1e7104d2ce86ec1b9fe52865a7dca4b4be` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 5531 |
| **Any found context length** | 1871 |
| **Avg pos** | 16.666666666666668 |
| **Min pos** | 8 |
| **Max pos** | 21 |
| **Top file pos** | 3 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/http/request.py b/django/http/request.py
--- a/django/http/request.py
+++ b/django/http/request.py
@@ -1,6 +1,7 @@
 import cgi
 import codecs
 import copy
+import warnings
 from io import BytesIO
 from itertools import chain
 from urllib.parse import quote, urlencode, urljoin, urlsplit
@@ -15,6 +16,7 @@
 from django.utils.datastructures import (
     CaseInsensitiveMapping, ImmutableList, MultiValueDict,
 )
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.encoding import escape_uri_path, iri_to_uri
 from django.utils.functional import cached_property
 from django.utils.http import is_same_domain, limited_parse_qsl
@@ -256,6 +258,11 @@ def is_secure(self):
         return self.scheme == 'https'
 
     def is_ajax(self):
+        warnings.warn(
+            'request.is_ajax() is deprecated. See Django 3.1 release notes '
+            'for more details about this deprecation.',
+            RemovedInDjango40Warning,
+        )
         return self.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
 
     @property
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -48,12 +48,12 @@ def technical_500_response(request, exc_type, exc_value, tb, status_code=500):
     the values returned from sys.exc_info() and friends.
     """
     reporter = get_exception_reporter_class(request)(request, exc_type, exc_value, tb)
-    if request.is_ajax():
-        text = reporter.get_traceback_text()
-        return HttpResponse(text, status=status_code, content_type='text/plain; charset=utf-8')
-    else:
+    if request.accepts('text/html'):
         html = reporter.get_traceback_html()
         return HttpResponse(html, status=status_code, content_type='text/html')
+    else:
+        text = reporter.get_traceback_text()
+        return HttpResponse(text, status=status_code, content_type='text/plain; charset=utf-8')
 
 
 @functools.lru_cache()
diff --git a/django/views/i18n.py b/django/views/i18n.py
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -33,7 +33,7 @@ def set_language(request):
     """
     next_url = request.POST.get('next', request.GET.get('next'))
     if (
-        (next_url or not request.is_ajax()) and
+        (next_url or request.accepts('text/html')) and
         not url_has_allowed_host_and_scheme(
             url=next_url,
             allowed_hosts={request.get_host()},

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/http/request.py | 4 | 4 | 21 | 3 | 5531
| django/http/request.py | 18 | 18 | 21 | 3 | 5531
| django/http/request.py | 259 | 259 | 8 | 3 | 1871
| django/views/debug.py | 51 | 54 | - | 30 | -
| django/views/i18n.py | 36 | 36 | - | 27 | -


## Problem Statement

```
Deprecate HttpRequest.is_ajax.
Description
	 
		(last modified by Mariusz Felisiak)
	 
As discussed on ​this django-developers thread this should be deprecated.
It inspects the non-standard header X-Requested-Wiith that is set by jQuery and maybe other frameworks. However jQuery's popularity, especially for making requests, is decreasing thanks to changes such as the new fetch() JS API.
Also in the cases this property is used to determine the kind of content to send to a client, it'd be better to inspect the HTTP standard Accept header.
For these reasons Flask has deprecated its similar property is_xhr.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/utils/deprecation.py | 73 | 95| 158 | 158 | 678 | 
| 2 | 2 django/views/decorators/clickjacking.py | 1 | 19| 138 | 296 | 1054 | 
| 3 | **3 django/http/request.py** | 355 | 386| 254 | 550 | 6247 | 
| 4 | **3 django/http/request.py** | 389 | 412| 185 | 735 | 6247 | 
| 5 | 3 django/utils/deprecation.py | 1 | 27| 181 | 916 | 6247 | 
| 6 | **3 django/http/request.py** | 304 | 324| 189 | 1105 | 6247 | 
| 7 | 3 django/views/decorators/clickjacking.py | 22 | 54| 238 | 1343 | 6247 | 
| **-> 8 <-** | **3 django/http/request.py** | 230 | 302| 528 | 1871 | 6247 | 
| 9 | 4 django/middleware/clickjacking.py | 1 | 46| 359 | 2230 | 6606 | 
| 10 | **4 django/http/request.py** | 151 | 158| 111 | 2341 | 6606 | 
| 11 | 5 django/http/response.py | 1 | 26| 157 | 2498 | 11125 | 
| 12 | **5 django/http/request.py** | 42 | 96| 394 | 2892 | 11125 | 
| 13 | 5 django/http/response.py | 135 | 156| 176 | 3068 | 11125 | 
| 14 | 6 django/views/decorators/http.py | 1 | 52| 350 | 3418 | 12077 | 
| 15 | **6 django/http/request.py** | 98 | 115| 183 | 3601 | 12077 | 
| 16 | 6 django/http/response.py | 536 | 561| 159 | 3760 | 12077 | 
| 17 | 7 django/views/csrf.py | 15 | 100| 835 | 4595 | 13621 | 
| 18 | 8 django/core/checks/security/csrf.py | 1 | 41| 299 | 4894 | 13920 | 
| 19 | **8 django/http/request.py** | 137 | 149| 133 | 5027 | 13920 | 
| 20 | **8 django/http/request.py** | 160 | 192| 235 | 5262 | 13920 | 
| **-> 21 <-** | **8 django/http/request.py** | 1 | 39| 269 | 5531 | 13920 | 
| 22 | **8 django/http/request.py** | 573 | 604| 241 | 5772 | 13920 | 
| 23 | 9 django/utils/http.py | 1 | 73| 714 | 6486 | 18098 | 
| 24 | 10 django/utils/cache.py | 136 | 151| 188 | 6674 | 21846 | 
| 25 | 11 django/core/checks/security/base.py | 85 | 180| 710 | 7384 | 23627 | 
| 26 | 12 django/contrib/admindocs/middleware.py | 1 | 29| 234 | 7618 | 23862 | 
| 27 | 13 django/middleware/csrf.py | 205 | 327| 1189 | 8807 | 26716 | 
| 28 | 14 django/contrib/gis/views.py | 1 | 21| 155 | 8962 | 26871 | 
| 29 | 14 django/utils/cache.py | 242 | 273| 256 | 9218 | 26871 | 
| 30 | 15 django/views/decorators/cache.py | 27 | 48| 153 | 9371 | 27234 | 
| 31 | 15 django/utils/http.py | 76 | 101| 195 | 9566 | 27234 | 
| 32 | **15 django/http/request.py** | 326 | 353| 279 | 9845 | 27234 | 
| 33 | 16 django/db/models/fields/__init__.py | 364 | 390| 199 | 10044 | 44802 | 
| 34 | 17 django/middleware/cache.py | 131 | 157| 252 | 10296 | 46420 | 
| 35 | **17 django/http/request.py** | 117 | 135| 187 | 10483 | 46420 | 
| 36 | 18 django/http/__init__.py | 1 | 22| 197 | 10680 | 46617 | 
| 37 | 19 django/core/servers/basehttp.py | 159 | 178| 170 | 10850 | 48362 | 
| 38 | 20 django/contrib/admin/options.py | 1 | 96| 769 | 11619 | 66863 | 
| 39 | 21 django/core/handlers/asgi.py | 1 | 124| 982 | 12601 | 69177 | 
| 40 | 21 django/http/response.py | 481 | 499| 186 | 12787 | 69177 | 
| 41 | 21 django/utils/deprecation.py | 30 | 70| 336 | 13123 | 69177 | 
| 42 | 21 django/http/response.py | 502 | 533| 153 | 13276 | 69177 | 
| 43 | 21 django/utils/http.py | 389 | 415| 329 | 13605 | 69177 | 
| 44 | 21 django/http/response.py | 204 | 220| 193 | 13798 | 69177 | 
| 45 | 21 django/views/csrf.py | 101 | 155| 577 | 14375 | 69177 | 
| 46 | 22 django/views/defaults.py | 100 | 119| 149 | 14524 | 70219 | 
| 47 | 23 django/contrib/sites/requests.py | 1 | 20| 131 | 14655 | 70350 | 
| 48 | 24 django/views/decorators/csrf.py | 1 | 57| 460 | 15115 | 70810 | 
| 49 | 24 django/db/models/fields/__init__.py | 1812 | 1840| 191 | 15306 | 70810 | 
| 50 | 24 django/utils/cache.py | 301 | 321| 217 | 15523 | 70810 | 
| 51 | 25 django/middleware/gzip.py | 1 | 52| 415 | 15938 | 71226 | 
| 52 | 25 django/utils/cache.py | 1 | 35| 284 | 16222 | 71226 | 
| 53 | 25 django/middleware/csrf.py | 1 | 42| 330 | 16552 | 71226 | 
| 54 | 25 django/views/decorators/http.py | 77 | 122| 347 | 16899 | 71226 | 
| 55 | 26 django/db/models/query_utils.py | 25 | 54| 185 | 17084 | 73942 | 
| 56 | **27 django/views/i18n.py** | 88 | 191| 711 | 17795 | 76487 | 
| 57 | 28 django/contrib/sitemaps/views.py | 1 | 19| 131 | 17926 | 77261 | 
| 58 | 28 django/middleware/csrf.py | 122 | 156| 267 | 18193 | 77261 | 
| 59 | 29 django/core/handlers/base.py | 64 | 83| 168 | 18361 | 78439 | 
| 60 | **30 django/views/debug.py** | 171 | 183| 143 | 18504 | 82781 | 
| 61 | 31 django/middleware/http.py | 1 | 42| 335 | 18839 | 83116 | 
| 62 | 32 django/middleware/common.py | 149 | 175| 254 | 19093 | 84627 | 
| 63 | 33 django/core/handlers/wsgi.py | 64 | 119| 486 | 19579 | 86324 | 
| 64 | 34 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 19924 | 86669 | 
| 65 | 35 django/views/decorators/debug.py | 77 | 92| 132 | 20056 | 87258 | 
| 66 | 35 django/views/decorators/debug.py | 1 | 44| 274 | 20330 | 87258 | 
| 67 | 35 django/contrib/admin/options.py | 1235 | 1308| 659 | 20989 | 87258 | 
| 68 | 36 django/contrib/admin/sites.py | 353 | 373| 157 | 21146 | 91449 | 
| 69 | **36 django/views/debug.py** | 146 | 169| 177 | 21323 | 91449 | 
| 70 | 37 django/forms/models.py | 309 | 348| 387 | 21710 | 103051 | 
| 71 | 38 django/utils/html.py | 1 | 75| 614 | 22324 | 106153 | 
| 72 | 38 django/db/models/fields/__init__.py | 1246 | 1287| 332 | 22656 | 106153 | 
| 73 | 38 django/core/checks/security/base.py | 213 | 226| 131 | 22787 | 106153 | 
| 74 | 38 django/db/models/fields/__init__.py | 1056 | 1085| 218 | 23005 | 106153 | 
| 75 | 38 django/middleware/common.py | 34 | 61| 257 | 23262 | 106153 | 
| 76 | 39 django/views/decorators/vary.py | 1 | 42| 232 | 23494 | 106386 | 
| 77 | 40 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 23641 | 106534 | 
| 78 | **40 django/views/i18n.py** | 1 | 20| 120 | 23761 | 106534 | 
| 79 | 40 django/core/servers/basehttp.py | 122 | 157| 280 | 24041 | 106534 | 
| 80 | 40 django/utils/http.py | 199 | 221| 166 | 24207 | 106534 | 
| 81 | 40 django/middleware/csrf.py | 181 | 203| 230 | 24437 | 106534 | 
| 82 | 40 django/http/response.py | 240 | 279| 319 | 24756 | 106534 | 
| 83 | 41 django/core/handlers/exception.py | 41 | 102| 499 | 25255 | 107475 | 
| 84 | 41 django/contrib/admin/options.py | 1404 | 1443| 309 | 25564 | 107475 | 
| 85 | 41 django/middleware/csrf.py | 158 | 179| 173 | 25737 | 107475 | 
| 86 | 42 django/core/checks/model_checks.py | 178 | 211| 332 | 26069 | 109262 | 
| 87 | 42 django/core/checks/security/base.py | 1 | 83| 732 | 26801 | 109262 | 
| 88 | 42 django/utils/http.py | 159 | 196| 380 | 27181 | 109262 | 
| 89 | 43 django/contrib/admin/utils.py | 287 | 305| 175 | 27356 | 113379 | 
| 90 | 43 django/contrib/admin/utils.py | 1 | 24| 228 | 27584 | 113379 | 
| 91 | 43 django/middleware/csrf.py | 93 | 119| 224 | 27808 | 113379 | 
| 92 | 43 django/utils/html.py | 352 | 379| 212 | 28020 | 113379 | 
| 93 | 44 django/middleware/security.py | 1 | 29| 275 | 28295 | 113882 | 
| 94 | 44 django/views/defaults.py | 1 | 24| 149 | 28444 | 113882 | 
| 95 | 45 django/conf/global_settings.py | 497 | 643| 924 | 29368 | 119520 | 
| 96 | 45 django/db/models/fields/__init__.py | 1106 | 1144| 293 | 29661 | 119520 | 
| 97 | 45 django/http/response.py | 313 | 345| 203 | 29864 | 119520 | 
| 98 | 45 django/utils/cache.py | 276 | 298| 255 | 30119 | 119520 | 
| 99 | 45 django/views/csrf.py | 1 | 13| 132 | 30251 | 119520 | 
| 100 | 46 django/contrib/admin/tests.py | 1 | 36| 264 | 30515 | 120997 | 
| 101 | 46 django/db/models/fields/__init__.py | 1843 | 1920| 567 | 31082 | 120997 | 
| 102 | **46 django/http/request.py** | 646 | 666| 184 | 31266 | 120997 | 
| 103 | 47 docs/_ext/djangodocs.py | 108 | 169| 526 | 31792 | 124042 | 
| 104 | 47 django/utils/html.py | 179 | 197| 161 | 31953 | 124042 | 
| 105 | **47 django/http/request.py** | 607 | 621| 118 | 32071 | 124042 | 
| 106 | 47 django/conf/global_settings.py | 147 | 262| 859 | 32930 | 124042 | 
| 107 | 48 django/template/defaultfilters.py | 56 | 91| 203 | 33133 | 130116 | 
| 108 | 48 django/template/defaultfilters.py | 325 | 409| 499 | 33632 | 130116 | 
| 109 | 48 django/core/servers/basehttp.py | 97 | 119| 211 | 33843 | 130116 | 
| 110 | 48 django/http/response.py | 29 | 106| 614 | 34457 | 130116 | 
| 111 | 49 django/http/cookie.py | 1 | 27| 188 | 34645 | 130305 | 
| 112 | 50 django/utils/translation/__init__.py | 68 | 147| 489 | 35134 | 132641 | 
| 113 | 50 django/utils/cache.py | 154 | 191| 447 | 35581 | 132641 | 
| 114 | 50 django/core/checks/model_checks.py | 155 | 176| 263 | 35844 | 132641 | 
| 115 | 50 django/core/checks/model_checks.py | 129 | 153| 268 | 36112 | 132641 | 
| 116 | 50 django/utils/cache.py | 215 | 239| 235 | 36347 | 132641 | 
| 117 | 51 django/contrib/admin/checks.py | 448 | 474| 190 | 36537 | 141689 | 
| 118 | 51 django/http/response.py | 108 | 133| 270 | 36807 | 141689 | 
| 119 | 51 django/contrib/admin/checks.py | 1019 | 1046| 204 | 37011 | 141689 | 
| 120 | 52 django/core/checks/security/sessions.py | 1 | 98| 572 | 37583 | 142262 | 
| 121 | 53 django/contrib/admindocs/views.py | 1 | 30| 223 | 37806 | 145570 | 
| 122 | 53 django/views/decorators/http.py | 55 | 76| 272 | 38078 | 145570 | 
| 123 | **53 django/http/request.py** | 194 | 228| 407 | 38485 | 145570 | 
| 124 | 53 django/middleware/common.py | 1 | 32| 247 | 38732 | 145570 | 
| 125 | 53 django/contrib/admin/options.py | 2070 | 2122| 451 | 39183 | 145570 | 
| 126 | 54 django/utils/asyncio.py | 1 | 35| 222 | 39405 | 145793 | 
| 127 | 55 django/utils/decorators.py | 114 | 153| 313 | 39718 | 147052 | 
| 128 | 55 django/db/models/fields/__init__.py | 1340 | 1368| 281 | 39999 | 147052 | 
| 129 | 55 django/middleware/cache.py | 117 | 129| 125 | 40124 | 147052 | 
| 130 | 56 django/contrib/auth/admin.py | 191 | 206| 185 | 40309 | 148778 | 
| 131 | 56 django/db/models/fields/__init__.py | 1146 | 1188| 287 | 40596 | 148778 | 
| 132 | 56 django/core/checks/security/base.py | 183 | 210| 208 | 40804 | 148778 | 
| 133 | 56 django/contrib/admin/options.py | 1099 | 1109| 125 | 40929 | 148778 | 
| 134 | 56 django/utils/cache.py | 106 | 133| 181 | 41110 | 148778 | 
| 135 | 57 django/contrib/auth/middleware.py | 46 | 82| 360 | 41470 | 149772 | 
| 136 | 57 django/contrib/auth/middleware.py | 26 | 44| 178 | 41648 | 149772 | 
| 137 | 58 django/utils/inspect.py | 36 | 64| 179 | 41827 | 150168 | 
| 138 | 59 django/db/models/base.py | 1 | 50| 330 | 42157 | 165528 | 
| 139 | 59 django/db/models/fields/__init__.py | 2085 | 2126| 325 | 42482 | 165528 | 
| 140 | 59 django/views/decorators/debug.py | 47 | 75| 199 | 42681 | 165528 | 
| 141 | 60 django/contrib/admin/helpers.py | 1 | 30| 198 | 42879 | 168721 | 
| 142 | 60 django/http/response.py | 222 | 238| 181 | 43060 | 168721 | 
| 143 | 60 django/utils/http.py | 418 | 480| 318 | 43378 | 168721 | 
| 144 | 60 django/utils/cache.py | 38 | 103| 557 | 43935 | 168721 | 
| 145 | 60 django/utils/translation/__init__.py | 1 | 37| 297 | 44232 | 168721 | 
| 146 | 61 django/forms/fields.py | 1169 | 1181| 113 | 44345 | 177734 | 
| 147 | 61 django/db/models/fields/__init__.py | 1289 | 1338| 342 | 44687 | 177734 | 
| 148 | 61 django/db/models/fields/__init__.py | 1088 | 1104| 175 | 44862 | 177734 | 
| 149 | 61 django/utils/http.py | 302 | 328| 268 | 45130 | 177734 | 
| 150 | 61 django/contrib/admin/options.py | 1158 | 1233| 664 | 45794 | 177734 | 
| 151 | 62 django/db/migrations/autodetector.py | 883 | 902| 184 | 45978 | 189469 | 
| 152 | 62 django/db/models/fields/__init__.py | 1190 | 1208| 180 | 46158 | 189469 | 
| 153 | 63 django/views/decorators/gzip.py | 1 | 6| 0 | 46158 | 189520 | 
| 154 | 64 django/db/models/__init__.py | 1 | 52| 591 | 46749 | 190111 | 
| 155 | 64 django/contrib/admin/checks.py | 414 | 423| 125 | 46874 | 190111 | 
| 156 | 64 django/http/response.py | 564 | 590| 264 | 47138 | 190111 | 
| 157 | 64 django/middleware/security.py | 31 | 56| 234 | 47372 | 190111 | 
| 158 | 65 django/urls/base.py | 90 | 157| 383 | 47755 | 191297 | 
| 159 | 66 django/views/static.py | 108 | 136| 206 | 47961 | 192349 | 
| 160 | **66 django/views/debug.py** | 75 | 102| 218 | 48179 | 192349 | 
| 161 | 67 django/core/validators.py | 1 | 14| 111 | 48290 | 196584 | 
| 162 | 67 django/contrib/admin/options.py | 1524 | 1610| 760 | 49050 | 196584 | 


### Hint

```
The first step would be to document current limitations of the method. Second step would be to avoid using it as much as possible in Django's own code. Finally the deprecation can take place. It remains to be shown how the ​request.accepts proposal can play a role here. A good exercise would be to replace that example: ​https://docs.djangoproject.com/en/2.2/topics/class-based-views/generic-editing/#ajax-example (or would you simply remove it?)
```

## Patch

```diff
diff --git a/django/http/request.py b/django/http/request.py
--- a/django/http/request.py
+++ b/django/http/request.py
@@ -1,6 +1,7 @@
 import cgi
 import codecs
 import copy
+import warnings
 from io import BytesIO
 from itertools import chain
 from urllib.parse import quote, urlencode, urljoin, urlsplit
@@ -15,6 +16,7 @@
 from django.utils.datastructures import (
     CaseInsensitiveMapping, ImmutableList, MultiValueDict,
 )
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.encoding import escape_uri_path, iri_to_uri
 from django.utils.functional import cached_property
 from django.utils.http import is_same_domain, limited_parse_qsl
@@ -256,6 +258,11 @@ def is_secure(self):
         return self.scheme == 'https'
 
     def is_ajax(self):
+        warnings.warn(
+            'request.is_ajax() is deprecated. See Django 3.1 release notes '
+            'for more details about this deprecation.',
+            RemovedInDjango40Warning,
+        )
         return self.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
 
     @property
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -48,12 +48,12 @@ def technical_500_response(request, exc_type, exc_value, tb, status_code=500):
     the values returned from sys.exc_info() and friends.
     """
     reporter = get_exception_reporter_class(request)(request, exc_type, exc_value, tb)
-    if request.is_ajax():
-        text = reporter.get_traceback_text()
-        return HttpResponse(text, status=status_code, content_type='text/plain; charset=utf-8')
-    else:
+    if request.accepts('text/html'):
         html = reporter.get_traceback_html()
         return HttpResponse(html, status=status_code, content_type='text/html')
+    else:
+        text = reporter.get_traceback_text()
+        return HttpResponse(text, status=status_code, content_type='text/plain; charset=utf-8')
 
 
 @functools.lru_cache()
diff --git a/django/views/i18n.py b/django/views/i18n.py
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -33,7 +33,7 @@ def set_language(request):
     """
     next_url = request.POST.get('next', request.GET.get('next'))
     if (
-        (next_url or not request.is_ajax()) and
+        (next_url or request.accepts('text/html')) and
         not url_has_allowed_host_and_scheme(
             url=next_url,
             allowed_hosts={request.get_host()},

```

## Test Patch

```diff
diff --git a/tests/requests/test_is_ajax_deprecations.py b/tests/requests/test_is_ajax_deprecations.py
new file mode 100644
--- /dev/null
+++ b/tests/requests/test_is_ajax_deprecations.py
@@ -0,0 +1,12 @@
+from django.http import HttpRequest
+from django.test import SimpleTestCase, ignore_warnings
+from django.utils.deprecation import RemovedInDjango40Warning
+
+
+@ignore_warnings(category=RemovedInDjango40Warning)
+class TestDeprecatedIsAjax(SimpleTestCase):
+    def test_is_ajax(self):
+        request = HttpRequest()
+        self.assertIs(request.is_ajax(), False)
+        request.META['HTTP_X_REQUESTED_WITH'] = 'XMLHttpRequest'
+        self.assertIs(request.is_ajax(), True)
diff --git a/tests/view_tests/tests/test_debug.py b/tests/view_tests/tests/test_debug.py
--- a/tests/view_tests/tests/test_debug.py
+++ b/tests/view_tests/tests/test_debug.py
@@ -1247,7 +1247,7 @@ def test_exception_report_uses_meta_filtering(self):
         response = self.client.get(
             '/raises500/',
             HTTP_SECRET_HEADER='super_secret',
-            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
+            HTTP_ACCEPT='application/json',
         )
         self.assertNotIn(b'super_secret', response.content)
 
@@ -1289,17 +1289,17 @@ def test_hidden_settings_override(self):
         )
 
 
-class AjaxResponseExceptionReporterFilter(ExceptionReportTestMixin, LoggingCaptureMixin, SimpleTestCase):
+class NonHTMLResponseExceptionReporterFilter(ExceptionReportTestMixin, LoggingCaptureMixin, SimpleTestCase):
     """
     Sensitive information can be filtered out of error reports.
 
-    Here we specifically test the plain text 500 debug-only error page served
-    when it has been detected the request was sent by JS code. We don't check
-    for (non)existence of frames vars in the traceback information section of
-    the response content because we don't include them in these error pages.
+    The plain text 500 debug-only error page is served when it has been
+    detected the request doesn't accept HTML content. Don't check for
+    (non)existence of frames vars in the traceback information section of the
+    response content because they're not included in these error pages.
     Refs #14614.
     """
-    rf = RequestFactory(HTTP_X_REQUESTED_WITH='XMLHttpRequest')
+    rf = RequestFactory(HTTP_ACCEPT='application/json')
 
     def test_non_sensitive_request(self):
         """
@@ -1346,8 +1346,8 @@ def test_custom_exception_reporter_filter(self):
             self.verify_unsafe_response(custom_exception_reporter_filter_view, check_for_vars=False)
 
     @override_settings(DEBUG=True, ROOT_URLCONF='view_tests.urls')
-    def test_ajax_response_encoding(self):
-        response = self.client.get('/raises500/', HTTP_X_REQUESTED_WITH='XMLHttpRequest')
+    def test_non_html_response_encoding(self):
+        response = self.client.get('/raises500/', HTTP_ACCEPT='application/json')
         self.assertEqual(response['Content-Type'], 'text/plain; charset=utf-8')
 
 
diff --git a/tests/view_tests/tests/test_i18n.py b/tests/view_tests/tests/test_i18n.py
--- a/tests/view_tests/tests/test_i18n.py
+++ b/tests/view_tests/tests/test_i18n.py
@@ -111,11 +111,12 @@ def test_setlang_default_redirect(self):
 
     def test_setlang_performs_redirect_for_ajax_if_explicitly_requested(self):
         """
-        The set_language view redirects to the "next" parameter for AJAX calls.
+        The set_language view redirects to the "next" parameter for requests
+        not accepting HTML response content.
         """
         lang_code = self._get_inactive_language_code()
         post_data = {'language': lang_code, 'next': '/'}
-        response = self.client.post('/i18n/setlang/', post_data, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
+        response = self.client.post('/i18n/setlang/', post_data, HTTP_ACCEPT='application/json')
         self.assertRedirects(response, '/')
         self.assertEqual(self.client.cookies[settings.LANGUAGE_COOKIE_NAME].value, lang_code)
         with ignore_warnings(category=RemovedInDjango40Warning):
@@ -123,12 +124,12 @@ def test_setlang_performs_redirect_for_ajax_if_explicitly_requested(self):
 
     def test_setlang_doesnt_perform_a_redirect_to_referer_for_ajax(self):
         """
-        The set_language view doesn't redirect to the HTTP referer header for
-        AJAX calls.
+        The set_language view doesn't redirect to the HTTP referer header if
+        the request doesn't accept HTML response content.
         """
         lang_code = self._get_inactive_language_code()
         post_data = {'language': lang_code}
-        headers = {'HTTP_REFERER': '/', 'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'}
+        headers = {'HTTP_REFERER': '/', 'HTTP_ACCEPT': 'application/json'}
         response = self.client.post('/i18n/setlang/', post_data, **headers)
         self.assertEqual(response.status_code, 204)
         self.assertEqual(self.client.cookies[settings.LANGUAGE_COOKIE_NAME].value, lang_code)
@@ -137,11 +138,12 @@ def test_setlang_doesnt_perform_a_redirect_to_referer_for_ajax(self):
 
     def test_setlang_doesnt_perform_a_default_redirect_for_ajax(self):
         """
-        The set_language view returns 204 for AJAX calls by default.
+        The set_language view returns 204 by default for requests not accepting
+        HTML response content.
         """
         lang_code = self._get_inactive_language_code()
         post_data = {'language': lang_code}
-        response = self.client.post('/i18n/setlang/', post_data, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
+        response = self.client.post('/i18n/setlang/', post_data, HTTP_ACCEPT='application/json')
         self.assertEqual(response.status_code, 204)
         self.assertEqual(self.client.cookies[settings.LANGUAGE_COOKIE_NAME].value, lang_code)
         with ignore_warnings(category=RemovedInDjango40Warning):
@@ -149,11 +151,12 @@ def test_setlang_doesnt_perform_a_default_redirect_for_ajax(self):
 
     def test_setlang_unsafe_next_for_ajax(self):
         """
-        The fallback to root URL for the set_language view works for AJAX calls.
+        The fallback to root URL for the set_language view works for requests
+        not accepting HTML response content.
         """
         lang_code = self._get_inactive_language_code()
         post_data = {'language': lang_code, 'next': '//unsafe/redirection/'}
-        response = self.client.post('/i18n/setlang/', post_data, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
+        response = self.client.post('/i18n/setlang/', post_data, HTTP_ACCEPT='application/json')
         self.assertEqual(response.url, '/')
         self.assertEqual(self.client.cookies[settings.LANGUAGE_COOKIE_NAME].value, lang_code)
 

```


## Code snippets

### 1 - django/utils/deprecation.py:

Start line: 73, End line: 95

```python
class DeprecationInstanceCheck(type):
    def __instancecheck__(self, instance):
        warnings.warn(
            "`%s` is deprecated, use `%s` instead." % (self.__name__, self.alternative),
            self.deprecation_warning, 2
        )
        return super().__instancecheck__(instance)


class MiddlewareMixin:
    def __init__(self, get_response=None):
        self.get_response = get_response
        super().__init__()

    def __call__(self, request):
        response = None
        if hasattr(self, 'process_request'):
            response = self.process_request(request)
        response = response or self.get_response(request)
        if hasattr(self, 'process_response'):
            response = self.process_response(request, response)
        return response
```
### 2 - django/views/decorators/clickjacking.py:

Start line: 1, End line: 19

```python
from functools import wraps


def xframe_options_deny(view_func):
    """
    Modify a view function so its response has the X-Frame-Options HTTP
    header set to 'DENY' as long as the response doesn't already have that
    header set. Usage:

    @xframe_options_deny
    def some_view(request):
        ...
    """
    def wrapped_view(*args, **kwargs):
        resp = view_func(*args, **kwargs)
        if resp.get('X-Frame-Options') is None:
            resp['X-Frame-Options'] = 'DENY'
        return resp
    return wraps(view_func)(wrapped_view)
```
### 3 - django/http/request.py:

Start line: 355, End line: 386

```python
class HttpRequest:

    def close(self):
        if hasattr(self, '_files'):
            for f in chain.from_iterable(l[1] for l in self._files.lists()):
                f.close()

    # File-like and iterator interface.
    #
    # Expects self._stream to be set to an appropriate source of bytes by
    # a corresponding request subclass (e.g. WSGIRequest).
    # Also when request data has already been read by request.POST or
    # request.body, self._stream points to a BytesIO instance
    # containing that data.

    def read(self, *args, **kwargs):
        self._read_started = True
        try:
            return self._stream.read(*args, **kwargs)
        except OSError as e:
            raise UnreadablePostError(*e.args) from e

    def readline(self, *args, **kwargs):
        self._read_started = True
        try:
            return self._stream.readline(*args, **kwargs)
        except OSError as e:
            raise UnreadablePostError(*e.args) from e

    def __iter__(self):
        return iter(self.readline, b'')

    def readlines(self):
        return list(self)
```
### 4 - django/http/request.py:

Start line: 389, End line: 412

```python
class HttpHeaders(CaseInsensitiveMapping):
    HTTP_PREFIX = 'HTTP_'
    # PEP 333 gives two headers which aren't prepended with HTTP_.
    UNPREFIXED_HEADERS = {'CONTENT_TYPE', 'CONTENT_LENGTH'}

    def __init__(self, environ):
        headers = {}
        for header, value in environ.items():
            name = self.parse_header_name(header)
            if name:
                headers[name] = value
        super().__init__(headers)

    def __getitem__(self, key):
        """Allow header lookup using underscores in place of hyphens."""
        return super().__getitem__(key.replace('_', '-'))

    @classmethod
    def parse_header_name(cls, header):
        if header.startswith(cls.HTTP_PREFIX):
            header = header[len(cls.HTTP_PREFIX):]
        elif header not in cls.UNPREFIXED_HEADERS:
            return None
        return header.replace('_', '-').title()
```
### 5 - django/utils/deprecation.py:

Start line: 1, End line: 27

```python
import inspect
import warnings


class RemovedInNextVersionWarning(DeprecationWarning):
    pass


class RemovedInDjango40Warning(PendingDeprecationWarning):
    pass


class warn_about_renamed_method:
    def __init__(self, class_name, old_method_name, new_method_name, deprecation_warning):
        self.class_name = class_name
        self.old_method_name = old_method_name
        self.new_method_name = new_method_name
        self.deprecation_warning = deprecation_warning

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            warnings.warn(
                "`%s.%s` is deprecated, use `%s` instead." %
                (self.class_name, self.old_method_name, self.new_method_name),
                self.deprecation_warning, 2)
            return f(*args, **kwargs)
        return wrapped
```
### 6 - django/http/request.py:

Start line: 304, End line: 324

```python
class HttpRequest:

    @property
    def body(self):
        if not hasattr(self, '_body'):
            if self._read_started:
                raise RawPostDataException("You cannot access body after reading from request's data stream")

            # Limit the maximum request data size that will be handled in-memory.
            if (settings.DATA_UPLOAD_MAX_MEMORY_SIZE is not None and
                    int(self.META.get('CONTENT_LENGTH') or 0) > settings.DATA_UPLOAD_MAX_MEMORY_SIZE):
                raise RequestDataTooBig('Request body exceeded settings.DATA_UPLOAD_MAX_MEMORY_SIZE.')

            try:
                self._body = self.read()
            except OSError as e:
                raise UnreadablePostError(*e.args) from e
            self._stream = BytesIO(self._body)
        return self._body

    def _mark_post_parse_error(self):
        self._post = QueryDict()
        self._files = MultiValueDict()
```
### 7 - django/views/decorators/clickjacking.py:

Start line: 22, End line: 54

```python
def xframe_options_sameorigin(view_func):
    """
    Modify a view function so its response has the X-Frame-Options HTTP
    header set to 'SAMEORIGIN' as long as the response doesn't already have
    that header set. Usage:

    @xframe_options_sameorigin
    def some_view(request):
        ...
    """
    def wrapped_view(*args, **kwargs):
        resp = view_func(*args, **kwargs)
        if resp.get('X-Frame-Options') is None:
            resp['X-Frame-Options'] = 'SAMEORIGIN'
        return resp
    return wraps(view_func)(wrapped_view)


def xframe_options_exempt(view_func):
    """
    Modify a view function by setting a response variable that instructs
    XFrameOptionsMiddleware to NOT set the X-Frame-Options HTTP header. Usage:

    @xframe_options_exempt
    def some_view(request):
        ...
    """
    def wrapped_view(*args, **kwargs):
        resp = view_func(*args, **kwargs)
        resp.xframe_options_exempt = True
        return resp
    return wraps(view_func)(wrapped_view)
```
### 8 - django/http/request.py:

Start line: 230, End line: 302

```python
class HttpRequest:

    @cached_property
    def _current_scheme_host(self):
        return '{}://{}'.format(self.scheme, self.get_host())

    def _get_scheme(self):
        """
        Hook for subclasses like WSGIRequest to implement. Return 'http' by
        default.
        """
        return 'http'

    @property
    def scheme(self):
        if settings.SECURE_PROXY_SSL_HEADER:
            try:
                header, secure_value = settings.SECURE_PROXY_SSL_HEADER
            except ValueError:
                raise ImproperlyConfigured(
                    'The SECURE_PROXY_SSL_HEADER setting must be a tuple containing two values.'
                )
            header_value = self.META.get(header)
            if header_value is not None:
                return 'https' if header_value == secure_value else 'http'
        return self._get_scheme()

    def is_secure(self):
        return self.scheme == 'https'

    def is_ajax(self):
        return self.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, val):
        """
        Set the encoding used for GET/POST accesses. If the GET or POST
        dictionary has already been created, remove and recreate it on the
        next access (so that it is decoded correctly).
        """
        self._encoding = val
        if hasattr(self, 'GET'):
            del self.GET
        if hasattr(self, '_post'):
            del self._post

    def _initialize_handlers(self):
        self._upload_handlers = [uploadhandler.load_handler(handler, self)
                                 for handler in settings.FILE_UPLOAD_HANDLERS]

    @property
    def upload_handlers(self):
        if not self._upload_handlers:
            # If there are no upload handlers defined, initialize them from settings.
            self._initialize_handlers()
        return self._upload_handlers

    @upload_handlers.setter
    def upload_handlers(self, upload_handlers):
        if hasattr(self, '_files'):
            raise AttributeError("You cannot set the upload handlers after the upload has been processed.")
        self._upload_handlers = upload_handlers

    def parse_file_upload(self, META, post_data):
        """Return a tuple of (POST QueryDict, FILES MultiValueDict)."""
        self.upload_handlers = ImmutableList(
            self.upload_handlers,
            warning="You cannot alter upload handlers after the upload has been processed."
        )
        parser = MultiPartParser(META, post_data, self.upload_handlers, self.encoding)
        return parser.parse()
```
### 9 - django/middleware/clickjacking.py:

Start line: 1, End line: 46

```python
"""
Clickjacking Protection Middleware.

This module provides a middleware that implements protection against a
malicious site loading resources from your site in a hidden frame.
"""

from django.conf import settings
from django.utils.deprecation import MiddlewareMixin


class XFrameOptionsMiddleware(MiddlewareMixin):
    """
    Set the X-Frame-Options HTTP header in HTTP responses.

    Do not set the header if it's already set or if the response contains
    a xframe_options_exempt value set to True.

    By default, set the X-Frame-Options header to 'SAMEORIGIN', meaning the
    response can only be loaded on a frame within the same site. To prevent the
    response from being loaded in a frame in any site, set X_FRAME_OPTIONS in
    your project's Django settings to 'DENY'.
    """
    def process_response(self, request, response):
        # Don't set it if it's already in the response
        if response.get('X-Frame-Options') is not None:
            return response

        # Don't set it if they used @xframe_options_exempt
        if getattr(response, 'xframe_options_exempt', False):
            return response

        response['X-Frame-Options'] = self.get_xframe_options_value(request,
                                                                    response)
        return response

    def get_xframe_options_value(self, request, response):
        """
        Get the value to set for the X_FRAME_OPTIONS header. Use the value from
        the X_FRAME_OPTIONS setting, or 'DENY' if not set.

        This method can be overridden if needed, allowing it to vary based on
        the request or response.
        """
        return getattr(settings, 'X_FRAME_OPTIONS', 'DENY').upper()
```
### 10 - django/http/request.py:

Start line: 151, End line: 158

```python
class HttpRequest:

    def _get_full_path(self, path, force_append_slash):
        # RFC 3986 requires query string arguments to be in the ASCII range.
        # Rather than crash if this doesn't happen, we encode defensively.
        return '%s%s%s' % (
            escape_uri_path(path),
            '/' if force_append_slash and not path.endswith('/') else '',
            ('?' + iri_to_uri(self.META.get('QUERY_STRING', ''))) if self.META.get('QUERY_STRING', '') else ''
        )
```
### 12 - django/http/request.py:

Start line: 42, End line: 96

```python
class HttpRequest:
    """A basic HTTP request."""

    # The encoding used in GET/POST dicts. None means use default setting.
    _encoding = None
    _upload_handlers = []

    def __init__(self):
        # WARNING: The `WSGIRequest` subclass doesn't call `super`.
        # Any variable assignment made here should also happen in
        # `WSGIRequest.__init__()`.

        self.GET = QueryDict(mutable=True)
        self.POST = QueryDict(mutable=True)
        self.COOKIES = {}
        self.META = {}
        self.FILES = MultiValueDict()

        self.path = ''
        self.path_info = ''
        self.method = None
        self.resolver_match = None
        self.content_type = None
        self.content_params = None

    def __repr__(self):
        if self.method is None or not self.get_full_path():
            return '<%s>' % self.__class__.__name__
        return '<%s: %s %r>' % (self.__class__.__name__, self.method, self.get_full_path())

    @cached_property
    def headers(self):
        return HttpHeaders(self.META)

    @cached_property
    def accepted_types(self):
        """Return a list of MediaType instances."""
        return parse_accept_header(self.headers.get('Accept', '*/*'))

    def accepts(self, media_type):
        return any(
            accepted_type.match(media_type)
            for accepted_type in self.accepted_types
        )

    def _set_content_type_params(self, meta):
        """Set content_type, content_params, and encoding."""
        self.content_type, self.content_params = cgi.parse_header(meta.get('CONTENT_TYPE', ''))
        if 'charset' in self.content_params:
            try:
                codecs.lookup(self.content_params['charset'])
            except LookupError:
                pass
            else:
                self.encoding = self.content_params['charset']
```
### 15 - django/http/request.py:

Start line: 98, End line: 115

```python
class HttpRequest:

    def _get_raw_host(self):
        """
        Return the HTTP host using the environment or request headers. Skip
        allowed hosts protection, so may return an insecure host.
        """
        # We try three options, in order of decreasing preference.
        if settings.USE_X_FORWARDED_HOST and (
                'HTTP_X_FORWARDED_HOST' in self.META):
            host = self.META['HTTP_X_FORWARDED_HOST']
        elif 'HTTP_HOST' in self.META:
            host = self.META['HTTP_HOST']
        else:
            # Reconstruct the host using the algorithm from PEP 333.
            host = self.META['SERVER_NAME']
            server_port = self.get_port()
            if server_port != ('443' if self.is_secure() else '80'):
                host = '%s:%s' % (host, server_port)
        return host
```
### 19 - django/http/request.py:

Start line: 137, End line: 149

```python
class HttpRequest:

    def get_port(self):
        """Return the port number for the request as a string."""
        if settings.USE_X_FORWARDED_PORT and 'HTTP_X_FORWARDED_PORT' in self.META:
            port = self.META['HTTP_X_FORWARDED_PORT']
        else:
            port = self.META['SERVER_PORT']
        return str(port)

    def get_full_path(self, force_append_slash=False):
        return self._get_full_path(self.path, force_append_slash)

    def get_full_path_info(self, force_append_slash=False):
        return self._get_full_path(self.path_info, force_append_slash)
```
### 20 - django/http/request.py:

Start line: 160, End line: 192

```python
class HttpRequest:

    def get_signed_cookie(self, key, default=RAISE_ERROR, salt='', max_age=None):
        """
        Attempt to return a signed cookie. If the signature fails or the
        cookie has expired, raise an exception, unless the `default` argument
        is provided,  in which case return that value.
        """
        try:
            cookie_value = self.COOKIES[key]
        except KeyError:
            if default is not RAISE_ERROR:
                return default
            else:
                raise
        try:
            value = signing.get_cookie_signer(salt=key + salt).unsign(
                cookie_value, max_age=max_age)
        except signing.BadSignature:
            if default is not RAISE_ERROR:
                return default
            else:
                raise
        return value

    def get_raw_uri(self):
        """
        Return an absolute URI from variables available in this request. Skip
        allowed hosts protection, so may return insecure URI.
        """
        return '{scheme}://{host}{path}'.format(
            scheme=self.scheme,
            host=self._get_raw_host(),
            path=self.get_full_path(),
        )
```
### 21 - django/http/request.py:

Start line: 1, End line: 39

```python
import cgi
import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import quote, urlencode, urljoin, urlsplit

from django.conf import settings
from django.core import signing
from django.core.exceptions import (
    DisallowedHost, ImproperlyConfigured, RequestDataTooBig,
)
from django.core.files import uploadhandler
from django.http.multipartparser import MultiPartParser, MultiPartParserError
from django.utils.datastructures import (
    CaseInsensitiveMapping, ImmutableList, MultiValueDict,
)
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, limited_parse_qsl
from django.utils.regex_helper import _lazy_re_compile

from .multipartparser import parse_header

RAISE_ERROR = object()
host_validation_re = _lazy_re_compile(r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:\d+)?$")


class UnreadablePostError(OSError):
    pass


class RawPostDataException(Exception):
    """
    You cannot access raw_post_data from a request that has
    multipart/* POST data if it has been accessed via POST,
    FILES, etc..
    """
    pass
```
### 22 - django/http/request.py:

Start line: 573, End line: 604

```python
class MediaType:
    def __init__(self, media_type_raw_line):
        full_type, self.params = parse_header(
            media_type_raw_line.encode('ascii') if media_type_raw_line else b''
        )
        self.main_type, _, self.sub_type = full_type.partition('/')

    def __str__(self):
        params_str = ''.join(
            '; %s=%s' % (k, v.decode('ascii'))
            for k, v in self.params.items()
        )
        return '%s%s%s' % (
            self.main_type,
            ('/%s' % self.sub_type) if self.sub_type else '',
            params_str,
        )

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__qualname__, self)

    @property
    def is_all_types(self):
        return self.main_type == '*' and self.sub_type == '*'

    def match(self, other):
        if self.is_all_types:
            return True
        other = MediaType(other)
        if self.main_type == other.main_type and self.sub_type in {'*', other.sub_type}:
            return True
        return False
```
### 32 - django/http/request.py:

Start line: 326, End line: 353

```python
class HttpRequest:

    def _load_post_and_files(self):
        """Populate self._post and self._files if the content-type is a form type"""
        if self.method != 'POST':
            self._post, self._files = QueryDict(encoding=self._encoding), MultiValueDict()
            return
        if self._read_started and not hasattr(self, '_body'):
            self._mark_post_parse_error()
            return

        if self.content_type == 'multipart/form-data':
            if hasattr(self, '_body'):
                # Use already read data
                data = BytesIO(self._body)
            else:
                data = self
            try:
                self._post, self._files = self.parse_file_upload(self.META, data)
            except MultiPartParserError:
                # An error occurred while parsing POST data. Since when
                # formatting the error the request handler might access
                # self.POST, set self._post and self._file to prevent
                # attempts to parse POST data again.
                self._mark_post_parse_error()
                raise
        elif self.content_type == 'application/x-www-form-urlencoded':
            self._post, self._files = QueryDict(self.body, encoding=self._encoding), MultiValueDict()
        else:
            self._post, self._files = QueryDict(encoding=self._encoding), MultiValueDict()
```
### 35 - django/http/request.py:

Start line: 117, End line: 135

```python
class HttpRequest:

    def get_host(self):
        """Return the HTTP host using the environment or request headers."""
        host = self._get_raw_host()

        # Allow variants of localhost if ALLOWED_HOSTS is empty and DEBUG=True.
        allowed_hosts = settings.ALLOWED_HOSTS
        if settings.DEBUG and not allowed_hosts:
            allowed_hosts = ['.localhost', '127.0.0.1', '[::1]']

        domain, port = split_domain_port(host)
        if domain and validate_host(domain, allowed_hosts):
            return host
        else:
            msg = "Invalid HTTP_HOST header: %r." % host
            if domain:
                msg += " You may need to add %r to ALLOWED_HOSTS." % domain
            else:
                msg += " The domain name provided is not valid according to RFC 1034/1035."
            raise DisallowedHost(msg)
```
### 56 - django/views/i18n.py:

Start line: 88, End line: 191

```python
js_catalog_template = r"""
{% autoescape off %}
(function(globals) {

  var django = globals.django || (globals.django = {});

  {% if plural %}
  django.pluralidx = function(n) {
    var v={{ plural }};
    if (typeof(v) == 'boolean') {
      return v ? 1 : 0;
    } else {
      return v;
    }
  };
  {% else %}
  django.pluralidx = function(count) { return (count == 1) ? 0 : 1; };
  {% endif %}

  /* gettext library */

  django.catalog = django.catalog || {};
  {% if catalog_str %}
  var newcatalog = {{ catalog_str }};
  for (var key in newcatalog) {
    django.catalog[key] = newcatalog[key];
  }
  {% endif %}

  if (!django.jsi18n_initialized) {
    django.gettext = function(msgid) {
      var value = django.catalog[msgid];
      if (typeof(value) == 'undefined') {
        return msgid;
      } else {
        return (typeof(value) == 'string') ? value : value[0];
      }
    };

    django.ngettext = function(singular, plural, count) {
      var value = django.catalog[singular];
      if (typeof(value) == 'undefined') {
        return (count == 1) ? singular : plural;
      } else {
        return value.constructor === Array ? value[django.pluralidx(count)] : value;
      }
    };

    django.gettext_noop = function(msgid) { return msgid; };

    django.pgettext = function(context, msgid) {
      var value = django.gettext(context + '\x04' + msgid);
      if (value.indexOf('\x04') != -1) {
        value = msgid;
      }
      return value;
    };

    django.npgettext = function(context, singular, plural, count) {
      var value = django.ngettext(context + '\x04' + singular, context + '\x04' + plural, count);
      if (value.indexOf('\x04') != -1) {
        value = django.ngettext(singular, plural, count);
      }
      return value;
    };

    django.interpolate = function(fmt, obj, named) {
      if (named) {
        return fmt.replace(/%\(\w+\)s/g, function(match){return String(obj[match.slice(2,-2)])});
      } else {
        return fmt.replace(/%s/g, function(match){return String(obj.shift())});
      }
    };


    /* formatting library */

    django.formats = {{ formats_str }};

    django.get_format = function(format_type) {
      var value = django.formats[format_type];
      if (typeof(value) == 'undefined') {
        return format_type;
      } else {
        return value;
      }
    };

    /* add to global namespace */
    globals.pluralidx = django.pluralidx;
    globals.gettext = django.gettext;
    globals.ngettext = django.ngettext;
    globals.gettext_noop = django.gettext_noop;
    globals.pgettext = django.pgettext;
    globals.npgettext = django.npgettext;
    globals.interpolate = django.interpolate;
    globals.get_format = django.get_format;

    django.jsi18n_initialized = true;
  }

}(this));
{% endautoescape %}
"""
```
### 60 - django/views/debug.py:

Start line: 171, End line: 183

```python
class SafeExceptionReporterFilter:

    def cleanse_special_types(self, request, value):
        try:
            # If value is lazy or a complex object of another kind, this check
            # might raise an exception. isinstance checks that lazy
            # MultiValueDicts will have a return value.
            is_multivalue_dict = isinstance(value, MultiValueDict)
        except Exception as e:
            return '{!r} while evaluating {!r}'.format(e, value)

        if is_multivalue_dict:
            # Cleanse MultiValueDicts (request.POST is the one we usually care about)
            value = self.get_cleansed_multivaluedict(request, value)
        return value
```
### 69 - django/views/debug.py:

Start line: 146, End line: 169

```python
class SafeExceptionReporterFilter:

    def get_post_parameters(self, request):
        """
        Replace the values of POST parameters marked as sensitive with
        stars (*********).
        """
        if request is None:
            return {}
        else:
            sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
            if self.is_active(request) and sensitive_post_parameters:
                cleansed = request.POST.copy()
                if sensitive_post_parameters == '__ALL__':
                    # Cleanse all parameters.
                    for k in cleansed:
                        cleansed[k] = self.cleansed_substitute
                    return cleansed
                else:
                    # Cleanse only the specified parameters.
                    for param in sensitive_post_parameters:
                        if param in cleansed:
                            cleansed[param] = self.cleansed_substitute
                    return cleansed
            else:
                return request.POST
```
### 78 - django/views/i18n.py:

Start line: 1, End line: 20

```python
import itertools
import json
import os
import re
from urllib.parse import unquote

from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import (
    LANGUAGE_SESSION_KEY, check_for_language, get_language,
)
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View

LANGUAGE_QUERY_PARAMETER = 'language'
```
### 102 - django/http/request.py:

Start line: 646, End line: 666

```python
def validate_host(host, allowed_hosts):
    """
    Validate the given host for this site.

    Check that the host looks valid and matches a host or host pattern in the
    given list of ``allowed_hosts``. Any pattern beginning with a period
    matches a domain and all its subdomains (e.g. ``.example.com`` matches
    ``example.com`` and any subdomain), ``*`` matches anything, and anything
    else must match exactly.

    Note: This function assumes that the given host is lowercased and has
    already had the port, if any, stripped off.

    Return ``True`` for a valid host, ``False`` otherwise.
    """
    return any(pattern == '*' or is_same_domain(host, pattern) for pattern in allowed_hosts)


def parse_accept_header(header):
    return [MediaType(token) for token in header.split(',') if token.strip()]
```
### 105 - django/http/request.py:

Start line: 607, End line: 621

```python
# It's neither necessary nor appropriate to use
# django.utils.encoding.force_str() for parsing URLs and form inputs. Thus,
# this slightly more restricted function, used by QueryDict.
def bytes_to_text(s, encoding):
    """
    Convert bytes objects to strings, using the given encoding. Illegally
    encoded input characters are replaced with Unicode "unknown" codepoint
    (\ufffd).

    Return any non-bytes objects without change.
    """
    if isinstance(s, bytes):
        return str(s, encoding, 'replace')
    else:
        return s
```
### 123 - django/http/request.py:

Start line: 194, End line: 228

```python
class HttpRequest:

    def build_absolute_uri(self, location=None):
        """
        Build an absolute URI from the location and the variables available in
        this request. If no ``location`` is specified, build the absolute URI
        using request.get_full_path(). If the location is absolute, convert it
        to an RFC 3987 compliant URI and return it. If location is relative or
        is scheme-relative (i.e., ``//example.com/``), urljoin() it to a base
        URL constructed from the request variables.
        """
        if location is None:
            # Make it an absolute url (but schemeless and domainless) for the
            # edge case that the path starts with '//'.
            location = '//%s' % self.get_full_path()
        else:
            # Coerce lazy locations.
            location = str(location)
        bits = urlsplit(location)
        if not (bits.scheme and bits.netloc):
            # Handle the simple, most common case. If the location is absolute
            # and a scheme or host (netloc) isn't provided, skip an expensive
            # urljoin() as long as no path segments are '.' or '..'.
            if (bits.path.startswith('/') and not bits.scheme and not bits.netloc and
                    '/./' not in bits.path and '/../' not in bits.path):
                # If location starts with '//' but has no netloc, reuse the
                # schema and netloc from the current request. Strip the double
                # slashes and continue as if it wasn't specified.
                if location.startswith('//'):
                    location = location[2:]
                location = self._current_scheme_host + location
            else:
                # Join the constructed URL with the provided location, which
                # allows the provided location to apply query strings to the
                # base path.
                location = urljoin(self._current_scheme_host + self.path, location)
        return iri_to_uri(location)
```
### 160 - django/views/debug.py:

Start line: 75, End line: 102

```python
class SafeExceptionReporterFilter:
    """
    Use annotations made by the sensitive_post_parameters and
    sensitive_variables decorators to filter out sensitive information.
    """
    cleansed_substitute = '********************'
    hidden_settings = _lazy_re_compile('API|TOKEN|KEY|SECRET|PASS|SIGNATURE', flags=re.I)

    def cleanse_setting(self, key, value):
        """
        Cleanse an individual setting key/value of sensitive content. If the
        value is a dictionary, recursively cleanse the keys in that dictionary.
        """
        try:
            if self.hidden_settings.search(key):
                cleansed = self.cleansed_substitute
            elif isinstance(value, dict):
                cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
            else:
                cleansed = value
        except TypeError:
            # If the key isn't regex-able, just return as-is.
            cleansed = value

        if callable(cleansed):
            cleansed = CallableSettingWrapper(cleansed)

        return cleansed
```
