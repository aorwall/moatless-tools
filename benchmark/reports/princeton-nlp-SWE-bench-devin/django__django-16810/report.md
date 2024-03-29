# django__django-16810

| **django/django** | `191f6a9a4586b5e5f79f4f42f190e7ad4bbacc84` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 1142 |
| **Any found context length** | 203 |
| **Avg pos** | 7.8 |
| **Min pos** | 1 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/middleware/locale.py b/django/middleware/locale.py
--- a/django/middleware/locale.py
+++ b/django/middleware/locale.py
@@ -16,37 +16,28 @@ class LocaleMiddleware(MiddlewareMixin):
 
     response_redirect_class = HttpResponseRedirect
 
-    def get_fallback_language(self, request):
-        """
-        Return the fallback language for the current request based on the
-        settings. If LANGUAGE_CODE is a variant not included in the supported
-        languages, get_fallback_language() will try to fallback to a supported
-        generic variant.
-
-        Can be overridden to have a fallback language depending on the request,
-        e.g. based on top level domain.
-        """
-        try:
-            return translation.get_supported_language_variant(settings.LANGUAGE_CODE)
-        except LookupError:
-            return settings.LANGUAGE_CODE
-
     def process_request(self, request):
         urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
-        i18n_patterns_used, _ = is_language_prefix_patterns_used(urlconf)
+        (
+            i18n_patterns_used,
+            prefixed_default_language,
+        ) = is_language_prefix_patterns_used(urlconf)
         language = translation.get_language_from_request(
             request, check_path=i18n_patterns_used
         )
-        if not language:
-            language = self.get_fallback_language(request)
-
+        language_from_path = translation.get_language_from_path(request.path_info)
+        if (
+            not language_from_path
+            and i18n_patterns_used
+            and not prefixed_default_language
+        ):
+            language = settings.LANGUAGE_CODE
         translation.activate(language)
         request.LANGUAGE_CODE = translation.get_language()
 
     def process_response(self, request, response):
         language = translation.get_language()
         language_from_path = translation.get_language_from_path(request.path_info)
-        language_from_request = translation.get_language_from_request(request)
         urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
         (
             i18n_patterns_used,
@@ -57,7 +48,7 @@ def process_response(self, request, response):
             response.status_code == 404
             and not language_from_path
             and i18n_patterns_used
-            and (prefixed_default_language or language_from_request)
+            and prefixed_default_language
         ):
             # Maybe the language code is missing in the URL? Try adding the
             # language prefix and redirecting to that URL.
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -23,7 +23,7 @@
 from django.utils.functional import cached_property
 from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
 from django.utils.regex_helper import _lazy_re_compile, normalize
-from django.utils.translation import get_language, get_supported_language_variant
+from django.utils.translation import get_language
 
 from .converters import get_converter
 from .exceptions import NoReverseMatch, Resolver404
@@ -351,8 +351,7 @@ def regex(self):
     @property
     def language_prefix(self):
         language_code = get_language() or settings.LANGUAGE_CODE
-        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
-        if language_code == default_language and not self.prefix_default_language:
+        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
             return ""
         else:
             return "%s/" % language_code
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -17,7 +17,6 @@
     "get_language_from_request",
     "get_language_info",
     "get_language_bidi",
-    "get_supported_language_variant",
     "check_for_language",
     "to_language",
     "to_locale",
diff --git a/django/utils/translation/trans_null.py b/django/utils/translation/trans_null.py
--- a/django/utils/translation/trans_null.py
+++ b/django/utils/translation/trans_null.py
@@ -53,7 +53,7 @@ def check_for_language(x):
 
 
 def get_language_from_request(request, check_path=False):
-    return None
+    return settings.LANGUAGE_CODE
 
 
 def get_language_from_path(request):
diff --git a/django/utils/translation/trans_real.py b/django/utils/translation/trans_real.py
--- a/django/utils/translation/trans_real.py
+++ b/django/utils/translation/trans_real.py
@@ -583,7 +583,11 @@ def get_language_from_request(request, check_path=False):
             return get_supported_language_variant(accept_lang)
         except LookupError:
             continue
-    return None
+
+    try:
+        return get_supported_language_variant(settings.LANGUAGE_CODE)
+    except LookupError:
+        return settings.LANGUAGE_CODE
 
 
 @functools.lru_cache(maxsize=1000)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/middleware/locale.py | 19 | 49 | - | 2 | -
| django/middleware/locale.py | 60 | 60 | 2 | 2 | 617
| django/urls/resolvers.py | 26 | 26 | - | 1 | -
| django/urls/resolvers.py | 354 | 355 | 1 | 1 | 203
| django/utils/translation/__init__.py | 20 | 20 | 22 | 7 | 5871
| django/utils/translation/trans_null.py | 56 | 56 | 10 | 8 | 2837
| django/utils/translation/trans_real.py | 586 | 586 | 4 | 4 | 1142


## Problem Statement

```
Translatable URL patterns raise 404 for non-English default language when prefix_default_language=False is used.
Description
	
A simple django project with instruction to replicate the bug can be found here:
​github repo
In brief: prefix_default_language = False raises HTTP 404 for the default unprefixed pages if LANGUAGE_CODE is not "en".
I think the problem is that the function get_language_from_path in django/utils/translation/trans_real.py returns None in case of failure instead of LANGUAGE_CODE: ​diff in 4.2
Consequently, other mechanisms are used to get the language (cookies or headers) that do not work neither.
Related issue with my last comment adding some extra context: https://code.djangoproject.com/ticket/34455
It is the first time I contribute to django, I hope the bug report is OK. I am also willing to write the patch and test if required.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/urls/resolvers.py** | 341 | 373| 203 | 203 | 6080 | 
| **-> 2 <-** | **2 django/middleware/locale.py** | 46 | 90| 414 | 617 | 6817 | 
| 3 | 3 django/conf/urls/i18n.py | 1 | 40| 251 | 868 | 7068 | 
| **-> 4 <-** | **4 django/utils/translation/trans_real.py** | 546 | 586| 274 | 1142 | 11805 | 
| 5 | 5 django/views/defaults.py | 1 | 26| 151 | 1293 | 12795 | 
| 6 | 6 django/urls/base.py | 160 | 188| 201 | 1494 | 13991 | 
| 7 | **6 django/utils/translation/trans_real.py** | 286 | 365| 458 | 1952 | 13991 | 
| 8 | **7 django/utils/translation/__init__.py** | 269 | 303| 236 | 2188 | 15874 | 
| 9 | 7 django/views/defaults.py | 29 | 79| 377 | 2565 | 15874 | 
| **-> 10 <-** | **8 django/utils/translation/trans_null.py** | 1 | 68| 272 | 2837 | 16146 | 
| 11 | **8 django/utils/translation/trans_real.py** | 234 | 249| 158 | 2995 | 16146 | 
| 12 | 8 django/views/defaults.py | 102 | 121| 144 | 3139 | 16146 | 
| 13 | 9 django/core/checks/translation.py | 1 | 67| 449 | 3588 | 16595 | 
| 14 | **9 django/utils/translation/trans_real.py** | 462 | 488| 248 | 3836 | 16595 | 
| 15 | **9 django/utils/translation/__init__.py** | 63 | 82| 138 | 3974 | 16595 | 
| 16 | 10 django/views/debug.py | 76 | 103| 178 | 4152 | 21651 | 
| 17 | **10 django/middleware/locale.py** | 1 | 44| 329 | 4481 | 21651 | 
| 18 | **10 django/utils/translation/__init__.py** | 231 | 266| 271 | 4752 | 21651 | 
| 19 | **10 django/utils/translation/trans_real.py** | 1 | 68| 571 | 5323 | 21651 | 
| 20 | 10 django/urls/base.py | 1 | 24| 170 | 5493 | 21651 | 
| 21 | **10 django/utils/translation/trans_real.py** | 529 | 543| 116 | 5609 | 21651 | 
| **-> 22 <-** | **10 django/utils/translation/__init__.py** | 1 | 44| 262 | 5871 | 21651 | 
| 23 | 11 django/views/i18n.py | 30 | 74| 382 | 6253 | 23520 | 
| 24 | 12 django/middleware/common.py | 153 | 179| 255 | 6508 | 25066 | 
| 25 | **12 django/utils/translation/__init__.py** | 167 | 228| 345 | 6853 | 25066 | 
| 26 | **12 django/urls/resolvers.py** | 740 | 828| 722 | 7575 | 25066 | 
| 27 | 13 django/conf/global_settings.py | 153 | 263| 832 | 8407 | 30904 | 
| 28 | 13 django/views/defaults.py | 124 | 150| 197 | 8604 | 30904 | 
| 29 | 14 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 8727 | 36732 | 
| 30 | 15 django/urls/__init__.py | 1 | 54| 269 | 8996 | 37001 | 
| 31 | 15 django/conf/global_settings.py | 597 | 668| 453 | 9449 | 37001 | 
| 32 | 16 django/contrib/gis/views.py | 1 | 23| 160 | 9609 | 37161 | 
| 33 | **16 django/utils/translation/trans_real.py** | 265 | 283| 140 | 9749 | 37161 | 
| 34 | 17 django/views/csrf.py | 30 | 88| 587 | 10336 | 37961 | 
| 35 | 17 django/middleware/common.py | 100 | 115| 165 | 10501 | 37961 | 
| 36 | 18 django/core/handlers/exception.py | 63 | 158| 605 | 11106 | 39091 | 
| 37 | 19 django/core/checks/urls.py | 76 | 118| 266 | 11372 | 39797 | 
| 38 | 19 django/views/debug.py | 590 | 648| 454 | 11826 | 39797 | 
| 39 | 19 django/middleware/common.py | 118 | 151| 284 | 12110 | 39797 | 
| 40 | 20 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 12257 | 39945 | 
| 41 | **20 django/urls/resolvers.py** | 656 | 702| 384 | 12641 | 39945 | 
| 42 | 20 django/middleware/common.py | 34 | 60| 265 | 12906 | 39945 | 
| 43 | **20 django/urls/resolvers.py** | 150 | 182| 235 | 13141 | 39945 | 
| 44 | 21 django/contrib/syndication/views.py | 1 | 26| 223 | 13364 | 41803 | 
| 45 | 21 django/urls/base.py | 91 | 157| 383 | 13747 | 41803 | 
| 46 | **21 django/urls/resolvers.py** | 704 | 725| 187 | 13934 | 41803 | 
| 47 | **21 django/utils/translation/trans_real.py** | 491 | 526| 342 | 14276 | 41803 | 
| 48 | 22 django/conf/locale/__init__.py | 1 | 624| 75 | 14351 | 46166 | 
| 49 | **22 django/utils/translation/trans_real.py** | 614 | 636| 243 | 14594 | 46166 | 
| 50 | 23 django/contrib/redirects/middleware.py | 1 | 51| 354 | 14948 | 46521 | 
| 51 | **23 django/utils/translation/__init__.py** | 85 | 112| 136 | 15084 | 46521 | 
| 52 | **23 django/utils/translation/trans_real.py** | 447 | 459| 112 | 15196 | 46521 | 
| 53 | **23 django/urls/resolvers.py** | 296 | 318| 173 | 15369 | 46521 | 
| 54 | 23 django/views/defaults.py | 82 | 99| 121 | 15490 | 46521 | 
| 55 | 24 django/template/defaultfilters.py | 368 | 455| 504 | 15994 | 53087 | 
| 56 | 25 django/core/checks/security/base.py | 1 | 79| 691 | 16685 | 55276 | 
| 57 | 26 django/urls/conf.py | 61 | 96| 266 | 16951 | 55993 | 
| 58 | **26 django/urls/resolvers.py** | 500 | 529| 292 | 17243 | 55993 | 
| 59 | **26 django/utils/translation/trans_real.py** | 394 | 444| 358 | 17601 | 55993 | 
| 60 | 26 django/core/handlers/exception.py | 1 | 21| 118 | 17719 | 55993 | 
| 61 | 27 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 24| 117 | 17836 | 56110 | 
| 62 | 27 django/middleware/common.py | 76 | 98| 228 | 18064 | 56110 | 
| 63 | 27 django/views/i18n.py | 114 | 124| 137 | 18201 | 56110 | 
| 64 | 28 django/template/backends/dummy.py | 1 | 52| 327 | 18528 | 56437 | 
| 65 | 28 django/middleware/common.py | 62 | 74| 136 | 18664 | 56437 | 
| 66 | 29 django/contrib/admindocs/utils.py | 127 | 168| 306 | 18970 | 58422 | 
| 67 | 29 django/middleware/common.py | 1 | 32| 247 | 19217 | 58422 | 
| 68 | **29 django/urls/resolvers.py** | 479 | 498| 167 | 19384 | 58422 | 
| 69 | **29 django/urls/resolvers.py** | 320 | 338| 163 | 19547 | 58422 | 
| 70 | 30 django/http/request.py | 166 | 175| 113 | 19660 | 63873 | 
| 71 | 31 django/http/response.py | 675 | 700| 159 | 19819 | 69178 | 
| 72 | **31 django/utils/translation/trans_real.py** | 589 | 611| 191 | 20010 | 69178 | 
| 73 | 32 django/core/management/commands/makemessages.py | 47 | 68| 143 | 20153 | 75199 | 
| 74 | **32 django/urls/resolvers.py** | 727 | 738| 120 | 20273 | 75199 | 
| 75 | **32 django/urls/resolvers.py** | 376 | 439| 404 | 20677 | 75199 | 
| 76 | 32 django/views/debug.py | 512 | 587| 626 | 21303 | 75199 | 
| 77 | **32 django/urls/resolvers.py** | 531 | 613| 575 | 21878 | 75199 | 
| 78 | 33 docs/conf.py | 54 | 131| 711 | 22589 | 78770 | 
| 79 | **33 django/urls/resolvers.py** | 457 | 477| 177 | 22766 | 78770 | 
| 80 | **33 django/utils/translation/trans_real.py** | 368 | 391| 179 | 22945 | 78770 | 
| 81 | 33 django/http/response.py | 82 | 99| 133 | 23078 | 78770 | 
| 82 | 34 django/core/servers/basehttp.py | 176 | 212| 279 | 23357 | 80911 | 
| 83 | **34 django/urls/resolvers.py** | 185 | 209| 203 | 23560 | 80911 | 
| 84 | 34 django/views/debug.py | 1 | 57| 357 | 23917 | 80911 | 
| 85 | 35 django/urls/exceptions.py | 1 | 10| 0 | 23917 | 80936 | 
| 86 | 36 django/contrib/auth/views.py | 35 | 62| 218 | 24135 | 83650 | 
| 87 | 36 django/http/response.py | 613 | 636| 195 | 24330 | 83650 | 
| 88 | 37 django/utils/http.py | 303 | 324| 179 | 24509 | 86852 | 
| 89 | 37 django/views/debug.py | 406 | 435| 247 | 24756 | 86852 | 
| 90 | 38 django/contrib/flatpages/urls.py | 1 | 7| 0 | 24756 | 86890 | 
| 91 | **38 django/urls/resolvers.py** | 105 | 124| 158 | 24914 | 86890 | 
| 92 | 38 django/http/response.py | 639 | 672| 157 | 25071 | 86890 | 
| 93 | **38 django/urls/resolvers.py** | 127 | 147| 180 | 25251 | 86890 | 
| 94 | 38 django/utils/http.py | 1 | 39| 460 | 25711 | 86890 | 
| 95 | 38 django/core/servers/basehttp.py | 214 | 233| 170 | 25881 | 86890 | 
| 96 | 39 django/utils/html.py | 276 | 326| 395 | 26276 | 90138 | 
| 97 | 40 django/db/models/options.py | 1 | 58| 353 | 26629 | 97834 | 
| 98 | 41 django/views/generic/__init__.py | 1 | 40| 204 | 26833 | 98039 | 
| 99 | 42 django/forms/fields.py | 751 | 771| 182 | 27015 | 107831 | 
| 100 | 43 django/conf/urls/__init__.py | 1 | 10| 0 | 27015 | 107896 | 
| 101 | 43 docs/conf.py | 133 | 237| 952 | 27967 | 107896 | 
| 102 | 44 django/contrib/flatpages/views.py | 1 | 45| 399 | 28366 | 108486 | 
| 103 | 45 django/contrib/auth/decorators.py | 1 | 40| 315 | 28681 | 109078 | 
| 104 | 46 django/core/checks/templates.py | 1 | 47| 313 | 28994 | 109560 | 
| 105 | 47 django/core/management/base.py | 85 | 112| 161 | 29155 | 114420 | 
| 106 | 48 django/core/checks/compatibility/django_4_0.py | 1 | 21| 142 | 29297 | 114563 | 
| 107 | 48 django/conf/global_settings.py | 1 | 50| 367 | 29664 | 114563 | 
| 108 | 49 django/http/__init__.py | 1 | 53| 241 | 29905 | 114804 | 
| 109 | 50 django/contrib/auth/password_validation.py | 217 | 267| 386 | 30291 | 116698 | 
| 110 | 51 django/views/generic/base.py | 256 | 286| 246 | 30537 | 118604 | 
| 111 | 52 django/__init__.py | 1 | 25| 173 | 30710 | 118777 | 
| 112 | 52 django/views/i18n.py | 1 | 27| 171 | 30881 | 118777 | 
| 113 | 53 scripts/manage_translations.py | 1 | 29| 195 | 31076 | 120475 | 
| 114 | 54 django/contrib/messages/__init__.py | 1 | 3| 0 | 31076 | 120499 | 
| 115 | **54 django/utils/translation/__init__.py** | 45 | 61| 154 | 31230 | 120499 | 
| 116 | 54 django/views/debug.py | 473 | 510| 301 | 31531 | 120499 | 
| 117 | 55 django/core/validators.py | 68 | 111| 609 | 32140 | 125120 | 
| 118 | 56 django/core/checks/security/csrf.py | 1 | 42| 305 | 32445 | 125585 | 
| 119 | 56 django/template/defaultfilters.py | 458 | 467| 111 | 32556 | 125585 | 
| 120 | 57 django/contrib/auth/admin.py | 1 | 25| 195 | 32751 | 127392 | 
| 121 | 58 django/core/handlers/wsgi.py | 56 | 110| 478 | 33229 | 129079 | 
| 122 | 59 django/middleware/csrf.py | 1 | 55| 480 | 33709 | 133190 | 
| 123 | 60 django/template/defaulttags.py | 454 | 482| 225 | 33934 | 143963 | 
| 124 | 61 django/contrib/redirects/migrations/0001_initial.py | 1 | 65| 309 | 34243 | 144272 | 
| 125 | 61 django/core/validators.py | 250 | 305| 367 | 34610 | 144272 | 
| 126 | 62 docs/_ext/djangodocs.py | 383 | 402| 204 | 34814 | 147496 | 
| 127 | 62 django/http/response.py | 154 | 167| 150 | 34964 | 147496 | 
| 128 | 63 django/contrib/sitemaps/__init__.py | 59 | 221| 1145 | 36109 | 149324 | 
| 129 | **63 django/urls/resolvers.py** | 615 | 654| 276 | 36385 | 149324 | 
| 130 | 63 django/core/management/commands/makemessages.py | 1 | 44| 307 | 36692 | 149324 | 
| 131 | 64 django/core/checks/messages.py | 59 | 82| 161 | 36853 | 149904 | 
| 132 | 65 django/conf/locale/fy/formats.py | 22 | 22| 0 | 36853 | 150056 | 
| 133 | 66 django/conf/locale/es_MX/formats.py | 3 | 27| 290 | 37143 | 150362 | 
| 134 | 67 django/utils/formats.py | 100 | 141| 396 | 37539 | 152771 | 
| 135 | 67 django/http/request.py | 424 | 472| 338 | 37877 | 152771 | 
| 136 | 67 django/middleware/csrf.py | 296 | 346| 450 | 38327 | 152771 | 
| 137 | 67 django/urls/conf.py | 1 | 58| 451 | 38778 | 152771 | 
| 138 | 67 django/views/csrf.py | 1 | 27| 212 | 38990 | 152771 | 
| 139 | 68 django/db/utils.py | 1 | 50| 177 | 39167 | 154670 | 
| 140 | 69 django/contrib/auth/urls.py | 1 | 37| 253 | 39420 | 154923 | 
| 141 | 69 django/core/validators.py | 1 | 16| 119 | 39539 | 154923 | 
| 142 | 69 docs/_ext/djangodocs.py | 111 | 175| 567 | 40106 | 154923 | 
| 143 | 69 django/urls/base.py | 27 | 88| 440 | 40546 | 154923 | 
| 144 | **69 django/urls/resolvers.py** | 211 | 247| 268 | 40814 | 154923 | 
| 145 | 70 django/template/context_processors.py | 58 | 90| 143 | 40957 | 155418 | 
| 146 | 71 django/db/models/fields/__init__.py | 2567 | 2589| 167 | 41124 | 174422 | 
| 147 | 72 django/conf/locale/de_CH/formats.py | 5 | 36| 409 | 41533 | 174876 | 
| 148 | 72 django/conf/global_settings.py | 481 | 596| 797 | 42330 | 174876 | 
| 149 | 73 django/templatetags/i18n.py | 35 | 69| 222 | 42552 | 179061 | 
| 150 | 73 django/core/checks/security/base.py | 259 | 284| 211 | 42763 | 179061 | 
| 151 | 73 django/templatetags/i18n.py | 599 | 617| 121 | 42884 | 179061 | 
| 152 | 73 django/templatetags/i18n.py | 292 | 332| 246 | 43130 | 179061 | 
| 153 | 74 django/conf/locale/es_CO/formats.py | 3 | 27| 263 | 43393 | 179340 | 
| 154 | 74 django/views/debug.py | 292 | 324| 236 | 43629 | 179340 | 
| 155 | 75 django/contrib/redirects/admin.py | 1 | 11| 0 | 43629 | 179409 | 
| 156 | 75 django/core/validators.py | 113 | 159| 474 | 44103 | 179409 | 
| 157 | 76 django/conf/locale/es_PR/formats.py | 3 | 28| 253 | 44356 | 179678 | 
| 158 | 77 django/contrib/auth/middleware.py | 59 | 96| 362 | 44718 | 180757 | 
| 159 | 78 django/contrib/admin/sites.py | 1 | 33| 224 | 44942 | 185254 | 
| 160 | **78 django/utils/translation/__init__.py** | 115 | 164| 341 | 45283 | 185254 | 
| 161 | 78 django/http/request.py | 1 | 48| 286 | 45569 | 185254 | 
| 162 | 78 scripts/manage_translations.py | 200 | 220| 130 | 45699 | 185254 | 
| 163 | 79 django/conf/locale/es_NI/formats.py | 3 | 27| 271 | 45970 | 185541 | 
| 164 | 79 django/http/request.py | 240 | 265| 191 | 46161 | 185541 | 
| 165 | 79 django/views/debug.py | 223 | 235| 143 | 46304 | 185541 | 
| 166 | 80 django/template/base.py | 1 | 91| 754 | 47058 | 193794 | 
| 167 | 80 django/views/debug.py | 196 | 221| 181 | 47239 | 193794 | 
| 168 | 80 django/core/management/commands/makemessages.py | 326 | 407| 819 | 48058 | 193794 | 
| 169 | 81 django/contrib/admin/widgets.py | 456 | 466| 117 | 48175 | 197984 | 
| 170 | 82 django/shortcuts.py | 117 | 156| 281 | 48456 | 199108 | 


### Hint

```
Expected behavior: ​django 4.2 documentation LocaleMiddleware tries to determine the user’s language preference by following this algorithm: First, it looks for the language prefix in the requested URL. This is only performed when you are using the i18n_patterns function in your root URLconf. See Internationalization: in URL patterns for more information about the language prefix and how to internationalize URL patterns. Failing that, it looks for a cookie. The name of the cookie used is set by the LANGUAGE_COOKIE_NAME setting. (The default name is django_language.) Failing that, it looks at the Accept-Language HTTP header. This header is sent by your browser and tells the server which language(s) you prefer, in order by priority. Django tries each language in the header until it finds one with available translations. Failing that, it uses the global LANGUAGE_CODE setting.
Thanks for the report. The use of URL patterns marked as translatable is crucial for this bug. Regression in 94e7f471c4edef845a4fe5e3160132997b4cca81. Reproduced at c24cd6575f948661fa0ed8b27b79098610dc3ccc.
Replying to ab: Expected behavior: ​django 4.2 documentation LocaleMiddleware tries to determine the user’s language preference by following this algorithm: First, it looks for the language prefix in the requested URL. This is only performed when you are using the i18n_patterns function in your root URLconf. See Internationalization: in URL patterns for more information about the language prefix and how to internationalize URL patterns. Failing that, it looks for a cookie. The name of the cookie used is set by the LANGUAGE_COOKIE_NAME setting. (The default name is django_language.) Failing that, it looks at the Accept-Language HTTP header. This header is sent by your browser and tells the server which language(s) you prefer, in order by priority. Django tries each language in the header until it finds one with available translations. Failing that, it uses the global LANGUAGE_CODE setting. IMO it still works that way. However, in Django 4.2 get_language_from_request() returns the language from a request (en for me) which is activated and the default path about/ is no longer translated to the a-propos/. This is definitely a change from the previous behavior.
Replying to Mariusz Felisiak: IMO it still works that way. However, in Django 4.2 get_language_from_request() returns the language from a request (en for me) which is activated and the default path about/ is no longer translated to the a-propos/. This is definitely a change from the previous behavior. Thank you Mariusz for the quick reaction. I agree it still globally works that way, nevertheless, in the case I describe, when django looks for the language prefix in the requested URL and there is not language prefix, I would expect django to return "fr", not to go to the next steps of the algorithm. Because I want prefix_default_language = False to take precedence on cookies or headers. Does it make sense? I need to add that I use translate_url to ​build the links in my templates. Consequently, my URLs are translated in the template only (hence the 404). So you're right about the default path not being translated anymore.
I have a PR with what I think the issue is, but not confident ​https://github.com/django/django/pull/16797 @ab I think what you're saying makes sense
I agree it still globally works that way, nevertheless, in the case I describe, when django looks for the language prefix in the requested URL and there is not language prefix, I would expect django to return "fr", not to go to the next steps of the algorithm. Because I want prefix_default_language = False to take precedence on cookies or headers. Does it make sense? My 2¢: Is ignoring the process for determining the language the least surprising choice here though? It all depends on whether no-prefix URL should refer to a user setting or the site's default language. I mean imho navigating to a prefix-less URL I might expect it to show the language I chose 🤷‍♂️
@Sarah: yes, it is the same problem. After investigating the code, the change in behavior is linked to the fact that get_language_from_path returns None when the url is not prefixed. So, the cookie is used or the Accept-Language header sent by the browser. In my case, I think it is the HTTP header. @David: thanks for your contribution, but I do not fully agree. If prefix_default_url is True, the language is correctly detected by django based on the URL. If I set prefix_default_url to False I expect the same behavior for the default language without prefix. When I decide do use i18n_patterns at least (I have just added this tag to the ticket). When i18n_patternsis not used, I agree with you. So the problem might come from i18n_patterns not calling/handling correctly the calls to the new get_language_* functions.
@sarah: I'll test your patch because your edits might solve the problem with HTTP headers too. Thanks!
Just to keep track of the current work on this issue, there is a discussion about how django should behave here: ​https://github.com/django/django/pull/16797#issuecomment-1524958085 As suggested by Sarah, I'll post to django-developers for a wider range of opinions.
​Django forum discussion.
```

## Patch

```diff
diff --git a/django/middleware/locale.py b/django/middleware/locale.py
--- a/django/middleware/locale.py
+++ b/django/middleware/locale.py
@@ -16,37 +16,28 @@ class LocaleMiddleware(MiddlewareMixin):
 
     response_redirect_class = HttpResponseRedirect
 
-    def get_fallback_language(self, request):
-        """
-        Return the fallback language for the current request based on the
-        settings. If LANGUAGE_CODE is a variant not included in the supported
-        languages, get_fallback_language() will try to fallback to a supported
-        generic variant.
-
-        Can be overridden to have a fallback language depending on the request,
-        e.g. based on top level domain.
-        """
-        try:
-            return translation.get_supported_language_variant(settings.LANGUAGE_CODE)
-        except LookupError:
-            return settings.LANGUAGE_CODE
-
     def process_request(self, request):
         urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
-        i18n_patterns_used, _ = is_language_prefix_patterns_used(urlconf)
+        (
+            i18n_patterns_used,
+            prefixed_default_language,
+        ) = is_language_prefix_patterns_used(urlconf)
         language = translation.get_language_from_request(
             request, check_path=i18n_patterns_used
         )
-        if not language:
-            language = self.get_fallback_language(request)
-
+        language_from_path = translation.get_language_from_path(request.path_info)
+        if (
+            not language_from_path
+            and i18n_patterns_used
+            and not prefixed_default_language
+        ):
+            language = settings.LANGUAGE_CODE
         translation.activate(language)
         request.LANGUAGE_CODE = translation.get_language()
 
     def process_response(self, request, response):
         language = translation.get_language()
         language_from_path = translation.get_language_from_path(request.path_info)
-        language_from_request = translation.get_language_from_request(request)
         urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
         (
             i18n_patterns_used,
@@ -57,7 +48,7 @@ def process_response(self, request, response):
             response.status_code == 404
             and not language_from_path
             and i18n_patterns_used
-            and (prefixed_default_language or language_from_request)
+            and prefixed_default_language
         ):
             # Maybe the language code is missing in the URL? Try adding the
             # language prefix and redirecting to that URL.
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -23,7 +23,7 @@
 from django.utils.functional import cached_property
 from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
 from django.utils.regex_helper import _lazy_re_compile, normalize
-from django.utils.translation import get_language, get_supported_language_variant
+from django.utils.translation import get_language
 
 from .converters import get_converter
 from .exceptions import NoReverseMatch, Resolver404
@@ -351,8 +351,7 @@ def regex(self):
     @property
     def language_prefix(self):
         language_code = get_language() or settings.LANGUAGE_CODE
-        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
-        if language_code == default_language and not self.prefix_default_language:
+        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
             return ""
         else:
             return "%s/" % language_code
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -17,7 +17,6 @@
     "get_language_from_request",
     "get_language_info",
     "get_language_bidi",
-    "get_supported_language_variant",
     "check_for_language",
     "to_language",
     "to_locale",
diff --git a/django/utils/translation/trans_null.py b/django/utils/translation/trans_null.py
--- a/django/utils/translation/trans_null.py
+++ b/django/utils/translation/trans_null.py
@@ -53,7 +53,7 @@ def check_for_language(x):
 
 
 def get_language_from_request(request, check_path=False):
-    return None
+    return settings.LANGUAGE_CODE
 
 
 def get_language_from_path(request):
diff --git a/django/utils/translation/trans_real.py b/django/utils/translation/trans_real.py
--- a/django/utils/translation/trans_real.py
+++ b/django/utils/translation/trans_real.py
@@ -583,7 +583,11 @@ def get_language_from_request(request, check_path=False):
             return get_supported_language_variant(accept_lang)
         except LookupError:
             continue
-    return None
+
+    try:
+        return get_supported_language_variant(settings.LANGUAGE_CODE)
+    except LookupError:
+        return settings.LANGUAGE_CODE
 
 
 @functools.lru_cache(maxsize=1000)

```

## Test Patch

```diff
diff --git a/tests/i18n/patterns/tests.py b/tests/i18n/patterns/tests.py
--- a/tests/i18n/patterns/tests.py
+++ b/tests/i18n/patterns/tests.py
@@ -431,6 +431,27 @@ def test_nl_path(self):
         self.assertEqual(response.context["LANGUAGE_CODE"], "nl")
 
 
+@override_settings(ROOT_URLCONF="i18n.urls_default_unprefixed", LANGUAGE_CODE="nl")
+class URLPrefixedFalseTranslatedTests(URLTestCaseBase):
+    def test_translated_path_unprefixed_language_other_than_accepted_header(self):
+        response = self.client.get("/gebruikers/", headers={"accept-language": "en"})
+        self.assertEqual(response.status_code, 200)
+
+    def test_translated_path_unprefixed_language_other_than_cookie_language(self):
+        self.client.cookies.load({settings.LANGUAGE_COOKIE_NAME: "en"})
+        response = self.client.get("/gebruikers/")
+        self.assertEqual(response.status_code, 200)
+
+    def test_translated_path_prefixed_language_other_than_accepted_header(self):
+        response = self.client.get("/en/users/", headers={"accept-language": "nl"})
+        self.assertEqual(response.status_code, 200)
+
+    def test_translated_path_prefixed_language_other_than_cookie_language(self):
+        self.client.cookies.load({settings.LANGUAGE_COOKIE_NAME: "nl"})
+        response = self.client.get("/en/users/")
+        self.assertEqual(response.status_code, 200)
+
+
 class URLRedirectWithScriptAliasTests(URLTestCaseBase):
     """
     #21579 - LocaleMiddleware should respect the script prefix.
diff --git a/tests/i18n/tests.py b/tests/i18n/tests.py
--- a/tests/i18n/tests.py
+++ b/tests/i18n/tests.py
@@ -1926,22 +1926,8 @@ def test_other_lang_with_prefix(self):
         response = self.client.get("/fr/simple/")
         self.assertEqual(response.content, b"Oui")
 
-    def test_unprefixed_language_with_accept_language(self):
-        """'Accept-Language' is respected."""
-        response = self.client.get("/simple/", headers={"accept-language": "fr"})
-        self.assertRedirects(response, "/fr/simple/")
-
-    def test_unprefixed_language_with_cookie_language(self):
-        """A language set in the cookies is respected."""
-        self.client.cookies.load({settings.LANGUAGE_COOKIE_NAME: "fr"})
-        response = self.client.get("/simple/")
-        self.assertRedirects(response, "/fr/simple/")
-
-    def test_unprefixed_language_with_non_valid_language(self):
-        response = self.client.get("/simple/", headers={"accept-language": "fi"})
-        self.assertEqual(response.content, b"Yes")
-        self.client.cookies.load({settings.LANGUAGE_COOKIE_NAME: "fi"})
-        response = self.client.get("/simple/")
+    def test_unprefixed_language_other_than_accept_language(self):
+        response = self.client.get("/simple/", HTTP_ACCEPT_LANGUAGE="fr")
         self.assertEqual(response.content, b"Yes")
 
     def test_page_with_dash(self):
@@ -2017,7 +2003,10 @@ def test_get_language_from_request(self):
 
     def test_get_language_from_request_null(self):
         lang = trans_null.get_language_from_request(None)
-        self.assertEqual(lang, None)
+        self.assertEqual(lang, "en")
+        with override_settings(LANGUAGE_CODE="de"):
+            lang = trans_null.get_language_from_request(None)
+            self.assertEqual(lang, "de")
 
     def test_specific_language_codes(self):
         # issue 11915
diff --git a/tests/i18n/urls_default_unprefixed.py b/tests/i18n/urls_default_unprefixed.py
--- a/tests/i18n/urls_default_unprefixed.py
+++ b/tests/i18n/urls_default_unprefixed.py
@@ -7,5 +7,6 @@
     re_path(r"^(?P<arg>[\w-]+)-page", lambda request, **arg: HttpResponse(_("Yes"))),
     path("simple/", lambda r: HttpResponse(_("Yes"))),
     re_path(r"^(.+)/(.+)/$", lambda *args: HttpResponse()),
+    re_path(_(r"^users/$"), lambda *args: HttpResponse(), name="users"),
     prefix_default_language=False,
 )

```


## Code snippets

### 1 - django/urls/resolvers.py:

Start line: 341, End line: 373

```python
class LocalePrefixPattern:
    def __init__(self, prefix_default_language=True):
        self.prefix_default_language = prefix_default_language
        self.converters = {}

    @property
    def regex(self):
        # This is only used by reverse() and cached in _reverse_dict.
        return re.compile(re.escape(self.language_prefix))

    @property
    def language_prefix(self):
        language_code = get_language() or settings.LANGUAGE_CODE
        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
        if language_code == default_language and not self.prefix_default_language:
            return ""
        else:
            return "%s/" % language_code

    def match(self, path):
        language_prefix = self.language_prefix
        if path.startswith(language_prefix):
            return path.removeprefix(language_prefix), (), {}
        return None

    def check(self):
        return []

    def describe(self):
        return "'{}'".format(self)

    def __str__(self):
        return self.language_prefix
```
### 2 - django/middleware/locale.py:

Start line: 46, End line: 90

```python
class LocaleMiddleware(MiddlewareMixin):

    def process_response(self, request, response):
        language = translation.get_language()
        language_from_path = translation.get_language_from_path(request.path_info)
        language_from_request = translation.get_language_from_request(request)
        urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
        (
            i18n_patterns_used,
            prefixed_default_language,
        ) = is_language_prefix_patterns_used(urlconf)

        if (
            response.status_code == 404
            and not language_from_path
            and i18n_patterns_used
            and (prefixed_default_language or language_from_request)
        ):
            # Maybe the language code is missing in the URL? Try adding the
            # language prefix and redirecting to that URL.
            language_path = "/%s%s" % (language, request.path_info)
            path_valid = is_valid_path(language_path, urlconf)
            path_needs_slash = not path_valid and (
                settings.APPEND_SLASH
                and not language_path.endswith("/")
                and is_valid_path("%s/" % language_path, urlconf)
            )

            if path_valid or path_needs_slash:
                script_prefix = get_script_prefix()
                # Insert language after the script prefix and before the
                # rest of the URL
                language_url = request.get_full_path(
                    force_append_slash=path_needs_slash
                ).replace(script_prefix, "%s%s/" % (script_prefix, language), 1)
                # Redirect to the language-specific URL as detected by
                # get_language_from_request(). HTTP caches may cache this
                # redirect, so add the Vary header.
                redirect = self.response_redirect_class(language_url)
                patch_vary_headers(redirect, ("Accept-Language", "Cookie"))
                return redirect

        if not (i18n_patterns_used and language_from_path):
            patch_vary_headers(response, ("Accept-Language",))
        response.headers.setdefault("Content-Language", language)
        return response
```
### 3 - django/conf/urls/i18n.py:

Start line: 1, End line: 40

```python
import functools

from django.conf import settings
from django.urls import LocalePrefixPattern, URLResolver, get_resolver, path
from django.views.i18n import set_language


def i18n_patterns(*urls, prefix_default_language=True):
    """
    Add the language code prefix to every URL pattern within this function.
    This may only be used in the root URLconf, not in an included URLconf.
    """
    if not settings.USE_I18N:
        return list(urls)
    return [
        URLResolver(
            LocalePrefixPattern(prefix_default_language=prefix_default_language),
            list(urls),
        )
    ]


@functools.cache
def is_language_prefix_patterns_used(urlconf):
    """
    Return a tuple of two booleans: (
        `True` if i18n_patterns() (LocalePrefixPattern) is used in the URLconf,
        `True` if the default language should be prefixed
    )
    """
    for url_pattern in get_resolver(urlconf).url_patterns:
        if isinstance(url_pattern.pattern, LocalePrefixPattern):
            return True, url_pattern.pattern.prefix_default_language
    return False, False


urlpatterns = [
    path("setlang/", set_language, name="set_language"),
]
```
### 4 - django/utils/translation/trans_real.py:

Start line: 546, End line: 586

```python
def get_language_from_request(request, check_path=False):
    """
    Analyze the request to find what language the user wants the system to
    show. Only languages listed in settings.LANGUAGES are taken into account.
    If the user requests a sublanguage where we have a main language, we send
    out the main language.

    If check_path is True, the URL path prefix will be checked for a language
    code, otherwise this is skipped for backwards compatibility.
    """
    if check_path:
        lang_code = get_language_from_path(request.path_info)
        if lang_code is not None:
            return lang_code

    lang_code = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME)
    if (
        lang_code is not None
        and lang_code in get_languages()
        and check_for_language(lang_code)
    ):
        return lang_code

    try:
        return get_supported_language_variant(lang_code)
    except LookupError:
        pass

    accept = request.META.get("HTTP_ACCEPT_LANGUAGE", "")
    for accept_lang, unused in parse_accept_lang_header(accept):
        if accept_lang == "*":
            break

        if not language_code_re.search(accept_lang):
            continue

        try:
            return get_supported_language_variant(accept_lang)
        except LookupError:
            continue
    return None
```
### 5 - django/views/defaults.py:

Start line: 1, End line: 26

```python
from urllib.parse import quote

from django.http import (
    HttpResponseBadRequest,
    HttpResponseForbidden,
    HttpResponseNotFound,
    HttpResponseServerError,
)
from django.template import Context, Engine, TemplateDoesNotExist, loader
from django.views.decorators.csrf import requires_csrf_token

ERROR_404_TEMPLATE_NAME = "404.html"
ERROR_403_TEMPLATE_NAME = "403.html"
ERROR_400_TEMPLATE_NAME = "400.html"
ERROR_500_TEMPLATE_NAME = "500.html"
ERROR_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <title>%(title)s</title>
</head>
<body>
  <h1>%(title)s</h1><p>%(details)s</p>
</body>
</html>
"""
```
### 6 - django/urls/base.py:

Start line: 160, End line: 188

```python
def translate_url(url, lang_code):
    """
    Given a URL (absolute or relative), try to get its translated version in
    the `lang_code` language (either by i18n_patterns or by translated regex).
    Return the original URL if no translated version is found.
    """
    parsed = urlsplit(url)
    try:
        # URL may be encoded.
        match = resolve(unquote(parsed.path))
    except Resolver404:
        pass
    else:
        to_be_reversed = (
            "%s:%s" % (match.namespace, match.url_name)
            if match.namespace
            else match.url_name
        )
        with override(lang_code):
            try:
                url = reverse(to_be_reversed, args=match.args, kwargs=match.kwargs)
            except NoReverseMatch:
                pass
            else:
                url = urlunsplit(
                    (parsed.scheme, parsed.netloc, url, parsed.query, parsed.fragment)
                )
    return url
```
### 7 - django/utils/translation/trans_real.py:

Start line: 286, End line: 365

```python
def translation(language):
    """
    Return a translation object in the default 'django' domain.
    """
    global _translations
    if language not in _translations:
        _translations[language] = DjangoTranslation(language)
    return _translations[language]


def activate(language):
    """
    Fetch the translation object for a given language and install it as the
    current translation object for the current thread.
    """
    if not language:
        return
    _active.value = translation(language)


def deactivate():
    """
    Uninstall the active translation object so that further _() calls resolve
    to the default translation object.
    """
    if hasattr(_active, "value"):
        del _active.value


def deactivate_all():
    """
    Make the active translation object a NullTranslations() instance. This is
    useful when we want delayed translations to appear as the original string
    for some reason.
    """
    _active.value = gettext_module.NullTranslations()
    _active.value.to_language = lambda *args: None


def get_language():
    """Return the currently selected language."""
    t = getattr(_active, "value", None)
    if t is not None:
        try:
            return t.to_language()
        except AttributeError:
            pass
    # If we don't have a real translation object, assume it's the default language.
    return settings.LANGUAGE_CODE


def get_language_bidi():
    """
    Return selected language's BiDi layout.

    * False = left-to-right layout
    * True = right-to-left layout
    """
    lang = get_language()
    if lang is None:
        return False
    else:
        base_lang = get_language().split("-")[0]
        return base_lang in settings.LANGUAGES_BIDI


def catalog():
    """
    Return the current active catalog for further processing.
    This can be used if you need to modify the catalog or want to access the
    whole message catalog instead of just translating one string.
    """
    global _default

    t = getattr(_active, "value", None)
    if t is not None:
        return t
    if _default is None:
        _default = translation(settings.LANGUAGE_CODE)
    return _default
```
### 8 - django/utils/translation/__init__.py:

Start line: 269, End line: 303

```python
def get_language_info(lang_code):
    from django.conf.locale import LANG_INFO

    try:
        lang_info = LANG_INFO[lang_code]
        if "fallback" in lang_info and "name" not in lang_info:
            info = get_language_info(lang_info["fallback"][0])
        else:
            info = lang_info
    except KeyError:
        if "-" not in lang_code:
            raise KeyError("Unknown language code %s." % lang_code)
        generic_lang_code = lang_code.split("-")[0]
        try:
            info = LANG_INFO[generic_lang_code]
        except KeyError:
            raise KeyError(
                "Unknown language code %s and %s." % (lang_code, generic_lang_code)
            )

    if info:
        info["name_translated"] = gettext_lazy(info["name"])
    return info


trim_whitespace_re = _lazy_re_compile(r"\s*\n\s*")


def trim_whitespace(s):
    return trim_whitespace_re.sub(" ", s.strip())


def round_away_from_one(value):
    return int(Decimal(value - 1).quantize(Decimal("0"), rounding=ROUND_UP)) + 1
```
### 9 - django/views/defaults.py:

Start line: 29, End line: 79

```python
# These views can be called when CsrfViewMiddleware.process_view() not run,
# therefore need @requires_csrf_token in case the template needs
# {% csrf_token %}.


@requires_csrf_token
def page_not_found(request, exception, template_name=ERROR_404_TEMPLATE_NAME):
    """
    Default 404 handler.

    Templates: :template:`404.html`
    Context:
        request_path
            The path of the requested URL (e.g., '/app/pages/bad_page/'). It's
            quoted to prevent a content injection attack.
        exception
            The message from the exception which triggered the 404 (if one was
            supplied), or the exception class name
    """
    exception_repr = exception.__class__.__name__
    # Try to get an "interesting" exception message, if any (and not the ugly
    # Resolver404 dictionary)
    try:
        message = exception.args[0]
    except (AttributeError, IndexError):
        pass
    else:
        if isinstance(message, str):
            exception_repr = message
    context = {
        "request_path": quote(request.path),
        "exception": exception_repr,
    }
    try:
        template = loader.get_template(template_name)
        body = template.render(context, request)
    except TemplateDoesNotExist:
        if template_name != ERROR_404_TEMPLATE_NAME:
            # Reraise if it's a missing custom template.
            raise
        # Render template (even though there are no substitutions) to allow
        # inspecting the context in tests.
        template = Engine().from_string(
            ERROR_PAGE_TEMPLATE
            % {
                "title": "Not Found",
                "details": "The requested resource was not found on this server.",
            },
        )
        body = template.render(Context(context))
    return HttpResponseNotFound(body)
```
### 10 - django/utils/translation/trans_null.py:

Start line: 1, End line: 68

```python
# These are versions of the functions in django.utils.translation.trans_real
# that don't actually do anything. This is purely for performance, so that
# settings.USE_I18N = False can use this module rather than trans_real.py.

from django.conf import settings


def gettext(message):
    return message


gettext_noop = gettext_lazy = _ = gettext


def ngettext(singular, plural, number):
    if number == 1:
        return singular
    return plural


ngettext_lazy = ngettext


def pgettext(context, message):
    return gettext(message)


def npgettext(context, singular, plural, number):
    return ngettext(singular, plural, number)


def activate(x):
    return None


def deactivate():
    return None


deactivate_all = deactivate


def get_language():
    return settings.LANGUAGE_CODE


def get_language_bidi():
    return settings.LANGUAGE_CODE in settings.LANGUAGES_BIDI


def check_for_language(x):
    return True


def get_language_from_request(request, check_path=False):
    return None


def get_language_from_path(request):
    return None


def get_supported_language_variant(lang_code, strict=False):
    if lang_code and lang_code.lower() == settings.LANGUAGE_CODE.lower():
        return lang_code
    else:
        raise LookupError(lang_code)
```
### 11 - django/utils/translation/trans_real.py:

Start line: 234, End line: 249

```python
class DjangoTranslation(gettext_module.GNUTranslations):

    def _add_fallback(self, localedirs=None):
        """Set the GNUTranslations() fallback with the default language."""
        # Don't set a fallback for the default language or any English variant
        # (as it's empty, so it'll ALWAYS fall back to the default language)
        if self.__language == settings.LANGUAGE_CODE or self.__language.startswith(
            "en"
        ):
            return
        if self.domain == "django":
            # Get from cache
            default_translation = translation(settings.LANGUAGE_CODE)
        else:
            default_translation = DjangoTranslation(
                settings.LANGUAGE_CODE, domain=self.domain, localedirs=localedirs
            )
        self.add_fallback(default_translation)
```
### 14 - django/utils/translation/trans_real.py:

Start line: 462, End line: 488

```python
@functools.lru_cache(maxsize=1000)
def check_for_language(lang_code):
    """
    Check whether there is a global language file for the given language
    code. This is used to decide whether a user-provided language is
    available.

    lru_cache should have a maxsize to prevent from memory exhaustion attacks,
    as the provided language codes are taken from the HTTP request. See also
    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.
    """
    # First, a quick check to make sure lang_code is well-formed (#21458)
    if lang_code is None or not language_code_re.search(lang_code):
        return False
    return any(
        gettext_module.find("django", path, [to_locale(lang_code)]) is not None
        for path in all_locale_paths()
    )


@functools.lru_cache
def get_languages():
    """
    Cache of settings.LANGUAGES in a dictionary for easy lookups by key.
    Convert keys to lowercase as they should be treated as case-insensitive.
    """
    return {key.lower(): value for key, value in dict(settings.LANGUAGES).items()}
```
### 15 - django/utils/translation/__init__.py:

Start line: 63, End line: 82

```python
class Trans:

    def __getattr__(self, real_name):
        from django.conf import settings

        if settings.USE_I18N:
            from django.utils.translation import trans_real as trans
            from django.utils.translation.reloader import (
                translation_file_changed,
                watch_for_translation_changes,
            )

            autoreload_started.connect(
                watch_for_translation_changes, dispatch_uid="translation_file_changed"
            )
            file_changed.connect(
                translation_file_changed, dispatch_uid="translation_file_changed"
            )
        else:
            from django.utils.translation import trans_null as trans
        setattr(self, real_name, getattr(trans, real_name))
        return getattr(trans, real_name)
```
### 17 - django/middleware/locale.py:

Start line: 1, End line: 44

```python
from django.conf import settings
from django.conf.urls.i18n import is_language_prefix_patterns_used
from django.http import HttpResponseRedirect
from django.urls import get_script_prefix, is_valid_path
from django.utils import translation
from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin


class LocaleMiddleware(MiddlewareMixin):
    """
    Parse a request and decide what translation object to install in the
    current thread context. This allows pages to be dynamically translated to
    the language the user desires (if the language is available).
    """

    response_redirect_class = HttpResponseRedirect

    def get_fallback_language(self, request):
        """
        Return the fallback language for the current request based on the
        settings. If LANGUAGE_CODE is a variant not included in the supported
        languages, get_fallback_language() will try to fallback to a supported
        generic variant.

        Can be overridden to have a fallback language depending on the request,
        e.g. based on top level domain.
        """
        try:
            return translation.get_supported_language_variant(settings.LANGUAGE_CODE)
        except LookupError:
            return settings.LANGUAGE_CODE

    def process_request(self, request):
        urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF)
        i18n_patterns_used, _ = is_language_prefix_patterns_used(urlconf)
        language = translation.get_language_from_request(
            request, check_path=i18n_patterns_used
        )
        if not language:
            language = self.get_fallback_language(request)

        translation.activate(language)
        request.LANGUAGE_CODE = translation.get_language()
```
### 18 - django/utils/translation/__init__.py:

Start line: 231, End line: 266

```python
def to_locale(language):
    """Turn a language name (en-us) into a locale name (en_US)."""
    lang, _, country = language.lower().partition("-")
    if not country:
        return language[:3].lower() + language[3:]
    # A language with > 2 characters after the dash only has its first
    # character after the dash capitalized; e.g. sr-latn becomes sr_Latn.
    # A language with 2 characters after the dash has both characters
    # capitalized; e.g. en-us becomes en_US.
    country, _, tail = country.partition("-")
    country = country.title() if len(country) > 2 else country.upper()
    if tail:
        country += "-" + tail
    return lang + "_" + country


def get_language_from_request(request, check_path=False):
    return _trans.get_language_from_request(request, check_path)


def get_language_from_path(path):
    return _trans.get_language_from_path(path)


def get_supported_language_variant(lang_code, *, strict=False):
    return _trans.get_supported_language_variant(lang_code, strict)


def templatize(src, **kwargs):
    from .template import templatize

    return templatize(src, **kwargs)


def deactivate_all():
    return _trans.deactivate_all()
```
### 19 - django/utils/translation/trans_real.py:

Start line: 1, End line: 68

```python
"""Translation helper functions."""
import functools
import gettext as gettext_module
import os
import re
import sys
import warnings

from asgiref.local import Local

from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe

from . import to_language, to_locale

# Translations are cached in a dictionary for every language.
# The active translations are stored by threadid to make them thread local.
_translations = {}
_active = Local()

# The default translation is based on the settings file.
_default = None

# magic gettext number to separate context from message
CONTEXT_SEPARATOR = "\x04"

# Maximum number of characters that will be parsed from the Accept-Language
# header to prevent possible denial of service or memory exhaustion attacks.
# About 10x longer than the longest value shown on MDN’s Accept-Language page.
ACCEPT_LANGUAGE_HEADER_MAX_LENGTH = 500

# Format of Accept-Language header values. From RFC 9110 Sections 12.4.2 and
# 12.5.4, and RFC 5646 Section 2.1.
accept_language_re = _lazy_re_compile(
    r"""
        # "en", "en-au", "x-y-z", "es-419", "*"
        ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\*)
        # Optional "q=1.00", "q=0.8"
        (?:\s*;\s*q=(0(?:\.[0-9]{,3})?|1(?:\.0{,3})?))?
        # Multiple accepts per header.
        (?:\s*,\s*|$)
    """,
    re.VERBOSE,
)

language_code_re = _lazy_re_compile(
    r"^[a-z]{1,8}(?:-[a-z0-9]{1,8})*(?:@[a-z0-9]{1,20})?$", re.IGNORECASE
)

language_code_prefix_re = _lazy_re_compile(r"^/(\w+([@-]\w+){0,2})(/|$)")


@receiver(setting_changed)
def reset_cache(*, setting, **kwargs):
    """
    Reset global state when LANGUAGES setting has been changed, as some
    languages should no longer be accepted.
    """
    if setting in ("LANGUAGES", "LANGUAGE_CODE"):
        check_for_language.cache_clear()
        get_languages.cache_clear()
        get_supported_language_variant.cache_clear()
```
### 21 - django/utils/translation/trans_real.py:

Start line: 529, End line: 543

```python
def get_language_from_path(path, strict=False):
    """
    Return the language code if there's a valid language code found in `path`.

    If `strict` is False (the default), look for a country-specific variant
    when neither the language code nor its generic variant is found.
    """
    regex_match = language_code_prefix_re.match(path)
    if not regex_match:
        return None
    lang_code = regex_match[1]
    try:
        return get_supported_language_variant(lang_code, strict=strict)
    except LookupError:
        return None
```
### 22 - django/utils/translation/__init__.py:

Start line: 1, End line: 44

```python
"""
Internationalization support.
"""
from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal

from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile

__all__ = [
    "activate",
    "deactivate",
    "override",
    "deactivate_all",
    "get_language",
    "get_language_from_request",
    "get_language_info",
    "get_language_bidi",
    "get_supported_language_variant",
    "check_for_language",
    "to_language",
    "to_locale",
    "templatize",
    "gettext",
    "gettext_lazy",
    "gettext_noop",
    "ngettext",
    "ngettext_lazy",
    "pgettext",
    "pgettext_lazy",
    "npgettext",
    "npgettext_lazy",
]


class TranslatorCommentWarning(SyntaxWarning):
    pass


# Here be dragons, so a short explanation of the logic won't hurt:
# We are trying to solve two problems: (1) access settings, in particular
# settings.USE_I18N, as late as possible, so that modules can be imported
# without having to first configure Django, and (2) if some other code creates
```
### 25 - django/utils/translation/__init__.py:

Start line: 167, End line: 228

```python
def _lazy_number_unpickle(func, resultclass, number, kwargs):
    return lazy_number(func, resultclass, number=number, **kwargs)


def ngettext_lazy(singular, plural, number=None):
    return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)


def npgettext_lazy(context, singular, plural, number=None):
    return lazy_number(
        npgettext, str, context=context, singular=singular, plural=plural, number=number
    )


def activate(language):
    return _trans.activate(language)


def deactivate():
    return _trans.deactivate()


class override(ContextDecorator):
    def __init__(self, language, deactivate=False):
        self.language = language
        self.deactivate = deactivate

    def __enter__(self):
        self.old_language = get_language()
        if self.language is not None:
            activate(self.language)
        else:
            deactivate_all()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_language is None:
            deactivate_all()
        elif self.deactivate:
            deactivate()
        else:
            activate(self.old_language)


def get_language():
    return _trans.get_language()


def get_language_bidi():
    return _trans.get_language_bidi()


def check_for_language(lang_code):
    return _trans.check_for_language(lang_code)


def to_language(locale):
    """Turn a locale name (en_US) into a language name (en-us)."""
    p = locale.find("_")
    if p >= 0:
        return locale[:p].lower() + "-" + locale[p + 1 :].lower()
    else:
        return locale.lower()
```
### 26 - django/urls/resolvers.py:

Start line: 740, End line: 828

```python
class URLResolver:

    def _reverse_with_prefix(self, lookup_view, _prefix, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Don't mix *args and **kwargs in call to reverse()!")

        if not self._populated:
            self._populate()

        possibilities = self.reverse_dict.getlist(lookup_view)

        for possibility, pattern, defaults, converters in possibilities:
            for result, params in possibility:
                if args:
                    if len(args) != len(params):
                        continue
                    candidate_subs = dict(zip(params, args))
                else:
                    if set(kwargs).symmetric_difference(params).difference(defaults):
                        continue
                    matches = True
                    for k, v in defaults.items():
                        if k in params:
                            continue
                        if kwargs.get(k, v) != v:
                            matches = False
                            break
                    if not matches:
                        continue
                    candidate_subs = kwargs
                # Convert the candidate subs to text using Converter.to_url().
                text_candidate_subs = {}
                match = True
                for k, v in candidate_subs.items():
                    if k in converters:
                        try:
                            text_candidate_subs[k] = converters[k].to_url(v)
                        except ValueError:
                            match = False
                            break
                    else:
                        text_candidate_subs[k] = str(v)
                if not match:
                    continue
                # WSGI provides decoded URLs, without %xx escapes, and the URL
                # resolver operates on such URLs. First substitute arguments
                # without quoting to build a decoded URL and look for a match.
                # Then, if we have a match, redo the substitution with quoted
                # arguments in order to return a properly encoded URL.
                candidate_pat = _prefix.replace("%", "%%") + result
                if re.search(
                    "^%s%s" % (re.escape(_prefix), pattern),
                    candidate_pat % text_candidate_subs,
                ):
                    # safe characters from `pchar` definition of RFC 3986
                    url = quote(
                        candidate_pat % text_candidate_subs,
                        safe=RFC3986_SUBDELIMS + "/~:@",
                    )
                    # Don't allow construction of scheme relative urls.
                    return escape_leading_slashes(url)
        # lookup_view can be URL name or callable, but callables are not
        # friendly in error messages.
        m = getattr(lookup_view, "__module__", None)
        n = getattr(lookup_view, "__name__", None)
        if m is not None and n is not None:
            lookup_view_s = "%s.%s" % (m, n)
        else:
            lookup_view_s = lookup_view

        patterns = [pattern for (_, pattern, _, _) in possibilities]
        if patterns:
            if args:
                arg_msg = "arguments '%s'" % (args,)
            elif kwargs:
                arg_msg = "keyword arguments '%s'" % kwargs
            else:
                arg_msg = "no arguments"
            msg = "Reverse for '%s' with %s not found. %d pattern(s) tried: %s" % (
                lookup_view_s,
                arg_msg,
                len(patterns),
                patterns,
            )
        else:
            msg = (
                "Reverse for '%(view)s' not found. '%(view)s' is not "
                "a valid view function or pattern name." % {"view": lookup_view_s}
            )
        raise NoReverseMatch(msg)
```
### 33 - django/utils/translation/trans_real.py:

Start line: 265, End line: 283

```python
class DjangoTranslation(gettext_module.GNUTranslations):

    def language(self):
        """Return the translation language."""
        return self.__language

    def to_language(self):
        """Return the translation language name."""
        return self.__to_language

    def ngettext(self, msgid1, msgid2, n):
        try:
            tmsg = self._catalog.plural(msgid1, n)
        except KeyError:
            if self._fallback:
                return self._fallback.ngettext(msgid1, msgid2, n)
            if n == 1:
                tmsg = msgid1
            else:
                tmsg = msgid2
        return tmsg
```
### 41 - django/urls/resolvers.py:

Start line: 656, End line: 702

```python
class URLResolver:

    def resolve(self, path):
        path = str(path)  # path may be a reverse_lazy object
        tried = []
        match = self.pattern.match(path)
        if match:
            new_path, args, kwargs = match
            for pattern in self.url_patterns:
                try:
                    sub_match = pattern.resolve(new_path)
                except Resolver404 as e:
                    self._extend_tried(tried, pattern, e.args[0].get("tried"))
                else:
                    if sub_match:
                        # Merge captured arguments in match with submatch
                        sub_match_dict = {**kwargs, **self.default_kwargs}
                        # Update the sub_match_dict with the kwargs from the sub_match.
                        sub_match_dict.update(sub_match.kwargs)
                        # If there are *any* named groups, ignore all non-named groups.
                        # Otherwise, pass all non-named arguments as positional
                        # arguments.
                        sub_match_args = sub_match.args
                        if not sub_match_dict:
                            sub_match_args = args + sub_match.args
                        current_route = (
                            ""
                            if isinstance(pattern, URLPattern)
                            else str(pattern.pattern)
                        )
                        self._extend_tried(tried, pattern, sub_match.tried)
                        return ResolverMatch(
                            sub_match.func,
                            sub_match_args,
                            sub_match_dict,
                            sub_match.url_name,
                            [self.app_name] + sub_match.app_names,
                            [self.namespace] + sub_match.namespaces,
                            self._join_route(current_route, sub_match.route),
                            tried,
                            captured_kwargs=sub_match.captured_kwargs,
                            extra_kwargs={
                                **self.default_kwargs,
                                **sub_match.extra_kwargs,
                            },
                        )
                    tried.append([pattern])
            raise Resolver404({"tried": tried, "path": new_path})
        raise Resolver404({"path": path})
```
### 43 - django/urls/resolvers.py:

Start line: 150, End line: 182

```python
class CheckURLMixin:
    def describe(self):
        """
        Format the URL pattern for display in warning messages.
        """
        description = "'{}'".format(self)
        if self.name:
            description += " [name='{}']".format(self.name)
        return description

    def _check_pattern_startswith_slash(self):
        """
        Check that the pattern does not begin with a forward slash.
        """
        regex_pattern = self.regex.pattern
        if not settings.APPEND_SLASH:
            # Skip check as it can be useful to start a URL pattern with a slash
            # when APPEND_SLASH=False.
            return []
        if regex_pattern.startswith(("/", "^/", "^\\/")) and not regex_pattern.endswith(
            "/"
        ):
            warning = Warning(
                "Your URL pattern {} has a route beginning with a '/'. Remove this "
                "slash as it is unnecessary. If this pattern is targeted in an "
                "include(), ensure the include() pattern has a trailing '/'.".format(
                    self.describe()
                ),
                id="urls.W002",
            )
            return [warning]
        else:
            return []
```
### 46 - django/urls/resolvers.py:

Start line: 704, End line: 725

```python
class URLResolver:

    @cached_property
    def urlconf_module(self):
        if isinstance(self.urlconf_name, str):
            return import_module(self.urlconf_name)
        else:
            return self.urlconf_name

    @cached_property
    def url_patterns(self):
        # urlconf_module might be a valid set of patterns, so we default to it
        patterns = getattr(self.urlconf_module, "urlpatterns", self.urlconf_module)
        try:
            iter(patterns)
        except TypeError as e:
            msg = (
                "The included URLconf '{name}' does not appear to have "
                "any patterns in it. If you see the 'urlpatterns' variable "
                "with valid patterns in the file then the issue is probably "
                "caused by a circular import."
            )
            raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
        return patterns
```
### 47 - django/utils/translation/trans_real.py:

Start line: 491, End line: 526

```python
@functools.lru_cache(maxsize=1000)
def get_supported_language_variant(lang_code, strict=False):
    """
    Return the language code that's listed in supported languages, possibly
    selecting a more generic variant. Raise LookupError if nothing is found.

    If `strict` is False (the default), look for a country-specific variant
    when neither the language code nor its generic variant is found.

    lru_cache should have a maxsize to prevent from memory exhaustion attacks,
    as the provided language codes are taken from the HTTP request. See also
    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.
    """
    if lang_code:
        # If 'zh-hant-tw' is not supported, try special fallback or subsequent
        # language codes i.e. 'zh-hant' and 'zh'.
        possible_lang_codes = [lang_code]
        try:
            possible_lang_codes.extend(LANG_INFO[lang_code]["fallback"])
        except KeyError:
            pass
        i = None
        while (i := lang_code.rfind("-", 0, i)) > -1:
            possible_lang_codes.append(lang_code[:i])
        generic_lang_code = possible_lang_codes[-1]
        supported_lang_codes = get_languages()

        for code in possible_lang_codes:
            if code.lower() in supported_lang_codes and check_for_language(code):
                return code
        if not strict:
            # if fr-fr is not supported, try fr-ca.
            for supported_code in supported_lang_codes:
                if supported_code.startswith(generic_lang_code + "-"):
                    return supported_code
    raise LookupError(lang_code)
```
### 49 - django/utils/translation/trans_real.py:

Start line: 614, End line: 636

```python
def parse_accept_lang_header(lang_string):
    """
    Parse the value of the Accept-Language header up to a maximum length.

    The value of the header is truncated to a maximum length to avoid potential
    denial of service and memory exhaustion attacks. Excessive memory could be
    used if the raw value is very large as it would be cached due to the use of
    functools.lru_cache() to avoid repetitive parsing of common header values.
    """
    # If the header value doesn't exceed the maximum allowed length, parse it.
    if len(lang_string) <= ACCEPT_LANGUAGE_HEADER_MAX_LENGTH:
        return _parse_accept_lang_header(lang_string)

    # If there is at least one comma in the value, parse up to the last comma
    # before the max length, skipping any truncated parts at the end of the
    # header value.
    if (index := lang_string.rfind(",", 0, ACCEPT_LANGUAGE_HEADER_MAX_LENGTH)) > 0:
        return _parse_accept_lang_header(lang_string[:index])

    # Don't attempt to parse if there is only one language-range value which is
    # longer than the maximum allowed length and so truncated.
    return ()
```
### 51 - django/utils/translation/__init__.py:

Start line: 85, End line: 112

```python
_trans = Trans()

# The Trans class is no more needed, so remove it from the namespace.
del Trans


def gettext_noop(message):
    return _trans.gettext_noop(message)


def gettext(message):
    return _trans.gettext(message)


def ngettext(singular, plural, number):
    return _trans.ngettext(singular, plural, number)


def pgettext(context, message):
    return _trans.pgettext(context, message)


def npgettext(context, singular, plural, number):
    return _trans.npgettext(context, singular, plural, number)


gettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)
```
### 52 - django/utils/translation/trans_real.py:

Start line: 447, End line: 459

```python
def all_locale_paths():
    """
    Return a list of paths to user-provides languages files.
    """
    globalpath = os.path.join(
        os.path.dirname(sys.modules[settings.__module__].__file__), "locale"
    )
    app_paths = []
    for app_config in apps.get_app_configs():
        locale_path = os.path.join(app_config.path, "locale")
        if os.path.exists(locale_path):
            app_paths.append(locale_path)
    return [globalpath, *settings.LOCALE_PATHS, *app_paths]
```
### 53 - django/urls/resolvers.py:

Start line: 296, End line: 318

```python
class RoutePattern(CheckURLMixin):
    regex = LocaleRegexDescriptor("_route")

    def __init__(self, route, name=None, is_endpoint=False):
        self._route = route
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = _route_to_regex(str(route), is_endpoint)[1]

    def match(self, path):
        match = self.regex.search(path)
        if match:
            # RoutePattern doesn't allow non-named groups so args are ignored.
            kwargs = match.groupdict()
            for key, value in kwargs.items():
                converter = self.converters[key]
                try:
                    kwargs[key] = converter.to_python(value)
                except ValueError:
                    return None
            return path[match.end() :], (), kwargs
        return None
```
### 58 - django/urls/resolvers.py:

Start line: 500, End line: 529

```python
class URLResolver:

    def _check_custom_error_handlers(self):
        messages = []
        # All handlers take (request, exception) arguments except handler500
        # which takes (request).
        for status_code, num_parameters in [(400, 2), (403, 2), (404, 2), (500, 1)]:
            try:
                handler = self.resolve_error_handler(status_code)
            except (ImportError, ViewDoesNotExist) as e:
                path = getattr(self.urlconf_module, "handler%s" % status_code)
                msg = (
                    "The custom handler{status_code} view '{path}' could not be "
                    "imported."
                ).format(status_code=status_code, path=path)
                messages.append(Error(msg, hint=str(e), id="urls.E008"))
                continue
            signature = inspect.signature(handler)
            args = [None] * num_parameters
            try:
                signature.bind(*args)
            except TypeError:
                msg = (
                    "The custom handler{status_code} view '{path}' does not "
                    "take the correct number of arguments ({args})."
                ).format(
                    status_code=status_code,
                    path=handler.__module__ + "." + handler.__qualname__,
                    args="request, exception" if num_parameters == 2 else "request",
                )
                messages.append(Error(msg, id="urls.E007"))
        return messages
```
### 59 - django/utils/translation/trans_real.py:

Start line: 394, End line: 444

```python
def pgettext(context, message):
    msg_with_ctxt = "%s%s%s" % (context, CONTEXT_SEPARATOR, message)
    result = gettext(msg_with_ctxt)
    if CONTEXT_SEPARATOR in result:
        # Translation not found
        result = message
    elif isinstance(message, SafeData):
        result = mark_safe(result)
    return result


def gettext_noop(message):
    """
    Mark strings for translation but don't translate them now. This can be
    used to store strings in global variables that should stay in the base
    language (because they might be used externally) and will be translated
    later.
    """
    return message


def do_ntranslate(singular, plural, number, translation_function):
    global _default

    t = getattr(_active, "value", None)
    if t is not None:
        return getattr(t, translation_function)(singular, plural, number)
    if _default is None:
        _default = translation(settings.LANGUAGE_CODE)
    return getattr(_default, translation_function)(singular, plural, number)


def ngettext(singular, plural, number):
    """
    Return a string of the translation of either the singular or plural,
    based on the number.
    """
    return do_ntranslate(singular, plural, number, "ngettext")


def npgettext(context, singular, plural, number):
    msgs_with_ctxt = (
        "%s%s%s" % (context, CONTEXT_SEPARATOR, singular),
        "%s%s%s" % (context, CONTEXT_SEPARATOR, plural),
        number,
    )
    result = ngettext(*msgs_with_ctxt)
    if CONTEXT_SEPARATOR in result:
        # Translation not found
        result = ngettext(singular, plural, number)
    return result
```
### 68 - django/urls/resolvers.py:

Start line: 479, End line: 498

```python
class URLResolver:

    def __repr__(self):
        if isinstance(self.urlconf_name, list) and self.urlconf_name:
            # Don't bother to output the whole list, it can be huge
            urlconf_repr = "<%s list>" % self.urlconf_name[0].__class__.__name__
        else:
            urlconf_repr = repr(self.urlconf_name)
        return "<%s %s (%s:%s) %s>" % (
            self.__class__.__name__,
            urlconf_repr,
            self.app_name,
            self.namespace,
            self.pattern.describe(),
        )

    def check(self):
        messages = []
        for pattern in self.url_patterns:
            messages.extend(check_resolver(pattern))
        messages.extend(self._check_custom_error_handlers())
        return messages or self.pattern.check()
```
### 69 - django/urls/resolvers.py:

Start line: 320, End line: 338

```python
class RoutePattern(CheckURLMixin):

    def check(self):
        warnings = self._check_pattern_startswith_slash()
        route = self._route
        if "(?P<" in route or route.startswith("^") or route.endswith("$"):
            warnings.append(
                Warning(
                    "Your URL pattern {} has a route that contains '(?P<', begins "
                    "with a '^', or ends with a '$'. This was likely an oversight "
                    "when migrating to django.urls.path().".format(self.describe()),
                    id="2_0.W001",
                )
            )
        return warnings

    def _compile(self, route):
        return re.compile(_route_to_regex(route, self._is_endpoint)[0])

    def __str__(self):
        return str(self._route)
```
### 72 - django/utils/translation/trans_real.py:

Start line: 589, End line: 611

```python
@functools.lru_cache(maxsize=1000)
def _parse_accept_lang_header(lang_string):
    """
    Parse the lang_string, which is the body of an HTTP Accept-Language
    header, and return a tuple of (lang, q-value), ordered by 'q' values.

    Return an empty tuple if there are any format errors in lang_string.
    """
    result = []
    pieces = accept_language_re.split(lang_string.lower())
    if pieces[-1]:
        return ()
    for i in range(0, len(pieces) - 1, 3):
        first, lang, priority = pieces[i : i + 3]
        if first:
            return ()
        if priority:
            priority = float(priority)
        else:
            priority = 1.0
        result.append((lang, priority))
    result.sort(key=lambda k: k[1], reverse=True)
    return tuple(result)
```
### 74 - django/urls/resolvers.py:

Start line: 727, End line: 738

```python
class URLResolver:

    def resolve_error_handler(self, view_type):
        callback = getattr(self.urlconf_module, "handler%s" % view_type, None)
        if not callback:
            # No handler specified in file; use lazy import, since
            # django.conf.urls imports this file.
            from django.conf import urls

            callback = getattr(urls, "handler%s" % view_type)
        return get_callable(callback)

    def reverse(self, lookup_view, *args, **kwargs):
        return self._reverse_with_prefix(lookup_view, "", *args, **kwargs)
```
### 75 - django/urls/resolvers.py:

Start line: 376, End line: 439

```python
class URLPattern:
    def __init__(self, pattern, callback, default_args=None, name=None):
        self.pattern = pattern
        self.callback = callback  # the view
        self.default_args = default_args or {}
        self.name = name

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.pattern.describe())

    def check(self):
        warnings = self._check_pattern_name()
        warnings.extend(self.pattern.check())
        warnings.extend(self._check_callback())
        return warnings

    def _check_pattern_name(self):
        """
        Check that the pattern name does not contain a colon.
        """
        if self.pattern.name is not None and ":" in self.pattern.name:
            warning = Warning(
                "Your URL pattern {} has a name including a ':'. Remove the colon, to "
                "avoid ambiguous namespace references.".format(self.pattern.describe()),
                id="urls.W003",
            )
            return [warning]
        else:
            return []

    def _check_callback(self):
        from django.views import View

        view = self.callback
        if inspect.isclass(view) and issubclass(view, View):
            return [
                Error(
                    "Your URL pattern %s has an invalid view, pass %s.as_view() "
                    "instead of %s."
                    % (
                        self.pattern.describe(),
                        view.__name__,
                        view.__name__,
                    ),
                    id="urls.E009",
                )
            ]
        return []

    def resolve(self, path):
        match = self.pattern.match(path)
        if match:
            new_path, args, captured_kwargs = match
            # Pass any default args as **kwargs.
            kwargs = {**captured_kwargs, **self.default_args}
            return ResolverMatch(
                self.callback,
                args,
                kwargs,
                self.pattern.name,
                route=str(self.pattern),
                captured_kwargs=captured_kwargs,
                extra_kwargs=self.default_args,
            )
```
### 77 - django/urls/resolvers.py:

Start line: 531, End line: 613

```python
class URLResolver:

    def _populate(self):
        # Short-circuit if called recursively in this thread to prevent
        # infinite recursion. Concurrent threads may call this at the same
        # time and will need to continue, so set 'populating' on a
        # thread-local variable.
        if getattr(self._local, "populating", False):
            return
        try:
            self._local.populating = True
            lookups = MultiValueDict()
            namespaces = {}
            apps = {}
            language_code = get_language()
            for url_pattern in reversed(self.url_patterns):
                p_pattern = url_pattern.pattern.regex.pattern
                p_pattern = p_pattern.removeprefix("^")
                if isinstance(url_pattern, URLPattern):
                    self._callback_strs.add(url_pattern.lookup_str)
                    bits = normalize(url_pattern.pattern.regex.pattern)
                    lookups.appendlist(
                        url_pattern.callback,
                        (
                            bits,
                            p_pattern,
                            url_pattern.default_args,
                            url_pattern.pattern.converters,
                        ),
                    )
                    if url_pattern.name is not None:
                        lookups.appendlist(
                            url_pattern.name,
                            (
                                bits,
                                p_pattern,
                                url_pattern.default_args,
                                url_pattern.pattern.converters,
                            ),
                        )
                else:  # url_pattern is a URLResolver.
                    url_pattern._populate()
                    if url_pattern.app_name:
                        apps.setdefault(url_pattern.app_name, []).append(
                            url_pattern.namespace
                        )
                        namespaces[url_pattern.namespace] = (p_pattern, url_pattern)
                    else:
                        for name in url_pattern.reverse_dict:
                            for (
                                matches,
                                pat,
                                defaults,
                                converters,
                            ) in url_pattern.reverse_dict.getlist(name):
                                new_matches = normalize(p_pattern + pat)
                                lookups.appendlist(
                                    name,
                                    (
                                        new_matches,
                                        p_pattern + pat,
                                        {**defaults, **url_pattern.default_kwargs},
                                        {
                                            **self.pattern.converters,
                                            **url_pattern.pattern.converters,
                                            **converters,
                                        },
                                    ),
                                )
                        for namespace, (
                            prefix,
                            sub_pattern,
                        ) in url_pattern.namespace_dict.items():
                            current_converters = url_pattern.pattern.converters
                            sub_pattern.pattern.converters.update(current_converters)
                            namespaces[namespace] = (p_pattern + prefix, sub_pattern)
                        for app_name, namespace_list in url_pattern.app_dict.items():
                            apps.setdefault(app_name, []).extend(namespace_list)
                    self._callback_strs.update(url_pattern._callback_strs)
            self._namespace_dict[language_code] = namespaces
            self._app_dict[language_code] = apps
            self._reverse_dict[language_code] = lookups
            self._populated = True
        finally:
            self._local.populating = False
```
### 79 - django/urls/resolvers.py:

Start line: 457, End line: 477

```python
class URLResolver:
    def __init__(
        self, pattern, urlconf_name, default_kwargs=None, app_name=None, namespace=None
    ):
        self.pattern = pattern
        # urlconf_name is the dotted Python path to the module defining
        # urlpatterns. It may also be an object with an urlpatterns attribute
        # or urlpatterns itself.
        self.urlconf_name = urlconf_name
        self.callback = None
        self.default_kwargs = default_kwargs or {}
        self.namespace = namespace
        self.app_name = app_name
        self._reverse_dict = {}
        self._namespace_dict = {}
        self._app_dict = {}
        # set of dotted paths to all functions and classes that are used in
        # urlpatterns
        self._callback_strs = set()
        self._populated = False
        self._local = Local()
```
### 80 - django/utils/translation/trans_real.py:

Start line: 368, End line: 391

```python
def gettext(message):
    """
    Translate the 'message' string. It uses the current thread to find the
    translation object to use. If no current translation is activated, the
    message will be run through the default translation object.
    """
    global _default

    eol_message = message.replace("\r\n", "\n").replace("\r", "\n")

    if eol_message:
        _default = _default or translation(settings.LANGUAGE_CODE)
        translation_object = getattr(_active, "value", _default)

        result = translation_object.gettext(eol_message)
    else:
        # Return an empty value of the corresponding type if an empty message
        # is given, instead of metadata, which is the default gettext behavior.
        result = type(message)("")

    if isinstance(message, SafeData):
        return mark_safe(result)

    return result
```
### 83 - django/urls/resolvers.py:

Start line: 185, End line: 209

```python
class RegexPattern(CheckURLMixin):
    regex = LocaleRegexDescriptor("_regex")

    def __init__(self, regex, name=None, is_endpoint=False):
        self._regex = regex
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = {}

    def match(self, path):
        match = (
            self.regex.fullmatch(path)
            if self._is_endpoint and self.regex.pattern.endswith("$")
            else self.regex.search(path)
        )
        if match:
            # If there are any named groups, use those as kwargs, ignoring
            # non-named groups. Otherwise, pass all non-named arguments as
            # positional arguments.
            kwargs = match.groupdict()
            args = () if kwargs else match.groups()
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return path[match.end() :], args, kwargs
        return None
```
### 91 - django/urls/resolvers.py:

Start line: 105, End line: 124

```python
def get_resolver(urlconf=None):
    if urlconf is None:
        urlconf = settings.ROOT_URLCONF
    return _get_cached_resolver(urlconf)


@functools.cache
def _get_cached_resolver(urlconf=None):
    return URLResolver(RegexPattern(r"^/"), urlconf)


@functools.cache
def get_ns_resolver(ns_pattern, resolver, converters):
    # Build a namespaced resolver for the given parent URLconf pattern.
    # This makes it possible to have captured parameters in the parent
    # URLconf pattern.
    pattern = RegexPattern(ns_pattern)
    pattern.converters = dict(converters)
    ns_resolver = URLResolver(pattern, resolver.url_patterns)
    return URLResolver(RegexPattern(r"^/"), [ns_resolver])
```
### 93 - django/urls/resolvers.py:

Start line: 127, End line: 147

```python
class LocaleRegexDescriptor:
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, instance, cls=None):
        """
        Return a compiled regular expression based on the active language.
        """
        if instance is None:
            return self
        # As a performance optimization, if the given regex string is a regular
        # string (not a lazily-translated string proxy), compile it once and
        # avoid per-language compilation.
        pattern = getattr(instance, self.attr)
        if isinstance(pattern, str):
            instance.__dict__["regex"] = instance._compile(pattern)
            return instance.__dict__["regex"]
        language_code = get_language()
        if language_code not in instance._regex_dict:
            instance._regex_dict[language_code] = instance._compile(str(pattern))
        return instance._regex_dict[language_code]
```
### 115 - django/utils/translation/__init__.py:

Start line: 45, End line: 61

```python
# a reference to one of these functions, don't break that reference when we
# replace the functions with their real counterparts (once we do access the
# settings).


class Trans:
    """
    The purpose of this class is to store the actual translation function upon
    receiving the first call to that function. After this is done, changes to
    USE_I18N will have no effect to which function is served upon request. If
    your tests rely on changing USE_I18N, you can delete all the functions
    from _trans.__dict__.

    Note that storing the function with setattr will have a noticeable
    performance effect, as access to the function goes the normal path,
    instead of using __getattr__.
    """
```
### 129 - django/urls/resolvers.py:

Start line: 615, End line: 654

```python
class URLResolver:

    @property
    def reverse_dict(self):
        language_code = get_language()
        if language_code not in self._reverse_dict:
            self._populate()
        return self._reverse_dict[language_code]

    @property
    def namespace_dict(self):
        language_code = get_language()
        if language_code not in self._namespace_dict:
            self._populate()
        return self._namespace_dict[language_code]

    @property
    def app_dict(self):
        language_code = get_language()
        if language_code not in self._app_dict:
            self._populate()
        return self._app_dict[language_code]

    @staticmethod
    def _extend_tried(tried, pattern, sub_tried=None):
        if sub_tried is None:
            tried.append([pattern])
        else:
            tried.extend([pattern, *t] for t in sub_tried)

    @staticmethod
    def _join_route(route1, route2):
        """Join two routes, without the starting ^ in the second route."""
        if not route1:
            return route2
        route2 = route2.removeprefix("^")
        return route1 + route2

    def _is_callback(self, name):
        if not self._populated:
            self._populate()
        return name in self._callback_strs
```
### 144 - django/urls/resolvers.py:

Start line: 211, End line: 247

```python
class RegexPattern(CheckURLMixin):

    def check(self):
        warnings = []
        warnings.extend(self._check_pattern_startswith_slash())
        if not self._is_endpoint:
            warnings.extend(self._check_include_trailing_dollar())
        return warnings

    def _check_include_trailing_dollar(self):
        regex_pattern = self.regex.pattern
        if regex_pattern.endswith("$") and not regex_pattern.endswith(r"\$"):
            return [
                Warning(
                    "Your URL pattern {} uses include with a route ending with a '$'. "
                    "Remove the dollar from the route to avoid problems including "
                    "URLs.".format(self.describe()),
                    id="urls.W001",
                )
            ]
        else:
            return []

    def _compile(self, regex):
        """Compile and return the given regular expression."""
        try:
            return re.compile(regex)
        except re.error as e:
            raise ImproperlyConfigured(
                '"%s" is not a valid regular expression: %s' % (regex, e)
            ) from e

    def __str__(self):
        return str(self._regex)


_PATH_PARAMETER_COMPONENT_RE = _lazy_re_compile(
    r"<(?:(?P<converter>[^>:]+):)?(?P<parameter>[^>]+)>"
)
```
### 160 - django/utils/translation/__init__.py:

Start line: 115, End line: 164

```python
def lazy_number(func, resultclass, number=None, **kwargs):
    if isinstance(number, int):
        kwargs["number"] = number
        proxy = lazy(func, resultclass)(**kwargs)
    else:
        original_kwargs = kwargs.copy()

        class NumberAwareString(resultclass):
            def __bool__(self):
                return bool(kwargs["singular"])

            def _get_number_value(self, values):
                try:
                    return values[number]
                except KeyError:
                    raise KeyError(
                        "Your dictionary lacks key '%s'. Please provide "
                        "it, because it is required to determine whether "
                        "string is singular or plural." % number
                    )

            def _translate(self, number_value):
                kwargs["number"] = number_value
                return func(**kwargs)

            def format(self, *args, **kwargs):
                number_value = (
                    self._get_number_value(kwargs) if kwargs and number else args[0]
                )
                return self._translate(number_value).format(*args, **kwargs)

            def __mod__(self, rhs):
                if isinstance(rhs, dict) and number:
                    number_value = self._get_number_value(rhs)
                else:
                    number_value = rhs
                translated = self._translate(number_value)
                try:
                    translated %= rhs
                except TypeError:
                    # String doesn't contain a placeholder for the number.
                    pass
                return translated

        proxy = lazy(lambda **kwargs: NumberAwareString(), NumberAwareString)(**kwargs)
        proxy.__reduce__ = lambda: (
            _lazy_number_unpickle,
            (func, resultclass, number, original_kwargs),
        )
    return proxy
```
