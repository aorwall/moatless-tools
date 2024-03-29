# django__django-16735

| **django/django** | `2eb1f37260f0e0b71ef3a77eb5522d2bb68d6489` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 41025 |
| **Any found context length** | 443 |
| **Avg pos** | 70.5 |
| **Min pos** | 2 |
| **Max pos** | 139 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -23,7 +23,7 @@
 from django.utils.functional import cached_property
 from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
 from django.utils.regex_helper import _lazy_re_compile, normalize
-from django.utils.translation import get_language
+from django.utils.translation import get_language, get_supported_language_variant
 
 from .converters import get_converter
 from .exceptions import NoReverseMatch, Resolver404
@@ -351,7 +351,8 @@ def regex(self):
     @property
     def language_prefix(self):
         language_code = get_language() or settings.LANGUAGE_CODE
-        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
+        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
+        if language_code == default_language and not self.prefix_default_language:
             return ""
         else:
             return "%s/" % language_code
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -17,6 +17,7 @@
     "get_language_from_request",
     "get_language_info",
     "get_language_bidi",
+    "get_supported_language_variant",
     "check_for_language",
     "to_language",
     "to_locale",

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/urls/resolvers.py | 26 | 26 | - | 2 | -
| django/urls/resolvers.py | 354 | 354 | 2 | 2 | 443
| django/utils/translation/__init__.py | 20 | 20 | 139 | 53 | 41025


## Problem Statement

```
i18n_patterns() not respecting prefix_default_language=False
Description
	 
		(last modified by Oussama Jarrousse)
	 
In my django project urls.py file I have the following setup:
from django.conf.urls.i18n import i18n_patterns
from django.contrib import admin
from django.urls import include
from django.urls import path
urlpatterns = []
# as an example... include the admin.site.urls 
urlpatterns += i18n_patterns(
	path("admin/", admin.site.urls), prefix_default_language=False
)
In versions Django==4.1.7 (or prior), I was able to navigating to /admin/ without having to add the language prefix.
Django==4.2.0, navigating to /admin/ will cause a HTTP 302 and only /en/admin/ works... although prefix_default_language=False is explicitly defined.
This change broke my API upon backend packages upgrade from 4.1.7 to 4.2.0

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/conf/urls/i18n.py | 1 | 40| 251 | 251 | 251 | 
| **-> 2 <-** | **2 django/urls/resolvers.py** | 341 | 372| 192 | 443 | 6315 | 
| 3 | 3 django/contrib/admindocs/urls.py | 1 | 51| 307 | 750 | 6622 | 
| 4 | 4 django/contrib/auth/urls.py | 1 | 37| 253 | 1003 | 6875 | 
| 5 | 5 django/conf/global_settings.py | 153 | 263| 832 | 1835 | 12713 | 
| 6 | 6 django/contrib/admin/options.py | 693 | 733| 280 | 2115 | 32008 | 
| 7 | 7 django/urls/__init__.py | 1 | 54| 269 | 2384 | 32277 | 
| 8 | 8 django/middleware/locale.py | 46 | 90| 414 | 2798 | 33014 | 
| 9 | 9 django/contrib/admin/sites.py | 251 | 313| 527 | 3325 | 37511 | 
| 10 | 10 django/urls/conf.py | 1 | 58| 451 | 3776 | 38228 | 
| 11 | 10 django/contrib/admin/sites.py | 1 | 33| 224 | 4000 | 38228 | 
| 12 | **10 django/urls/resolvers.py** | 703 | 724| 187 | 4187 | 38228 | 
| 13 | **10 django/urls/resolvers.py** | 739 | 827| 722 | 4909 | 38228 | 
| 14 | 10 django/contrib/admin/sites.py | 443 | 459| 136 | 5045 | 38228 | 
| 15 | 11 django/contrib/admin/__init__.py | 1 | 53| 292 | 5337 | 38520 | 
| 16 | 12 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 5756 | 38939 | 
| 17 | 13 django/urls/base.py | 1 | 24| 170 | 5926 | 40135 | 
| 18 | **13 django/urls/resolvers.py** | 185 | 209| 203 | 6129 | 40135 | 
| 19 | 13 django/contrib/admin/sites.py | 600 | 614| 116 | 6245 | 40135 | 
| 20 | 13 django/urls/base.py | 91 | 157| 383 | 6628 | 40135 | 
| 21 | 14 django/contrib/auth/admin.py | 1 | 25| 195 | 6823 | 41906 | 
| 22 | **14 django/urls/resolvers.py** | 320 | 338| 163 | 6986 | 41906 | 
| 23 | 14 django/urls/conf.py | 61 | 96| 266 | 7252 | 41906 | 
| 24 | **14 django/urls/resolvers.py** | 296 | 318| 173 | 7425 | 41906 | 
| 25 | **14 django/urls/resolvers.py** | 456 | 476| 177 | 7602 | 41906 | 
| 26 | **14 django/urls/resolvers.py** | 375 | 438| 404 | 8006 | 41906 | 
| 27 | 15 django/views/defaults.py | 1 | 26| 151 | 8157 | 42896 | 
| 28 | 16 django/__init__.py | 1 | 25| 173 | 8330 | 43069 | 
| 29 | **16 django/urls/resolvers.py** | 655 | 701| 384 | 8714 | 43069 | 
| 30 | 16 django/contrib/admin/options.py | 735 | 752| 128 | 8842 | 43069 | 
| 31 | 17 django/conf/urls/__init__.py | 1 | 10| 0 | 8842 | 43134 | 
| 32 | **17 django/urls/resolvers.py** | 211 | 247| 268 | 9110 | 43134 | 
| 33 | 18 django/contrib/sitemaps/__init__.py | 59 | 221| 1145 | 10255 | 44962 | 
| 34 | 18 django/urls/base.py | 160 | 188| 201 | 10456 | 44962 | 
| 35 | **18 django/urls/resolvers.py** | 150 | 182| 235 | 10691 | 44962 | 
| 36 | 18 django/urls/base.py | 27 | 88| 440 | 11131 | 44962 | 
| 37 | 18 django/contrib/admin/options.py | 1 | 119| 796 | 11927 | 44962 | 
| 38 | 18 django/conf/global_settings.py | 597 | 668| 453 | 12380 | 44962 | 
| 39 | 19 docs/conf.py | 133 | 236| 934 | 13314 | 48515 | 
| 40 | 20 django/contrib/auth/decorators.py | 1 | 40| 315 | 13629 | 49107 | 
| 41 | 21 docs/_ext/djangodocs.py | 26 | 71| 398 | 14027 | 52331 | 
| 42 | 22 django/views/i18n.py | 30 | 74| 382 | 14409 | 54200 | 
| 43 | 22 django/views/i18n.py | 126 | 140| 138 | 14547 | 54200 | 
| 44 | 22 django/contrib/admin/sites.py | 570 | 597| 196 | 14743 | 54200 | 
| 45 | **22 django/urls/resolvers.py** | 530 | 612| 575 | 15318 | 54200 | 
| 46 | 23 django/contrib/admin/templatetags/admin_list.py | 1 | 33| 194 | 15512 | 58006 | 
| 47 | **23 django/urls/resolvers.py** | 105 | 124| 158 | 15670 | 58006 | 
| 48 | 24 django/middleware/common.py | 153 | 179| 255 | 15925 | 59552 | 
| 49 | **24 django/urls/resolvers.py** | 614 | 653| 276 | 16201 | 59552 | 
| 50 | 24 django/views/i18n.py | 114 | 124| 137 | 16338 | 59552 | 
| 51 | 25 django/contrib/admin/widgets.py | 384 | 453| 373 | 16711 | 63742 | 
| 52 | 26 django/contrib/admin/models.py | 1 | 21| 123 | 16834 | 64935 | 
| 53 | 27 django/contrib/redirects/admin.py | 1 | 11| 0 | 16834 | 65004 | 
| 54 | **27 django/urls/resolvers.py** | 478 | 497| 167 | 17001 | 65004 | 
| 55 | 28 scripts/manage_translations.py | 1 | 29| 195 | 17196 | 66702 | 
| 56 | 28 docs/conf.py | 54 | 131| 711 | 17907 | 66702 | 
| 57 | 29 django/db/models/options.py | 1 | 58| 353 | 18260 | 74398 | 
| 58 | 30 django/contrib/admin/tests.py | 1 | 37| 265 | 18525 | 76040 | 
| 59 | 31 django/core/checks/urls.py | 76 | 118| 266 | 18791 | 76746 | 
| 60 | 31 django/contrib/admin/sites.py | 228 | 249| 221 | 19012 | 76746 | 
| 61 | 32 django/contrib/admindocs/views.py | 1 | 36| 252 | 19264 | 80234 | 
| 62 | 33 django/core/management/base.py | 85 | 112| 161 | 19425 | 85094 | 
| 63 | 34 django/contrib/flatpages/urls.py | 1 | 7| 0 | 19425 | 85132 | 
| 64 | **34 django/urls/resolvers.py** | 499 | 528| 292 | 19717 | 85132 | 
| 65 | 34 django/conf/global_settings.py | 1 | 50| 367 | 20084 | 85132 | 
| 66 | 34 django/contrib/admin/options.py | 121 | 154| 235 | 20319 | 85132 | 
| 67 | 35 django/contrib/admin/views/main.py | 1 | 61| 357 | 20676 | 89890 | 
| 68 | 35 django/middleware/locale.py | 1 | 44| 329 | 21005 | 89890 | 
| 69 | 36 django/contrib/admin/checks.py | 55 | 173| 772 | 21777 | 99416 | 
| 70 | 37 django/contrib/admindocs/utils.py | 127 | 168| 306 | 22083 | 101401 | 
| 71 | 37 django/contrib/admindocs/views.py | 455 | 483| 194 | 22277 | 101401 | 
| 72 | 38 django/contrib/gis/admin/__init__.py | 1 | 30| 130 | 22407 | 101531 | 
| 73 | 38 django/contrib/admindocs/views.py | 486 | 499| 122 | 22529 | 101531 | 
| 74 | 39 django/contrib/flatpages/admin.py | 1 | 23| 148 | 22677 | 101679 | 
| 75 | 39 django/core/checks/urls.py | 57 | 73| 128 | 22805 | 101679 | 
| 76 | 40 django/contrib/admin/migrations/0001_initial.py | 1 | 76| 363 | 23168 | 102042 | 
| 77 | 41 django/contrib/admin/helpers.py | 246 | 259| 115 | 23283 | 105654 | 
| 78 | 42 django/contrib/auth/__init__.py | 1 | 38| 240 | 23523 | 107368 | 
| 79 | 42 django/views/defaults.py | 29 | 79| 377 | 23900 | 107368 | 
| 80 | 42 django/middleware/common.py | 34 | 60| 265 | 24165 | 107368 | 
| 81 | 42 django/contrib/auth/admin.py | 149 | 214| 477 | 24642 | 107368 | 
| 82 | 43 django/contrib/admin/exceptions.py | 1 | 14| 0 | 24642 | 107435 | 
| 83 | 43 django/conf/global_settings.py | 481 | 596| 797 | 25439 | 107435 | 
| 84 | 44 django/conf/__init__.py | 129 | 142| 117 | 25556 | 109800 | 
| 85 | 45 django/contrib/gis/views.py | 1 | 23| 160 | 25716 | 109960 | 
| 86 | 46 django/contrib/sites/admin.py | 1 | 9| 0 | 25716 | 110006 | 
| 87 | 47 django/core/management/utils.py | 130 | 176| 316 | 26032 | 111239 | 
| 88 | 47 django/contrib/admin/widgets.py | 171 | 204| 244 | 26276 | 111239 | 
| 89 | 47 django/conf/__init__.py | 175 | 241| 563 | 26839 | 111239 | 
| 90 | 47 django/contrib/admin/tests.py | 184 | 202| 177 | 27016 | 111239 | 
| 91 | 47 django/contrib/admindocs/utils.py | 198 | 212| 207 | 27223 | 111239 | 
| 92 | 48 django/contrib/admin/utils.py | 79 | 119| 279 | 27502 | 115629 | 
| 93 | 48 scripts/manage_translations.py | 200 | 220| 130 | 27632 | 115629 | 
| 94 | 48 django/contrib/admin/options.py | 1524 | 1550| 233 | 27865 | 115629 | 
| 95 | 49 django/utils/http.py | 303 | 324| 179 | 28044 | 118831 | 
| 96 | 50 django/http/request.py | 441 | 489| 338 | 28382 | 124409 | 
| 97 | 50 django/middleware/common.py | 1 | 32| 247 | 28629 | 124409 | 
| 98 | 50 django/contrib/admin/options.py | 439 | 499| 494 | 29123 | 124409 | 
| 99 | 51 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 24| 117 | 29240 | 124526 | 
| 100 | 51 django/contrib/admin/helpers.py | 195 | 244| 386 | 29626 | 124526 | 
| 101 | 52 django/contrib/admin/apps.py | 1 | 28| 164 | 29790 | 124690 | 
| 102 | 52 django/conf/global_settings.py | 405 | 480| 790 | 30580 | 124690 | 
| 103 | 52 django/contrib/admin/options.py | 2481 | 2516| 315 | 30895 | 124690 | 
| 104 | **53 django/utils/translation/__init__.py** | 62 | 81| 138 | 31033 | 126566 | 
| 105 | 54 django/core/checks/security/base.py | 1 | 79| 691 | 31724 | 128755 | 
| 106 | 55 django/db/utils.py | 197 | 235| 246 | 31970 | 130654 | 
| 107 | 56 django/contrib/admin/decorators.py | 80 | 112| 142 | 32112 | 131308 | 
| 108 | 56 django/contrib/admin/options.py | 1763 | 1865| 780 | 32892 | 131308 | 
| 109 | 56 django/contrib/admin/tests.py | 132 | 147| 169 | 33061 | 131308 | 
| 110 | 56 django/contrib/admin/sites.py | 360 | 381| 182 | 33243 | 131308 | 
| 111 | 56 django/contrib/admin/views/main.py | 64 | 170| 866 | 34109 | 131308 | 
| 112 | 56 django/db/utils.py | 1 | 50| 177 | 34286 | 131308 | 
| 113 | 57 django/conf/urls/static.py | 1 | 31| 200 | 34486 | 131508 | 
| 114 | 58 django/contrib/redirects/migrations/0001_initial.py | 1 | 65| 309 | 34795 | 131817 | 
| 115 | 58 django/middleware/common.py | 100 | 115| 165 | 34960 | 131817 | 
| 116 | 58 django/contrib/admin/checks.py | 176 | 192| 155 | 35115 | 131817 | 
| 117 | 58 django/contrib/admin/options.py | 243 | 256| 139 | 35254 | 131817 | 
| 118 | 59 django/contrib/auth/views.py | 1 | 32| 255 | 35509 | 134531 | 
| 119 | 59 django/contrib/admin/widgets.py | 456 | 466| 117 | 35626 | 134531 | 
| 120 | 60 django/contrib/admin/filters.py | 1 | 21| 144 | 35770 | 140257 | 
| 121 | 60 django/contrib/admin/options.py | 676 | 691| 123 | 35893 | 140257 | 
| 122 | 60 django/contrib/admin/options.py | 1176 | 1200| 196 | 36089 | 140257 | 
| 123 | 60 django/contrib/admin/views/main.py | 276 | 292| 125 | 36214 | 140257 | 
| 124 | **60 django/urls/resolvers.py** | 726 | 737| 120 | 36334 | 140257 | 
| 125 | 60 django/contrib/admin/widgets.py | 361 | 381| 172 | 36506 | 140257 | 
| 126 | 60 django/contrib/admindocs/utils.py | 215 | 229| 225 | 36731 | 140257 | 
| 127 | **60 django/utils/translation/__init__.py** | 230 | 265| 271 | 37002 | 140257 | 
| 128 | 61 django/core/checks/compatibility/django_4_0.py | 1 | 21| 142 | 37144 | 140400 | 
| 129 | 61 django/contrib/admin/options.py | 2187 | 2245| 444 | 37588 | 140400 | 
| 130 | **61 django/utils/translation/__init__.py** | 166 | 227| 345 | 37933 | 140400 | 
| 131 | 61 django/contrib/admin/options.py | 1692 | 1728| 337 | 38270 | 140400 | 
| 132 | 62 django/core/checks/translation.py | 1 | 67| 449 | 38719 | 140849 | 
| 133 | 62 django/contrib/admin/options.py | 2017 | 2108| 784 | 39503 | 140849 | 
| 134 | 62 django/contrib/admin/views/main.py | 331 | 361| 270 | 39773 | 140849 | 
| 135 | 63 django/template/context_processors.py | 58 | 90| 143 | 39916 | 141344 | 
| 136 | 63 django/contrib/admindocs/views.py | 141 | 161| 170 | 40086 | 141344 | 
| 137 | 64 django/template/defaultfilters.py | 368 | 455| 504 | 40590 | 147899 | 
| 138 | 64 django/contrib/admin/widgets.py | 95 | 120| 180 | 40770 | 147899 | 
| **-> 139 <-** | **64 django/utils/translation/__init__.py** | 1 | 43| 255 | 41025 | 147899 | 
| 140 | 65 django/contrib/auth/password_validation.py | 1 | 38| 218 | 41243 | 149793 | 
| 141 | 65 django/views/defaults.py | 102 | 121| 144 | 41387 | 149793 | 
| 142 | 66 django/db/models/__init__.py | 1 | 116| 682 | 42069 | 150475 | 
| 143 | 66 django/contrib/admin/sites.py | 36 | 79| 339 | 42408 | 150475 | 
| 144 | 67 django/core/validators.py | 68 | 111| 609 | 43017 | 155096 | 
| 145 | 68 django/core/management/commands/makemessages.py | 227 | 324| 674 | 43691 | 161117 | 
| 146 | 68 django/contrib/admin/checks.py | 1243 | 1272| 196 | 43887 | 161117 | 
| 147 | 69 django/utils/translation/trans_real.py | 1 | 68| 571 | 44458 | 165854 | 
| 148 | 69 django/contrib/admin/checks.py | 789 | 807| 183 | 44641 | 165854 | 
| 149 | 70 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 44641 | 165951 | 
| 150 | 71 django/utils/log.py | 1 | 76| 501 | 45142 | 167625 | 
| 151 | 71 django/contrib/admin/templatetags/admin_list.py | 180 | 196| 140 | 45282 | 167625 | 
| 152 | 71 django/contrib/admin/options.py | 1271 | 1333| 511 | 45793 | 167625 | 
| 153 | 71 django/contrib/admin/options.py | 1866 | 1897| 303 | 46096 | 167625 | 
| 154 | 71 django/contrib/admin/utils.py | 340 | 402| 469 | 46565 | 167625 | 
| 155 | 71 django/conf/global_settings.py | 51 | 152| 1188 | 47753 | 167625 | 
| 156 | 72 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 47900 | 167773 | 
| 157 | 72 django/conf/global_settings.py | 264 | 355| 832 | 48732 | 167773 | 
| 158 | 73 django/apps/registry.py | 1 | 59| 475 | 49207 | 171193 | 
| 159 | 74 django/contrib/syndication/views.py | 1 | 26| 223 | 49430 | 173051 | 
| 160 | 74 django/views/defaults.py | 124 | 150| 197 | 49627 | 173051 | 
| 161 | 74 django/contrib/admin/sites.py | 315 | 340| 202 | 49829 | 173051 | 
| 162 | 75 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 49952 | 178879 | 
| 163 | 75 django/middleware/common.py | 62 | 74| 136 | 50088 | 178879 | 
| 164 | 75 django/contrib/admindocs/utils.py | 91 | 124| 210 | 50298 | 178879 | 
| 165 | 76 django/urls/exceptions.py | 1 | 10| 0 | 50298 | 178904 | 
| 166 | 77 django/contrib/sitemaps/views.py | 91 | 141| 369 | 50667 | 179976 | 
| 167 | 77 django/contrib/admin/options.py | 364 | 437| 500 | 51167 | 179976 | 
| 168 | 78 django/contrib/flatpages/models.py | 1 | 50| 368 | 51535 | 180344 | 
| 169 | 78 docs/conf.py | 1 | 53| 485 | 52020 | 180344 | 
| 170 | 79 django/http/response.py | 656 | 689| 157 | 52177 | 185747 | 
| 171 | 79 django/contrib/admin/views/main.py | 172 | 274| 880 | 53057 | 185747 | 
| 172 | 79 django/views/i18n.py | 77 | 95| 129 | 53186 | 185747 | 
| 173 | 80 django/contrib/postgres/utils.py | 1 | 30| 219 | 53405 | 185966 | 
| 174 | 81 django/contrib/auth/migrations/0001_initial.py | 1 | 205| 1007 | 54412 | 186973 | 
| 175 | 82 django/contrib/flatpages/migrations/0001_initial.py | 1 | 69| 355 | 54767 | 187328 | 
| 176 | 83 django/conf/locale/id/formats.py | 5 | 50| 678 | 55445 | 188051 | 
| 177 | 83 django/utils/translation/trans_real.py | 447 | 459| 112 | 55557 | 188051 | 
| 178 | 84 django/core/checks/security/csrf.py | 1 | 42| 305 | 55862 | 188516 | 
| 179 | 85 django/conf/locale/__init__.py | 1 | 624| 75 | 55937 | 192879 | 
| 180 | 86 django/utils/translation/trans_null.py | 1 | 68| 272 | 56209 | 193151 | 
| 181 | 86 django/db/models/options.py | 496 | 519| 155 | 56364 | 193151 | 
| 182 | 86 django/contrib/admin/views/main.py | 294 | 329| 312 | 56676 | 193151 | 
| 183 | 86 django/core/checks/urls.py | 1 | 28| 142 | 56818 | 193151 | 
| 184 | 86 django/utils/translation/trans_real.py | 234 | 249| 158 | 56976 | 193151 | 
| 185 | **86 django/utils/translation/__init__.py** | 268 | 302| 236 | 57212 | 193151 | 
| 186 | 86 scripts/manage_translations.py | 89 | 109| 191 | 57403 | 193151 | 
| 187 | 86 django/contrib/admin/checks.py | 460 | 478| 137 | 57540 | 193151 | 


### Hint

```
Thanks for the ticket, however I'm not able to reproduce this issue. Can you provide a small sample project that reproduces this? (it seems to be related with 94e7f471c4edef845a4fe5e3160132997b4cca81.)
I will provide the project shortly on github... In the meanwhile, I assume you were not able to reproduce the issue because you did not include: django.middleware.locale.LocaleMiddleware here is MIDDLEWARE list in settings.py MIDDLEWARE = [ "django.middleware.security.SecurityMiddleware", "django.contrib.sessions.middleware.SessionMiddleware", "django.middleware.locale.LocaleMiddleware", # This line is important "django.middleware.common.CommonMiddleware", "django.middleware.csrf.CsrfViewMiddleware", "django.contrib.auth.middleware.AuthenticationMiddleware", "django.contrib.messages.middleware.MessageMiddleware", "django.middleware.clickjacking.XFrameOptionsMiddleware", ]
I prepared a simple (pytest) test that navigates to a path that should return HTTP status_code == 200. I prepare a tox.ini file that runs pytest in two different Django environments once with Django==4.1.7 and another with Django==4.2.0 Tox will run the test twice consecutively. The first will pass. the second will fail. ​https://github.com/oussjarrousse/djangoproject-ticket-34455
Oussama, thanks! Regression in 94e7f471c4edef845a4fe5e3160132997b4cca81. Reproduced at 0e1aae7a5f51408b73c5a29e18bd1803dd030930.
I want to work on this bug, I think the real problem is in getting the language from request. Sending the pull request.
I have created a pull request. Can you please review it? ​https://github.com/django/django/pull/16727 Replying to Mariusz Felisiak: Oussama, thanks! Regression in 94e7f471c4edef845a4fe5e3160132997b4cca81. Reproduced at 0e1aae7a5f51408b73c5a29e18bd1803dd030930.
I want to work on this bug, I think the real problem is in getting the language from request. Sending the pull request. I don't think think there is an issue. I'd rather suspect ​this line.
Unable to replicate the bug. For me, it works for both version 4.2 and 4.1.7. I used LocaleMiddleware
Replying to Mohit Singh Sinsniwal: Unable to replicate the bug. For me, it works for both version 4.2 and 4.1.7. I used LocaleMiddleware Please don't close already accepted tickets. I'm still able to reproduce the issue.
Replying to Mohit Singh Sinsniwal: Unable to replicate the bug. For me, it works for both version 4.2 and 4.1.7. I used LocaleMiddleware here is a project to replicate the issue... it uses tox to setup two different environments and run a simple test in each environment. ​​https://github.com/oussjarrousse/djangoproject-ticket-34455
Oussama, thanks, would you like to prepare a patch?
Replying to Mariusz Felisiak: Oussama, thanks, would you like to prepare a patch? In theory, I would love to. However, I am not familiar enough with the core source code.
Replying to Mariusz Felisiak: Replying to Mohit Singh Sinsniwal: Unable to replicate the bug. For me, it works for both version 4.2 and 4.1.7. I used LocaleMiddleware Please don't close already accepted tickets. I'm still able to reproduce the issue. Mariusz, sorry for closing it, I went on a different track while solving the issue, and now I can replicate. I need your help in understanding the middleware. Locale class, what should be done with /admin/login/?next=/admin ? When /admin/login/?next=/admin is requested, it calls get_fallback_lanuage and redirects afterward to /en/admin/login/?next=/en/admin/ get_faalback_language is taking the prefixed language. If we dont want that, then we can update the process_request function: OLD: def process_request(self, request): urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF) i18n_patterns_used, _ = is_language_prefix_patterns_used(urlconf) language = translation.get_language_from_request( request, check_path=i18n_patterns_used ) if not language: language = self.get_fallback_language(request) translation.activate(language) request.LANGUAGE_CODE = translation.get_language() New: def process_request(self, request): urlconf = getattr(request, "urlconf", settings.ROOT_URLCONF) ( i18n_patterns_used, prefixed_default_language, ) = is_language_prefix_patterns_used(urlconf) language = translation.get_language_from_request( request, check_path=i18n_patterns_used ) language_from_path = translation.get_language_from_path(request.path_info) if ( not language_from_path and i18n_patterns_used and prefixed_default_language ): language = settings.LANGUAGE_CODE translation.activate(language) request.LANGUAGE_CODE = translation.get_language() I want to work on this issue and need your suggestion if I am on right track.
```

## Patch

```diff
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -23,7 +23,7 @@
 from django.utils.functional import cached_property
 from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
 from django.utils.regex_helper import _lazy_re_compile, normalize
-from django.utils.translation import get_language
+from django.utils.translation import get_language, get_supported_language_variant
 
 from .converters import get_converter
 from .exceptions import NoReverseMatch, Resolver404
@@ -351,7 +351,8 @@ def regex(self):
     @property
     def language_prefix(self):
         language_code = get_language() or settings.LANGUAGE_CODE
-        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
+        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
+        if language_code == default_language and not self.prefix_default_language:
             return ""
         else:
             return "%s/" % language_code
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -17,6 +17,7 @@
     "get_language_from_request",
     "get_language_info",
     "get_language_bidi",
+    "get_supported_language_variant",
     "check_for_language",
     "to_language",
     "to_locale",

```

## Test Patch

```diff
diff --git a/tests/i18n/tests.py b/tests/i18n/tests.py
--- a/tests/i18n/tests.py
+++ b/tests/i18n/tests.py
@@ -1916,6 +1916,12 @@ def test_default_lang_without_prefix(self):
         response = self.client.get("/simple/")
         self.assertEqual(response.content, b"Yes")
 
+    @override_settings(LANGUAGE_CODE="en-us")
+    def test_default_lang_fallback_without_prefix(self):
+        response = self.client.get("/simple/")
+        self.assertEqual(response.status_code, 200)
+        self.assertEqual(response.content, b"Yes")
+
     def test_other_lang_with_prefix(self):
         response = self.client.get("/fr/simple/")
         self.assertEqual(response.content, b"Oui")

```


## Code snippets

### 1 - django/conf/urls/i18n.py:

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
### 2 - django/urls/resolvers.py:

Start line: 341, End line: 372

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
        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
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
### 3 - django/contrib/admindocs/urls.py:

Start line: 1, End line: 51

```python
from django.contrib.admindocs import views
from django.urls import path, re_path

urlpatterns = [
    path(
        "",
        views.BaseAdminDocsView.as_view(template_name="admin_doc/index.html"),
        name="django-admindocs-docroot",
    ),
    path(
        "bookmarklets/",
        views.BookmarkletsView.as_view(),
        name="django-admindocs-bookmarklets",
    ),
    path(
        "tags/",
        views.TemplateTagIndexView.as_view(),
        name="django-admindocs-tags",
    ),
    path(
        "filters/",
        views.TemplateFilterIndexView.as_view(),
        name="django-admindocs-filters",
    ),
    path(
        "views/",
        views.ViewIndexView.as_view(),
        name="django-admindocs-views-index",
    ),
    path(
        "views/<view>/",
        views.ViewDetailView.as_view(),
        name="django-admindocs-views-detail",
    ),
    path(
        "models/",
        views.ModelIndexView.as_view(),
        name="django-admindocs-models-index",
    ),
    re_path(
        r"^models/(?P<app_label>[^\.]+)\.(?P<model_name>[^/]+)/$",
        views.ModelDetailView.as_view(),
        name="django-admindocs-models-detail",
    ),
    path(
        "templates/<path:template>/",
        views.TemplateDetailView.as_view(),
        name="django-admindocs-templates",
    ),
]
```
### 4 - django/contrib/auth/urls.py:

Start line: 1, End line: 37

```python
# The views used below are normally mapped in the AdminSite instance.
# This URLs file is used to provide a reliable view deployment for test purposes.
# It is also provided as a convenience to those who want to deploy these URLs
# elsewhere.

from django.contrib.auth import views
from django.urls import path

urlpatterns = [
    path("login/", views.LoginView.as_view(), name="login"),
    path("logout/", views.LogoutView.as_view(), name="logout"),
    path(
        "password_change/", views.PasswordChangeView.as_view(), name="password_change"
    ),
    path(
        "password_change/done/",
        views.PasswordChangeDoneView.as_view(),
        name="password_change_done",
    ),
    path("password_reset/", views.PasswordResetView.as_view(), name="password_reset"),
    path(
        "password_reset/done/",
        views.PasswordResetDoneView.as_view(),
        name="password_reset_done",
    ),
    path(
        "reset/<uidb64>/<token>/",
        views.PasswordResetConfirmView.as_view(),
        name="password_reset_confirm",
    ),
    path(
        "reset/done/",
        views.PasswordResetCompleteView.as_view(),
        name="password_reset_complete",
    ),
]
```
### 5 - django/conf/global_settings.py:

Start line: 153, End line: 263

```python
LANGUAGES_BIDI = ["he", "ar", "ar-dz", "ckb", "fa", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = "django_language"
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = "/"
LANGUAGE_COOKIE_SECURE = False
LANGUAGE_COOKIE_HTTPONLY = False
LANGUAGE_COOKIE_SAMESITE = None

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default charset to use for all HttpResponse objects, if a MIME type isn't
# manually specified. It's used to construct the Content-Type header.
DEFAULT_CHARSET = "utf-8"

# Email address that error messages come from.
SERVER_EMAIL = "root@localhost"

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# Host for sending email.
EMAIL_HOST = "localhost"

# Port for sending email.
EMAIL_PORT = 25

# Whether to send SMTP 'Date' header in the local time zone or in UTC.
EMAIL_USE_LOCALTIME = False

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ""
EMAIL_HOST_PASSWORD = ""
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default form rendering class.
FORM_RENDERER = "django.forms.renderers.DjangoTemplates"

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = "webmaster@localhost"

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = "[Django] "

# Whether to append trailing slashes to URLs.
APPEND_SLASH = True

# Whether to prepend the "www." subdomain to URLs that don't have it.
PREPEND_WWW = False

# Override the server-derived value of SCRIPT_NAME
FORCE_SCRIPT_NAME = None

# List of compiled regular expression objects representing User-Agent strings
# that are not allowed to visit any page, systemwide. Use this for bad
# robots/crawlers. Here are a few examples:
#     import re
#     DISALLOWED_USER_AGENTS = [
#         re.compile(r'^NaverBot.*'),
#         re.compile(r'^EmailSiphon.*'),
#         re.compile(r'^SiteSucker.*'),
#         re.compile(r'^sohu-search'),
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$'),
#        re.compile(r'^/robots.txt$'),
#        re.compile(r'^/phpmyadmin/'),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
```
### 6 - django/contrib/admin/options.py:

Start line: 693, End line: 733

```python
class ModelAdmin(BaseModelAdmin):

    def get_urls(self):
        from django.urls import path

        def wrap(view):
            def wrapper(*args, **kwargs):
                return self.admin_site.admin_view(view)(*args, **kwargs)

            wrapper.model_admin = self
            return update_wrapper(wrapper, view)

        info = self.opts.app_label, self.opts.model_name

        return [
            path("", wrap(self.changelist_view), name="%s_%s_changelist" % info),
            path("add/", wrap(self.add_view), name="%s_%s_add" % info),
            path(
                "<path:object_id>/history/",
                wrap(self.history_view),
                name="%s_%s_history" % info,
            ),
            path(
                "<path:object_id>/delete/",
                wrap(self.delete_view),
                name="%s_%s_delete" % info,
            ),
            path(
                "<path:object_id>/change/",
                wrap(self.change_view),
                name="%s_%s_change" % info,
            ),
            # For backwards compatibility (was the change url before 1.9)
            path(
                "<path:object_id>/",
                wrap(
                    RedirectView.as_view(
                        pattern_name="%s:%s_%s_change"
                        % ((self.admin_site.name,) + info)
                    )
                ),
            ),
        ]
```
### 7 - django/urls/__init__.py:

Start line: 1, End line: 54

```python
from .base import (
    clear_script_prefix,
    clear_url_caches,
    get_script_prefix,
    get_urlconf,
    is_valid_path,
    resolve,
    reverse,
    reverse_lazy,
    set_script_prefix,
    set_urlconf,
    translate_url,
)
from .conf import include, path, re_path
from .converters import register_converter
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import (
    LocalePrefixPattern,
    ResolverMatch,
    URLPattern,
    URLResolver,
    get_ns_resolver,
    get_resolver,
)
from .utils import get_callable, get_mod_func

__all__ = [
    "LocalePrefixPattern",
    "NoReverseMatch",
    "URLPattern",
    "URLResolver",
    "Resolver404",
    "ResolverMatch",
    "clear_script_prefix",
    "clear_url_caches",
    "get_callable",
    "get_mod_func",
    "get_ns_resolver",
    "get_resolver",
    "get_script_prefix",
    "get_urlconf",
    "include",
    "is_valid_path",
    "path",
    "re_path",
    "register_converter",
    "resolve",
    "reverse",
    "reverse_lazy",
    "set_script_prefix",
    "set_urlconf",
    "translate_url",
]
```
### 8 - django/middleware/locale.py:

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
### 9 - django/contrib/admin/sites.py:

Start line: 251, End line: 313

```python
class AdminSite:

    def get_urls(self):
        # Since this module gets imported in the application's root package,
        # it cannot import models from other applications at the module level,
        # and django.contrib.contenttypes.views imports ContentType.
        from django.contrib.contenttypes import views as contenttype_views
        from django.urls import include, path, re_path

        def wrap(view, cacheable=False):
            def wrapper(*args, **kwargs):
                return self.admin_view(view, cacheable)(*args, **kwargs)

            wrapper.admin_site = self
            return update_wrapper(wrapper, view)

        # Admin-site-wide views.
        urlpatterns = [
            path("", wrap(self.index), name="index"),
            path("login/", self.login, name="login"),
            path("logout/", wrap(self.logout), name="logout"),
            path(
                "password_change/",
                wrap(self.password_change, cacheable=True),
                name="password_change",
            ),
            path(
                "password_change/done/",
                wrap(self.password_change_done, cacheable=True),
                name="password_change_done",
            ),
            path("autocomplete/", wrap(self.autocomplete_view), name="autocomplete"),
            path("jsi18n/", wrap(self.i18n_javascript, cacheable=True), name="jsi18n"),
            path(
                "r/<int:content_type_id>/<path:object_id>/",
                wrap(contenttype_views.shortcut),
                name="view_on_site",
            ),
        ]

        # Add in each model's views, and create a list of valid URLS for the
        # app_index
        valid_app_labels = []
        for model, model_admin in self._registry.items():
            urlpatterns += [
                path(
                    "%s/%s/" % (model._meta.app_label, model._meta.model_name),
                    include(model_admin.urls),
                ),
            ]
            if model._meta.app_label not in valid_app_labels:
                valid_app_labels.append(model._meta.app_label)

        # If there were ModelAdmins registered, we should have a list of app
        # labels for which we need to allow access to the app_index view,
        if valid_app_labels:
            regex = r"^(?P<app_label>" + "|".join(valid_app_labels) + ")/$"
            urlpatterns += [
                re_path(regex, wrap(self.app_index), name="app_list"),
            ]

        if self.final_catch_all_view:
            urlpatterns.append(re_path(r"(?P<url>.*)$", wrap(self.catch_all_view)))

        return urlpatterns
```
### 10 - django/urls/conf.py:

Start line: 1, End line: 58

```python
"""Functions for use in URLsconfs."""
from functools import partial
from importlib import import_module

from django.core.exceptions import ImproperlyConfigured

from .resolvers import (
    LocalePrefixPattern,
    RegexPattern,
    RoutePattern,
    URLPattern,
    URLResolver,
)


def include(arg, namespace=None):
    app_name = None
    if isinstance(arg, tuple):
        # Callable returning a namespace hint.
        try:
            urlconf_module, app_name = arg
        except ValueError:
            if namespace:
                raise ImproperlyConfigured(
                    "Cannot override the namespace for a dynamic module that "
                    "provides a namespace."
                )
            raise ImproperlyConfigured(
                "Passing a %d-tuple to include() is not supported. Pass a "
                "2-tuple containing the list of patterns and app_name, and "
                "provide the namespace argument to include() instead." % len(arg)
            )
    else:
        # No namespace hint - use manually provided namespace.
        urlconf_module = arg

    if isinstance(urlconf_module, str):
        urlconf_module = import_module(urlconf_module)
    patterns = getattr(urlconf_module, "urlpatterns", urlconf_module)
    app_name = getattr(urlconf_module, "app_name", app_name)
    if namespace and not app_name:
        raise ImproperlyConfigured(
            "Specifying a namespace in include() without providing an app_name "
            "is not supported. Set the app_name attribute in the included "
            "module, or pass a 2-tuple containing the list of patterns and "
            "app_name instead.",
        )
    namespace = namespace or app_name
    # Make sure the patterns can be iterated through (without this, some
    # testcases will break).
    if isinstance(patterns, (list, tuple)):
        for url_pattern in patterns:
            pattern = getattr(url_pattern, "pattern", None)
            if isinstance(pattern, LocalePrefixPattern):
                raise ImproperlyConfigured(
                    "Using i18n_patterns in an included URLconf is not allowed."
                )
    return (urlconf_module, app_name, namespace)
```
### 12 - django/urls/resolvers.py:

Start line: 703, End line: 724

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
### 13 - django/urls/resolvers.py:

Start line: 739, End line: 827

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
### 18 - django/urls/resolvers.py:

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
### 22 - django/urls/resolvers.py:

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
### 24 - django/urls/resolvers.py:

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
### 25 - django/urls/resolvers.py:

Start line: 456, End line: 476

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
### 26 - django/urls/resolvers.py:

Start line: 375, End line: 438

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
### 29 - django/urls/resolvers.py:

Start line: 655, End line: 701

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
### 32 - django/urls/resolvers.py:

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
### 35 - django/urls/resolvers.py:

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
### 45 - django/urls/resolvers.py:

Start line: 530, End line: 612

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
### 47 - django/urls/resolvers.py:

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
### 49 - django/urls/resolvers.py:

Start line: 614, End line: 653

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
### 54 - django/urls/resolvers.py:

Start line: 478, End line: 497

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
### 64 - django/urls/resolvers.py:

Start line: 499, End line: 528

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
### 104 - django/utils/translation/__init__.py:

Start line: 62, End line: 81

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
### 124 - django/urls/resolvers.py:

Start line: 726, End line: 737

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
### 127 - django/utils/translation/__init__.py:

Start line: 230, End line: 265

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
### 130 - django/utils/translation/__init__.py:

Start line: 166, End line: 227

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
### 139 - django/utils/translation/__init__.py:

Start line: 1, End line: 43

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
### 185 - django/utils/translation/__init__.py:

Start line: 268, End line: 302

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
