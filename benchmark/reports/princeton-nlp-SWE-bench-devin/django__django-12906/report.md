# django__django-12906

| **django/django** | `42c08ee46539ef44f8658ebb1cbefb408e0d03fe` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11441 |
| **Any found context length** | 11441 |
| **Avg pos** | 82.0 |
| **Min pos** | 41 |
| **Max pos** | 41 |
| **Top file pos** | 19 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -59,6 +59,7 @@ def check_dependencies(**kwargs):
     """
     Check that the admin's dependencies are correctly installed.
     """
+    from django.contrib.admin.sites import all_sites
     if not apps.is_installed('django.contrib.admin'):
         return []
     errors = []
@@ -105,6 +106,15 @@ def check_dependencies(**kwargs):
                 "the admin application.",
                 id='admin.E404',
             ))
+        sidebar_enabled = any(site.enable_nav_sidebar for site in all_sites)
+        if (sidebar_enabled and 'django.template.context_processors.request'
+                not in django_templates_instance.context_processors):
+            errors.append(checks.Warning(
+                "'django.template.context_processors.request' must be enabled "
+                "in DjangoTemplates (TEMPLATES) in order to use the admin "
+                "navigation sidebar.",
+                id='admin.W411',
+            ))
 
     if not _contains_subclass('django.contrib.auth.middleware.AuthenticationMiddleware', settings.MIDDLEWARE):
         errors.append(checks.Error(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 62 | 62 | 41 | 19 | 11441
| django/contrib/admin/checks.py | 108 | 108 | 41 | 19 | 11441


## Problem Statement

```
Document admin's requirement on django.template.context_processors.request context processor.
Description
	
Since commit d24ba1be7a53a113d19e2860c03aff9922efec24, the admin templates use the implied request variable normally added by django.template.context_processors.request.
As Django templates silence errors, this went unnoticed during testing, and won't immediately break the templates, but certain expected rendering features won't work.
Django should document this change:
In the release notes (provide a deprecation period where it is a warning only)
In the admin docs
In the system check framework as a warning (but eventually an error)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admindocs/views.py | 318 | 347| 201 | 201 | 3296 | 
| 2 | 2 django/template/context.py | 233 | 260| 199 | 400 | 5177 | 
| 3 | 3 django/template/context_processors.py | 35 | 49| 126 | 526 | 5666 | 
| 4 | 4 django/contrib/admin/options.py | 1123 | 1168| 482 | 1008 | 24221 | 
| 5 | 4 django/contrib/admin/options.py | 1623 | 1647| 279 | 1287 | 24221 | 
| 6 | 5 django/contrib/auth/admin.py | 1 | 22| 188 | 1475 | 25947 | 
| 7 | 6 django/contrib/admin/sites.py | 293 | 315| 187 | 1662 | 30157 | 
| 8 | 6 django/template/context_processors.py | 1 | 32| 218 | 1880 | 30157 | 
| 9 | 6 django/template/context_processors.py | 52 | 82| 143 | 2023 | 30157 | 
| 10 | 6 django/contrib/admindocs/views.py | 56 | 84| 285 | 2308 | 30157 | 
| 11 | 6 django/contrib/admindocs/views.py | 87 | 115| 285 | 2593 | 30157 | 
| 12 | 6 django/contrib/admin/options.py | 1906 | 1943| 330 | 2923 | 30157 | 
| 13 | 7 django/views/generic/base.py | 1 | 30| 173 | 3096 | 31938 | 
| 14 | 7 django/template/context.py | 133 | 167| 288 | 3384 | 31938 | 
| 15 | 8 django/bin/django-admin.py | 1 | 22| 138 | 3522 | 32076 | 
| 16 | 8 django/contrib/admin/options.py | 1536 | 1622| 760 | 4282 | 32076 | 
| 17 | 8 django/contrib/admin/sites.py | 515 | 549| 293 | 4575 | 32076 | 
| 18 | 9 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 4980 | 32481 | 
| 19 | 9 django/template/context.py | 213 | 231| 185 | 5165 | 32481 | 
| 20 | 10 django/contrib/admin/tests.py | 1 | 36| 264 | 5429 | 33958 | 
| 21 | 10 django/contrib/admindocs/views.py | 1 | 30| 223 | 5652 | 33958 | 
| 22 | 10 django/contrib/auth/admin.py | 128 | 189| 465 | 6117 | 33958 | 
| 23 | 11 django/contrib/admin/templatetags/log.py | 26 | 60| 317 | 6434 | 34437 | 
| 24 | 12 django/contrib/admindocs/utils.py | 1 | 25| 151 | 6585 | 36342 | 
| 25 | 12 django/contrib/admin/options.py | 1753 | 1834| 744 | 7329 | 36342 | 
| 26 | 12 django/contrib/admin/options.py | 1457 | 1476| 133 | 7462 | 36342 | 
| 27 | 12 django/template/context.py | 1 | 24| 128 | 7590 | 36342 | 
| 28 | 13 django/contrib/admin/templatetags/admin_list.py | 198 | 212| 136 | 7726 | 40284 | 
| 29 | 13 django/contrib/admindocs/views.py | 118 | 133| 154 | 7880 | 40284 | 
| 30 | 13 django/contrib/admindocs/views.py | 156 | 180| 234 | 8114 | 40284 | 
| 31 | 14 django/contrib/admin/templatetags/base.py | 22 | 34| 134 | 8248 | 40582 | 
| 32 | 15 django/views/csrf.py | 1 | 13| 132 | 8380 | 42126 | 
| 33 | 16 django/contrib/admin/templatetags/admin_modify.py | 1 | 45| 372 | 8752 | 43094 | 
| 34 | 16 django/contrib/admindocs/views.py | 183 | 249| 584 | 9336 | 43094 | 
| 35 | 16 django/contrib/admin/sites.py | 221 | 240| 221 | 9557 | 43094 | 
| 36 | 17 django/views/defaults.py | 100 | 119| 149 | 9706 | 44136 | 
| 37 | 17 django/contrib/admin/templatetags/admin_list.py | 1 | 27| 178 | 9884 | 44136 | 
| 38 | 18 django/contrib/auth/context_processors.py | 24 | 64| 247 | 10131 | 44550 | 
| 39 | 18 django/contrib/admindocs/views.py | 250 | 315| 573 | 10704 | 44550 | 
| 40 | 18 django/views/defaults.py | 1 | 24| 149 | 10853 | 44550 | 
| **-> 41 <-** | **19 django/contrib/admin/checks.py** | 58 | 127| 588 | 11441 | 53557 | 
| 42 | 19 django/contrib/admindocs/views.py | 136 | 154| 187 | 11628 | 53557 | 
| 43 | 19 django/contrib/admin/options.py | 1322 | 1347| 232 | 11860 | 53557 | 
| 44 | 20 django/contrib/messages/context_processors.py | 1 | 14| 0 | 11860 | 53628 | 
| 45 | 20 django/contrib/admin/options.py | 1247 | 1320| 659 | 12519 | 53628 | 
| 46 | 20 django/contrib/admin/options.py | 2163 | 2198| 315 | 12834 | 53628 | 
| 47 | 20 django/template/context.py | 263 | 279| 132 | 12966 | 53628 | 
| 48 | 20 django/contrib/admin/sites.py | 317 | 332| 158 | 13124 | 53628 | 
| 49 | 21 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 13376 | 53925 | 
| 50 | 21 django/contrib/admin/options.py | 1681 | 1752| 653 | 14029 | 53925 | 
| 51 | 21 django/views/csrf.py | 15 | 100| 835 | 14864 | 53925 | 
| 52 | 22 django/contrib/admin/views/main.py | 1 | 45| 324 | 15188 | 58223 | 
| 53 | 23 docs/_ext/djangodocs.py | 26 | 71| 398 | 15586 | 61376 | 
| 54 | 23 django/views/defaults.py | 122 | 149| 214 | 15800 | 61376 | 
| 55 | 23 django/contrib/admin/options.py | 1945 | 1988| 403 | 16203 | 61376 | 
| 56 | 23 django/contrib/admin/options.py | 1 | 95| 756 | 16959 | 61376 | 
| 57 | 23 django/contrib/auth/context_processors.py | 1 | 21| 167 | 17126 | 61376 | 
| 58 | 24 django/template/response.py | 60 | 145| 587 | 17713 | 62455 | 
| 59 | 25 django/contrib/redirects/admin.py | 1 | 11| 0 | 17713 | 62523 | 
| 60 | 25 django/contrib/auth/admin.py | 101 | 126| 286 | 17999 | 62523 | 
| 61 | 25 django/contrib/admindocs/views.py | 33 | 53| 159 | 18158 | 62523 | 
| 62 | 26 django/contrib/admin/utils.py | 308 | 365| 466 | 18624 | 66675 | 
| 63 | 26 django/contrib/admin/options.py | 1503 | 1534| 295 | 18919 | 66675 | 
| 64 | 26 django/contrib/admin/sites.py | 356 | 376| 157 | 19076 | 66675 | 
| 65 | 26 django/contrib/admin/utils.py | 121 | 156| 303 | 19379 | 66675 | 
| 66 | 26 django/contrib/admin/options.py | 1349 | 1414| 581 | 19960 | 66675 | 
| 67 | 27 django/contrib/admin/models.py | 1 | 20| 118 | 20078 | 67798 | 
| 68 | 27 django/contrib/admin/options.py | 1478 | 1501| 319 | 20397 | 67798 | 
| 69 | 28 django/template/base.py | 1 | 94| 779 | 21176 | 75676 | 
| 70 | 28 django/contrib/admin/options.py | 2082 | 2134| 451 | 21627 | 75676 | 
| 71 | 29 django/contrib/admin/__init__.py | 1 | 28| 257 | 21884 | 75933 | 
| 72 | 30 django/contrib/admin/helpers.py | 152 | 190| 343 | 22227 | 79124 | 
| 73 | 31 django/template/backends/dummy.py | 1 | 53| 325 | 22552 | 79449 | 
| 74 | 31 django/contrib/admin/options.py | 594 | 607| 122 | 22674 | 79449 | 
| 75 | 32 django/views/decorators/debug.py | 77 | 92| 132 | 22806 | 80038 | 
| 76 | 33 django/contrib/admindocs/__init__.py | 1 | 2| 0 | 22806 | 80053 | 
| 77 | 33 django/contrib/admin/options.py | 1836 | 1904| 584 | 23390 | 80053 | 
| 78 | 33 django/contrib/admin/options.py | 835 | 856| 182 | 23572 | 80053 | 
| 79 | 33 django/contrib/auth/admin.py | 40 | 99| 504 | 24076 | 80053 | 
| 80 | 34 django/utils/cache.py | 136 | 151| 188 | 24264 | 83801 | 
| 81 | 34 django/contrib/admin/sites.py | 334 | 354| 182 | 24446 | 83801 | 
| 82 | 35 django/contrib/admin/exceptions.py | 1 | 12| 0 | 24446 | 83868 | 
| 83 | 36 django/contrib/admindocs/urls.py | 1 | 51| 307 | 24753 | 84175 | 
| 84 | 37 django/utils/log.py | 1 | 75| 484 | 25237 | 85817 | 
| 85 | 37 django/contrib/admin/sites.py | 1 | 29| 175 | 25412 | 85817 | 
| 86 | 38 django/conf/global_settings.py | 147 | 262| 859 | 26271 | 91454 | 
| 87 | 38 django/contrib/admin/options.py | 609 | 631| 280 | 26551 | 91454 | 
| 88 | 38 django/contrib/admin/options.py | 98 | 128| 223 | 26774 | 91454 | 
| 89 | 39 django/contrib/admin/bin/compress.py | 1 | 53| 393 | 27167 | 91847 | 
| 90 | 39 django/template/response.py | 1 | 43| 383 | 27550 | 91847 | 
| 91 | 39 django/contrib/admin/options.py | 1416 | 1455| 309 | 27859 | 91847 | 
| 92 | 40 django/contrib/sites/admin.py | 1 | 9| 0 | 27859 | 91893 | 
| 93 | 40 django/contrib/admin/templatetags/log.py | 1 | 23| 161 | 28020 | 91893 | 
| 94 | 40 django/contrib/admin/options.py | 1170 | 1245| 664 | 28684 | 91893 | 
| 95 | 41 django/http/request.py | 1 | 41| 285 | 28969 | 97145 | 
| 96 | 42 django/contrib/admin/forms.py | 1 | 31| 185 | 29154 | 97330 | 
| 97 | 43 django/template/engine.py | 149 | 163| 138 | 29292 | 98640 | 
| 98 | 44 django/contrib/auth/forms.py | 403 | 455| 358 | 29650 | 101848 | 
| 99 | 44 django/contrib/admin/views/main.py | 48 | 119| 636 | 30286 | 101848 | 
| 100 | 44 django/views/decorators/debug.py | 1 | 44| 274 | 30560 | 101848 | 
| 101 | 45 django/core/mail/__init__.py | 90 | 104| 175 | 30735 | 102968 | 
| 102 | 46 django/core/checks/security/csrf.py | 1 | 41| 299 | 31034 | 103267 | 
| 103 | 46 django/views/csrf.py | 101 | 155| 577 | 31611 | 103267 | 
| 104 | 46 django/contrib/admin/options.py | 532 | 546| 169 | 31780 | 103267 | 
| 105 | 47 django/middleware/security.py | 34 | 59| 252 | 32032 | 103811 | 
| 106 | 48 django/core/checks/templates.py | 1 | 36| 259 | 32291 | 104071 | 
| 107 | 48 docs/_ext/djangodocs.py | 74 | 106| 255 | 32546 | 104071 | 
| 108 | 48 django/contrib/admin/options.py | 498 | 511| 165 | 32711 | 104071 | 
| 109 | 48 django/contrib/admin/views/main.py | 206 | 255| 423 | 33134 | 104071 | 
| 110 | 48 django/contrib/admin/sites.py | 378 | 410| 288 | 33422 | 104071 | 
| 111 | 48 django/template/engine.py | 1 | 53| 388 | 33810 | 104071 | 
| 112 | 48 django/contrib/admin/tests.py | 126 | 137| 153 | 33963 | 104071 | 
| 113 | 49 django/template/__init__.py | 1 | 69| 360 | 34323 | 104431 | 
| 114 | 50 django/core/management/templates.py | 58 | 118| 526 | 34849 | 107106 | 
| 115 | 50 django/contrib/admin/options.py | 1649 | 1663| 173 | 35022 | 107106 | 
| 116 | 50 django/contrib/admin/templatetags/base.py | 1 | 20| 173 | 35195 | 107106 | 
| 117 | 51 django/middleware/csrf.py | 205 | 330| 1222 | 36417 | 109992 | 
| 118 | 51 django/contrib/admin/helpers.py | 33 | 67| 230 | 36647 | 109992 | 
| 119 | 51 django/contrib/admin/templatetags/admin_list.py | 442 | 500| 343 | 36990 | 109992 | 
| 120 | 52 django/core/handlers/exception.py | 54 | 115| 499 | 37489 | 111013 | 
| 121 | 52 django/contrib/auth/admin.py | 191 | 206| 185 | 37674 | 111013 | 
| 122 | 53 django/contrib/sessions/middleware.py | 1 | 79| 623 | 38297 | 111637 | 
| 123 | 53 django/core/management/templates.py | 120 | 183| 560 | 38857 | 111637 | 
| 124 | 54 django/template/backends/utils.py | 1 | 15| 0 | 38857 | 111726 | 
| 125 | 54 django/contrib/admin/sites.py | 242 | 291| 472 | 39329 | 111726 | 
| 126 | 55 django/core/servers/basehttp.py | 159 | 178| 170 | 39499 | 113471 | 
| 127 | 56 django/utils/deprecation.py | 1 | 33| 209 | 39708 | 114519 | 
| 128 | 57 django/template/loader.py | 52 | 67| 117 | 39825 | 114935 | 
| 129 | 57 django/template/context.py | 170 | 210| 296 | 40121 | 114935 | 
| 130 | 57 django/contrib/admin/helpers.py | 355 | 364| 134 | 40255 | 114935 | 
| 131 | **57 django/contrib/admin/checks.py** | 130 | 147| 155 | 40410 | 114935 | 
| 132 | 58 django/contrib/auth/middleware.py | 46 | 82| 360 | 40770 | 115929 | 
| 133 | 59 django/contrib/auth/views.py | 208 | 222| 133 | 40903 | 118593 | 
| 134 | 59 django/views/decorators/debug.py | 47 | 75| 199 | 41102 | 118593 | 
| 135 | 60 django/middleware/common.py | 34 | 61| 257 | 41359 | 120104 | 
| 136 | 60 django/contrib/admin/options.py | 474 | 496| 241 | 41600 | 120104 | 
| 137 | 60 django/contrib/auth/middleware.py | 1 | 23| 171 | 41771 | 120104 | 
| 138 | 61 django/template/defaulttags.py | 251 | 261| 140 | 41911 | 131247 | 
| 139 | 61 django/contrib/admin/options.py | 1033 | 1055| 198 | 42109 | 131247 | 
| 140 | 62 django/contrib/admindocs/middleware.py | 1 | 29| 234 | 42343 | 131482 | 
| 141 | 62 django/contrib/auth/views.py | 330 | 362| 239 | 42582 | 131482 | 
| 142 | 63 django/template/backends/jinja2.py | 54 | 88| 232 | 42814 | 132304 | 
| 143 | 64 django/template/loader_tags.py | 276 | 322| 392 | 43206 | 134883 | 
| 144 | 64 django/utils/deprecation.py | 88 | 142| 434 | 43640 | 134883 | 
| 145 | 64 django/contrib/admin/options.py | 513 | 530| 200 | 43840 | 134883 | 
| 146 | 65 django/contrib/admin/filters.py | 1 | 17| 127 | 43967 | 138976 | 
| 147 | 65 django/contrib/admin/options.py | 1111 | 1121| 125 | 44092 | 138976 | 
| 148 | 65 django/contrib/admin/options.py | 549 | 592| 297 | 44389 | 138976 | 
| 149 | 66 django/views/debug.py | 150 | 173| 177 | 44566 | 143369 | 
| 150 | 66 django/contrib/admin/options.py | 429 | 472| 350 | 44916 | 143369 | 
| 151 | 66 django/views/defaults.py | 27 | 76| 401 | 45317 | 143369 | 
| 152 | **66 django/contrib/admin/checks.py** | 622 | 641| 183 | 45500 | 143369 | 
| 153 | 67 django/core/checks/security/base.py | 1 | 83| 732 | 46232 | 145150 | 
| 154 | 68 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 46232 | 145226 | 
| 155 | 68 django/contrib/admin/views/main.py | 121 | 204| 827 | 47059 | 145226 | 
| 156 | 68 django/contrib/admin/options.py | 652 | 666| 136 | 47195 | 145226 | 
| 157 | 69 django/template/library.py | 201 | 234| 304 | 47499 | 147763 | 
| 158 | 69 django/contrib/admin/options.py | 716 | 749| 245 | 47744 | 147763 | 
| 159 | 70 django/conf/urls/__init__.py | 1 | 22| 146 | 47890 | 147909 | 
| 160 | 71 django/contrib/gis/views.py | 1 | 21| 155 | 48045 | 148064 | 
| 161 | 71 django/contrib/admin/tests.py | 157 | 171| 160 | 48205 | 148064 | 
| 162 | 72 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 48615 | 149089 | 
| 163 | 72 django/template/base.py | 816 | 881| 540 | 49155 | 149089 | 
| 164 | 72 django/template/engine.py | 81 | 147| 457 | 49612 | 149089 | 
| 165 | 73 django/contrib/gis/db/models/functions.py | 193 | 251| 395 | 50007 | 153016 | 
| 166 | 73 django/contrib/admin/options.py | 803 | 833| 216 | 50223 | 153016 | 
| 167 | 73 django/views/defaults.py | 79 | 97| 129 | 50352 | 153016 | 
| 168 | 73 django/conf/global_settings.py | 497 | 642| 923 | 51275 | 153016 | 
| 169 | 73 django/contrib/admin/options.py | 284 | 373| 641 | 51916 | 153016 | 
| 170 | 74 django/views/i18n.py | 88 | 191| 711 | 52627 | 155562 | 
| 171 | 74 django/middleware/csrf.py | 122 | 156| 267 | 52894 | 155562 | 
| 172 | 74 django/template/context.py | 27 | 130| 655 | 53549 | 155562 | 
| 173 | 74 django/contrib/admin/options.py | 2024 | 2045| 221 | 53770 | 155562 | 
| 174 | 74 django/views/debug.py | 189 | 237| 467 | 54237 | 155562 | 
| 175 | 75 django/views/decorators/http.py | 77 | 122| 347 | 54584 | 156514 | 
| 176 | 75 django/contrib/auth/admin.py | 25 | 37| 128 | 54712 | 156514 | 
| 177 | 76 django/contrib/admindocs/apps.py | 1 | 8| 0 | 54712 | 156556 | 
| 178 | 77 django/contrib/admin/apps.py | 1 | 25| 148 | 54860 | 156704 | 
| 179 | 78 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 55180 | 157024 | 
| 180 | 78 django/contrib/admin/helpers.py | 366 | 382| 138 | 55318 | 157024 | 
| 181 | 78 django/template/base.py | 792 | 814| 190 | 55508 | 157024 | 
| 182 | 78 django/contrib/admin/options.py | 2136 | 2161| 250 | 55758 | 157024 | 
| 183 | 78 django/contrib/admin/models.py | 23 | 36| 111 | 55869 | 157024 | 
| 184 | 78 django/core/checks/security/base.py | 85 | 180| 710 | 56579 | 157024 | 
| 185 | 78 django/utils/cache.py | 154 | 191| 447 | 57026 | 157024 | 
| 186 | 78 django/middleware/security.py | 1 | 32| 316 | 57342 | 157024 | 
| 187 | 78 django/middleware/csrf.py | 93 | 119| 224 | 57566 | 157024 | 
| 188 | 78 django/contrib/admin/helpers.py | 1 | 30| 196 | 57762 | 157024 | 
| 189 | 78 django/contrib/admin/options.py | 1020 | 1031| 149 | 57911 | 157024 | 
| 190 | 78 django/contrib/admin/views/main.py | 434 | 477| 390 | 58301 | 157024 | 
| 191 | 79 django/template/backends/django.py | 79 | 111| 225 | 58526 | 157880 | 
| 192 | 79 django/contrib/admin/options.py | 633 | 650| 136 | 58662 | 157880 | 
| 193 | 79 django/contrib/admin/helpers.py | 335 | 353| 181 | 58843 | 157880 | 


### Hint

```
​https://github.com/django/django/pull/12906
Yes, good point. Thanks Jon.
Suggestion from Mariusz on the PR, to the system check an error but only if the sidebar is enabled.
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -59,6 +59,7 @@ def check_dependencies(**kwargs):
     """
     Check that the admin's dependencies are correctly installed.
     """
+    from django.contrib.admin.sites import all_sites
     if not apps.is_installed('django.contrib.admin'):
         return []
     errors = []
@@ -105,6 +106,15 @@ def check_dependencies(**kwargs):
                 "the admin application.",
                 id='admin.E404',
             ))
+        sidebar_enabled = any(site.enable_nav_sidebar for site in all_sites)
+        if (sidebar_enabled and 'django.template.context_processors.request'
+                not in django_templates_instance.context_processors):
+            errors.append(checks.Warning(
+                "'django.template.context_processors.request' must be enabled "
+                "in DjangoTemplates (TEMPLATES) in order to use the admin "
+                "navigation sidebar.",
+                id='admin.W411',
+            ))
 
     if not _contains_subclass('django.contrib.auth.middleware.AuthenticationMiddleware', settings.MIDDLEWARE):
         errors.append(checks.Error(

```

## Test Patch

```diff
diff --git a/tests/admin_checks/tests.py b/tests/admin_checks/tests.py
--- a/tests/admin_checks/tests.py
+++ b/tests/admin_checks/tests.py
@@ -134,6 +134,12 @@ def test_context_processor_dependencies(self):
                 "be enabled in DjangoTemplates (TEMPLATES) in order to use "
                 "the admin application.",
                 id='admin.E404',
+            ),
+            checks.Warning(
+                "'django.template.context_processors.request' must be enabled "
+                "in DjangoTemplates (TEMPLATES) in order to use the admin "
+                "navigation sidebar.",
+                id='admin.W411',
             )
         ]
         self.assertEqual(admin.checks.check_dependencies(), expected)
@@ -150,7 +156,10 @@ def test_context_processor_dependencies(self):
             'DIRS': [],
             'APP_DIRS': True,
             'OPTIONS': {
-                'context_processors': ['django.contrib.messages.context_processors.messages'],
+                'context_processors': [
+                    'django.template.context_processors.request',
+                    'django.contrib.messages.context_processors.messages',
+                ],
             },
         }],
     )
@@ -177,6 +186,7 @@ def test_context_processor_dependencies_model_backend_subclass(self):
                 'APP_DIRS': True,
                 'OPTIONS': {
                     'context_processors': [
+                        'django.template.context_processors.request',
                         'django.contrib.auth.context_processors.auth',
                         'django.contrib.messages.context_processors.messages',
                     ],
diff --git a/tests/admin_scripts/tests.py b/tests/admin_scripts/tests.py
--- a/tests/admin_scripts/tests.py
+++ b/tests/admin_scripts/tests.py
@@ -1124,6 +1124,7 @@ def test_complex_app(self):
                         'APP_DIRS': True,
                         'OPTIONS': {
                             'context_processors': [
+                                'django.template.context_processors.request',
                                 'django.contrib.auth.context_processors.auth',
                                 'django.contrib.messages.context_processors.messages',
                             ],
diff --git a/tests/admin_views/test_nav_sidebar.py b/tests/admin_views/test_nav_sidebar.py
--- a/tests/admin_views/test_nav_sidebar.py
+++ b/tests/admin_views/test_nav_sidebar.py
@@ -51,9 +51,31 @@ def test_sidebar_unauthenticated(self):
         self.assertNotContains(response, '<nav class="sticky" id="nav-sidebar">')
 
     def test_sidebar_aria_current_page(self):
-        response = self.client.get(reverse('test_with_sidebar:auth_user_changelist'))
+        url = reverse('test_with_sidebar:auth_user_changelist')
+        response = self.client.get(url)
         self.assertContains(response, '<nav class="sticky" id="nav-sidebar">')
-        self.assertContains(response, 'aria-current="page">Users</a>')
+        self.assertContains(response, '<a href="%s" aria-current="page">Users</a>' % url)
+
+    @override_settings(
+        TEMPLATES=[{
+            'BACKEND': 'django.template.backends.django.DjangoTemplates',
+            'DIRS': [],
+            'APP_DIRS': True,
+            'OPTIONS': {
+                'context_processors': [
+                    'django.contrib.auth.context_processors.auth',
+                    'django.contrib.messages.context_processors.messages',
+                ],
+            },
+        }]
+    )
+    def test_sidebar_aria_current_page_missing_without_request_context_processor(self):
+        url = reverse('test_with_sidebar:auth_user_changelist')
+        response = self.client.get(url)
+        self.assertContains(response, '<nav class="sticky" id="nav-sidebar">')
+        # Does not include aria-current attribute.
+        self.assertContains(response, '<a href="%s">Users</a>' % url)
+        self.assertNotContains(response, 'aria-current')
 
 
 @override_settings(ROOT_URLCONF='admin_views.test_nav_sidebar')
diff --git a/tests/admin_views/tests.py b/tests/admin_views/tests.py
--- a/tests/admin_views/tests.py
+++ b/tests/admin_views/tests.py
@@ -1425,6 +1425,7 @@ def get_perm(Model, codename):
         'APP_DIRS': True,
         'OPTIONS': {
             'context_processors': [
+                'django.template.context_processors.request',
                 'django.contrib.auth.context_processors.auth',
                 'django.contrib.messages.context_processors.messages',
             ],
@@ -2424,6 +2425,7 @@ def test_post_save_message_no_forbidden_links_visible(self):
         'APP_DIRS': True,
         'OPTIONS': {
             'context_processors': [
+                'django.template.context_processors.request',
                 'django.contrib.auth.context_processors.auth',
                 'django.contrib.messages.context_processors.messages',
             ],
diff --git a/tests/auth_tests/settings.py b/tests/auth_tests/settings.py
--- a/tests/auth_tests/settings.py
+++ b/tests/auth_tests/settings.py
@@ -11,6 +11,7 @@
     'APP_DIRS': True,
     'OPTIONS': {
         'context_processors': [
+            'django.template.context_processors.request',
             'django.contrib.auth.context_processors.auth',
             'django.contrib.messages.context_processors.messages',
         ],

```


## Code snippets

### 1 - django/contrib/admindocs/views.py:

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
### 2 - django/template/context.py:

Start line: 233, End line: 260

```python
class RequestContext(Context):

    @contextmanager
    def bind_template(self, template):
        if self.template is not None:
            raise RuntimeError("Context is already bound to a template")

        self.template = template
        # Set context processors according to the template engine's settings.
        processors = (template.engine.template_context_processors +
                      self._processors)
        updates = {}
        for processor in processors:
            updates.update(processor(self.request))
        self.dicts[self._processors_index] = updates

        try:
            yield
        finally:
            self.template = None
            # Unset context processors.
            self.dicts[self._processors_index] = {}

    def new(self, values=None):
        new_context = super().new(values)
        # This is for backwards-compatibility: RequestContexts created via
        # Context.new don't include values from context processors.
        if hasattr(new_context, '_processors_index'):
            del new_context._processors_index
        return new_context
```
### 3 - django/template/context_processors.py:

Start line: 35, End line: 49

```python
def debug(request):
    """
    Return context variables helpful for debugging.
    """
    context_extras = {}
    if settings.DEBUG and request.META.get('REMOTE_ADDR') in settings.INTERNAL_IPS:
        context_extras['debug'] = True
        from django.db import connections
        # Return a lazy reference that computes connection.queries on access,
        # to ensure it contains queries triggered after this function runs.
        context_extras['sql_queries'] = lazy(
            lambda: list(itertools.chain.from_iterable(connections[x].queries for x in connections)),
            list
        )
    return context_extras
```
### 4 - django/contrib/admin/options.py:

Start line: 1123, End line: 1168

```python
class ModelAdmin(BaseModelAdmin):

    def render_change_form(self, request, context, add=False, change=False, form_url='', obj=None):
        opts = self.model._meta
        app_label = opts.app_label
        preserved_filters = self.get_preserved_filters(request)
        form_url = add_preserved_filters({'preserved_filters': preserved_filters, 'opts': opts}, form_url)
        view_on_site_url = self.get_view_on_site_url(obj)
        has_editable_inline_admin_formsets = False
        for inline in context['inline_admin_formsets']:
            if inline.has_add_permission or inline.has_change_permission or inline.has_delete_permission:
                has_editable_inline_admin_formsets = True
                break
        context.update({
            'add': add,
            'change': change,
            'has_view_permission': self.has_view_permission(request, obj),
            'has_add_permission': self.has_add_permission(request),
            'has_change_permission': self.has_change_permission(request, obj),
            'has_delete_permission': self.has_delete_permission(request, obj),
            'has_editable_inline_admin_formsets': has_editable_inline_admin_formsets,
            'has_file_field': context['adminform'].form.is_multipart() or any(
                admin_formset.formset.is_multipart()
                for admin_formset in context['inline_admin_formsets']
            ),
            'has_absolute_url': view_on_site_url is not None,
            'absolute_url': view_on_site_url,
            'form_url': form_url,
            'opts': opts,
            'content_type_id': get_content_type_for_model(self.model).pk,
            'save_as': self.save_as,
            'save_on_top': self.save_on_top,
            'to_field_var': TO_FIELD_VAR,
            'is_popup_var': IS_POPUP_VAR,
            'app_label': app_label,
        })
        if add and self.add_form_template is not None:
            form_template = self.add_form_template
        else:
            form_template = self.change_form_template

        request.current_app = self.admin_site.name

        return TemplateResponse(request, form_template or [
            "admin/%s/%s/change_form.html" % (app_label, opts.model_name),
            "admin/%s/change_form.html" % app_label,
            "admin/change_form.html"
        ], context)
```
### 5 - django/contrib/admin/options.py:

Start line: 1623, End line: 1647

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        # ... other code
        context = {
            **self.admin_site.each_context(request),
            'title': title % opts.verbose_name,
            'adminform': adminForm,
            'object_id': object_id,
            'original': obj,
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'to_field': to_field,
            'media': media,
            'inline_admin_formsets': inline_formsets,
            'errors': helpers.AdminErrorList(form, formsets),
            'preserved_filters': self.get_preserved_filters(request),
        }

        # Hide the "Save" and "Save and continue" buttons if "Save as New" was
        # previously chosen to prevent the interface from getting confusing.
        if request.method == 'POST' and not form_validated and "_saveasnew" in request.POST:
            context['show_save'] = False
            context['show_save_and_continue'] = False
            # Use the change template instead of the add template.
            add = False

        context.update(extra_context or {})

        return self.render_change_form(request, context, add=add, change=not add, obj=obj, form_url=form_url)
```
### 6 - django/contrib/auth/admin.py:

Start line: 1, End line: 22

```python
from django.conf import settings
from django.contrib import admin, messages
from django.contrib.admin.options import IS_POPUP_VAR
from django.contrib.admin.utils import unquote
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import (
    AdminPasswordChangeForm, UserChangeForm, UserCreationForm,
)
from django.contrib.auth.models import Group, User
from django.core.exceptions import PermissionDenied
from django.db import router, transaction
from django.http import Http404, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.decorators import method_decorator
from django.utils.html import escape
from django.utils.translation import gettext, gettext_lazy as _
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters

csrf_protect_m = method_decorator(csrf_protect)
sensitive_post_parameters_m = method_decorator(sensitive_post_parameters())
```
### 7 - django/contrib/admin/sites.py:

Start line: 293, End line: 315

```python
class AdminSite:

    @property
    def urls(self):
        return self.get_urls(), 'admin', self.name

    def each_context(self, request):
        """
        Return a dictionary of variables to put in the template context for
        *every* page in the admin site.

        For sites running on a subpath, use the SCRIPT_NAME value if site_url
        hasn't been customized.
        """
        script_name = request.META['SCRIPT_NAME']
        site_url = script_name if self.site_url == '/' and script_name else self.site_url
        return {
            'site_title': self.site_title,
            'site_header': self.site_header,
            'site_url': site_url,
            'has_permission': self.has_permission(request),
            'available_apps': self.get_app_list(request),
            'is_popup': False,
            'is_nav_sidebar_enabled': self.enable_nav_sidebar,
        }
```
### 8 - django/template/context_processors.py:

Start line: 1, End line: 32

```python
"""
A set of request processors that return dictionaries to be merged into a
template context. Each function takes the request object as its only parameter
and returns a dictionary to add to the context.

These are referenced from the 'context_processors' option of the configuration
of a DjangoTemplates backend and used by RequestContext.
"""

import itertools

from django.conf import settings
from django.middleware.csrf import get_token
from django.utils.functional import SimpleLazyObject, lazy


def csrf(request):
    """
    Context processor that provides a CSRF token, or the string 'NOTPROVIDED' if
    it has not been provided by either a view decorator or the middleware
    """
    def _get_val():
        token = get_token(request)
        if token is None:
            # In order to be able to provide debugging info in the
            # case of misconfiguration, we use a sentinel value
            # instead of returning an empty dict.
            return 'NOTPROVIDED'
        else:
            return token

    return {'csrf_token': SimpleLazyObject(_get_val)}
```
### 9 - django/template/context_processors.py:

Start line: 52, End line: 82

```python
def i18n(request):
    from django.utils import translation
    return {
        'LANGUAGES': settings.LANGUAGES,
        'LANGUAGE_CODE': translation.get_language(),
        'LANGUAGE_BIDI': translation.get_language_bidi(),
    }


def tz(request):
    from django.utils import timezone
    return {'TIME_ZONE': timezone.get_current_timezone_name()}


def static(request):
    """
    Add static-related context variables to the context.
    """
    return {'STATIC_URL': settings.STATIC_URL}


def media(request):
    """
    Add media-related context variables to the context.
    """
    return {'MEDIA_URL': settings.MEDIA_URL}


def request(request):
    return {'request': request}
```
### 10 - django/contrib/admindocs/views.py:

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
### 41 - django/contrib/admin/checks.py:

Start line: 58, End line: 127

```python
def check_dependencies(**kwargs):
    """
    Check that the admin's dependencies are correctly installed.
    """
    if not apps.is_installed('django.contrib.admin'):
        return []
    errors = []
    app_dependencies = (
        ('django.contrib.contenttypes', 401),
        ('django.contrib.auth', 405),
        ('django.contrib.messages', 406),
    )
    for app_name, error_code in app_dependencies:
        if not apps.is_installed(app_name):
            errors.append(checks.Error(
                "'%s' must be in INSTALLED_APPS in order to use the admin "
                "application." % app_name,
                id='admin.E%d' % error_code,
            ))
    for engine in engines.all():
        if isinstance(engine, DjangoTemplates):
            django_templates_instance = engine.engine
            break
    else:
        django_templates_instance = None
    if not django_templates_instance:
        errors.append(checks.Error(
            "A 'django.template.backends.django.DjangoTemplates' instance "
            "must be configured in TEMPLATES in order to use the admin "
            "application.",
            id='admin.E403',
        ))
    else:
        if ('django.contrib.auth.context_processors.auth'
                not in django_templates_instance.context_processors and
                _contains_subclass('django.contrib.auth.backends.ModelBackend', settings.AUTHENTICATION_BACKENDS)):
            errors.append(checks.Error(
                "'django.contrib.auth.context_processors.auth' must be "
                "enabled in DjangoTemplates (TEMPLATES) if using the default "
                "auth backend in order to use the admin application.",
                id='admin.E402',
            ))
        if ('django.contrib.messages.context_processors.messages'
                not in django_templates_instance.context_processors):
            errors.append(checks.Error(
                "'django.contrib.messages.context_processors.messages' must "
                "be enabled in DjangoTemplates (TEMPLATES) in order to use "
                "the admin application.",
                id='admin.E404',
            ))

    if not _contains_subclass('django.contrib.auth.middleware.AuthenticationMiddleware', settings.MIDDLEWARE):
        errors.append(checks.Error(
            "'django.contrib.auth.middleware.AuthenticationMiddleware' must "
            "be in MIDDLEWARE in order to use the admin application.",
            id='admin.E408',
        ))
    if not _contains_subclass('django.contrib.messages.middleware.MessageMiddleware', settings.MIDDLEWARE):
        errors.append(checks.Error(
            "'django.contrib.messages.middleware.MessageMiddleware' must "
            "be in MIDDLEWARE in order to use the admin application.",
            id='admin.E409',
        ))
    if not _contains_subclass('django.contrib.sessions.middleware.SessionMiddleware', settings.MIDDLEWARE):
        errors.append(checks.Error(
            "'django.contrib.sessions.middleware.SessionMiddleware' must "
            "be in MIDDLEWARE in order to use the admin application.",
            id='admin.E410',
        ))
    return errors
```
### 131 - django/contrib/admin/checks.py:

Start line: 130, End line: 147

```python
class BaseModelAdminChecks:

    def check(self, admin_obj, **kwargs):
        return [
            *self._check_autocomplete_fields(admin_obj),
            *self._check_raw_id_fields(admin_obj),
            *self._check_fields(admin_obj),
            *self._check_fieldsets(admin_obj),
            *self._check_exclude(admin_obj),
            *self._check_form(admin_obj),
            *self._check_filter_vertical(admin_obj),
            *self._check_filter_horizontal(admin_obj),
            *self._check_radio_fields(admin_obj),
            *self._check_prepopulated_fields(admin_obj),
            *self._check_view_on_site_url(admin_obj),
            *self._check_ordering(admin_obj),
            *self._check_readonly_fields(admin_obj),
        ]
```
### 152 - django/contrib/admin/checks.py:

Start line: 622, End line: 641

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def check(self, admin_obj, **kwargs):
        return [
            *super().check(admin_obj),
            *self._check_save_as(admin_obj),
            *self._check_save_on_top(admin_obj),
            *self._check_inlines(admin_obj),
            *self._check_list_display(admin_obj),
            *self._check_list_display_links(admin_obj),
            *self._check_list_filter(admin_obj),
            *self._check_list_select_related(admin_obj),
            *self._check_list_per_page(admin_obj),
            *self._check_list_max_show_all(admin_obj),
            *self._check_list_editable(admin_obj),
            *self._check_search_fields(admin_obj),
            *self._check_date_hierarchy(admin_obj),
            *self._check_action_permission_methods(admin_obj),
            *self._check_actions_uniqueness(admin_obj),
        ]
```
