# django__django-15790

| **django/django** | `c627226d05dd52aef59447dcfb29cec2c2b11b8a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 166 |
| **Any found context length** | 166 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/checks/templates.py b/django/core/checks/templates.py
--- a/django/core/checks/templates.py
+++ b/django/core/checks/templates.py
@@ -50,15 +50,15 @@ def check_string_if_invalid_is_string(app_configs, **kwargs):
 @register(Tags.templates)
 def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
     errors = []
-    libraries = defaultdict(list)
+    libraries = defaultdict(set)
 
     for conf in settings.TEMPLATES:
         custom_libraries = conf.get("OPTIONS", {}).get("libraries", {})
         for module_name, module_path in custom_libraries.items():
-            libraries[module_name].append(module_path)
+            libraries[module_name].add(module_path)
 
     for module_name, module_path in get_template_tag_modules():
-        libraries[module_name].append(module_path)
+        libraries[module_name].add(module_path)
 
     for library_name, items in libraries.items():
         if len(items) > 1:
@@ -66,7 +66,7 @@ def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
                 Error(
                     E003.msg.format(
                         repr(library_name),
-                        ", ".join(repr(item) for item in items),
+                        ", ".join(repr(item) for item in sorted(items)),
                     ),
                     id=E003.id,
                 )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/checks/templates.py | 53 | 61 | 1 | 1 | 166
| django/core/checks/templates.py | 69 | 69 | 1 | 1 | 166


## Problem Statement

```
check_for_template_tags_with_the_same_name with libraries in TEMPLATES
Description
	
I didn't explore this thoroughly, but I think there might be an issue with the check_for_template_tags_with_the_same_name when you add a template tag library into TEMPLATES['OPTIONS']['librairies'].
I'm getting an error like: 
(templates.E003) 'my_tags' is used for multiple template tag modules: 'someapp.templatetags.my_tags', 'someapp.templatetags.my_tags'

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/checks/templates.py** | 50 | 76| 166 | 166 | 479 | 
| 2 | **1 django/core/checks/templates.py** | 1 | 47| 311 | 477 | 479 | 
| 3 | 2 django/template/defaulttags.py | 1025 | 1059| 199 | 676 | 11252 | 
| 4 | 3 django/template/library.py | 1 | 54| 338 | 1014 | 13804 | 
| 5 | 3 django/template/defaulttags.py | 1062 | 1091| 228 | 1242 | 13804 | 
| 6 | 3 django/template/defaulttags.py | 1 | 64| 338 | 1580 | 13804 | 
| 7 | 3 django/template/defaulttags.py | 869 | 889| 133 | 1713 | 13804 | 
| 8 | 3 django/template/library.py | 99 | 154| 318 | 2031 | 13804 | 
| 9 | 4 django/template/backends/django.py | 110 | 138| 208 | 2239 | 14708 | 
| 10 | 4 django/template/backends/django.py | 1 | 45| 303 | 2542 | 14708 | 
| 11 | 4 django/template/defaulttags.py | 892 | 980| 694 | 3236 | 14708 | 
| 12 | 4 django/template/library.py | 281 | 366| 648 | 3884 | 14708 | 
| 13 | 5 django/core/checks/registry.py | 1 | 25| 130 | 4014 | 15386 | 
| 14 | 5 django/template/library.py | 369 | 386| 110 | 4124 | 15386 | 
| 15 | 5 django/template/defaulttags.py | 983 | 1022| 364 | 4488 | 15386 | 
| 16 | 6 django/contrib/admin/templatetags/admin_list.py | 176 | 192| 140 | 4628 | 19182 | 
| 17 | 6 django/template/backends/django.py | 80 | 107| 176 | 4804 | 19182 | 
| 18 | 6 django/template/defaulttags.py | 418 | 434| 121 | 4925 | 19182 | 
| 19 | 6 django/contrib/admin/templatetags/admin_list.py | 1 | 33| 193 | 5118 | 19182 | 
| 20 | 6 django/template/defaulttags.py | 295 | 324| 171 | 5289 | 19182 | 
| 21 | 7 django/template/utils.py | 1 | 65| 407 | 5696 | 19898 | 
| 22 | 7 django/template/library.py | 228 | 243| 126 | 5822 | 19898 | 
| 23 | 7 django/template/defaulttags.py | 248 | 278| 263 | 6085 | 19898 | 
| 24 | 7 django/template/library.py | 156 | 204| 230 | 6315 | 19898 | 
| 25 | 8 django/templatetags/cache.py | 1 | 55| 426 | 6741 | 20640 | 
| 26 | 8 django/template/defaulttags.py | 183 | 245| 532 | 7273 | 20640 | 
| 27 | 9 django/contrib/admin/templatetags/base.py | 1 | 30| 186 | 7459 | 20955 | 
| 28 | 9 django/template/defaulttags.py | 527 | 542| 142 | 7601 | 20955 | 
| 29 | 9 django/template/defaulttags.py | 280 | 292| 147 | 7748 | 20955 | 
| 30 | 9 django/contrib/admin/templatetags/base.py | 32 | 46| 138 | 7886 | 20955 | 
| 31 | 9 django/template/defaulttags.py | 159 | 181| 186 | 8072 | 20955 | 
| 32 | 10 django/template/base.py | 574 | 606| 207 | 8279 | 29228 | 
| 33 | 11 django/contrib/admindocs/views.py | 65 | 99| 297 | 8576 | 32710 | 
| 34 | 11 django/template/defaulttags.py | 1289 | 1323| 293 | 8869 | 32710 | 
| 35 | 11 django/template/base.py | 1 | 91| 754 | 9623 | 32710 | 
| 36 | 11 django/template/defaulttags.py | 1395 | 1415| 121 | 9744 | 32710 | 
| 37 | 12 django/templatetags/i18n.py | 568 | 596| 216 | 9960 | 36892 | 
| 38 | 13 django/core/checks/model_checks.py | 1 | 90| 671 | 10631 | 38702 | 
| 39 | 14 django/template/loader_tags.py | 1 | 39| 207 | 10838 | 41414 | 
| 40 | 14 django/template/defaulttags.py | 327 | 345| 151 | 10989 | 41414 | 
| 41 | 14 django/template/loader_tags.py | 277 | 298| 243 | 11232 | 41414 | 
| 42 | 15 django/core/management/templates.py | 256 | 289| 240 | 11472 | 44416 | 
| 43 | 15 django/template/defaulttags.py | 572 | 662| 812 | 12284 | 44416 | 
| 44 | 16 django/core/checks/__init__.py | 1 | 48| 327 | 12611 | 44743 | 
| 45 | 16 django/template/base.py | 436 | 456| 156 | 12767 | 44743 | 
| 46 | 16 django/template/loader_tags.py | 301 | 353| 402 | 13169 | 44743 | 
| 47 | 17 django/template/__init__.py | 1 | 76| 394 | 13563 | 45137 | 
| 48 | 18 django/template/engine.py | 1 | 63| 397 | 13960 | 46635 | 
| 49 | 18 django/templatetags/i18n.py | 1 | 32| 197 | 14157 | 46635 | 
| 50 | 18 django/template/defaulttags.py | 455 | 483| 225 | 14382 | 46635 | 
| 51 | 18 django/template/defaulttags.py | 117 | 156| 236 | 14618 | 46635 | 
| 52 | 19 django/core/checks/security/base.py | 242 | 256| 121 | 14739 | 48824 | 
| 53 | 19 django/template/loader_tags.py | 114 | 129| 155 | 14894 | 48824 | 
| 54 | 19 django/template/utils.py | 67 | 94| 197 | 15091 | 48824 | 
| 55 | 19 django/template/defaulttags.py | 1460 | 1495| 250 | 15341 | 48824 | 
| 56 | 19 django/template/library.py | 246 | 278| 304 | 15645 | 48824 | 
| 57 | 19 django/template/defaulttags.py | 437 | 453| 117 | 15762 | 48824 | 
| 58 | 19 django/template/defaulttags.py | 545 | 569| 191 | 15953 | 48824 | 
| 59 | 19 django/template/defaulttags.py | 665 | 716| 339 | 16292 | 48824 | 
| 60 | 19 django/contrib/admin/templatetags/admin_list.py | 463 | 534| 372 | 16664 | 48824 | 
| 61 | 19 django/template/base.py | 521 | 541| 181 | 16845 | 48824 | 
| 62 | 19 django/template/library.py | 207 | 225| 157 | 17002 | 48824 | 
| 63 | 20 django/core/checks/urls.py | 31 | 54| 168 | 17170 | 49530 | 
| 64 | 20 django/template/loader_tags.py | 83 | 112| 227 | 17397 | 49530 | 
| 65 | 20 django/template/library.py | 56 | 97| 325 | 17722 | 49530 | 
| 66 | 20 django/core/checks/model_checks.py | 187 | 228| 345 | 18067 | 49530 | 
| 67 | 20 django/template/defaulttags.py | 376 | 415| 218 | 18285 | 49530 | 
| 68 | 21 django/core/checks/translation.py | 1 | 67| 449 | 18734 | 49979 | 
| 69 | 21 django/core/checks/security/base.py | 183 | 223| 306 | 19040 | 49979 | 
| 70 | 21 django/core/checks/model_checks.py | 161 | 185| 267 | 19307 | 49979 | 
| 71 | 22 django/templatetags/static.py | 95 | 131| 242 | 19549 | 50998 | 
| 72 | 22 django/core/management/templates.py | 229 | 254| 212 | 19761 | 50998 | 
| 73 | 23 django/templatetags/l10n.py | 1 | 38| 251 | 20012 | 51440 | 
| 74 | 23 django/template/defaulttags.py | 1326 | 1392| 508 | 20520 | 51440 | 
| 75 | 24 django/core/checks/security/sessions.py | 1 | 100| 580 | 21100 | 52021 | 
| 76 | 25 django/contrib/admin/templatetags/log.py | 1 | 25| 165 | 21265 | 52517 | 
| 77 | 26 django/core/checks/caches.py | 1 | 19| 119 | 21384 | 53044 | 
| 78 | 26 django/template/defaulttags.py | 1232 | 1255| 176 | 21560 | 53044 | 
| 79 | 27 django/template/backends/dummy.py | 1 | 53| 327 | 21887 | 53371 | 
| 80 | 27 django/template/defaulttags.py | 486 | 524| 252 | 22139 | 53371 | 
| 81 | 27 django/template/base.py | 543 | 572| 245 | 22384 | 53371 | 
| 82 | 27 django/core/checks/caches.py | 22 | 58| 294 | 22678 | 53371 | 
| 83 | 28 django/core/checks/compatibility/django_4_0.py | 1 | 21| 142 | 22820 | 53514 | 
| 84 | 28 django/template/defaulttags.py | 67 | 89| 164 | 22984 | 53514 | 
| 85 | 29 django/core/checks/security/csrf.py | 1 | 42| 305 | 23289 | 53979 | 
| 86 | 30 django/utils/cache.py | 213 | 231| 184 | 23473 | 57772 | 
| 87 | 31 django/template/loaders/app_directories.py | 1 | 14| 0 | 23473 | 57831 | 
| 88 | 32 django/template/backends/jinja2.py | 1 | 52| 344 | 23817 | 58647 | 
| 89 | 33 django/templatetags/tz.py | 116 | 161| 246 | 24063 | 59999 | 
| 90 | 34 django/contrib/contenttypes/checks.py | 28 | 47| 129 | 24192 | 60260 | 
| 91 | 34 django/core/management/templates.py | 153 | 227| 609 | 24801 | 60260 | 
| 92 | 35 django/contrib/admin/templatetags/admin_modify.py | 115 | 130| 103 | 24904 | 61304 | 
| 93 | 35 django/core/management/templates.py | 42 | 80| 265 | 25169 | 61304 | 
| 94 | 35 django/template/engine.py | 112 | 179| 459 | 25628 | 61304 | 
| 95 | 36 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 26047 | 61723 | 
| 96 | 36 django/core/checks/security/csrf.py | 45 | 68| 159 | 26206 | 61723 | 
| 97 | 36 django/template/defaulttags.py | 1094 | 1138| 370 | 26576 | 61723 | 
| 98 | 36 django/core/checks/security/base.py | 81 | 180| 732 | 27308 | 61723 | 
| 99 | 36 django/template/base.py | 94 | 135| 228 | 27536 | 61723 | 
| 100 | 36 django/template/base.py | 179 | 204| 154 | 27690 | 61723 | 
| 101 | 36 django/template/defaulttags.py | 1164 | 1229| 647 | 28337 | 61723 | 
| 102 | 36 django/contrib/contenttypes/checks.py | 1 | 25| 130 | 28467 | 61723 | 
| 103 | 36 django/template/base.py | 749 | 773| 210 | 28677 | 61723 | 
| 104 | 37 django/template/loader.py | 1 | 49| 299 | 28976 | 62139 | 
| 105 | 37 django/core/management/templates.py | 82 | 151| 579 | 29555 | 62139 | 
| 106 | 37 django/template/engine.py | 197 | 213| 134 | 29689 | 62139 | 
| 107 | 38 django/contrib/admin/checks.py | 955 | 979| 197 | 29886 | 71671 | 
| 108 | 38 django/core/checks/caches.py | 61 | 77| 112 | 29998 | 71671 | 
| 109 | 38 django/core/checks/security/base.py | 1 | 79| 691 | 30689 | 71671 | 
| 110 | 38 django/template/base.py | 712 | 747| 270 | 30959 | 71671 | 
| 111 | 39 django/template/exceptions.py | 1 | 45| 262 | 31221 | 71934 | 
| 112 | 39 django/template/defaulttags.py | 1258 | 1286| 164 | 31385 | 71934 | 
| 113 | 39 django/template/defaulttags.py | 768 | 866| 774 | 32159 | 71934 | 
| 114 | 39 django/template/defaulttags.py | 1418 | 1457| 369 | 32528 | 71934 | 
| 115 | 39 django/template/defaulttags.py | 348 | 373| 233 | 32761 | 71934 | 
| 116 | 40 django/template/response.py | 144 | 164| 106 | 32867 | 73046 | 
| 117 | 41 django/contrib/auth/checks.py | 107 | 221| 786 | 33653 | 74562 | 
| 118 | 41 django/template/backends/jinja2.py | 55 | 91| 233 | 33886 | 74562 | 
| 119 | 41 django/core/checks/security/base.py | 259 | 284| 211 | 34097 | 74562 | 
| 120 | 41 django/template/base.py | 609 | 644| 361 | 34458 | 74562 | 
| 121 | 41 django/templatetags/tz.py | 1 | 62| 305 | 34763 | 74562 | 
| 122 | 42 django/template/autoreload.py | 33 | 55| 149 | 34912 | 74919 | 
| 123 | 43 django/utils/translation/template.py | 65 | 247| 1459 | 36371 | 76907 | 
| 124 | 43 django/contrib/admin/checks.py | 1322 | 1351| 176 | 36547 | 76907 | 
| 125 | 43 django/core/checks/model_checks.py | 135 | 159| 268 | 36815 | 76907 | 
| 126 | 44 django/views/i18n.py | 88 | 191| 702 | 37517 | 79410 | 
| 127 | 44 django/core/checks/urls.py | 76 | 118| 266 | 37783 | 79410 | 
| 128 | 44 django/templatetags/i18n.py | 72 | 101| 245 | 38028 | 79410 | 
| 129 | 44 django/templatetags/l10n.py | 41 | 64| 190 | 38218 | 79410 | 
| 130 | 45 django/template/smartif.py | 154 | 214| 426 | 38644 | 80938 | 
| 131 | 45 django/template/loader_tags.py | 211 | 240| 292 | 38936 | 80938 | 
| 132 | 45 django/template/autoreload.py | 1 | 30| 206 | 39142 | 80938 | 
| 133 | 45 django/core/management/templates.py | 381 | 403| 200 | 39342 | 80938 | 
| 134 | 45 django/core/checks/registry.py | 28 | 69| 254 | 39596 | 80938 | 
| 135 | 45 django/contrib/admindocs/views.py | 102 | 138| 301 | 39897 | 80938 | 
| 136 | 45 django/contrib/admin/checks.py | 55 | 173| 772 | 40669 | 80938 | 
| 137 | 45 django/template/defaulttags.py | 92 | 114| 175 | 40844 | 80938 | 
| 138 | 46 django/template/backends/utils.py | 1 | 16| 0 | 40844 | 81029 | 
| 139 | 46 django/template/engine.py | 65 | 83| 180 | 41024 | 81029 | 
| 140 | 46 django/template/base.py | 341 | 365| 169 | 41193 | 81029 | 
| 141 | 46 django/contrib/admin/templatetags/admin_modify.py | 1 | 58| 393 | 41586 | 81029 | 
| 142 | 46 django/template/smartif.py | 117 | 151| 188 | 41774 | 81029 | 
| 143 | 46 django/template/smartif.py | 1 | 41| 299 | 42073 | 81029 | 
| 144 | 46 django/contrib/auth/checks.py | 1 | 104| 728 | 42801 | 81029 | 
| 145 | 47 django/contrib/flatpages/templatetags/flatpages.py | 1 | 43| 302 | 43103 | 81808 | 
| 146 | 47 django/contrib/admin/checks.py | 1091 | 1143| 430 | 43533 | 81808 | 
| 147 | 47 django/templatetags/cache.py | 58 | 101| 315 | 43848 | 81808 | 
| 148 | 47 django/templatetags/i18n.py | 207 | 239| 214 | 44062 | 81808 | 
| 149 | 48 django/db/models/base.py | 1785 | 1807| 171 | 44233 | 100349 | 
| 150 | 48 django/contrib/admin/checks.py | 841 | 877| 268 | 44501 | 100349 | 
| 151 | 48 django/templatetags/i18n.py | 599 | 617| 121 | 44622 | 100349 | 
| 152 | 49 django/db/backends/ddl_references.py | 184 | 220| 253 | 44875 | 101978 | 
| 153 | 50 docs/_ext/djangodocs.py | 383 | 402| 204 | 45079 | 105202 | 
| 154 | 51 django/core/management/commands/check.py | 1 | 45| 263 | 45342 | 105693 | 
| 155 | 52 django/template/backends/base.py | 1 | 82| 516 | 45858 | 106210 | 
| 156 | 52 django/contrib/admindocs/views.py | 394 | 427| 211 | 46069 | 106210 | 
| 157 | 52 django/template/base.py | 402 | 433| 262 | 46331 | 106210 | 
| 158 | 52 django/core/management/commands/check.py | 47 | 84| 233 | 46564 | 106210 | 
| 159 | 52 django/core/checks/urls.py | 1 | 28| 142 | 46706 | 106210 | 
| 160 | 53 django/template/loaders/base.py | 1 | 52| 286 | 46992 | 106497 | 
| 161 | 54 django/views/debug.py | 1 | 56| 351 | 47343 | 111243 | 
| 162 | 55 django/views/csrf.py | 1 | 13| 132 | 47475 | 112796 | 
| 163 | 55 django/contrib/admin/templatetags/log.py | 28 | 70| 330 | 47805 | 112796 | 
| 164 | 56 django/template/context.py | 244 | 270| 197 | 48002 | 114688 | 
| 165 | 56 django/template/base.py | 138 | 177| 296 | 48298 | 114688 | 
| 166 | 56 django/templatetags/i18n.py | 358 | 453| 649 | 48947 | 114688 | 
| 167 | 57 django/forms/renderers.py | 32 | 103| 443 | 49390 | 115337 | 
| 168 | 57 django/utils/translation/template.py | 1 | 36| 384 | 49774 | 115337 | 
| 169 | 58 django/views/defaults.py | 1 | 26| 151 | 49925 | 116320 | 
| 170 | 58 django/utils/translation/template.py | 39 | 63| 165 | 50090 | 116320 | 
| 171 | 58 django/template/base.py | 458 | 519| 567 | 50657 | 116320 | 
| 172 | 59 django/db/models/fields/related.py | 302 | 339| 296 | 50953 | 130932 | 
| 173 | 59 django/templatetags/i18n.py | 104 | 146| 283 | 51236 | 130932 | 
| 174 | 60 django/forms/utils.py | 48 | 55| 113 | 51349 | 132651 | 
| 175 | 60 django/db/models/base.py | 1760 | 1783| 176 | 51525 | 132651 | 
| 176 | 60 django/db/models/base.py | 1809 | 1842| 232 | 51757 | 132651 | 
| 177 | 60 django/db/models/fields/related.py | 226 | 301| 696 | 52453 | 132651 | 
| 178 | 60 django/template/loader_tags.py | 160 | 208| 366 | 52819 | 132651 | 
| 179 | 60 django/templatetags/i18n.py | 456 | 567| 774 | 53593 | 132651 | 
| 180 | 61 django/utils/formats.py | 62 | 97| 246 | 53839 | 135141 | 
| 181 | 61 django/contrib/admin/templatetags/admin_list.py | 54 | 81| 196 | 54035 | 135141 | 
| 182 | 61 django/contrib/admin/checks.py | 1210 | 1228| 140 | 54175 | 135141 | 
| 183 | 61 django/templatetags/i18n.py | 35 | 69| 222 | 54397 | 135141 | 
| 184 | 61 django/template/base.py | 776 | 841| 558 | 54955 | 135141 | 
| 185 | 61 django/db/models/base.py | 1903 | 1992| 675 | 55630 | 135141 | 
| 186 | 62 django/utils/html.py | 139 | 165| 149 | 55779 | 138393 | 
| 187 | 62 django/core/checks/model_checks.py | 93 | 116| 170 | 55949 | 138393 | 
| 188 | 62 django/templatetags/i18n.py | 335 | 355| 173 | 56122 | 138393 | 
| 189 | 62 django/contrib/admin/checks.py | 1 | 52| 321 | 56443 | 138393 | 
| 190 | 62 django/template/loader_tags.py | 243 | 274| 272 | 56715 | 138393 | 
| 191 | 62 django/contrib/admin/checks.py | 348 | 367| 146 | 56861 | 138393 | 
| 192 | 63 django/contrib/sites/checks.py | 1 | 13| 0 | 56861 | 138470 | 
| 193 | 63 django/contrib/admin/checks.py | 930 | 953| 195 | 57056 | 138470 | 
| 194 | 63 django/contrib/admindocs/views.py | 1 | 36| 256 | 57312 | 138470 | 
| 195 | 64 django/core/checks/messages.py | 59 | 82| 161 | 57473 | 139050 | 
| 196 | 64 django/contrib/admin/checks.py | 894 | 928| 222 | 57695 | 139050 | 
| 197 | 64 django/contrib/admin/checks.py | 826 | 839| 117 | 57812 | 139050 | 
| 198 | 65 django/template/loaders/cached.py | 1 | 66| 501 | 58313 | 139775 | 
| 199 | 65 django/templatetags/i18n.py | 265 | 289| 234 | 58547 | 139775 | 
| 200 | 65 django/template/loader.py | 52 | 67| 117 | 58664 | 139775 | 
| 201 | 65 django/template/loader_tags.py | 42 | 80| 321 | 58985 | 139775 | 
| 202 | 65 django/core/checks/security/base.py | 226 | 239| 127 | 59112 | 139775 | 
| 203 | 65 django/contrib/admin/checks.py | 314 | 346| 233 | 59345 | 139775 | 
| 204 | 65 django/db/models/base.py | 2255 | 2446| 1302 | 60647 | 139775 | 


### Hint

```
Thanks for the report. It's a bug in the new system check (see 004b4620f6f4ad87261e149898940f2dcd5757ef and #32987).
```

## Patch

```diff
diff --git a/django/core/checks/templates.py b/django/core/checks/templates.py
--- a/django/core/checks/templates.py
+++ b/django/core/checks/templates.py
@@ -50,15 +50,15 @@ def check_string_if_invalid_is_string(app_configs, **kwargs):
 @register(Tags.templates)
 def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
     errors = []
-    libraries = defaultdict(list)
+    libraries = defaultdict(set)
 
     for conf in settings.TEMPLATES:
         custom_libraries = conf.get("OPTIONS", {}).get("libraries", {})
         for module_name, module_path in custom_libraries.items():
-            libraries[module_name].append(module_path)
+            libraries[module_name].add(module_path)
 
     for module_name, module_path in get_template_tag_modules():
-        libraries[module_name].append(module_path)
+        libraries[module_name].add(module_path)
 
     for library_name, items in libraries.items():
         if len(items) > 1:
@@ -66,7 +66,7 @@ def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
                 Error(
                     E003.msg.format(
                         repr(library_name),
-                        ", ".join(repr(item) for item in items),
+                        ", ".join(repr(item) for item in sorted(items)),
                     ),
                     id=E003.id,
                 )

```

## Test Patch

```diff
diff --git a/tests/check_framework/test_templates.py b/tests/check_framework/test_templates.py
--- a/tests/check_framework/test_templates.py
+++ b/tests/check_framework/test_templates.py
@@ -158,6 +158,19 @@ def test_template_tags_with_same_library_name(self):
                 [self.error_same_tags],
             )
 
+    @override_settings(
+        INSTALLED_APPS=["check_framework.template_test_apps.same_tags_app_1"]
+    )
+    def test_template_tags_same_library_in_installed_apps_libraries(self):
+        with self.settings(
+            TEMPLATES=[
+                self.get_settings(
+                    "same_tags", "same_tags_app_1.templatetags.same_tags"
+                ),
+            ]
+        ):
+            self.assertEqual(check_for_template_tags_with_the_same_name(None), [])
+
     @override_settings(
         INSTALLED_APPS=["check_framework.template_test_apps.same_tags_app_1"]
     )

```


## Code snippets

### 1 - django/core/checks/templates.py:

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
### 2 - django/core/checks/templates.py:

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
### 3 - django/template/defaulttags.py:

Start line: 1025, End line: 1059

```python
def find_library(parser, name):
    try:
        return parser.libraries[name]
    except KeyError:
        raise TemplateSyntaxError(
            "'%s' is not a registered tag library. Must be one of:\n%s"
            % (
                name,
                "\n".join(sorted(parser.libraries)),
            ),
        )


def load_from_library(library, label, names):
    """
    Return a subset of tags and filters from a library.
    """
    subset = Library()
    for name in names:
        found = False
        if name in library.tags:
            found = True
            subset.tags[name] = library.tags[name]
        if name in library.filters:
            found = True
            subset.filters[name] = library.filters[name]
        if found is False:
            raise TemplateSyntaxError(
                "'%s' is not a valid tag or filter in tag library '%s'"
                % (
                    name,
                    label,
                ),
            )
    return subset
```
### 4 - django/template/library.py:

Start line: 1, End line: 54

```python
from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap

from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable

from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError


class InvalidTemplateLibrary(Exception):
    pass


class Library:
    """
    A class for registering template tags and filters. Compiled filter and
    template tag functions are stored in the filters and tags attributes.
    The filter, simple_tag, and inclusion_tag methods provide a convenient
    way to register callables as tags.
    """

    def __init__(self):
        self.filters = {}
        self.tags = {}

    def tag(self, name=None, compile_function=None):
        if name is None and compile_function is None:
            # @register.tag()
            return self.tag_function
        elif name is not None and compile_function is None:
            if callable(name):
                # @register.tag
                return self.tag_function(name)
            else:
                # @register.tag('somename') or @register.tag(name='somename')
                def dec(func):
                    return self.tag(name, func)

                return dec
        elif name is not None and compile_function is not None:
            # register.tag('somename', somefunc)
            self.tags[name] = compile_function
            return compile_function
        else:
            raise ValueError(
                "Unsupported arguments to Library.tag: (%r, %r)"
                % (name, compile_function),
            )

    def tag_function(self, func):
        self.tags[func.__name__] = func
        return func
```
### 5 - django/template/defaulttags.py:

Start line: 1062, End line: 1091

```python
@register.tag
def load(parser, token):
    """
    Load a custom template tag library into the parser.

    For example, to load the template tags in
    ``django/templatetags/news/photos.py``::

        {% load news.photos %}

    Can also be used to load an individual tag/filter from
    a library::

        {% load byline from news %}
    """
    # token.split_contents() isn't useful here because this tag doesn't accept
    # variable as arguments.
    bits = token.contents.split()
    if len(bits) >= 4 and bits[-2] == "from":
        # from syntax is used; load individual tags from the library
        name = bits[-1]
        lib = find_library(parser, name)
        subset = load_from_library(lib, name, bits[1:-2])
        parser.add_library(subset)
    else:
        # one or more libraries are specified; load and add them to the parser
        for name in bits[1:]:
            lib = find_library(parser, name)
            parser.add_library(lib)
    return LoadNode()
```
### 6 - django/template/defaulttags.py:

Start line: 1, End line: 64

```python
"""Default tags used by the template system, available to all templates."""
import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby

from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe

from .base import (
    BLOCK_TAG_END,
    BLOCK_TAG_START,
    COMMENT_TAG_END,
    COMMENT_TAG_START,
    FILTER_SEPARATOR,
    SINGLE_BRACE_END,
    SINGLE_BRACE_START,
    VARIABLE_ATTRIBUTE_SEPARATOR,
    VARIABLE_TAG_END,
    VARIABLE_TAG_START,
    Node,
    NodeList,
    TemplateSyntaxError,
    VariableDoesNotExist,
    kwarg_re,
    render_value_in_context,
    token_kwargs,
)
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal

register = Library()


class AutoEscapeControlNode(Node):
    """Implement the actions of the autoescape tag."""

    def __init__(self, setting, nodelist):
        self.setting, self.nodelist = setting, nodelist

    def render(self, context):
        old_setting = context.autoescape
        context.autoescape = self.setting
        output = self.nodelist.render(context)
        context.autoescape = old_setting
        if self.setting:
            return mark_safe(output)
        else:
            return output


class CommentNode(Node):
    child_nodelists = ()

    def render(self, context):
        return ""
```
### 7 - django/template/defaulttags.py:

Start line: 869, End line: 889

```python
class TemplateLiteral(Literal):
    def __init__(self, value, text):
        self.value = value
        self.text = text  # for better error messages

    def display(self):
        return self.text

    def eval(self, context):
        return self.value.resolve(context, ignore_failures=True)


class TemplateIfParser(IfParser):
    error_class = TemplateSyntaxError

    def __init__(self, parser, *args, **kwargs):
        self.template_parser = parser
        super().__init__(*args, **kwargs)

    def create_var(self, value):
        return TemplateLiteral(self.template_parser.compile_filter(value), value)
```
### 8 - django/template/library.py:

Start line: 99, End line: 154

```python
class Library:

    def filter_function(self, func, **flags):
        return self.filter(func.__name__, func, **flags)

    def simple_tag(self, func=None, takes_context=None, name=None):
        """
        Register a callable as a compiled template tag. Example:

        @register.simple_tag
        def hello(*args, **kwargs):
            return 'world'
        """

        def dec(func):
            (
                params,
                varargs,
                varkw,
                defaults,
                kwonly,
                kwonly_defaults,
                _,
            ) = getfullargspec(unwrap(func))
            function_name = name or func.__name__

            @wraps(func)
            def compile_func(parser, token):
                bits = token.split_contents()[1:]
                target_var = None
                if len(bits) >= 2 and bits[-2] == "as":
                    target_var = bits[-1]
                    bits = bits[:-2]
                args, kwargs = parse_bits(
                    parser,
                    bits,
                    params,
                    varargs,
                    varkw,
                    defaults,
                    kwonly,
                    kwonly_defaults,
                    takes_context,
                    function_name,
                )
                return SimpleNode(func, takes_context, args, kwargs, target_var)

            self.tag(function_name, compile_func)
            return func

        if func is None:
            # @register.simple_tag(...)
            return dec
        elif callable(func):
            # @register.simple_tag
            return dec(func)
        else:
            raise ValueError("Invalid arguments provided to simple_tag")
```
### 9 - django/template/backends/django.py:

Start line: 110, End line: 138

```python
def get_installed_libraries():
    """
    Return the built-in template tag libraries and those from installed
    applications. Libraries are stored in a dictionary where keys are the
    individual module names, not the full module paths. Example:
    django.templatetags.i18n is stored as i18n.
    """
    return {
        module_name: full_name for module_name, full_name in get_template_tag_modules()
    }


def get_package_libraries(pkg):
    """
    Recursively yield template tag libraries defined in submodules of a
    package.
    """
    for entry in walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            module = import_module(entry[1])
        except ImportError as e:
            raise InvalidTemplateLibrary(
                "Invalid template library specified. ImportError raised when "
                "trying to load '%s': %s" % (entry[1], e)
            ) from e

        if hasattr(module, "register"):
            yield entry[1]
```
### 10 - django/template/backends/django.py:

Start line: 1, End line: 45

```python
from importlib import import_module
from pkgutil import walk_packages

from django.apps import apps
from django.conf import settings
from django.template import TemplateDoesNotExist
from django.template.context import make_context
from django.template.engine import Engine
from django.template.library import InvalidTemplateLibrary

from .base import BaseEngine


class DjangoTemplates(BaseEngine):

    app_dirname = "templates"

    def __init__(self, params):
        params = params.copy()
        options = params.pop("OPTIONS").copy()
        options.setdefault("autoescape", True)
        options.setdefault("debug", settings.DEBUG)
        options.setdefault("file_charset", "utf-8")
        libraries = options.get("libraries", {})
        options["libraries"] = self.get_templatetag_libraries(libraries)
        super().__init__(params)
        self.engine = Engine(self.dirs, self.app_dirs, **options)

    def from_string(self, template_code):
        return Template(self.engine.from_string(template_code), self)

    def get_template(self, template_name):
        try:
            return Template(self.engine.get_template(template_name), self)
        except TemplateDoesNotExist as exc:
            reraise(exc, self)

    def get_templatetag_libraries(self, custom_libraries):
        """
        Return a collation of template tag libraries from installed
        applications and the supplied custom_libraries argument.
        """
        libraries = get_installed_libraries()
        libraries.update(custom_libraries)
        return libraries
```
