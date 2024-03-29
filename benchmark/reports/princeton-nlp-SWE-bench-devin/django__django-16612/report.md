# django__django-16612

| **django/django** | `55bcbd8d172b689811fae17cde2f09218dd74e9c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 129 |
| **Any found context length** | 129 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/sites.py b/django/contrib/admin/sites.py
--- a/django/contrib/admin/sites.py
+++ b/django/contrib/admin/sites.py
@@ -453,7 +453,9 @@ def catch_all_view(self, request, url):
                 pass
             else:
                 if getattr(match.func, "should_append_slash", True):
-                    return HttpResponsePermanentRedirect("%s/" % request.path)
+                    return HttpResponsePermanentRedirect(
+                        request.get_full_path(force_append_slash=True)
+                    )
         raise Http404
 
     def _build_app_dict(self, request, label=None):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/sites.py | 456 | 456 | 1 | 1 | 129


## Problem Statement

```
AdminSite.catch_all_view() drops query string in redirects
Description
	
#31747 introduced AdminSite.catch_all_view(). However, in the process it broke the ability to redirect with settings.APPEND_SLASH = True when there are query strings.
Provided URL: ​http://127.0.0.1:8000/admin/auth/foo?id=123
Expected redirect: ​http://127.0.0.1:8000/admin/auth/foo/?id=123
Actual redirect: ​http://127.0.0.1:8000/admin/auth/foo/
This seems to be because the redirect in question does not include the query strings (such as via request.META['QUERY_STRING']):
return HttpResponsePermanentRedirect("%s/" % request.path)
​https://github.com/django/django/blob/c57ff9ba5e251cd4c2761105a6046662c08f951e/django/contrib/admin/sites.py#L456

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/sites.py** | 443 | 457| 129 | 129 | 4490 | 
| 2 | **1 django/contrib/admin/sites.py** | 228 | 249| 221 | 350 | 4490 | 
| 3 | 2 django/contrib/auth/password_validation.py | 217 | 267| 386 | 736 | 6384 | 
| 4 | 3 django/views/generic/base.py | 256 | 286| 246 | 982 | 8290 | 
| 5 | 4 django/__init__.py | 1 | 25| 173 | 1155 | 8463 | 
| 6 | 5 django/contrib/redirects/admin.py | 1 | 11| 0 | 1155 | 8532 | 
| 7 | **5 django/contrib/admin/sites.py** | 251 | 313| 527 | 1682 | 8532 | 
| 8 | 6 django/contrib/redirects/middleware.py | 1 | 51| 354 | 2036 | 8887 | 
| 9 | 7 django/contrib/admin/options.py | 682 | 722| 280 | 2316 | 28146 | 
| 10 | **7 django/contrib/admin/sites.py** | 568 | 595| 196 | 2512 | 28146 | 
| 11 | 8 django/utils/http.py | 303 | 324| 179 | 2691 | 31348 | 
| 12 | 9 django/middleware/common.py | 62 | 74| 136 | 2827 | 32894 | 
| 13 | 10 django/contrib/admin/views/main.py | 256 | 309| 428 | 3255 | 37428 | 
| 14 | 10 django/contrib/admin/options.py | 1165 | 1189| 196 | 3451 | 37428 | 
| 15 | **10 django/contrib/admin/sites.py** | 1 | 33| 224 | 3675 | 37428 | 
| 16 | 11 django/contrib/redirects/models.py | 1 | 36| 238 | 3913 | 37666 | 
| 17 | 11 django/contrib/admin/options.py | 1513 | 1539| 233 | 4146 | 37666 | 
| 18 | 12 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 4565 | 38085 | 
| 19 | 13 django/contrib/redirects/apps.py | 1 | 9| 0 | 4565 | 38136 | 
| 20 | 14 django/views/decorators/common.py | 1 | 18| 112 | 4677 | 38249 | 
| 21 | 15 django/contrib/admindocs/urls.py | 1 | 51| 307 | 4984 | 38556 | 
| 22 | 15 django/middleware/common.py | 34 | 60| 265 | 5249 | 38556 | 
| 23 | 16 django/contrib/admindocs/views.py | 184 | 210| 238 | 5487 | 42044 | 
| 24 | 16 django/contrib/admin/options.py | 2176 | 2234| 444 | 5931 | 42044 | 
| 25 | 16 django/contrib/admindocs/views.py | 1 | 36| 252 | 6183 | 42044 | 
| 26 | 17 django/contrib/auth/views.py | 35 | 62| 218 | 6401 | 44758 | 
| 27 | **17 django/contrib/admin/sites.py** | 204 | 226| 167 | 6568 | 44758 | 
| 28 | 17 django/contrib/auth/views.py | 1 | 32| 255 | 6823 | 44758 | 
| 29 | 18 django/contrib/auth/admin.py | 1 | 25| 195 | 7018 | 46529 | 
| 30 | 18 django/contrib/admin/options.py | 1855 | 1886| 303 | 7321 | 46529 | 
| 31 | **18 django/contrib/admin/sites.py** | 36 | 79| 339 | 7660 | 46529 | 
| 32 | 18 django/middleware/common.py | 100 | 115| 165 | 7825 | 46529 | 
| 33 | 19 django/contrib/auth/urls.py | 1 | 37| 253 | 8078 | 46782 | 
| 34 | 19 django/middleware/common.py | 1 | 32| 247 | 8325 | 46782 | 
| 35 | 19 django/contrib/admin/options.py | 2099 | 2174| 599 | 8924 | 46782 | 
| 36 | 19 django/contrib/admin/options.py | 2006 | 2097| 784 | 9708 | 46782 | 
| 37 | 20 django/urls/base.py | 27 | 88| 440 | 10148 | 47978 | 
| 38 | 21 django/views/defaults.py | 1 | 26| 151 | 10299 | 48968 | 
| 39 | **21 django/contrib/admin/sites.py** | 315 | 340| 202 | 10501 | 48968 | 
| 40 | 21 django/contrib/admin/options.py | 1917 | 2005| 676 | 11177 | 48968 | 
| 41 | 21 django/contrib/admin/options.py | 1260 | 1322| 511 | 11688 | 48968 | 
| 42 | **21 django/contrib/admin/sites.py** | 547 | 566| 125 | 11813 | 48968 | 
| 43 | 21 django/contrib/admin/options.py | 1 | 113| 768 | 12581 | 48968 | 
| 44 | 22 django/views/generic/__init__.py | 1 | 40| 204 | 12785 | 49173 | 
| 45 | 23 django/contrib/admin/helpers.py | 246 | 259| 115 | 12900 | 52785 | 
| 46 | 23 django/contrib/admin/options.py | 1752 | 1854| 780 | 13680 | 52785 | 
| 47 | 24 django/contrib/admin/__init__.py | 1 | 51| 281 | 13961 | 53066 | 
| 48 | 24 django/contrib/admin/options.py | 1719 | 1750| 293 | 14254 | 53066 | 
| 49 | 24 django/contrib/admindocs/views.py | 141 | 161| 170 | 14424 | 53066 | 
| 50 | 25 django/views/csrf.py | 30 | 88| 587 | 15011 | 53866 | 
| 51 | 25 django/middleware/common.py | 76 | 98| 228 | 15239 | 53866 | 
| 52 | 25 django/contrib/admin/options.py | 1658 | 1679| 133 | 15372 | 53866 | 
| 53 | 26 django/views/debug.py | 196 | 221| 181 | 15553 | 58922 | 
| 54 | 27 django/shortcuts.py | 28 | 61| 247 | 15800 | 60046 | 
| 55 | 27 django/contrib/admindocs/views.py | 164 | 182| 187 | 15987 | 60046 | 
| 56 | 27 django/contrib/admindocs/views.py | 102 | 138| 301 | 16288 | 60046 | 
| 57 | 27 django/contrib/admin/views/main.py | 1 | 51| 324 | 16612 | 60046 | 
| 58 | 28 django/contrib/gis/views.py | 1 | 23| 160 | 16772 | 60206 | 
| 59 | 29 django/middleware/security.py | 1 | 31| 281 | 17053 | 60732 | 
| 60 | 30 django/http/response.py | 654 | 687| 157 | 17210 | 66114 | 
| 61 | 30 django/contrib/admin/options.py | 356 | 429| 500 | 17710 | 66114 | 
| 62 | 30 django/contrib/admin/views/main.py | 54 | 151| 736 | 18446 | 66114 | 
| 63 | 31 django/core/checks/security/csrf.py | 45 | 68| 159 | 18605 | 66579 | 
| 64 | 31 django/contrib/admin/options.py | 2236 | 2285| 450 | 19055 | 66579 | 
| 65 | 31 django/contrib/admin/options.py | 724 | 741| 128 | 19183 | 66579 | 
| 66 | 32 django/contrib/redirects/migrations/0001_initial.py | 1 | 65| 309 | 19492 | 66888 | 
| 67 | 33 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 19917 | 67729 | 
| 68 | 33 django/contrib/auth/views.py | 90 | 121| 216 | 20133 | 67729 | 
| 69 | **33 django/contrib/admin/sites.py** | 598 | 612| 116 | 20249 | 67729 | 
| 70 | 33 django/contrib/admin/options.py | 1541 | 1608| 586 | 20835 | 67729 | 
| 71 | 33 django/contrib/admindocs/views.py | 213 | 297| 615 | 21450 | 67729 | 
| 72 | 33 django/contrib/admindocs/views.py | 298 | 392| 640 | 22090 | 67729 | 
| 73 | 33 django/contrib/admindocs/views.py | 395 | 428| 211 | 22301 | 67729 | 
| 74 | 34 django/http/request.py | 630 | 660| 208 | 22509 | 73307 | 
| 75 | 34 django/http/response.py | 628 | 651| 195 | 22704 | 73307 | 
| 76 | 35 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 24| 117 | 22821 | 73424 | 
| 77 | 35 django/contrib/admindocs/views.py | 39 | 62| 163 | 22984 | 73424 | 
| 78 | 35 django/contrib/admin/options.py | 1888 | 1899| 149 | 23133 | 73424 | 
| 79 | 36 django/contrib/sites/admin.py | 1 | 9| 0 | 23133 | 73470 | 
| 80 | 37 django/contrib/admin/views/decorators.py | 1 | 20| 137 | 23270 | 73608 | 
| 81 | 37 django/middleware/common.py | 153 | 179| 255 | 23525 | 73608 | 
| 82 | 37 django/http/request.py | 183 | 192| 113 | 23638 | 73608 | 
| 83 | 37 django/contrib/admin/options.py | 1143 | 1163| 201 | 23839 | 73608 | 
| 84 | 37 django/contrib/admin/views/main.py | 554 | 586| 227 | 24066 | 73608 | 
| 85 | **37 django/contrib/admin/sites.py** | 81 | 97| 132 | 24198 | 73608 | 
| 86 | 38 django/contrib/syndication/views.py | 1 | 26| 223 | 24421 | 75466 | 
| 87 | 39 django/middleware/csrf.py | 296 | 346| 450 | 24871 | 79577 | 
| 88 | 40 django/urls/resolvers.py | 739 | 827| 722 | 25593 | 85641 | 
| 89 | 41 django/contrib/flatpages/views.py | 1 | 45| 399 | 25992 | 86231 | 
| 90 | 41 django/views/generic/base.py | 145 | 179| 204 | 26196 | 86231 | 
| 91 | 42 django/contrib/admindocs/middleware.py | 1 | 34| 257 | 26453 | 86489 | 
| 92 | **42 django/contrib/admin/sites.py** | 383 | 403| 152 | 26605 | 86489 | 
| 93 | 43 django/contrib/auth/decorators.py | 1 | 40| 315 | 26920 | 87081 | 
| 94 | 43 django/middleware/csrf.py | 413 | 468| 577 | 27497 | 87081 | 
| 95 | 43 django/contrib/admin/options.py | 1610 | 1656| 322 | 27819 | 87081 | 
| 96 | 43 django/contrib/admin/options.py | 1414 | 1511| 710 | 28529 | 87081 | 
| 97 | 44 django/contrib/contenttypes/views.py | 1 | 89| 712 | 29241 | 87793 | 
| 98 | **44 django/contrib/admin/sites.py** | 405 | 441| 300 | 29541 | 87793 | 
| 99 | 45 django/conf/global_settings.py | 153 | 263| 832 | 30373 | 93631 | 
| 100 | 46 django/contrib/sitemaps/views.py | 42 | 88| 423 | 30796 | 94703 | 
| 101 | 46 django/contrib/auth/admin.py | 121 | 147| 286 | 31082 | 94703 | 
| 102 | 47 django/contrib/admin/utils.py | 124 | 162| 310 | 31392 | 98971 | 
| 103 | 47 django/views/defaults.py | 102 | 121| 144 | 31536 | 98971 | 
| 104 | 47 django/contrib/admindocs/views.py | 65 | 99| 297 | 31833 | 98971 | 
| 105 | 48 django/contrib/sitemaps/__init__.py | 30 | 56| 243 | 32076 | 100799 | 
| 106 | 48 django/contrib/sitemaps/views.py | 91 | 141| 369 | 32445 | 100799 | 
| 107 | 48 django/contrib/admin/options.py | 997 | 1011| 125 | 32570 | 100799 | 
| 108 | 49 django/contrib/admin/templatetags/admin_list.py | 1 | 32| 187 | 32757 | 104585 | 
| 109 | 50 django/contrib/flatpages/urls.py | 1 | 7| 0 | 32757 | 104623 | 
| 110 | 50 django/views/defaults.py | 124 | 150| 197 | 32954 | 104623 | 
| 111 | **50 django/contrib/admin/sites.py** | 360 | 381| 182 | 33136 | 104623 | 
| 112 | 51 django/contrib/admin/widgets.py | 171 | 204| 244 | 33380 | 108813 | 
| 113 | 51 django/contrib/admin/options.py | 116 | 146| 223 | 33603 | 108813 | 
| 114 | 52 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 33726 | 114641 | 
| 115 | 53 django/contrib/sites/managers.py | 1 | 46| 277 | 34003 | 115036 | 
| 116 | 53 django/contrib/admin/options.py | 2382 | 2439| 465 | 34468 | 115036 | 
| 117 | **53 django/contrib/admin/sites.py** | 342 | 358| 158 | 34626 | 115036 | 
| 118 | 54 django/urls/conf.py | 61 | 96| 266 | 34892 | 115753 | 
| 119 | 54 django/middleware/csrf.py | 348 | 411| 585 | 35477 | 115753 | 
| 120 | 55 django/contrib/auth/__init__.py | 1 | 38| 240 | 35717 | 117357 | 
| 121 | 56 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 35864 | 117505 | 
| 122 | 56 django/contrib/admin/options.py | 2470 | 2505| 315 | 36179 | 117505 | 
| 123 | 56 django/views/defaults.py | 29 | 79| 377 | 36556 | 117505 | 
| 124 | 56 django/contrib/admin/options.py | 431 | 489| 509 | 37065 | 117505 | 
| 125 | 57 django/contrib/sites/models.py | 25 | 46| 192 | 37257 | 118294 | 
| 126 | **57 django/contrib/admin/sites.py** | 531 | 545| 118 | 37375 | 118294 | 
| 127 | 57 django/contrib/admin/views/main.py | 495 | 552| 465 | 37840 | 118294 | 
| 128 | 57 django/contrib/auth/admin.py | 149 | 214| 477 | 38317 | 118294 | 
| 129 | 58 django/core/checks/security/base.py | 1 | 79| 691 | 39008 | 120483 | 
| 130 | 58 django/contrib/admin/options.py | 235 | 248| 139 | 39147 | 120483 | 
| 131 | 58 django/contrib/admin/options.py | 1036 | 1061| 216 | 39363 | 120483 | 
| 132 | 58 django/contrib/admin/options.py | 1324 | 1412| 689 | 40052 | 120483 | 
| 133 | 58 django/contrib/admin/utils.py | 295 | 321| 189 | 40241 | 120483 | 
| 134 | **58 django/contrib/admin/sites.py** | 150 | 202| 349 | 40590 | 120483 | 
| 135 | **58 django/contrib/admin/sites.py** | 459 | 529| 481 | 41071 | 120483 | 
| 136 | 58 django/core/checks/security/csrf.py | 1 | 42| 305 | 41376 | 120483 | 
| 137 | 59 django/core/handlers/asgi.py | 1 | 25| 123 | 41499 | 122915 | 
| 138 | 59 django/contrib/admin/views/autocomplete.py | 1 | 42| 241 | 41740 | 122915 | 
| 139 | 59 django/contrib/admin/templatetags/admin_list.py | 179 | 195| 140 | 41880 | 122915 | 
| 140 | 59 django/contrib/admin/views/main.py | 153 | 254| 863 | 42743 | 122915 | 
| 141 | 59 django/contrib/admin/options.py | 1901 | 1915| 132 | 42875 | 122915 | 
| 142 | 59 django/contrib/sitemaps/views.py | 1 | 27| 162 | 43037 | 122915 | 
| 143 | 59 django/contrib/admin/options.py | 594 | 610| 173 | 43210 | 122915 | 
| 144 | 60 django/contrib/admin/apps.py | 1 | 28| 164 | 43374 | 123079 | 
| 145 | 60 django/contrib/sites/models.py | 79 | 121| 236 | 43610 | 123079 | 
| 146 | 60 django/core/checks/security/base.py | 259 | 284| 211 | 43821 | 123079 | 
| 147 | 61 django/views/i18n.py | 30 | 74| 382 | 44203 | 124948 | 
| 148 | 61 django/middleware/csrf.py | 199 | 218| 151 | 44354 | 124948 | 
| 149 | 61 django/contrib/admin/options.py | 613 | 663| 349 | 44703 | 124948 | 
| 150 | 61 django/contrib/admin/options.py | 332 | 354| 169 | 44872 | 124948 | 
| 151 | 62 django/contrib/admin/models.py | 1 | 21| 123 | 44995 | 126141 | 
| 152 | 62 django/middleware/csrf.py | 252 | 268| 186 | 45181 | 126141 | 
| 153 | 62 django/contrib/admin/widgets.py | 361 | 381| 172 | 45353 | 126141 | 
| 154 | 62 django/contrib/admindocs/views.py | 486 | 499| 122 | 45475 | 126141 | 
| 155 | 62 django/views/debug.py | 106 | 144| 284 | 45759 | 126141 | 
| 156 | 63 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 59 | 77| 113 | 45872 | 126706 | 
| 157 | 64 docs/_ext/djangodocs.py | 26 | 71| 398 | 46270 | 129930 | 
| 158 | 65 django/contrib/admin/tests.py | 132 | 147| 169 | 46439 | 131572 | 
| 159 | 66 django/contrib/gis/admin/__init__.py | 1 | 30| 130 | 46569 | 131702 | 
| 160 | 66 django/contrib/sites/models.py | 48 | 76| 237 | 46806 | 131702 | 
| 161 | 66 django/contrib/sites/managers.py | 48 | 66| 125 | 46931 | 131702 | 
| 162 | 67 django/contrib/flatpages/sitemaps.py | 1 | 15| 116 | 47047 | 131818 | 
| 163 | 67 django/contrib/admin/options.py | 979 | 995| 169 | 47216 | 131818 | 
| 164 | 67 django/contrib/admin/tests.py | 1 | 37| 265 | 47481 | 131818 | 
| 165 | 67 django/contrib/admin/templatetags/admin_list.py | 53 | 80| 196 | 47677 | 131818 | 
| 166 | 68 django/contrib/admindocs/utils.py | 1 | 28| 185 | 47862 | 133803 | 
| 167 | 68 django/views/debug.py | 182 | 194| 148 | 48010 | 133803 | 
| 168 | 68 django/middleware/csrf.py | 270 | 294| 176 | 48186 | 133803 | 
| 169 | 68 django/middleware/csrf.py | 1 | 55| 480 | 48666 | 133803 | 
| 170 | 69 django/contrib/admin/checks.py | 176 | 192| 155 | 48821 | 143329 | 
| 171 | 70 django/core/servers/basehttp.py | 214 | 233| 170 | 48991 | 145470 | 
| 172 | 71 django/views/decorators/clickjacking.py | 25 | 63| 243 | 49234 | 145855 | 
| 173 | 71 django/contrib/auth/views.py | 252 | 294| 382 | 49616 | 145855 | 
| 174 | 72 django/conf/urls/__init__.py | 1 | 10| 0 | 49616 | 145920 | 
| 175 | 72 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 56| 452 | 50068 | 145920 | 
| 176 | 72 django/middleware/csrf.py | 470 | 483| 163 | 50231 | 145920 | 
| 177 | 72 django/contrib/admin/checks.py | 954 | 978| 197 | 50428 | 145920 | 
| 178 | 73 django/contrib/admin/filters.py | 72 | 127| 415 | 50843 | 150197 | 
| 179 | 74 django/contrib/admin/exceptions.py | 1 | 14| 0 | 50843 | 150264 | 
| 180 | 74 django/contrib/admin/checks.py | 789 | 807| 183 | 51026 | 150264 | 
| 181 | 75 django/utils/html.py | 191 | 225| 298 | 51324 | 153512 | 
| 182 | 75 django/views/decorators/clickjacking.py | 1 | 22| 140 | 51464 | 153512 | 
| 183 | 75 django/middleware/csrf.py | 184 | 197| 116 | 51580 | 153512 | 
| 184 | 75 django/core/checks/security/base.py | 81 | 180| 732 | 52312 | 153512 | 
| 185 | 75 django/contrib/admin/options.py | 665 | 680| 123 | 52435 | 153512 | 
| 186 | 76 django/views/decorators/debug.py | 49 | 77| 199 | 52634 | 154103 | 
| 187 | 76 django/views/defaults.py | 82 | 99| 121 | 52755 | 154103 | 
| 188 | 76 django/views/debug.py | 237 | 289| 471 | 53226 | 154103 | 
| 189 | 77 django/contrib/admin/actions.py | 1 | 97| 647 | 53873 | 154750 | 
| 190 | 77 django/urls/resolvers.py | 726 | 737| 120 | 53993 | 154750 | 
| 191 | 77 django/contrib/auth/views.py | 348 | 380| 239 | 54232 | 154750 | 
| 192 | 77 django/contrib/admin/checks.py | 55 | 173| 772 | 55004 | 154750 | 
| 193 | 78 django/contrib/flatpages/models.py | 1 | 50| 368 | 55372 | 155118 | 
| 194 | 78 django/urls/base.py | 1 | 24| 170 | 55542 | 155118 | 
| 195 | 78 django/contrib/flatpages/views.py | 48 | 71| 191 | 55733 | 155118 | 
| 196 | 78 django/utils/http.py | 42 | 80| 301 | 56034 | 155118 | 
| 197 | 79 django/template/defaultfilters.py | 368 | 455| 504 | 56538 | 161673 | 


### Hint

```
Thanks for the report! Using get_full_path() should fix the issue: django/contrib/admin/sites.py diff --git a/django/contrib/admin/sites.py b/django/contrib/admin/sites.py index 61be31d890..96c54e44ad 100644 a b class AdminSite: 453453 pass 454454 else: 455455 if getattr(match.func, "should_append_slash", True): 456 return HttpResponsePermanentRedirect("%s/" % request.path) 456 return HttpResponsePermanentRedirect(request.get_full_path(force_append_slash=True)) 457457 raise Http404 458458 459459 def _build_app_dict(self, request, label=None): Would you like to prepare PR via GitHub? (a regression test is required.) Regression in ba31b0103442ac891fb3cb98f316781254e366c3.
​https://github.com/django/django/pull/16612
```

## Patch

```diff
diff --git a/django/contrib/admin/sites.py b/django/contrib/admin/sites.py
--- a/django/contrib/admin/sites.py
+++ b/django/contrib/admin/sites.py
@@ -453,7 +453,9 @@ def catch_all_view(self, request, url):
                 pass
             else:
                 if getattr(match.func, "should_append_slash", True):
-                    return HttpResponsePermanentRedirect("%s/" % request.path)
+                    return HttpResponsePermanentRedirect(
+                        request.get_full_path(force_append_slash=True)
+                    )
         raise Http404
 
     def _build_app_dict(self, request, label=None):

```

## Test Patch

```diff
diff --git a/tests/admin_views/tests.py b/tests/admin_views/tests.py
--- a/tests/admin_views/tests.py
+++ b/tests/admin_views/tests.py
@@ -8463,6 +8463,24 @@ def test_missing_slash_append_slash_true(self):
             response, known_url, status_code=301, target_status_code=403
         )
 
+    @override_settings(APPEND_SLASH=True)
+    def test_missing_slash_append_slash_true_query_string(self):
+        superuser = User.objects.create_user(
+            username="staff",
+            password="secret",
+            email="staff@example.com",
+            is_staff=True,
+        )
+        self.client.force_login(superuser)
+        known_url = reverse("admin:admin_views_article_changelist")
+        response = self.client.get("%s?id=1" % known_url[:-1])
+        self.assertRedirects(
+            response,
+            f"{known_url}?id=1",
+            status_code=301,
+            fetch_redirect_response=False,
+        )
+
     @override_settings(APPEND_SLASH=True)
     def test_missing_slash_append_slash_true_script_name(self):
         superuser = User.objects.create_user(
@@ -8481,6 +8499,24 @@ def test_missing_slash_append_slash_true_script_name(self):
             fetch_redirect_response=False,
         )
 
+    @override_settings(APPEND_SLASH=True)
+    def test_missing_slash_append_slash_true_script_name_query_string(self):
+        superuser = User.objects.create_user(
+            username="staff",
+            password="secret",
+            email="staff@example.com",
+            is_staff=True,
+        )
+        self.client.force_login(superuser)
+        known_url = reverse("admin:admin_views_article_changelist")
+        response = self.client.get("%s?id=1" % known_url[:-1], SCRIPT_NAME="/prefix/")
+        self.assertRedirects(
+            response,
+            f"/prefix{known_url}?id=1",
+            status_code=301,
+            fetch_redirect_response=False,
+        )
+
     @override_settings(APPEND_SLASH=True, FORCE_SCRIPT_NAME="/prefix/")
     def test_missing_slash_append_slash_true_force_script_name(self):
         superuser = User.objects.create_user(
@@ -8515,6 +8551,23 @@ def test_missing_slash_append_slash_true_non_staff_user(self):
             "/test_admin/admin/login/?next=/test_admin/admin/admin_views/article",
         )
 
+    @override_settings(APPEND_SLASH=True)
+    def test_missing_slash_append_slash_true_non_staff_user_query_string(self):
+        user = User.objects.create_user(
+            username="user",
+            password="secret",
+            email="user@example.com",
+            is_staff=False,
+        )
+        self.client.force_login(user)
+        known_url = reverse("admin:admin_views_article_changelist")
+        response = self.client.get("%s?id=1" % known_url[:-1])
+        self.assertRedirects(
+            response,
+            "/test_admin/admin/login/?next=/test_admin/admin/admin_views/article"
+            "%3Fid%3D1",
+        )
+
     @override_settings(APPEND_SLASH=False)
     def test_missing_slash_append_slash_false(self):
         superuser = User.objects.create_user(
@@ -8629,6 +8682,24 @@ def test_missing_slash_append_slash_true_without_final_catch_all_view(self):
             response, known_url, status_code=301, target_status_code=403
         )
 
+    @override_settings(APPEND_SLASH=True)
+    def test_missing_slash_append_slash_true_query_without_final_catch_all_view(self):
+        superuser = User.objects.create_user(
+            username="staff",
+            password="secret",
+            email="staff@example.com",
+            is_staff=True,
+        )
+        self.client.force_login(superuser)
+        known_url = reverse("admin10:admin_views_article_changelist")
+        response = self.client.get("%s?id=1" % known_url[:-1])
+        self.assertRedirects(
+            response,
+            f"{known_url}?id=1",
+            status_code=301,
+            fetch_redirect_response=False,
+        )
+
     @override_settings(APPEND_SLASH=False)
     def test_missing_slash_append_slash_false_without_final_catch_all_view(self):
         superuser = User.objects.create_user(

```


## Code snippets

### 1 - django/contrib/admin/sites.py:

Start line: 443, End line: 457

```python
class AdminSite:

    def autocomplete_view(self, request):
        return AutocompleteJsonView.as_view(admin_site=self)(request)

    @no_append_slash
    def catch_all_view(self, request, url):
        if settings.APPEND_SLASH and not url.endswith("/"):
            urlconf = getattr(request, "urlconf", None)
            try:
                match = resolve("%s/" % request.path_info, urlconf)
            except Resolver404:
                pass
            else:
                if getattr(match.func, "should_append_slash", True):
                    return HttpResponsePermanentRedirect("%s/" % request.path)
        raise Http404
```
### 2 - django/contrib/admin/sites.py:

Start line: 228, End line: 249

```python
class AdminSite:

    def admin_view(self, view, cacheable=False):

        def inner(request, *args, **kwargs):
            if not self.has_permission(request):
                if request.path == reverse("admin:logout", current_app=self.name):
                    index_path = reverse("admin:index", current_app=self.name)
                    return HttpResponseRedirect(index_path)
                # Inner import to prevent django.contrib.admin (app) from
                # importing django.contrib.auth.models.User (unrelated model).
                from django.contrib.auth.views import redirect_to_login

                return redirect_to_login(
                    request.get_full_path(),
                    reverse("admin:login", current_app=self.name),
                )
            return view(request, *args, **kwargs)

        if not cacheable:
            inner = never_cache(inner)
        # We add csrf_protect here so this function can be used as a utility
        # function for any view, without having to repeat 'csrf_protect'.
        if not getattr(view, "csrf_exempt", False):
            inner = csrf_protect(inner)
        return update_wrapper(inner, view)
```
### 3 - django/contrib/auth/password_validation.py:

Start line: 217, End line: 267

```python
class CommonPasswordValidator:
    """
    Validate that the password is not a common password.

    The password is rejected if it occurs in a provided list of passwords,
    which may be gzipped. The list Django ships with contains 20000 common
    passwords (lowercased and deduplicated), created by Royce Williams:
    https://gist.github.com/roycewilliams/226886fd01572964e1431ac8afc999ce
    The password list must be lowercased to match the comparison in validate().
    """

    @cached_property
    def DEFAULT_PASSWORD_LIST_PATH(self):
        return Path(__file__).resolve().parent / "common-passwords.txt.gz"

    def __init__(self, password_list_path=DEFAULT_PASSWORD_LIST_PATH):
        if password_list_path is CommonPasswordValidator.DEFAULT_PASSWORD_LIST_PATH:
            password_list_path = self.DEFAULT_PASSWORD_LIST_PATH
        try:
            with gzip.open(password_list_path, "rt", encoding="utf-8") as f:
                self.passwords = {x.strip() for x in f}
        except OSError:
            with open(password_list_path) as f:
                self.passwords = {x.strip() for x in f}

    def validate(self, password, user=None):
        if password.lower().strip() in self.passwords:
            raise ValidationError(
                _("This password is too common."),
                code="password_too_common",
            )

    def get_help_text(self):
        return _("Your password can’t be a commonly used password.")


class NumericPasswordValidator:
    """
    Validate that the password is not entirely numeric.
    """

    def validate(self, password, user=None):
        if password.isdigit():
            raise ValidationError(
                _("This password is entirely numeric."),
                code="password_entirely_numeric",
            )

    def get_help_text(self):
        return _("Your password can’t be entirely numeric.")
```
### 4 - django/views/generic/base.py:

Start line: 256, End line: 286

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
                "Gone: %s", request.path, extra={"status_code": 410, "request": request}
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
### 5 - django/__init__.py:

Start line: 1, End line: 25

```python
from django.utils.version import get_version

VERSION = (5, 0, 0, "alpha", 0)

__version__ = get_version(VERSION)


def setup(set_prefix=True):
    """
    Configure the settings (this happens as a side effect of accessing the
    first setting), configure logging and populate the app registry.
    Set the thread-local urlresolvers script prefix if `set_prefix` is True.
    """
    from django.apps import apps
    from django.conf import settings
    from django.urls import set_script_prefix
    from django.utils.log import configure_logging

    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
    if set_prefix:
        set_script_prefix(
            "/" if settings.FORCE_SCRIPT_NAME is None else settings.FORCE_SCRIPT_NAME
        )
    apps.populate(settings.INSTALLED_APPS)
```
### 6 - django/contrib/redirects/admin.py:

Start line: 1, End line: 11

```python

```
### 7 - django/contrib/admin/sites.py:

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
### 8 - django/contrib/redirects/middleware.py:

Start line: 1, End line: 51

```python
from django.apps import apps
from django.conf import settings
from django.contrib.redirects.models import Redirect
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ImproperlyConfigured
from django.http import HttpResponseGone, HttpResponsePermanentRedirect
from django.utils.deprecation import MiddlewareMixin


class RedirectFallbackMiddleware(MiddlewareMixin):
    # Defined as class-level attributes to be subclassing-friendly.
    response_gone_class = HttpResponseGone
    response_redirect_class = HttpResponsePermanentRedirect

    def __init__(self, get_response):
        if not apps.is_installed("django.contrib.sites"):
            raise ImproperlyConfigured(
                "You cannot use RedirectFallbackMiddleware when "
                "django.contrib.sites is not installed."
            )
        super().__init__(get_response)

    def process_response(self, request, response):
        # No need to check for a redirect for non-404 responses.
        if response.status_code != 404:
            return response

        full_path = request.get_full_path()
        current_site = get_current_site(request)

        r = None
        try:
            r = Redirect.objects.get(site=current_site, old_path=full_path)
        except Redirect.DoesNotExist:
            pass
        if r is None and settings.APPEND_SLASH and not request.path.endswith("/"):
            try:
                r = Redirect.objects.get(
                    site=current_site,
                    old_path=request.get_full_path(force_append_slash=True),
                )
            except Redirect.DoesNotExist:
                pass
        if r is not None:
            if r.new_path == "":
                return self.response_gone_class()
            return self.response_redirect_class(r.new_path)

        # No redirect was found. Return the response.
        return response
```
### 9 - django/contrib/admin/options.py:

Start line: 682, End line: 722

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
### 10 - django/contrib/admin/sites.py:

Start line: 568, End line: 595

```python
class AdminSite:

    def app_index(self, request, app_label, extra_context=None):
        app_list = self.get_app_list(request, app_label)

        if not app_list:
            raise Http404("The requested admin page does not exist.")

        context = {
            **self.each_context(request),
            "title": _("%(app)s administration") % {"app": app_list[0]["name"]},
            "subtitle": None,
            "app_list": app_list,
            "app_label": app_label,
            **(extra_context or {}),
        }

        request.current_app = self.name

        return TemplateResponse(
            request,
            self.app_index_template
            or ["admin/%s/app_index.html" % app_label, "admin/app_index.html"],
            context,
        )

    def get_log_entries(self, request):
        from django.contrib.admin.models import LogEntry

        return LogEntry.objects.select_related("content_type", "user")
```
### 15 - django/contrib/admin/sites.py:

Start line: 1, End line: 33

```python
from functools import update_wrapper
from weakref import WeakSet

from django.apps import apps
from django.conf import settings
from django.contrib.admin import ModelAdmin, actions
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, Resolver404, resolve, reverse
from django.utils.decorators import method_decorator
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.common import no_append_slash
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog

all_sites = WeakSet()


class AlreadyRegistered(Exception):
    pass


class NotRegistered(Exception):
    pass
```
### 27 - django/contrib/admin/sites.py:

Start line: 204, End line: 226

```python
class AdminSite:

    def admin_view(self, view, cacheable=False):
        """
        Decorator to create an admin view attached to this ``AdminSite``. This
        wraps the view and provides permission checking by calling
        ``self.has_permission``.

        You'll want to use this from within ``AdminSite.get_urls()``:

            class MyAdminSite(AdminSite):

                def get_urls(self):
                    from django.urls import path

                    urls = super().get_urls()
                    urls += [
                        path('my_view/', self.admin_view(some_view))
                    ]
                    return urls

        By default, admin_views are marked non-cacheable using the
        ``never_cache`` decorator. If the view can be safely cached, set
        cacheable=True.
        """
        # ... other code
```
### 31 - django/contrib/admin/sites.py:

Start line: 36, End line: 79

```python
class AdminSite:
    """
    An AdminSite object encapsulates an instance of the Django admin application, ready
    to be hooked in to your URLconf. Models are registered with the AdminSite using the
    register() method, and the get_urls() method can then be used to access Django view
    functions that present a full admin interface for the collection of registered
    models.
    """

    # Text to put at the end of each page's <title>.
    site_title = gettext_lazy("Django site admin")

    # Text to put in each page's <h1>.
    site_header = gettext_lazy("Django administration")

    # Text to put at the top of the admin index page.
    index_title = gettext_lazy("Site administration")

    # URL for the "View site" link at the top of each admin page.
    site_url = "/"

    enable_nav_sidebar = True

    empty_value_display = "-"

    login_form = None
    index_template = None
    app_index_template = None
    login_template = None
    logout_template = None
    password_change_template = None
    password_change_done_template = None

    final_catch_all_view = True

    def __init__(self, name="admin"):
        self._registry = {}  # model_class class -> admin_class instance
        self.name = name
        self._actions = {"delete_selected": actions.delete_selected}
        self._global_actions = self._actions.copy()
        all_sites.add(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
```
### 39 - django/contrib/admin/sites.py:

Start line: 315, End line: 340

```python
class AdminSite:

    @property
    def urls(self):
        return self.get_urls(), "admin", self.name

    def each_context(self, request):
        """
        Return a dictionary of variables to put in the template context for
        *every* page in the admin site.

        For sites running on a subpath, use the SCRIPT_NAME value if site_url
        hasn't been customized.
        """
        script_name = request.META["SCRIPT_NAME"]
        site_url = (
            script_name if self.site_url == "/" and script_name else self.site_url
        )
        return {
            "site_title": self.site_title,
            "site_header": self.site_header,
            "site_url": site_url,
            "has_permission": self.has_permission(request),
            "available_apps": self.get_app_list(request),
            "is_popup": False,
            "is_nav_sidebar_enabled": self.enable_nav_sidebar,
            "log_entries": self.get_log_entries(request),
        }
```
### 42 - django/contrib/admin/sites.py:

Start line: 547, End line: 566

```python
class AdminSite:

    def index(self, request, extra_context=None):
        """
        Display the main admin index page, which lists all of the installed
        apps that have been registered in this site.
        """
        app_list = self.get_app_list(request)

        context = {
            **self.each_context(request),
            "title": self.index_title,
            "subtitle": None,
            "app_list": app_list,
            **(extra_context or {}),
        }

        request.current_app = self.name

        return TemplateResponse(
            request, self.index_template or "admin/index.html", context
        )
```
### 69 - django/contrib/admin/sites.py:

Start line: 598, End line: 612

```python
class DefaultAdminSite(LazyObject):
    def _setup(self):
        AdminSiteClass = import_string(apps.get_app_config("admin").default_site)
        self._wrapped = AdminSiteClass()

    def __repr__(self):
        return repr(self._wrapped)


# This global object represents the default admin site, for the common case.
# You can provide your own AdminSite using the (Simple)AdminConfig.default_site
# attribute. You can also instantiate AdminSite in your own code to create a
# custom admin site.
site = DefaultAdminSite()
```
### 85 - django/contrib/admin/sites.py:

Start line: 81, End line: 97

```python
class AdminSite:

    def check(self, app_configs):
        """
        Run the system checks on all ModelAdmins, except if they aren't
        customized at all.
        """
        if app_configs is None:
            app_configs = apps.get_app_configs()
        app_configs = set(app_configs)  # Speed up lookups below

        errors = []
        modeladmins = (
            o for o in self._registry.values() if o.__class__ is not ModelAdmin
        )
        for modeladmin in modeladmins:
            if modeladmin.model._meta.app_config in app_configs:
                errors.extend(modeladmin.check())
        return errors
```
### 92 - django/contrib/admin/sites.py:

Start line: 383, End line: 403

```python
class AdminSite:

    def logout(self, request, extra_context=None):
        """
        Log out the user for the given HttpRequest.

        This should *not* assume the user is already logged in.
        """
        from django.contrib.auth.views import LogoutView

        defaults = {
            "extra_context": {
                **self.each_context(request),
                # Since the user isn't logged out at this point, the value of
                # has_permission must be overridden.
                "has_permission": False,
                **(extra_context or {}),
            },
        }
        if self.logout_template is not None:
            defaults["template_name"] = self.logout_template
        request.current_app = self.name
        return LogoutView.as_view(**defaults)(request)
```
### 98 - django/contrib/admin/sites.py:

Start line: 405, End line: 441

```python
class AdminSite:

    @method_decorator(never_cache)
    def login(self, request, extra_context=None):
        """
        Display the login form for the given HttpRequest.
        """
        if request.method == "GET" and self.has_permission(request):
            # Already logged-in, redirect to admin index
            index_path = reverse("admin:index", current_app=self.name)
            return HttpResponseRedirect(index_path)

        # Since this module gets imported in the application's root package,
        # it cannot import models from other applications at the module level,
        # and django.contrib.admin.forms eventually imports User.
        from django.contrib.admin.forms import AdminAuthenticationForm
        from django.contrib.auth.views import LoginView

        context = {
            **self.each_context(request),
            "title": _("Log in"),
            "subtitle": None,
            "app_path": request.get_full_path(),
            "username": request.user.get_username(),
        }
        if (
            REDIRECT_FIELD_NAME not in request.GET
            and REDIRECT_FIELD_NAME not in request.POST
        ):
            context[REDIRECT_FIELD_NAME] = reverse("admin:index", current_app=self.name)
        context.update(extra_context or {})

        defaults = {
            "extra_context": context,
            "authentication_form": self.login_form or AdminAuthenticationForm,
            "template_name": self.login_template or "admin/login.html",
        }
        request.current_app = self.name
        return LoginView.as_view(**defaults)(request)
```
### 111 - django/contrib/admin/sites.py:

Start line: 360, End line: 381

```python
class AdminSite:

    def password_change_done(self, request, extra_context=None):
        """
        Display the "success" page after a password change.
        """
        from django.contrib.auth.views import PasswordChangeDoneView

        defaults = {
            "extra_context": {**self.each_context(request), **(extra_context or {})},
        }
        if self.password_change_done_template is not None:
            defaults["template_name"] = self.password_change_done_template
        request.current_app = self.name
        return PasswordChangeDoneView.as_view(**defaults)(request)

    def i18n_javascript(self, request, extra_context=None):
        """
        Display the i18n JavaScript that the Django admin requires.

        `extra_context` is unused but present for consistency with the other
        admin views.
        """
        return JavaScriptCatalog.as_view(packages=["django.contrib.admin"])(request)
```
### 117 - django/contrib/admin/sites.py:

Start line: 342, End line: 358

```python
class AdminSite:

    def password_change(self, request, extra_context=None):
        """
        Handle the "change password" task -- both form display and validation.
        """
        from django.contrib.admin.forms import AdminPasswordChangeForm
        from django.contrib.auth.views import PasswordChangeView

        url = reverse("admin:password_change_done", current_app=self.name)
        defaults = {
            "form_class": AdminPasswordChangeForm,
            "success_url": url,
            "extra_context": {**self.each_context(request), **(extra_context or {})},
        }
        if self.password_change_template is not None:
            defaults["template_name"] = self.password_change_template
        request.current_app = self.name
        return PasswordChangeView.as_view(**defaults)(request)
```
### 126 - django/contrib/admin/sites.py:

Start line: 531, End line: 545

```python
class AdminSite:

    def get_app_list(self, request, app_label=None):
        """
        Return a sorted list of all the installed apps that have been
        registered in this site.
        """
        app_dict = self._build_app_dict(request, app_label)

        # Sort the apps alphabetically.
        app_list = sorted(app_dict.values(), key=lambda x: x["name"].lower())

        # Sort the models alphabetically within each app.
        for app in app_list:
            app["models"].sort(key=lambda x: x["name"])

        return app_list
```
### 134 - django/contrib/admin/sites.py:

Start line: 150, End line: 202

```python
class AdminSite:

    def unregister(self, model_or_iterable):
        """
        Unregister the given model(s).

        If a model isn't already registered, raise NotRegistered.
        """
        if isinstance(model_or_iterable, ModelBase):
            model_or_iterable = [model_or_iterable]
        for model in model_or_iterable:
            if model not in self._registry:
                raise NotRegistered("The model %s is not registered" % model.__name__)
            del self._registry[model]

    def is_registered(self, model):
        """
        Check if a model class is registered with this `AdminSite`.
        """
        return model in self._registry

    def add_action(self, action, name=None):
        """
        Register an action to be available globally.
        """
        name = name or action.__name__
        self._actions[name] = action
        self._global_actions[name] = action

    def disable_action(self, name):
        """
        Disable a globally-registered action. Raise KeyError for invalid names.
        """
        del self._actions[name]

    def get_action(self, name):
        """
        Explicitly get a registered global action whether it's enabled or
        not. Raise KeyError for invalid names.
        """
        return self._global_actions[name]

    @property
    def actions(self):
        """
        Get all the enabled actions as an iterable of (name, func).
        """
        return self._actions.items()

    def has_permission(self, request):
        """
        Return True if the given HttpRequest has permission to view
        *at least one* page in the admin site.
        """
        return request.user.is_active and request.user.is_staff
```
### 135 - django/contrib/admin/sites.py:

Start line: 459, End line: 529

```python
class AdminSite:

    def _build_app_dict(self, request, label=None):
        """
        Build the app dictionary. The optional `label` parameter filters models
        of a specific app.
        """
        app_dict = {}

        if label:
            models = {
                m: m_a
                for m, m_a in self._registry.items()
                if m._meta.app_label == label
            }
        else:
            models = self._registry

        for model, model_admin in models.items():
            app_label = model._meta.app_label

            has_module_perms = model_admin.has_module_permission(request)
            if not has_module_perms:
                continue

            perms = model_admin.get_model_perms(request)

            # Check whether user has any perm for this module.
            # If so, add the module to the model_list.
            if True not in perms.values():
                continue

            info = (app_label, model._meta.model_name)
            model_dict = {
                "model": model,
                "name": capfirst(model._meta.verbose_name_plural),
                "object_name": model._meta.object_name,
                "perms": perms,
                "admin_url": None,
                "add_url": None,
            }
            if perms.get("change") or perms.get("view"):
                model_dict["view_only"] = not perms.get("change")
                try:
                    model_dict["admin_url"] = reverse(
                        "admin:%s_%s_changelist" % info, current_app=self.name
                    )
                except NoReverseMatch:
                    pass
            if perms.get("add"):
                try:
                    model_dict["add_url"] = reverse(
                        "admin:%s_%s_add" % info, current_app=self.name
                    )
                except NoReverseMatch:
                    pass

            if app_label in app_dict:
                app_dict[app_label]["models"].append(model_dict)
            else:
                app_dict[app_label] = {
                    "name": apps.get_app_config(app_label).verbose_name,
                    "app_label": app_label,
                    "app_url": reverse(
                        "admin:app_list",
                        kwargs={"app_label": app_label},
                        current_app=self.name,
                    ),
                    "has_module_perms": has_module_perms,
                    "models": [model_dict],
                }

        return app_dict
```
