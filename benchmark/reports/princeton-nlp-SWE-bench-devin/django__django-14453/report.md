# django__django-14453

| **django/django** | `ee408309d2007ecec4f43756360bd855d424cbf6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 177 |
| **Any found context length** | 177 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -626,9 +626,10 @@ def url_patterns(self):
             iter(patterns)
         except TypeError as e:
             msg = (
-                "The included URLconf '{name}' does not appear to have any "
-                "patterns in it. If you see valid patterns in the file then "
-                "the issue is probably caused by a circular import."
+                "The included URLconf '{name}' does not appear to have "
+                "any patterns in it. If you see the 'urlpatterns' variable "
+                "with valid patterns in the file then the issue is probably "
+                "caused by a circular import."
             )
             raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
         return patterns

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/urls/resolvers.py | 629 | 631 | 1 | 1 | 177


## Problem Statement

```
Added message when user mispells 'urlpatterns' in some 'urls' module
Description
	
I found this kind of error when I mispelled urlspattern instead of urlpatterns inside my blog/urls.py file.
So the console was throwing an error, but this error do not helped me to found the problem. Check it:
django.core.exceptions.ImproperlyConfigured: The included URLconf '<module 'blog.urls'
from '.../my_project/blog/urls.py'>' does not
 appear to have any patterns in it. If you see valid patterns in the file then the
 issue is probably caused by a circular import.
The problem is not with a circular import, but with the mispelled urlpatterns variable itself, so I'm doing this ticket. 
OBS.: I have already created a pull request for this: ​https://github.com/django/django/pull/14453
I appreciate any feedback.
Thanks,
Igor

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/urls/resolvers.py** | 614 | 634| 177 | 177 | 5745 | 
| 2 | **1 django/urls/resolvers.py** | 120 | 150| 232 | 409 | 5745 | 
| 3 | 2 django/core/checks/urls.py | 71 | 111| 264 | 673 | 6446 | 
| 4 | **2 django/urls/resolvers.py** | 282 | 298| 160 | 833 | 6446 | 
| 5 | **2 django/urls/resolvers.py** | 335 | 387| 370 | 1203 | 6446 | 
| 6 | 3 django/urls/conf.py | 1 | 54| 447 | 1650 | 7115 | 
| 7 | **3 django/urls/resolvers.py** | 443 | 471| 288 | 1938 | 7115 | 
| 8 | **3 django/urls/resolvers.py** | 153 | 204| 400 | 2338 | 7115 | 
| 9 | **3 django/urls/resolvers.py** | 425 | 441| 164 | 2502 | 7115 | 
| 10 | 4 django/middleware/common.py | 149 | 175| 254 | 2756 | 8644 | 
| 11 | 5 django/core/checks/templates.py | 1 | 36| 259 | 3015 | 8904 | 
| 12 | 6 django/views/i18n.py | 79 | 182| 702 | 3717 | 11365 | 
| 13 | 7 django/conf/global_settings.py | 51 | 150| 1160 | 4877 | 17052 | 
| 14 | 8 django/db/migrations/exceptions.py | 1 | 55| 249 | 5126 | 17302 | 
| 15 | 9 django/views/csrf.py | 15 | 100| 835 | 5961 | 18846 | 
| 16 | 9 django/urls/conf.py | 57 | 86| 222 | 6183 | 18846 | 
| 17 | **9 django/urls/resolvers.py** | 576 | 612| 347 | 6530 | 18846 | 
| 18 | 10 django/conf/urls/__init__.py | 1 | 10| 0 | 6530 | 18911 | 
| 19 | 11 django/urls/exceptions.py | 1 | 10| 0 | 6530 | 18936 | 
| 20 | 12 django/core/checks/security/csrf.py | 1 | 42| 304 | 6834 | 19398 | 
| 21 | **12 django/urls/resolvers.py** | 648 | 721| 681 | 7515 | 19398 | 
| 22 | 13 django/core/validators.py | 1 | 16| 127 | 7642 | 24097 | 
| 23 | 14 django/contrib/admindocs/views.py | 407 | 420| 127 | 7769 | 27436 | 
| 24 | 14 django/core/validators.py | 159 | 210| 518 | 8287 | 27436 | 
| 25 | 15 django/core/checks/messages.py | 54 | 77| 161 | 8448 | 28013 | 
| 26 | 16 django/contrib/auth/urls.py | 1 | 21| 224 | 8672 | 28237 | 
| 27 | 17 django/utils/autoreload.py | 1 | 56| 287 | 8959 | 33335 | 
| 28 | 18 django/urls/__init__.py | 1 | 24| 239 | 9198 | 33574 | 
| 29 | 18 django/core/checks/messages.py | 27 | 51| 259 | 9457 | 33574 | 
| 30 | **18 django/urls/resolvers.py** | 258 | 280| 173 | 9630 | 33574 | 
| 31 | 18 django/core/checks/security/csrf.py | 45 | 68| 157 | 9787 | 33574 | 
| 32 | 18 django/core/checks/urls.py | 1 | 27| 142 | 9929 | 33574 | 
| 33 | 18 django/core/validators.py | 102 | 156| 512 | 10441 | 33574 | 
| 34 | **18 django/urls/resolvers.py** | 636 | 646| 120 | 10561 | 33574 | 
| 35 | 19 django/contrib/messages/__init__.py | 1 | 3| 0 | 10561 | 33598 | 
| 36 | 20 django/contrib/admindocs/urls.py | 1 | 51| 307 | 10868 | 33905 | 
| 37 | 20 django/core/validators.py | 64 | 100| 575 | 11443 | 33905 | 
| 38 | 21 django/core/checks/security/base.py | 1 | 72| 660 | 12103 | 35946 | 
| 39 | 21 django/contrib/admindocs/views.py | 1 | 31| 234 | 12337 | 35946 | 
| 40 | 22 django/contrib/messages/views.py | 1 | 19| 0 | 12337 | 36042 | 
| 41 | 23 django/views/generic/__init__.py | 1 | 23| 189 | 12526 | 36232 | 
| 42 | 24 django/core/checks/model_checks.py | 155 | 176| 263 | 12789 | 38017 | 
| 43 | 24 django/conf/global_settings.py | 496 | 646| 930 | 13719 | 38017 | 
| 44 | 25 django/contrib/sites/models.py | 1 | 22| 130 | 13849 | 38805 | 
| 45 | 25 django/conf/global_settings.py | 151 | 266| 859 | 14708 | 38805 | 
| 46 | **25 django/urls/resolvers.py** | 405 | 423| 174 | 14882 | 38805 | 
| 47 | 26 django/contrib/admin/checks.py | 232 | 253| 208 | 15090 | 47987 | 
| 48 | 27 django/contrib/auth/forms.py | 57 | 75| 124 | 15214 | 51113 | 
| 49 | 28 django/db/utils.py | 1 | 49| 177 | 15391 | 53120 | 
| 50 | 29 django/contrib/postgres/utils.py | 1 | 30| 218 | 15609 | 53338 | 
| 51 | 30 django/contrib/syndication/views.py | 1 | 24| 214 | 15823 | 55065 | 
| 52 | **30 django/urls/resolvers.py** | 473 | 532| 548 | 16371 | 55065 | 
| 53 | 30 django/middleware/common.py | 118 | 147| 277 | 16648 | 55065 | 
| 54 | 30 django/core/checks/urls.py | 30 | 50| 165 | 16813 | 55065 | 
| 55 | 30 django/views/csrf.py | 1 | 13| 132 | 16945 | 55065 | 
| 56 | 31 django/core/mail/message.py | 1 | 52| 346 | 17291 | 58716 | 
| 57 | 32 django/views/defaults.py | 1 | 24| 149 | 17440 | 59744 | 
| 58 | 33 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 17578 | 59883 | 
| 59 | 34 django/core/checks/security/sessions.py | 1 | 98| 572 | 18150 | 60456 | 
| 60 | 35 django/contrib/gis/views.py | 1 | 21| 155 | 18305 | 60611 | 
| 61 | 35 django/core/checks/model_checks.py | 178 | 211| 332 | 18637 | 60611 | 
| 62 | 36 django/contrib/sitemaps/__init__.py | 1 | 29| 245 | 18882 | 62366 | 
| 63 | 36 django/core/validators.py | 256 | 271| 135 | 19017 | 62366 | 
| 64 | 37 django/views/debug.py | 1 | 55| 351 | 19368 | 67119 | 
| 65 | 38 django/contrib/flatpages/urls.py | 1 | 7| 0 | 19368 | 67157 | 
| 66 | 39 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 19505 | 67294 | 
| 67 | 40 docs/conf.py | 54 | 126| 672 | 20177 | 70703 | 
| 68 | 41 django/contrib/admin/sites.py | 1 | 35| 225 | 20402 | 75092 | 
| 69 | 42 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 20519 | 75209 | 
| 70 | 43 django/utils/module_loading.py | 1 | 24| 165 | 20684 | 75922 | 
| 71 | 43 django/core/checks/model_checks.py | 129 | 153| 268 | 20952 | 75922 | 
| 72 | 44 django/core/management/commands/makemessages.py | 283 | 362| 814 | 21766 | 81509 | 
| 73 | 44 docs/conf.py | 128 | 231| 914 | 22680 | 81509 | 
| 74 | 45 django/utils/html.py | 306 | 349| 438 | 23118 | 84611 | 
| 75 | 45 django/middleware/common.py | 1 | 32| 247 | 23365 | 84611 | 
| 76 | 45 django/core/management/commands/makemessages.py | 1 | 34| 260 | 23625 | 84611 | 
| 77 | 46 django/urls/base.py | 1 | 24| 170 | 23795 | 85794 | 
| 78 | 46 django/middleware/common.py | 100 | 115| 165 | 23960 | 85794 | 
| 79 | 47 django/forms/utils.py | 144 | 151| 132 | 24092 | 87095 | 
| 80 | 48 django/contrib/messages/apps.py | 1 | 8| 0 | 24092 | 87132 | 
| 81 | 49 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 24360 | 87400 | 
| 82 | 49 django/utils/module_loading.py | 27 | 60| 300 | 24660 | 87400 | 
| 83 | 50 django/core/checks/__init__.py | 1 | 28| 307 | 24967 | 87707 | 
| 84 | **50 django/urls/resolvers.py** | 32 | 72| 381 | 25348 | 87707 | 
| 85 | 50 django/core/checks/model_checks.py | 1 | 86| 665 | 26013 | 87707 | 
| 86 | 51 django/contrib/gis/geos/error.py | 1 | 4| 0 | 26013 | 87731 | 
| 87 | 52 django/__init__.py | 1 | 25| 173 | 26186 | 87904 | 
| 88 | 52 django/core/validators.py | 212 | 231| 170 | 26356 | 87904 | 
| 89 | 53 django/utils/deprecation.py | 1 | 33| 209 | 26565 | 88943 | 
| 90 | **53 django/urls/resolvers.py** | 1 | 29| 209 | 26774 | 88943 | 
| 91 | 54 django/db/models/base.py | 1521 | 1543| 171 | 26945 | 106255 | 
| 92 | 55 django/contrib/messages/context_processors.py | 1 | 14| 0 | 26945 | 106326 | 
| 93 | 56 django/template/backends/django.py | 79 | 111| 225 | 27170 | 107185 | 
| 94 | 57 django/contrib/sites/checks.py | 1 | 14| 0 | 27170 | 107264 | 
| 95 | 58 django/core/management/utils.py | 128 | 154| 198 | 27368 | 108378 | 
| 96 | 59 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 27490 | 108627 | 
| 97 | 60 django/contrib/messages/utils.py | 1 | 13| 0 | 27490 | 108677 | 
| 98 | 61 django/core/checks/translation.py | 1 | 65| 445 | 27935 | 109122 | 
| 99 | 62 setup.py | 1 | 48| 326 | 28261 | 109435 | 
| 100 | 62 django/core/checks/messages.py | 1 | 25| 160 | 28421 | 109435 | 
| 101 | 63 django/contrib/auth/views.py | 1 | 37| 278 | 28699 | 112129 | 
| 102 | **63 django/urls/resolvers.py** | 534 | 574| 282 | 28981 | 112129 | 
| 103 | 63 django/core/checks/security/base.py | 234 | 258| 210 | 29191 | 112129 | 
| 104 | 64 django/core/management/commands/migrate.py | 1 | 18| 140 | 29331 | 115393 | 
| 105 | 65 django/contrib/admin/exceptions.py | 1 | 12| 0 | 29331 | 115460 | 
| 106 | 66 django/conf/urls/i18n.py | 1 | 20| 127 | 29458 | 115716 | 
| 107 | 66 django/core/management/commands/makemessages.py | 216 | 281| 633 | 30091 | 115716 | 
| 108 | 67 django/core/management/commands/compilemessages.py | 59 | 116| 504 | 30595 | 117035 | 
| 109 | 68 django/conf/__init__.py | 133 | 189| 511 | 31106 | 118872 | 
| 110 | 68 django/views/defaults.py | 102 | 121| 149 | 31255 | 118872 | 
| 111 | 68 django/core/validators.py | 294 | 313| 168 | 31423 | 118872 | 
| 112 | 69 django/template/defaultfilters.py | 182 | 203| 206 | 31629 | 125102 | 
| 113 | 70 django/core/management/templates.py | 211 | 242| 236 | 31865 | 127778 | 
| 114 | 71 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 32056 | 127969 | 
| 115 | **71 django/urls/resolvers.py** | 389 | 402| 121 | 32177 | 127969 | 
| 116 | 71 django/middleware/common.py | 34 | 61| 257 | 32434 | 127969 | 
| 117 | 72 django/core/handlers/exception.py | 54 | 122| 557 | 32991 | 129037 | 
| 118 | 73 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 33146 | 131876 | 
| 119 | 73 django/template/defaultfilters.py | 340 | 424| 499 | 33645 | 131876 | 
| 120 | 74 django/core/signals.py | 1 | 7| 0 | 33645 | 131903 | 
| 121 | 75 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 33645 | 132000 | 
| 122 | 76 django/apps/config.py | 193 | 254| 580 | 34225 | 134543 | 
| 123 | 77 django/middleware/csrf.py | 1 | 53| 480 | 34705 | 138229 | 
| 124 | 77 django/db/models/base.py | 1269 | 1300| 267 | 34972 | 138229 | 
| 125 | 77 django/template/backends/django.py | 1 | 45| 303 | 35275 | 138229 | 
| 126 | 77 django/urls/base.py | 89 | 155| 383 | 35658 | 138229 | 
| 127 | 77 django/core/mail/message.py | 244 | 269| 322 | 35980 | 138229 | 
| 128 | 77 django/middleware/common.py | 63 | 75| 136 | 36116 | 138229 | 
| 129 | 78 django/contrib/auth/admin.py | 1 | 22| 188 | 36304 | 139966 | 
| 130 | 79 django/db/migrations/loader.py | 288 | 312| 205 | 36509 | 143071 | 
| 131 | 80 django/core/mail/backends/__init__.py | 1 | 2| 0 | 36509 | 143079 | 
| 132 | 80 django/core/checks/model_checks.py | 89 | 110| 168 | 36677 | 143079 | 
| 133 | 81 django/contrib/flatpages/sitemaps.py | 1 | 13| 112 | 36789 | 143191 | 
| 134 | 82 django/utils/log.py | 1 | 75| 484 | 37273 | 144833 | 
| 135 | 83 django/apps/registry.py | 213 | 233| 237 | 37510 | 148240 | 
| 136 | 84 django/utils/http.py | 317 | 355| 402 | 37912 | 151482 | 
| 137 | **84 django/urls/resolvers.py** | 301 | 332| 190 | 38102 | 151482 | 
| 138 | 85 django/contrib/sitemaps/views.py | 1 | 19| 132 | 38234 | 152275 | 
| 139 | 86 django/contrib/gis/gdal/prototypes/errcheck.py | 1 | 35| 221 | 38455 | 153258 | 
| 140 | 87 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 38455 | 153334 | 
| 141 | 87 django/contrib/admin/sites.py | 242 | 296| 516 | 38971 | 153334 | 
| 142 | 87 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 39096 | 153334 | 
| 143 | 88 django/http/request.py | 1 | 39| 273 | 39369 | 158540 | 
| 144 | 89 django/views/__init__.py | 1 | 4| 0 | 39369 | 158555 | 
| 145 | 89 django/core/management/commands/makemessages.py | 363 | 400| 272 | 39641 | 158555 | 
| 146 | 89 django/utils/autoreload.py | 109 | 116| 114 | 39755 | 158555 | 
| 147 | 90 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 39902 | 158703 | 
| 148 | 90 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 40724 | 158703 | 
| 149 | 90 django/views/debug.py | 189 | 201| 143 | 40867 | 158703 | 
| 150 | 91 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 41119 | 159000 | 
| 151 | 91 django/core/management/commands/compilemessages.py | 1 | 27| 161 | 41280 | 159000 | 
| 152 | 92 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 41587 | 159307 | 
| 153 | 92 django/middleware/csrf.py | 273 | 316| 372 | 41959 | 159307 | 
| 154 | 93 django/utils/baseconv.py | 1 | 55| 147 | 42106 | 160149 | 
| 155 | 94 django/db/backends/base/creation.py | 301 | 322| 258 | 42364 | 162937 | 
| 156 | 94 django/core/checks/security/base.py | 74 | 168| 746 | 43110 | 162937 | 
| 157 | 94 django/contrib/sitemaps/views.py | 48 | 93| 409 | 43519 | 162937 | 
| 158 | 94 django/utils/html.py | 352 | 379| 212 | 43731 | 162937 | 
| 159 | 95 django/contrib/admin/options.py | 1 | 97| 761 | 44492 | 181549 | 
| 160 | 95 django/utils/html.py | 291 | 304| 154 | 44646 | 181549 | 
| 161 | 96 django/contrib/sessions/exceptions.py | 1 | 17| 0 | 44646 | 181620 | 
| 162 | 96 django/utils/autoreload.py | 59 | 87| 156 | 44802 | 181620 | 
| 163 | 96 django/contrib/admin/checks.py | 794 | 815| 190 | 44992 | 181620 | 
| 164 | 97 django/contrib/sites/middleware.py | 1 | 13| 0 | 44992 | 181679 | 
| 165 | 97 django/views/csrf.py | 101 | 155| 577 | 45569 | 181679 | 
| 166 | 98 django/utils/translation/reloader.py | 1 | 36| 228 | 45797 | 181908 | 
| 167 | 99 django/contrib/gis/geoip2/base.py | 1 | 21| 145 | 45942 | 183924 | 
| 168 | 99 django/utils/html.py | 259 | 289| 321 | 46263 | 183924 | 
| 169 | 100 django/db/migrations/graph.py | 259 | 280| 179 | 46442 | 186527 | 
| 170 | 101 django/contrib/auth/__init__.py | 1 | 38| 241 | 46683 | 188108 | 
| 171 | 101 django/core/validators.py | 19 | 61| 336 | 47019 | 188108 | 
| 172 | 102 django/template/library.py | 312 | 329| 110 | 47129 | 190645 | 
| 173 | 102 django/urls/base.py | 27 | 86| 438 | 47567 | 190645 | 
| 174 | 102 django/core/checks/urls.py | 53 | 68| 128 | 47695 | 190645 | 
| 175 | 102 django/forms/utils.py | 79 | 142| 379 | 48074 | 190645 | 
| 176 | 102 django/core/validators.py | 274 | 291| 150 | 48224 | 190645 | 
| 177 | 103 django/contrib/redirects/middleware.py | 1 | 51| 354 | 48578 | 191000 | 
| 178 | 103 django/utils/autoreload.py | 315 | 330| 146 | 48724 | 191000 | 
| 179 | 103 django/views/defaults.py | 124 | 151| 198 | 48922 | 191000 | 
| 180 | 104 django/contrib/redirects/apps.py | 1 | 9| 0 | 48922 | 191051 | 
| 181 | 104 django/contrib/admin/checks.py | 58 | 143| 718 | 49640 | 191051 | 
| 182 | 105 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 50045 | 191456 | 
| 183 | 106 django/contrib/gis/db/backends/oracle/operations.py | 38 | 49| 202 | 50247 | 193538 | 
| 184 | 107 django/http/response.py | 496 | 514| 186 | 50433 | 198120 | 


## Patch

```diff
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -626,9 +626,10 @@ def url_patterns(self):
             iter(patterns)
         except TypeError as e:
             msg = (
-                "The included URLconf '{name}' does not appear to have any "
-                "patterns in it. If you see valid patterns in the file then "
-                "the issue is probably caused by a circular import."
+                "The included URLconf '{name}' does not appear to have "
+                "any patterns in it. If you see the 'urlpatterns' variable "
+                "with valid patterns in the file then the issue is probably "
+                "caused by a circular import."
             )
             raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
         return patterns

```

## Test Patch

```diff
diff --git a/tests/urlpatterns_reverse/tests.py b/tests/urlpatterns_reverse/tests.py
--- a/tests/urlpatterns_reverse/tests.py
+++ b/tests/urlpatterns_reverse/tests.py
@@ -271,8 +271,9 @@ def test_no_urls_exception(self):
         with self.assertRaisesMessage(
             ImproperlyConfigured,
             "The included URLconf 'urlpatterns_reverse.no_urls' does not "
-            "appear to have any patterns in it. If you see valid patterns in "
-            "the file then the issue is probably caused by a circular import."
+            "appear to have any patterns in it. If you see the 'urlpatterns' "
+            "variable with valid patterns in the file then the issue is "
+            "probably caused by a circular import."
         ):
             getattr(resolver, 'url_patterns')
 
@@ -1095,8 +1096,9 @@ class NoRootUrlConfTests(SimpleTestCase):
     def test_no_handler_exception(self):
         msg = (
             "The included URLconf 'None' does not appear to have any patterns "
-            "in it. If you see valid patterns in the file then the issue is "
-            "probably caused by a circular import."
+            "in it. If you see the 'urlpatterns' variable with valid patterns "
+            "in the file then the issue is probably caused by a circular "
+            "import."
         )
         with self.assertRaisesMessage(ImproperlyConfigured, msg):
             self.client.get('/test/me/')

```


## Code snippets

### 1 - django/urls/resolvers.py:

Start line: 614, End line: 634

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
                "The included URLconf '{name}' does not appear to have any "
                "patterns in it. If you see valid patterns in the file then "
                "the issue is probably caused by a circular import."
            )
            raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
        return patterns
```
### 2 - django/urls/resolvers.py:

Start line: 120, End line: 150

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
        if regex_pattern.startswith(('/', '^/', '^\\/')) and not regex_pattern.endswith('/'):
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
### 3 - django/core/checks/urls.py:

Start line: 71, End line: 111

```python
def get_warning_for_invalid_pattern(pattern):
    """
    Return a list containing a warning that the pattern is invalid.

    describe_pattern() cannot be used here, because we cannot rely on the
    urlpattern having regex or name attributes.
    """
    if isinstance(pattern, str):
        hint = (
            "Try removing the string '{}'. The list of urlpatterns should not "
            "have a prefix string as the first element.".format(pattern)
        )
    elif isinstance(pattern, tuple):
        hint = "Try using path() instead of a tuple."
    else:
        hint = None

    return [Error(
        "Your URL pattern {!r} is invalid. Ensure that urlpatterns is a list "
        "of path() and/or re_path() instances.".format(pattern),
        hint=hint,
        id="urls.E004",
    )]


@register(Tags.urls)
def check_url_settings(app_configs, **kwargs):
    errors = []
    for name in ('STATIC_URL', 'MEDIA_URL'):
        value = getattr(settings, name)
        if value and not value.endswith('/'):
            errors.append(E006(name))
    return errors


def E006(name):
    return Error(
        'The {} setting must end with a slash.'.format(name),
        id='urls.E006',
    )
```
### 4 - django/urls/resolvers.py:

Start line: 282, End line: 298

```python
class RoutePattern(CheckURLMixin):

    def check(self):
        warnings = self._check_pattern_startswith_slash()
        route = self._route
        if '(?P<' in route or route.startswith('^') or route.endswith('$'):
            warnings.append(Warning(
                "Your URL pattern {} has a route that contains '(?P<', begins "
                "with a '^', or ends with a '$'. This was likely an oversight "
                "when migrating to django.urls.path().".format(self.describe()),
                id='2_0.W001',
            ))
        return warnings

    def _compile(self, route):
        return re.compile(_route_to_regex(route, self._is_endpoint)[0])

    def __str__(self):
        return str(self._route)
```
### 5 - django/urls/resolvers.py:

Start line: 335, End line: 387

```python
class URLPattern:
    def __init__(self, pattern, callback, default_args=None, name=None):
        self.pattern = pattern
        self.callback = callback  # the view
        self.default_args = default_args or {}
        self.name = name

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.pattern.describe())

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
            return [Error(
                'Your URL pattern %s has an invalid view, pass %s.as_view() '
                'instead of %s.' % (
                    self.pattern.describe(),
                    view.__name__,
                    view.__name__,
                ),
                id='urls.E009',
            )]
        return []

    def resolve(self, path):
        match = self.pattern.match(path)
        if match:
            new_path, args, kwargs = match
            # Pass any extra_kwargs as **kwargs.
            kwargs.update(self.default_args)
            return ResolverMatch(self.callback, args, kwargs, self.pattern.name, route=str(self.pattern))
```
### 6 - django/urls/conf.py:

Start line: 1, End line: 54

```python
"""Functions for use in URLsconfs."""
from functools import partial
from importlib import import_module

from django.core.exceptions import ImproperlyConfigured

from .resolvers import (
    LocalePrefixPattern, RegexPattern, RoutePattern, URLPattern, URLResolver,
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
                    'Cannot override the namespace for a dynamic module that '
                    'provides a namespace.'
                )
            raise ImproperlyConfigured(
                'Passing a %d-tuple to include() is not supported. Pass a '
                '2-tuple containing the list of patterns and app_name, and '
                'provide the namespace argument to include() instead.' % len(arg)
            )
    else:
        # No namespace hint - use manually provided namespace.
        urlconf_module = arg

    if isinstance(urlconf_module, str):
        urlconf_module = import_module(urlconf_module)
    patterns = getattr(urlconf_module, 'urlpatterns', urlconf_module)
    app_name = getattr(urlconf_module, 'app_name', app_name)
    if namespace and not app_name:
        raise ImproperlyConfigured(
            'Specifying a namespace in include() without providing an app_name '
            'is not supported. Set the app_name attribute in the included '
            'module, or pass a 2-tuple containing the list of patterns and '
            'app_name instead.',
        )
    namespace = namespace or app_name
    # Make sure the patterns can be iterated through (without this, some
    # testcases will break).
    if isinstance(patterns, (list, tuple)):
        for url_pattern in patterns:
            pattern = getattr(url_pattern, 'pattern', None)
            if isinstance(pattern, LocalePrefixPattern):
                raise ImproperlyConfigured(
                    'Using i18n_patterns in an included URLconf is not allowed.'
                )
    return (urlconf_module, app_name, namespace)
```
### 7 - django/urls/resolvers.py:

Start line: 443, End line: 471

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
                path = getattr(self.urlconf_module, 'handler%s' % status_code)
                msg = (
                    "The custom handler{status_code} view '{path}' could not be imported."
                ).format(status_code=status_code, path=path)
                messages.append(Error(msg, hint=str(e), id='urls.E008'))
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
                    path=handler.__module__ + '.' + handler.__qualname__,
                    args='request, exception' if num_parameters == 2 else 'request',
                )
                messages.append(Error(msg, id='urls.E007'))
        return messages
```
### 8 - django/urls/resolvers.py:

Start line: 153, End line: 204

```python
class RegexPattern(CheckURLMixin):
    regex = LocaleRegexDescriptor('_regex')

    def __init__(self, regex, name=None, is_endpoint=False):
        self._regex = regex
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = {}

    def match(self, path):
        match = self.regex.search(path)
        if match:
            # If there are any named groups, use those as kwargs, ignoring
            # non-named groups. Otherwise, pass all non-named arguments as
            # positional arguments.
            kwargs = match.groupdict()
            args = () if kwargs else match.groups()
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return path[match.end():], args, kwargs
        return None

    def check(self):
        warnings = []
        warnings.extend(self._check_pattern_startswith_slash())
        if not self._is_endpoint:
            warnings.extend(self._check_include_trailing_dollar())
        return warnings

    def _check_include_trailing_dollar(self):
        regex_pattern = self.regex.pattern
        if regex_pattern.endswith('$') and not regex_pattern.endswith(r'\$'):
            return [Warning(
                "Your URL pattern {} uses include with a route ending with a '$'. "
                "Remove the dollar from the route to avoid problems including "
                "URLs.".format(self.describe()),
                id='urls.W001',
            )]
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
```
### 9 - django/urls/resolvers.py:

Start line: 425, End line: 441

```python
class URLResolver:

    def __repr__(self):
        if isinstance(self.urlconf_name, list) and self.urlconf_name:
            # Don't bother to output the whole list, it can be huge
            urlconf_repr = '<%s list>' % self.urlconf_name[0].__class__.__name__
        else:
            urlconf_repr = repr(self.urlconf_name)
        return '<%s %s (%s:%s) %s>' % (
            self.__class__.__name__, urlconf_repr, self.app_name,
            self.namespace, self.pattern.describe(),
        )

    def check(self):
        messages = []
        for pattern in self.url_patterns:
            messages.extend(check_resolver(pattern))
        messages.extend(self._check_custom_error_handlers())
        return messages or self.pattern.check()
```
### 10 - django/middleware/common.py:

Start line: 149, End line: 175

```python
class BrokenLinkEmailsMiddleware(MiddlewareMixin):

    def is_ignorable_request(self, request, uri, domain, referer):
        """
        Return True if the given request *shouldn't* notify the site managers
        according to project settings or in situations outlined by the inline
        comments.
        """
        # The referer is empty.
        if not referer:
            return True

        # APPEND_SLASH is enabled and the referer is equal to the current URL
        # without a trailing slash indicating an internal redirect.
        if settings.APPEND_SLASH and uri.endswith('/') and referer == uri[:-1]:
            return True

        # A '?' in referer is identified as a search engine source.
        if not self.is_internal_request(domain, referer) and '?' in referer:
            return True

        # The referer is equal to the current URL, ignoring the scheme (assumed
        # to be a poorly implemented bot).
        parsed_referer = urlparse(referer)
        if parsed_referer.netloc in ['', domain] and parsed_referer.path == uri:
            return True

        return any(pattern.search(uri) for pattern in settings.IGNORABLE_404_URLS)
```
### 17 - django/urls/resolvers.py:

Start line: 576, End line: 612

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
                    self._extend_tried(tried, pattern, e.args[0].get('tried'))
                else:
                    if sub_match:
                        # Merge captured arguments in match with submatch
                        sub_match_dict = {**kwargs, **self.default_kwargs}
                        # Update the sub_match_dict with the kwargs from the sub_match.
                        sub_match_dict.update(sub_match.kwargs)
                        # If there are *any* named groups, ignore all non-named groups.
                        # Otherwise, pass all non-named arguments as positional arguments.
                        sub_match_args = sub_match.args
                        if not sub_match_dict:
                            sub_match_args = args + sub_match.args
                        current_route = '' if isinstance(pattern, URLPattern) else str(pattern.pattern)
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
                        )
                    tried.append([pattern])
            raise Resolver404({'tried': tried, 'path': new_path})
        raise Resolver404({'path': path})
```
### 21 - django/urls/resolvers.py:

Start line: 648, End line: 721

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
                    if any(kwargs.get(k, v) != v for k, v in defaults.items()):
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
                candidate_pat = _prefix.replace('%', '%%') + result
                if re.search('^%s%s' % (re.escape(_prefix), pattern), candidate_pat % text_candidate_subs):
                    # safe characters from `pchar` definition of RFC 3986
                    url = quote(candidate_pat % text_candidate_subs, safe=RFC3986_SUBDELIMS + '/~:@')
                    # Don't allow construction of scheme relative urls.
                    return escape_leading_slashes(url)
        # lookup_view can be URL name or callable, but callables are not
        # friendly in error messages.
        m = getattr(lookup_view, '__module__', None)
        n = getattr(lookup_view, '__name__', None)
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
            msg = (
                "Reverse for '%s' with %s not found. %d pattern(s) tried: %s" %
                (lookup_view_s, arg_msg, len(patterns), patterns)
            )
        else:
            msg = (
                "Reverse for '%(view)s' not found. '%(view)s' is not "
                "a valid view function or pattern name." % {'view': lookup_view_s}
            )
        raise NoReverseMatch(msg)
```
### 30 - django/urls/resolvers.py:

Start line: 258, End line: 280

```python
class RoutePattern(CheckURLMixin):
    regex = LocaleRegexDescriptor('_route')

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
            return path[match.end():], (), kwargs
        return None
```
### 34 - django/urls/resolvers.py:

Start line: 636, End line: 646

```python
class URLResolver:

    def resolve_error_handler(self, view_type):
        callback = getattr(self.urlconf_module, 'handler%s' % view_type, None)
        if not callback:
            # No handler specified in file; use lazy import, since
            # django.conf.urls imports this file.
            from django.conf import urls
            callback = getattr(urls, 'handler%s' % view_type)
        return get_callable(callback)

    def reverse(self, lookup_view, *args, **kwargs):
        return self._reverse_with_prefix(lookup_view, '', *args, **kwargs)
```
### 46 - django/urls/resolvers.py:

Start line: 405, End line: 423

```python
class URLResolver:
    def __init__(self, pattern, urlconf_name, default_kwargs=None, app_name=None, namespace=None):
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
### 52 - django/urls/resolvers.py:

Start line: 473, End line: 532

```python
class URLResolver:

    def _populate(self):
        # Short-circuit if called recursively in this thread to prevent
        # infinite recursion. Concurrent threads may call this at the same
        # time and will need to continue, so set 'populating' on a
        # thread-local variable.
        if getattr(self._local, 'populating', False):
            return
        try:
            self._local.populating = True
            lookups = MultiValueDict()
            namespaces = {}
            apps = {}
            language_code = get_language()
            for url_pattern in reversed(self.url_patterns):
                p_pattern = url_pattern.pattern.regex.pattern
                if p_pattern.startswith('^'):
                    p_pattern = p_pattern[1:]
                if isinstance(url_pattern, URLPattern):
                    self._callback_strs.add(url_pattern.lookup_str)
                    bits = normalize(url_pattern.pattern.regex.pattern)
                    lookups.appendlist(
                        url_pattern.callback,
                        (bits, p_pattern, url_pattern.default_args, url_pattern.pattern.converters)
                    )
                    if url_pattern.name is not None:
                        lookups.appendlist(
                            url_pattern.name,
                            (bits, p_pattern, url_pattern.default_args, url_pattern.pattern.converters)
                        )
                else:  # url_pattern is a URLResolver.
                    url_pattern._populate()
                    if url_pattern.app_name:
                        apps.setdefault(url_pattern.app_name, []).append(url_pattern.namespace)
                        namespaces[url_pattern.namespace] = (p_pattern, url_pattern)
                    else:
                        for name in url_pattern.reverse_dict:
                            for matches, pat, defaults, converters in url_pattern.reverse_dict.getlist(name):
                                new_matches = normalize(p_pattern + pat)
                                lookups.appendlist(
                                    name,
                                    (
                                        new_matches,
                                        p_pattern + pat,
                                        {**defaults, **url_pattern.default_kwargs},
                                        {**self.pattern.converters, **url_pattern.pattern.converters, **converters}
                                    )
                                )
                        for namespace, (prefix, sub_pattern) in url_pattern.namespace_dict.items():
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
### 84 - django/urls/resolvers.py:

Start line: 32, End line: 72

```python
class ResolverMatch:
    def __init__(self, func, args, kwargs, url_name=None, app_names=None, namespaces=None, route=None, tried=None):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.url_name = url_name
        self.route = route
        self.tried = tried

        # If a URLRegexResolver doesn't have a namespace or app_name, it passes
        # in an empty value.
        self.app_names = [x for x in app_names if x] if app_names else []
        self.app_name = ':'.join(self.app_names)
        self.namespaces = [x for x in namespaces if x] if namespaces else []
        self.namespace = ':'.join(self.namespaces)

        if not hasattr(func, '__name__'):
            # A class-based view
            self._func_path = func.__class__.__module__ + '.' + func.__class__.__name__
        else:
            # A function-based view
            self._func_path = func.__module__ + '.' + func.__name__

        view_path = url_name or self._func_path
        self.view_name = ':'.join(self.namespaces + [view_path])

    def __getitem__(self, index):
        return (self.func, self.args, self.kwargs)[index]

    def __repr__(self):
        if isinstance(self.func, functools.partial):
            func = repr(self.func)
        else:
            func = self._func_path
        return (
            'ResolverMatch(func=%s, args=%r, kwargs=%r, url_name=%r, '
            'app_names=%r, namespaces=%r, route=%r)' % (
                func, self.args, self.kwargs, self.url_name,
                self.app_names, self.namespaces, self.route,
            )
        )
```
### 90 - django/urls/resolvers.py:

Start line: 1, End line: 29

```python
"""
This module converts requested URLs to callback view functions.

URLResolver is the main class here. Its resolve() method takes a URL (as
a string) and returns a ResolverMatch object which provides access to all
attributes of the resolved URL match.
"""
import functools
import inspect
import re
import string
from importlib import import_module
from urllib.parse import quote

from asgiref.local import Local

from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language

from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
```
### 102 - django/urls/resolvers.py:

Start line: 534, End line: 574

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
        if route2.startswith('^'):
            route2 = route2[1:]
        return route1 + route2

    def _is_callback(self, name):
        if not self._populated:
            self._populate()
        return name in self._callback_strs
```
### 115 - django/urls/resolvers.py:

Start line: 389, End line: 402

```python
class URLPattern:

    @cached_property
    def lookup_str(self):
        """
        A string that identifies the view (e.g. 'path.to.view_function' or
        'path.to.ClassBasedView').
        """
        callback = self.callback
        if isinstance(callback, functools.partial):
            callback = callback.func
        if hasattr(callback, 'view_class'):
            callback = callback.view_class
        elif not hasattr(callback, '__name__'):
            return callback.__module__ + "." + callback.__class__.__name__
        return callback.__module__ + "." + callback.__qualname__
```
### 137 - django/urls/resolvers.py:

Start line: 301, End line: 332

```python
class LocalePrefixPattern:
    def __init__(self, prefix_default_language=True):
        self.prefix_default_language = prefix_default_language
        self.converters = {}

    @property
    def regex(self):
        # This is only used by reverse() and cached in _reverse_dict.
        return re.compile(self.language_prefix)

    @property
    def language_prefix(self):
        language_code = get_language() or settings.LANGUAGE_CODE
        if language_code == settings.LANGUAGE_CODE and not self.prefix_default_language:
            return ''
        else:
            return '%s/' % language_code

    def match(self, path):
        language_prefix = self.language_prefix
        if path.startswith(language_prefix):
            return path[len(language_prefix):], (), {}
        return None

    def check(self):
        return []

    def describe(self):
        return "'{}'".format(self)

    def __str__(self):
        return self.language_prefix
```
