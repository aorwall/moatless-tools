# django__django-11166

| **django/django** | `85676979a4845fa9b586ec42d4ddbdb9f28b7cc8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 535 |
| **Any found context length** | 535 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -65,7 +65,6 @@ def check_dependencies(**kwargs):
         ('django.contrib.contenttypes', 401),
         ('django.contrib.auth', 405),
         ('django.contrib.messages', 406),
-        ('django.contrib.sessions', 407),
     )
     for app_name, error_code in app_dependencies:
         if not apps.is_installed(app_name):
@@ -118,6 +117,12 @@ def check_dependencies(**kwargs):
             "be in MIDDLEWARE in order to use the admin application.",
             id='admin.E409',
         ))
+    if not _contains_subclass('django.contrib.sessions.middleware.SessionMiddleware', settings.MIDDLEWARE):
+        errors.append(checks.Error(
+            "'django.contrib.sessions.middleware.SessionMiddleware' must "
+            "be in MIDDLEWARE in order to use the admin application.",
+            id='admin.E410',
+        ))
     return errors
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 68 | 68 | 1 | 1 | 535
| django/contrib/admin/checks.py | 121 | 121 | 1 | 1 | 535


## Problem Statement

```
Admin app has too hard a dependency on sessions app
Description
	
Since #29695 (371ece2f0682e51f2f796854d3e091827a7cea63), released in 2.2, the admin app checks whether the django.contrib.sessions app is in INSTALLED_APPS.
Some projects may have opted to use a replacement session management app such as ​https://github.com/QueraTeam/django-qsessions – the admin app claims to be incompatible with such a configuration, even if it actually means "I'm going to need _some_ session management that works like django.contrib.sessions".
Maybe it would be better to get rid of the app check and do what's being done for various middleware in the checks function anyway, e.g. something like
if not _contains_subclass('django.contrib.sessions.middleware.SessionMiddleware', settings.MIDDLEWARE):
	errors.append(checks.Error(
		"'django.contrib.sessions.middleware.SessionMiddleware' must "
		"be in MIDDLEWARE in order to use the admin application.",
		id='admin.E4XX',
	))
– this would be out-of-the-box compatible with e.g. Qsessions.
The obvious workaround is to just re-add django.contrib.sessions back into INSTALLED_APPS which kinda works, but has the mild but unfortunate side effect of forcibly enabling the django.contrib.sessions.models.Session model and migrations, (re-)adding a useless django_session table into the database upon migration.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/checks.py** | 57 | 121| 535 | 535 | 8961 | 
| 2 | 2 django/core/checks/security/sessions.py | 1 | 98| 572 | 1107 | 9534 | 
| 3 | **2 django/contrib/admin/checks.py** | 1 | 54| 329 | 1436 | 9534 | 
| 4 | 3 django/contrib/admin/apps.py | 1 | 25| 148 | 1584 | 9682 | 
| 5 | 4 django/contrib/auth/admin.py | 1 | 22| 188 | 1772 | 11408 | 
| 6 | **4 django/contrib/admin/checks.py** | 124 | 141| 155 | 1927 | 11408 | 
| 7 | 5 django/contrib/admin/sites.py | 1 | 29| 175 | 2102 | 15599 | 
| 8 | **5 django/contrib/admin/checks.py** | 615 | 634| 183 | 2285 | 15599 | 
| 9 | 6 django/contrib/admin/__init__.py | 1 | 30| 286 | 2571 | 15885 | 
| 10 | 7 django/contrib/sessions/middleware.py | 1 | 76| 578 | 3149 | 16464 | 
| 11 | 7 django/contrib/admin/sites.py | 70 | 84| 129 | 3278 | 16464 | 
| 12 | 8 django/core/checks/security/base.py | 88 | 190| 747 | 4025 | 18090 | 
| 13 | **8 django/contrib/admin/checks.py** | 1008 | 1035| 204 | 4229 | 18090 | 
| 14 | 8 django/contrib/admin/sites.py | 512 | 546| 293 | 4522 | 18090 | 
| 15 | 9 django/contrib/sessions/apps.py | 1 | 8| 0 | 4522 | 18127 | 
| 16 | 10 django/contrib/auth/checks.py | 1 | 94| 646 | 5168 | 19300 | 
| 17 | 11 django/contrib/admin/tests.py | 1 | 112| 812 | 5980 | 20716 | 
| 18 | **11 django/contrib/admin/checks.py** | 955 | 991| 281 | 6261 | 20716 | 
| 19 | **11 django/contrib/admin/checks.py** | 591 | 612| 162 | 6423 | 20716 | 
| 20 | 12 django/contrib/admindocs/views.py | 1 | 29| 216 | 6639 | 24026 | 
| 21 | 12 django/contrib/admin/sites.py | 219 | 238| 221 | 6860 | 24026 | 
| 22 | 13 django/contrib/admin/options.py | 525 | 539| 169 | 7029 | 42379 | 
| 23 | **13 django/contrib/admin/checks.py** | 578 | 589| 128 | 7157 | 42379 | 
| 24 | 13 django/contrib/auth/checks.py | 97 | 167| 525 | 7682 | 42379 | 
| 25 | 14 django/core/checks/security/csrf.py | 1 | 41| 299 | 7981 | 42678 | 
| 26 | 15 django/contrib/sessions/__init__.py | 1 | 2| 0 | 7981 | 42691 | 
| 27 | 15 django/contrib/admin/sites.py | 240 | 289| 472 | 8453 | 42691 | 
| 28 | 16 django/contrib/auth/apps.py | 1 | 29| 213 | 8666 | 42904 | 
| 29 | **16 django/contrib/admin/checks.py** | 320 | 346| 221 | 8887 | 42904 | 
| 30 | 17 django/contrib/sessions/backends/base.py | 1 | 35| 202 | 9089 | 45429 | 
| 31 | 18 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 9411 | 45751 | 
| 32 | 19 django/core/checks/model_checks.py | 134 | 167| 332 | 9743 | 47153 | 
| 33 | **19 django/contrib/admin/checks.py** | 1037 | 1079| 343 | 10086 | 47153 | 
| 34 | 19 django/contrib/sessions/backends/base.py | 127 | 207| 547 | 10633 | 47153 | 
| 35 | 19 django/core/checks/security/base.py | 193 | 211| 127 | 10760 | 47153 | 
| 36 | 19 django/contrib/admin/tests.py | 114 | 126| 154 | 10914 | 47153 | 
| 37 | **19 django/contrib/admin/checks.py** | 994 | 1006| 116 | 11030 | 47153 | 
| 38 | **19 django/contrib/admin/checks.py** | 348 | 364| 134 | 11164 | 47153 | 
| 39 | **19 django/contrib/admin/checks.py** | 874 | 922| 416 | 11580 | 47153 | 
| 40 | 20 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 11742 | 47315 | 
| 41 | 20 django/core/checks/model_checks.py | 111 | 132| 263 | 12005 | 47315 | 
| 42 | 20 django/contrib/auth/admin.py | 128 | 189| 465 | 12470 | 47315 | 
| 43 | 21 django/contrib/auth/middleware.py | 1 | 24| 193 | 12663 | 48331 | 
| 44 | 21 django/core/checks/model_checks.py | 85 | 109| 268 | 12931 | 48331 | 
| 45 | 22 django/conf/global_settings.py | 499 | 638| 853 | 13784 | 53935 | 
| 46 | **22 django/contrib/admin/checks.py** | 665 | 699| 265 | 14049 | 53935 | 
| 47 | 23 django/contrib/admindocs/apps.py | 1 | 8| 0 | 14049 | 53977 | 
| 48 | 24 django/contrib/auth/__init__.py | 1 | 58| 393 | 14442 | 55544 | 
| 49 | **24 django/contrib/admin/checks.py** | 198 | 208| 127 | 14569 | 55544 | 
| 50 | **24 django/contrib/admin/checks.py** | 636 | 663| 232 | 14801 | 55544 | 
| 51 | 25 django/core/management/commands/check.py | 1 | 34| 226 | 15027 | 55979 | 
| 52 | 25 django/contrib/admin/options.py | 2135 | 2170| 315 | 15342 | 55979 | 
| 53 | 25 django/contrib/admin/tests.py | 148 | 162| 160 | 15502 | 55979 | 
| 54 | 26 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 15502 | 56080 | 
| 55 | 27 django/utils/module_loading.py | 27 | 60| 300 | 15802 | 56823 | 
| 56 | 28 django/apps/registry.py | 298 | 329| 236 | 16038 | 60230 | 
| 57 | 29 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 16653 | 61255 | 
| 58 | **29 django/contrib/admin/checks.py** | 155 | 196| 325 | 16978 | 61255 | 
| 59 | **29 django/contrib/admin/checks.py** | 713 | 744| 227 | 17205 | 61255 | 
| 60 | 30 django/db/migrations/loader.py | 277 | 301| 205 | 17410 | 64172 | 
| 61 | **30 django/contrib/admin/checks.py** | 225 | 255| 229 | 17639 | 64172 | 
| 62 | 31 django/contrib/sessions/base_session.py | 26 | 48| 139 | 17778 | 64461 | 
| 63 | 31 django/contrib/admin/options.py | 1 | 96| 769 | 18547 | 64461 | 
| 64 | 32 django/contrib/admin/utils.py | 306 | 363| 468 | 19015 | 68329 | 
| 65 | 33 django/contrib/sessions/models.py | 1 | 36| 250 | 19265 | 68579 | 
| 66 | 33 django/contrib/admin/options.py | 1517 | 1596| 719 | 19984 | 68579 | 
| 67 | 33 django/contrib/auth/middleware.py | 47 | 83| 360 | 20344 | 68579 | 
| 68 | **33 django/contrib/admin/checks.py** | 924 | 953| 243 | 20587 | 68579 | 
| 69 | 33 django/core/checks/model_checks.py | 1 | 42| 282 | 20869 | 68579 | 
| 70 | 33 django/contrib/auth/admin.py | 101 | 126| 286 | 21155 | 68579 | 
| 71 | 34 django/utils/deprecation.py | 76 | 98| 158 | 21313 | 69271 | 
| 72 | 35 django/contrib/admin/views/main.py | 1 | 34| 244 | 21557 | 73348 | 
| 73 | 36 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 21679 | 73597 | 
| 74 | 36 django/core/checks/security/base.py | 1 | 86| 752 | 22431 | 73597 | 
| 75 | 36 django/contrib/sessions/backends/base.py | 109 | 125| 173 | 22604 | 73597 | 
| 76 | **36 django/contrib/admin/checks.py** | 210 | 223| 161 | 22765 | 73597 | 
| 77 | 36 django/contrib/admin/sites.py | 409 | 475| 476 | 23241 | 73597 | 
| 78 | 37 django/contrib/admin/helpers.py | 152 | 190| 343 | 23584 | 76773 | 
| 79 | 37 django/contrib/admin/options.py | 1725 | 1806| 744 | 24328 | 76773 | 
| 80 | **37 django/contrib/admin/checks.py** | 1082 | 1112| 188 | 24516 | 76773 | 
| 81 | 37 django/contrib/auth/admin.py | 25 | 37| 128 | 24644 | 76773 | 
| 82 | 37 django/contrib/admin/options.py | 99 | 129| 223 | 24867 | 76773 | 
| 83 | **37 django/contrib/admin/checks.py** | 763 | 784| 190 | 25057 | 76773 | 
| 84 | **37 django/contrib/admin/checks.py** | 143 | 153| 123 | 25180 | 76773 | 
| 85 | **37 django/contrib/admin/checks.py** | 257 | 270| 135 | 25315 | 76773 | 
| 86 | 37 django/contrib/sessions/base_session.py | 1 | 23| 149 | 25464 | 76773 | 
| 87 | 37 django/contrib/admin/options.py | 368 | 420| 504 | 25968 | 76773 | 
| 88 | 37 django/contrib/admin/options.py | 1917 | 1960| 403 | 26371 | 76773 | 
| 89 | **37 django/contrib/admin/checks.py** | 470 | 480| 149 | 26520 | 76773 | 
| 90 | 37 django/contrib/sessions/backends/base.py | 38 | 107| 513 | 27033 | 76773 | 
| 91 | 37 django/contrib/admin/sites.py | 477 | 510| 228 | 27261 | 76773 | 
| 92 | 38 django/contrib/admin/forms.py | 1 | 31| 184 | 27445 | 76957 | 
| 93 | 38 django/contrib/admin/sites.py | 32 | 68| 298 | 27743 | 76957 | 
| 94 | **38 django/contrib/admin/checks.py** | 541 | 576| 303 | 28046 | 76957 | 
| 95 | 38 django/contrib/admin/options.py | 864 | 878| 125 | 28171 | 76957 | 
| 96 | 38 django/apps/registry.py | 234 | 260| 219 | 28390 | 76957 | 
| 97 | 39 django/db/migrations/autodetector.py | 239 | 263| 267 | 28657 | 88628 | 
| 98 | 40 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 28910 | 89049 | 
| 99 | **40 django/contrib/admin/checks.py** | 366 | 392| 281 | 29191 | 89049 | 
| 100 | 41 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 29979 | 91798 | 
| 101 | 42 django/contrib/auth/management/__init__.py | 35 | 86| 471 | 30450 | 92872 | 
| 102 | 42 django/conf/global_settings.py | 398 | 497| 793 | 31243 | 92872 | 
| 103 | 43 django/contrib/admin/exceptions.py | 1 | 12| 0 | 31243 | 92939 | 
| 104 | 43 django/contrib/admin/options.py | 1878 | 1915| 330 | 31573 | 92939 | 
| 105 | 43 django/contrib/admin/helpers.py | 33 | 67| 230 | 31803 | 92939 | 
| 106 | 44 django/contrib/sessions/exceptions.py | 1 | 12| 0 | 31803 | 92990 | 
| 107 | 45 django/db/models/options.py | 1 | 36| 304 | 32107 | 99856 | 
| 108 | 46 django/contrib/sessions/management/commands/clearsessions.py | 1 | 20| 122 | 32229 | 99978 | 
| 109 | 47 django/contrib/admindocs/__init__.py | 1 | 2| 0 | 32229 | 99993 | 
| 110 | 47 django/core/checks/model_checks.py | 45 | 66| 168 | 32397 | 99993 | 
| 111 | 47 django/contrib/admin/options.py | 1466 | 1482| 231 | 32628 | 99993 | 
| 112 | 47 django/contrib/admin/options.py | 1597 | 1621| 279 | 32907 | 99993 | 
| 113 | 47 django/apps/registry.py | 127 | 145| 166 | 33073 | 99993 | 
| 114 | 47 django/contrib/sessions/backends/base.py | 292 | 359| 459 | 33532 | 99993 | 
| 115 | 47 django/contrib/admin/options.py | 422 | 465| 350 | 33882 | 99993 | 
| 116 | 47 django/contrib/admin/options.py | 277 | 366| 641 | 34523 | 99993 | 
| 117 | **47 django/contrib/admin/checks.py** | 408 | 417| 125 | 34648 | 99993 | 
| 118 | **47 django/contrib/admin/checks.py** | 516 | 539| 230 | 34878 | 99993 | 
| 119 | 48 django/core/checks/templates.py | 1 | 36| 259 | 35137 | 100253 | 
| 120 | 48 django/contrib/admin/options.py | 1337 | 1402| 581 | 35718 | 100253 | 
| 121 | 48 django/core/management/commands/check.py | 36 | 66| 214 | 35932 | 100253 | 
| 122 | 49 django/db/models/base.py | 1 | 45| 289 | 36221 | 115143 | 
| 123 | 50 docs/_ext/djangodocs.py | 26 | 70| 385 | 36606 | 118216 | 
| 124 | 50 django/contrib/admin/options.py | 602 | 624| 280 | 36886 | 118216 | 
| 125 | 51 django/contrib/auth/models.py | 161 | 197| 235 | 37121 | 121144 | 
| 126 | **51 django/contrib/admin/checks.py** | 838 | 860| 217 | 37338 | 121144 | 
| 127 | 52 django/db/migrations/state.py | 245 | 291| 444 | 37782 | 126362 | 
| 128 | 52 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 37954 | 126362 | 
| 129 | 52 django/contrib/admin/utils.py | 285 | 303| 175 | 38129 | 126362 | 
| 130 | 52 django/core/checks/model_checks.py | 68 | 83| 176 | 38305 | 126362 | 
| 131 | 52 django/contrib/admin/options.py | 1111 | 1156| 482 | 38787 | 126362 | 
| 132 | 52 django/contrib/admin/helpers.py | 350 | 359| 134 | 38921 | 126362 | 
| 133 | 53 django/db/utils.py | 266 | 308| 322 | 39243 | 128380 | 
| 134 | 54 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 39648 | 128785 | 
| 135 | 55 django/contrib/sites/admin.py | 1 | 9| 0 | 39648 | 128831 | 
| 136 | 56 django/contrib/contenttypes/apps.py | 1 | 23| 150 | 39798 | 128981 | 
| 137 | 57 django/db/migrations/recorder.py | 24 | 45| 145 | 39943 | 129651 | 
| 138 | 58 django/db/backends/base/base.py | 413 | 490| 525 | 40468 | 134436 | 
| 139 | 58 django/contrib/admin/options.py | 467 | 489| 241 | 40709 | 134436 | 
| 140 | 59 django/contrib/admindocs/urls.py | 1 | 51| 307 | 41016 | 134743 | 
| 141 | 60 django/contrib/admin/models.py | 1 | 20| 118 | 41134 | 135853 | 
| 142 | 60 django/contrib/admin/options.py | 1310 | 1335| 232 | 41366 | 135853 | 
| 143 | 61 django/core/checks/__init__.py | 1 | 25| 254 | 41620 | 136107 | 
| 144 | **61 django/contrib/admin/checks.py** | 504 | 514| 134 | 41754 | 136107 | 
| 145 | 62 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 41964 | 137609 | 
| 146 | **62 django/contrib/admin/checks.py** | 701 | 711| 115 | 42079 | 137609 | 
| 147 | 62 django/conf/global_settings.py | 145 | 263| 876 | 42955 | 137609 | 
| 148 | 63 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 43153 | 138176 | 
| 149 | 63 django/contrib/admin/options.py | 2108 | 2133| 250 | 43403 | 138176 | 
| 150 | 64 django/contrib/sessions/backends/db.py | 1 | 72| 461 | 43864 | 138900 | 
| 151 | 65 django/db/migrations/executor.py | 298 | 377| 712 | 44576 | 142192 | 
| 152 | 66 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 44687 | 142303 | 
| 153 | **66 django/contrib/admin/checks.py** | 307 | 318| 138 | 44825 | 142303 | 
| 154 | 66 django/db/migrations/autodetector.py | 358 | 372| 141 | 44966 | 142303 | 
| 155 | 67 django/db/models/__init__.py | 1 | 49| 548 | 45514 | 142851 | 
| 156 | 68 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 46363 | 143700 | 
| 157 | 69 django/utils/autoreload.py | 248 | 264| 162 | 46525 | 148312 | 
| 158 | 70 django/contrib/auth/backends.py | 1 | 43| 308 | 46833 | 149749 | 
| 159 | 71 django/db/migrations/questioner.py | 1 | 54| 468 | 47301 | 151823 | 
| 160 | 72 django/contrib/redirects/admin.py | 1 | 11| 0 | 47301 | 151891 | 
| 161 | 73 django/db/models/fields/__init__.py | 1114 | 1143| 218 | 47519 | 168890 | 
| 162 | 73 django/contrib/admin/options.py | 1653 | 1724| 653 | 48172 | 168890 | 
| 163 | 73 django/db/migrations/loader.py | 148 | 174| 291 | 48463 | 168890 | 
| 164 | 73 django/contrib/admin/models.py | 23 | 36| 111 | 48574 | 168890 | 
| 165 | 73 django/db/migrations/autodetector.py | 1265 | 1288| 240 | 48814 | 168890 | 
| 166 | 73 django/db/models/base.py | 1230 | 1259| 242 | 49056 | 168890 | 
| 167 | 73 django/contrib/admin/options.py | 626 | 643| 136 | 49192 | 168890 | 
| 168 | 74 django/contrib/admin/bin/compress.py | 1 | 64| 473 | 49665 | 169363 | 
| 169 | 74 django/contrib/admin/options.py | 587 | 600| 122 | 49787 | 169363 | 
| 170 | 74 django/contrib/admin/helpers.py | 1 | 30| 198 | 49985 | 169363 | 
| 171 | 74 django/db/models/base.py | 1261 | 1286| 184 | 50169 | 169363 | 
| 172 | 75 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 50960 | 171234 | 
| 173 | 75 django/contrib/sessions/backends/file.py | 75 | 109| 253 | 51213 | 171234 | 
| 174 | 75 django/contrib/auth/middleware.py | 113 | 124| 107 | 51320 | 171234 | 
| 175 | 76 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 51478 | 172420 | 
| 176 | **76 django/contrib/admin/checks.py** | 419 | 440| 191 | 51669 | 172420 | 
| 177 | **76 django/contrib/admin/checks.py** | 482 | 502| 200 | 51869 | 172420 | 
| 178 | 76 django/apps/registry.py | 212 | 232| 237 | 52106 | 172420 | 
| 179 | 77 django/core/management/commands/migrate.py | 1 | 18| 148 | 52254 | 175560 | 
| 180 | 77 django/apps/registry.py | 61 | 125| 438 | 52692 | 175560 | 
| 181 | **77 django/contrib/admin/checks.py** | 272 | 305| 381 | 53073 | 175560 | 
| 182 | 77 django/db/migrations/executor.py | 281 | 296| 165 | 53238 | 175560 | 
| 183 | 78 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 33 | 49| 109 | 53347 | 175938 | 
| 184 | 78 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 53697 | 175938 | 
| 185 | 79 django/contrib/admindocs/utils.py | 1 | 23| 133 | 53830 | 177838 | 
| 186 | 80 django/contrib/admin/filters.py | 1 | 17| 127 | 53957 | 181507 | 
| 187 | 80 django/core/management/commands/migrate.py | 21 | 65| 369 | 54326 | 181507 | 
| 188 | 80 django/contrib/sessions/backends/file.py | 111 | 170| 530 | 54856 | 181507 | 
| 189 | 80 django/db/migrations/autodetector.py | 437 | 463| 256 | 55112 | 181507 | 
| 190 | 80 django/contrib/admin/views/main.py | 448 | 479| 225 | 55337 | 181507 | 
| 191 | 80 django/db/migrations/loader.py | 176 | 197| 213 | 55550 | 181507 | 
| 192 | 80 django/db/migrations/state.py | 293 | 317| 266 | 55816 | 181507 | 
| 193 | 80 django/apps/registry.py | 356 | 376| 183 | 55999 | 181507 | 
| 194 | 81 django/core/management/base.py | 379 | 444| 614 | 56613 | 185862 | 
| 195 | 81 django/contrib/auth/admin.py | 40 | 99| 504 | 57117 | 185862 | 
| 196 | 81 django/contrib/admin/sites.py | 134 | 194| 397 | 57514 | 185862 | 
| 197 | 81 django/contrib/sessions/backends/db.py | 74 | 110| 269 | 57783 | 185862 | 
| 198 | 81 django/contrib/admindocs/views.py | 155 | 179| 234 | 58017 | 185862 | 
| 199 | 81 django/db/migrations/autodetector.py | 1038 | 1058| 136 | 58153 | 185862 | 
| 200 | 81 django/contrib/admin/options.py | 1808 | 1876| 584 | 58737 | 185862 | 
| 201 | 81 django/db/models/base.py | 1288 | 1317| 205 | 58942 | 185862 | 
| 202 | 82 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 58942 | 185938 | 
| 203 | 83 django/contrib/auth/views.py | 1 | 35| 272 | 59214 | 188590 | 
| 204 | 84 django/middleware/common.py | 34 | 61| 257 | 59471 | 190101 | 
| 205 | 84 django/db/migrations/autodetector.py | 264 | 335| 748 | 60219 | 190101 | 
| 206 | 85 django/contrib/postgres/apps.py | 20 | 37| 188 | 60407 | 190667 | 
| 207 | 85 django/contrib/admindocs/views.py | 182 | 248| 584 | 60991 | 190667 | 
| 208 | 85 django/contrib/admin/utils.py | 121 | 156| 303 | 61294 | 190667 | 
| 209 | 86 django/contrib/messages/apps.py | 1 | 8| 0 | 61294 | 190704 | 
| 210 | 86 django/core/management/commands/migrate.py | 293 | 340| 401 | 61695 | 190704 | 
| 211 | 86 django/core/management/commands/migrate.py | 67 | 160| 825 | 62520 | 190704 | 


### Hint

```
System checks are ​designed to be silenced if not appropriate. I'm inclined to think this just such an edge-case at first glance. But OK, yes, I guess subclasses are as legitimate here as elsewhere, so accepting as a Cleanup/Optimisation.
I'll work on a PR, then. :)
​PR
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -65,7 +65,6 @@ def check_dependencies(**kwargs):
         ('django.contrib.contenttypes', 401),
         ('django.contrib.auth', 405),
         ('django.contrib.messages', 406),
-        ('django.contrib.sessions', 407),
     )
     for app_name, error_code in app_dependencies:
         if not apps.is_installed(app_name):
@@ -118,6 +117,12 @@ def check_dependencies(**kwargs):
             "be in MIDDLEWARE in order to use the admin application.",
             id='admin.E409',
         ))
+    if not _contains_subclass('django.contrib.sessions.middleware.SessionMiddleware', settings.MIDDLEWARE):
+        errors.append(checks.Error(
+            "'django.contrib.sessions.middleware.SessionMiddleware' must "
+            "be in MIDDLEWARE in order to use the admin application.",
+            id='admin.E410',
+        ))
     return errors
 
 

```

## Test Patch

```diff
diff --git a/tests/admin_checks/tests.py b/tests/admin_checks/tests.py
--- a/tests/admin_checks/tests.py
+++ b/tests/admin_checks/tests.py
@@ -5,6 +5,7 @@
 from django.contrib.auth.middleware import AuthenticationMiddleware
 from django.contrib.contenttypes.admin import GenericStackedInline
 from django.contrib.messages.middleware import MessageMiddleware
+from django.contrib.sessions.middleware import SessionMiddleware
 from django.core import checks
 from django.test import SimpleTestCase, override_settings
 
@@ -52,13 +53,16 @@ class ModelBackendSubclass(ModelBackend):
     pass
 
 
+class SessionMiddlewareSubclass(SessionMiddleware):
+    pass
+
+
 @override_settings(
     SILENCED_SYSTEM_CHECKS=['fields.W342'],  # ForeignKey(unique=True)
     INSTALLED_APPS=[
         'django.contrib.admin',
         'django.contrib.auth',
         'django.contrib.contenttypes',
-        'django.contrib.sessions',
         'django.contrib.messages',
         'admin_checks',
     ],
@@ -93,11 +97,6 @@ def test_apps_dependencies(self):
                 "to use the admin application.",
                 id='admin.E406',
             ),
-            checks.Error(
-                "'django.contrib.sessions' must be in INSTALLED_APPS in order "
-                "to use the admin application.",
-                id='admin.E407',
-            )
         ]
         self.assertEqual(errors, expected)
 
@@ -201,13 +200,19 @@ def test_middleware_dependencies(self):
                 "'django.contrib.messages.middleware.MessageMiddleware' "
                 "must be in MIDDLEWARE in order to use the admin application.",
                 id='admin.E409',
-            )
+            ),
+            checks.Error(
+                "'django.contrib.sessions.middleware.SessionMiddleware' "
+                "must be in MIDDLEWARE in order to use the admin application.",
+                id='admin.E410',
+            ),
         ]
         self.assertEqual(errors, expected)
 
     @override_settings(MIDDLEWARE=[
         'admin_checks.tests.AuthenticationMiddlewareSubclass',
         'admin_checks.tests.MessageMiddlewareSubclass',
+        'admin_checks.tests.SessionMiddlewareSubclass',
     ])
     def test_middleware_subclasses(self):
         self.assertEqual(admin.checks.check_dependencies(), [])
@@ -216,6 +221,7 @@ def test_middleware_subclasses(self):
         'django.contrib.does.not.Exist',
         'django.contrib.auth.middleware.AuthenticationMiddleware',
         'django.contrib.messages.middleware.MessageMiddleware',
+        'django.contrib.sessions.middleware.SessionMiddleware',
     ])
     def test_admin_check_ignores_import_error_in_middleware(self):
         self.assertEqual(admin.checks.check_dependencies(), [])
diff --git a/tests/admin_scripts/tests.py b/tests/admin_scripts/tests.py
--- a/tests/admin_scripts/tests.py
+++ b/tests/admin_scripts/tests.py
@@ -1103,13 +1103,13 @@ def test_complex_app(self):
                 'django.contrib.auth',
                 'django.contrib.contenttypes',
                 'django.contrib.messages',
-                'django.contrib.sessions',
             ],
             sdict={
                 'DEBUG': True,
                 'MIDDLEWARE': [
                     'django.contrib.messages.middleware.MessageMiddleware',
                     'django.contrib.auth.middleware.AuthenticationMiddleware',
+                    'django.contrib.sessions.middleware.SessionMiddleware',
                 ],
                 'TEMPLATES': [
                     {

```


## Code snippets

### 1 - django/contrib/admin/checks.py:

Start line: 57, End line: 121

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
        ('django.contrib.sessions', 407),
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
    return errors
```
### 2 - django/core/checks/security/sessions.py:

Start line: 1, End line: 98

```python
from django.conf import settings

from .. import Tags, Warning, register


def add_session_cookie_message(message):
    return message + (
        " Using a secure-only session cookie makes it more difficult for "
        "network traffic sniffers to hijack user sessions."
    )


W010 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_SECURE to True."
    ),
    id='security.W010',
)

W011 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_SECURE to True."
    ),
    id='security.W011',
)

W012 = Warning(
    add_session_cookie_message("SESSION_COOKIE_SECURE is not set to True."),
    id='security.W012',
)


def add_httponly_message(message):
    return message + (
        " Using an HttpOnly session cookie makes it more difficult for "
        "cross-site scripting attacks to hijack user sessions."
    )


W013 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_HTTPONLY to True.",
    ),
    id='security.W013',
)

W014 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_HTTPONLY to True."
    ),
    id='security.W014',
)

W015 = Warning(
    add_httponly_message("SESSION_COOKIE_HTTPONLY is not set to True."),
    id='security.W015',
)


@register(Tags.security, deploy=True)
def check_session_cookie_secure(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_SECURE:
        if _session_app():
            errors.append(W010)
        if _session_middleware():
            errors.append(W011)
        if len(errors) > 1:
            errors = [W012]
    return errors


@register(Tags.security, deploy=True)
def check_session_cookie_httponly(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_HTTPONLY:
        if _session_app():
            errors.append(W013)
        if _session_middleware():
            errors.append(W014)
        if len(errors) > 1:
            errors = [W015]
    return errors


def _session_middleware():
    return 'django.contrib.sessions.middleware.SessionMiddleware' in settings.MIDDLEWARE


def _session_app():
    return "django.contrib.sessions" in settings.INSTALLED_APPS
```
### 3 - django/contrib/admin/checks.py:

Start line: 1, End line: 54

```python
from itertools import chain

from django.apps import apps
from django.conf import settings
from django.contrib.admin.utils import (
    NotRelationField, flatten, get_fields_from_path,
)
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable, F, OrderBy
from django.forms.models import (
    BaseModelForm, BaseModelFormSet, _get_foreign_key,
)
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string


def _issubclass(cls, classinfo):
    """
    issubclass() variant that doesn't raise an exception if cls isn't a
    class.
    """
    try:
        return issubclass(cls, classinfo)
    except TypeError:
        return False


def _contains_subclass(class_path, candidate_paths):
    """
    Return whether or not a dotted class path (or a subclass of that class) is
    found in a list of candidate paths.
    """
    cls = import_string(class_path)
    for path in candidate_paths:
        try:
            candidate_cls = import_string(path)
        except ImportError:
            # ImportErrors are raised elsewhere.
            continue
        if _issubclass(candidate_cls, cls):
            return True
    return False


def check_admin_app(app_configs, **kwargs):
    from django.contrib.admin.sites import all_sites
    errors = []
    for site in all_sites:
        errors.extend(site.check(app_configs))
    return errors
```
### 4 - django/contrib/admin/apps.py:

Start line: 1, End line: 25

```python
from django.apps import AppConfig
from django.contrib.admin.checks import check_admin_app, check_dependencies
from django.core import checks
from django.utils.translation import gettext_lazy as _


class SimpleAdminConfig(AppConfig):
    """Simple AppConfig which does not do automatic discovery."""

    default_site = 'django.contrib.admin.sites.AdminSite'
    name = 'django.contrib.admin'
    verbose_name = _("Administration")

    def ready(self):
        checks.register(check_dependencies, checks.Tags.admin)
        checks.register(check_admin_app, checks.Tags.admin)


class AdminConfig(SimpleAdminConfig):
    """The default AppConfig for admin which does autodiscovery."""

    def ready(self):
        super().ready()
        self.module.autodiscover()
```
### 5 - django/contrib/auth/admin.py:

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
### 6 - django/contrib/admin/checks.py:

Start line: 124, End line: 141

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
### 7 - django/contrib/admin/sites.py:

Start line: 1, End line: 29

```python
import re
from functools import update_wrapper
from weakref import WeakSet

from django.apps import apps
from django.contrib.admin import ModelAdmin, actions
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import Http404, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, reverse
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _, gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog

all_sites = WeakSet()


class AlreadyRegistered(Exception):
    pass


class NotRegistered(Exception):
    pass
```
### 8 - django/contrib/admin/checks.py:

Start line: 615, End line: 634

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
### 9 - django/contrib/admin/__init__.py:

Start line: 1, End line: 30

```python
# ACTION_CHECKBOX_NAME is unused, but should stay since its import from here
# has been referenced in documentation.
from django.contrib.admin.decorators import register
from django.contrib.admin.filters import (
    AllValuesFieldListFilter, BooleanFieldListFilter, ChoicesFieldListFilter,
    DateFieldListFilter, FieldListFilter, ListFilter, RelatedFieldListFilter,
    RelatedOnlyFieldListFilter, SimpleListFilter,
)
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.admin.options import (
    HORIZONTAL, VERTICAL, ModelAdmin, StackedInline, TabularInline,
)
from django.contrib.admin.sites import AdminSite, site
from django.utils.module_loading import autodiscover_modules

__all__ = [
    "register", "ACTION_CHECKBOX_NAME", "ModelAdmin", "HORIZONTAL", "VERTICAL",
    "StackedInline", "TabularInline", "AdminSite", "site", "ListFilter",
    "SimpleListFilter", "FieldListFilter", "BooleanFieldListFilter",
    "RelatedFieldListFilter", "ChoicesFieldListFilter", "DateFieldListFilter",
    "AllValuesFieldListFilter", "RelatedOnlyFieldListFilter", "autodiscover",
]


def autodiscover():
    autodiscover_modules('admin', register_to=site)


default_app_config = 'django.contrib.admin.apps.AdminConfig'
```
### 10 - django/contrib/sessions/middleware.py:

Start line: 1, End line: 76

```python
import time
from importlib import import_module

from django.conf import settings
from django.contrib.sessions.backends.base import UpdateError
from django.core.exceptions import SuspiciousOperation
from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import http_date


class SessionMiddleware(MiddlewareMixin):
    def __init__(self, get_response=None):
        self.get_response = get_response
        engine = import_module(settings.SESSION_ENGINE)
        self.SessionStore = engine.SessionStore

    def process_request(self, request):
        session_key = request.COOKIES.get(settings.SESSION_COOKIE_NAME)
        request.session = self.SessionStore(session_key)

    def process_response(self, request, response):
        """
        If request.session was modified, or if the configuration is to save the
        session every time, save the changes and set a session cookie or delete
        the session cookie if the session has been emptied.
        """
        try:
            accessed = request.session.accessed
            modified = request.session.modified
            empty = request.session.is_empty()
        except AttributeError:
            pass
        else:
            # First check if we need to delete this cookie.
            # The session should be deleted only if the session is entirely empty
            if settings.SESSION_COOKIE_NAME in request.COOKIES and empty:
                response.delete_cookie(
                    settings.SESSION_COOKIE_NAME,
                    path=settings.SESSION_COOKIE_PATH,
                    domain=settings.SESSION_COOKIE_DOMAIN,
                )
                patch_vary_headers(response, ('Cookie',))
            else:
                if accessed:
                    patch_vary_headers(response, ('Cookie',))
                if (modified or settings.SESSION_SAVE_EVERY_REQUEST) and not empty:
                    if request.session.get_expire_at_browser_close():
                        max_age = None
                        expires = None
                    else:
                        max_age = request.session.get_expiry_age()
                        expires_time = time.time() + max_age
                        expires = http_date(expires_time)
                    # Save the session data and refresh the client cookie.
                    # Skip session save for 500 responses, refs #3881.
                    if response.status_code != 500:
                        try:
                            request.session.save()
                        except UpdateError:
                            raise SuspiciousOperation(
                                "The request's session was deleted before the "
                                "request completed. The user may have logged "
                                "out in a concurrent request, for example."
                            )
                        response.set_cookie(
                            settings.SESSION_COOKIE_NAME,
                            request.session.session_key, max_age=max_age,
                            expires=expires, domain=settings.SESSION_COOKIE_DOMAIN,
                            path=settings.SESSION_COOKIE_PATH,
                            secure=settings.SESSION_COOKIE_SECURE or None,
                            httponly=settings.SESSION_COOKIE_HTTPONLY or None,
                            samesite=settings.SESSION_COOKIE_SAMESITE,
                        )
        return response
```
### 13 - django/contrib/admin/checks.py:

Start line: 1008, End line: 1035

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_exclude_of_parent_model(self, obj, parent_model):
        # Do not perform more specific checks if the base checks result in an
        # error.
        errors = super()._check_exclude(obj)
        if errors:
            return []

        # Skip if `fk_name` is invalid.
        if self._check_relation(obj, parent_model):
            return []

        if obj.exclude is None:
            return []

        fk = _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        if fk.name in obj.exclude:
            return [
                checks.Error(
                    "Cannot exclude the field '%s', because it is the foreign key "
                    "to the parent model '%s.%s'." % (
                        fk.name, parent_model._meta.app_label, parent_model._meta.object_name
                    ),
                    obj=obj.__class__,
                    id='admin.E201',
                )
            ]
        else:
            return []
```
### 18 - django/contrib/admin/checks.py:

Start line: 955, End line: 991

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_action_permission_methods(self, obj):
        """
        Actions with an allowed_permission attribute require the ModelAdmin to
        implement a has_<perm>_permission() method for each permission.
        """
        actions = obj._get_base_actions()
        errors = []
        for func, name, _ in actions:
            if not hasattr(func, 'allowed_permissions'):
                continue
            for permission in func.allowed_permissions:
                method_name = 'has_%s_permission' % permission
                if not hasattr(obj, method_name):
                    errors.append(
                        checks.Error(
                            '%s must define a %s() method for the %s action.' % (
                                obj.__class__.__name__,
                                method_name,
                                func.__name__,
                            ),
                            obj=obj.__class__,
                            id='admin.E129',
                        )
                    )
        return errors

    def _check_actions_uniqueness(self, obj):
        """Check that every action has a unique __name__."""
        names = [name for _, name, _ in obj._get_base_actions()]
        if len(names) != len(set(names)):
            return [checks.Error(
                '__name__ attributes of actions defined in %s must be '
                'unique.' % obj.__class__,
                obj=obj.__class__,
                id='admin.E130',
            )]
        return []
```
### 19 - django/contrib/admin/checks.py:

Start line: 591, End line: 612

```python
class BaseModelAdminChecks:

    def _check_readonly_fields_item(self, obj, field_name, label):
        if callable(field_name):
            return []
        elif hasattr(obj, field_name):
            return []
        elif hasattr(obj.model, field_name):
            return []
        else:
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return [
                    checks.Error(
                        "The value of '%s' is not a callable, an attribute of '%s', or an attribute of '%s.%s'." % (
                            label, obj.__class__.__name__, obj.model._meta.app_label, obj.model._meta.object_name
                        ),
                        obj=obj.__class__,
                        id='admin.E035',
                    )
                ]
            else:
                return []
```
### 23 - django/contrib/admin/checks.py:

Start line: 578, End line: 589

```python
class BaseModelAdminChecks:

    def _check_readonly_fields(self, obj):
        """ Check that readonly_fields refers to proper attribute or field. """

        if obj.readonly_fields == ():
            return []
        elif not isinstance(obj.readonly_fields, (list, tuple)):
            return must_be('a list or tuple', option='readonly_fields', obj=obj, id='admin.E034')
        else:
            return list(chain.from_iterable(
                self._check_readonly_fields_item(obj, field_name, "readonly_fields[%d]" % index)
                for index, field_name in enumerate(obj.readonly_fields)
            ))
```
### 29 - django/contrib/admin/checks.py:

Start line: 320, End line: 346

```python
class BaseModelAdminChecks:

    def _check_field_spec_item(self, obj, field_name, label):
        if field_name in obj.readonly_fields:
            # Stuff can be put in fields that isn't actually a model field if
            # it's in readonly_fields, readonly_fields will handle the
            # validation of such things.
            return []
        else:
            try:
                field = obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                # If we can't find a field on the model that matches, it could
                # be an extra field on the form.
                return []
            else:
                if (isinstance(field, models.ManyToManyField) and
                        not field.remote_field.through._meta.auto_created):
                    return [
                        checks.Error(
                            "The value of '%s' cannot include the ManyToManyField '%s', "
                            "because that field manually specifies a relationship model."
                            % (label, field_name),
                            obj=obj.__class__,
                            id='admin.E013',
                        )
                    ]
                else:
                    return []
```
### 33 - django/contrib/admin/checks.py:

Start line: 1037, End line: 1079

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_relation(self, obj, parent_model):
        try:
            _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        except ValueError as e:
            return [checks.Error(e.args[0], obj=obj.__class__, id='admin.E202')]
        else:
            return []

    def _check_extra(self, obj):
        """ Check that extra is an integer. """

        if not isinstance(obj.extra, int):
            return must_be('an integer', option='extra', obj=obj, id='admin.E203')
        else:
            return []

    def _check_max_num(self, obj):
        """ Check that max_num is an integer. """

        if obj.max_num is None:
            return []
        elif not isinstance(obj.max_num, int):
            return must_be('an integer', option='max_num', obj=obj, id='admin.E204')
        else:
            return []

    def _check_min_num(self, obj):
        """ Check that min_num is an integer. """

        if obj.min_num is None:
            return []
        elif not isinstance(obj.min_num, int):
            return must_be('an integer', option='min_num', obj=obj, id='admin.E205')
        else:
            return []

    def _check_formset(self, obj):
        """ Check formset is a subclass of BaseModelFormSet. """

        if not _issubclass(obj.formset, BaseModelFormSet):
            return must_inherit_from(parent='BaseModelFormSet', option='formset', obj=obj, id='admin.E206')
        else:
            return []
```
### 37 - django/contrib/admin/checks.py:

Start line: 994, End line: 1006

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def check(self, inline_obj, **kwargs):
        parent_model = inline_obj.parent_model
        return [
            *super().check(inline_obj),
            *self._check_relation(inline_obj, parent_model),
            *self._check_exclude_of_parent_model(inline_obj, parent_model),
            *self._check_extra(inline_obj),
            *self._check_max_num(inline_obj),
            *self._check_min_num(inline_obj),
            *self._check_formset(inline_obj),
        ]
```
### 38 - django/contrib/admin/checks.py:

Start line: 348, End line: 364

```python
class BaseModelAdminChecks:

    def _check_exclude(self, obj):
        """ Check that exclude is a sequence without duplicates. """

        if obj.exclude is None:  # default value is None
            return []
        elif not isinstance(obj.exclude, (list, tuple)):
            return must_be('a list or tuple', option='exclude', obj=obj, id='admin.E014')
        elif len(obj.exclude) > len(set(obj.exclude)):
            return [
                checks.Error(
                    "The value of 'exclude' contains duplicate field(s).",
                    obj=obj.__class__,
                    id='admin.E015',
                )
            ]
        else:
            return []
```
### 39 - django/contrib/admin/checks.py:

Start line: 874, End line: 922

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable_item(self, obj, field_name, label):
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E121')
        else:
            if field_name not in obj.list_display:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not "
                        "contained in 'list_display'." % (label, field_name),
                        obj=obj.__class__,
                        id='admin.E122',
                    )
                ]
            elif obj.list_display_links and field_name in obj.list_display_links:
                return [
                    checks.Error(
                        "The value of '%s' cannot be in both 'list_editable' and 'list_display_links'." % field_name,
                        obj=obj.__class__,
                        id='admin.E123',
                    )
                ]
            # If list_display[0] is in list_editable, check that
            # list_display_links is set. See #22792 and #26229 for use cases.
            elif (obj.list_display[0] == field_name and not obj.list_display_links and
                    obj.list_display_links is not None):
                return [
                    checks.Error(
                        "The value of '%s' refers to the first field in 'list_display' ('%s'), "
                        "which cannot be used unless 'list_display_links' is set." % (
                            label, obj.list_display[0]
                        ),
                        obj=obj.__class__,
                        id='admin.E124',
                    )
                ]
            elif not field.editable:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not editable through the admin." % (
                            label, field_name
                        ),
                        obj=obj.__class__,
                        id='admin.E125',
                    )
                ]
            else:
                return []
```
### 46 - django/contrib/admin/checks.py:

Start line: 665, End line: 699

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_inlines_item(self, obj, inline, label):
        """ Check one inline model admin. """
        try:
            inline_label = inline.__module__ + '.' + inline.__name__
        except AttributeError:
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % obj,
                    obj=obj.__class__,
                    id='admin.E104',
                )
            ]

        from django.contrib.admin.options import InlineModelAdmin

        if not _issubclass(inline, InlineModelAdmin):
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % inline_label,
                    obj=obj.__class__,
                    id='admin.E104',
                )
            ]
        elif not inline.model:
            return [
                checks.Error(
                    "'%s' must have a 'model' attribute." % inline_label,
                    obj=obj.__class__,
                    id='admin.E105',
                )
            ]
        elif not _issubclass(inline.model, models.Model):
            return must_be('a Model', option='%s.model' % inline_label, obj=obj, id='admin.E106')
        else:
            return inline(obj.model, obj.admin_site).check()
```
### 49 - django/contrib/admin/checks.py:

Start line: 198, End line: 208

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields(self, obj):
        """ Check that `raw_id_fields` only contains field names that are listed
        on the model. """

        if not isinstance(obj.raw_id_fields, (list, tuple)):
            return must_be('a list or tuple', option='raw_id_fields', obj=obj, id='admin.E001')
        else:
            return list(chain.from_iterable(
                self._check_raw_id_fields_item(obj, field_name, 'raw_id_fields[%d]' % index)
                for index, field_name in enumerate(obj.raw_id_fields)
            ))
```
### 50 - django/contrib/admin/checks.py:

Start line: 636, End line: 663

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_save_as(self, obj):
        """ Check save_as is a boolean. """

        if not isinstance(obj.save_as, bool):
            return must_be('a boolean', option='save_as',
                           obj=obj, id='admin.E101')
        else:
            return []

    def _check_save_on_top(self, obj):
        """ Check save_on_top is a boolean. """

        if not isinstance(obj.save_on_top, bool):
            return must_be('a boolean', option='save_on_top',
                           obj=obj, id='admin.E102')
        else:
            return []

    def _check_inlines(self, obj):
        """ Check all inline model admin classes. """

        if not isinstance(obj.inlines, (list, tuple)):
            return must_be('a list or tuple', option='inlines', obj=obj, id='admin.E103')
        else:
            return list(chain.from_iterable(
                self._check_inlines_item(obj, item, "inlines[%d]" % index)
                for index, item in enumerate(obj.inlines)
            ))
```
### 58 - django/contrib/admin/checks.py:

Start line: 155, End line: 196

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields_item(self, obj, field_name, label):
        """
        Check that an item in `autocomplete_fields` is a ForeignKey or a
        ManyToManyField and that the item has a related ModelAdmin with
        search_fields defined.
        """
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E037')
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    'a foreign key or a many-to-many field',
                    option=label, obj=obj, id='admin.E038'
                )
            related_admin = obj.admin_site._registry.get(field.remote_field.model)
            if related_admin is None:
                return [
                    checks.Error(
                        'An admin for model "%s" has to be registered '
                        'to be referenced by %s.autocomplete_fields.' % (
                            field.remote_field.model.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id='admin.E039',
                    )
                ]
            elif not related_admin.search_fields:
                return [
                    checks.Error(
                        '%s must define "search_fields", because it\'s '
                        'referenced by %s.autocomplete_fields.' % (
                            related_admin.__class__.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id='admin.E040',
                    )
                ]
            return []
```
### 59 - django/contrib/admin/checks.py:

Start line: 713, End line: 744

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_item(self, obj, item, label):
        if callable(item):
            return []
        elif hasattr(obj, item):
            return []
        elif hasattr(obj.model, item):
            try:
                field = obj.model._meta.get_field(item)
            except FieldDoesNotExist:
                return []
            else:
                if isinstance(field, models.ManyToManyField):
                    return [
                        checks.Error(
                            "The value of '%s' must not be a ManyToManyField." % label,
                            obj=obj.__class__,
                            id='admin.E109',
                        )
                    ]
                return []
        else:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not a callable, "
                    "an attribute of '%s', or an attribute or method on '%s.%s'." % (
                        label, item, obj.__class__.__name__,
                        obj.model._meta.app_label, obj.model._meta.object_name,
                    ),
                    obj=obj.__class__,
                    id='admin.E108',
                )
            ]
```
### 61 - django/contrib/admin/checks.py:

Start line: 225, End line: 255

```python
class BaseModelAdminChecks:

    def _check_fields(self, obj):
        """ Check that `fields` only refer to existing fields, doesn't contain
        duplicates. Check if at most one of `fields` and `fieldsets` is defined.
        """

        if obj.fields is None:
            return []
        elif not isinstance(obj.fields, (list, tuple)):
            return must_be('a list or tuple', option='fields', obj=obj, id='admin.E004')
        elif obj.fieldsets:
            return [
                checks.Error(
                    "Both 'fieldsets' and 'fields' are specified.",
                    obj=obj.__class__,
                    id='admin.E005',
                )
            ]
        fields = flatten(obj.fields)
        if len(fields) != len(set(fields)):
            return [
                checks.Error(
                    "The value of 'fields' contains duplicate field(s).",
                    obj=obj.__class__,
                    id='admin.E006',
                )
            ]

        return list(chain.from_iterable(
            self._check_field_spec(obj, field_name, 'fields')
            for field_name in obj.fields
        ))
```
### 68 - django/contrib/admin/checks.py:

Start line: 924, End line: 953

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_search_fields(self, obj):
        """ Check search_fields is a sequence. """

        if not isinstance(obj.search_fields, (list, tuple)):
            return must_be('a list or tuple', option='search_fields', obj=obj, id='admin.E126')
        else:
            return []

    def _check_date_hierarchy(self, obj):
        """ Check that date_hierarchy refers to DateField or DateTimeField. """

        if obj.date_hierarchy is None:
            return []
        else:
            try:
                field = get_fields_from_path(obj.model, obj.date_hierarchy)[-1]
            except (NotRelationField, FieldDoesNotExist):
                return [
                    checks.Error(
                        "The value of 'date_hierarchy' refers to '%s', which "
                        "does not refer to a Field." % obj.date_hierarchy,
                        obj=obj.__class__,
                        id='admin.E127',
                    )
                ]
            else:
                if not isinstance(field, (models.DateField, models.DateTimeField)):
                    return must_be('a DateField or DateTimeField', option='date_hierarchy', obj=obj, id='admin.E128')
                else:
                    return []
```
### 76 - django/contrib/admin/checks.py:

Start line: 210, End line: 223

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields_item(self, obj, field_name, label):
        """ Check an item of `raw_id_fields`, i.e. check that field named
        `field_name` exists in model `model` and is a ForeignKey or a
        ManyToManyField. """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E002')
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be('a foreign key or a many-to-many field', option=label, obj=obj, id='admin.E003')
            else:
                return []
```
### 80 - django/contrib/admin/checks.py:

Start line: 1082, End line: 1112

```python
def must_be(type, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must be %s." % (option, type),
            obj=obj.__class__,
            id=id,
        ),
    ]


def must_inherit_from(parent, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must inherit from '%s'." % (option, parent),
            obj=obj.__class__,
            id=id,
        ),
    ]


def refer_to_missing_field(field, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' refers to '%s', which is not an attribute of '%s.%s'." % (
                option, field, obj.model._meta.app_label, obj.model._meta.object_name
            ),
            obj=obj.__class__,
            id=id,
        ),
    ]
```
### 83 - django/contrib/admin/checks.py:

Start line: 763, End line: 784

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links_item(self, obj, field_name, label):
        if field_name not in obj.list_display:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not defined in 'list_display'." % (
                        label, field_name
                    ),
                    obj=obj.__class__,
                    id='admin.E111',
                )
            ]
        else:
            return []

    def _check_list_filter(self, obj):
        if not isinstance(obj.list_filter, (list, tuple)):
            return must_be('a list or tuple', option='list_filter', obj=obj, id='admin.E112')
        else:
            return list(chain.from_iterable(
                self._check_list_filter_item(obj, item, "list_filter[%d]" % index)
                for index, item in enumerate(obj.list_filter)
            ))
```
### 84 - django/contrib/admin/checks.py:

Start line: 143, End line: 153

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields(self, obj):
        """
        Check that `autocomplete_fields` is a list or tuple of model fields.
        """
        if not isinstance(obj.autocomplete_fields, (list, tuple)):
            return must_be('a list or tuple', option='autocomplete_fields', obj=obj, id='admin.E036')
        else:
            return list(chain.from_iterable([
                self._check_autocomplete_fields_item(obj, field_name, 'autocomplete_fields[%d]' % index)
                for index, field_name in enumerate(obj.autocomplete_fields)
            ]))
```
### 85 - django/contrib/admin/checks.py:

Start line: 257, End line: 270

```python
class BaseModelAdminChecks:

    def _check_fieldsets(self, obj):
        """ Check that fieldsets is properly formatted and doesn't contain
        duplicates. """

        if obj.fieldsets is None:
            return []
        elif not isinstance(obj.fieldsets, (list, tuple)):
            return must_be('a list or tuple', option='fieldsets', obj=obj, id='admin.E007')
        else:
            seen_fields = []
            return list(chain.from_iterable(
                self._check_fieldsets_item(obj, fieldset, 'fieldsets[%d]' % index, seen_fields)
                for index, fieldset in enumerate(obj.fieldsets)
            ))
```
### 89 - django/contrib/admin/checks.py:

Start line: 470, End line: 480

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields(self, obj):
        """ Check that `prepopulated_fields` is a dictionary containing allowed
        field types. """
        if not isinstance(obj.prepopulated_fields, dict):
            return must_be('a dictionary', option='prepopulated_fields', obj=obj, id='admin.E026')
        else:
            return list(chain.from_iterable(
                self._check_prepopulated_fields_key(obj, field_name, 'prepopulated_fields') +
                self._check_prepopulated_fields_value(obj, val, 'prepopulated_fields["%s"]' % field_name)
                for field_name, val in obj.prepopulated_fields.items()
            ))
```
### 94 - django/contrib/admin/checks.py:

Start line: 541, End line: 576

```python
class BaseModelAdminChecks:

    def _check_ordering_item(self, obj, field_name, label):
        """ Check that `ordering` refers to existing fields. """
        if isinstance(field_name, (Combinable, OrderBy)):
            if not isinstance(field_name, OrderBy):
                field_name = field_name.asc()
            if isinstance(field_name.expression, F):
                field_name = field_name.expression.name
            else:
                return []
        if field_name == '?' and len(obj.ordering) != 1:
            return [
                checks.Error(
                    "The value of 'ordering' has the random ordering marker '?', "
                    "but contains other fields as well.",
                    hint='Either remove the "?", or remove the other fields.',
                    obj=obj.__class__,
                    id='admin.E032',
                )
            ]
        elif field_name == '?':
            return []
        elif LOOKUP_SEP in field_name:
            # Skip ordering in the format field1__field2 (FIXME: checking
            # this format would be nice, but it's a little fiddly).
            return []
        else:
            if field_name.startswith('-'):
                field_name = field_name[1:]
            if field_name == 'pk':
                return []
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E033')
            else:
                return []
```
### 99 - django/contrib/admin/checks.py:

Start line: 366, End line: 392

```python
class BaseModelAdminChecks:

    def _check_form(self, obj):
        """ Check that form subclasses BaseModelForm. """
        if not _issubclass(obj.form, BaseModelForm):
            return must_inherit_from(parent='BaseModelForm', option='form',
                                     obj=obj, id='admin.E016')
        else:
            return []

    def _check_filter_vertical(self, obj):
        """ Check that filter_vertical is a sequence of field names. """
        if not isinstance(obj.filter_vertical, (list, tuple)):
            return must_be('a list or tuple', option='filter_vertical', obj=obj, id='admin.E017')
        else:
            return list(chain.from_iterable(
                self._check_filter_item(obj, field_name, "filter_vertical[%d]" % index)
                for index, field_name in enumerate(obj.filter_vertical)
            ))

    def _check_filter_horizontal(self, obj):
        """ Check that filter_horizontal is a sequence of field names. """
        if not isinstance(obj.filter_horizontal, (list, tuple)):
            return must_be('a list or tuple', option='filter_horizontal', obj=obj, id='admin.E018')
        else:
            return list(chain.from_iterable(
                self._check_filter_item(obj, field_name, "filter_horizontal[%d]" % index)
                for index, field_name in enumerate(obj.filter_horizontal)
            ))
```
### 117 - django/contrib/admin/checks.py:

Start line: 408, End line: 417

```python
class BaseModelAdminChecks:

    def _check_radio_fields(self, obj):
        """ Check that `radio_fields` is a dictionary. """
        if not isinstance(obj.radio_fields, dict):
            return must_be('a dictionary', option='radio_fields', obj=obj, id='admin.E021')
        else:
            return list(chain.from_iterable(
                self._check_radio_fields_key(obj, field_name, 'radio_fields') +
                self._check_radio_fields_value(obj, val, 'radio_fields["%s"]' % field_name)
                for field_name, val in obj.radio_fields.items()
            ))
```
### 118 - django/contrib/admin/checks.py:

Start line: 516, End line: 539

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value_item(self, obj, field_name, label):
        """ For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title". """

        try:
            obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E030')
        else:
            return []

    def _check_ordering(self, obj):
        """ Check that ordering refers to existing fields or is random. """

        # ordering = None
        if obj.ordering is None:  # The default value is None
            return []
        elif not isinstance(obj.ordering, (list, tuple)):
            return must_be('a list or tuple', option='ordering', obj=obj, id='admin.E031')
        else:
            return list(chain.from_iterable(
                self._check_ordering_item(obj, field_name, 'ordering[%d]' % index)
                for index, field_name in enumerate(obj.ordering)
            ))
```
### 126 - django/contrib/admin/checks.py:

Start line: 838, End line: 860

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_select_related(self, obj):
        """ Check that list_select_related is a boolean, a list or a tuple. """

        if not isinstance(obj.list_select_related, (bool, list, tuple)):
            return must_be('a boolean, tuple or list', option='list_select_related', obj=obj, id='admin.E117')
        else:
            return []

    def _check_list_per_page(self, obj):
        """ Check that list_per_page is an integer. """

        if not isinstance(obj.list_per_page, int):
            return must_be('an integer', option='list_per_page', obj=obj, id='admin.E118')
        else:
            return []

    def _check_list_max_show_all(self, obj):
        """ Check that list_max_show_all is an integer. """

        if not isinstance(obj.list_max_show_all, int):
            return must_be('an integer', option='list_max_show_all', obj=obj, id='admin.E119')
        else:
            return []
```
### 144 - django/contrib/admin/checks.py:

Start line: 504, End line: 514

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value(self, obj, val, label):
        """ Check a value of `prepopulated_fields` dictionary, i.e. it's an
        iterable of existing fields. """

        if not isinstance(val, (list, tuple)):
            return must_be('a list or tuple', option=label, obj=obj, id='admin.E029')
        else:
            return list(chain.from_iterable(
                self._check_prepopulated_fields_value_item(obj, subfield_name, "%s[%r]" % (label, index))
                for index, subfield_name in enumerate(val)
            ))
```
### 146 - django/contrib/admin/checks.py:

Start line: 701, End line: 711

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display(self, obj):
        """ Check that list_display only contains fields or usable attributes.
        """

        if not isinstance(obj.list_display, (list, tuple)):
            return must_be('a list or tuple', option='list_display', obj=obj, id='admin.E107')
        else:
            return list(chain.from_iterable(
                self._check_list_display_item(obj, item, "list_display[%d]" % index)
                for index, item in enumerate(obj.list_display)
            ))
```
### 153 - django/contrib/admin/checks.py:

Start line: 307, End line: 318

```python
class BaseModelAdminChecks:

    def _check_field_spec(self, obj, fields, label):
        """ `fields` should be an item of `fields` or an item of
        fieldset[1]['fields'] for any `fieldset` in `fieldsets`. It should be a
        field name or a tuple of field names. """

        if isinstance(fields, tuple):
            return list(chain.from_iterable(
                self._check_field_spec_item(obj, field_name, "%s[%d]" % (label, index))
                for index, field_name in enumerate(fields)
            ))
        else:
            return self._check_field_spec_item(obj, fields, label)
```
### 176 - django/contrib/admin/checks.py:

Start line: 419, End line: 440

```python
class BaseModelAdminChecks:

    def _check_radio_fields_key(self, obj, field_name, label):
        """ Check that a key of `radio_fields` dictionary is name of existing
        field and that the field is a ForeignKey or has `choices` defined. """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E022')
        else:
            if not (isinstance(field, models.ForeignKey) or field.choices):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not an "
                        "instance of ForeignKey, and does not have a 'choices' definition." % (
                            label, field_name
                        ),
                        obj=obj.__class__,
                        id='admin.E023',
                    )
                ]
            else:
                return []
```
### 177 - django/contrib/admin/checks.py:

Start line: 482, End line: 502

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_key(self, obj, field_name, label):
        """ Check a key of `prepopulated_fields` dictionary, i.e. check that it
        is a name of existing field and the field is one of the allowed types.
        """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E027')
        else:
            if isinstance(field, (models.DateTimeField, models.ForeignKey, models.ManyToManyField)):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which must not be a DateTimeField, "
                        "a ForeignKey, a OneToOneField, or a ManyToManyField." % (label, field_name),
                        obj=obj.__class__,
                        id='admin.E028',
                    )
                ]
            else:
                return []
```
### 181 - django/contrib/admin/checks.py:

Start line: 272, End line: 305

```python
class BaseModelAdminChecks:

    def _check_fieldsets_item(self, obj, fieldset, label, seen_fields):
        """ Check an item of `fieldsets`, i.e. check that this is a pair of a
        set name and a dictionary containing "fields" key. """

        if not isinstance(fieldset, (list, tuple)):
            return must_be('a list or tuple', option=label, obj=obj, id='admin.E008')
        elif len(fieldset) != 2:
            return must_be('of length 2', option=label, obj=obj, id='admin.E009')
        elif not isinstance(fieldset[1], dict):
            return must_be('a dictionary', option='%s[1]' % label, obj=obj, id='admin.E010')
        elif 'fields' not in fieldset[1]:
            return [
                checks.Error(
                    "The value of '%s[1]' must contain the key 'fields'." % label,
                    obj=obj.__class__,
                    id='admin.E011',
                )
            ]
        elif not isinstance(fieldset[1]['fields'], (list, tuple)):
            return must_be('a list or tuple', option="%s[1]['fields']" % label, obj=obj, id='admin.E008')

        seen_fields.extend(flatten(fieldset[1]['fields']))
        if len(seen_fields) != len(set(seen_fields)):
            return [
                checks.Error(
                    "There are duplicate field(s) in '%s[1]'." % label,
                    obj=obj.__class__,
                    id='admin.E012',
                )
            ]
        return list(chain.from_iterable(
            self._check_field_spec(obj, fieldset_fields, '%s[1]["fields"]' % label)
            for fieldset_fields in fieldset[1]['fields']
        ))
```
