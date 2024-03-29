# django__django-13837

| **django/django** | `415f50298f97fb17f841a9df38d995ccf347dfcc` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 841 |
| **Any found context length** | 841 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -216,14 +216,14 @@ def get_child_arguments():
     executable is reported to not have the .exe extension which can cause bugs
     on reloading.
     """
-    import django.__main__
-    django_main_path = Path(django.__main__.__file__)
+    import __main__
     py_script = Path(sys.argv[0])
 
     args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
-    if py_script == django_main_path:
-        # The server was started with `python -m django runserver`.
-        args += ['-m', 'django']
+    # __spec__ is set when the server was started with the `-m` option,
+    # see https://docs.python.org/3/reference/import.html#main-spec
+    if __main__.__spec__ is not None and __main__.__spec__.parent:
+        args += ['-m', __main__.__spec__.parent]
         args += sys.argv[1:]
     elif not py_script.exists():
         # sys.argv[0] may not exist for several reasons on Windows.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/autoreload.py | 219 | 226 | 3 | 1 | 841


## Problem Statement

```
Allow autoreloading of `python -m pkg_other_than_django runserver`
Description
	 
		(last modified by William Schwartz)
	 
​django.utils.autoreload.get_child_arguments detects if Python was launched as python -m django. Currently it detects only when ​-m was passed specifically django (and only in Python environments in which __file__ is set on modules, which is ​not true of all Python environments). Like #32177, this ticket aims to remove one impediment to creating Django-based command-line utilities that have their own ​__main__ sub-module while overriding Django's built-in management commands—in this case, runserver.
The fix, which I have submitted in the ​attached PR, is to use Python's ​documented way of determining if -m was used in get_child_arguments:
The top-level __main__ module is always the entry point of a ​complete Python program.
 __main__.__spec__ is not None ​if and only if Python was launched with -m or the name of a "directory, zipfile or other sys.path entry." In the latter cases, the ​documentation says
If the script name refers to a directory or zipfile, the script name is added to the start of sys.path and the __main__.py file in that location is executed as the __main__ module.
Hence __main__.__spec__.parent (which is ​usually but not always __main__.__package__) exists and is the empty string when Python is started with the name of a directory or zip file.
Therefore Python was started with -m pkg if and only if __main__.__spec__.parent == "pkg".
Following this algorithm is guaranteed to work as long as Python obeys its own documentation, and has the side benefit of avoiding use of __file__.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/autoreload.py** | 631 | 643| 117 | 117 | 5082 | 
| 2 | 2 django/contrib/gis/db/models/proxy.py | 1 | 47| 363 | 480 | 5751 | 
| **-> 3 <-** | **2 django/utils/autoreload.py** | 213 | 246| 361 | 841 | 5751 | 
| 4 | 3 django/core/management/commands/runserver.py | 67 | 105| 397 | 1238 | 7198 | 
| 5 | **3 django/utils/autoreload.py** | 313 | 328| 146 | 1384 | 7198 | 
| 6 | **3 django/utils/autoreload.py** | 1 | 56| 287 | 1671 | 7198 | 
| 7 | **3 django/utils/autoreload.py** | 603 | 628| 217 | 1888 | 7198 | 
| 8 | 3 django/core/management/commands/runserver.py | 107 | 159| 502 | 2390 | 7198 | 
| 9 | **3 django/utils/autoreload.py** | 109 | 116| 114 | 2504 | 7198 | 
| 10 | **3 django/utils/autoreload.py** | 410 | 440| 349 | 2853 | 7198 | 
| 11 | **3 django/utils/autoreload.py** | 249 | 293| 298 | 3151 | 7198 | 
| 12 | 3 django/core/management/commands/runserver.py | 24 | 53| 239 | 3390 | 7198 | 
| 13 | 4 django/core/management/__init__.py | 334 | 420| 755 | 4145 | 10706 | 
| 14 | **4 django/utils/autoreload.py** | 368 | 407| 266 | 4411 | 10706 | 
| 15 | **4 django/utils/autoreload.py** | 330 | 365| 259 | 4670 | 10706 | 
| 16 | **4 django/utils/autoreload.py** | 466 | 478| 145 | 4815 | 10706 | 
| 17 | **4 django/utils/autoreload.py** | 503 | 523| 268 | 5083 | 10706 | 
| 18 | **4 django/utils/autoreload.py** | 525 | 546| 205 | 5288 | 10706 | 
| 19 | 4 django/core/management/commands/runserver.py | 1 | 21| 204 | 5492 | 10706 | 
| 20 | **4 django/utils/autoreload.py** | 548 | 571| 177 | 5669 | 10706 | 
| 21 | **4 django/utils/autoreload.py** | 295 | 311| 162 | 5831 | 10706 | 
| 22 | **4 django/utils/autoreload.py** | 59 | 87| 156 | 5987 | 10706 | 
| 23 | **4 django/utils/autoreload.py** | 442 | 464| 221 | 6208 | 10706 | 
| 24 | 5 django/__main__.py | 1 | 10| 0 | 6208 | 10751 | 
| 25 | **5 django/utils/autoreload.py** | 573 | 600| 230 | 6438 | 10751 | 
| 26 | 6 django/template/autoreload.py | 31 | 51| 136 | 6574 | 11061 | 
| 27 | 7 django/contrib/auth/forms.py | 54 | 72| 124 | 6698 | 14173 | 
| 28 | 8 django/db/migrations/questioner.py | 1 | 54| 467 | 7165 | 16246 | 
| 29 | 9 django/db/migrations/autodetector.py | 356 | 370| 138 | 7303 | 27865 | 
| 30 | 10 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 7555 | 28118 | 
| 31 | 11 django/core/servers/basehttp.py | 204 | 221| 210 | 7765 | 29879 | 
| 32 | 12 django/core/management/commands/shell.py | 42 | 82| 401 | 8166 | 30703 | 
| 33 | 12 django/template/autoreload.py | 1 | 28| 172 | 8338 | 30703 | 
| 34 | 13 django/db/migrations/operations/special.py | 181 | 204| 246 | 8584 | 32261 | 
| 35 | 13 django/db/migrations/autodetector.py | 35 | 45| 120 | 8704 | 32261 | 
| 36 | **13 django/utils/autoreload.py** | 119 | 162| 412 | 9116 | 32261 | 
| 37 | 13 django/core/management/commands/runserver.py | 55 | 65| 120 | 9236 | 32261 | 
| 38 | 13 django/db/migrations/autodetector.py | 1191 | 1216| 245 | 9481 | 32261 | 
| 39 | 14 django/db/utils.py | 1 | 49| 177 | 9658 | 34274 | 
| 40 | 15 django/core/management/commands/testserver.py | 29 | 55| 234 | 9892 | 34707 | 
| 41 | 16 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 10714 | 37546 | 
| 42 | 17 django/utils/module_loading.py | 27 | 60| 300 | 11014 | 38289 | 
| 43 | **17 django/utils/autoreload.py** | 480 | 501| 228 | 11242 | 38289 | 
| 44 | 17 django/core/management/commands/shell.py | 1 | 40| 267 | 11509 | 38289 | 
| 45 | 18 docs/conf.py | 1 | 101| 799 | 12308 | 41326 | 
| 46 | 18 django/core/management/__init__.py | 184 | 226| 343 | 12651 | 41326 | 
| 47 | 18 django/db/migrations/autodetector.py | 1218 | 1230| 131 | 12782 | 41326 | 
| 48 | 19 django/core/management/base.py | 323 | 371| 376 | 13158 | 45963 | 
| 49 | 19 django/db/migrations/autodetector.py | 262 | 333| 748 | 13906 | 45963 | 
| 50 | 20 django/core/management/commands/migrate.py | 169 | 251| 808 | 14714 | 49219 | 
| 51 | **20 django/utils/autoreload.py** | 90 | 106| 161 | 14875 | 49219 | 
| 52 | 20 django/core/management/base.py | 158 | 238| 762 | 15637 | 49219 | 
| 53 | 20 django/core/management/commands/migrate.py | 71 | 167| 834 | 16471 | 49219 | 
| 54 | 20 django/db/migrations/autodetector.py | 463 | 507| 424 | 16895 | 49219 | 
| 55 | 20 django/db/migrations/autodetector.py | 237 | 261| 267 | 17162 | 49219 | 
| 56 | 21 django/urls/base.py | 89 | 155| 383 | 17545 | 50390 | 
| 57 | 21 django/db/migrations/autodetector.py | 435 | 461| 256 | 17801 | 50390 | 
| 58 | 22 django/urls/__init__.py | 1 | 24| 239 | 18040 | 50629 | 
| 59 | 22 django/core/management/commands/migrate.py | 1 | 18| 140 | 18180 | 50629 | 
| 60 | 22 django/db/migrations/autodetector.py | 1155 | 1189| 296 | 18476 | 50629 | 
| 61 | 23 django/db/models/options.py | 1 | 35| 300 | 18776 | 57996 | 
| 62 | 23 django/db/migrations/autodetector.py | 1095 | 1130| 312 | 19088 | 57996 | 
| 63 | 24 django/core/management/commands/test.py | 25 | 59| 296 | 19384 | 58447 | 
| 64 | 25 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 19676 | 59633 | 
| 65 | 26 django/db/migrations/loader.py | 156 | 182| 291 | 19967 | 62738 | 
| 66 | 27 django/core/management/commands/loaddata.py | 1 | 35| 177 | 20144 | 65657 | 
| 67 | 28 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 20523 | 66290 | 
| 68 | 29 django/bin/django-admin.py | 1 | 22| 138 | 20661 | 66428 | 
| 69 | 29 django/core/management/commands/migrate.py | 21 | 69| 407 | 21068 | 66428 | 
| 70 | 30 django/conf/global_settings.py | 488 | 619| 777 | 21845 | 72205 | 
| 71 | 30 django/db/migrations/autodetector.py | 1054 | 1074| 136 | 21981 | 72205 | 
| 72 | 30 django/core/management/commands/testserver.py | 1 | 27| 204 | 22185 | 72205 | 
| 73 | 31 docs/_ext/djangodocs.py | 26 | 71| 398 | 22583 | 75361 | 
| 74 | 31 django/urls/base.py | 1 | 24| 167 | 22750 | 75361 | 
| 75 | 32 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 22945 | 75556 | 
| 76 | 32 django/core/management/__init__.py | 1 | 40| 278 | 23223 | 75556 | 
| 77 | 33 django/core/management/commands/makemessages.py | 283 | 362| 814 | 24037 | 81147 | 
| 78 | 33 django/conf/global_settings.py | 151 | 266| 859 | 24896 | 81147 | 
| 79 | 33 django/core/management/commands/makemigrations.py | 24 | 59| 284 | 25180 | 81147 | 
| 80 | 34 django/middleware/common.py | 1 | 32| 247 | 25427 | 82676 | 
| 81 | 35 django/db/models/base.py | 2066 | 2117| 351 | 25778 | 99534 | 
| 82 | 35 django/db/migrations/autodetector.py | 1036 | 1052| 188 | 25966 | 99534 | 
| 83 | 36 scripts/manage_translations.py | 1 | 29| 197 | 26163 | 101190 | 
| 84 | 37 django/core/management/commands/dbshell.py | 23 | 44| 175 | 26338 | 101499 | 
| 85 | 37 django/db/migrations/autodetector.py | 1014 | 1034| 134 | 26472 | 101499 | 
| 86 | 37 django/db/migrations/autodetector.py | 1282 | 1317| 311 | 26783 | 101499 | 
| 87 | 38 django/dispatch/__init__.py | 1 | 10| 0 | 26783 | 101564 | 
| 88 | 38 django/core/management/commands/shell.py | 84 | 104| 166 | 26949 | 101564 | 
| 89 | 39 django/contrib/auth/__init__.py | 1 | 38| 241 | 27190 | 103184 | 
| 90 | 40 django/core/handlers/wsgi.py | 64 | 119| 486 | 27676 | 104943 | 
| 91 | 40 django/utils/module_loading.py | 63 | 79| 147 | 27823 | 104943 | 
| 92 | 40 django/core/management/commands/loaddata.py | 87 | 157| 640 | 28463 | 104943 | 
| 93 | 41 django/utils/translation/reloader.py | 1 | 36| 228 | 28691 | 105172 | 
| 94 | 41 django/db/migrations/loader.py | 68 | 132| 551 | 29242 | 105172 | 
| 95 | 42 django/core/management/commands/dumpdata.py | 67 | 139| 624 | 29866 | 106782 | 
| 96 | 43 django/contrib/admin/__init__.py | 1 | 25| 245 | 30111 | 107027 | 
| 97 | **43 django/utils/autoreload.py** | 165 | 210| 374 | 30485 | 107027 | 
| 98 | 43 django/db/migrations/autodetector.py | 526 | 681| 1240 | 31725 | 107027 | 
| 99 | 44 django/utils/deprecation.py | 36 | 76| 336 | 32061 | 108093 | 
| 100 | 45 django/apps/config.py | 294 | 307| 117 | 32178 | 110642 | 
| 101 | 45 django/core/management/commands/test.py | 1 | 23| 160 | 32338 | 110642 | 
| 102 | 45 scripts/manage_translations.py | 176 | 186| 116 | 32454 | 110642 | 
| 103 | 46 django/core/mail/backends/__init__.py | 1 | 2| 0 | 32454 | 110650 | 
| 104 | 46 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 32612 | 110650 | 
| 105 | 47 django/template/loaders/app_directories.py | 1 | 15| 0 | 32612 | 110709 | 
| 106 | 47 django/db/migrations/operations/special.py | 133 | 179| 304 | 32916 | 110709 | 
| 107 | 48 django/core/management/commands/startapp.py | 1 | 15| 0 | 32916 | 110810 | 
| 108 | 48 django/conf/global_settings.py | 401 | 487| 781 | 33697 | 110810 | 
| 109 | 49 django/db/__init__.py | 1 | 43| 272 | 33969 | 111082 | 
| 110 | 49 django/db/migrations/autodetector.py | 996 | 1012| 188 | 34157 | 111082 | 
| 111 | 50 django/core/checks/security/base.py | 71 | 161| 716 | 34873 | 113038 | 
| 112 | 51 django/core/management/templates.py | 120 | 183| 560 | 35433 | 115713 | 
| 113 | 51 django/core/servers/basehttp.py | 1 | 23| 164 | 35597 | 115713 | 
| 114 | 51 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 35752 | 115713 | 
| 115 | 51 django/db/migrations/loader.py | 1 | 53| 409 | 36161 | 115713 | 
| 116 | 51 django/db/migrations/autodetector.py | 1132 | 1153| 231 | 36392 | 115713 | 
| 117 | 52 django/conf/urls/__init__.py | 1 | 23| 152 | 36544 | 115865 | 
| 118 | 53 django/contrib/admin/sites.py | 1 | 29| 175 | 36719 | 120014 | 
| 119 | 54 django/utils/inspect.py | 42 | 71| 198 | 36917 | 120458 | 
| 120 | 54 django/conf/global_settings.py | 620 | 655| 244 | 37161 | 120458 | 
| 121 | 54 django/core/servers/basehttp.py | 163 | 182| 170 | 37331 | 120458 | 
| 122 | 55 django/core/checks/urls.py | 53 | 68| 128 | 37459 | 121159 | 
| 123 | 55 django/core/management/commands/dbshell.py | 1 | 21| 139 | 37598 | 121159 | 
| 124 | 55 django/db/utils.py | 255 | 297| 322 | 37920 | 121159 | 
| 125 | 55 django/core/management/commands/loaddata.py | 38 | 67| 261 | 38181 | 121159 | 
| 126 | 56 django/db/backends/sqlite3/base.py | 1 | 80| 503 | 38684 | 127207 | 
| 127 | 57 django/db/models/__init__.py | 1 | 53| 619 | 39303 | 127826 | 
| 128 | 57 docs/conf.py | 102 | 206| 899 | 40202 | 127826 | 
| 129 | 58 django/core/mail/backends/dummy.py | 1 | 11| 0 | 40202 | 127869 | 
| 130 | 59 django/utils/translation/__init__.py | 55 | 67| 131 | 40333 | 130209 | 
| 131 | 59 django/db/models/options.py | 414 | 440| 175 | 40508 | 130209 | 
| 132 | 60 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 40508 | 130310 | 
| 133 | 61 django/__init__.py | 1 | 25| 173 | 40681 | 130483 | 
| 134 | 61 django/db/migrations/autodetector.py | 1076 | 1093| 180 | 40861 | 130483 | 
| 135 | 61 django/core/management/commands/migrate.py | 253 | 270| 208 | 41069 | 130483 | 
| 136 | 62 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 41723 | 132356 | 
| 137 | 62 django/db/migrations/autodetector.py | 805 | 854| 567 | 42290 | 132356 | 
| 138 | 63 django/conf/__init__.py | 160 | 219| 546 | 42836 | 134510 | 
| 139 | 63 django/core/management/commands/loaddata.py | 69 | 85| 187 | 43023 | 134510 | 
| 140 | 64 django/http/request.py | 241 | 319| 576 | 43599 | 139982 | 
| 141 | 65 django/core/management/commands/check.py | 1 | 38| 256 | 43855 | 140454 | 
| 142 | 65 django/utils/deprecation.py | 79 | 120| 343 | 44198 | 140454 | 
| 143 | 65 django/utils/deprecation.py | 1 | 33| 209 | 44407 | 140454 | 
| 144 | 65 django/core/management/commands/makemigrations.py | 154 | 192| 313 | 44720 | 140454 | 
| 145 | 66 django/core/checks/security/csrf.py | 1 | 41| 299 | 45019 | 140753 | 
| 146 | 66 django/core/management/commands/makemessages.py | 1 | 34| 260 | 45279 | 140753 | 
| 147 | 66 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 45629 | 140753 | 
| 148 | 66 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 46136 | 140753 | 
| 149 | 66 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 46395 | 140753 | 
| 150 | 66 django/db/migrations/autodetector.py | 47 | 85| 322 | 46717 | 140753 | 
| 151 | 66 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 47508 | 140753 | 
| 152 | 66 django/core/management/base.py | 239 | 273| 313 | 47821 | 140753 | 
| 153 | 66 django/core/management/commands/makemessages.py | 363 | 400| 272 | 48093 | 140753 | 
| 154 | 66 django/core/management/templates.py | 58 | 118| 526 | 48619 | 140753 | 
| 155 | 66 django/core/management/__init__.py | 260 | 332| 721 | 49340 | 140753 | 
| 156 | 67 django/db/migrations/state.py | 105 | 151| 367 | 49707 | 145856 | 
| 157 | 67 django/db/migrations/loader.py | 55 | 66| 116 | 49823 | 145856 | 
| 158 | 68 django/contrib/sites/middleware.py | 1 | 13| 0 | 49823 | 145915 | 
| 159 | 69 django/contrib/admindocs/utils.py | 1 | 25| 151 | 49974 | 147820 | 
| 160 | 70 django/db/backends/postgresql/client.py | 1 | 55| 387 | 50361 | 148207 | 
| 161 | 71 django/db/backends/base/client.py | 1 | 27| 192 | 50553 | 148399 | 
| 162 | 71 django/core/management/base.py | 1 | 18| 115 | 50668 | 148399 | 
| 163 | 71 django/db/migrations/state.py | 153 | 163| 132 | 50800 | 148399 | 
| 164 | 72 django/db/backends/base/base.py | 1 | 23| 138 | 50938 | 153302 | 
| 165 | 72 django/core/checks/security/base.py | 164 | 189| 188 | 51126 | 153302 | 
| 166 | 72 django/utils/module_loading.py | 82 | 98| 128 | 51254 | 153302 | 
| 167 | 72 django/db/migrations/autodetector.py | 335 | 354| 196 | 51450 | 153302 | 
| 168 | 72 django/http/request.py | 1 | 50| 397 | 51847 | 153302 | 
| 169 | 72 django/db/models/base.py | 404 | 505| 871 | 52718 | 153302 | 
| 170 | 73 django/contrib/admin/utils.py | 309 | 366| 466 | 53184 | 157464 | 
| 171 | 74 django/core/management/commands/inspectdb.py | 1 | 36| 266 | 53450 | 160097 | 
| 172 | 75 django/template/backends/django.py | 1 | 45| 303 | 53753 | 160956 | 
| 173 | 75 django/core/management/commands/makemessages.py | 216 | 281| 633 | 54386 | 160956 | 
| 174 | 75 django/core/checks/security/base.py | 1 | 69| 631 | 55017 | 160956 | 
| 175 | 75 django/core/servers/basehttp.py | 101 | 123| 211 | 55228 | 160956 | 
| 176 | 76 django/contrib/admindocs/views.py | 1 | 30| 223 | 55451 | 164252 | 
| 177 | 76 django/utils/translation/__init__.py | 1 | 37| 297 | 55748 | 164252 | 
| 178 | 76 django/db/models/base.py | 1 | 50| 328 | 56076 | 164252 | 
| 179 | 77 django/contrib/auth/management/commands/createsuperuser.py | 81 | 202| 1158 | 57234 | 166315 | 
| 180 | 77 django/core/management/templates.py | 211 | 242| 236 | 57470 | 166315 | 
| 181 | 77 django/db/utils.py | 101 | 131| 329 | 57799 | 166315 | 
| 182 | 78 django/utils/deconstruct.py | 1 | 56| 385 | 58184 | 166700 | 
| 183 | 78 django/apps/config.py | 1 | 70| 558 | 58742 | 166700 | 
| 184 | 79 django/core/checks/templates.py | 1 | 36| 259 | 59001 | 166960 | 
| 185 | 80 django/db/migrations/utils.py | 1 | 18| 0 | 59001 | 167048 | 
| 186 | 81 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 59275 | 167322 | 
| 187 | 82 django/contrib/staticfiles/management/commands/collectstatic.py | 38 | 69| 297 | 59572 | 170170 | 
| 188 | 82 django/db/migrations/state.py | 165 | 189| 213 | 59785 | 170170 | 
| 189 | 82 django/core/checks/security/base.py | 227 | 248| 184 | 59969 | 170170 | 
| 190 | 83 django/db/migrations/__init__.py | 1 | 3| 0 | 59969 | 170194 | 
| 191 | 83 django/core/servers/basehttp.py | 26 | 50| 228 | 60197 | 170194 | 
| 192 | 84 django/core/checks/__init__.py | 1 | 26| 270 | 60467 | 170464 | 
| 193 | 84 django/db/migrations/autodetector.py | 913 | 994| 876 | 61343 | 170464 | 
| 194 | 84 django/core/management/commands/check.py | 40 | 71| 221 | 61564 | 170464 | 
| 195 | 84 django/db/migrations/autodetector.py | 683 | 714| 278 | 61842 | 170464 | 
| 196 | 85 django/utils/version.py | 42 | 68| 192 | 62034 | 171262 | 
| 197 | 85 django/db/migrations/autodetector.py | 509 | 525| 186 | 62220 | 171262 | 
| 198 | 86 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 62220 | 171338 | 
| 199 | 86 django/db/migrations/autodetector.py | 372 | 433| 552 | 62772 | 171338 | 
| 200 | 86 django/db/models/base.py | 212 | 322| 866 | 63638 | 171338 | 
| 201 | 87 django/core/management/utils.py | 1 | 27| 173 | 63811 | 172452 | 
| 202 | 88 django/contrib/admin/checks.py | 58 | 143| 718 | 64529 | 181589 | 
| 203 | 89 django/db/migrations/operations/base.py | 1 | 109| 804 | 65333 | 182619 | 
| 204 | 89 django/core/servers/basehttp.py | 126 | 161| 280 | 65613 | 182619 | 
| 205 | 89 django/utils/module_loading.py | 1 | 24| 165 | 65778 | 182619 | 
| 206 | 90 django/db/backends/postgresql/base.py | 1 | 62| 456 | 66234 | 185468 | 
| 207 | 91 django/core/asgi.py | 1 | 14| 0 | 66234 | 185553 | 


## Patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -216,14 +216,14 @@ def get_child_arguments():
     executable is reported to not have the .exe extension which can cause bugs
     on reloading.
     """
-    import django.__main__
-    django_main_path = Path(django.__main__.__file__)
+    import __main__
     py_script = Path(sys.argv[0])
 
     args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
-    if py_script == django_main_path:
-        # The server was started with `python -m django runserver`.
-        args += ['-m', 'django']
+    # __spec__ is set when the server was started with the `-m` option,
+    # see https://docs.python.org/3/reference/import.html#main-spec
+    if __main__.__spec__ is not None and __main__.__spec__.parent:
+        args += ['-m', __main__.__spec__.parent]
         args += sys.argv[1:]
     elif not py_script.exists():
         # sys.argv[0] may not exist for several reasons on Windows.

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -23,6 +23,7 @@
 from django.utils import autoreload
 from django.utils.autoreload import WatchmanUnavailable
 
+from .test_module import __main__ as test_main
 from .utils import on_macos_with_hfs
 
 
@@ -157,6 +158,7 @@ def test_path_with_embedded_null_bytes(self):
 
 
 class TestChildArguments(SimpleTestCase):
+    @mock.patch.dict(sys.modules, {'__main__': django.__main__})
     @mock.patch('sys.argv', [django.__main__.__file__, 'runserver'])
     @mock.patch('sys.warnoptions', [])
     def test_run_as_module(self):
@@ -165,6 +167,15 @@ def test_run_as_module(self):
             [sys.executable, '-m', 'django', 'runserver']
         )
 
+    @mock.patch.dict(sys.modules, {'__main__': test_main})
+    @mock.patch('sys.argv', [test_main.__file__, 'runserver'])
+    @mock.patch('sys.warnoptions', [])
+    def test_run_as_non_django_module(self):
+        self.assertEqual(
+            autoreload.get_child_arguments(),
+            [sys.executable, '-m', 'utils_tests.test_module', 'runserver'],
+        )
+
     @mock.patch('sys.argv', [__file__, 'runserver'])
     @mock.patch('sys.warnoptions', ['error'])
     def test_warnoptions(self):
@@ -447,7 +458,8 @@ def test_python_m_django(self):
         argv = [main, 'runserver']
         mock_call = self.patch_autoreload(argv)
         with mock.patch('django.__main__.__file__', main):
-            autoreload.restart_with_reloader()
+            with mock.patch.dict(sys.modules, {'__main__': django.__main__}):
+                autoreload.restart_with_reloader()
             self.assertEqual(mock_call.call_count, 1)
             self.assertEqual(mock_call.call_args[0][0], [self.executable, '-Wall', '-m', 'django'] + argv[1:])
 
diff --git a/tests/utils_tests/test_module/__main__.py b/tests/utils_tests/test_module/__main__.py
new file mode 100644

```


## Code snippets

### 1 - django/utils/autoreload.py:

Start line: 631, End line: 643

```python
def run_with_reloader(main_func, *args, **kwargs):
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    try:
        if os.environ.get(DJANGO_AUTORELOAD_ENV) == 'true':
            reloader = get_reloader()
            logger.info('Watching for file changes with %s', reloader.__class__.__name__)
            start_django(reloader, main_func, *args, **kwargs)
        else:
            exit_code = restart_with_reloader()
            sys.exit(exit_code)
    except KeyboardInterrupt:
        pass
```
### 2 - django/contrib/gis/db/models/proxy.py:

Start line: 1, End line: 47

```python
"""
The SpatialProxy object allows for lazy-geometries and lazy-rasters. The proxy
uses Python descriptors for instantiating and setting Geometry or Raster
objects corresponding to geographic model fields.

Thanks to Robert Coup for providing this functionality (see #4322).
"""
from django.db.models.query_utils import DeferredAttribute


class SpatialProxy(DeferredAttribute):
    def __init__(self, klass, field, load_func=None):
        """
        Initialize on the given Geometry or Raster class (not an instance)
        and the corresponding field.
        """
        self._klass = klass
        self._load_func = load_func or klass
        super().__init__(field)

    def __get__(self, instance, cls=None):
        """
        Retrieve the geometry or raster, initializing it using the
        corresponding class specified during initialization and the value of
        the field. Currently, GEOS or OGR geometries as well as GDALRasters are
        supported.
        """
        if instance is None:
            # Accessed on a class, not an instance
            return self

        # Getting the value of the field.
        try:
            geo_value = instance.__dict__[self.field.attname]
        except KeyError:
            geo_value = super().__get__(instance, cls)

        if isinstance(geo_value, self._klass):
            geo_obj = geo_value
        elif (geo_value is None) or (geo_value == ''):
            geo_obj = None
        else:
            # Otherwise, a geometry or raster object is built using the field's
            # contents, and the model's corresponding attribute is set.
            geo_obj = self._load_func(geo_value)
            setattr(instance, self.field.attname, geo_obj)
        return geo_obj
```
### 3 - django/utils/autoreload.py:

Start line: 213, End line: 246

```python
def get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.
    """
    import django.__main__
    django_main_path = Path(django.__main__.__file__)
    py_script = Path(sys.argv[0])

    args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
    if py_script == django_main_path:
        # The server was started with `python -m django runserver`.
        args += ['-m', 'django']
        args += sys.argv[1:]
    elif not py_script.exists():
        # sys.argv[0] may not exist for several reasons on Windows.
        # It may exist with a .exe extension or have a -script.py suffix.
        exe_entrypoint = py_script.with_suffix('.exe')
        if exe_entrypoint.exists():
            # Should be executed directly, ignoring sys.executable.
            # TODO: Remove str() when dropping support for PY37.
            # args parameter accepts path-like on Windows from Python 3.8.
            return [str(exe_entrypoint), *sys.argv[1:]]
        script_entrypoint = py_script.with_name('%s-script.py' % py_script.name)
        if script_entrypoint.exists():
            # Should be executed as usual.
            # TODO: Remove str() when dropping support for PY37.
            # args parameter accepts path-like on Windows from Python 3.8.
            return [*args, str(script_entrypoint), *sys.argv[1:]]
        raise RuntimeError('Script %s does not exist.' % py_script)
    else:
        args += sys.argv
    return args
```
### 4 - django/core/management/commands/runserver.py:

Start line: 67, End line: 105

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        if not settings.DEBUG and not settings.ALLOWED_HOSTS:
            raise CommandError('You must set settings.ALLOWED_HOSTS if DEBUG is False.')

        self.use_ipv6 = options['use_ipv6']
        if self.use_ipv6 and not socket.has_ipv6:
            raise CommandError('Your Python does not support IPv6.')
        self._raw_ipv6 = False
        if not options['addrport']:
            self.addr = ''
            self.port = self.default_port
        else:
            m = re.match(naiveip_re, options['addrport'])
            if m is None:
                raise CommandError('"%s" is not a valid port number '
                                   'or address:port pair.' % options['addrport'])
            self.addr, _ipv4, _ipv6, _fqdn, self.port = m.groups()
            if not self.port.isdigit():
                raise CommandError("%r is not a valid port number." % self.port)
            if self.addr:
                if _ipv6:
                    self.addr = self.addr[1:-1]
                    self.use_ipv6 = True
                    self._raw_ipv6 = True
                elif self.use_ipv6 and not _fqdn:
                    raise CommandError('"%s" is not a valid IPv6 address.' % self.addr)
        if not self.addr:
            self.addr = self.default_addr_ipv6 if self.use_ipv6 else self.default_addr
            self._raw_ipv6 = self.use_ipv6
        self.run(**options)

    def run(self, **options):
        """Run the server, using the autoreloader if needed."""
        use_reloader = options['use_reloader']

        if use_reloader:
            autoreload.run_with_reloader(self.inner_run, **options)
        else:
            self.inner_run(None, **options)
```
### 5 - django/utils/autoreload.py:

Start line: 313, End line: 328

```python
class BaseReloader:

    def run(self, django_main_thread):
        logger.debug('Waiting for apps ready_event.')
        self.wait_for_apps_ready(apps, django_main_thread)
        from django.urls import get_resolver

        # Prevent a race condition where URL modules aren't loaded when the
        # reloader starts by accessing the urlconf_module property.
        try:
            get_resolver().urlconf_module
        except Exception:
            # Loading the urlconf can result in errors during development.
            # If this occurs then swallow the error and continue.
            pass
        logger.debug('Apps ready_event triggered. Sending autoreload_started signal.')
        autoreload_started.send(sender=self)
        self.run_loop()
```
### 6 - django/utils/autoreload.py:

Start line: 1, End line: 56

```python
import functools
import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter

import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple

autoreload_started = Signal()
file_changed = Signal()

DJANGO_AUTORELOAD_ENV = 'RUN_MAIN'

logger = logging.getLogger('django.utils.autoreload')

# If an error is raised while importing a file, it's not placed in sys.modules.
# This means that any future modifications aren't caught. Keep a list of these
# file paths to allow watching them in the future.
_error_files = []
_exception = None

try:
    import termios
except ImportError:
    termios = None


try:
    import pywatchman
except ImportError:
    pywatchman = None


def is_django_module(module):
    """Return True if the given module is nested under Django."""
    return module.__name__.startswith('django.')


def is_django_path(path):
    """Return True if the given file path is nested under Django."""
    return Path(django.__file__).parent in Path(path).parents
```
### 7 - django/utils/autoreload.py:

Start line: 603, End line: 628

```python
def get_reloader():
    """Return the most suitable reloader for this environment."""
    try:
        WatchmanReloader.check_availability()
    except WatchmanUnavailable:
        return StatReloader()
    return WatchmanReloader()


def start_django(reloader, main_func, *args, **kwargs):
    ensure_echo_on()

    main_func = check_errors(main_func)
    django_main_thread = threading.Thread(target=main_func, args=args, kwargs=kwargs, name='django-main-thread')
    django_main_thread.setDaemon(True)
    django_main_thread.start()

    while not reloader.should_stop:
        try:
            reloader.run(django_main_thread)
        except WatchmanUnavailable as ex:
            # It's possible that the watchman service shuts down or otherwise
            # becomes unavailable. In that case, use the StatReloader.
            reloader = StatReloader()
            logger.error('Error connecting to Watchman: %s', ex)
            logger.info('Watching for file changes with %s', reloader.__class__.__name__)
```
### 8 - django/core/management/commands/runserver.py:

Start line: 107, End line: 159

```python
class Command(BaseCommand):

    def inner_run(self, *args, **options):
        # If an exception was silenced in ManagementUtility.execute in order
        # to be raised in the child process, raise it now.
        autoreload.raise_last_exception()

        threading = options['use_threading']
        # 'shutdown_message' is a stealth option.
        shutdown_message = options.get('shutdown_message', '')
        quit_command = 'CTRL-BREAK' if sys.platform == 'win32' else 'CONTROL-C'

        self.stdout.write("Performing system checks...\n\n")
        self.check(display_num_errors=True)
        # Need to check migrations here, so can't use the
        # requires_migrations_check attribute.
        self.check_migrations()
        now = datetime.now().strftime('%B %d, %Y - %X')
        self.stdout.write(now)
        self.stdout.write((
            "Django version %(version)s, using settings %(settings)r\n"
            "Starting development server at %(protocol)s://%(addr)s:%(port)s/\n"
            "Quit the server with %(quit_command)s."
        ) % {
            "version": self.get_version(),
            "settings": settings.SETTINGS_MODULE,
            "protocol": self.protocol,
            "addr": '[%s]' % self.addr if self._raw_ipv6 else self.addr,
            "port": self.port,
            "quit_command": quit_command,
        })

        try:
            handler = self.get_handler(*args, **options)
            run(self.addr, int(self.port), handler,
                ipv6=self.use_ipv6, threading=threading, server_cls=self.server_cls)
        except OSError as e:
            # Use helpful error messages instead of ugly tracebacks.
            ERRORS = {
                errno.EACCES: "You don't have permission to access that port.",
                errno.EADDRINUSE: "That port is already in use.",
                errno.EADDRNOTAVAIL: "That IP address can't be assigned to.",
            }
            try:
                error_text = ERRORS[e.errno]
            except KeyError:
                error_text = e
            self.stderr.write("Error: %s" % error_text)
            # Need to use an OS exit because sys.exit doesn't work in a thread
            os._exit(1)
        except KeyboardInterrupt:
            if shutdown_message:
                self.stdout.write(shutdown_message)
            sys.exit(0)
```
### 9 - django/utils/autoreload.py:

Start line: 109, End line: 116

```python
def iter_all_python_module_files():
    # This is a hot path during reloading. Create a stable sorted list of
    # modules based on the module name and pass it to iter_modules_and_files().
    # This ensures cached results are returned in the usual case that modules
    # aren't loaded on the fly.
    keys = sorted(sys.modules)
    modules = tuple(m for m in map(sys.modules.__getitem__, keys) if not isinstance(m, weakref.ProxyTypes))
    return iter_modules_and_files(modules, frozenset(_error_files))
```
### 10 - django/utils/autoreload.py:

Start line: 410, End line: 440

```python
class WatchmanReloader(BaseReloader):
    def __init__(self):
        self.roots = defaultdict(set)
        self.processed_request = threading.Event()
        self.client_timeout = int(os.environ.get('DJANGO_WATCHMAN_TIMEOUT', 5))
        super().__init__()

    @cached_property
    def client(self):
        return pywatchman.client(timeout=self.client_timeout)

    def _watch_root(self, root):
        # In practice this shouldn't occur, however, it's possible that a
        # directory that doesn't exist yet is being watched. If it's outside of
        # sys.path then this will end up a new root. How to handle this isn't
        # clear: Not adding the root will likely break when subscribing to the
        # changes, however, as this is currently an internal API,  no files
        # will be being watched outside of sys.path. Fixing this by checking
        # inside watch_glob() and watch_dir() is expensive, instead this could
        # could fall back to the StatReloader if this case is detected? For
        # now, watching its parent, if possible, is sufficient.
        if not root.exists():
            if not root.parent.exists():
                logger.warning('Unable to watch root dir %s as neither it or its parent exist.', root)
                return
            root = root.parent
        result = self.client.query('watch-project', str(root.absolute()))
        if 'warning' in result:
            logger.warning('Watchman warning: %s', result['warning'])
        logger.debug('Watchman watch-project result: %s', result)
        return result['watch'], result.get('relative_path')
```
### 11 - django/utils/autoreload.py:

Start line: 249, End line: 293

```python
def trigger_reload(filename):
    logger.info('%s changed, reloading.', filename)
    sys.exit(3)


def restart_with_reloader():
    new_environ = {**os.environ, DJANGO_AUTORELOAD_ENV: 'true'}
    args = get_child_arguments()
    while True:
        p = subprocess.run(args, env=new_environ, close_fds=False)
        if p.returncode != 3:
            return p.returncode


class BaseReloader:
    def __init__(self):
        self.extra_files = set()
        self.directory_globs = defaultdict(set)
        self._stop_condition = threading.Event()

    def watch_dir(self, path, glob):
        path = Path(path)
        try:
            path = path.absolute()
        except FileNotFoundError:
            logger.debug(
                'Unable to watch directory %s as it cannot be resolved.',
                path,
                exc_info=True,
            )
            return
        logger.debug('Watching dir %s with glob %s.', path, glob)
        self.directory_globs[path].add(glob)

    def watched_files(self, include_globs=True):
        """
        Yield all files that need to be watched, including module files and
        files within globs.
        """
        yield from iter_all_python_module_files()
        yield from self.extra_files
        if include_globs:
            for directory, patterns in self.directory_globs.items():
                for pattern in patterns:
                    yield from directory.glob(pattern)
```
### 14 - django/utils/autoreload.py:

Start line: 368, End line: 407

```python
class StatReloader(BaseReloader):
    SLEEP_TIME = 1  # Check for changes once per second.

    def tick(self):
        mtimes = {}
        while True:
            for filepath, mtime in self.snapshot_files():
                old_time = mtimes.get(filepath)
                mtimes[filepath] = mtime
                if old_time is None:
                    logger.debug('File %s first seen with mtime %s', filepath, mtime)
                    continue
                elif mtime > old_time:
                    logger.debug('File %s previous mtime: %s, current mtime: %s', filepath, old_time, mtime)
                    self.notify_file_changed(filepath)

            time.sleep(self.SLEEP_TIME)
            yield

    def snapshot_files(self):
        # watched_files may produce duplicate paths if globs overlap.
        seen_files = set()
        for file in self.watched_files():
            if file in seen_files:
                continue
            try:
                mtime = file.stat().st_mtime
            except OSError:
                # This is thrown when the file does not exist.
                continue
            seen_files.add(file)
            yield file, mtime

    @classmethod
    def check_availability(cls):
        return True


class WatchmanUnavailable(RuntimeError):
    pass
```
### 15 - django/utils/autoreload.py:

Start line: 330, End line: 365

```python
class BaseReloader:

    def run_loop(self):
        ticker = self.tick()
        while not self.should_stop:
            try:
                next(ticker)
            except StopIteration:
                break
        self.stop()

    def tick(self):
        """
        This generator is called in a loop from run_loop. It's important that
        the method takes care of pausing or otherwise waiting for a period of
        time. This split between run_loop() and tick() is to improve the
        testability of the reloader implementations by decoupling the work they
        do from the loop.
        """
        raise NotImplementedError('subclasses must implement tick().')

    @classmethod
    def check_availability(cls):
        raise NotImplementedError('subclasses must implement check_availability().')

    def notify_file_changed(self, path):
        results = file_changed.send(sender=self, file_path=path)
        logger.debug('%s notified as changed. Signal results: %s.', path, results)
        if not any(res[1] for res in results):
            trigger_reload(path)

    # These are primarily used for testing.
    @property
    def should_stop(self):
        return self._stop_condition.is_set()

    def stop(self):
        self._stop_condition.set()
```
### 16 - django/utils/autoreload.py:

Start line: 466, End line: 478

```python
class WatchmanReloader(BaseReloader):

    def _subscribe_dir(self, directory, filenames):
        if not directory.exists():
            if not directory.parent.exists():
                logger.warning('Unable to watch directory %s as neither it or its parent exist.', directory)
                return
            prefix = 'files-parent-%s' % directory.name
            filenames = ['%s/%s' % (directory.name, filename) for filename in filenames]
            directory = directory.parent
            expression = ['name', filenames, 'wholename']
        else:
            prefix = 'files'
            expression = ['name', filenames]
        self._subscribe(directory, '%s:%s' % (prefix, directory), expression)
```
### 17 - django/utils/autoreload.py:

Start line: 503, End line: 523

```python
class WatchmanReloader(BaseReloader):

    def watched_roots(self, watched_files):
        extra_directories = self.directory_globs.keys()
        watched_file_dirs = [f.parent for f in watched_files]
        sys_paths = list(sys_path_directories())
        return frozenset((*extra_directories, *watched_file_dirs, *sys_paths))

    def _update_watches(self):
        watched_files = list(self.watched_files(include_globs=False))
        found_roots = common_roots(self.watched_roots(watched_files))
        logger.debug('Watching %s files', len(watched_files))
        logger.debug('Found common roots: %s', found_roots)
        # Setup initial roots for performance, shortest roots first.
        for root in sorted(found_roots):
            self._watch_root(root)
        for directory, patterns in self.directory_globs.items():
            self._watch_glob(directory, patterns)
        # Group sorted watched_files by their parent directory.
        sorted_files = sorted(watched_files, key=lambda p: p.parent)
        for directory, group in itertools.groupby(sorted_files, key=lambda p: p.parent):
            # These paths need to be relative to the parent directory.
            self._subscribe_dir(directory, [str(p.relative_to(directory)) for p in group])
```
### 18 - django/utils/autoreload.py:

Start line: 525, End line: 546

```python
class WatchmanReloader(BaseReloader):

    def update_watches(self):
        try:
            self._update_watches()
        except Exception as ex:
            # If the service is still available, raise the original exception.
            if self.check_server_status(ex):
                raise

    def _check_subscription(self, sub):
        subscription = self.client.getSubscription(sub)
        if not subscription:
            return
        logger.debug('Watchman subscription %s has results.', sub)
        for result in subscription:
            # When using watch-project, it's not simple to get the relative
            # directory without storing some specific state. Store the full
            # path to the directory in the subscription name, prefixed by its
            # type (glob, files).
            root_directory = Path(result['subscription'].split(':', 1)[1])
            logger.debug('Found root directory %s', root_directory)
            for file in result.get('files', []):
                self.notify_file_changed(root_directory / file)
```
### 20 - django/utils/autoreload.py:

Start line: 548, End line: 571

```python
class WatchmanReloader(BaseReloader):

    def request_processed(self, **kwargs):
        logger.debug('Request processed. Setting update_watches event.')
        self.processed_request.set()

    def tick(self):
        request_finished.connect(self.request_processed)
        self.update_watches()
        while True:
            if self.processed_request.is_set():
                self.update_watches()
                self.processed_request.clear()
            try:
                self.client.receive()
            except pywatchman.SocketTimeout:
                pass
            except pywatchman.WatchmanError as ex:
                logger.debug('Watchman error: %s, checking server status.', ex)
                self.check_server_status(ex)
            else:
                for sub in list(self.client.subs.keys()):
                    self._check_subscription(sub)
            yield
            # Protect against busy loops.
            time.sleep(0.1)
```
### 21 - django/utils/autoreload.py:

Start line: 295, End line: 311

```python
class BaseReloader:

    def wait_for_apps_ready(self, app_reg, django_main_thread):
        """
        Wait until Django reports that the apps have been loaded. If the given
        thread has terminated before the apps are ready, then a SyntaxError or
        other non-recoverable error has been raised. In that case, stop waiting
        for the apps_ready event and continue processing.

        Return True if the thread is alive and the ready event has been
        triggered, or False if the thread is terminated while waiting for the
        event.
        """
        while django_main_thread.is_alive():
            if app_reg.ready_event.wait(timeout=0.1):
                return True
        else:
            logger.debug('Main Django thread has terminated before apps are ready.')
            return False
```
### 22 - django/utils/autoreload.py:

Start line: 59, End line: 87

```python
def check_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _exception
        try:
            fn(*args, **kwargs)
        except Exception:
            _exception = sys.exc_info()

            et, ev, tb = _exception

            if getattr(ev, 'filename', None) is None:
                # get the filename from the last item in the stack
                filename = traceback.extract_tb(tb)[-1][0]
            else:
                filename = ev.filename

            if filename not in _error_files:
                _error_files.append(filename)

            raise

    return wrapper


def raise_last_exception():
    global _exception
    if _exception is not None:
        raise _exception[1]
```
### 23 - django/utils/autoreload.py:

Start line: 442, End line: 464

```python
class WatchmanReloader(BaseReloader):

    @functools.lru_cache()
    def _get_clock(self, root):
        return self.client.query('clock', root)['clock']

    def _subscribe(self, directory, name, expression):
        root, rel_path = self._watch_root(directory)
        # Only receive notifications of files changing, filtering out other types
        # like special files: https://facebook.github.io/watchman/docs/type
        only_files_expression = [
            'allof',
            ['anyof', ['type', 'f'], ['type', 'l']],
            expression
        ]
        query = {
            'expression': only_files_expression,
            'fields': ['name'],
            'since': self._get_clock(root),
            'dedup_results': True,
        }
        if rel_path:
            query['relative_root'] = rel_path
        logger.debug('Issuing watchman subscription %s, for root %s. Query: %s', name, root, query)
        self.client.query('subscribe', root, name, query)
```
### 25 - django/utils/autoreload.py:

Start line: 573, End line: 600

```python
class WatchmanReloader(BaseReloader):

    def stop(self):
        self.client.close()
        super().stop()

    def check_server_status(self, inner_ex=None):
        """Return True if the server is available."""
        try:
            self.client.query('version')
        except Exception:
            raise WatchmanUnavailable(str(inner_ex)) from inner_ex
        return True

    @classmethod
    def check_availability(cls):
        if not pywatchman:
            raise WatchmanUnavailable('pywatchman not installed.')
        client = pywatchman.client(timeout=0.1)
        try:
            result = client.capabilityCheck()
        except Exception:
            # The service is down?
            raise WatchmanUnavailable('Cannot connect to the watchman service.')
        version = get_version_tuple(result['version'])
        # Watchman 4.9 includes multiple improvements to watching project
        # directories as well as case insensitive filesystems.
        logger.debug('Watchman version %s', version)
        if version < (4, 9):
            raise WatchmanUnavailable('Watchman 4.9 or later is required.')
```
### 36 - django/utils/autoreload.py:

Start line: 119, End line: 162

```python
@functools.lru_cache(maxsize=1)
def iter_modules_and_files(modules, extra_files):
    """Iterate through all modules needed to be watched."""
    sys_file_paths = []
    for module in modules:
        # During debugging (with PyDev) the 'typing.io' and 'typing.re' objects
        # are added to sys.modules, however they are types not modules and so
        # cause issues here.
        if not isinstance(module, ModuleType):
            continue
        if module.__name__ == '__main__':
            # __main__ (usually manage.py) doesn't always have a __spec__ set.
            # Handle this by falling back to using __file__, resolved below.
            # See https://docs.python.org/reference/import.html#main-spec
            # __file__ may not exists, e.g. when running ipdb debugger.
            if hasattr(module, '__file__'):
                sys_file_paths.append(module.__file__)
            continue
        if getattr(module, '__spec__', None) is None:
            continue
        spec = module.__spec__
        # Modules could be loaded from places without a concrete location. If
        # this is the case, skip them.
        if spec.has_location:
            origin = spec.loader.archive if isinstance(spec.loader, zipimporter) else spec.origin
            sys_file_paths.append(origin)

    results = set()
    for filename in itertools.chain(sys_file_paths, extra_files):
        if not filename:
            continue
        path = Path(filename)
        try:
            if not path.exists():
                # The module could have been removed, don't fail loudly if this
                # is the case.
                continue
        except ValueError as e:
            # Network filesystems may return null bytes in file paths.
            logger.debug('"%s" raised when resolving path: "%s"', e, path)
            continue
        resolved_path = path.resolve().absolute()
        results.add(resolved_path)
    return frozenset(results)
```
### 43 - django/utils/autoreload.py:

Start line: 480, End line: 501

```python
class WatchmanReloader(BaseReloader):

    def _watch_glob(self, directory, patterns):
        """
        Watch a directory with a specific glob. If the directory doesn't yet
        exist, attempt to watch the parent directory and amend the patterns to
        include this. It's important this method isn't called more than one per
        directory when updating all subscriptions. Subsequent calls will
        overwrite the named subscription, so it must include all possible glob
        expressions.
        """
        prefix = 'glob'
        if not directory.exists():
            if not directory.parent.exists():
                logger.warning('Unable to watch directory %s as neither it or its parent exist.', directory)
                return
            prefix = 'glob-parent-%s' % directory.name
            patterns = ['%s/%s' % (directory.name, pattern) for pattern in patterns]
            directory = directory.parent

        expression = ['anyof']
        for pattern in patterns:
            expression.append(['match', pattern, 'wholename'])
        self._subscribe(directory, '%s:%s' % (prefix, directory), expression)
```
### 51 - django/utils/autoreload.py:

Start line: 90, End line: 106

```python
def ensure_echo_on():
    """
    Ensure that echo mode is enabled. Some tools such as PDB disable
    it which causes usability issues after reload.
    """
    if not termios or not sys.stdin.isatty():
        return
    attr_list = termios.tcgetattr(sys.stdin)
    if not attr_list[3] & termios.ECHO:
        attr_list[3] |= termios.ECHO
        if hasattr(signal, 'SIGTTOU'):
            old_handler = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        else:
            old_handler = None
        termios.tcsetattr(sys.stdin, termios.TCSANOW, attr_list)
        if old_handler is not None:
            signal.signal(signal.SIGTTOU, old_handler)
```
### 97 - django/utils/autoreload.py:

Start line: 165, End line: 210

```python
@functools.lru_cache(maxsize=1)
def common_roots(paths):
    """
    Return a tuple of common roots that are shared between the given paths.
    File system watchers operate on directories and aren't cheap to create.
    Try to find the minimum set of directories to watch that encompass all of
    the files that need to be watched.
    """
    # Inspired from Werkzeug:
    # https://github.com/pallets/werkzeug/blob/7477be2853df70a022d9613e765581b9411c3c39/werkzeug/_reloader.py
    # Create a sorted list of the path components, longest first.
    path_parts = sorted([x.parts for x in paths], key=len, reverse=True)
    tree = {}
    for chunks in path_parts:
        node = tree
        # Add each part of the path to the tree.
        for chunk in chunks:
            node = node.setdefault(chunk, {})
        # Clear the last leaf in the tree.
        node.clear()

    # Turn the tree into a list of Path instances.
    def _walk(node, path):
        for prefix, child in node.items():
            yield from _walk(child, path + (prefix,))
        if not node:
            yield Path(*path)

    return tuple(_walk(tree, ()))


def sys_path_directories():
    """
    Yield absolute directories from sys.path, ignoring entries that don't
    exist.
    """
    for path in sys.path:
        path = Path(path)
        if not path.exists():
            continue
        resolved_path = path.resolve().absolute()
        # If the path is a file (like a zip file), watch the parent directory.
        if resolved_path.is_file():
            yield resolved_path.parent
        else:
            yield resolved_path
```
