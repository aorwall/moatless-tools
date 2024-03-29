# django__django-11422

| **django/django** | `df46b329e0900e9e4dc1d60816c1dce6dfc1094e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 37143 |
| **Any found context length** | 37143 |
| **Avg pos** | 122.0 |
| **Min pos** | 122 |
| **Max pos** | 122 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -114,7 +114,15 @@ def iter_modules_and_files(modules, extra_files):
         # During debugging (with PyDev) the 'typing.io' and 'typing.re' objects
         # are added to sys.modules, however they are types not modules and so
         # cause issues here.
-        if not isinstance(module, ModuleType) or getattr(module, '__spec__', None) is None:
+        if not isinstance(module, ModuleType):
+            continue
+        if module.__name__ == '__main__':
+            # __main__ (usually manage.py) doesn't always have a __spec__ set.
+            # Handle this by falling back to using __file__, resolved below.
+            # See https://docs.python.org/reference/import.html#main-spec
+            sys_file_paths.append(module.__file__)
+            continue
+        if getattr(module, '__spec__', None) is None:
             continue
         spec = module.__spec__
         # Modules could be loaded from places without a concrete location. If

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/autoreload.py | 117 | 117 | 122 | 1 | 37143


## Problem Statement

```
Autoreloader with StatReloader doesn't track changes in manage.py.
Description
	 
		(last modified by Mariusz Felisiak)
	 
This is a bit convoluted, but here we go.
Environment (OSX 10.11):
$ python -V
Python 3.6.2
$ pip -V
pip 19.1.1
$ pip install Django==2.2.1
Steps to reproduce:
Run a server python manage.py runserver
Edit the manage.py file, e.g. add print(): 
def main():
	print('sth')
	os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ticket_30479.settings')
	...
Under 2.1.8 (and prior), this will trigger the auto-reloading mechanism. Under 2.2.1, it won't. As far as I can tell from the django.utils.autoreload log lines, it never sees the manage.py itself.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/autoreload.py** | 323 | 362| 266 | 266 | 4594 | 
| 2 | **1 django/utils/autoreload.py** | 577 | 589| 117 | 383 | 4594 | 
| 3 | **1 django/utils/autoreload.py** | 558 | 574| 169 | 552 | 4594 | 
| 4 | **1 django/utils/autoreload.py** | 269 | 283| 146 | 698 | 4594 | 
| 5 | **1 django/utils/autoreload.py** | 1 | 46| 230 | 928 | 4594 | 
| 6 | **1 django/utils/autoreload.py** | 473 | 494| 205 | 1133 | 4594 | 
| 7 | **1 django/utils/autoreload.py** | 531 | 555| 212 | 1345 | 4594 | 
| 8 | **1 django/utils/autoreload.py** | 285 | 320| 259 | 1604 | 4594 | 
| 9 | **1 django/utils/autoreload.py** | 365 | 395| 349 | 1953 | 4594 | 
| 10 | 2 django/core/management/commands/runserver.py | 66 | 104| 397 | 2350 | 6045 | 
| 11 | **2 django/utils/autoreload.py** | 414 | 426| 145 | 2495 | 6045 | 
| 12 | **2 django/utils/autoreload.py** | 451 | 471| 268 | 2763 | 6045 | 
| 13 | **2 django/utils/autoreload.py** | 397 | 412| 156 | 2919 | 6045 | 
| 14 | **2 django/utils/autoreload.py** | 496 | 529| 228 | 3147 | 6045 | 
| 15 | **2 django/utils/autoreload.py** | 219 | 249| 234 | 3381 | 6045 | 
| 16 | **2 django/utils/autoreload.py** | 49 | 77| 169 | 3550 | 6045 | 
| 17 | 2 django/core/management/commands/runserver.py | 106 | 162| 517 | 4067 | 6045 | 
| 18 | **2 django/utils/autoreload.py** | 99 | 106| 114 | 4181 | 6045 | 
| 19 | 2 django/core/management/commands/runserver.py | 23 | 52| 240 | 4421 | 6045 | 
| 20 | 3 django/db/migrations/autodetector.py | 1202 | 1214| 131 | 4552 | 17716 | 
| 21 | **3 django/utils/autoreload.py** | 251 | 267| 162 | 4714 | 17716 | 
| 22 | **3 django/utils/autoreload.py** | 80 | 96| 161 | 4875 | 17716 | 
| 23 | 4 django/core/management/commands/loaddata.py | 81 | 148| 593 | 5468 | 20584 | 
| 24 | 5 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 5468 | 20685 | 
| 25 | 6 django/core/management/commands/migrate.py | 161 | 242| 793 | 6261 | 23837 | 
| 26 | 6 django/db/migrations/autodetector.py | 1175 | 1200| 245 | 6506 | 23837 | 
| 27 | 7 django/db/migrations/state.py | 154 | 164| 132 | 6638 | 29055 | 
| 28 | 7 django/db/migrations/state.py | 106 | 152| 368 | 7006 | 29055 | 
| 29 | 7 django/core/management/commands/runserver.py | 1 | 20| 191 | 7197 | 29055 | 
| 30 | 7 django/db/migrations/state.py | 166 | 190| 213 | 7410 | 29055 | 
| 31 | 7 django/core/management/commands/loaddata.py | 1 | 29| 151 | 7561 | 29055 | 
| 32 | 8 django/core/management/commands/testserver.py | 29 | 55| 234 | 7795 | 29489 | 
| 33 | 8 django/core/management/commands/loaddata.py | 63 | 79| 187 | 7982 | 29489 | 
| 34 | 8 django/db/migrations/autodetector.py | 1139 | 1173| 296 | 8278 | 29489 | 
| 35 | 9 django/utils/translation/reloader.py | 1 | 30| 205 | 8483 | 29695 | 
| 36 | 10 setup.py | 1 | 66| 508 | 8991 | 30712 | 
| 37 | 10 django/db/migrations/autodetector.py | 358 | 372| 141 | 9132 | 30712 | 
| 38 | 11 django/utils/translation/__init__.py | 54 | 64| 127 | 9259 | 32996 | 
| 39 | 12 django/db/migrations/loader.py | 148 | 174| 291 | 9550 | 35913 | 
| 40 | 13 django/contrib/admin/models.py | 23 | 36| 111 | 9661 | 37023 | 
| 41 | 14 django/db/models/base.py | 585 | 644| 527 | 10188 | 51981 | 
| 42 | 15 django/core/management/__init__.py | 301 | 382| 743 | 10931 | 55253 | 
| 43 | **15 django/utils/autoreload.py** | 428 | 449| 228 | 11159 | 55253 | 
| 44 | 16 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 11947 | 58002 | 
| 45 | 16 django/core/management/commands/migrate.py | 67 | 160| 825 | 12772 | 58002 | 
| 46 | 17 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 12906 | 58693 | 
| 47 | 17 django/db/migrations/autodetector.py | 1020 | 1036| 188 | 13094 | 58693 | 
| 48 | **17 django/utils/autoreload.py** | 170 | 216| 313 | 13407 | 58693 | 
| 49 | 17 django/db/migrations/autodetector.py | 465 | 506| 418 | 13825 | 58693 | 
| 50 | 17 django/db/migrations/autodetector.py | 1038 | 1058| 136 | 13961 | 58693 | 
| 51 | 18 django/contrib/admin/sites.py | 1 | 29| 175 | 14136 | 62884 | 
| 52 | 19 django/contrib/auth/password_validation.py | 160 | 204| 351 | 14487 | 64368 | 
| 53 | 20 django/conf/global_settings.py | 499 | 638| 853 | 15340 | 69972 | 
| 54 | 20 django/core/management/commands/loaddata.py | 32 | 61| 261 | 15601 | 69972 | 
| 55 | 20 django/db/migrations/autodetector.py | 998 | 1018| 134 | 15735 | 69972 | 
| 56 | 20 django/core/management/commands/migrate.py | 1 | 18| 148 | 15883 | 69972 | 
| 57 | 21 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 16041 | 71158 | 
| 58 | 21 django/db/migrations/autodetector.py | 980 | 996| 188 | 16229 | 71158 | 
| 59 | 21 django/core/management/commands/migrate.py | 21 | 65| 369 | 16598 | 71158 | 
| 60 | 21 django/db/migrations/autodetector.py | 525 | 671| 1109 | 17707 | 71158 | 
| 61 | 22 django/db/migrations/recorder.py | 24 | 45| 145 | 17852 | 71828 | 
| 62 | 23 scripts/manage_translations.py | 1 | 29| 200 | 18052 | 73521 | 
| 63 | 24 django/db/models/signals.py | 37 | 54| 231 | 18283 | 74008 | 
| 64 | 24 django/db/migrations/autodetector.py | 1 | 15| 110 | 18393 | 74008 | 
| 65 | 25 django/utils/module_loading.py | 27 | 60| 300 | 18693 | 74751 | 
| 66 | 26 django/core/checks/templates.py | 1 | 36| 259 | 18952 | 75011 | 
| 67 | 27 django/contrib/auth/models.py | 1 | 30| 200 | 19152 | 77939 | 
| 68 | 28 django/db/backends/base/base.py | 1 | 22| 128 | 19280 | 82728 | 
| 69 | 28 django/db/migrations/autodetector.py | 508 | 524| 186 | 19466 | 82728 | 
| 70 | 28 django/db/migrations/autodetector.py | 1116 | 1137| 231 | 19697 | 82728 | 
| 71 | 28 django/conf/global_settings.py | 145 | 263| 876 | 20573 | 82728 | 
| 72 | 29 django/contrib/auth/apps.py | 1 | 29| 213 | 20786 | 82941 | 
| 73 | 29 django/db/migrations/autodetector.py | 1079 | 1114| 312 | 21098 | 82941 | 
| 74 | 29 django/core/management/commands/migrate.py | 259 | 291| 349 | 21447 | 82941 | 
| 75 | 30 django/core/management/commands/dumpdata.py | 170 | 194| 224 | 21671 | 84476 | 
| 76 | 31 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 21923 | 84729 | 
| 77 | 31 django/utils/translation/__init__.py | 1 | 36| 281 | 22204 | 84729 | 
| 78 | 32 django/core/management/commands/makemessages.py | 283 | 362| 816 | 23020 | 90289 | 
| 79 | 32 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 23322 | 90289 | 
| 80 | 33 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 23644 | 90611 | 
| 81 | 34 django/core/management/sql.py | 37 | 52| 116 | 23760 | 90996 | 
| 82 | 35 docs/_ext/djangodocs.py | 26 | 70| 385 | 24145 | 94069 | 
| 83 | 36 django/core/checks/security/csrf.py | 1 | 41| 299 | 24444 | 94368 | 
| 84 | 37 django/core/servers/basehttp.py | 197 | 214| 210 | 24654 | 96080 | 
| 85 | 37 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 24803 | 96080 | 
| 86 | 37 django/core/management/commands/migrate.py | 243 | 257| 170 | 24973 | 96080 | 
| 87 | 37 django/db/models/base.py | 1 | 45| 289 | 25262 | 96080 | 
| 88 | 37 scripts/manage_translations.py | 176 | 186| 116 | 25378 | 96080 | 
| 89 | 37 django/contrib/admin/models.py | 1 | 20| 118 | 25496 | 96080 | 
| 90 | 38 django/contrib/admin/options.py | 1310 | 1335| 232 | 25728 | 114433 | 
| 91 | 38 django/core/management/sql.py | 20 | 34| 116 | 25844 | 114433 | 
| 92 | 38 django/db/migrations/autodetector.py | 796 | 845| 570 | 26414 | 114433 | 
| 93 | 38 django/db/migrations/loader.py | 64 | 124| 548 | 26962 | 114433 | 
| 94 | 39 django/core/management/commands/sqlmigrate.py | 32 | 67| 347 | 27309 | 115041 | 
| 95 | 39 django/db/migrations/autodetector.py | 707 | 794| 789 | 28098 | 115041 | 
| 96 | 40 django/core/management/commands/check.py | 36 | 66| 214 | 28312 | 115476 | 
| 97 | 41 django/core/management/commands/shell.py | 42 | 81| 401 | 28713 | 116297 | 
| 98 | 41 django/db/migrations/loader.py | 277 | 301| 205 | 28918 | 116297 | 
| 99 | 42 django/contrib/admin/__init__.py | 1 | 30| 286 | 29204 | 116583 | 
| 100 | 43 django/db/models/options.py | 244 | 276| 343 | 29547 | 123449 | 
| 101 | 44 django/db/backends/base/creation.py | 126 | 156| 264 | 29811 | 125774 | 
| 102 | 45 django/core/checks/security/base.py | 1 | 86| 752 | 30563 | 127400 | 
| 103 | 46 django/contrib/redirects/admin.py | 1 | 11| 0 | 30563 | 127468 | 
| 104 | 46 django/core/management/commands/makemessages.py | 394 | 416| 200 | 30763 | 127468 | 
| 105 | 46 django/contrib/admin/options.py | 1725 | 1806| 744 | 31507 | 127468 | 
| 106 | 47 django/utils/log.py | 1 | 76| 492 | 31999 | 129076 | 
| 107 | 47 django/db/migrations/autodetector.py | 1060 | 1077| 180 | 32179 | 129076 | 
| 108 | 48 django/db/utils.py | 1 | 48| 150 | 32329 | 131094 | 
| 109 | 48 django/db/migrations/autodetector.py | 904 | 978| 812 | 33141 | 131094 | 
| 110 | 49 django/dispatch/__init__.py | 1 | 10| 0 | 33141 | 131159 | 
| 111 | 50 django/contrib/staticfiles/management/commands/collectstatic.py | 147 | 205| 503 | 33644 | 134003 | 
| 112 | 50 django/db/models/options.py | 1 | 36| 304 | 33948 | 134003 | 
| 113 | 50 django/db/migrations/loader.py | 1 | 49| 390 | 34338 | 134003 | 
| 114 | 50 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 34622 | 134003 | 
| 115 | 50 django/db/migrations/autodetector.py | 264 | 335| 748 | 35370 | 134003 | 
| 116 | 51 django/template/loaders/app_directories.py | 1 | 15| 0 | 35370 | 134062 | 
| 117 | 51 django/db/models/base.py | 549 | 565| 142 | 35512 | 134062 | 
| 118 | 51 django/core/management/commands/testserver.py | 1 | 27| 205 | 35717 | 134062 | 
| 119 | 51 django/db/models/base.py | 1814 | 1865| 351 | 36068 | 134062 | 
| 120 | 51 django/db/migrations/state.py | 1 | 24| 191 | 36259 | 134062 | 
| 121 | 51 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 36885 | 134062 | 
| **-> 122 <-** | **51 django/utils/autoreload.py** | 109 | 136| 258 | 37143 | 134062 | 
| 123 | 52 django/db/backends/signals.py | 1 | 4| 0 | 37143 | 134079 | 
| 124 | 53 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 37254 | 134190 | 
| 125 | 53 django/core/management/commands/check.py | 1 | 34| 226 | 37480 | 134190 | 
| 126 | 54 django/db/migrations/writer.py | 2 | 115| 886 | 38366 | 136437 | 
| 127 | 54 django/db/migrations/autodetector.py | 673 | 705| 283 | 38649 | 136437 | 
| 128 | 55 django/contrib/auth/admin.py | 1 | 22| 188 | 38837 | 138163 | 
| 129 | 56 django/http/multipartparser.py | 406 | 425| 205 | 39042 | 143190 | 
| 130 | 57 django/db/backends/postgresql/base.py | 1 | 61| 487 | 39529 | 145719 | 
| 131 | 58 django/core/signals.py | 1 | 7| 0 | 39529 | 145771 | 
| 132 | 59 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 39658 | 149666 | 
| 133 | 60 django/contrib/sessions/middleware.py | 1 | 76| 578 | 40236 | 150245 | 
| 134 | 60 django/core/management/commands/makemigrations.py | 231 | 311| 824 | 41060 | 150245 | 
| 135 | 60 django/core/management/commands/runserver.py | 54 | 64| 120 | 41180 | 150245 | 
| 136 | 61 django/db/migrations/questioner.py | 1 | 54| 468 | 41648 | 152319 | 
| 137 | 62 django/core/checks/model_checks.py | 143 | 164| 263 | 41911 | 154000 | 
| 138 | 62 django/core/management/commands/makemessages.py | 363 | 392| 231 | 42142 | 154000 | 
| 139 | 63 django/db/migrations/operations/special.py | 181 | 204| 246 | 42388 | 155558 | 
| 140 | 64 django/core/serializers/base.py | 219 | 230| 157 | 42545 | 157951 | 
| 141 | 64 django/core/management/commands/makemigrations.py | 186 | 229| 450 | 42995 | 157951 | 
| 142 | 65 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 42995 | 158027 | 
| 143 | 65 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 43287 | 158027 | 
| 144 | 65 django/core/management/commands/shell.py | 1 | 40| 268 | 43555 | 158027 | 
| 145 | 66 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 273 | 43828 | 158300 | 
| 146 | 67 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 44677 | 159149 | 
| 147 | 68 django/contrib/sites/admin.py | 1 | 9| 0 | 44677 | 159195 | 
| 148 | 68 django/db/migrations/autodetector.py | 37 | 47| 120 | 44797 | 159195 | 
| 149 | 69 django/core/management/base.py | 347 | 382| 292 | 45089 | 163582 | 
| 150 | 70 django/utils/cache.py | 116 | 131| 188 | 45277 | 167131 | 
| 151 | 71 django/db/__init__.py | 40 | 62| 118 | 45395 | 167524 | 
| 152 | 72 django/contrib/contenttypes/fields.py | 664 | 689| 254 | 45649 | 172827 | 
| 153 | 72 django/contrib/admin/options.py | 2135 | 2170| 315 | 45964 | 172827 | 
| 154 | 73 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 1 | 81| 598 | 46562 | 173445 | 
| 155 | 74 django/core/checks/messages.py | 53 | 76| 161 | 46723 | 174018 | 
| 156 | 75 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 47227 | 175284 | 
| 157 | 76 django/db/backends/utils.py | 1 | 47| 287 | 47514 | 177181 | 
| 158 | 77 django/contrib/redirects/__init__.py | 1 | 2| 0 | 47514 | 177195 | 
| 159 | 78 django/contrib/admin/checks.py | 999 | 1011| 116 | 47630 | 186209 | 
| 160 | 79 django/contrib/staticfiles/checks.py | 1 | 15| 0 | 47630 | 186285 | 
| 161 | 80 django/core/management/commands/test.py | 25 | 57| 260 | 47890 | 186694 | 
| 162 | 80 django/core/checks/model_checks.py | 166 | 199| 332 | 48222 | 186694 | 
| 163 | 81 django/contrib/syndication/__init__.py | 1 | 2| 0 | 48222 | 186711 | 
| 164 | 81 django/contrib/admin/checks.py | 620 | 639| 183 | 48405 | 186711 | 
| 165 | 82 django/core/checks/__init__.py | 1 | 25| 254 | 48659 | 186965 | 
| 166 | 82 django/contrib/admin/models.py | 39 | 72| 243 | 48902 | 186965 | 
| 167 | 83 django/contrib/sites/managers.py | 1 | 61| 385 | 49287 | 187350 | 
| 168 | 84 django/core/management/commands/startapp.py | 1 | 15| 0 | 49287 | 187451 | 
| 169 | 84 django/db/models/options.py | 369 | 395| 175 | 49462 | 187451 | 
| 170 | 85 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 49799 | 187990 | 
| 171 | 86 django/views/debug.py | 388 | 456| 575 | 50374 | 192208 | 
| 172 | 86 django/db/migrations/autodetector.py | 1265 | 1288| 240 | 50614 | 192208 | 
| 173 | 87 django/conf/__init__.py | 132 | 185| 472 | 51086 | 193994 | 
| 174 | 88 docs/conf.py | 1 | 95| 746 | 51832 | 196971 | 
| 175 | 88 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 52339 | 196971 | 
| 176 | 88 django/core/management/commands/migrate.py | 293 | 340| 401 | 52740 | 196971 | 
| 177 | 88 django/core/checks/security/base.py | 88 | 190| 747 | 53487 | 196971 | 
| 178 | 89 django/core/serializers/__init__.py | 143 | 156| 132 | 53619 | 198633 | 
| 179 | 89 django/db/utils.py | 266 | 308| 322 | 53941 | 198633 | 
| 180 | 90 django/__main__.py | 1 | 10| 0 | 53941 | 198678 | 
| 181 | 90 django/core/management/commands/compilemessages.py | 1 | 26| 157 | 54098 | 198678 | 
| 182 | 90 django/db/__init__.py | 1 | 18| 141 | 54239 | 198678 | 
| 183 | 91 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 54239 | 198771 | 
| 184 | 91 django/core/management/commands/makemessages.py | 36 | 57| 143 | 54382 | 198771 | 


### Hint

```
Thanks for the report. I simplified scenario. Regression in c8720e7696ca41f3262d5369365cc1bd72a216ca. Reproduced at 8d010f39869f107820421631111417298d1c5bb9.
Argh. I guess this is because manage.py isn't showing up in the sys.modules. I'm not sure I remember any specific manage.py handling in the old implementation, so I'm not sure how it used to work, but I should be able to fix this pretty easily.
Done a touch of debugging: iter_modules_and_files is where it gets lost. Specifically, it ends up in there twice: (<module '__future__' from '/../lib/python3.6/__future__.py'>, <module '__main__' from 'manage.py'>, <module '__main__' from 'manage.py'>, ...,) But getattr(module, "__spec__", None) is None is True so it continues onwards. I thought I managed to get one of them to have a __spec__ attr but no has_location, but I can't seem to get that again (stepping around with pdb) Digging into wtf __spec__ is None: ​Here's the py3 docs on it, which helpfully mentions that ​The one exception is __main__, where __spec__ is set to None in some cases
Tom, will you have time to work on this in the next few days?
I'm sorry for assigning it to myself Mariusz, I intended to work on it on Tuesday but work overtook me and now I am travelling for a wedding this weekend. So I doubt it I'm afraid. It seems Keryn's debugging is a great help, it should be somewhat simple to add special case handling for __main__, while __spec__ is None we can still get the filename and watch on that.
np, Tom, thanks for info. Keryn, it looks that you've already made most of the work. Would you like to prepare a patch?
```

## Patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -114,7 +114,15 @@ def iter_modules_and_files(modules, extra_files):
         # During debugging (with PyDev) the 'typing.io' and 'typing.re' objects
         # are added to sys.modules, however they are types not modules and so
         # cause issues here.
-        if not isinstance(module, ModuleType) or getattr(module, '__spec__', None) is None:
+        if not isinstance(module, ModuleType):
+            continue
+        if module.__name__ == '__main__':
+            # __main__ (usually manage.py) doesn't always have a __spec__ set.
+            # Handle this by falling back to using __file__, resolved below.
+            # See https://docs.python.org/reference/import.html#main-spec
+            sys_file_paths.append(module.__file__)
+            continue
+        if getattr(module, '__spec__', None) is None:
             continue
         spec = module.__spec__
         # Modules could be loaded from places without a concrete location. If

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -132,6 +132,10 @@ def test_module_without_spec(self):
         del module.__spec__
         self.assertEqual(autoreload.iter_modules_and_files((module,), frozenset()), frozenset())
 
+    def test_main_module_is_resolved(self):
+        main_module = sys.modules['__main__']
+        self.assertFileFound(Path(main_module.__file__))
+
 
 class TestCommonRoots(SimpleTestCase):
     def test_common_roots(self):

```


## Code snippets

### 1 - django/utils/autoreload.py:

Start line: 323, End line: 362

```python
class StatReloader(BaseReloader):
    SLEEP_TIME = 1  # Check for changes once per second.

    def tick(self):
        mtimes = {}
        while True:
            for filepath, mtime in self.snapshot_files():
                old_time = mtimes.get(filepath)
                if old_time is None:
                    logger.debug('File %s first seen with mtime %s', filepath, mtime)
                    mtimes[filepath] = mtime
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
### 2 - django/utils/autoreload.py:

Start line: 577, End line: 589

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
### 3 - django/utils/autoreload.py:

Start line: 558, End line: 574

```python
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
### 4 - django/utils/autoreload.py:

Start line: 269, End line: 283

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
### 5 - django/utils/autoreload.py:

Start line: 1, End line: 46

```python
import functools
import itertools
import logging
import os
import pathlib
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

from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple

autoreload_started = Signal()
file_changed = Signal(providing_args=['file_path', 'kind'])

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
```
### 6 - django/utils/autoreload.py:

Start line: 473, End line: 494

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
### 7 - django/utils/autoreload.py:

Start line: 531, End line: 555

```python
class WatchmanReloader(BaseReloader):

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


def get_reloader():
    """Return the most suitable reloader for this environment."""
    try:
        WatchmanReloader.check_availability()
    except WatchmanUnavailable:
        return StatReloader()
    return WatchmanReloader()
```
### 8 - django/utils/autoreload.py:

Start line: 285, End line: 320

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
### 9 - django/utils/autoreload.py:

Start line: 365, End line: 395

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
### 10 - django/core/management/commands/runserver.py:

Start line: 66, End line: 104

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
### 11 - django/utils/autoreload.py:

Start line: 414, End line: 426

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
### 12 - django/utils/autoreload.py:

Start line: 451, End line: 471

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
### 13 - django/utils/autoreload.py:

Start line: 397, End line: 412

```python
class WatchmanReloader(BaseReloader):

    @functools.lru_cache()
    def _get_clock(self, root):
        return self.client.query('clock', root)['clock']

    def _subscribe(self, directory, name, expression):
        root, rel_path = self._watch_root(directory)
        query = {
            'expression': expression,
            'fields': ['name'],
            'since': self._get_clock(root),
            'dedup_results': True,
        }
        if rel_path:
            query['relative_root'] = rel_path
        logger.debug('Issuing watchman subscription %s, for root %s. Query: %s', name, root, query)
        self.client.query('subscribe', root, name, query)
```
### 14 - django/utils/autoreload.py:

Start line: 496, End line: 529

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
```
### 15 - django/utils/autoreload.py:

Start line: 219, End line: 249

```python
class BaseReloader:
    def __init__(self):
        self.extra_files = set()
        self.directory_globs = defaultdict(set)
        self._stop_condition = threading.Event()

    def watch_dir(self, path, glob):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('%s must be absolute.' % path)
        logger.debug('Watching dir %s with glob %s.', path, glob)
        self.directory_globs[path].add(glob)

    def watch_file(self, path):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('%s must be absolute.' % path)
        logger.debug('Watching file %s.', path)
        self.extra_files.add(path)

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
### 16 - django/utils/autoreload.py:

Start line: 49, End line: 77

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
        raise _exception[0](_exception[1]).with_traceback(_exception[2])
```
### 18 - django/utils/autoreload.py:

Start line: 99, End line: 106

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
### 21 - django/utils/autoreload.py:

Start line: 251, End line: 267

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

Start line: 80, End line: 96

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
### 43 - django/utils/autoreload.py:

Start line: 428, End line: 449

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
### 48 - django/utils/autoreload.py:

Start line: 170, End line: 216

```python
def sys_path_directories():
    """
    Yield absolute directories from sys.path, ignoring entries that don't
    exist.
    """
    for path in sys.path:
        path = Path(path)
        if not path.exists():
            continue
        path = path.resolve().absolute()
        # If the path is a file (like a zip file), watch the parent directory.
        if path.is_file():
            yield path.parent
        else:
            yield path


def get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.
    """
    import django.__main__

    args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
    if sys.argv[0] == django.__main__.__file__:
        # The server was started with `python -m django runserver`.
        args += ['-m', 'django']
        args += sys.argv[1:]
    else:
        args += sys.argv
    return args


def trigger_reload(filename):
    logger.info('%s changed, reloading.', filename)
    sys.exit(3)


def restart_with_reloader():
    new_environ = {**os.environ, DJANGO_AUTORELOAD_ENV: 'true'}
    args = get_child_arguments()
    while True:
        exit_code = subprocess.call(args, env=new_environ, close_fds=False)
        if exit_code != 3:
            return exit_code
```
### 122 - django/utils/autoreload.py:

Start line: 109, End line: 136

```python
@functools.lru_cache(maxsize=1)
def iter_modules_and_files(modules, extra_files):
    """Iterate through all modules needed to be watched."""
    sys_file_paths = []
    for module in modules:
        # During debugging (with PyDev) the 'typing.io' and 'typing.re' objects
        # are added to sys.modules, however they are types not modules and so
        # cause issues here.
        if not isinstance(module, ModuleType) or getattr(module, '__spec__', None) is None:
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
        path = pathlib.Path(filename)
        if not path.exists():
            # The module could have been removed, don't fail loudly if this
            # is the case.
            continue
        results.add(path.resolve().absolute())
    return frozenset(results)
```
