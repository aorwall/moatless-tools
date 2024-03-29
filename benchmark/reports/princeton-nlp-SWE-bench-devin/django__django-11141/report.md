# django__django-11141

| **django/django** | `5d9cf79baf07fc4aed7ad1b06990532a65378155` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 938 |
| **Any found context length** | 938 |
| **Avg pos** | 6.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/loader.py b/django/db/migrations/loader.py
--- a/django/db/migrations/loader.py
+++ b/django/db/migrations/loader.py
@@ -84,11 +84,6 @@ def load_disk(self):
                     continue
                 raise
             else:
-                # Empty directories are namespaces.
-                # getattr() needed on PY36 and older (replace w/attribute access).
-                if getattr(module, '__file__', None) is None:
-                    self.unmigrated_apps.add(app_config.label)
-                    continue
                 # Module is not a package (e.g. migrations.py).
                 if not hasattr(module, '__path__'):
                     self.unmigrated_apps.add(app_config.label)
@@ -96,11 +91,14 @@ def load_disk(self):
                 # Force a reload if it's already loaded (tests need this)
                 if was_loaded:
                     reload(module)
-            self.migrated_apps.add(app_config.label)
             migration_names = {
                 name for _, name, is_pkg in pkgutil.iter_modules(module.__path__)
                 if not is_pkg and name[0] not in '_~'
             }
+            if migration_names or self.ignore_no_migrations:
+                self.migrated_apps.add(app_config.label)
+            else:
+                self.unmigrated_apps.add(app_config.label)
             # Load migrations
             for migration_name in migration_names:
                 migration_path = '%s.%s' % (module_name, migration_name)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/loader.py | 87 | 91 | 3 | 1 | 938
| django/db/migrations/loader.py | 99 | 99 | 3 | 1 | 938


## Problem Statement

```
Allow migrations directories without __init__.py files
Description
	 
		(last modified by Tim Graham)
	 
Background: In python 3 a package with no __init__.py is implicitly a namespace package, so it has no __file__ attribute. 
The migrate command currently checks for existence of a __file__ attribute on the migrations package. This check was introduced in #21015, because the __file__ attribute was used in migration file discovery. 
However, in #23406 migration file discovery was changed to use pkgutil.iter_modules (), instead of direct filesystem access. pkgutil. iter_modules() uses the package's __path__ list, which exists on implicit namespace packages.
As a result, the __file__ check is no longer needed, and in fact prevents migrate from working on namespace packages (implicit or otherwise). 
Related work: #29091

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/migrations/loader.py** | 1 | 49| 390 | 390 | 2917 | 
| 2 | 2 django/db/migrations/__init__.py | 1 | 3| 0 | 390 | 2941 | 
| **-> 3 <-** | **2 django/db/migrations/loader.py** | 64 | 124| 548 | 938 | 2941 | 
| 4 | **2 django/db/migrations/loader.py** | 148 | 174| 291 | 1229 | 2941 | 
| 5 | 3 django/db/migrations/executor.py | 298 | 377| 712 | 1941 | 6233 | 
| 6 | 4 django/core/management/commands/migrate.py | 21 | 65| 369 | 2310 | 9385 | 
| 7 | 5 django/db/migrations/autodetector.py | 358 | 372| 141 | 2451 | 21078 | 
| 8 | 6 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 3239 | 23827 | 
| 9 | 6 django/core/management/commands/migrate.py | 67 | 160| 825 | 4064 | 23827 | 
| 10 | 7 django/db/migrations/writer.py | 201 | 301| 619 | 4683 | 26074 | 
| 11 | 8 django/db/migrations/questioner.py | 1 | 54| 468 | 5151 | 28148 | 
| 12 | 9 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 308 | 5459 | 28456 | 
| 13 | 9 django/core/management/commands/migrate.py | 1 | 18| 148 | 5607 | 28456 | 
| 14 | 9 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 5909 | 28456 | 
| 15 | 9 django/core/management/commands/migrate.py | 161 | 242| 793 | 6702 | 28456 | 
| 16 | 9 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 6986 | 28456 | 
| 17 | 9 django/db/migrations/writer.py | 118 | 199| 744 | 7730 | 28456 | 
| 18 | 9 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 7879 | 28456 | 
| 19 | 10 django/db/migrations/utils.py | 1 | 18| 0 | 7879 | 28544 | 
| 20 | 11 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 8041 | 28706 | 
| 21 | 12 django/core/management/commands/squashmigrations.py | 202 | 215| 112 | 8153 | 30577 | 
| 22 | 13 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 8153 | 30653 | 
| 23 | **13 django/db/migrations/loader.py** | 51 | 62| 116 | 8269 | 30653 | 
| 24 | 13 django/db/migrations/questioner.py | 227 | 240| 123 | 8392 | 30653 | 
| 25 | 14 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 275 | 8667 | 30928 | 
| 26 | 15 django/db/migrations/graph.py | 44 | 58| 121 | 8788 | 33531 | 
| 27 | 16 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 9110 | 33853 | 
| 28 | 16 django/core/management/commands/makemigrations.py | 186 | 229| 450 | 9560 | 33853 | 
| 29 | 17 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 10409 | 34702 | 
| 30 | **17 django/db/migrations/loader.py** | 277 | 301| 205 | 10614 | 34702 | 
| 31 | **17 django/db/migrations/loader.py** | 176 | 197| 213 | 10827 | 34702 | 
| 32 | 18 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 11034 | 34909 | 
| 33 | 19 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 192 | 11226 | 35101 | 
| 34 | 19 django/core/management/commands/migrate.py | 259 | 291| 349 | 11575 | 35101 | 
| 35 | 19 django/db/migrations/autodetector.py | 1178 | 1203| 245 | 11820 | 35101 | 
| 36 | 20 django/db/migrations/exceptions.py | 1 | 55| 250 | 12070 | 35352 | 
| 37 | 20 django/db/migrations/autodetector.py | 239 | 263| 267 | 12337 | 35352 | 
| 38 | 21 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 12495 | 36538 | 
| 39 | 22 django/core/management/commands/sqlmigrate.py | 32 | 69| 371 | 12866 | 37170 | 
| 40 | 23 django/db/migrations/recorder.py | 24 | 45| 145 | 13011 | 37840 | 
| 41 | 23 django/db/migrations/autodetector.py | 89 | 101| 118 | 13129 | 37840 | 
| 42 | **23 django/db/migrations/loader.py** | 199 | 275| 768 | 13897 | 37840 | 
| 43 | 23 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 14189 | 37840 | 
| 44 | 23 django/db/migrations/autodetector.py | 1001 | 1021| 134 | 14323 | 37840 | 
| 45 | 23 django/db/migrations/autodetector.py | 264 | 335| 748 | 15071 | 37840 | 
| 46 | 23 django/core/management/commands/showmigrations.py | 65 | 103| 411 | 15482 | 37840 | 
| 47 | 23 django/core/management/commands/makemigrations.py | 231 | 311| 824 | 16306 | 37840 | 
| 48 | 24 django/core/files/__init__.py | 1 | 4| 0 | 16306 | 37855 | 
| 49 | 24 django/db/migrations/graph.py | 61 | 97| 337 | 16643 | 37855 | 
| 50 | **24 django/db/migrations/loader.py** | 126 | 146| 211 | 16854 | 37855 | 
| 51 | 25 django/db/migrations/migration.py | 1 | 88| 714 | 17568 | 39488 | 
| 52 | 25 django/db/migrations/autodetector.py | 1205 | 1217| 131 | 17699 | 39488 | 
| 53 | 25 django/db/migrations/executor.py | 281 | 296| 165 | 17864 | 39488 | 
| 54 | 25 django/core/management/commands/migrate.py | 243 | 257| 170 | 18034 | 39488 | 
| 55 | 25 django/db/migrations/autodetector.py | 1041 | 1061| 136 | 18170 | 39488 | 
| 56 | 25 django/db/migrations/autodetector.py | 983 | 999| 188 | 18358 | 39488 | 
| 57 | 25 django/db/migrations/autodetector.py | 1 | 15| 110 | 18468 | 39488 | 
| 58 | 25 django/db/migrations/autodetector.py | 1268 | 1291| 240 | 18708 | 39488 | 
| 59 | 26 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 18903 | 39683 | 
| 60 | 26 django/db/migrations/recorder.py | 47 | 96| 378 | 19281 | 39683 | 
| 61 | 26 django/db/migrations/autodetector.py | 1063 | 1080| 180 | 19461 | 39683 | 
| 62 | 27 django/utils/module_loading.py | 63 | 79| 147 | 19608 | 40426 | 
| 63 | 27 django/db/migrations/autodetector.py | 437 | 463| 256 | 19864 | 40426 | 
| 64 | 28 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 19975 | 40537 | 
| 65 | 28 django/db/migrations/autodetector.py | 37 | 47| 120 | 20095 | 40537 | 
| 66 | 29 django/db/utils.py | 272 | 314| 322 | 20417 | 42649 | 
| 67 | 30 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 20634 | 42866 | 
| 68 | 30 django/db/migrations/autodetector.py | 1142 | 1176| 296 | 20930 | 42866 | 
| 69 | 30 django/db/migrations/recorder.py | 1 | 22| 153 | 21083 | 42866 | 
| 70 | 30 django/db/migrations/autodetector.py | 1082 | 1117| 312 | 21395 | 42866 | 
| 71 | 31 django/core/management/base.py | 451 | 483| 282 | 21677 | 47253 | 
| 72 | 31 django/db/migrations/autodetector.py | 1023 | 1039| 188 | 21865 | 47253 | 
| 73 | 32 django/utils/autoreload.py | 98 | 105| 114 | 21979 | 51970 | 
| 74 | 32 django/core/management/commands/showmigrations.py | 105 | 148| 340 | 22319 | 51970 | 
| 75 | 32 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 23110 | 51970 | 
| 76 | 32 django/db/migrations/migration.py | 127 | 194| 585 | 23695 | 51970 | 
| 77 | 32 django/core/management/commands/squashmigrations.py | 136 | 200| 652 | 24347 | 51970 | 
| 78 | 32 django/utils/module_loading.py | 27 | 60| 300 | 24647 | 51970 | 
| 79 | 32 django/db/migrations/autodetector.py | 1119 | 1140| 231 | 24878 | 51970 | 
| 80 | 32 django/db/migrations/autodetector.py | 1219 | 1266| 429 | 25307 | 51970 | 
| 81 | 32 django/db/migrations/executor.py | 64 | 80| 168 | 25475 | 51970 | 
| 82 | 32 django/db/migrations/autodetector.py | 200 | 222| 239 | 25714 | 51970 | 
| 83 | 33 django/db/migrations/operations/special.py | 181 | 204| 246 | 25960 | 53528 | 
| 84 | 33 django/db/migrations/autodetector.py | 103 | 198| 769 | 26729 | 53528 | 
| 85 | 34 django/core/files/move.py | 1 | 27| 148 | 26877 | 54212 | 
| 86 | 34 django/db/migrations/autodetector.py | 883 | 902| 184 | 27061 | 54212 | 
| 87 | 34 django/db/migrations/autodetector.py | 224 | 237| 199 | 27260 | 54212 | 
| 88 | 35 django/dispatch/__init__.py | 1 | 10| 0 | 27260 | 54277 | 
| 89 | 35 django/db/migrations/autodetector.py | 465 | 506| 418 | 27678 | 54277 | 
| 90 | 36 setup.py | 69 | 138| 536 | 28214 | 55300 | 
| 91 | 37 django/db/migrations/operations/base.py | 1 | 102| 783 | 28997 | 56379 | 
| 92 | 37 django/db/migrations/graph.py | 259 | 280| 179 | 29176 | 56379 | 
| 93 | 37 django/db/migrations/operations/base.py | 104 | 142| 299 | 29475 | 56379 | 
| 94 | 37 django/core/management/commands/sqlmigrate.py | 1 | 30| 266 | 29741 | 56379 | 
| 95 | 37 django/db/migrations/autodetector.py | 337 | 356| 196 | 29937 | 56379 | 
| 96 | 37 django/db/migrations/autodetector.py | 525 | 671| 1109 | 31046 | 56379 | 
| 97 | 37 django/db/migrations/questioner.py | 187 | 205| 237 | 31283 | 56379 | 
| 98 | 37 django/core/management/commands/migrate.py | 342 | 365| 180 | 31463 | 56379 | 
| 99 | 37 django/db/migrations/autodetector.py | 508 | 524| 186 | 31649 | 56379 | 
| 100 | 37 django/db/migrations/questioner.py | 56 | 81| 220 | 31869 | 56379 | 
| 101 | 37 django/db/migrations/executor.py | 1 | 62| 496 | 32365 | 56379 | 
| 102 | 37 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 32715 | 56379 | 
| 103 | 38 django/utils/deconstruct.py | 1 | 56| 385 | 33100 | 56764 | 
| 104 | 39 django/contrib/postgres/__init__.py | 1 | 2| 0 | 33100 | 56778 | 
| 105 | 39 setup.py | 1 | 66| 508 | 33608 | 56778 | 
| 106 | 40 django/db/migrations/state.py | 154 | 164| 132 | 33740 | 61996 | 
| 107 | 40 django/utils/module_loading.py | 82 | 98| 128 | 33868 | 61996 | 
| 108 | 41 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 33868 | 62097 | 
| 109 | 42 django/core/management/sql.py | 20 | 34| 116 | 33984 | 62482 | 
| 110 | 43 django/urls/__init__.py | 1 | 24| 239 | 34223 | 62721 | 
| 111 | 43 django/core/management/commands/migrate.py | 293 | 340| 401 | 34624 | 62721 | 
| 112 | 44 django/utils/translation/__init__.py | 55 | 65| 127 | 34751 | 65047 | 
| 113 | 44 django/db/migrations/state.py | 580 | 599| 188 | 34939 | 65047 | 
| 114 | 45 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 34939 | 65100 | 
| 115 | **45 django/db/migrations/loader.py** | 303 | 325| 206 | 35145 | 65100 | 
| 116 | 46 django/forms/__init__.py | 1 | 12| 0 | 35145 | 65190 | 
| 117 | 47 django/db/models/fields/__init__.py | 1597 | 1622| 206 | 35351 | 82597 | 
| 118 | 48 django/views/__init__.py | 1 | 4| 0 | 35351 | 82612 | 
| 119 | 49 django/contrib/gis/__init__.py | 1 | 2| 0 | 35351 | 82626 | 
| 120 | 50 django/contrib/staticfiles/storage.py | 252 | 322| 575 | 35926 | 86156 | 
| 121 | 50 django/db/migrations/graph.py | 193 | 243| 450 | 36376 | 86156 | 
| 122 | 51 django/contrib/staticfiles/utils.py | 1 | 39| 248 | 36624 | 86609 | 
| 123 | 51 django/db/migrations/autodetector.py | 904 | 981| 834 | 37458 | 86609 | 
| 124 | 51 django/db/migrations/graph.py | 99 | 120| 192 | 37650 | 86609 | 
| 125 | 51 django/db/migrations/executor.py | 255 | 279| 227 | 37877 | 86609 | 
| 126 | 52 django/contrib/humanize/__init__.py | 1 | 2| 0 | 37877 | 86625 | 
| 127 | 52 django/utils/module_loading.py | 1 | 24| 165 | 38042 | 86625 | 
| 128 | 52 django/db/migrations/state.py | 166 | 190| 213 | 38255 | 86625 | 
| 129 | 52 django/core/files/move.py | 30 | 88| 535 | 38790 | 86625 | 
| 130 | 53 django/db/migrations/serializer.py | 141 | 160| 223 | 39013 | 89173 | 
| 131 | 53 django/db/migrations/writer.py | 2 | 115| 886 | 39899 | 89173 | 
| 132 | 54 django/utils/deprecation.py | 30 | 70| 336 | 40235 | 89851 | 
| 133 | 55 django/conf/global_settings.py | 496 | 640| 900 | 41135 | 95479 | 
| 134 | 56 django/db/models/__init__.py | 1 | 51| 576 | 41711 | 96055 | 
| 135 | 56 django/db/migrations/questioner.py | 84 | 107| 187 | 41898 | 96055 | 
| 136 | 57 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 42035 | 96192 | 
| 137 | 57 django/utils/autoreload.py | 430 | 442| 145 | 42180 | 96192 | 
| 138 | 57 django/core/management/sql.py | 37 | 52| 116 | 42296 | 96192 | 
| 139 | 57 django/db/migrations/state.py | 1 | 24| 191 | 42487 | 96192 | 
| 140 | 57 django/db/migrations/state.py | 106 | 152| 368 | 42855 | 96192 | 
| 141 | 57 django/utils/autoreload.py | 221 | 265| 298 | 43153 | 96192 | 
| 142 | 58 django/apps/__init__.py | 1 | 5| 0 | 43153 | 96215 | 
| 143 | 59 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 43153 | 96290 | 
| 144 | 60 django/contrib/staticfiles/__init__.py | 1 | 2| 0 | 43153 | 96304 | 
| 145 | 61 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 43153 | 96381 | 
| 146 | 61 django/utils/autoreload.py | 108 | 151| 415 | 43568 | 96381 | 
| 147 | 62 django/core/management/commands/makemessages.py | 36 | 57| 143 | 43711 | 101926 | 
| 148 | 62 django/db/migrations/executor.py | 152 | 211| 567 | 44278 | 101926 | 
| 149 | 63 django/core/mail/backends/__init__.py | 1 | 2| 0 | 44278 | 101934 | 
| 150 | 64 django/template/backends/django.py | 114 | 130| 115 | 44393 | 102790 | 
| 151 | 65 django/db/models/sql/__init__.py | 1 | 7| 0 | 44393 | 102850 | 
| 152 | 65 django/db/migrations/state.py | 496 | 528| 250 | 44643 | 102850 | 
| 153 | 66 django/db/backends/base/features.py | 1 | 115| 900 | 45543 | 105402 | 
| 154 | 66 django/db/migrations/state.py | 601 | 612| 136 | 45679 | 105402 | 
| 155 | 66 django/db/models/fields/__init__.py | 363 | 389| 199 | 45878 | 105402 | 
| 156 | 67 django/utils/translation/reloader.py | 1 | 30| 201 | 46079 | 105604 | 
| 157 | 67 django/db/migrations/state.py | 27 | 54| 233 | 46312 | 105604 | 
| 158 | 68 django/contrib/postgres/forms/__init__.py | 1 | 5| 0 | 46312 | 105646 | 
| 159 | 69 django/conf/__init__.py | 161 | 220| 541 | 46853 | 107705 | 
| 160 | 69 django/db/migrations/questioner.py | 109 | 141| 290 | 47143 | 107705 | 
| 161 | 70 django/template/loaders/app_directories.py | 1 | 15| 0 | 47143 | 107764 | 
| 162 | 70 django/db/migrations/autodetector.py | 18 | 35| 185 | 47328 | 107764 | 
| 163 | 70 django/core/management/commands/makemessages.py | 116 | 151| 269 | 47597 | 107764 | 
| 164 | 71 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 47747 | 107914 | 
| 165 | 71 django/db/migrations/autodetector.py | 707 | 794| 789 | 48536 | 107914 | 
| 166 | 71 django/db/migrations/autodetector.py | 847 | 881| 339 | 48875 | 107914 | 
| 167 | 72 django/db/backends/base/schema.py | 31 | 41| 120 | 48995 | 119230 | 
| 168 | 73 django/core/checks/__init__.py | 1 | 25| 254 | 49249 | 119484 | 
| 169 | 73 django/db/migrations/autodetector.py | 374 | 435| 552 | 49801 | 119484 | 
| 170 | 74 django/core/checks/urls.py | 53 | 68| 128 | 49929 | 120185 | 
| 171 | 75 django/core/files/storage.py | 1 | 22| 158 | 50087 | 123054 | 
| 172 | 76 django/db/models/base.py | 1832 | 1856| 175 | 50262 | 138343 | 
| 173 | 76 django/utils/autoreload.py | 467 | 487| 268 | 50530 | 138343 | 
| 174 | 76 django/utils/autoreload.py | 1 | 45| 227 | 50757 | 138343 | 
| 175 | 77 django/contrib/gis/db/models/sql/__init__.py | 1 | 8| 0 | 50757 | 138378 | 
| 176 | 78 django/contrib/gis/sitemaps/__init__.py | 1 | 5| 0 | 50757 | 138420 | 
| 177 | 78 django/utils/autoreload.py | 185 | 218| 231 | 50988 | 138420 | 
| 178 | 78 django/core/files/storage.py | 233 | 294| 524 | 51512 | 138420 | 
| 179 | 79 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 51512 | 138488 | 
| 180 | 79 django/db/migrations/autodetector.py | 1293 | 1324| 314 | 51826 | 138488 | 
| 181 | 80 django/core/checks/templates.py | 1 | 36| 259 | 52085 | 138748 | 
| 182 | 81 django/contrib/gis/forms/__init__.py | 1 | 9| 0 | 52085 | 138822 | 
| 183 | 82 django/core/validators.py | 468 | 503| 249 | 52334 | 143144 | 
| 184 | 83 django/core/management/commands/loaddata.py | 217 | 273| 549 | 52883 | 146012 | 
| 185 | 83 django/db/migrations/questioner.py | 162 | 185| 246 | 53129 | 146012 | 
| 186 | 84 django/core/cache/backends/filebased.py | 115 | 158| 337 | 53466 | 147178 | 
| 187 | 85 django/contrib/admin/utils.py | 285 | 303| 175 | 53641 | 151266 | 
| 188 | 85 django/db/migrations/executor.py | 127 | 150| 235 | 53876 | 151266 | 
| 189 | 86 django/db/migrations/operations/models.py | 839 | 874| 347 | 54223 | 157962 | 
| 190 | 87 django/contrib/admin/checks.py | 1 | 54| 329 | 54552 | 166978 | 
| 191 | 88 django/contrib/sites/__init__.py | 1 | 2| 0 | 54552 | 166992 | 
| 192 | 89 django/core/management/utils.py | 128 | 154| 198 | 54750 | 168106 | 
| 193 | 90 django/template/loaders/filesystem.py | 1 | 47| 287 | 55037 | 168393 | 
| 194 | 91 django/contrib/gis/utils/__init__.py | 1 | 15| 139 | 55176 | 168533 | 
| 195 | 91 django/utils/deprecation.py | 73 | 95| 158 | 55334 | 168533 | 
| 196 | 92 django/core/checks/caches.py | 1 | 17| 0 | 55334 | 168633 | 
| 197 | 92 django/utils/deprecation.py | 1 | 27| 181 | 55515 | 168633 | 
| 198 | 92 django/db/migrations/state.py | 229 | 242| 133 | 55648 | 168633 | 
| 199 | 92 django/db/utils.py | 1 | 49| 154 | 55802 | 168633 | 
| 200 | 92 django/core/management/commands/makemessages.py | 82 | 95| 118 | 55920 | 168633 | 
| 201 | 93 django/core/files/base.py | 75 | 118| 303 | 56223 | 169685 | 
| 202 | 94 django/contrib/messages/__init__.py | 1 | 5| 0 | 56223 | 169721 | 
| 203 | 94 django/db/migrations/operations/models.py | 102 | 118| 147 | 56370 | 169721 | 
| 204 | 94 django/db/migrations/autodetector.py | 796 | 845| 570 | 56940 | 169721 | 
| 205 | 95 django/core/checks/model_checks.py | 178 | 211| 332 | 57272 | 171508 | 
| 206 | 96 django/db/models/fields/files.py | 150 | 209| 645 | 57917 | 175230 | 
| 207 | 97 django/contrib/flatpages/__init__.py | 1 | 2| 0 | 57917 | 175244 | 
| 208 | 98 django/utils/archive.py | 185 | 227| 311 | 58228 | 176773 | 
| 209 | 99 django/contrib/staticfiles/finders.py | 70 | 93| 202 | 58430 | 178814 | 
| 210 | 99 django/core/management/commands/loaddata.py | 81 | 148| 593 | 59023 | 178814 | 
| 211 | 99 django/db/migrations/questioner.py | 207 | 224| 171 | 59194 | 178814 | 
| 212 | 100 django/db/models/options.py | 1 | 36| 304 | 59498 | 185913 | 
| 213 | 101 django/__main__.py | 1 | 10| 0 | 59498 | 185958 | 
| 214 | 102 django/conf/urls/__init__.py | 1 | 14| 0 | 59498 | 186053 | 
| 215 | 102 django/db/migrations/executor.py | 231 | 253| 194 | 59692 | 186053 | 
| 216 | 103 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 59896 | 186257 | 
| 217 | 104 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 59896 | 186354 | 
| 218 | 104 django/db/migrations/state.py | 79 | 104| 252 | 60148 | 186354 | 
| 219 | 104 django/db/migrations/autodetector.py | 673 | 705| 283 | 60431 | 186354 | 
| 220 | 104 django/db/models/fields/__init__.py | 337 | 361| 184 | 60615 | 186354 | 


## Patch

```diff
diff --git a/django/db/migrations/loader.py b/django/db/migrations/loader.py
--- a/django/db/migrations/loader.py
+++ b/django/db/migrations/loader.py
@@ -84,11 +84,6 @@ def load_disk(self):
                     continue
                 raise
             else:
-                # Empty directories are namespaces.
-                # getattr() needed on PY36 and older (replace w/attribute access).
-                if getattr(module, '__file__', None) is None:
-                    self.unmigrated_apps.add(app_config.label)
-                    continue
                 # Module is not a package (e.g. migrations.py).
                 if not hasattr(module, '__path__'):
                     self.unmigrated_apps.add(app_config.label)
@@ -96,11 +91,14 @@ def load_disk(self):
                 # Force a reload if it's already loaded (tests need this)
                 if was_loaded:
                     reload(module)
-            self.migrated_apps.add(app_config.label)
             migration_names = {
                 name for _, name, is_pkg in pkgutil.iter_modules(module.__path__)
                 if not is_pkg and name[0] not in '_~'
             }
+            if migration_names or self.ignore_no_migrations:
+                self.migrated_apps.add(app_config.label)
+            else:
+                self.unmigrated_apps.add(app_config.label)
             # Load migrations
             for migration_name in migration_names:
                 migration_path = '%s.%s' % (module_name, migration_name)

```

## Test Patch

```diff
diff --git a/tests/migrations/test_loader.py b/tests/migrations/test_loader.py
--- a/tests/migrations/test_loader.py
+++ b/tests/migrations/test_loader.py
@@ -508,6 +508,17 @@ def test_ignore_files(self):
         migrations = [name for app, name in loader.disk_migrations if app == 'migrations']
         self.assertEqual(migrations, ['0001_initial'])
 
+    @override_settings(
+        MIGRATION_MODULES={'migrations': 'migrations.test_migrations_namespace_package'},
+    )
+    def test_loading_namespace_package(self):
+        """Migration directories without an __init__.py file are loaded."""
+        migration_loader = MigrationLoader(connection)
+        self.assertEqual(
+            migration_loader.graph.forwards_plan(('migrations', '0001_initial')),
+            [('migrations', '0001_initial')],
+        )
+
 
 class PycLoaderTests(MigrationTestBase):
 
diff --git a/tests/migrations/test_migrations_namespace_package/0001_initial.py b/tests/migrations/test_migrations_namespace_package/0001_initial.py
new file mode 100644
--- /dev/null
+++ b/tests/migrations/test_migrations_namespace_package/0001_initial.py
@@ -0,0 +1,15 @@
+from django.db import migrations, models
+
+
+class Migration(migrations.Migration):
+    initial = True
+
+    operations = [
+        migrations.CreateModel(
+            "Author",
+            [
+                ("id", models.AutoField(primary_key=True)),
+                ("name", models.CharField(max_length=255)),
+            ],
+        ),
+    ]

```


## Code snippets

### 1 - django/db/migrations/loader.py:

Start line: 1, End line: 49

```python
import pkgutil
import sys
from importlib import import_module, reload

from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder

from .exceptions import (
    AmbiguityError, BadMigrationError, InconsistentMigrationHistory,
    NodeNotFoundError,
)

MIGRATIONS_MODULE_NAME = 'migrations'


class MigrationLoader:
    """
    Load migration files from disk and their status from the database.

    Migration files are expected to live in the "migrations" directory of
    an app. Their names are entirely unimportant from a code perspective,
    but will probably follow the 1234_name.py convention.

    On initialization, this class will scan those directories, and open and
    read the Python files, looking for a class called Migration, which should
    inherit from django.db.migrations.Migration. See
    django.db.migrations.migration for what that looks like.

    Some migrations will be marked as "replacing" another set of migrations.
    These are loaded into a separate set of migrations away from the main ones.
    If all the migrations they replace are either unapplied or missing from
    disk, then they are injected into the main set, replacing the named migrations.
    Any dependency pointers to the replaced migrations are re-pointed to the
    new migration.

    This does mean that this class MUST also talk to the database as well as
    to disk, but this is probably fine. We're already not just operating
    in memory.
    """

    def __init__(self, connection, load=True, ignore_no_migrations=False):
        self.connection = connection
        self.disk_migrations = None
        self.applied_migrations = None
        self.ignore_no_migrations = ignore_no_migrations
        if load:
            self.build_graph()
```
### 2 - django/db/migrations/__init__.py:

Start line: 1, End line: 3

```python

```
### 3 - django/db/migrations/loader.py:

Start line: 64, End line: 124

```python
class MigrationLoader:

    def load_disk(self):
        """Load the migrations from all INSTALLED_APPS from disk."""
        self.disk_migrations = {}
        self.unmigrated_apps = set()
        self.migrated_apps = set()
        for app_config in apps.get_app_configs():
            # Get the migrations module directory
            module_name, explicit = self.migrations_module(app_config.label)
            if module_name is None:
                self.unmigrated_apps.add(app_config.label)
                continue
            was_loaded = module_name in sys.modules
            try:
                module = import_module(module_name)
            except ImportError as e:
                # I hate doing this, but I don't want to squash other import errors.
                # Might be better to try a directory check directly.
                if ((explicit and self.ignore_no_migrations) or (
                        not explicit and "No module named" in str(e) and MIGRATIONS_MODULE_NAME in str(e))):
                    self.unmigrated_apps.add(app_config.label)
                    continue
                raise
            else:
                # Empty directories are namespaces.
                # getattr() needed on PY36 and older (replace w/attribute access).
                if getattr(module, '__file__', None) is None:
                    self.unmigrated_apps.add(app_config.label)
                    continue
                # Module is not a package (e.g. migrations.py).
                if not hasattr(module, '__path__'):
                    self.unmigrated_apps.add(app_config.label)
                    continue
                # Force a reload if it's already loaded (tests need this)
                if was_loaded:
                    reload(module)
            self.migrated_apps.add(app_config.label)
            migration_names = {
                name for _, name, is_pkg in pkgutil.iter_modules(module.__path__)
                if not is_pkg and name[0] not in '_~'
            }
            # Load migrations
            for migration_name in migration_names:
                migration_path = '%s.%s' % (module_name, migration_name)
                try:
                    migration_module = import_module(migration_path)
                except ImportError as e:
                    if 'bad magic number' in str(e):
                        raise ImportError(
                            "Couldn't import %r as it appears to be a stale "
                            ".pyc file." % migration_path
                        ) from e
                    else:
                        raise
                if not hasattr(migration_module, "Migration"):
                    raise BadMigrationError(
                        "Migration %s in app %s has no Migration class" % (migration_name, app_config.label)
                    )
                self.disk_migrations[app_config.label, migration_name] = migration_module.Migration(
                    migration_name,
                    app_config.label,
                )
```
### 4 - django/db/migrations/loader.py:

Start line: 148, End line: 174

```python
class MigrationLoader:

    def check_key(self, key, current_app):
        if (key[1] != "__first__" and key[1] != "__latest__") or key in self.graph:
            return key
        # Special-case __first__, which means "the first migration" for
        # migrated apps, and is ignored for unmigrated apps. It allows
        # makemigrations to declare dependencies on apps before they even have
        # migrations.
        if key[0] == current_app:
            # Ignore __first__ references to the same app (#22325)
            return
        if key[0] in self.unmigrated_apps:
            # This app isn't migrated, but something depends on it.
            # The models will get auto-added into the state, though
            # so we're fine.
            return
        if key[0] in self.migrated_apps:
            try:
                if key[1] == "__first__":
                    return self.graph.root_nodes(key[0])[0]
                else:  # "__latest__"
                    return self.graph.leaf_nodes(key[0])[0]
            except IndexError:
                if self.ignore_no_migrations:
                    return None
                else:
                    raise ValueError("Dependency on app with no migrations: %s" % key[0])
        raise ValueError("Dependency on unknown app: %s" % key[0])
```
### 5 - django/db/migrations/executor.py:

Start line: 298, End line: 377

```python
class MigrationExecutor:

    def detect_soft_applied(self, project_state, migration):
        """
        Test whether a migration has been implicitly applied - that the
        tables or columns it would create exist. This is intended only for use
        on initial migrations (as it only looks for CreateModel and AddField).
        """
        def should_skip_detecting_model(migration, model):
            """
            No need to detect tables for proxy models, unmanaged models, or
            models that can't be migrated on the current database.
            """
            return (
                model._meta.proxy or not model._meta.managed or not
                router.allow_migrate(
                    self.connection.alias, migration.app_label,
                    model_name=model._meta.model_name,
                )
            )

        if migration.initial is None:
            # Bail if the migration isn't the first one in its app
            if any(app == migration.app_label for app, name in migration.dependencies):
                return False, project_state
        elif migration.initial is False:
            # Bail if it's NOT an initial migration
            return False, project_state

        if project_state is None:
            after_state = self.loader.project_state((migration.app_label, migration.name), at_end=True)
        else:
            after_state = migration.mutate_state(project_state)
        apps = after_state.apps
        found_create_model_migration = False
        found_add_field_migration = False
        with self.connection.cursor() as cursor:
            existing_table_names = self.connection.introspection.table_names(cursor)
        # Make sure all create model and add field operations are done
        for operation in migration.operations:
            if isinstance(operation, migrations.CreateModel):
                model = apps.get_model(migration.app_label, operation.name)
                if model._meta.swapped:
                    # We have to fetch the model to test with from the
                    # main app cache, as it's not a direct dependency.
                    model = global_apps.get_model(model._meta.swapped)
                if should_skip_detecting_model(migration, model):
                    continue
                if model._meta.db_table not in existing_table_names:
                    return False, project_state
                found_create_model_migration = True
            elif isinstance(operation, migrations.AddField):
                model = apps.get_model(migration.app_label, operation.model_name)
                if model._meta.swapped:
                    # We have to fetch the model to test with from the
                    # main app cache, as it's not a direct dependency.
                    model = global_apps.get_model(model._meta.swapped)
                if should_skip_detecting_model(migration, model):
                    continue

                table = model._meta.db_table
                field = model._meta.get_field(operation.name)

                # Handle implicit many-to-many tables created by AddField.
                if field.many_to_many:
                    if field.remote_field.through._meta.db_table not in existing_table_names:
                        return False, project_state
                    else:
                        found_add_field_migration = True
                        continue

                column_names = [
                    column.name for column in
                    self.connection.introspection.get_table_description(self.connection.cursor(), table)
                ]
                if field.column not in column_names:
                    return False, project_state
                found_add_field_migration = True
        # If we get this far and we found at least one CreateModel or AddField migration,
        # the migration is considered implicitly applied.
        return (found_create_model_migration or found_add_field_migration), after_state
```
### 6 - django/core/management/commands/migrate.py:

Start line: 21, End line: 65

```python
class Command(BaseCommand):
    help = "Updates database schema. Manages both apps with migrations and those without."

    def add_arguments(self, parser):
        parser.add_argument(
            'app_label', nargs='?',
            help='App label of an application to synchronize the state.',
        )
        parser.add_argument(
            'migration_name', nargs='?',
            help='Database state will be brought to the state after that '
                 'migration. Use the name "zero" to unapply all migrations.',
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--database',
            default=DEFAULT_DB_ALIAS,
            help='Nominates a database to synchronize. Defaults to the "default" database.',
        )
        parser.add_argument(
            '--fake', action='store_true',
            help='Mark migrations as run without actually running them.',
        )
        parser.add_argument(
            '--fake-initial', action='store_true',
            help='Detect if tables already exist and fake-apply initial migrations if so. Make sure '
                 'that the current database schema matches your initial migration before using this '
                 'flag. Django will only check for an existing table name.',
        )
        parser.add_argument(
            '--plan', action='store_true',
            help='Shows a list of the migration actions that will be performed.',
        )
        parser.add_argument(
            '--run-syncdb', action='store_true',
            help='Creates tables for apps without migrations.',
        )

    def _run_checks(self, **kwargs):
        issues = run_checks(tags=[Tags.database])
        issues.extend(super()._run_checks(**kwargs))
        return issues
```
### 7 - django/db/migrations/autodetector.py:

Start line: 358, End line: 372

```python
class MigrationAutodetector:

    def _optimize_migrations(self):
        # Add in internal dependencies among the migrations
        for app_label, migrations in self.migrations.items():
            for m1, m2 in zip(migrations, migrations[1:]):
                m2.dependencies.append((app_label, m1.name))

        # De-dupe dependencies
        for migrations in self.migrations.values():
            for migration in migrations:
                migration.dependencies = list(set(migration.dependencies))

        # Optimize migrations
        for app_label, migrations in self.migrations.items():
            for migration in migrations:
                migration.operations = MigrationOptimizer().optimize(migration.operations, app_label=app_label)
```
### 8 - django/core/management/commands/makemigrations.py:

Start line: 60, End line: 146

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        self.verbosity = options['verbosity']
        self.interactive = options['interactive']
        self.dry_run = options['dry_run']
        self.merge = options['merge']
        self.empty = options['empty']
        self.migration_name = options['name']
        if self.migration_name and not self.migration_name.isidentifier():
            raise CommandError('The migration name must be a valid Python identifier.')
        self.include_header = options['include_header']
        check_changes = options['check_changes']

        # Make sure the app they asked for exists
        app_labels = set(app_labels)
        has_bad_labels = False
        for app_label in app_labels:
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                self.stderr.write(str(err))
                has_bad_labels = True
        if has_bad_labels:
            sys.exit(2)

        # Load the current graph state. Pass in None for the connection so
        # the loader doesn't try to resolve replaced migrations from DB.
        loader = MigrationLoader(None, ignore_no_migrations=True)

        # Raise an error if any migrations are applied before their dependencies.
        consistency_check_labels = {config.label for config in apps.get_app_configs()}
        # Non-default databases are only checked if database routers used.
        aliases_to_check = connections if settings.DATABASE_ROUTERS else [DEFAULT_DB_ALIAS]
        for alias in sorted(aliases_to_check):
            connection = connections[alias]
            if (connection.settings_dict['ENGINE'] != 'django.db.backends.dummy' and any(
                    # At least one model must be migrated to the database.
                    router.allow_migrate(connection.alias, app_label, model_name=model._meta.object_name)
                    for app_label in consistency_check_labels
                    for model in apps.get_app_config(app_label).get_models()
            )):
                loader.check_consistent_history(connection)

        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any and they don't want to merge
        conflicts = loader.detect_conflicts()

        # If app_labels is specified, filter out conflicting migrations for unspecified apps
        if app_labels:
            conflicts = {
                app_label: conflict for app_label, conflict in conflicts.items()
                if app_label in app_labels
            }

        if conflicts and not self.merge:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they want to merge and there's nothing to merge, then politely exit
        if self.merge and not conflicts:
            self.stdout.write("No conflicts detected to merge.")
            return

        # If they want to merge and there is something to merge, then
        # divert into the merge code
        if self.merge and conflicts:
            return self.handle_merge(loader, conflicts)

        if self.interactive:
            questioner = InteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        else:
            questioner = NonInteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        # Set up autodetector
        autodetector = MigrationAutodetector(
            loader.project_state(),
            ProjectState.from_apps(apps),
            questioner,
        )

        # If they want to make an empty migration, make one for each app
        # ... other code
```
### 9 - django/core/management/commands/migrate.py:

Start line: 67, End line: 160

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):

        self.verbosity = options['verbosity']
        self.interactive = options['interactive']

        # Import the 'management' module within each installed app, to register
        # dispatcher events.
        for app_config in apps.get_app_configs():
            if module_has_submodule(app_config.module, "management"):
                import_module('.management', app_config.name)

        # Get the database we're operating from
        db = options['database']
        connection = connections[db]

        # Hook for backends needing any database preparation
        connection.prepare_database()
        # Work out which apps have migrations and which do not
        executor = MigrationExecutor(connection, self.migration_progress_callback)

        # Raise an error if any migrations are applied before their dependencies.
        executor.loader.check_consistent_history(connection)

        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any
        conflicts = executor.loader.detect_conflicts()
        if conflicts:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they supplied command line arguments, work out what they mean.
        run_syncdb = options['run_syncdb']
        target_app_labels_only = True
        if options['app_label']:
            # Validate app_label.
            app_label = options['app_label']
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                raise CommandError(str(err))
            if run_syncdb:
                if app_label in executor.loader.migrated_apps:
                    raise CommandError("Can't use run_syncdb with app '%s' as it has migrations." % app_label)
            elif app_label not in executor.loader.migrated_apps:
                raise CommandError("App '%s' does not have migrations." % app_label)

        if options['app_label'] and options['migration_name']:
            migration_name = options['migration_name']
            if migration_name == "zero":
                targets = [(app_label, None)]
            else:
                try:
                    migration = executor.loader.get_migration_by_prefix(app_label, migration_name)
                except AmbiguityError:
                    raise CommandError(
                        "More than one migration matches '%s' in app '%s'. "
                        "Please be more specific." %
                        (migration_name, app_label)
                    )
                except KeyError:
                    raise CommandError("Cannot find a migration matching '%s' from app '%s'." % (
                        migration_name, app_label))
                targets = [(app_label, migration.name)]
            target_app_labels_only = False
        elif options['app_label']:
            targets = [key for key in executor.loader.graph.leaf_nodes() if key[0] == app_label]
        else:
            targets = executor.loader.graph.leaf_nodes()

        plan = executor.migration_plan(targets)

        if options['plan']:
            self.stdout.write('Planned operations:', self.style.MIGRATE_LABEL)
            if not plan:
                self.stdout.write('  No planned migration operations.')
            for migration, backwards in plan:
                self.stdout.write(str(migration), self.style.MIGRATE_HEADING)
                for operation in migration.operations:
                    message, is_error = self.describe_operation(operation, backwards)
                    style = self.style.WARNING if is_error else None
                    self.stdout.write('    ' + message, style)
            return

        # At this point, ignore run_syncdb if there aren't any apps to sync.
        run_syncdb = options['run_syncdb'] and executor.loader.unmigrated_apps
        # Print some useful info
        # ... other code
```
### 10 - django/db/migrations/writer.py:

Start line: 201, End line: 301

```python
class MigrationWriter:

    @property
    def basedir(self):
        migrations_package_name, _ = MigrationLoader.migrations_module(self.migration.app_label)

        if migrations_package_name is None:
            raise ValueError(
                "Django can't create migrations for app '%s' because "
                "migrations have been disabled via the MIGRATION_MODULES "
                "setting." % self.migration.app_label
            )

        # See if we can import the migrations module directly
        try:
            migrations_module = import_module(migrations_package_name)
        except ImportError:
            pass
        else:
            try:
                return module_dir(migrations_module)
            except ValueError:
                pass

        # Alright, see if it's a direct submodule of the app
        app_config = apps.get_app_config(self.migration.app_label)
        maybe_app_name, _, migrations_package_basename = migrations_package_name.rpartition(".")
        if app_config.name == maybe_app_name:
            return os.path.join(app_config.path, migrations_package_basename)

        # In case of using MIGRATION_MODULES setting and the custom package
        # doesn't exist, create one, starting from an existing package
        existing_dirs, missing_dirs = migrations_package_name.split("."), []
        while existing_dirs:
            missing_dirs.insert(0, existing_dirs.pop(-1))
            try:
                base_module = import_module(".".join(existing_dirs))
            except (ImportError, ValueError):
                continue
            else:
                try:
                    base_dir = module_dir(base_module)
                except ValueError:
                    continue
                else:
                    break
        else:
            raise ValueError(
                "Could not locate an appropriate location to create "
                "migrations package %s. Make sure the toplevel "
                "package exists and can be imported." %
                migrations_package_name)

        final_dir = os.path.join(base_dir, *missing_dirs)
        os.makedirs(final_dir, exist_ok=True)
        for missing_dir in missing_dirs:
            base_dir = os.path.join(base_dir, missing_dir)
            with open(os.path.join(base_dir, "__init__.py"), "w"):
                pass

        return final_dir

    @property
    def filename(self):
        return "%s.py" % self.migration.name

    @property
    def path(self):
        return os.path.join(self.basedir, self.filename)

    @classmethod
    def serialize(cls, value):
        return serializer_factory(value).serialize()

    @classmethod
    def register_serializer(cls, type_, serializer):
        Serializer.register(type_, serializer)

    @classmethod
    def unregister_serializer(cls, type_):
        Serializer.unregister(type_)


MIGRATION_HEADER_TEMPLATE = """\
# Generated by Django %(version)s on %(timestamp)s

"""


MIGRATION_TEMPLATE = """\
%(migration_header)s%(imports)s

class Migration(migrations.Migration):
%(replaces_str)s%(initial_str)s
    dependencies = [
%(dependencies)s\
    ]

    operations = [
%(operations)s\
    ]
"""
```
### 23 - django/db/migrations/loader.py:

Start line: 51, End line: 62

```python
class MigrationLoader:

    @classmethod
    def migrations_module(cls, app_label):
        """
        Return the path to the migrations module for the specified app_label
        and a boolean indicating if the module is specified in
        settings.MIGRATION_MODULE.
        """
        if app_label in settings.MIGRATION_MODULES:
            return settings.MIGRATION_MODULES[app_label], True
        else:
            app_package_name = apps.get_app_config(app_label).name
            return '%s.%s' % (app_package_name, MIGRATIONS_MODULE_NAME), False
```
### 30 - django/db/migrations/loader.py:

Start line: 277, End line: 301

```python
class MigrationLoader:

    def check_consistent_history(self, connection):
        """
        Raise InconsistentMigrationHistory if any applied migrations have
        unapplied dependencies.
        """
        recorder = MigrationRecorder(connection)
        applied = recorder.applied_migrations()
        for migration in applied:
            # If the migration is unknown, skip it.
            if migration not in self.graph.nodes:
                continue
            for parent in self.graph.node_map[migration].parents:
                if parent not in applied:
                    # Skip unapplied squashed migrations that have all of their
                    # `replaces` applied.
                    if parent in self.replacements:
                        if all(m in applied for m in self.replacements[parent].replaces):
                            continue
                    raise InconsistentMigrationHistory(
                        "Migration {}.{} is applied before its dependency "
                        "{}.{} on database '{}'.".format(
                            migration[0], migration[1], parent[0], parent[1],
                            connection.alias,
                        )
                    )
```
### 31 - django/db/migrations/loader.py:

Start line: 176, End line: 197

```python
class MigrationLoader:

    def add_internal_dependencies(self, key, migration):
        """
        Internal dependencies need to be added first to ensure `__first__`
        dependencies find the correct root node.
        """
        for parent in migration.dependencies:
            # Ignore __first__ references to the same app.
            if parent[0] == key[0] and parent[1] != '__first__':
                self.graph.add_dependency(migration, key, parent, skip_validation=True)

    def add_external_dependencies(self, key, migration):
        for parent in migration.dependencies:
            # Skip internal dependencies
            if key[0] == parent[0]:
                continue
            parent = self.check_key(parent, key[0])
            if parent is not None:
                self.graph.add_dependency(migration, key, parent, skip_validation=True)
        for child in migration.run_before:
            child = self.check_key(child, key[0])
            if child is not None:
                self.graph.add_dependency(migration, child, key, skip_validation=True)
```
### 42 - django/db/migrations/loader.py:

Start line: 199, End line: 275

```python
class MigrationLoader:

    def build_graph(self):
        """
        Build a migration dependency graph using both the disk and database.
        You'll need to rebuild the graph if you apply migrations. This isn't
        usually a problem as generally migration stuff runs in a one-shot process.
        """
        # Load disk data
        self.load_disk()
        # Load database data
        if self.connection is None:
            self.applied_migrations = {}
        else:
            recorder = MigrationRecorder(self.connection)
            self.applied_migrations = recorder.applied_migrations()
        # To start, populate the migration graph with nodes for ALL migrations
        # and their dependencies. Also make note of replacing migrations at this step.
        self.graph = MigrationGraph()
        self.replacements = {}
        for key, migration in self.disk_migrations.items():
            self.graph.add_node(key, migration)
            # Replacing migrations.
            if migration.replaces:
                self.replacements[key] = migration
        for key, migration in self.disk_migrations.items():
            # Internal (same app) dependencies.
            self.add_internal_dependencies(key, migration)
        # Add external dependencies now that the internal ones have been resolved.
        for key, migration in self.disk_migrations.items():
            self.add_external_dependencies(key, migration)
        # Carry out replacements where possible.
        for key, migration in self.replacements.items():
            # Get applied status of each of this migration's replacement targets.
            applied_statuses = [(target in self.applied_migrations) for target in migration.replaces]
            # Ensure the replacing migration is only marked as applied if all of
            # its replacement targets are.
            if all(applied_statuses):
                self.applied_migrations[key] = migration
            else:
                self.applied_migrations.pop(key, None)
            # A replacing migration can be used if either all or none of its
            # replacement targets have been applied.
            if all(applied_statuses) or (not any(applied_statuses)):
                self.graph.remove_replaced_nodes(key, migration.replaces)
            else:
                # This replacing migration cannot be used because it is partially applied.
                # Remove it from the graph and remap dependencies to it (#25945).
                self.graph.remove_replacement_node(key, migration.replaces)
        # Ensure the graph is consistent.
        try:
            self.graph.validate_consistency()
        except NodeNotFoundError as exc:
            # Check if the missing node could have been replaced by any squash
            # migration but wasn't because the squash migration was partially
            # applied before. In that case raise a more understandable exception
            # (#23556).
            # Get reverse replacements.
            reverse_replacements = {}
            for key, migration in self.replacements.items():
                for replaced in migration.replaces:
                    reverse_replacements.setdefault(replaced, set()).add(key)
            # Try to reraise exception with more detail.
            if exc.node in reverse_replacements:
                candidates = reverse_replacements.get(exc.node, set())
                is_replaced = any(candidate in self.graph.nodes for candidate in candidates)
                if not is_replaced:
                    tries = ', '.join('%s.%s' % c for c in candidates)
                    raise NodeNotFoundError(
                        "Migration {0} depends on nonexistent node ('{1}', '{2}'). "
                        "Django tried to replace migration {1}.{2} with any of [{3}] "
                        "but wasn't able to because some of the replaced migrations "
                        "are already applied.".format(
                            exc.origin, exc.node[0], exc.node[1], tries
                        ),
                        exc.node
                    ) from exc
            raise exc
        self.graph.ensure_not_cyclic()
```
### 50 - django/db/migrations/loader.py:

Start line: 126, End line: 146

```python
class MigrationLoader:

    def get_migration(self, app_label, name_prefix):
        """Return the named migration or raise NodeNotFoundError."""
        return self.graph.nodes[app_label, name_prefix]

    def get_migration_by_prefix(self, app_label, name_prefix):
        """
        Return the migration(s) which match the given app label and name_prefix.
        """
        # Do the search
        results = []
        for migration_app_label, migration_name in self.disk_migrations:
            if migration_app_label == app_label and migration_name.startswith(name_prefix):
                results.append((migration_app_label, migration_name))
        if len(results) > 1:
            raise AmbiguityError(
                "There is more than one migration for '%s' with the prefix '%s'" % (app_label, name_prefix)
            )
        elif not results:
            raise KeyError("There no migrations for '%s' with the prefix '%s'" % (app_label, name_prefix))
        else:
            return self.disk_migrations[results[0]]
```
### 115 - django/db/migrations/loader.py:

Start line: 303, End line: 325

```python
class MigrationLoader:

    def detect_conflicts(self):
        """
        Look through the loaded graph and detect any conflicts - apps
        with more than one leaf migration. Return a dict of the app labels
        that conflict with the migration names that conflict.
        """
        seen_apps = {}
        conflicting_apps = set()
        for app_label, migration_name in self.graph.leaf_nodes():
            if app_label in seen_apps:
                conflicting_apps.add(app_label)
            seen_apps.setdefault(app_label, set()).add(migration_name)
        return {app_label: seen_apps[app_label] for app_label in conflicting_apps}

    def project_state(self, nodes=None, at_end=True):
        """
        Return a ProjectState object representing the most recent state
        that the loaded migrations represent.

        See graph.make_state() for the meaning of "nodes" and "at_end".
        """
        return self.graph.make_state(nodes=nodes, at_end=at_end, real_apps=list(self.unmigrated_apps))
```
