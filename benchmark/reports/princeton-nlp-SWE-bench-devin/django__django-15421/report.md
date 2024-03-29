# django__django-15421

| **django/django** | `be80aa55ec120b3b6645b3efb77316704d7ad948` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 3834 |
| **Avg pos** | 35.0 |
| **Min pos** | 14 |
| **Max pos** | 56 |
| **Top file pos** | 9 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/sqlite3/creation.py b/django/db/backends/sqlite3/creation.py
--- a/django/db/backends/sqlite3/creation.py
+++ b/django/db/backends/sqlite3/creation.py
@@ -1,8 +1,11 @@
+import multiprocessing
 import os
 import shutil
+import sqlite3
 import sys
 from pathlib import Path
 
+from django.db import NotSupportedError
 from django.db.backends.base.creation import BaseDatabaseCreation
 
 
@@ -51,16 +54,26 @@ def _create_test_db(self, verbosity, autoclobber, keepdb=False):
     def get_test_db_clone_settings(self, suffix):
         orig_settings_dict = self.connection.settings_dict
         source_database_name = orig_settings_dict["NAME"]
-        if self.is_in_memory_db(source_database_name):
+
+        if not self.is_in_memory_db(source_database_name):
+            root, ext = os.path.splitext(source_database_name)
+            return {**orig_settings_dict, "NAME": f"{root}_{suffix}{ext}"}
+
+        start_method = multiprocessing.get_start_method()
+        if start_method == "fork":
             return orig_settings_dict
-        else:
-            root, ext = os.path.splitext(orig_settings_dict["NAME"])
-            return {**orig_settings_dict, "NAME": "{}_{}{}".format(root, suffix, ext)}
+        if start_method == "spawn":
+            return {
+                **orig_settings_dict,
+                "NAME": f"{self.connection.alias}_{suffix}.sqlite3",
+            }
+        raise NotSupportedError(
+            f"Cloning with start method {start_method!r} is not supported."
+        )
 
     def _clone_test_db(self, suffix, verbosity, keepdb=False):
         source_database_name = self.connection.settings_dict["NAME"]
         target_database_name = self.get_test_db_clone_settings(suffix)["NAME"]
-        # Forking automatically makes a copy of an in-memory database.
         if not self.is_in_memory_db(source_database_name):
             # Erase the old test database
             if os.access(target_database_name, os.F_OK):
@@ -85,6 +98,12 @@ def _clone_test_db(self, suffix, verbosity, keepdb=False):
             except Exception as e:
                 self.log("Got an error cloning the test database: %s" % e)
                 sys.exit(2)
+        # Forking automatically makes a copy of an in-memory database.
+        # Spawn requires migrating to disk which will be re-opened in
+        # setup_worker_connection.
+        elif multiprocessing.get_start_method() == "spawn":
+            ondisk_db = sqlite3.connect(target_database_name, uri=True)
+            self.connection.connection.backup(ondisk_db)
 
     def _destroy_test_db(self, test_database_name, verbosity):
         if test_database_name and not self.is_in_memory_db(test_database_name):
@@ -106,3 +125,34 @@ def test_db_signature(self):
         else:
             sig.append(test_database_name)
         return tuple(sig)
+
+    def setup_worker_connection(self, _worker_id):
+        settings_dict = self.get_test_db_clone_settings(_worker_id)
+        # connection.settings_dict must be updated in place for changes to be
+        # reflected in django.db.connections. Otherwise new threads would
+        # connect to the default database instead of the appropriate clone.
+        start_method = multiprocessing.get_start_method()
+        if start_method == "fork":
+            # Update settings_dict in place.
+            self.connection.settings_dict.update(settings_dict)
+            self.connection.close()
+        elif start_method == "spawn":
+            alias = self.connection.alias
+            connection_str = (
+                f"file:memorydb_{alias}_{_worker_id}?mode=memory&cache=shared"
+            )
+            source_db = self.connection.Database.connect(
+                f"file:{alias}_{_worker_id}.sqlite3", uri=True
+            )
+            target_db = sqlite3.connect(connection_str, uri=True)
+            source_db.backup(target_db)
+            source_db.close()
+            # Update settings_dict in place.
+            self.connection.settings_dict.update(settings_dict)
+            self.connection.settings_dict["NAME"] = connection_str
+            # Re-open connection to in-memory database before closing copy
+            # connection.
+            self.connection.connect()
+            target_db.close()
+            if os.environ.get("RUNNING_DJANGOS_TEST_SUITE") == "true":
+                self.mark_expected_failures_and_skips()
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -130,7 +130,7 @@ def iter_modules_and_files(modules, extra_files):
         # cause issues here.
         if not isinstance(module, ModuleType):
             continue
-        if module.__name__ == "__main__":
+        if module.__name__ in ("__main__", "__mp_main__"):
             # __main__ (usually manage.py) doesn't always have a __spec__ set.
             # Handle this by falling back to using __file__, resolved below.
             # See https://docs.python.org/reference/import.html#main-spec

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/sqlite3/creation.py | 1 | 1 | 56 | 10 | 16484
| django/db/backends/sqlite3/creation.py | 54 | 63 | 14 | 10 | 3834
| django/db/backends/sqlite3/creation.py | 88 | 88 | - | 10 | -
| django/db/backends/sqlite3/creation.py | 109 | 109 | - | 10 | -
| django/utils/autoreload.py | 133 | 133 | - | 9 | -


## Problem Statement

```
Allow parallel test runner to work with Windows/macOS `spawn` process start method.
Description
	 
		(last modified by Brandon Navra)
	 
Python 3.8 on MacOS has changed the default start method for the multiprocessing module from fork to spawn: ​https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods.
When running tests with the --parallel flag, this causes the worker processes to fail with django.core.exceptions.AppRegistryNotReady: Apps aren't loaded yet. as they no longer have a copy of the parent memory state. It can also cause the workers to fail to find the cloned dbs ( {{django.db.utils.OperationalError: FATAL: database "xxx_1" does not exist}} ) as the db test prefix is missing.
I have attached a patch which changes django.test.runner._init_worker (the worker initialiser for ParallelTestSuite) to run django.setup() and set the db name to one with the test_ prefix.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/management/commands/test.py | 1 | 24| 170 | 170 | 500 | 
| 2 | 1 django/core/management/commands/test.py | 59 | 72| 121 | 291 | 500 | 
| 3 | 2 django/db/backends/base/creation.py | 327 | 351| 282 | 573 | 3441 | 
| 4 | 3 django/contrib/admin/tests.py | 40 | 48| 138 | 711 | 5083 | 
| 5 | 4 django/db/backends/base/base.py | 1 | 36| 214 | 925 | 10514 | 
| 6 | 5 django/core/management/commands/testserver.py | 38 | 66| 237 | 1162 | 10960 | 
| 7 | 5 django/db/backends/base/creation.py | 1 | 104| 754 | 1916 | 10960 | 
| 8 | 6 django/db/backends/oracle/creation.py | 29 | 124| 768 | 2684 | 14975 | 
| 9 | 6 django/db/backends/base/creation.py | 235 | 255| 159 | 2843 | 14975 | 
| 10 | 7 django/core/management/commands/startapp.py | 1 | 15| 0 | 2843 | 15076 | 
| 11 | 8 django/db/backends/mysql/creation.py | 1 | 29| 221 | 3064 | 15745 | 
| 12 | 8 django/db/backends/oracle/creation.py | 126 | 157| 324 | 3388 | 15745 | 
| 13 | **9 django/utils/autoreload.py** | 672 | 686| 121 | 3509 | 20957 | 
| **-> 14 <-** | **10 django/db/backends/sqlite3/creation.py** | 51 | 87| 325 | 3834 | 21819 | 
| 15 | **10 django/utils/autoreload.py** | 329 | 344| 146 | 3980 | 21819 | 
| 16 | 10 django/db/backends/base/creation.py | 353 | 381| 216 | 4196 | 21819 | 
| 17 | 11 django/db/backends/postgresql/creation.py | 58 | 88| 255 | 4451 | 22502 | 
| 18 | 11 django/db/backends/base/creation.py | 257 | 277| 179 | 4630 | 22502 | 
| 19 | 11 django/db/backends/mysql/creation.py | 31 | 60| 261 | 4891 | 22502 | 
| 20 | 12 django/db/backends/base/features.py | 220 | 354| 1162 | 6053 | 25556 | 
| 21 | **12 django/utils/autoreload.py** | 1 | 56| 287 | 6340 | 25556 | 
| 22 | 12 django/core/management/commands/test.py | 26 | 57| 219 | 6559 | 25556 | 
| 23 | 13 django/core/management/commands/runserver.py | 80 | 120| 401 | 6960 | 27072 | 
| 24 | 14 django/apps/config.py | 262 | 275| 117 | 7077 | 29301 | 
| 25 | 15 django/db/migrations/operations/special.py | 182 | 209| 254 | 7331 | 30874 | 
| 26 | 15 django/db/backends/base/features.py | 6 | 219| 1745 | 9076 | 30874 | 
| 27 | 15 django/db/backends/postgresql/creation.py | 41 | 56| 174 | 9250 | 30874 | 
| 28 | 15 django/db/backends/base/base.py | 617 | 649| 230 | 9480 | 30874 | 
| 29 | 15 django/core/management/commands/runserver.py | 122 | 184| 522 | 10002 | 30874 | 
| 30 | 16 django/core/management/utils.py | 1 | 29| 173 | 10175 | 32052 | 
| 31 | 17 django/core/files/temp.py | 1 | 80| 520 | 10695 | 32573 | 
| 32 | 17 django/db/backends/oracle/creation.py | 1 | 27| 225 | 10920 | 32573 | 
| 33 | 17 django/core/management/commands/testserver.py | 1 | 36| 214 | 11134 | 32573 | 
| 34 | 17 django/contrib/admin/tests.py | 1 | 38| 265 | 11399 | 32573 | 
| 35 | 17 django/db/backends/base/creation.py | 188 | 233| 375 | 11774 | 32573 | 
| 36 | **17 django/utils/autoreload.py** | 221 | 262| 429 | 12203 | 32573 | 
| 37 | 17 django/db/backends/base/base.py | 700 | 714| 144 | 12347 | 32573 | 
| 38 | 18 django/core/management/base.py | 385 | 419| 297 | 12644 | 37356 | 
| 39 | 18 django/db/backends/oracle/creation.py | 262 | 300| 403 | 13047 | 37356 | 
| 40 | **18 django/utils/autoreload.py** | 311 | 327| 162 | 13209 | 37356 | 
| 41 | 18 django/db/backends/oracle/creation.py | 159 | 201| 411 | 13620 | 37356 | 
| 42 | 19 django/db/migrations/operations/__init__.py | 1 | 41| 218 | 13838 | 37574 | 
| 43 | 20 django/db/backends/postgresql/base.py | 271 | 297| 228 | 14066 | 40583 | 
| 44 | 20 django/db/backends/base/creation.py | 106 | 142| 287 | 14353 | 40583 | 
| 45 | 21 django/__init__.py | 1 | 25| 173 | 14526 | 40756 | 
| 46 | 22 docs/_ext/djangodocs.py | 26 | 71| 398 | 14924 | 43980 | 
| 47 | **22 django/utils/autoreload.py** | 431 | 464| 354 | 15278 | 43980 | 
| 48 | 23 django/db/backends/base/client.py | 1 | 29| 197 | 15475 | 44177 | 
| 49 | 23 django/db/backends/oracle/creation.py | 363 | 378| 193 | 15668 | 44177 | 
| 50 | 24 django/core/checks/async_checks.py | 1 | 17| 0 | 15668 | 44271 | 
| 51 | 25 django/db/utils.py | 185 | 218| 224 | 15892 | 46291 | 
| 52 | 26 django/core/mail/backends/__init__.py | 1 | 2| 0 | 15892 | 46299 | 
| 53 | 27 django/contrib/postgres/apps.py | 47 | 78| 275 | 16167 | 46954 | 
| 54 | 28 django/dispatch/__init__.py | 1 | 10| 0 | 16167 | 47019 | 
| 55 | 28 django/contrib/admin/tests.py | 185 | 203| 177 | 16344 | 47019 | 
| **-> 56 <-** | **28 django/db/backends/sqlite3/creation.py** | 1 | 20| 140 | 16484 | 47019 | 
| 57 | 28 django/db/backends/base/base.py | 651 | 698| 300 | 16784 | 47019 | 
| 58 | 29 django/core/mail/backends/locmem.py | 1 | 32| 183 | 16967 | 47203 | 
| 59 | 29 django/core/management/commands/runserver.py | 25 | 66| 284 | 17251 | 47203 | 
| 60 | 30 django/db/backends/sqlite3/base.py | 162 | 219| 511 | 17762 | 50234 | 
| 61 | 30 django/db/backends/oracle/creation.py | 227 | 260| 323 | 18085 | 50234 | 
| 62 | 30 django/db/backends/oracle/creation.py | 313 | 334| 180 | 18265 | 50234 | 
| 63 | 30 django/db/migrations/operations/special.py | 136 | 180| 302 | 18567 | 50234 | 
| 64 | 31 django/db/migrations/executor.py | 307 | 411| 862 | 19429 | 53675 | 
| 65 | **31 django/db/backends/sqlite3/creation.py** | 22 | 49| 242 | 19671 | 53675 | 
| 66 | 32 django/conf/global_settings.py | 512 | 643| 807 | 20478 | 59531 | 
| 67 | 33 django/db/migrations/__init__.py | 1 | 3| 0 | 20478 | 59555 | 
| 68 | 34 django/db/backends/postgresql/client.py | 1 | 65| 467 | 20945 | 60022 | 
| 69 | 35 django/db/backends/utils.py | 48 | 64| 176 | 21121 | 62057 | 
| 70 | 36 django/apps/registry.py | 1 | 59| 475 | 21596 | 65485 | 
| 71 | 36 django/db/backends/base/base.py | 741 | 771| 187 | 21783 | 65485 | 
| 72 | 37 django/core/management/commands/migrate.py | 93 | 186| 765 | 22548 | 69392 | 
| 73 | 37 django/db/backends/mysql/creation.py | 62 | 88| 200 | 22748 | 69392 | 
| 74 | 37 django/db/migrations/executor.py | 236 | 261| 206 | 22954 | 69392 | 
| 75 | 37 django/db/backends/oracle/creation.py | 380 | 465| 741 | 23695 | 69392 | 
| 76 | 38 django/db/migrations/questioner.py | 1 | 55| 469 | 24164 | 72088 | 
| 77 | 38 django/db/backends/postgresql/base.py | 322 | 377| 376 | 24540 | 72088 | 
| 78 | 39 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 24540 | 72181 | 
| 79 | 39 django/contrib/postgres/apps.py | 1 | 18| 160 | 24700 | 72181 | 
| 80 | 40 django/db/backends/oracle/base.py | 1 | 37| 247 | 24947 | 77330 | 
| 81 | 41 django/contrib/postgres/functions.py | 1 | 12| 0 | 24947 | 77382 | 
| 82 | 41 django/core/management/commands/migrate.py | 263 | 361| 813 | 25760 | 77382 | 
| 83 | 42 django/db/backends/postgresql/features.py | 1 | 109| 872 | 26632 | 78254 | 
| 84 | 43 django/urls/base.py | 91 | 157| 383 | 27015 | 79453 | 
| 85 | 44 django/db/models/options.py | 1 | 54| 319 | 27334 | 86955 | 
| 86 | 45 django/core/management/commands/dbshell.py | 27 | 49| 176 | 27510 | 87271 | 
| 87 | 45 django/db/utils.py | 138 | 161| 227 | 27737 | 87271 | 
| 88 | 45 django/db/backends/postgresql/base.py | 234 | 269| 271 | 28008 | 87271 | 
| 89 | 46 django/db/backends/dummy/features.py | 1 | 7| 0 | 28008 | 87303 | 
| 90 | **46 django/utils/autoreload.py** | 346 | 381| 259 | 28267 | 87303 | 
| 91 | 47 django/core/mail/backends/dummy.py | 1 | 11| 0 | 28267 | 87346 | 
| 92 | 48 django/db/backends/sqlite3/features.py | 1 | 57| 581 | 28848 | 88542 | 
| 93 | **48 django/utils/autoreload.py** | 538 | 560| 272 | 29120 | 88542 | 
| 94 | 49 django/db/models/sql/__init__.py | 1 | 7| 0 | 29120 | 88608 | 
| 95 | 50 django/core/management/commands/loaddata.py | 91 | 109| 191 | 29311 | 91787 | 
| 96 | **50 django/utils/autoreload.py** | 640 | 669| 226 | 29537 | 91787 | 
| 97 | 50 django/core/management/commands/dbshell.py | 1 | 25| 145 | 29682 | 91787 | 
| 98 | **50 django/utils/autoreload.py** | 265 | 309| 298 | 29980 | 91787 | 
| 99 | 50 django/contrib/admin/tests.py | 50 | 131| 580 | 30560 | 91787 | 
| 100 | 51 django/core/servers/basehttp.py | 230 | 247| 210 | 30770 | 93696 | 
| 101 | 51 django/db/backends/utils.py | 66 | 94| 255 | 31025 | 93696 | 
| 102 | 51 django/db/backends/oracle/creation.py | 302 | 311| 114 | 31139 | 93696 | 
| 103 | 51 django/db/backends/postgresql/base.py | 1 | 64| 462 | 31601 | 93696 | 
| 104 | **51 django/utils/autoreload.py** | 585 | 608| 177 | 31778 | 93696 | 
| 105 | 51 django/db/backends/sqlite3/features.py | 59 | 106| 383 | 32161 | 93696 | 
| 106 | 52 django/contrib/humanize/apps.py | 1 | 8| 0 | 32161 | 93737 | 
| 107 | 53 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 32556 | 94403 | 
| 108 | 54 django/apps/__init__.py | 1 | 5| 0 | 32556 | 94426 | 
| 109 | 54 django/core/management/commands/migrate.py | 425 | 480| 409 | 32965 | 94426 | 
| 110 | 54 django/db/backends/base/base.py | 270 | 344| 505 | 33470 | 94426 | 
| 111 | 55 django/core/management/commands/sendtestemail.py | 1 | 30| 196 | 33666 | 94737 | 
| 112 | 55 django/db/backends/oracle/base.py | 260 | 301| 439 | 34105 | 94737 | 
| 113 | 56 django/core/management/commands/shell.py | 1 | 55| 291 | 34396 | 95636 | 
| 114 | 56 django/db/backends/utils.py | 97 | 144| 327 | 34723 | 95636 | 
| 115 | 57 django/contrib/postgres/serializers.py | 1 | 11| 0 | 34723 | 95737 | 
| 116 | 57 django/db/migrations/executor.py | 263 | 288| 228 | 34951 | 95737 | 
| 117 | 58 django/template/loaders/app_directories.py | 1 | 14| 0 | 34951 | 95796 | 
| 118 | 58 django/db/models/options.py | 359 | 377| 124 | 35075 | 95796 | 
| 119 | 59 django/core/checks/database.py | 1 | 15| 0 | 35075 | 95865 | 
| 120 | 59 django/core/management/commands/migrate.py | 1 | 14| 134 | 35209 | 95865 | 
| 121 | 60 django/__main__.py | 1 | 10| 0 | 35209 | 95910 | 
| 122 | 60 django/db/backends/base/base.py | 493 | 588| 604 | 35813 | 95910 | 
| 123 | 61 django/db/models/base.py | 1 | 63| 336 | 36149 | 113857 | 
| 124 | 62 django/db/backends/mysql/features.py | 76 | 165| 701 | 36850 | 116214 | 
| 125 | 62 django/core/management/commands/shell.py | 57 | 112| 442 | 37292 | 116214 | 
| 126 | **62 django/utils/autoreload.py** | 384 | 428| 273 | 37565 | 116214 | 
| 127 | 63 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 37565 | 116282 | 
| 128 | 63 django/core/management/commands/runserver.py | 1 | 22| 204 | 37769 | 116282 | 
| 129 | 64 django/db/backends/dummy/base.py | 51 | 75| 173 | 37942 | 116723 | 
| 130 | 65 django/contrib/postgres/forms/__init__.py | 1 | 4| 0 | 37942 | 116754 | 
| 131 | 66 django/template/autoreload.py | 33 | 55| 149 | 38091 | 117107 | 
| 132 | 67 django/db/backends/oracle/features.py | 1 | 75| 644 | 38735 | 118309 | 
| 133 | 68 django/core/management/commands/makemigrations.py | 1 | 21| 163 | 38898 | 121424 | 
| 134 | 68 django/db/backends/postgresql/creation.py | 1 | 39| 266 | 39164 | 121424 | 
| 135 | 69 django/db/backends/mysql/base.py | 397 | 421| 211 | 39375 | 124936 | 
| 136 | 70 django/db/backends/mysql/client.py | 1 | 61| 560 | 39935 | 125497 | 
| 137 | 71 django/core/mail/backends/console.py | 1 | 45| 285 | 40220 | 125783 | 
| 138 | 72 docs/conf.py | 54 | 127| 672 | 40892 | 129278 | 
| 139 | 73 django/contrib/auth/migrations/0012_alter_user_first_name_max_length.py | 1 | 19| 0 | 40892 | 129357 | 
| 140 | 73 django/core/management/commands/runserver.py | 68 | 78| 120 | 41012 | 129357 | 
| 141 | 74 django/db/__init__.py | 1 | 62| 291 | 41303 | 129648 | 
| 142 | **74 django/utils/autoreload.py** | 466 | 493| 228 | 41531 | 129648 | 
| 143 | 74 django/db/migrations/executor.py | 1 | 71| 571 | 42102 | 129648 | 
| 144 | 74 django/db/backends/base/creation.py | 279 | 312| 241 | 42343 | 129648 | 
| 145 | 75 django/db/models/sql/compiler.py | 65 | 77| 155 | 42498 | 144861 | 
| 146 | 75 django/db/backends/dummy/base.py | 1 | 48| 266 | 42764 | 144861 | 
| 147 | 75 django/conf/global_settings.py | 157 | 272| 859 | 43623 | 144861 | 
| 148 | 76 django/db/migrations/autodetector.py | 476 | 506| 267 | 43890 | 157324 | 
| 149 | 77 django/contrib/auth/management/commands/createsuperuser.py | 89 | 247| 1288 | 45178 | 159540 | 
| 150 | 77 django/db/backends/oracle/base.py | 64 | 102| 325 | 45503 | 159540 | 
| 151 | 77 django/db/backends/base/creation.py | 314 | 325| 117 | 45620 | 159540 | 
| 152 | 77 django/contrib/postgres/apps.py | 21 | 44| 220 | 45840 | 159540 | 
| 153 | 78 django/contrib/sessions/apps.py | 1 | 8| 0 | 45840 | 159577 | 
| 154 | 79 django/db/migrations/loader.py | 169 | 197| 295 | 46135 | 162730 | 
| 155 | 80 django/contrib/gis/db/backends/mysql/base.py | 1 | 15| 0 | 46135 | 162836 | 
| 156 | 80 django/db/backends/utils.py | 1 | 46| 287 | 46422 | 162836 | 
| 157 | 80 django/db/backends/oracle/base.py | 303 | 351| 334 | 46756 | 162836 | 
| 158 | 80 django/core/management/commands/shell.py | 114 | 140| 176 | 46932 | 162836 | 
| 159 | 81 django/contrib/sites/checks.py | 1 | 13| 0 | 46932 | 162913 | 
| 160 | 81 django/core/management/commands/migrate.py | 188 | 262| 672 | 47604 | 162913 | 
| 161 | 81 django/db/utils.py | 1 | 50| 177 | 47781 | 162913 | 
| 162 | 82 django/db/backends/signals.py | 1 | 4| 0 | 47781 | 162924 | 
| 163 | **82 django/utils/autoreload.py** | 495 | 510| 150 | 47931 | 162924 | 
| 164 | 82 django/urls/base.py | 1 | 24| 170 | 48101 | 162924 | 
| 165 | 82 django/db/backends/oracle/base.py | 40 | 64| 214 | 48315 | 162924 | 
| 166 | 83 django/contrib/auth/management/__init__.py | 37 | 106| 495 | 48810 | 164066 | 
| 167 | 84 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 48810 | 164119 | 
| 168 | 85 scripts/manage_translations.py | 200 | 220| 130 | 48940 | 165817 | 
| 169 | 86 django/contrib/gis/db/models/sql/__init__.py | 1 | 7| 0 | 48940 | 165850 | 
| 170 | 87 django/db/backends/sqlite3/client.py | 1 | 11| 0 | 48940 | 165917 | 
| 171 | 87 django/db/backends/oracle/base.py | 235 | 258| 181 | 49121 | 165917 | 
| 172 | 87 django/core/management/commands/loaddata.py | 1 | 41| 181 | 49302 | 165917 | 
| 173 | 88 django/db/backends/sqlite3/_functions.py | 40 | 84| 671 | 49973 | 169773 | 
| 174 | 88 django/db/migrations/autodetector.py | 1 | 17| 110 | 50083 | 169773 | 
| 175 | 88 django/db/backends/sqlite3/base.py | 133 | 160| 272 | 50355 | 169773 | 
| 176 | 89 django/db/migrations/writer.py | 206 | 312| 632 | 50987 | 172044 | 
| 177 | 90 django/contrib/auth/__init__.py | 1 | 38| 240 | 51227 | 173648 | 
| 178 | **90 django/utils/autoreload.py** | 562 | 583| 206 | 51433 | 173648 | 
| 179 | 90 django/db/backends/mysql/base.py | 177 | 204| 212 | 51645 | 173648 | 
| 180 | 91 django/utils/version.py | 1 | 18| 158 | 51803 | 174543 | 
| 181 | 92 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 51803 | 174620 | 
| 182 | 92 django/core/management/commands/makemigrations.py | 90 | 179| 783 | 52586 | 174620 | 
| 183 | 92 django/db/migrations/loader.py | 73 | 139| 553 | 53139 | 174620 | 
| 184 | 92 django/conf/global_settings.py | 415 | 511| 788 | 53927 | 174620 | 
| 185 | 93 django/db/models/query.py | 2320 | 2388| 785 | 54712 | 193583 | 
| 186 | **93 django/utils/autoreload.py** | 512 | 536| 233 | 54945 | 193583 | 
| 187 | 94 django/contrib/gis/db/backends/spatialite/base.py | 40 | 80| 321 | 55266 | 194231 | 
| 188 | 95 django/contrib/admindocs/apps.py | 1 | 8| 0 | 55266 | 194273 | 
| 189 | 96 django/utils/timezone.py | 136 | 163| 188 | 55454 | 196504 | 
| 190 | 96 django/db/backends/postgresql/base.py | 208 | 232| 257 | 55711 | 196504 | 
| 191 | 97 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 19| 0 | 55711 | 196586 | 
| 192 | 97 django/core/management/commands/loaddata.py | 44 | 89| 286 | 55997 | 196586 | 
| 193 | 97 django/core/management/commands/loaddata.py | 281 | 312| 305 | 56302 | 196586 | 
| 194 | 97 django/db/backends/sqlite3/base.py | 221 | 340| 887 | 57189 | 196586 | 
| 195 | 97 django/db/backends/mysql/base.py | 256 | 274| 153 | 57342 | 196586 | 
| 196 | 98 django/core/management/commands/compilemessages.py | 1 | 27| 162 | 57504 | 197945 | 
| 197 | 98 django/db/backends/base/base.py | 243 | 268| 247 | 57751 | 197945 | 
| 198 | 98 django/db/backends/mysql/base.py | 276 | 312| 259 | 58010 | 197945 | 
| 199 | 98 django/db/migrations/operations/special.py | 119 | 133| 139 | 58149 | 197945 | 
| 200 | 99 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 58149 | 197965 | 


### Hint

```
spawn() is also a default method on Windows, and we don't encounter any issues with it 🤔.
I'm still trying to research the exact root cause. The Python issue which triggered this change has snippets of info: https://code.djangoproject.com/ticket/31169 but nothing conclusive. My theory is that the memory copying semantics between MacOS and Windows are different and hence the spawn method doesn't have identical behaviour between the two.
Ahhh, sorry we don't use parallel on Windows.
Parallel running is disabled on Windows: ​https://github.com/django/django/blob/59b4e99dd00b9c36d56055b889f96885995e4240/django/test/runner.py#L286-L295 def default_test_processes(): """Default number of test processes when using the --parallel option.""" # The current implementation of the parallel test runner requires # multiprocessing to start subprocesses with fork(). if multiprocessing.get_start_method() != 'fork': return 1 try: return int(os.environ['DJANGO_TEST_PROCESSES']) except KeyError: return multiprocessing.cpu_count() I'll accept this as a new feature: the limitation has been there since it was implemented. Brandon, your patch is tiny. Is it really that simple? We'd need tests and a few other adjustments (like to the function above) but, fancy opening a PR?
So this occurs on macOS 10.15. (I have 10.14 currently so can't experiment there.) Applying the patch on Windows, alas, doesn't immediately solve the issue, but it is INSTALLED_APPS/AppRegistry errors that are raised, so it's going to be in the right ball-park. More investigating needed, but this would be a good one to land.
I created a PR with the changes from my patch: ​https://github.com/django/django/pull/12321 FYI" I am on macOS 10.14.6 I'm not sure how best to adjust default_test_processes as i've always used the --parallel flag with a parameter. Also, could you provide some guidance on how you'd like this tested
FYI" I am on macOS 10.14.6 Super. I had a 3.7. env active. I can reproduce with Python 3.8.
​PR
Thanks for the report. I had the same issue but did not find the root cause in https://code.djangoproject.com/ticket/31116. I would love to see that being resolved.
I ran into this while running the Django test suite, and when applying the patch in ​PR 12321, I get the same problem with a different exception: Traceback (most recent call last): File "/Users/inglesp/.pyenv/versions/3.8.0/lib/python3.8/multiprocessing/process.py", line 313, in _bootstrap self.run() File "/Users/inglesp/.pyenv/versions/3.8.0/lib/python3.8/multiprocessing/process.py", line 108, in run self._target(*self._args, **self._kwargs) File "/Users/inglesp/.pyenv/versions/3.8.0/lib/python3.8/multiprocessing/pool.py", line 114, in worker task = get() File "/Users/inglesp/.pyenv/versions/3.8.0/lib/python3.8/multiprocessing/queues.py", line 358, in get return _ForkingPickler.loads(res) File "/Users/inglesp/src/django/django/tests/fixtures_regress/tests.py", line 18, in <module> from .models import ( File "/Users/inglesp/src/django/django/tests/fixtures_regress/models.py", line 1, in <module> from django.contrib.auth.models import User File "/Users/inglesp/src/django/django/django/contrib/auth/models.py", line 3, in <module> from django.contrib.contenttypes.models import ContentType File "/Users/inglesp/src/django/django/django/contrib/contenttypes/models.py", line 133, in <module> class ContentType(models.Model): File "/Users/inglesp/src/django/django/django/db/models/base.py", line 113, in __new__ raise RuntimeError( RuntimeError: Model class django.contrib.contenttypes.models.ContentType doesn't declare an explicit app_label and isn't in an application in INSTALLED_APPS. I'm on OSX 10.15 with Python 3.8.
An attempt at a fix: ​https://github.com/django/django/pull/12547
For those looking for a workaround, here's how to add the appropriate call to reset back to fork mode: ​https://adamj.eu/tech/2020/07/21/how-to-use-djangos-parallel-testing-on-macos-with-python-3.8-plus/
PR is working nicely on macOS but needs a rebase, and a refactoring for review comments.
```

## Patch

```diff
diff --git a/django/db/backends/sqlite3/creation.py b/django/db/backends/sqlite3/creation.py
--- a/django/db/backends/sqlite3/creation.py
+++ b/django/db/backends/sqlite3/creation.py
@@ -1,8 +1,11 @@
+import multiprocessing
 import os
 import shutil
+import sqlite3
 import sys
 from pathlib import Path
 
+from django.db import NotSupportedError
 from django.db.backends.base.creation import BaseDatabaseCreation
 
 
@@ -51,16 +54,26 @@ def _create_test_db(self, verbosity, autoclobber, keepdb=False):
     def get_test_db_clone_settings(self, suffix):
         orig_settings_dict = self.connection.settings_dict
         source_database_name = orig_settings_dict["NAME"]
-        if self.is_in_memory_db(source_database_name):
+
+        if not self.is_in_memory_db(source_database_name):
+            root, ext = os.path.splitext(source_database_name)
+            return {**orig_settings_dict, "NAME": f"{root}_{suffix}{ext}"}
+
+        start_method = multiprocessing.get_start_method()
+        if start_method == "fork":
             return orig_settings_dict
-        else:
-            root, ext = os.path.splitext(orig_settings_dict["NAME"])
-            return {**orig_settings_dict, "NAME": "{}_{}{}".format(root, suffix, ext)}
+        if start_method == "spawn":
+            return {
+                **orig_settings_dict,
+                "NAME": f"{self.connection.alias}_{suffix}.sqlite3",
+            }
+        raise NotSupportedError(
+            f"Cloning with start method {start_method!r} is not supported."
+        )
 
     def _clone_test_db(self, suffix, verbosity, keepdb=False):
         source_database_name = self.connection.settings_dict["NAME"]
         target_database_name = self.get_test_db_clone_settings(suffix)["NAME"]
-        # Forking automatically makes a copy of an in-memory database.
         if not self.is_in_memory_db(source_database_name):
             # Erase the old test database
             if os.access(target_database_name, os.F_OK):
@@ -85,6 +98,12 @@ def _clone_test_db(self, suffix, verbosity, keepdb=False):
             except Exception as e:
                 self.log("Got an error cloning the test database: %s" % e)
                 sys.exit(2)
+        # Forking automatically makes a copy of an in-memory database.
+        # Spawn requires migrating to disk which will be re-opened in
+        # setup_worker_connection.
+        elif multiprocessing.get_start_method() == "spawn":
+            ondisk_db = sqlite3.connect(target_database_name, uri=True)
+            self.connection.connection.backup(ondisk_db)
 
     def _destroy_test_db(self, test_database_name, verbosity):
         if test_database_name and not self.is_in_memory_db(test_database_name):
@@ -106,3 +125,34 @@ def test_db_signature(self):
         else:
             sig.append(test_database_name)
         return tuple(sig)
+
+    def setup_worker_connection(self, _worker_id):
+        settings_dict = self.get_test_db_clone_settings(_worker_id)
+        # connection.settings_dict must be updated in place for changes to be
+        # reflected in django.db.connections. Otherwise new threads would
+        # connect to the default database instead of the appropriate clone.
+        start_method = multiprocessing.get_start_method()
+        if start_method == "fork":
+            # Update settings_dict in place.
+            self.connection.settings_dict.update(settings_dict)
+            self.connection.close()
+        elif start_method == "spawn":
+            alias = self.connection.alias
+            connection_str = (
+                f"file:memorydb_{alias}_{_worker_id}?mode=memory&cache=shared"
+            )
+            source_db = self.connection.Database.connect(
+                f"file:{alias}_{_worker_id}.sqlite3", uri=True
+            )
+            target_db = sqlite3.connect(connection_str, uri=True)
+            source_db.backup(target_db)
+            source_db.close()
+            # Update settings_dict in place.
+            self.connection.settings_dict.update(settings_dict)
+            self.connection.settings_dict["NAME"] = connection_str
+            # Re-open connection to in-memory database before closing copy
+            # connection.
+            self.connection.connect()
+            target_db.close()
+            if os.environ.get("RUNNING_DJANGOS_TEST_SUITE") == "true":
+                self.mark_expected_failures_and_skips()
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -130,7 +130,7 @@ def iter_modules_and_files(modules, extra_files):
         # cause issues here.
         if not isinstance(module, ModuleType):
             continue
-        if module.__name__ == "__main__":
+        if module.__name__ in ("__main__", "__mp_main__"):
             # __main__ (usually manage.py) doesn't always have a __spec__ set.
             # Handle this by falling back to using __file__, resolved below.
             # See https://docs.python.org/reference/import.html#main-spec

```

## Test Patch

```diff
diff --git a/django/test/runner.py b/django/test/runner.py
--- a/django/test/runner.py
+++ b/django/test/runner.py
@@ -20,7 +20,12 @@
 from django.core.management import call_command
 from django.db import connections
 from django.test import SimpleTestCase, TestCase
-from django.test.utils import NullTimeKeeper, TimeKeeper, iter_test_cases
+from django.test.utils import (
+    NullTimeKeeper,
+    TimeKeeper,
+    captured_stdout,
+    iter_test_cases,
+)
 from django.test.utils import setup_databases as _setup_databases
 from django.test.utils import setup_test_environment
 from django.test.utils import teardown_databases as _teardown_databases
@@ -367,8 +372,8 @@ def get_max_test_processes():
     The maximum number of test processes when using the --parallel option.
     """
     # The current implementation of the parallel test runner requires
-    # multiprocessing to start subprocesses with fork().
-    if multiprocessing.get_start_method() != "fork":
+    # multiprocessing to start subprocesses with fork() or spawn().
+    if multiprocessing.get_start_method() not in {"fork", "spawn"}:
         return 1
     try:
         return int(os.environ["DJANGO_TEST_PROCESSES"])
@@ -391,7 +396,13 @@ def parallel_type(value):
 _worker_id = 0
 
 
-def _init_worker(counter):
+def _init_worker(
+    counter,
+    initial_settings=None,
+    serialized_contents=None,
+    process_setup=None,
+    process_setup_args=None,
+):
     """
     Switch to databases dedicated to this worker.
 
@@ -405,9 +416,22 @@ def _init_worker(counter):
         counter.value += 1
         _worker_id = counter.value
 
+    start_method = multiprocessing.get_start_method()
+
+    if start_method == "spawn":
+        process_setup(*process_setup_args)
+        setup_test_environment()
+
     for alias in connections:
         connection = connections[alias]
+        if start_method == "spawn":
+            # Restore initial settings in spawned processes.
+            connection.settings_dict.update(initial_settings[alias])
+            if value := serialized_contents.get(alias):
+                connection._test_serialized_contents = value
         connection.creation.setup_worker_connection(_worker_id)
+        with captured_stdout():
+            call_command("check", databases=connections)
 
 
 def _run_subsuite(args):
@@ -449,6 +473,8 @@ def __init__(self, subsuites, processes, failfast=False, buffer=False):
         self.processes = processes
         self.failfast = failfast
         self.buffer = buffer
+        self.initial_settings = None
+        self.serialized_contents = None
         super().__init__()
 
     def run(self, result):
@@ -469,8 +495,12 @@ def run(self, result):
         counter = multiprocessing.Value(ctypes.c_int, 0)
         pool = multiprocessing.Pool(
             processes=self.processes,
-            initializer=self.init_worker.__func__,
-            initargs=[counter],
+            initializer=self.init_worker,
+            initargs=[
+                counter,
+                self.initial_settings,
+                self.serialized_contents,
+            ],
         )
         args = [
             (self.runner_class, index, subsuite, self.failfast, self.buffer)
@@ -508,6 +538,17 @@ def run(self, result):
     def __iter__(self):
         return iter(self.subsuites)
 
+    def initialize_suite(self):
+        if multiprocessing.get_start_method() == "spawn":
+            self.initial_settings = {
+                alias: connections[alias].settings_dict for alias in connections
+            }
+            self.serialized_contents = {
+                alias: connections[alias]._test_serialized_contents
+                for alias in connections
+                if alias in self.serialized_aliases
+            }
+
 
 class Shuffler:
     """
@@ -921,6 +962,8 @@ def run_checks(self, databases):
     def run_suite(self, suite, **kwargs):
         kwargs = self.get_test_runner_kwargs()
         runner = self.test_runner(**kwargs)
+        if hasattr(suite, "initialize_suite"):
+            suite.initialize_suite()
         try:
             return runner.run(suite)
         finally:
@@ -989,13 +1032,13 @@ def run_tests(self, test_labels, extra_tests=None, **kwargs):
         self.setup_test_environment()
         suite = self.build_suite(test_labels, extra_tests)
         databases = self.get_databases(suite)
-        serialized_aliases = set(
+        suite.serialized_aliases = set(
             alias for alias, serialize in databases.items() if serialize
         )
         with self.time_keeper.timed("Total database setup"):
             old_config = self.setup_databases(
                 aliases=databases,
-                serialized_aliases=serialized_aliases,
+                serialized_aliases=suite.serialized_aliases,
             )
         run_failed = False
         try:
diff --git a/tests/admin_checks/tests.py b/tests/admin_checks/tests.py
--- a/tests/admin_checks/tests.py
+++ b/tests/admin_checks/tests.py
@@ -70,6 +70,8 @@ class SessionMiddlewareSubclass(SessionMiddleware):
     ],
 )
 class SystemChecksTestCase(SimpleTestCase):
+    databases = "__all__"
+
     def test_checks_are_performed(self):
         admin.site.register(Song, MyAdmin)
         try:
diff --git a/tests/backends/sqlite/test_creation.py b/tests/backends/sqlite/test_creation.py
--- a/tests/backends/sqlite/test_creation.py
+++ b/tests/backends/sqlite/test_creation.py
@@ -1,7 +1,9 @@
 import copy
+import multiprocessing
 import unittest
+from unittest import mock
 
-from django.db import DEFAULT_DB_ALIAS, connection, connections
+from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connection, connections
 from django.test import SimpleTestCase
 
 
@@ -33,3 +35,9 @@ def test_get_test_db_clone_settings_name(self):
                 creation_class = test_connection.creation_class(test_connection)
                 clone_settings_dict = creation_class.get_test_db_clone_settings("1")
                 self.assertEqual(clone_settings_dict["NAME"], expected_clone_name)
+
+    @mock.patch.object(multiprocessing, "get_start_method", return_value="forkserver")
+    def test_get_test_db_clone_settings_not_supported(self, *mocked_objects):
+        msg = "Cloning with start method 'forkserver' is not supported."
+        with self.assertRaisesMessage(NotSupportedError, msg):
+            connection.creation.get_test_db_clone_settings(1)
diff --git a/tests/check_framework/tests.py b/tests/check_framework/tests.py
--- a/tests/check_framework/tests.py
+++ b/tests/check_framework/tests.py
@@ -362,5 +362,7 @@ class ModelWithDescriptorCalledCheck(models.Model):
 
 
 class ChecksRunDuringTests(SimpleTestCase):
+    databases = "__all__"
+
     def test_registered_check_did_run(self):
         self.assertTrue(my_check.did_run)
diff --git a/tests/contenttypes_tests/test_checks.py b/tests/contenttypes_tests/test_checks.py
--- a/tests/contenttypes_tests/test_checks.py
+++ b/tests/contenttypes_tests/test_checks.py
@@ -11,6 +11,8 @@
 
 @isolate_apps("contenttypes_tests", attr_name="apps")
 class GenericForeignKeyTests(SimpleTestCase):
+    databases = "__all__"
+
     def test_missing_content_type_field(self):
         class TaggedItem(models.Model):
             # no content_type field
diff --git a/tests/contenttypes_tests/test_management.py b/tests/contenttypes_tests/test_management.py
--- a/tests/contenttypes_tests/test_management.py
+++ b/tests/contenttypes_tests/test_management.py
@@ -22,6 +22,13 @@ class RemoveStaleContentTypesTests(TestCase):
 
     @classmethod
     def setUpTestData(cls):
+        with captured_stdout():
+            call_command(
+                "remove_stale_contenttypes",
+                interactive=False,
+                include_stale_apps=True,
+                verbosity=2,
+            )
         cls.before_count = ContentType.objects.count()
         cls.content_type = ContentType.objects.create(
             app_label="contenttypes_tests", model="Fake"
diff --git a/tests/postgres_tests/test_bulk_update.py b/tests/postgres_tests/test_bulk_update.py
--- a/tests/postgres_tests/test_bulk_update.py
+++ b/tests/postgres_tests/test_bulk_update.py
@@ -1,5 +1,7 @@
 from datetime import date
 
+from django.test import modify_settings
+
 from . import PostgreSQLTestCase
 from .models import (
     HStoreModel,
@@ -16,6 +18,7 @@
     pass  # psycopg2 isn't installed.
 
 
+@modify_settings(INSTALLED_APPS={"append": "django.contrib.postgres"})
 class BulkSaveTests(PostgreSQLTestCase):
     def test_bulk_update(self):
         test_data = [
diff --git a/tests/runtests.py b/tests/runtests.py
--- a/tests/runtests.py
+++ b/tests/runtests.py
@@ -3,6 +3,7 @@
 import atexit
 import copy
 import gc
+import multiprocessing
 import os
 import shutil
 import socket
@@ -10,6 +11,7 @@
 import sys
 import tempfile
 import warnings
+from functools import partial
 from pathlib import Path
 
 try:
@@ -24,7 +26,7 @@
     from django.core.exceptions import ImproperlyConfigured
     from django.db import connection, connections
     from django.test import TestCase, TransactionTestCase
-    from django.test.runner import get_max_test_processes, parallel_type
+    from django.test.runner import _init_worker, get_max_test_processes, parallel_type
     from django.test.selenium import SeleniumTestCaseBase
     from django.test.utils import NullTimeKeeper, TimeKeeper, get_runner
     from django.utils.deprecation import RemovedInDjango50Warning
@@ -382,7 +384,8 @@ def django_tests(
             msg += " with up to %d processes" % max_parallel
         print(msg)
 
-    test_labels, state = setup_run_tests(verbosity, start_at, start_after, test_labels)
+    process_setup_args = (verbosity, start_at, start_after, test_labels)
+    test_labels, state = setup_run_tests(*process_setup_args)
     # Run the test suite, including the extra validation tests.
     if not hasattr(settings, "TEST_RUNNER"):
         settings.TEST_RUNNER = "django.test.runner.DiscoverRunner"
@@ -395,6 +398,11 @@ def django_tests(
             parallel = 1
 
     TestRunner = get_runner(settings)
+    TestRunner.parallel_test_suite.init_worker = partial(
+        _init_worker,
+        process_setup=setup_run_tests,
+        process_setup_args=process_setup_args,
+    )
     test_runner = TestRunner(
         verbosity=verbosity,
         interactive=interactive,
@@ -718,6 +726,11 @@ def paired_tests(paired_test, options, test_labels, start_at, start_after):
         options.settings = os.environ["DJANGO_SETTINGS_MODULE"]
 
     if options.selenium:
+        if multiprocessing.get_start_method() == "spawn" and options.parallel != 1:
+            parser.error(
+                "You cannot use --selenium with parallel tests on this system. "
+                "Pass --parallel=1 to use --selenium."
+            )
         if not options.tags:
             options.tags = ["selenium"]
         elif "selenium" not in options.tags:
diff --git a/tests/test_runner/test_discover_runner.py b/tests/test_runner/test_discover_runner.py
--- a/tests/test_runner/test_discover_runner.py
+++ b/tests/test_runner/test_discover_runner.py
@@ -86,6 +86,16 @@ def test_get_max_test_processes_spawn(
         mocked_cpu_count,
     ):
         mocked_get_start_method.return_value = "spawn"
+        self.assertEqual(get_max_test_processes(), 12)
+        with mock.patch.dict(os.environ, {"DJANGO_TEST_PROCESSES": "7"}):
+            self.assertEqual(get_max_test_processes(), 7)
+
+    def test_get_max_test_processes_forkserver(
+        self,
+        mocked_get_start_method,
+        mocked_cpu_count,
+    ):
+        mocked_get_start_method.return_value = "forkserver"
         self.assertEqual(get_max_test_processes(), 1)
         with mock.patch.dict(os.environ, {"DJANGO_TEST_PROCESSES": "7"}):
             self.assertEqual(get_max_test_processes(), 1)
diff --git a/tests/test_runner/tests.py b/tests/test_runner/tests.py
--- a/tests/test_runner/tests.py
+++ b/tests/test_runner/tests.py
@@ -480,8 +480,6 @@ def test_time_recorded(self):
 # Isolate from the real environment.
 @mock.patch.dict(os.environ, {}, clear=True)
 @mock.patch.object(multiprocessing, "cpu_count", return_value=12)
-# Python 3.8 on macOS defaults to 'spawn' mode.
-@mock.patch.object(multiprocessing, "get_start_method", return_value="fork")
 class ManageCommandParallelTests(SimpleTestCase):
     def test_parallel_default(self, *mocked_objects):
         with captured_stderr() as stderr:
@@ -507,8 +505,8 @@ def test_no_parallel(self, *mocked_objects):
         # Parallel is disabled by default.
         self.assertEqual(stderr.getvalue(), "")
 
-    def test_parallel_spawn(self, mocked_get_start_method, mocked_cpu_count):
-        mocked_get_start_method.return_value = "spawn"
+    @mock.patch.object(multiprocessing, "get_start_method", return_value="spawn")
+    def test_parallel_spawn(self, *mocked_objects):
         with captured_stderr() as stderr:
             call_command(
                 "test",
@@ -517,8 +515,8 @@ def test_parallel_spawn(self, mocked_get_start_method, mocked_cpu_count):
             )
         self.assertIn("parallel=1", stderr.getvalue())
 
-    def test_no_parallel_spawn(self, mocked_get_start_method, mocked_cpu_count):
-        mocked_get_start_method.return_value = "spawn"
+    @mock.patch.object(multiprocessing, "get_start_method", return_value="spawn")
+    def test_no_parallel_spawn(self, *mocked_objects):
         with captured_stderr() as stderr:
             call_command(
                 "test",

```


## Code snippets

### 1 - django/core/management/commands/test.py:

Start line: 1, End line: 24

```python
import sys

from django.conf import settings
from django.core.management.base import BaseCommand
from django.core.management.utils import get_command_line_option
from django.test.runner import get_max_test_processes
from django.test.utils import NullTimeKeeper, TimeKeeper, get_runner


class Command(BaseCommand):
    help = "Discover and run tests in the specified modules or the current directory."

    # DiscoverRunner runs the checks after databases are set up.
    requires_system_checks = []
    test_runner = None

    def run_from_argv(self, argv):
        """
        Pre-parse the command line to extract the value of the --testrunner
        option. This allows a test runner to define additional command line
        arguments.
        """
        self.test_runner = get_command_line_option(argv, "--testrunner")
        super().run_from_argv(argv)
```
### 2 - django/core/management/commands/test.py:

Start line: 59, End line: 72

```python
class Command(BaseCommand):

    def handle(self, *test_labels, **options):
        TestRunner = get_runner(settings, options["testrunner"])

        time_keeper = TimeKeeper() if options.get("timing", False) else NullTimeKeeper()
        parallel = options.get("parallel")
        if parallel == "auto":
            options["parallel"] = get_max_test_processes()
        test_runner = TestRunner(**options)
        with time_keeper.timed("Total run"):
            failures = test_runner.run_tests(test_labels)
        time_keeper.print_results()
        if failures:
            sys.exit(1)
```
### 3 - django/db/backends/base/creation.py:

Start line: 327, End line: 351

```python
class BaseDatabaseCreation:

    def mark_expected_failures_and_skips(self):
        """
        Mark tests in Django's test suite which are expected failures on this
        database and test which should be skipped on this database.
        """
        # Only load unittest if we're actually testing.
        from unittest import expectedFailure, skip

        for test_name in self.connection.features.django_test_expected_failures:
            test_case_name, _, test_method_name = test_name.rpartition(".")
            test_app = test_name.split(".")[0]
            # Importing a test app that isn't installed raises RuntimeError.
            if test_app in settings.INSTALLED_APPS:
                test_case = import_string(test_case_name)
                test_method = getattr(test_case, test_method_name)
                setattr(test_case, test_method_name, expectedFailure(test_method))
        for reason, tests in self.connection.features.django_test_skips.items():
            for test_name in tests:
                test_case_name, _, test_method_name = test_name.rpartition(".")
                test_app = test_name.split(".")[0]
                # Importing a test app that isn't installed raises RuntimeError.
                if test_app in settings.INSTALLED_APPS:
                    test_case = import_string(test_case_name)
                    test_method = getattr(test_case, test_method_name)
                    setattr(test_case, test_method_name, skip(reason)(test_method))
```
### 4 - django/contrib/admin/tests.py:

Start line: 40, End line: 48

```python
@modify_settings(MIDDLEWARE={"append": "django.contrib.admin.tests.CSPMiddleware"})
class AdminSeleniumTestCase(SeleniumTestCase, StaticLiveServerTestCase):

    def wait_for_and_switch_to_popup(self, num_windows=2, timeout=10):
        """
        Block until `num_windows` are present and are ready (usually 2, but can
        be overridden in the case of pop-ups opening other pop-ups). Switch the
        current window to the new pop-up.
        """
        self.wait_until(lambda d: len(d.window_handles) == num_windows, timeout)
        self.selenium.switch_to.window(self.selenium.window_handles[-1])
        self.wait_page_ready()
```
### 5 - django/db/backends/base/base.py:

Start line: 1, End line: 36

```python
import _thread
import copy
import threading
import time
import warnings
from collections import deque
from contextlib import contextmanager

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DEFAULT_DB_ALIAS, DatabaseError, NotSupportedError
from django.db.backends import utils
from django.db.backends.base.validation import BaseDatabaseValidation
from django.db.backends.signals import connection_created
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseErrorWrapper
from django.utils import timezone
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property

NO_DB_ALIAS = "__no_db__"
RAN_DB_VERSION_CHECK = set()


# RemovedInDjango50Warning
def timezone_constructor(tzname):
    if settings.USE_DEPRECATED_PYTZ:
        import pytz

        return pytz.timezone(tzname)
    return zoneinfo.ZoneInfo(tzname)
```
### 6 - django/core/management/commands/testserver.py:

Start line: 38, End line: 66

```python
class Command(BaseCommand):

    def handle(self, *fixture_labels, **options):
        verbosity = options["verbosity"]
        interactive = options["interactive"]

        # Create a test database.
        db_name = connection.creation.create_test_db(
            verbosity=verbosity, autoclobber=not interactive, serialize=False
        )

        # Import the fixture data into the test database.
        call_command("loaddata", *fixture_labels, **{"verbosity": verbosity})

        # Run the development server. Turn off auto-reloading because it causes
        # a strange error -- it causes this handle() method to be called
        # multiple times.
        shutdown_message = (
            "\nServer stopped.\nNote that the test database, %r, has not been "
            "deleted. You can explore it on your own." % db_name
        )
        use_threading = connection.features.test_db_allows_multiple_connections
        call_command(
            "runserver",
            addrport=options["addrport"],
            shutdown_message=shutdown_message,
            use_reloader=False,
            use_ipv6=options["use_ipv6"],
            use_threading=use_threading,
        )
```
### 7 - django/db/backends/base/creation.py:

Start line: 1, End line: 104

```python
import os
import sys
from io import StringIO

from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string

# The prefix to put on the default database name when creating
# the test database.
TEST_DATABASE_PREFIX = "test_"


class BaseDatabaseCreation:
    """
    Encapsulate backend-specific differences pertaining to creation and
    destruction of the test database.
    """

    def __init__(self, connection):
        self.connection = connection

    def _nodb_cursor(self):
        return self.connection._nodb_cursor()

    def log(self, msg):
        sys.stderr.write(msg + os.linesep)

    def create_test_db(
        self, verbosity=1, autoclobber=False, serialize=True, keepdb=False
    ):
        """
        Create a test database, prompting the user for confirmation if the
        database already exists. Return the name of the test database created.
        """
        # Don't import django.core.management if it isn't needed.
        from django.core.management import call_command

        test_database_name = self._get_test_db_name()

        if verbosity >= 1:
            action = "Creating"
            if keepdb:
                action = "Using existing"

            self.log(
                "%s test database for alias %s..."
                % (
                    action,
                    self._get_database_display_str(verbosity, test_database_name),
                )
            )

        # We could skip this call if keepdb is True, but we instead
        # give it the keepdb param. This is to handle the case
        # where the test DB doesn't exist, in which case we need to
        # create it, then just not destroy it. If we instead skip
        # this, we will get an exception.
        self._create_test_db(verbosity, autoclobber, keepdb)

        self.connection.close()
        settings.DATABASES[self.connection.alias]["NAME"] = test_database_name
        self.connection.settings_dict["NAME"] = test_database_name

        try:
            if self.connection.settings_dict["TEST"]["MIGRATE"] is False:
                # Disable migrations for all apps.
                old_migration_modules = settings.MIGRATION_MODULES
                settings.MIGRATION_MODULES = {
                    app.label: None for app in apps.get_app_configs()
                }
            # We report migrate messages at one level lower than that
            # requested. This ensures we don't get flooded with messages during
            # testing (unless you really ask to be flooded).
            call_command(
                "migrate",
                verbosity=max(verbosity - 1, 0),
                interactive=False,
                database=self.connection.alias,
                run_syncdb=True,
            )
        finally:
            if self.connection.settings_dict["TEST"]["MIGRATE"] is False:
                settings.MIGRATION_MODULES = old_migration_modules

        # We then serialize the current state of the database into a string
        # and store it on the connection. This slightly horrific process is so people
        # who are testing on databases without transactions or who are using
        # a TransactionTestCase still get a clean database on every test run.
        if serialize:
            self.connection._test_serialized_contents = self.serialize_db_to_string()

        call_command("createcachetable", database=self.connection.alias)

        # Ensure a connection for the side effect of initializing the test database.
        self.connection.ensure_connection()

        if os.environ.get("RUNNING_DJANGOS_TEST_SUITE") == "true":
            self.mark_expected_failures_and_skips()

        return test_database_name
```
### 8 - django/db/backends/oracle/creation.py:

Start line: 29, End line: 124

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _create_test_db(self, verbosity=1, autoclobber=False, keepdb=False):
        parameters = self._get_test_db_params()
        with self._maindb_connection.cursor() as cursor:
            if self._test_database_create():
                try:
                    self._execute_test_db_creation(
                        cursor, parameters, verbosity, keepdb
                    )
                except Exception as e:
                    if "ORA-01543" not in str(e):
                        # All errors except "tablespace already exists" cancel tests
                        self.log("Got an error creating the test database: %s" % e)
                        sys.exit(2)
                    if not autoclobber:
                        confirm = input(
                            "It appears the test database, %s, already exists. "
                            "Type 'yes' to delete it, or 'no' to cancel: "
                            % parameters["user"]
                        )
                    if autoclobber or confirm == "yes":
                        if verbosity >= 1:
                            self.log(
                                "Destroying old test database for alias '%s'..."
                                % self.connection.alias
                            )
                        try:
                            self._execute_test_db_destruction(
                                cursor, parameters, verbosity
                            )
                        except DatabaseError as e:
                            if "ORA-29857" in str(e):
                                self._handle_objects_preventing_db_destruction(
                                    cursor, parameters, verbosity, autoclobber
                                )
                            else:
                                # Ran into a database error that isn't about
                                # leftover objects in the tablespace.
                                self.log(
                                    "Got an error destroying the old test database: %s"
                                    % e
                                )
                                sys.exit(2)
                        except Exception as e:
                            self.log(
                                "Got an error destroying the old test database: %s" % e
                            )
                            sys.exit(2)
                        try:
                            self._execute_test_db_creation(
                                cursor, parameters, verbosity, keepdb
                            )
                        except Exception as e:
                            self.log(
                                "Got an error recreating the test database: %s" % e
                            )
                            sys.exit(2)
                    else:
                        self.log("Tests cancelled.")
                        sys.exit(1)

            if self._test_user_create():
                if verbosity >= 1:
                    self.log("Creating test user...")
                try:
                    self._create_test_user(cursor, parameters, verbosity, keepdb)
                except Exception as e:
                    if "ORA-01920" not in str(e):
                        # All errors except "user already exists" cancel tests
                        self.log("Got an error creating the test user: %s" % e)
                        sys.exit(2)
                    if not autoclobber:
                        confirm = input(
                            "It appears the test user, %s, already exists. Type "
                            "'yes' to delete it, or 'no' to cancel: "
                            % parameters["user"]
                        )
                    if autoclobber or confirm == "yes":
                        try:
                            if verbosity >= 1:
                                self.log("Destroying old test user...")
                            self._destroy_test_user(cursor, parameters, verbosity)
                            if verbosity >= 1:
                                self.log("Creating test user...")
                            self._create_test_user(
                                cursor, parameters, verbosity, keepdb
                            )
                        except Exception as e:
                            self.log("Got an error recreating the test user: %s" % e)
                            sys.exit(2)
                    else:
                        self.log("Tests cancelled.")
                        sys.exit(1)
        # Done with main user -- test user and tablespaces created.
        self._maindb_connection.close()
        self._switch_to_test_user(parameters)
        return self.connection.settings_dict["NAME"]
```
### 9 - django/db/backends/base/creation.py:

Start line: 235, End line: 255

```python
class BaseDatabaseCreation:

    def clone_test_db(self, suffix, verbosity=1, autoclobber=False, keepdb=False):
        """
        Clone a test database.
        """
        source_database_name = self.connection.settings_dict["NAME"]

        if verbosity >= 1:
            action = "Cloning test database"
            if keepdb:
                action = "Using existing clone"
            self.log(
                "%s for alias %s..."
                % (
                    action,
                    self._get_database_display_str(verbosity, source_database_name),
                )
            )

        # We could skip this call if keepdb is True, but we instead
        # give it the keepdb param. See create_test_db for details.
        self._clone_test_db(suffix, verbosity, keepdb)
```
### 10 - django/core/management/commands/startapp.py:

Start line: 1, End line: 15

```python

```
### 13 - django/utils/autoreload.py:

Start line: 672, End line: 686

```python
def run_with_reloader(main_func, *args, **kwargs):
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    try:
        if os.environ.get(DJANGO_AUTORELOAD_ENV) == "true":
            reloader = get_reloader()
            logger.info(
                "Watching for file changes with %s", reloader.__class__.__name__
            )
            start_django(reloader, main_func, *args, **kwargs)
        else:
            exit_code = restart_with_reloader()
            sys.exit(exit_code)
    except KeyboardInterrupt:
        pass
```
### 14 - django/db/backends/sqlite3/creation.py:

Start line: 51, End line: 87

```python
class DatabaseCreation(BaseDatabaseCreation):

    def get_test_db_clone_settings(self, suffix):
        orig_settings_dict = self.connection.settings_dict
        source_database_name = orig_settings_dict["NAME"]
        if self.is_in_memory_db(source_database_name):
            return orig_settings_dict
        else:
            root, ext = os.path.splitext(orig_settings_dict["NAME"])
            return {**orig_settings_dict, "NAME": "{}_{}{}".format(root, suffix, ext)}

    def _clone_test_db(self, suffix, verbosity, keepdb=False):
        source_database_name = self.connection.settings_dict["NAME"]
        target_database_name = self.get_test_db_clone_settings(suffix)["NAME"]
        # Forking automatically makes a copy of an in-memory database.
        if not self.is_in_memory_db(source_database_name):
            # Erase the old test database
            if os.access(target_database_name, os.F_OK):
                if keepdb:
                    return
                if verbosity >= 1:
                    self.log(
                        "Destroying old test database for alias %s..."
                        % (
                            self._get_database_display_str(
                                verbosity, target_database_name
                            ),
                        )
                    )
                try:
                    os.remove(target_database_name)
                except Exception as e:
                    self.log("Got an error deleting the old test database: %s" % e)
                    sys.exit(2)
            try:
                shutil.copy(source_database_name, target_database_name)
            except Exception as e:
                self.log("Got an error cloning the test database: %s" % e)
                sys.exit(2)
```
### 15 - django/utils/autoreload.py:

Start line: 329, End line: 344

```python
class BaseReloader:

    def run(self, django_main_thread):
        logger.debug("Waiting for apps ready_event.")
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
        logger.debug("Apps ready_event triggered. Sending autoreload_started signal.")
        autoreload_started.send(sender=self)
        self.run_loop()
```
### 21 - django/utils/autoreload.py:

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

DJANGO_AUTORELOAD_ENV = "RUN_MAIN"

logger = logging.getLogger("django.utils.autoreload")

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
    return module.__name__.startswith("django.")


def is_django_path(path):
    """Return True if the given file path is nested under Django."""
    return Path(django.__file__).parent in Path(path).parents
```
### 36 - django/utils/autoreload.py:

Start line: 221, End line: 262

```python
def get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.
    """
    import __main__

    py_script = Path(sys.argv[0])

    args = [sys.executable] + ["-W%s" % o for o in sys.warnoptions]
    if sys.implementation.name == "cpython":
        args.extend(
            f"-X{key}" if value is True else f"-X{key}={value}"
            for key, value in sys._xoptions.items()
        )
    # __spec__ is set when the server was started with the `-m` option,
    # see https://docs.python.org/3/reference/import.html#main-spec
    # __spec__ may not exist, e.g. when running in a Conda env.
    if getattr(__main__, "__spec__", None) is not None:
        spec = __main__.__spec__
        if (spec.name == "__main__" or spec.name.endswith(".__main__")) and spec.parent:
            name = spec.parent
        else:
            name = spec.name
        args += ["-m", name]
        args += sys.argv[1:]
    elif not py_script.exists():
        # sys.argv[0] may not exist for several reasons on Windows.
        # It may exist with a .exe extension or have a -script.py suffix.
        exe_entrypoint = py_script.with_suffix(".exe")
        if exe_entrypoint.exists():
            # Should be executed directly, ignoring sys.executable.
            return [exe_entrypoint, *sys.argv[1:]]
        script_entrypoint = py_script.with_name("%s-script.py" % py_script.name)
        if script_entrypoint.exists():
            # Should be executed as usual.
            return [*args, script_entrypoint, *sys.argv[1:]]
        raise RuntimeError("Script %s does not exist." % py_script)
    else:
        args += sys.argv
    return args
```
### 40 - django/utils/autoreload.py:

Start line: 311, End line: 327

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
            logger.debug("Main Django thread has terminated before apps are ready.")
            return False
```
### 47 - django/utils/autoreload.py:

Start line: 431, End line: 464

```python
class WatchmanReloader(BaseReloader):
    def __init__(self):
        self.roots = defaultdict(set)
        self.processed_request = threading.Event()
        self.client_timeout = int(os.environ.get("DJANGO_WATCHMAN_TIMEOUT", 5))
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
                logger.warning(
                    "Unable to watch root dir %s as neither it or its parent exist.",
                    root,
                )
                return
            root = root.parent
        result = self.client.query("watch-project", str(root.absolute()))
        if "warning" in result:
            logger.warning("Watchman warning: %s", result["warning"])
        logger.debug("Watchman watch-project result: %s", result)
        return result["watch"], result.get("relative_path")
```
### 56 - django/db/backends/sqlite3/creation.py:

Start line: 1, End line: 20

```python
import os
import shutil
import sys
from pathlib import Path

from django.db.backends.base.creation import BaseDatabaseCreation


class DatabaseCreation(BaseDatabaseCreation):
    @staticmethod
    def is_in_memory_db(database_name):
        return not isinstance(database_name, Path) and (
            database_name == ":memory:" or "mode=memory" in database_name
        )

    def _get_test_db_name(self):
        test_database_name = self.connection.settings_dict["TEST"]["NAME"] or ":memory:"
        if test_database_name == ":memory:":
            return "file:memorydb_%s?mode=memory&cache=shared" % self.connection.alias
        return test_database_name
```
### 65 - django/db/backends/sqlite3/creation.py:

Start line: 22, End line: 49

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _create_test_db(self, verbosity, autoclobber, keepdb=False):
        test_database_name = self._get_test_db_name()

        if keepdb:
            return test_database_name
        if not self.is_in_memory_db(test_database_name):
            # Erase the old test database
            if verbosity >= 1:
                self.log(
                    "Destroying old test database for alias %s..."
                    % (self._get_database_display_str(verbosity, test_database_name),)
                )
            if os.access(test_database_name, os.F_OK):
                if not autoclobber:
                    confirm = input(
                        "Type 'yes' if you would like to try deleting the test "
                        "database '%s', or 'no' to cancel: " % test_database_name
                    )
                if autoclobber or confirm == "yes":
                    try:
                        os.remove(test_database_name)
                    except Exception as e:
                        self.log("Got an error deleting the old test database: %s" % e)
                        sys.exit(2)
                else:
                    self.log("Tests cancelled.")
                    sys.exit(1)
        return test_database_name
```
### 90 - django/utils/autoreload.py:

Start line: 346, End line: 381

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
        raise NotImplementedError("subclasses must implement tick().")

    @classmethod
    def check_availability(cls):
        raise NotImplementedError("subclasses must implement check_availability().")

    def notify_file_changed(self, path):
        results = file_changed.send(sender=self, file_path=path)
        logger.debug("%s notified as changed. Signal results: %s.", path, results)
        if not any(res[1] for res in results):
            trigger_reload(path)

    # These are primarily used for testing.
    @property
    def should_stop(self):
        return self._stop_condition.is_set()

    def stop(self):
        self._stop_condition.set()
```
### 93 - django/utils/autoreload.py:

Start line: 538, End line: 560

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
        logger.debug("Watching %s files", len(watched_files))
        logger.debug("Found common roots: %s", found_roots)
        # Setup initial roots for performance, shortest roots first.
        for root in sorted(found_roots):
            self._watch_root(root)
        for directory, patterns in self.directory_globs.items():
            self._watch_glob(directory, patterns)
        # Group sorted watched_files by their parent directory.
        sorted_files = sorted(watched_files, key=lambda p: p.parent)
        for directory, group in itertools.groupby(sorted_files, key=lambda p: p.parent):
            # These paths need to be relative to the parent directory.
            self._subscribe_dir(
                directory, [str(p.relative_to(directory)) for p in group]
            )
```
### 96 - django/utils/autoreload.py:

Start line: 640, End line: 669

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
    django_main_thread = threading.Thread(
        target=main_func, args=args, kwargs=kwargs, name="django-main-thread"
    )
    django_main_thread.daemon = True
    django_main_thread.start()

    while not reloader.should_stop:
        try:
            reloader.run(django_main_thread)
        except WatchmanUnavailable as ex:
            # It's possible that the watchman service shuts down or otherwise
            # becomes unavailable. In that case, use the StatReloader.
            reloader = StatReloader()
            logger.error("Error connecting to Watchman: %s", ex)
            logger.info(
                "Watching for file changes with %s", reloader.__class__.__name__
            )
```
### 98 - django/utils/autoreload.py:

Start line: 265, End line: 309

```python
def trigger_reload(filename):
    logger.info("%s changed, reloading.", filename)
    sys.exit(3)


def restart_with_reloader():
    new_environ = {**os.environ, DJANGO_AUTORELOAD_ENV: "true"}
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
                "Unable to watch directory %s as it cannot be resolved.",
                path,
                exc_info=True,
            )
            return
        logger.debug("Watching dir %s with glob %s.", path, glob)
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
### 104 - django/utils/autoreload.py:

Start line: 585, End line: 608

```python
class WatchmanReloader(BaseReloader):

    def request_processed(self, **kwargs):
        logger.debug("Request processed. Setting update_watches event.")
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
                logger.debug("Watchman error: %s, checking server status.", ex)
                self.check_server_status(ex)
            else:
                for sub in list(self.client.subs.keys()):
                    self._check_subscription(sub)
            yield
            # Protect against busy loops.
            time.sleep(0.1)
```
### 126 - django/utils/autoreload.py:

Start line: 384, End line: 428

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
                    logger.debug("File %s first seen with mtime %s", filepath, mtime)
                    continue
                elif mtime > old_time:
                    logger.debug(
                        "File %s previous mtime: %s, current mtime: %s",
                        filepath,
                        old_time,
                        mtime,
                    )
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
### 142 - django/utils/autoreload.py:

Start line: 466, End line: 493

```python
class WatchmanReloader(BaseReloader):

    @functools.lru_cache
    def _get_clock(self, root):
        return self.client.query("clock", root)["clock"]

    def _subscribe(self, directory, name, expression):
        root, rel_path = self._watch_root(directory)
        # Only receive notifications of files changing, filtering out other types
        # like special files: https://facebook.github.io/watchman/docs/type
        only_files_expression = [
            "allof",
            ["anyof", ["type", "f"], ["type", "l"]],
            expression,
        ]
        query = {
            "expression": only_files_expression,
            "fields": ["name"],
            "since": self._get_clock(root),
            "dedup_results": True,
        }
        if rel_path:
            query["relative_root"] = rel_path
        logger.debug(
            "Issuing watchman subscription %s, for root %s. Query: %s",
            name,
            root,
            query,
        )
        self.client.query("subscribe", root, name, query)
```
### 163 - django/utils/autoreload.py:

Start line: 495, End line: 510

```python
class WatchmanReloader(BaseReloader):

    def _subscribe_dir(self, directory, filenames):
        if not directory.exists():
            if not directory.parent.exists():
                logger.warning(
                    "Unable to watch directory %s as neither it or its parent exist.",
                    directory,
                )
                return
            prefix = "files-parent-%s" % directory.name
            filenames = ["%s/%s" % (directory.name, filename) for filename in filenames]
            directory = directory.parent
            expression = ["name", filenames, "wholename"]
        else:
            prefix = "files"
            expression = ["name", filenames]
        self._subscribe(directory, "%s:%s" % (prefix, directory), expression)
```
### 178 - django/utils/autoreload.py:

Start line: 562, End line: 583

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
        logger.debug("Watchman subscription %s has results.", sub)
        for result in subscription:
            # When using watch-project, it's not simple to get the relative
            # directory without storing some specific state. Store the full
            # path to the directory in the subscription name, prefixed by its
            # type (glob, files).
            root_directory = Path(result["subscription"].split(":", 1)[1])
            logger.debug("Found root directory %s", root_directory)
            for file in result.get("files", []):
                self.notify_file_changed(root_directory / file)
```
### 186 - django/utils/autoreload.py:

Start line: 512, End line: 536

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
        prefix = "glob"
        if not directory.exists():
            if not directory.parent.exists():
                logger.warning(
                    "Unable to watch directory %s as neither it or its parent exist.",
                    directory,
                )
                return
            prefix = "glob-parent-%s" % directory.name
            patterns = ["%s/%s" % (directory.name, pattern) for pattern in patterns]
            directory = directory.parent

        expression = ["anyof"]
        for pattern in patterns:
            expression.append(["match", pattern, "wholename"])
        self._subscribe(directory, "%s:%s" % (prefix, directory), expression)
```
