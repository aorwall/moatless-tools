# django__django-16532

| **django/django** | `ce8189eea007882bbe6db22f86b0965e718bd341` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 499 |
| **Any found context length** | 499 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -407,20 +407,12 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
                     or not new_field.remote_field.through._meta.auto_created
                 ):
                     continue
-                # Rename the M2M table that's based on this model's name.
-                old_m2m_model = old_field.remote_field.through
-                new_m2m_model = new_field.remote_field.through
-                schema_editor.alter_db_table(
-                    new_m2m_model,
-                    old_m2m_model._meta.db_table,
-                    new_m2m_model._meta.db_table,
-                )
-                # Rename the column in the M2M table that's based on this
-                # model's name.
-                schema_editor.alter_field(
-                    new_m2m_model,
-                    old_m2m_model._meta.get_field(old_model._meta.model_name),
-                    new_m2m_model._meta.get_field(new_model._meta.model_name),
+                # Rename columns and the M2M table.
+                schema_editor._alter_many_to_many(
+                    new_model,
+                    old_field,
+                    new_field,
+                    strict=False,
                 )
 
     def database_backwards(self, app_label, schema_editor, from_state, to_state):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/operations/models.py | 410 | 423 | 1 | 1 | 499


## Problem Statement

```
Duplicate model names in M2M relationship causes RenameModel migration failure
Description
	
Example code is here: ​https://github.com/jzmiller1/edemo
I have a django project with two apps, incidents and vault, that both have a model named Incident. The vault Incident model has an M2M involving the incidents Incident model. When the table is created for this M2M relationship the automatic field names are "from_incident_id" and "to_incident_id" since models have the same names.
If I then try to use a RenameModel in a migration... 
	operations = [
		migrations.RenameModel(
			old_name='Incident',
			new_name='Folder',
		),
	]
it fails with this traceback:
Tracking file by folder pattern: migrations
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, incidents, sessions, vault
Running migrations:
 Applying vault.0002_rename_incident_folder...Traceback (most recent call last):
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/models/options.py", line 668, in get_field
	return self.fields_map[field_name]
KeyError: 'incident'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
 File "/Users/zacmiller/Library/Application Support/JetBrains/Toolbox/apps/PyCharm-P/ch-0/222.4345.23/PyCharm.app/Contents/plugins/python/helpers/pycharm/django_manage.py", line 52, in <module>
	run_command()
 File "/Users/zacmiller/Library/Application Support/JetBrains/Toolbox/apps/PyCharm-P/ch-0/222.4345.23/PyCharm.app/Contents/plugins/python/helpers/pycharm/django_manage.py", line 46, in run_command
	run_module(manage_file, None, '__main__', True)
 File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 209, in run_module
	return _run_module_code(code, init_globals, run_name, mod_spec)
 File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 96, in _run_module_code
	_run_code(code, mod_globals, init_globals,
 File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 86, in _run_code
	exec(code, run_globals)
 File "/Users/zacmiller/PycharmProjects/edemo/manage.py", line 22, in <module>
	main()
 File "/Users/zacmiller/PycharmProjects/edemo/manage.py", line 18, in main
	execute_from_command_line(sys.argv)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/__init__.py", line 446, in execute_from_command_line
	utility.execute()
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/__init__.py", line 440, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/base.py", line 402, in run_from_argv
	self.execute(*args, **cmd_options)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/base.py", line 448, in execute
	output = self.handle(*args, **options)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/base.py", line 96, in wrapped
	res = handle_func(*args, **kwargs)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/core/management/commands/migrate.py", line 349, in handle
	post_migrate_state = executor.migrate(
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/migrations/executor.py", line 135, in migrate
	state = self._migrate_all_forwards(
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/migrations/executor.py", line 167, in _migrate_all_forwards
	state = self.apply_migration(
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/migrations/executor.py", line 252, in apply_migration
	state = migration.apply(state, schema_editor)
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/migrations/migration.py", line 130, in apply
	operation.database_forwards(
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/migrations/operations/models.py", line 422, in database_forwards
	old_m2m_model._meta.get_field(old_model._meta.model_name),
 File "/Users/zacmiller/PycharmProjects/virtualenvs/edemo/lib/python3.10/site-packages/django/db/models/options.py", line 670, in get_field
	raise FieldDoesNotExist(
django.core.exceptions.FieldDoesNotExist: Incident_incidents has no field named 'incident'

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/migrations/operations/models.py** | 370 | 424| 499 | 499 | 7999 | 
| 2 | **1 django/db/migrations/operations/models.py** | 426 | 439| 127 | 626 | 7999 | 
| 3 | 2 django/db/migrations/autodetector.py | 512 | 577| 482 | 1108 | 21613 | 
| 4 | 3 django/db/migrations/state.py | 291 | 345| 476 | 1584 | 29781 | 
| 5 | 4 django/db/backends/sqlite3/schema.py | 489 | 551| 472 | 2056 | 34496 | 
| 6 | **4 django/db/migrations/operations/models.py** | 441 | 478| 267 | 2323 | 34496 | 
| 7 | 4 django/db/migrations/state.py | 142 | 179| 395 | 2718 | 34496 | 
| 8 | 5 django/db/models/base.py | 1680 | 1714| 252 | 2970 | 53223 | 
| 9 | 6 django/db/migrations/operations/fields.py | 270 | 337| 521 | 3491 | 55735 | 
| 10 | **6 django/db/migrations/operations/models.py** | 344 | 368| 157 | 3648 | 55735 | 
| 11 | **6 django/db/migrations/operations/models.py** | 481 | 529| 407 | 4055 | 55735 | 
| 12 | 6 django/db/migrations/autodetector.py | 1613 | 1645| 258 | 4313 | 55735 | 
| 13 | 7 django/db/models/fields/related.py | 1440 | 1569| 984 | 5297 | 70275 | 
| 14 | 7 django/db/migrations/autodetector.py | 807 | 902| 712 | 6009 | 70275 | 
| 15 | 8 django/db/migrations/exceptions.py | 1 | 61| 249 | 6258 | 70525 | 
| 16 | 9 django/db/backends/base/schema.py | 1300 | 1344| 412 | 6670 | 84941 | 
| 17 | 9 django/db/migrations/autodetector.py | 904 | 977| 623 | 7293 | 84941 | 
| 18 | 9 django/db/migrations/autodetector.py | 1530 | 1549| 187 | 7480 | 84941 | 
| 19 | 9 django/db/migrations/autodetector.py | 1647 | 1661| 135 | 7615 | 84941 | 
| 20 | 10 django/core/management/commands/migrate.py | 96 | 189| 765 | 8380 | 88867 | 
| 21 | 10 django/db/migrations/autodetector.py | 1428 | 1473| 318 | 8698 | 88867 | 
| 22 | **10 django/db/migrations/operations/models.py** | 971 | 1006| 319 | 9017 | 88867 | 
| 23 | 10 django/db/models/fields/related.py | 1670 | 1720| 431 | 9448 | 88867 | 
| 24 | 11 django/db/migrations/questioner.py | 217 | 247| 252 | 9700 | 91563 | 
| 25 | 11 django/db/migrations/autodetector.py | 596 | 772| 1231 | 10931 | 91563 | 
| 26 | 11 django/core/management/commands/migrate.py | 270 | 368| 813 | 11744 | 91563 | 
| 27 | 11 django/db/migrations/state.py | 437 | 458| 204 | 11948 | 91563 | 
| 28 | 11 django/db/migrations/autodetector.py | 1097 | 1214| 982 | 12930 | 91563 | 
| 29 | 11 django/db/models/base.py | 1815 | 1837| 175 | 13105 | 91563 | 
| 30 | 11 django/db/models/base.py | 2187 | 2268| 592 | 13697 | 91563 | 
| 31 | **11 django/db/migrations/operations/models.py** | 136 | 306| 968 | 14665 | 91563 | 
| 32 | 12 django/db/migrations/operations/__init__.py | 1 | 45| 240 | 14905 | 91803 | 
| 33 | **12 django/db/migrations/operations/models.py** | 1008 | 1025| 149 | 15054 | 91803 | 
| 34 | 12 django/db/backends/base/schema.py | 39 | 72| 214 | 15268 | 91803 | 
| 35 | 12 django/db/backends/base/schema.py | 620 | 637| 154 | 15422 | 91803 | 
| 36 | 13 django/contrib/redirects/migrations/0001_initial.py | 1 | 65| 309 | 15731 | 92112 | 
| 37 | 13 django/db/migrations/autodetector.py | 1551 | 1571| 192 | 15923 | 92112 | 
| 38 | 13 django/db/migrations/autodetector.py | 979 | 1021| 297 | 16220 | 92112 | 
| 39 | **13 django/db/migrations/operations/models.py** | 1027 | 1062| 249 | 16469 | 92112 | 
| 40 | 13 django/db/migrations/autodetector.py | 579 | 595| 186 | 16655 | 92112 | 
| 41 | 13 django/db/migrations/state.py | 1 | 30| 236 | 16891 | 92112 | 
| 42 | **13 django/db/migrations/operations/models.py** | 113 | 134| 164 | 17055 | 92112 | 
| 43 | 14 django/core/management/commands/makemigrations.py | 104 | 194| 791 | 17846 | 96060 | 
| 44 | 14 django/db/models/base.py | 1529 | 1565| 288 | 18134 | 96060 | 
| 45 | **14 django/db/migrations/operations/models.py** | 708 | 724| 159 | 18293 | 96060 | 
| 46 | 14 django/db/backends/sqlite3/schema.py | 175 | 255| 760 | 19053 | 96060 | 
| 47 | 14 django/db/models/fields/related.py | 1871 | 1918| 505 | 19558 | 96060 | 
| 48 | 14 django/core/management/commands/migrate.py | 191 | 269| 678 | 20236 | 96060 | 
| 49 | 15 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 20631 | 96726 | 
| 50 | **15 django/db/migrations/operations/models.py** | 682 | 706| 231 | 20862 | 96726 | 
| 51 | 16 django/contrib/auth/migrations/0001_initial.py | 1 | 205| 1007 | 21869 | 97733 | 
| 52 | 16 django/db/backends/sqlite3/schema.py | 256 | 361| 885 | 22754 | 97733 | 
| 53 | **16 django/db/migrations/operations/models.py** | 1 | 18| 137 | 22891 | 97733 | 
| 54 | 16 django/db/migrations/state.py | 265 | 289| 238 | 23129 | 97733 | 
| 55 | 17 django/contrib/admin/migrations/0001_initial.py | 1 | 76| 363 | 23492 | 98096 | 
| 56 | 17 django/db/migrations/autodetector.py | 234 | 265| 256 | 23748 | 98096 | 
| 57 | 17 django/db/models/fields/related.py | 302 | 339| 296 | 24044 | 98096 | 
| 58 | 18 django/contrib/sites/migrations/0001_initial.py | 1 | 44| 210 | 24254 | 98306 | 
| 59 | 18 django/db/models/fields/related.py | 1571 | 1668| 655 | 24909 | 98306 | 
| 60 | 18 django/db/migrations/state.py | 240 | 263| 247 | 25156 | 98306 | 
| 61 | 19 django/core/serializers/base.py | 327 | 359| 227 | 25383 | 100843 | 
| 62 | 19 django/db/models/base.py | 1839 | 1874| 246 | 25629 | 100843 | 
| 63 | 20 django/contrib/contenttypes/management/__init__.py | 1 | 43| 358 | 25987 | 101831 | 
| 64 | 21 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 43| 232 | 26219 | 102063 | 
| 65 | 21 django/db/models/base.py | 1790 | 1813| 180 | 26399 | 102063 | 
| 66 | 21 django/core/management/commands/migrate.py | 369 | 390| 204 | 26603 | 102063 | 
| 67 | 21 django/db/migrations/state.py | 460 | 486| 150 | 26753 | 102063 | 
| 68 | 21 django/core/management/commands/migrate.py | 1 | 14| 134 | 26887 | 102063 | 
| 69 | 22 django/core/management/commands/showmigrations.py | 56 | 77| 158 | 27045 | 103359 | 
| 70 | 22 django/db/migrations/autodetector.py | 1393 | 1426| 289 | 27334 | 103359 | 
| 71 | 22 django/db/models/fields/related.py | 1920 | 1947| 305 | 27639 | 103359 | 
| 72 | 22 django/core/management/commands/makemigrations.py | 196 | 259| 458 | 28097 | 103359 | 
| 73 | 22 django/db/migrations/questioner.py | 57 | 87| 255 | 28352 | 103359 | 
| 74 | 22 django/db/migrations/autodetector.py | 1500 | 1528| 235 | 28587 | 103359 | 
| 75 | **22 django/db/migrations/operations/models.py** | 41 | 111| 524 | 29111 | 103359 | 
| 76 | 22 django/db/migrations/autodetector.py | 1342 | 1363| 197 | 29308 | 103359 | 
| 77 | 22 django/core/management/commands/makemigrations.py | 1 | 23| 185 | 29493 | 103359 | 
| 78 | 22 django/db/backends/sqlite3/schema.py | 402 | 428| 256 | 29749 | 103359 | 
| 79 | 22 django/db/migrations/autodetector.py | 1475 | 1498| 161 | 29910 | 103359 | 
| 80 | **22 django/db/migrations/operations/models.py** | 916 | 969| 348 | 30258 | 103359 | 
| 81 | 22 django/db/backends/sqlite3/schema.py | 122 | 173| 527 | 30785 | 103359 | 
| 82 | 22 django/db/migrations/state.py | 181 | 238| 598 | 31383 | 103359 | 
| 83 | 22 django/db/migrations/state.py | 973 | 989| 140 | 31523 | 103359 | 
| 84 | 22 django/core/management/commands/makemigrations.py | 261 | 331| 572 | 32095 | 103359 | 
| 85 | 23 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 24| 117 | 32212 | 103476 | 
| 86 | 24 django/contrib/flatpages/migrations/0001_initial.py | 1 | 69| 355 | 32567 | 103831 | 
| 87 | 25 django/core/management/commands/squashmigrations.py | 255 | 268| 112 | 32679 | 105871 | 
| 88 | 25 django/db/migrations/autodetector.py | 1216 | 1303| 719 | 33398 | 105871 | 
| 89 | **25 django/db/migrations/operations/models.py** | 532 | 567| 290 | 33688 | 105871 | 
| 90 | 25 django/db/models/base.py | 2025 | 2078| 359 | 34047 | 105871 | 
| 91 | 25 django/db/migrations/autodetector.py | 1023 | 1072| 384 | 34431 | 105871 | 
| 92 | 26 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 46| 225 | 34656 | 106096 | 
| 93 | **26 django/db/migrations/operations/models.py** | 21 | 38| 116 | 34772 | 106096 | 
| 94 | 26 django/core/management/commands/squashmigrations.py | 161 | 253| 766 | 35538 | 106096 | 
| 95 | 26 django/db/migrations/state.py | 126 | 140| 163 | 35701 | 106096 | 
| 96 | 26 django/db/migrations/state.py | 872 | 904| 250 | 35951 | 106096 | 
| 97 | 26 django/db/migrations/autodetector.py | 1305 | 1340| 232 | 36183 | 106096 | 
| 98 | 26 django/db/models/fields/related.py | 208 | 224| 142 | 36325 | 106096 | 
| 99 | 26 django/db/models/base.py | 1069 | 1121| 520 | 36845 | 106096 | 
| 100 | 26 django/db/models/base.py | 459 | 572| 957 | 37802 | 106096 | 
| 101 | 26 django/db/models/fields/related.py | 226 | 301| 696 | 38498 | 106096 | 
| 102 | 27 django/db/migrations/recorder.py | 1 | 22| 148 | 38646 | 106783 | 
| 103 | **27 django/db/migrations/operations/models.py** | 597 | 606| 129 | 38775 | 106783 | 
| 104 | 27 django/db/models/fields/related.py | 1722 | 1760| 413 | 39188 | 106783 | 
| 105 | 27 django/db/backends/sqlite3/schema.py | 99 | 120| 195 | 39383 | 106783 | 
| 106 | 27 django/db/models/base.py | 1 | 66| 361 | 39744 | 106783 | 
| 107 | 27 django/db/models/base.py | 1735 | 1788| 494 | 40238 | 106783 | 
| 108 | 27 django/db/backends/base/schema.py | 544 | 563| 196 | 40434 | 106783 | 
| 109 | 28 django/db/migrations/migration.py | 200 | 240| 292 | 40726 | 108690 | 
| 110 | **28 django/db/migrations/operations/models.py** | 1106 | 1146| 337 | 41063 | 108690 | 
| 111 | 28 django/db/backends/base/schema.py | 745 | 775| 293 | 41356 | 108690 | 
| 112 | 28 django/db/models/base.py | 1398 | 1428| 220 | 41576 | 108690 | 
| 113 | 28 django/db/migrations/autodetector.py | 281 | 379| 806 | 42382 | 108690 | 
| 114 | 28 django/db/migrations/autodetector.py | 1074 | 1095| 188 | 42570 | 108690 | 
| 115 | 28 django/db/migrations/questioner.py | 291 | 342| 367 | 42937 | 108690 | 
| 116 | 28 django/db/models/base.py | 1123 | 1150| 238 | 43175 | 108690 | 
| 117 | 28 django/db/migrations/operations/fields.py | 227 | 237| 146 | 43321 | 108690 | 
| 118 | 28 django/db/models/base.py | 1622 | 1647| 188 | 43509 | 108690 | 
| 119 | 28 django/db/backends/sqlite3/schema.py | 430 | 487| 489 | 43998 | 108690 | 
| 120 | 28 django/db/migrations/operations/fields.py | 339 | 358| 153 | 44151 | 108690 | 
| 121 | 29 django/db/backends/mysql/schema.py | 46 | 55| 134 | 44285 | 110762 | 
| 122 | 29 django/db/backends/sqlite3/schema.py | 363 | 379| 132 | 44417 | 110762 | 
| 123 | 29 django/core/management/commands/makemigrations.py | 405 | 515| 927 | 45344 | 110762 | 
| 124 | 29 django/db/backends/base/schema.py | 1566 | 1588| 199 | 45543 | 110762 | 
| 125 | 29 django/core/management/commands/makemigrations.py | 26 | 102| 446 | 45989 | 110762 | 
| 126 | 29 django/db/models/base.py | 250 | 367| 874 | 46863 | 110762 | 
| 127 | 30 django/contrib/sessions/migrations/0001_initial.py | 1 | 38| 173 | 47036 | 110935 | 
| 128 | 30 django/core/management/commands/squashmigrations.py | 62 | 159| 809 | 47845 | 110935 | 
| 129 | 30 django/db/migrations/operations/fields.py | 239 | 267| 206 | 48051 | 110935 | 
| 130 | 30 django/db/migrations/autodetector.py | 399 | 415| 141 | 48192 | 110935 | 
| 131 | 30 django/db/backends/base/schema.py | 1524 | 1543| 175 | 48367 | 110935 | 
| 132 | 30 django/core/management/commands/sqlmigrate.py | 1 | 38| 276 | 48643 | 110935 | 
| 133 | **30 django/db/migrations/operations/models.py** | 782 | 816| 225 | 48868 | 110935 | 
| 134 | 30 django/db/migrations/autodetector.py | 1663 | 1713| 439 | 49307 | 110935 | 
| 135 | 30 django/db/models/base.py | 1301 | 1348| 413 | 49720 | 110935 | 
| 136 | 30 django/db/migrations/recorder.py | 48 | 104| 400 | 50120 | 110935 | 
| 137 | 31 django/db/models/options.py | 1 | 58| 353 | 50473 | 118631 | 
| 138 | 31 django/db/migrations/autodetector.py | 1365 | 1391| 144 | 50617 | 118631 | 
| 139 | 31 django/db/migrations/autodetector.py | 381 | 397| 161 | 50778 | 118631 | 
| 140 | 31 django/db/migrations/state.py | 397 | 409| 136 | 50914 | 118631 | 
| 141 | 31 django/core/management/commands/migrate.py | 17 | 94| 487 | 51401 | 118631 | 
| 142 | 31 django/db/migrations/state.py | 488 | 510| 225 | 51626 | 118631 | 
| 143 | 32 django/core/management/commands/inspectdb.py | 265 | 323| 490 | 52116 | 121702 | 
| 144 | 32 django/db/migrations/autodetector.py | 1 | 18| 113 | 52229 | 121702 | 
| 145 | 32 django/db/migrations/autodetector.py | 1573 | 1611| 304 | 52533 | 121702 | 
| 146 | 32 django/db/models/fields/related.py | 1244 | 1298| 421 | 52954 | 121702 | 
| 147 | 32 django/db/migrations/autodetector.py | 480 | 510| 267 | 53221 | 121702 | 
| 148 | 33 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 54029 | 133360 | 
| 149 | 34 django/core/serializers/python.py | 64 | 91| 198 | 54227 | 134705 | 
| 150 | 34 django/db/backends/base/schema.py | 639 | 657| 162 | 54389 | 134705 | 
| 151 | **34 django/db/migrations/operations/models.py** | 727 | 779| 320 | 54709 | 134705 | 
| 152 | **34 django/db/migrations/operations/models.py** | 608 | 632| 213 | 54922 | 134705 | 
| 153 | 34 django/db/migrations/autodetector.py | 774 | 805| 278 | 55200 | 134705 | 
| 154 | 35 django/forms/models.py | 494 | 524| 247 | 55447 | 146915 | 
| 155 | 35 django/db/backends/mysql/schema.py | 1 | 44| 484 | 55931 | 146915 | 
| 156 | 35 django/core/management/commands/migrate.py | 432 | 487| 409 | 56340 | 146915 | 
| 157 | 35 django/db/migrations/operations/fields.py | 154 | 195| 348 | 56688 | 146915 | 
| 158 | 36 django/db/backends/mysql/operations.py | 436 | 465| 274 | 56962 | 151093 | 
| 159 | 36 django/db/migrations/operations/fields.py | 101 | 113| 130 | 57092 | 151093 | 
| 160 | 37 django/db/migrations/utils.py | 27 | 50| 194 | 57286 | 151964 | 
| 161 | 37 django/db/migrations/operations/fields.py | 115 | 127| 131 | 57417 | 151964 | 
| 162 | 37 django/db/migrations/state.py | 958 | 971| 138 | 57555 | 151964 | 
| 163 | 37 django/db/models/fields/related.py | 1016 | 1052| 261 | 57816 | 151964 | 
| 164 | **37 django/db/migrations/operations/models.py** | 309 | 341| 249 | 58065 | 151964 | 
| 165 | 37 django/db/migrations/state.py | 711 | 765| 472 | 58537 | 151964 | 
| 166 | 38 django/core/management/commands/optimizemigration.py | 1 | 130| 940 | 59477 | 152904 | 
| 167 | 38 django/db/models/base.py | 574 | 612| 326 | 59803 | 152904 | 
| 168 | 39 django/db/utils.py | 237 | 279| 322 | 60125 | 154803 | 
| 169 | 39 django/db/migrations/utils.py | 1 | 24| 134 | 60259 | 154803 | 
| 170 | 39 django/core/management/commands/inspectdb.py | 54 | 263| 1616 | 61875 | 154803 | 
| 171 | 39 django/db/migrations/state.py | 347 | 395| 371 | 62246 | 154803 | 
| 172 | 40 django/db/models/__init__.py | 1 | 116| 682 | 62928 | 155485 | 
| 173 | 40 django/db/migrations/state.py | 33 | 65| 243 | 63171 | 155485 | 
| 174 | 40 django/db/models/base.py | 1381 | 1396| 142 | 63313 | 155485 | 
| 175 | 40 django/db/models/base.py | 1649 | 1678| 209 | 63522 | 155485 | 
| 176 | 40 django/db/models/fields/related.py | 154 | 185| 209 | 63731 | 155485 | 
| 177 | 40 django/db/models/fields/related.py | 1793 | 1818| 222 | 63953 | 155485 | 
| 178 | 40 django/db/models/fields/related.py | 187 | 206| 155 | 64108 | 155485 | 
| 179 | 40 django/db/models/fields/related.py | 580 | 600| 138 | 64246 | 155485 | 
| 180 | 40 django/db/migrations/state.py | 411 | 435| 213 | 64459 | 155485 | 
| 181 | 40 django/db/models/fields/related_descriptors.py | 1457 | 1507| 407 | 64866 | 155485 | 
| 182 | 40 django/db/migrations/autodetector.py | 90 | 102| 119 | 64985 | 155485 | 
| 183 | 41 django/db/migrations/serializer.py | 223 | 258| 281 | 65266 | 158253 | 
| 184 | 42 django/db/backends/postgresql/schema.py | 142 | 255| 920 | 66186 | 161090 | 
| 185 | 42 django/db/models/fields/related.py | 1085 | 1102| 133 | 66319 | 161090 | 
| 186 | 43 django/core/management/base.py | 566 | 606| 293 | 66612 | 165950 | 
| 187 | **43 django/db/migrations/operations/models.py** | 659 | 680| 162 | 66774 | 165950 | 
| 188 | 43 django/db/models/fields/related.py | 1 | 40| 251 | 67025 | 165950 | 
| 189 | 43 django/db/backends/base/schema.py | 841 | 936| 799 | 67824 | 165950 | 
| 190 | 43 django/db/models/fields/related.py | 670 | 703| 335 | 68159 | 165950 | 
| 191 | 44 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 27| 143 | 68302 | 166093 | 
| 192 | 44 django/db/models/fields/related_descriptors.py | 1133 | 1151| 174 | 68476 | 166093 | 


### Hint

```
Thanks for the report.
Mariusz , if the names of models are same (but different apps), are there any expected names of the cloumns for m2m db table? Can it be incident_id and incidents_incident_id since two db columns cannot have same name?
I did some further testing and you don't encounter the same issue going in the other direction. For example, in the demo project if I rename incidents.Incident the migration generated works fine and properly updates the M2M on vault.Incident. The M2M table is updated and the from/to are removed from both. After that you can do a rename on the model with the M2M. Just an FYI for anyone who might be looking for a work around.
Thanks zac! I did some digging in the code and found out that even though the field names are from/to_incident , they are referring to the correct models: CREATE TABLE "vault_incident" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "name" varchar(10) NOT NULL); CREATE TABLE "vault_incident_incidents" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "from_incident_id" bigint NOT NULL REFERENCES "vault_incident" ("id") DEFERRABLE INITIALLY DEFERRED, "to_incident_id" bigint NOT NULL REFERENCE S "incidents_incident" ("id") DEFERRABLE INITIALLY DEFERRED); The problem is caused in renaming fields of m2m model ​here. One of the solutions can be checking app names + model name together and if model names are same but apps are different the name of fields can be (like in above case) incident_id and incidents_incident_id. Feel free to share if anyone has a better solution. Thanks!
​Draft PR
Hi, the PR is ready for review !
Per ​Simon's comment.
```

## Patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -407,20 +407,12 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
                     or not new_field.remote_field.through._meta.auto_created
                 ):
                     continue
-                # Rename the M2M table that's based on this model's name.
-                old_m2m_model = old_field.remote_field.through
-                new_m2m_model = new_field.remote_field.through
-                schema_editor.alter_db_table(
-                    new_m2m_model,
-                    old_m2m_model._meta.db_table,
-                    new_m2m_model._meta.db_table,
-                )
-                # Rename the column in the M2M table that's based on this
-                # model's name.
-                schema_editor.alter_field(
-                    new_m2m_model,
-                    old_m2m_model._meta.get_field(old_model._meta.model_name),
-                    new_m2m_model._meta.get_field(new_model._meta.model_name),
+                # Rename columns and the M2M table.
+                schema_editor._alter_many_to_many(
+                    new_model,
+                    old_field,
+                    new_field,
+                    strict=False,
                 )
 
     def database_backwards(self, app_label, schema_editor, from_state, to_state):

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -1058,6 +1058,75 @@ def test_rename_model_with_m2m(self):
             Pony._meta.get_field("riders").remote_field.through.objects.count(), 2
         )
 
+    def test_rename_model_with_m2m_models_in_different_apps_with_same_name(self):
+        app_label_1 = "test_rmw_m2m_1"
+        app_label_2 = "test_rmw_m2m_2"
+        project_state = self.apply_operations(
+            app_label_1,
+            ProjectState(),
+            operations=[
+                migrations.CreateModel(
+                    "Rider",
+                    fields=[
+                        ("id", models.AutoField(primary_key=True)),
+                    ],
+                ),
+            ],
+        )
+        project_state = self.apply_operations(
+            app_label_2,
+            project_state,
+            operations=[
+                migrations.CreateModel(
+                    "Rider",
+                    fields=[
+                        ("id", models.AutoField(primary_key=True)),
+                        ("riders", models.ManyToManyField(f"{app_label_1}.Rider")),
+                    ],
+                ),
+            ],
+        )
+        m2m_table = f"{app_label_2}_rider_riders"
+        self.assertColumnExists(m2m_table, "from_rider_id")
+        self.assertColumnExists(m2m_table, "to_rider_id")
+
+        Rider_1 = project_state.apps.get_model(app_label_1, "Rider")
+        Rider_2 = project_state.apps.get_model(app_label_2, "Rider")
+        rider_2 = Rider_2.objects.create()
+        rider_2.riders.add(Rider_1.objects.create())
+        # Rename model.
+        project_state_2 = project_state.clone()
+        project_state = self.apply_operations(
+            app_label_2,
+            project_state,
+            operations=[migrations.RenameModel("Rider", "Pony")],
+            atomic=connection.features.supports_atomic_references_rename,
+        )
+
+        m2m_table = f"{app_label_2}_pony_riders"
+        self.assertColumnExists(m2m_table, "pony_id")
+        self.assertColumnExists(m2m_table, "rider_id")
+
+        Rider_1 = project_state.apps.get_model(app_label_1, "Rider")
+        Rider_2 = project_state.apps.get_model(app_label_2, "Pony")
+        rider_2 = Rider_2.objects.create()
+        rider_2.riders.add(Rider_1.objects.create())
+        self.assertEqual(Rider_1.objects.count(), 2)
+        self.assertEqual(Rider_2.objects.count(), 2)
+        self.assertEqual(
+            Rider_2._meta.get_field("riders").remote_field.through.objects.count(), 2
+        )
+        # Reversal.
+        self.unapply_operations(
+            app_label_2,
+            project_state_2,
+            operations=[migrations.RenameModel("Rider", "Pony")],
+            atomic=connection.features.supports_atomic_references_rename,
+        )
+        m2m_table = f"{app_label_2}_rider_riders"
+        self.assertColumnExists(m2m_table, "to_rider_id")
+        self.assertColumnExists(m2m_table, "from_rider_id")
+
     def test_rename_model_with_db_table_rename_m2m(self):
         app_label = "test_rmwdbrm2m"
         project_state = self.apply_operations(

```


## Code snippets

### 1 - django/db/migrations/operations/models.py:

Start line: 370, End line: 424

```python
class RenameModel(ModelOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.new_name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.old_name)
            # Move the main table
            schema_editor.alter_db_table(
                new_model,
                old_model._meta.db_table,
                new_model._meta.db_table,
            )
            # Alter the fields pointing to us
            for related_object in old_model._meta.related_objects:
                if related_object.related_model == old_model:
                    model = new_model
                    related_key = (app_label, self.new_name_lower)
                else:
                    model = related_object.related_model
                    related_key = (
                        related_object.related_model._meta.app_label,
                        related_object.related_model._meta.model_name,
                    )
                to_field = to_state.apps.get_model(*related_key)._meta.get_field(
                    related_object.field.name
                )
                schema_editor.alter_field(
                    model,
                    related_object.field,
                    to_field,
                )
            # Rename M2M fields whose name is based on this model's name.
            fields = zip(
                old_model._meta.local_many_to_many, new_model._meta.local_many_to_many
            )
            for old_field, new_field in fields:
                # Skip self-referential fields as these are renamed above.
                if (
                    new_field.model == new_field.related_model
                    or not new_field.remote_field.through._meta.auto_created
                ):
                    continue
                # Rename the M2M table that's based on this model's name.
                old_m2m_model = old_field.remote_field.through
                new_m2m_model = new_field.remote_field.through
                schema_editor.alter_db_table(
                    new_m2m_model,
                    old_m2m_model._meta.db_table,
                    new_m2m_model._meta.db_table,
                )
                # Rename the column in the M2M table that's based on this
                # model's name.
                schema_editor.alter_field(
                    new_m2m_model,
                    old_m2m_model._meta.get_field(old_model._meta.model_name),
                    new_m2m_model._meta.get_field(new_model._meta.model_name),
                )
```
### 2 - django/db/migrations/operations/models.py:

Start line: 426, End line: 439

```python
class RenameModel(ModelOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name

        self.database_forwards(app_label, schema_editor, from_state, to_state)

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name
```
### 3 - django/db/migrations/autodetector.py:

Start line: 512, End line: 577

```python
class MigrationAutodetector:

    def generate_renamed_models(self):
        """
        Find any renamed models, generate the operations for them, and remove
        the old entry from the model lists. Must be run before other
        model-level generation.
        """
        self.renamed_models = {}
        self.renamed_models_rel = {}
        added_models = self.new_model_keys - self.old_model_keys
        for app_label, model_name in sorted(added_models):
            model_state = self.to_state.models[app_label, model_name]
            model_fields_def = self.only_relation_agnostic_fields(model_state.fields)

            removed_models = self.old_model_keys - self.new_model_keys
            for rem_app_label, rem_model_name in removed_models:
                if rem_app_label == app_label:
                    rem_model_state = self.from_state.models[
                        rem_app_label, rem_model_name
                    ]
                    rem_model_fields_def = self.only_relation_agnostic_fields(
                        rem_model_state.fields
                    )
                    if model_fields_def == rem_model_fields_def:
                        if self.questioner.ask_rename_model(
                            rem_model_state, model_state
                        ):
                            dependencies = []
                            fields = list(model_state.fields.values()) + [
                                field.remote_field
                                for relations in self.to_state.relations[
                                    app_label, model_name
                                ].values()
                                for field in relations.values()
                            ]
                            for field in fields:
                                if field.is_relation:
                                    dependencies.extend(
                                        self._get_dependencies_for_foreign_key(
                                            app_label,
                                            model_name,
                                            field,
                                            self.to_state,
                                        )
                                    )
                            self.add_operation(
                                app_label,
                                operations.RenameModel(
                                    old_name=rem_model_state.name,
                                    new_name=model_state.name,
                                ),
                                dependencies=dependencies,
                            )
                            self.renamed_models[app_label, model_name] = rem_model_name
                            renamed_models_rel_key = "%s.%s" % (
                                rem_model_state.app_label,
                                rem_model_state.name_lower,
                            )
                            self.renamed_models_rel[
                                renamed_models_rel_key
                            ] = "%s.%s" % (
                                model_state.app_label,
                                model_state.name_lower,
                            )
                            self.old_model_keys.remove((rem_app_label, rem_model_name))
                            self.old_model_keys.add((app_label, model_name))
                            break
```
### 4 - django/db/migrations/state.py:

Start line: 291, End line: 345

```python
class ProjectState:

    def rename_field(self, app_label, model_name, old_name, new_name):
        model_key = app_label, model_name
        model_state = self.models[model_key]
        # Rename the field.
        fields = model_state.fields
        try:
            found = fields.pop(old_name)
        except KeyError:
            raise FieldDoesNotExist(
                f"{app_label}.{model_name} has no field named '{old_name}'"
            )
        fields[new_name] = found
        for field in fields.values():
            # Fix from_fields to refer to the new field.
            from_fields = getattr(field, "from_fields", None)
            if from_fields:
                field.from_fields = tuple(
                    [
                        new_name if from_field_name == old_name else from_field_name
                        for from_field_name in from_fields
                    ]
                )
        # Fix index/unique_together to refer to the new field.
        options = model_state.options
        for option in ("index_together", "unique_together"):
            if option in options:
                options[option] = [
                    [new_name if n == old_name else n for n in together]
                    for together in options[option]
                ]
        # Fix to_fields to refer to the new field.
        delay = True
        references = get_references(self, model_key, (old_name, found))
        for *_, field, reference in references:
            delay = False
            if reference.to:
                remote_field, to_fields = reference.to
                if getattr(remote_field, "field_name", None) == old_name:
                    remote_field.field_name = new_name
                if to_fields:
                    field.to_fields = tuple(
                        [
                            new_name if to_field_name == old_name else to_field_name
                            for to_field_name in to_fields
                        ]
                    )
        if self._relations is not None:
            old_name_lower = old_name.lower()
            new_name_lower = new_name.lower()
            for to_model in self._relations.values():
                if old_name_lower in to_model[model_key]:
                    field = to_model[model_key].pop(old_name_lower)
                    field.name = new_name_lower
                    to_model[model_key][new_name_lower] = field
        self.reload_model(*model_key, delay=delay)
```
### 5 - django/db/backends/sqlite3/schema.py:

Start line: 489, End line: 551

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        if (
            old_field.remote_field.through._meta.db_table
            == new_field.remote_field.through._meta.db_table
        ):
            # The field name didn't change, but some options did, so we have to
            # propagate this altering.
            self._remake_table(
                old_field.remote_field.through,
                alter_fields=[
                    (
                        # The field that points to the target model is needed,
                        # so that table can be remade with the new m2m field -
                        # this is m2m_reverse_field_name().
                        old_field.remote_field.through._meta.get_field(
                            old_field.m2m_reverse_field_name()
                        ),
                        new_field.remote_field.through._meta.get_field(
                            new_field.m2m_reverse_field_name()
                        ),
                    ),
                    (
                        # The field that points to the model itself is needed,
                        # so that table can be remade with the new self field -
                        # this is m2m_field_name().
                        old_field.remote_field.through._meta.get_field(
                            old_field.m2m_field_name()
                        ),
                        new_field.remote_field.through._meta.get_field(
                            new_field.m2m_field_name()
                        ),
                    ),
                ],
            )
            return

        # Make a new through table
        self.create_model(new_field.remote_field.through)
        # Copy the data across
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_field.remote_field.through._meta.db_table),
                ", ".join(
                    [
                        "id",
                        new_field.m2m_column_name(),
                        new_field.m2m_reverse_name(),
                    ]
                ),
                ", ".join(
                    [
                        "id",
                        old_field.m2m_column_name(),
                        old_field.m2m_reverse_name(),
                    ]
                ),
                self.quote_name(old_field.remote_field.through._meta.db_table),
            )
        )
        # Delete the old through table
        self.delete_model(old_field.remote_field.through)
```
### 6 - django/db/migrations/operations/models.py:

Start line: 441, End line: 478

```python
class RenameModel(ModelOperation):

    def references_model(self, name, app_label):
        return (
            name.lower() == self.old_name_lower or name.lower() == self.new_name_lower
        )

    def describe(self):
        return "Rename model %s to %s" % (self.old_name, self.new_name)

    @property
    def migration_name_fragment(self):
        return "rename_%s_%s" % (self.old_name_lower, self.new_name_lower)

    def reduce(self, operation, app_label):
        if (
            isinstance(operation, RenameModel)
            and self.new_name_lower == operation.old_name_lower
        ):
            return [
                RenameModel(
                    self.old_name,
                    operation.new_name,
                ),
            ]
        # Skip `ModelOperation.reduce` as we want to run `references_model`
        # against self.new_name.
        return super(ModelOperation, self).reduce(
            operation, app_label
        ) or not operation.references_model(self.new_name, app_label)


class ModelOptionOperation(ModelOperation):
    def reduce(self, operation, app_label):
        if (
            isinstance(operation, (self.__class__, DeleteModel))
            and self.name_lower == operation.name_lower
        ):
            return [operation]
        return super().reduce(operation, app_label)
```
### 7 - django/db/migrations/state.py:

Start line: 142, End line: 179

```python
class ProjectState:

    def rename_model(self, app_label, old_name, new_name):
        # Add a new model.
        old_name_lower = old_name.lower()
        new_name_lower = new_name.lower()
        renamed_model = self.models[app_label, old_name_lower].clone()
        renamed_model.name = new_name
        self.models[app_label, new_name_lower] = renamed_model
        # Repoint all fields pointing to the old model to the new one.
        old_model_tuple = (app_label, old_name_lower)
        new_remote_model = f"{app_label}.{new_name}"
        to_reload = set()
        for model_state, name, field, reference in get_references(
            self, old_model_tuple
        ):
            changed_field = None
            if reference.to:
                changed_field = field.clone()
                changed_field.remote_field.model = new_remote_model
            if reference.through:
                if changed_field is None:
                    changed_field = field.clone()
                changed_field.remote_field.through = new_remote_model
            if changed_field:
                model_state.fields[name] = changed_field
                to_reload.add((model_state.app_label, model_state.name_lower))
        if self._relations is not None:
            old_name_key = app_label, old_name_lower
            new_name_key = app_label, new_name_lower
            if old_name_key in self._relations:
                self._relations[new_name_key] = self._relations.pop(old_name_key)
            for model_relations in self._relations.values():
                if old_name_key in model_relations:
                    model_relations[new_name_key] = model_relations.pop(old_name_key)
        # Reload models related to old model before removing the old model.
        self.reload_models(to_reload, delay=True)
        # Remove the old model.
        self.remove_model(app_label, old_name_lower)
        self.reload_model(app_label, new_name_lower, delay=True)
```
### 8 - django/db/models/base.py:

Start line: 1680, End line: 1714

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_m2m_through_same_relationship(cls):
        """Check if no relationship model is used by more than one m2m field."""

        errors = []
        seen_intermediary_signatures = []

        fields = cls._meta.local_many_to_many

        # Skip when the target model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.model, ModelBase))

        # Skip when the relationship model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.through, ModelBase))

        for f in fields:
            signature = (
                f.remote_field.model,
                cls,
                f.remote_field.through,
                f.remote_field.through_fields,
            )
            if signature in seen_intermediary_signatures:
                errors.append(
                    checks.Error(
                        "The model has two identical many-to-many relations "
                        "through the intermediate model '%s'."
                        % f.remote_field.through._meta.label,
                        obj=cls,
                        id="models.E003",
                    )
                )
            else:
                seen_intermediary_signatures.append(signature)
        return errors
```
### 9 - django/db/migrations/operations/fields.py:

Start line: 270, End line: 337

```python
class RenameField(FieldOperation):
    """Rename a field on the model. Might affect db_column too."""

    def __init__(self, model_name, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        super().__init__(model_name, old_name)

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {
            "model_name": self.model_name,
            "old_name": self.old_name,
            "new_name": self.new_name,
        }
        return (self.__class__.__name__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.rename_field(
            app_label, self.model_name_lower, self.old_name, self.new_name
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            schema_editor.alter_field(
                from_model,
                from_model._meta.get_field(self.old_name),
                to_model._meta.get_field(self.new_name),
            )

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            schema_editor.alter_field(
                from_model,
                from_model._meta.get_field(self.new_name),
                to_model._meta.get_field(self.old_name),
            )

    def describe(self):
        return "Rename field %s on %s to %s" % (
            self.old_name,
            self.model_name,
            self.new_name,
        )

    @property
    def migration_name_fragment(self):
        return "rename_%s_%s_%s" % (
            self.old_name_lower,
            self.model_name_lower,
            self.new_name_lower,
        )

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (
            name.lower() == self.old_name_lower or name.lower() == self.new_name_lower
        )
```
### 10 - django/db/migrations/operations/models.py:

Start line: 344, End line: 368

```python
class RenameModel(ModelOperation):
    """Rename a model."""

    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        super().__init__(old_name)

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {
            "old_name": self.old_name,
            "new_name": self.new_name,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.rename_model(app_label, self.old_name, self.new_name)
```
### 11 - django/db/migrations/operations/models.py:

Start line: 481, End line: 529

```python
class AlterModelTable(ModelOptionOperation):
    """Rename a model's table."""

    def __init__(self, name, table):
        self.table = table
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "table": self.table,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(app_label, self.name_lower, {"db_table": self.table})

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.name)
            schema_editor.alter_db_table(
                new_model,
                old_model._meta.db_table,
                new_model._meta.db_table,
            )
            # Rename M2M fields whose name is based on this model's db_table
            for old_field, new_field in zip(
                old_model._meta.local_many_to_many, new_model._meta.local_many_to_many
            ):
                if new_field.remote_field.through._meta.auto_created:
                    schema_editor.alter_db_table(
                        new_field.remote_field.through,
                        old_field.remote_field.through._meta.db_table,
                        new_field.remote_field.through._meta.db_table,
                    )

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        return self.database_forwards(app_label, schema_editor, from_state, to_state)

    def describe(self):
        return "Rename table for %s to %s" % (
            self.name,
            self.table if self.table is not None else "(default)",
        )

    @property
    def migration_name_fragment(self):
        return "alter_%s_table" % self.name_lower
```
### 22 - django/db/migrations/operations/models.py:

Start line: 971, End line: 1006

```python
class RenameIndex(IndexOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if not self.allow_migrate_model(schema_editor.connection.alias, model):
            return

        if self.old_fields:
            from_model = from_state.apps.get_model(app_label, self.model_name)
            columns = [
                from_model._meta.get_field(field).column for field in self.old_fields
            ]
            matching_index_name = schema_editor._constraint_names(
                from_model, column_names=columns, index=True
            )
            if len(matching_index_name) != 1:
                raise ValueError(
                    "Found wrong number (%s) of indexes for %s(%s)."
                    % (
                        len(matching_index_name),
                        from_model._meta.db_table,
                        ", ".join(columns),
                    )
                )
            old_index = models.Index(
                fields=self.old_fields,
                name=matching_index_name[0],
            )
        else:
            from_model_state = from_state.models[app_label, self.model_name_lower]
            old_index = from_model_state.get_index_by_name(self.old_name)
        # Don't alter when the index name is not changed.
        if old_index.name == self.new_name:
            return

        to_model_state = to_state.models[app_label, self.model_name_lower]
        new_index = to_model_state.get_index_by_name(self.new_name)
        schema_editor.rename_index(model, old_index, new_index)
```
### 31 - django/db/migrations/operations/models.py:

Start line: 136, End line: 306

```python
class CreateModel(ModelOperation):

    def reduce(self, operation, app_label):
        if (
            isinstance(operation, DeleteModel)
            and self.name_lower == operation.name_lower
            and not self.options.get("proxy", False)
        ):
            return []
        elif (
            isinstance(operation, RenameModel)
            and self.name_lower == operation.old_name_lower
        ):
            return [
                CreateModel(
                    operation.new_name,
                    fields=self.fields,
                    options=self.options,
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif (
            isinstance(operation, AlterModelOptions)
            and self.name_lower == operation.name_lower
        ):
            options = {**self.options, **operation.options}
            for key in operation.ALTER_OPTION_KEYS:
                if key not in operation.options:
                    options.pop(key, None)
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options=options,
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif (
            isinstance(operation, AlterModelManagers)
            and self.name_lower == operation.name_lower
        ):
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options=self.options,
                    bases=self.bases,
                    managers=operation.managers,
                ),
            ]
        elif (
            isinstance(operation, AlterTogetherOptionOperation)
            and self.name_lower == operation.name_lower
        ):
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options={
                        **self.options,
                        **{operation.option_name: operation.option_value},
                    },
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif (
            isinstance(operation, AlterOrderWithRespectTo)
            and self.name_lower == operation.name_lower
        ):
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options={
                        **self.options,
                        "order_with_respect_to": operation.order_with_respect_to,
                    },
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif (
            isinstance(operation, FieldOperation)
            and self.name_lower == operation.model_name_lower
        ):
            if isinstance(operation, AddField):
                return [
                    CreateModel(
                        self.name,
                        fields=self.fields + [(operation.name, operation.field)],
                        options=self.options,
                        bases=self.bases,
                        managers=self.managers,
                    ),
                ]
            elif isinstance(operation, AlterField):
                return [
                    CreateModel(
                        self.name,
                        fields=[
                            (n, operation.field if n == operation.name else v)
                            for n, v in self.fields
                        ],
                        options=self.options,
                        bases=self.bases,
                        managers=self.managers,
                    ),
                ]
            elif isinstance(operation, RemoveField):
                options = self.options.copy()
                for option_name in ("unique_together", "index_together"):
                    option = options.pop(option_name, None)
                    if option:
                        option = set(
                            filter(
                                bool,
                                (
                                    tuple(
                                        f for f in fields if f != operation.name_lower
                                    )
                                    for fields in option
                                ),
                            )
                        )
                        if option:
                            options[option_name] = option
                order_with_respect_to = options.get("order_with_respect_to")
                if order_with_respect_to == operation.name_lower:
                    del options["order_with_respect_to"]
                return [
                    CreateModel(
                        self.name,
                        fields=[
                            (n, v)
                            for n, v in self.fields
                            if n.lower() != operation.name_lower
                        ],
                        options=options,
                        bases=self.bases,
                        managers=self.managers,
                    ),
                ]
            elif isinstance(operation, RenameField):
                options = self.options.copy()
                for option_name in ("unique_together", "index_together"):
                    option = options.get(option_name)
                    if option:
                        options[option_name] = {
                            tuple(
                                operation.new_name if f == operation.old_name else f
                                for f in fields
                            )
                            for fields in option
                        }
                order_with_respect_to = options.get("order_with_respect_to")
                if order_with_respect_to == operation.old_name:
                    options["order_with_respect_to"] = operation.new_name
                return [
                    CreateModel(
                        self.name,
                        fields=[
                            (operation.new_name if n == operation.old_name else n, v)
                            for n, v in self.fields
                        ],
                        options=options,
                        bases=self.bases,
                        managers=self.managers,
                    ),
                ]
        return super().reduce(operation, app_label)
```
### 33 - django/db/migrations/operations/models.py:

Start line: 1008, End line: 1025

```python
class RenameIndex(IndexOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if self.old_fields:
            # Backward operation with unnamed index is a no-op.
            return

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name

        self.database_forwards(app_label, schema_editor, from_state, to_state)

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name
```
### 39 - django/db/migrations/operations/models.py:

Start line: 1027, End line: 1062

```python
class RenameIndex(IndexOperation):

    def describe(self):
        if self.old_name:
            return (
                f"Rename index {self.old_name} on {self.model_name} to {self.new_name}"
            )
        return (
            f"Rename unnamed index for {self.old_fields} on {self.model_name} to "
            f"{self.new_name}"
        )

    @property
    def migration_name_fragment(self):
        if self.old_name:
            return "rename_%s_%s" % (self.old_name_lower, self.new_name_lower)
        return "rename_%s_%s_%s" % (
            self.model_name_lower,
            "_".join(self.old_fields),
            self.new_name_lower,
        )

    def reduce(self, operation, app_label):
        if (
            isinstance(operation, RenameIndex)
            and self.model_name_lower == operation.model_name_lower
            and operation.old_name
            and self.new_name_lower == operation.old_name_lower
        ):
            return [
                RenameIndex(
                    self.model_name,
                    new_name=operation.new_name,
                    old_name=self.old_name,
                    old_fields=self.old_fields,
                )
            ]
        return super().reduce(operation, app_label)
```
### 42 - django/db/migrations/operations/models.py:

Start line: 113, End line: 134

```python
class CreateModel(ModelOperation):

    def references_model(self, name, app_label):
        name_lower = name.lower()
        if name_lower == self.name_lower:
            return True

        # Check we didn't inherit from the model
        reference_model_tuple = (app_label, name_lower)
        for base in self.bases:
            if (
                base is not models.Model
                and isinstance(base, (models.base.ModelBase, str))
                and resolve_relation(base, app_label) == reference_model_tuple
            ):
                return True

        # Check we have no FKs/M2Ms with it
        for _name, field in self.fields:
            if field_references(
                (app_label, self.name_lower), field, reference_model_tuple
            ):
                return True
        return False
```
### 45 - django/db/migrations/operations/models.py:

Start line: 708, End line: 724

```python
class AlterOrderWithRespectTo(ModelOptionOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.database_forwards(app_label, schema_editor, from_state, to_state)

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (
            self.order_with_respect_to is None or name == self.order_with_respect_to
        )

    def describe(self):
        return "Set order_with_respect_to on %s to %s" % (
            self.name,
            self.order_with_respect_to,
        )

    @property
    def migration_name_fragment(self):
        return "alter_%s_order_with_respect_to" % self.name_lower
```
### 50 - django/db/migrations/operations/models.py:

Start line: 682, End line: 706

```python
class AlterOrderWithRespectTo(ModelOptionOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.name)
            # Remove a field if we need to
            if (
                from_model._meta.order_with_respect_to
                and not to_model._meta.order_with_respect_to
            ):
                schema_editor.remove_field(
                    from_model, from_model._meta.get_field("_order")
                )
            # Add a field if we need to (altering the column is untouched as
            # it's likely a rename)
            elif (
                to_model._meta.order_with_respect_to
                and not from_model._meta.order_with_respect_to
            ):
                field = to_model._meta.get_field("_order")
                if not field.has_default():
                    field.default = 0
                schema_editor.add_field(
                    from_model,
                    field,
                )
```
### 53 - django/db/migrations/operations/models.py:

Start line: 1, End line: 18

```python
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property

from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField


def _check_for_duplicates(arg_name, objs):
    used_vals = set()
    for val in objs:
        if val in used_vals:
            raise ValueError(
                "Found duplicate value %s in CreateModel %s argument." % (val, arg_name)
            )
        used_vals.add(val)
```
### 75 - django/db/migrations/operations/models.py:

Start line: 41, End line: 111

```python
class CreateModel(ModelOperation):
    """Create a model's table."""

    serialization_expand_args = ["fields", "options", "managers"]

    def __init__(self, name, fields, options=None, bases=None, managers=None):
        self.fields = fields
        self.options = options or {}
        self.bases = bases or (models.Model,)
        self.managers = managers or []
        super().__init__(name)
        # Sanity-check that there are no duplicated field names, bases, or
        # manager names
        _check_for_duplicates("fields", (name for name, _ in self.fields))
        _check_for_duplicates(
            "bases",
            (
                base._meta.label_lower
                if hasattr(base, "_meta")
                else base.lower()
                if isinstance(base, str)
                else base
                for base in self.bases
            ),
        )
        _check_for_duplicates("managers", (name for name, _ in self.managers))

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "fields": self.fields,
        }
        if self.options:
            kwargs["options"] = self.options
        if self.bases and self.bases != (models.Model,):
            kwargs["bases"] = self.bases
        if self.managers and self.managers != [("objects", models.Manager())]:
            kwargs["managers"] = self.managers
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.add_model(
            ModelState(
                app_label,
                self.name,
                list(self.fields),
                dict(self.options),
                tuple(self.bases),
                list(self.managers),
            )
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def describe(self):
        return "Create %smodel %s" % (
            "proxy " if self.options.get("proxy", False) else "",
            self.name,
        )

    @property
    def migration_name_fragment(self):
        return self.name_lower
```
### 80 - django/db/migrations/operations/models.py:

Start line: 916, End line: 969

```python
class RenameIndex(IndexOperation):
    """Rename an index."""

    def __init__(self, model_name, new_name, old_name=None, old_fields=None):
        if not old_name and not old_fields:
            raise ValueError(
                "RenameIndex requires one of old_name and old_fields arguments to be "
                "set."
            )
        if old_name and old_fields:
            raise ValueError(
                "RenameIndex.old_name and old_fields are mutually exclusive."
            )
        self.model_name = model_name
        self.new_name = new_name
        self.old_name = old_name
        self.old_fields = old_fields

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {
            "model_name": self.model_name,
            "new_name": self.new_name,
        }
        if self.old_name:
            kwargs["old_name"] = self.old_name
        if self.old_fields:
            kwargs["old_fields"] = self.old_fields
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        if self.old_fields:
            state.add_index(
                app_label,
                self.model_name_lower,
                models.Index(fields=self.old_fields, name=self.new_name),
            )
            state.remove_model_options(
                app_label,
                self.model_name_lower,
                AlterIndexTogether.option_name,
                self.old_fields,
            )
        else:
            state.rename_index(
                app_label, self.model_name_lower, self.old_name, self.new_name
            )
```
### 89 - django/db/migrations/operations/models.py:

Start line: 532, End line: 567

```python
class AlterModelTableComment(ModelOptionOperation):
    def __init__(self, name, table_comment):
        self.table_comment = table_comment
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "table_comment": self.table_comment,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(
            app_label, self.name_lower, {"db_table_comment": self.table_comment}
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.name)
            schema_editor.alter_db_table_comment(
                new_model,
                old_model._meta.db_table_comment,
                new_model._meta.db_table_comment,
            )

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        return self.database_forwards(app_label, schema_editor, from_state, to_state)

    def describe(self):
        return f"Alter {self.name} table comment"

    @property
    def migration_name_fragment(self):
        return f"alter_{self.name_lower}_table_comment"
```
### 93 - django/db/migrations/operations/models.py:

Start line: 21, End line: 38

```python
class ModelOperation(Operation):
    def __init__(self, name):
        self.name = name

    @cached_property
    def name_lower(self):
        return self.name.lower()

    def references_model(self, name, app_label):
        return name.lower() == self.name_lower

    def reduce(self, operation, app_label):
        return super().reduce(operation, app_label) or self.can_reduce_through(
            operation, app_label
        )

    def can_reduce_through(self, operation, app_label):
        return not operation.references_model(self.name, app_label)
```
### 103 - django/db/migrations/operations/models.py:

Start line: 597, End line: 606

```python
class AlterTogetherOptionOperation(ModelOptionOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.name)
            alter_together = getattr(schema_editor, "alter_%s" % self.option_name)
            alter_together(
                new_model,
                getattr(old_model._meta, self.option_name, set()),
                getattr(new_model._meta, self.option_name, set()),
            )
```
### 110 - django/db/migrations/operations/models.py:

Start line: 1106, End line: 1146

```python
class RemoveConstraint(IndexOperation):
    option_name = "constraints"

    def __init__(self, model_name, name):
        self.model_name = model_name
        self.name = name

    def state_forwards(self, app_label, state):
        state.remove_constraint(app_label, self.model_name_lower, self.name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            from_model_state = from_state.models[app_label, self.model_name_lower]
            constraint = from_model_state.get_constraint_by_name(self.name)
            schema_editor.remove_constraint(model, constraint)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            to_model_state = to_state.models[app_label, self.model_name_lower]
            constraint = to_model_state.get_constraint_by_name(self.name)
            schema_editor.add_constraint(model, constraint)

    def deconstruct(self):
        return (
            self.__class__.__name__,
            [],
            {
                "model_name": self.model_name,
                "name": self.name,
            },
        )

    def describe(self):
        return "Remove constraint %s from model %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return "remove_%s_%s" % (self.model_name_lower, self.name.lower())
```
### 133 - django/db/migrations/operations/models.py:

Start line: 782, End line: 816

```python
class AlterModelManagers(ModelOptionOperation):
    """Alter the model's managers."""

    serialization_expand_args = ["managers"]

    def __init__(self, name, managers):
        self.managers = managers
        super().__init__(name)

    def deconstruct(self):
        return (self.__class__.__qualname__, [self.name, self.managers], {})

    def state_forwards(self, app_label, state):
        state.alter_model_managers(app_label, self.name_lower, self.managers)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return "Change managers on %s" % self.name

    @property
    def migration_name_fragment(self):
        return "alter_%s_managers" % self.name_lower


class IndexOperation(Operation):
    option_name = "indexes"

    @cached_property
    def model_name_lower(self):
        return self.model_name.lower()
```
### 151 - django/db/migrations/operations/models.py:

Start line: 727, End line: 779

```python
class AlterModelOptions(ModelOptionOperation):
    """
    Set new model options that don't directly affect the database schema
    (like verbose_name, permissions, ordering). Python code in migrations
    may still need them.
    """

    # Model options we want to compare and preserve in an AlterModelOptions op
    ALTER_OPTION_KEYS = [
        "base_manager_name",
        "default_manager_name",
        "default_related_name",
        "get_latest_by",
        "managed",
        "ordering",
        "permissions",
        "default_permissions",
        "select_on_save",
        "verbose_name",
        "verbose_name_plural",
    ]

    def __init__(self, name, options):
        self.options = options
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "options": self.options,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(
            app_label,
            self.name_lower,
            self.options,
            self.ALTER_OPTION_KEYS,
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return "Change Meta options on %s" % self.name

    @property
    def migration_name_fragment(self):
        return "alter_%s_options" % self.name_lower
```
### 152 - django/db/migrations/operations/models.py:

Start line: 608, End line: 632

```python
class AlterTogetherOptionOperation(ModelOptionOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        return self.database_forwards(app_label, schema_editor, from_state, to_state)

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (
            not self.option_value
            or any((name in fields) for fields in self.option_value)
        )

    def describe(self):
        return "Alter %s for %s (%s constraint(s))" % (
            self.option_name,
            self.name,
            len(self.option_value or ""),
        )

    @property
    def migration_name_fragment(self):
        return "alter_%s_%s" % (self.name_lower, self.option_name)

    def can_reduce_through(self, operation, app_label):
        return super().can_reduce_through(operation, app_label) or (
            isinstance(operation, AlterTogetherOptionOperation)
            and type(operation) is not type(self)
        )
```
### 164 - django/db/migrations/operations/models.py:

Start line: 309, End line: 341

```python
class DeleteModel(ModelOperation):
    """Drop a model's table."""

    def deconstruct(self):
        kwargs = {
            "name": self.name,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.remove_model(app_label, self.name_lower)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def references_model(self, name, app_label):
        # The deleted model could be referencing the specified model through
        # related fields.
        return True

    def describe(self):
        return "Delete model %s" % self.name

    @property
    def migration_name_fragment(self):
        return "delete_%s" % self.name_lower
```
### 187 - django/db/migrations/operations/models.py:

Start line: 659, End line: 680

```python
class AlterOrderWithRespectTo(ModelOptionOperation):
    """Represent a change with the order_with_respect_to option."""

    option_name = "order_with_respect_to"

    def __init__(self, name, order_with_respect_to):
        self.order_with_respect_to = order_with_respect_to
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "order_with_respect_to": self.order_with_respect_to,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(
            app_label,
            self.name_lower,
            {self.option_name: self.order_with_respect_to},
        )
```
