# django__django-12049

| **django/django** | `24b9f5082344a127147266dd52d5d2dcd1c9cb44` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 888 |
| **Any found context length** | 888 |
| **Avg pos** | 6.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/executor.py b/django/db/migrations/executor.py
--- a/django/db/migrations/executor.py
+++ b/django/db/migrations/executor.py
@@ -329,8 +329,11 @@ def should_skip_detecting_model(migration, model):
         apps = after_state.apps
         found_create_model_migration = False
         found_add_field_migration = False
+        fold_identifier_case = self.connection.features.ignores_table_name_case
         with self.connection.cursor() as cursor:
-            existing_table_names = self.connection.introspection.table_names(cursor)
+            existing_table_names = set(self.connection.introspection.table_names(cursor))
+            if fold_identifier_case:
+                existing_table_names = {name.casefold() for name in existing_table_names}
         # Make sure all create model and add field operations are done
         for operation in migration.operations:
             if isinstance(operation, migrations.CreateModel):
@@ -341,7 +344,10 @@ def should_skip_detecting_model(migration, model):
                     model = global_apps.get_model(model._meta.swapped)
                 if should_skip_detecting_model(migration, model):
                     continue
-                if model._meta.db_table not in existing_table_names:
+                db_table = model._meta.db_table
+                if fold_identifier_case:
+                    db_table = db_table.casefold()
+                if db_table not in existing_table_names:
                     return False, project_state
                 found_create_model_migration = True
             elif isinstance(operation, migrations.AddField):
@@ -358,19 +364,29 @@ def should_skip_detecting_model(migration, model):
 
                 # Handle implicit many-to-many tables created by AddField.
                 if field.many_to_many:
-                    if field.remote_field.through._meta.db_table not in existing_table_names:
+                    through_db_table = field.remote_field.through._meta.db_table
+                    if fold_identifier_case:
+                        through_db_table = through_db_table.casefold()
+                    if through_db_table not in existing_table_names:
                         return False, project_state
                     else:
                         found_add_field_migration = True
                         continue
-
-                column_names = [
-                    column.name for column in
-                    self.connection.introspection.get_table_description(self.connection.cursor(), table)
-                ]
-                if field.column not in column_names:
+                columns = self.connection.introspection.get_table_description(
+                    self.connection.cursor(),
+                    table,
+                )
+                for column in columns:
+                    field_column = field.column
+                    column_name = column.name
+                    if fold_identifier_case:
+                        column_name = column_name.casefold()
+                        field_column = field_column.casefold()
+                    if column_name == field_column:
+                        found_add_field_migration = True
+                        break
+                else:
                     return False, project_state
-                found_add_field_migration = True
         # If we get this far and we found at least one CreateModel or AddField migration,
         # the migration is considered implicitly applied.
         return (found_create_model_migration or found_add_field_migration), after_state

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/executor.py | 332 | 332 | 2 | 2 | 888
| django/db/migrations/executor.py | 344 | 344 | 2 | 2 | 888
| django/db/migrations/executor.py | 361 | 373 | 2 | 2 | 888


## Problem Statement

```
Applied migration detection may fail when using a case-insensitive collation
Description
	 
		(last modified by Tim Graham)
	 
Hello, 
I'm using this guide ​https://datascience.blog.wzb.eu/2017/03/21/using-django-with-an-existinglegacy-database for my studies with camelCasing together with Django (yes, I'm still trying to keep the naming convention we have inside our DB, also for the model's names)
Now, I'm really new to Django and I don't know if it's intended but this part of code inside django/db/migrations/executor.py' is doing case sensitive comparison to check if a column is already present in a database
column_names = [
	column.name for column in
	self.connection.introspection.get_table_description(self.connection.cursor(), table)
]
if field.column not in column_names:
	return False, project_state
so if my migration file contains something like this
		migrations.AddField(
			model_name='city',
			name='countrycode',
			field=models.ForeignKey(db_column='countryCode', on_delete=django.db.models.deletion.CASCADE, to='my_DB.country'),
and I run python3 manage.py migrate --database my_DB --fake-initial my_first_app
it fires an error saying that that table already exists 
django.db.utils.OperationalError: (1050, "Table 'city' already exists")
If I run python3 manage.py migrate --database my_DB --fake my_first_app it correctly fakes my_first_app
The my_DB collation is case insensitive, while MySql is running with the ' --lower-case-table-names=0' option

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/base.py | 1445 | 1468| 176 | 176 | 15311 | 
| **-> 2 <-** | **2 django/db/migrations/executor.py** | 298 | 377| 712 | 888 | 18603 | 
| 3 | 2 django/db/models/base.py | 1761 | 1832| 565 | 1453 | 18603 | 
| 4 | 3 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 1684 | 30338 | 
| 5 | 4 django/db/migrations/exceptions.py | 1 | 55| 250 | 1934 | 30589 | 
| 6 | 4 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 2179 | 30589 | 
| 7 | 4 django/db/migrations/autodetector.py | 1027 | 1043| 188 | 2367 | 30589 | 
| 8 | 4 django/db/migrations/autodetector.py | 1086 | 1121| 312 | 2679 | 30589 | 
| 9 | 4 django/db/models/base.py | 1470 | 1492| 171 | 2850 | 30589 | 
| 10 | 4 django/db/migrations/autodetector.py | 1209 | 1221| 131 | 2981 | 30589 | 
| 11 | 4 django/db/migrations/autodetector.py | 437 | 463| 256 | 3237 | 30589 | 
| 12 | 5 django/core/management/commands/migrate.py | 21 | 65| 369 | 3606 | 33747 | 
| 13 | 5 django/core/management/commands/migrate.py | 161 | 242| 793 | 4399 | 33747 | 
| 14 | 6 django/db/migrations/loader.py | 146 | 172| 291 | 4690 | 36638 | 
| 15 | 6 django/db/models/base.py | 1250 | 1279| 242 | 4932 | 36638 | 
| 16 | 6 django/db/migrations/autodetector.py | 358 | 372| 141 | 5073 | 36638 | 
| 17 | 6 django/db/migrations/autodetector.py | 18 | 35| 185 | 5258 | 36638 | 
| 18 | 7 django/db/migrations/state.py | 601 | 612| 136 | 5394 | 41856 | 
| 19 | 7 django/db/migrations/autodetector.py | 987 | 1003| 188 | 5582 | 41856 | 
| 20 | 7 django/db/migrations/autodetector.py | 1 | 15| 110 | 5692 | 41856 | 
| 21 | 7 django/db/models/base.py | 1 | 50| 330 | 6022 | 41856 | 
| 22 | 7 django/db/models/base.py | 1494 | 1526| 231 | 6253 | 41856 | 
| 23 | 7 django/db/migrations/autodetector.py | 337 | 356| 196 | 6449 | 41856 | 
| 24 | 8 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 6644 | 42051 | 
| 25 | 8 django/core/management/commands/migrate.py | 1 | 18| 148 | 6792 | 42051 | 
| 26 | 9 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 7112 | 42371 | 
| 27 | 9 django/db/migrations/state.py | 1 | 24| 191 | 7303 | 42371 | 
| 28 | 10 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 8152 | 43220 | 
| 29 | 11 django/core/management/commands/sqlmigrate.py | 32 | 69| 371 | 8523 | 43852 | 
| 30 | 11 django/db/migrations/autodetector.py | 1146 | 1180| 296 | 8819 | 43852 | 
| 31 | 11 django/db/migrations/autodetector.py | 1045 | 1065| 136 | 8955 | 43852 | 
| 32 | 12 django/db/migrations/operations/models.py | 345 | 394| 493 | 9448 | 50548 | 
| 33 | 13 django/db/backends/mysql/schema.py | 1 | 77| 751 | 10199 | 51943 | 
| 34 | 14 django/core/management/commands/inspectdb.py | 176 | 230| 478 | 10677 | 54560 | 
| 35 | 14 django/core/management/commands/migrate.py | 67 | 160| 825 | 11502 | 54560 | 
| 36 | 15 django/db/utils.py | 272 | 314| 322 | 11824 | 56672 | 
| 37 | 15 django/db/migrations/operations/models.py | 1 | 38| 238 | 12062 | 56672 | 
| 38 | 16 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 12850 | 59421 | 
| 39 | 17 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 13041 | 59612 | 
| 40 | 18 django/db/migrations/recorder.py | 47 | 96| 378 | 13419 | 60282 | 
| 41 | 18 django/db/migrations/state.py | 580 | 599| 188 | 13607 | 60282 | 
| 42 | 18 django/db/models/base.py | 1834 | 1858| 175 | 13782 | 60282 | 
| 43 | 19 django/db/backends/sqlite3/schema.py | 1 | 37| 318 | 14100 | 64247 | 
| 44 | 19 django/db/migrations/autodetector.py | 1067 | 1084| 180 | 14280 | 64247 | 
| 45 | 19 django/core/management/commands/sqlmigrate.py | 1 | 30| 266 | 14546 | 64247 | 
| 46 | 19 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 14695 | 64247 | 
| 47 | 19 django/db/models/base.py | 1281 | 1306| 184 | 14879 | 64247 | 
| 48 | 20 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 15086 | 64454 | 
| 49 | 20 django/db/migrations/recorder.py | 1 | 22| 153 | 15239 | 64454 | 
| 50 | 20 django/db/migrations/operations/models.py | 102 | 118| 147 | 15386 | 64454 | 
| 51 | 21 django/utils/datastructures.py | 296 | 340| 298 | 15684 | 66711 | 
| 52 | 22 django/db/models/options.py | 1 | 36| 304 | 15988 | 73810 | 
| 53 | 23 django/core/management/commands/squashmigrations.py | 202 | 215| 112 | 16100 | 75681 | 
| 54 | 24 django/db/backends/mysql/features.py | 1 | 101| 834 | 16934 | 77006 | 
| 55 | 25 django/db/backends/mysql/validation.py | 1 | 27| 248 | 17182 | 77494 | 
| 56 | 26 django/db/backends/base/features.py | 1 | 115| 904 | 18086 | 80073 | 
| 57 | 26 django/db/migrations/autodetector.py | 465 | 506| 418 | 18504 | 80073 | 
| 58 | 26 django/db/migrations/autodetector.py | 904 | 985| 876 | 19380 | 80073 | 
| 59 | 26 django/db/migrations/operations/models.py | 396 | 412| 182 | 19562 | 80073 | 
| 60 | 26 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 19743 | 80073 | 
| 61 | 27 django/db/migrations/questioner.py | 1 | 54| 468 | 20211 | 82147 | 
| 62 | 28 django/core/management/base.py | 451 | 483| 282 | 20493 | 86534 | 
| 63 | 29 django/db/backends/oracle/schema.py | 125 | 173| 419 | 20912 | 88290 | 
| 64 | 29 django/db/migrations/autodetector.py | 525 | 671| 1109 | 22021 | 88290 | 
| 65 | 29 django/db/backends/base/features.py | 117 | 215| 844 | 22865 | 88290 | 
| 66 | 29 django/db/migrations/operations/models.py | 460 | 485| 279 | 23144 | 88290 | 
| 67 | 29 django/db/migrations/autodetector.py | 508 | 524| 186 | 23330 | 88290 | 
| 68 | 29 django/db/migrations/questioner.py | 227 | 240| 123 | 23453 | 88290 | 
| 69 | 30 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 23611 | 89476 | 
| 70 | 31 django/core/management/sql.py | 20 | 34| 116 | 23727 | 89861 | 
| 71 | 32 django/db/backends/oracle/introspection.py | 108 | 141| 286 | 24013 | 92220 | 
| 72 | 33 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 24150 | 92357 | 
| 73 | 33 django/db/migrations/loader.py | 275 | 299| 205 | 24355 | 92357 | 
| 74 | 33 django/core/management/commands/migrate.py | 293 | 340| 401 | 24756 | 92357 | 
| 75 | **33 django/db/migrations/executor.py** | 281 | 296| 165 | 24921 | 92357 | 
| 76 | 33 django/db/models/base.py | 1388 | 1443| 491 | 25412 | 92357 | 
| 77 | 33 django/db/migrations/autodetector.py | 1005 | 1025| 134 | 25546 | 92357 | 
| 78 | 34 django/db/migrations/operations/fields.py | 241 | 251| 146 | 25692 | 95624 | 
| 79 | 34 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 26178 | 95624 | 
| 80 | 35 django/db/backends/mysql/operations.py | 191 | 234| 329 | 26507 | 98917 | 
| 81 | 35 django/db/backends/mysql/operations.py | 1 | 33| 254 | 26761 | 98917 | 
| 82 | 36 django/db/backends/sqlite3/base.py | 299 | 383| 829 | 27590 | 104656 | 
| 83 | 36 django/db/migrations/autodetector.py | 1297 | 1328| 314 | 27904 | 104656 | 
| 84 | 36 django/db/migrations/state.py | 154 | 164| 132 | 28036 | 104656 | 
| 85 | 36 django/db/migrations/autodetector.py | 200 | 222| 239 | 28275 | 104656 | 
| 86 | 37 django/db/backends/mysql/creation.py | 1 | 29| 219 | 28494 | 105265 | 
| 87 | 38 django/contrib/postgres/apps.py | 40 | 67| 249 | 28743 | 105831 | 
| 88 | 38 django/db/migrations/operations/fields.py | 357 | 384| 289 | 29032 | 105831 | 
| 89 | 39 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 29249 | 106048 | 
| 90 | 39 django/db/models/base.py | 1068 | 1111| 404 | 29653 | 106048 | 
| 91 | 39 django/db/backends/mysql/operations.py | 315 | 328| 176 | 29829 | 106048 | 
| 92 | 39 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 30131 | 106048 | 
| 93 | 39 django/db/models/base.py | 1611 | 1659| 348 | 30479 | 106048 | 
| 94 | 40 django/db/migrations/migration.py | 127 | 194| 585 | 31064 | 107681 | 
| 95 | 41 django/db/migrations/operations/utils.py | 1 | 14| 138 | 31202 | 108161 | 
| 96 | 41 django/db/migrations/autodetector.py | 264 | 335| 748 | 31950 | 108161 | 
| 97 | 42 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 32257 | 108468 | 
| 98 | 42 django/db/migrations/operations/fields.py | 1 | 37| 237 | 32494 | 108468 | 
| 99 | 43 django/db/backends/postgresql/operations.py | 88 | 105| 202 | 32696 | 111227 | 
| 100 | 43 django/db/migrations/operations/models.py | 304 | 343| 406 | 33102 | 111227 | 
| 101 | 44 django/db/models/__init__.py | 1 | 51| 576 | 33678 | 111803 | 
| 102 | 45 django/contrib/contenttypes/apps.py | 1 | 23| 150 | 33828 | 111953 | 
| 103 | 45 django/db/backends/mysql/schema.py | 91 | 104| 148 | 33976 | 111953 | 
| 104 | 45 django/db/migrations/questioner.py | 187 | 205| 237 | 34213 | 111953 | 
| 105 | 46 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 34375 | 112115 | 
| 106 | 46 django/core/management/commands/migrate.py | 243 | 257| 170 | 34545 | 112115 | 
| 107 | 47 django/db/backends/oracle/creation.py | 130 | 165| 399 | 34944 | 116010 | 
| 108 | 47 django/db/models/base.py | 1159 | 1187| 213 | 35157 | 116010 | 
| 109 | 47 django/db/migrations/state.py | 496 | 528| 250 | 35407 | 116010 | 
| 110 | 48 django/db/models/fields/related.py | 255 | 282| 269 | 35676 | 129522 | 
| 111 | 49 django/db/backends/postgresql/base.py | 66 | 141| 772 | 36448 | 132282 | 
| 112 | 49 django/db/models/base.py | 1661 | 1759| 717 | 37165 | 132282 | 
| 113 | 50 django/db/backends/mysql/base.py | 1 | 49| 457 | 37622 | 135380 | 
| 114 | 50 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 37906 | 135380 | 
| 115 | 51 django/db/backends/postgresql/schema.py | 79 | 147| 539 | 38445 | 137323 | 
| 116 | 52 django/db/migrations/operations/special.py | 181 | 204| 246 | 38691 | 138881 | 
| 117 | 53 django/db/migrations/operations/base.py | 1 | 102| 783 | 39474 | 139960 | 
| 118 | 53 django/db/migrations/autodetector.py | 89 | 101| 118 | 39592 | 139960 | 
| 119 | 53 django/core/management/sql.py | 37 | 52| 116 | 39708 | 139960 | 
| 120 | 54 django/db/backends/postgresql/creation.py | 36 | 51| 174 | 39882 | 140608 | 
| 121 | 54 django/db/models/fields/related.py | 190 | 254| 673 | 40555 | 140608 | 
| 122 | 54 django/db/migrations/operations/models.py | 591 | 607| 215 | 40770 | 140608 | 
| 123 | 54 django/db/migrations/operations/fields.py | 274 | 300| 158 | 40928 | 140608 | 
| 124 | 54 django/db/migrations/autodetector.py | 1272 | 1295| 240 | 41168 | 140608 | 
| 125 | 54 django/core/management/commands/migrate.py | 259 | 291| 349 | 41517 | 140608 | 
| 126 | 54 django/db/migrations/operations/models.py | 609 | 622| 137 | 41654 | 140608 | 
| 127 | 54 django/db/migrations/operations/fields.py | 302 | 355| 535 | 42189 | 140608 | 
| 128 | 55 django/db/backends/base/base.py | 1 | 23| 138 | 42327 | 145466 | 
| 129 | 56 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 42438 | 145577 | 
| 130 | 57 django/db/backends/oracle/operations.py | 364 | 401| 369 | 42807 | 151448 | 
| 131 | 57 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 43045 | 151448 | 
| 132 | 57 django/db/migrations/autodetector.py | 224 | 237| 199 | 43244 | 151448 | 
| 133 | 57 django/db/migrations/questioner.py | 162 | 185| 246 | 43490 | 151448 | 
| 134 | 58 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 43628 | 151586 | 
| 135 | 59 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 43778 | 151736 | 
| 136 | 59 django/db/utils.py | 1 | 49| 154 | 43932 | 151736 | 
| 137 | 60 django/db/migrations/__init__.py | 1 | 3| 0 | 43932 | 151760 | 
| 138 | 60 django/db/migrations/recorder.py | 24 | 45| 145 | 44077 | 151760 | 
| 139 | 61 django/db/models/fields/__init__.py | 1052 | 1081| 218 | 44295 | 169200 | 
| 140 | 61 django/db/migrations/operations/models.py | 517 | 526| 129 | 44424 | 169200 | 
| 141 | 62 django/contrib/admin/utils.py | 1 | 24| 228 | 44652 | 173298 | 
| 142 | 63 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 44926 | 173572 | 
| 143 | 63 django/db/migrations/state.py | 106 | 152| 368 | 45294 | 173572 | 
| 144 | 64 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 45294 | 173649 | 
| 145 | 64 django/db/migrations/operations/models.py | 839 | 874| 347 | 45641 | 173649 | 
| 146 | 64 django/db/backends/oracle/operations.py | 20 | 71| 556 | 46197 | 173649 | 
| 147 | 65 django/db/models/signals.py | 37 | 54| 231 | 46428 | 174136 | 
| 148 | 65 django/db/backends/oracle/operations.py | 467 | 499| 367 | 46795 | 174136 | 
| 149 | 65 django/db/migrations/questioner.py | 56 | 81| 220 | 47015 | 174136 | 
| 150 | 65 django/db/backends/mysql/base.py | 286 | 324| 404 | 47419 | 174136 | 
| 151 | 66 django/db/backends/base/schema.py | 44 | 119| 790 | 48209 | 185452 | 
| 152 | 66 django/db/backends/postgresql/operations.py | 1 | 28| 253 | 48462 | 185452 | 
| 153 | 67 django/core/checks/model_checks.py | 129 | 153| 268 | 48730 | 187239 | 
| 154 | 67 django/db/backends/sqlite3/base.py | 80 | 151| 729 | 49459 | 187239 | 
| 155 | 67 django/db/migrations/operations/base.py | 104 | 142| 299 | 49758 | 187239 | 
| 156 | 67 django/db/migrations/autodetector.py | 707 | 794| 789 | 50547 | 187239 | 
| 157 | 68 django/contrib/gis/db/backends/mysql/schema.py | 25 | 38| 146 | 50693 | 187872 | 
| 158 | 68 django/db/backends/base/schema.py | 626 | 698| 792 | 51485 | 187872 | 
| 159 | 68 django/db/backends/oracle/operations.py | 579 | 594| 221 | 51706 | 187872 | 
| 160 | 68 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 52056 | 187872 | 
| 161 | 68 django/db/migrations/operations/special.py | 44 | 60| 180 | 52236 | 187872 | 
| 162 | 69 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 52480 | 189321 | 
| 163 | 69 django/db/models/base.py | 1371 | 1386| 153 | 52633 | 189321 | 
| 164 | 69 django/db/backends/oracle/operations.py | 298 | 323| 243 | 52876 | 189321 | 
| 165 | 69 django/db/migrations/migration.py | 1 | 88| 714 | 53590 | 189321 | 
| 166 | 69 django/core/checks/model_checks.py | 178 | 211| 332 | 53922 | 189321 | 
| 167 | 69 django/db/backends/base/schema.py | 1080 | 1104| 193 | 54115 | 189321 | 
| 168 | 69 django/db/backends/oracle/schema.py | 57 | 77| 249 | 54364 | 189321 | 
| 169 | 69 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 55184 | 189321 | 
| 170 | 69 django/db/models/base.py | 1113 | 1140| 286 | 55470 | 189321 | 
| 171 | 69 django/core/checks/model_checks.py | 1 | 86| 667 | 56137 | 189321 | 
| 172 | 69 django/db/backends/mysql/creation.py | 57 | 67| 149 | 56286 | 189321 | 
| 173 | 69 django/db/migrations/operations/fields.py | 220 | 239| 205 | 56491 | 189321 | 
| 174 | 69 django/db/migrations/autodetector.py | 796 | 845| 570 | 57061 | 189321 | 
| 175 | 69 django/db/models/base.py | 1339 | 1369| 244 | 57305 | 189321 | 
| 176 | 69 django/db/backends/mysql/features.py | 103 | 156| 497 | 57802 | 189321 | 
| 177 | 69 django/db/migrations/operations/fields.py | 253 | 271| 161 | 57963 | 189321 | 
| 178 | 70 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 58085 | 189570 | 
| 179 | 70 django/db/models/fields/__init__.py | 2322 | 2371| 311 | 58396 | 189570 | 
| 180 | 70 django/contrib/postgres/apps.py | 1 | 17| 129 | 58525 | 189570 | 
| 181 | 70 django/db/models/base.py | 1861 | 1912| 351 | 58876 | 189570 | 
| 182 | 70 django/db/backends/oracle/schema.py | 1 | 39| 406 | 59282 | 189570 | 
| 183 | 71 django/contrib/auth/apps.py | 1 | 29| 213 | 59495 | 189783 | 
| 184 | 71 django/db/backends/mysql/base.py | 98 | 166| 718 | 60213 | 189783 | 
| 185 | 71 django/db/models/fields/__init__.py | 338 | 362| 184 | 60397 | 189783 | 
| 186 | 71 django/db/backends/mysql/base.py | 250 | 284| 247 | 60644 | 189783 | 
| 187 | 71 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 61435 | 189783 | 
| 188 | 71 django/db/models/fields/related.py | 964 | 991| 215 | 61650 | 189783 | 
| 189 | 71 django/db/backends/oracle/operations.py | 325 | 336| 227 | 61877 | 189783 | 
| 190 | 71 django/db/backends/mysql/creation.py | 31 | 55| 254 | 62131 | 189783 | 
| 191 | 71 django/db/migrations/autodetector.py | 1223 | 1270| 429 | 62560 | 189783 | 
| 192 | 71 django/db/backends/base/schema.py | 1048 | 1062| 123 | 62683 | 189783 | 
| 193 | 71 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 63414 | 189783 | 
| 194 | 71 django/db/migrations/operations/models.py | 528 | 541| 139 | 63553 | 189783 | 
| 195 | 71 django/db/models/fields/__init__.py | 208 | 242| 235 | 63788 | 189783 | 


### Hint

```
Not sure about the solution. PR just created for proposing the solution.
Left a few comments for improvements.
```

## Patch

```diff
diff --git a/django/db/migrations/executor.py b/django/db/migrations/executor.py
--- a/django/db/migrations/executor.py
+++ b/django/db/migrations/executor.py
@@ -329,8 +329,11 @@ def should_skip_detecting_model(migration, model):
         apps = after_state.apps
         found_create_model_migration = False
         found_add_field_migration = False
+        fold_identifier_case = self.connection.features.ignores_table_name_case
         with self.connection.cursor() as cursor:
-            existing_table_names = self.connection.introspection.table_names(cursor)
+            existing_table_names = set(self.connection.introspection.table_names(cursor))
+            if fold_identifier_case:
+                existing_table_names = {name.casefold() for name in existing_table_names}
         # Make sure all create model and add field operations are done
         for operation in migration.operations:
             if isinstance(operation, migrations.CreateModel):
@@ -341,7 +344,10 @@ def should_skip_detecting_model(migration, model):
                     model = global_apps.get_model(model._meta.swapped)
                 if should_skip_detecting_model(migration, model):
                     continue
-                if model._meta.db_table not in existing_table_names:
+                db_table = model._meta.db_table
+                if fold_identifier_case:
+                    db_table = db_table.casefold()
+                if db_table not in existing_table_names:
                     return False, project_state
                 found_create_model_migration = True
             elif isinstance(operation, migrations.AddField):
@@ -358,19 +364,29 @@ def should_skip_detecting_model(migration, model):
 
                 # Handle implicit many-to-many tables created by AddField.
                 if field.many_to_many:
-                    if field.remote_field.through._meta.db_table not in existing_table_names:
+                    through_db_table = field.remote_field.through._meta.db_table
+                    if fold_identifier_case:
+                        through_db_table = through_db_table.casefold()
+                    if through_db_table not in existing_table_names:
                         return False, project_state
                     else:
                         found_add_field_migration = True
                         continue
-
-                column_names = [
-                    column.name for column in
-                    self.connection.introspection.get_table_description(self.connection.cursor(), table)
-                ]
-                if field.column not in column_names:
+                columns = self.connection.introspection.get_table_description(
+                    self.connection.cursor(),
+                    table,
+                )
+                for column in columns:
+                    field_column = field.column
+                    column_name = column.name
+                    if fold_identifier_case:
+                        column_name = column_name.casefold()
+                        field_column = field_column.casefold()
+                    if column_name == field_column:
+                        found_add_field_migration = True
+                        break
+                else:
                     return False, project_state
-                found_add_field_migration = True
         # If we get this far and we found at least one CreateModel or AddField migration,
         # the migration is considered implicitly applied.
         return (found_create_model_migration or found_add_field_migration), after_state

```

## Test Patch

```diff
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -14,7 +14,7 @@
 from django.db.backends.utils import truncate_name
 from django.db.migrations.exceptions import InconsistentMigrationHistory
 from django.db.migrations.recorder import MigrationRecorder
-from django.test import TestCase, override_settings
+from django.test import TestCase, override_settings, skipUnlessDBFeature
 
 from .models import UnicodeModel, UnserializableModel
 from .routers import TestRouter
@@ -197,6 +197,32 @@ def test_migrate_fake_initial(self):
             self.assertTableNotExists("migrations_tribble", using=db)
             self.assertTableNotExists("migrations_book", using=db)
 
+    @skipUnlessDBFeature('ignores_table_name_case')
+    def test_migrate_fake_initial_case_insensitive(self):
+        with override_settings(MIGRATION_MODULES={
+            'migrations': 'migrations.test_fake_initial_case_insensitive.initial',
+        }):
+            call_command('migrate', 'migrations', '0001', verbosity=0)
+            call_command('migrate', 'migrations', 'zero', fake=True, verbosity=0)
+
+        with override_settings(MIGRATION_MODULES={
+            'migrations': 'migrations.test_fake_initial_case_insensitive.fake_initial',
+        }):
+            out = io.StringIO()
+            call_command(
+                'migrate',
+                'migrations',
+                '0001',
+                fake_initial=True,
+                stdout=out,
+                verbosity=1,
+                no_color=True,
+            )
+            self.assertIn(
+                'migrations.0001_initial... faked',
+                out.getvalue().lower(),
+            )
+
     @override_settings(MIGRATION_MODULES={"migrations": "migrations.test_migrations_fake_split_initial"})
     def test_migrate_fake_split_initial(self):
         """
diff --git a/tests/migrations/test_fake_initial_case_insensitive/fake_initial/0001_initial.py b/tests/migrations/test_fake_initial_case_insensitive/fake_initial/0001_initial.py
new file mode 100644
--- /dev/null
+++ b/tests/migrations/test_fake_initial_case_insensitive/fake_initial/0001_initial.py
@@ -0,0 +1,28 @@
+from django.db import migrations, models
+
+
+class Migration(migrations.Migration):
+    initial = True
+
+    operations = [
+        migrations.CreateModel(
+            'fakeinitialmodel',
+            [
+                ('id', models.AutoField(primary_key=True)),
+                ('field', models.CharField(max_length=20)),
+            ],
+            options={
+                'db_table': 'migrations_mIxEd_cAsE_iNiTiAl_mOdEl',
+            },
+        ),
+        migrations.AddField(
+            model_name='fakeinitialmodel',
+            name='field_mixed_case',
+            field=models.CharField(max_length=20, db_column='fIeLd_mIxEd_cAsE'),
+        ),
+        migrations.AddField(
+            model_name='fakeinitialmodel',
+            name='fake_initial_model',
+            field=models.ManyToManyField(to='migrations.fakeinitialmodel', db_table='m2m_mIxEd_cAsE'),
+        ),
+    ]
diff --git a/tests/migrations/test_fake_initial_case_insensitive/fake_initial/__init__.py b/tests/migrations/test_fake_initial_case_insensitive/fake_initial/__init__.py
new file mode 100644
diff --git a/tests/migrations/test_fake_initial_case_insensitive/initial/0001_initial.py b/tests/migrations/test_fake_initial_case_insensitive/initial/0001_initial.py
new file mode 100644
--- /dev/null
+++ b/tests/migrations/test_fake_initial_case_insensitive/initial/0001_initial.py
@@ -0,0 +1,23 @@
+from django.db import migrations, models
+
+
+class Migration(migrations.Migration):
+    initial = True
+
+    operations = [
+        migrations.CreateModel(
+            name='fakeinitialmodel',
+            fields=[
+                ('id', models.AutoField(primary_key=True)),
+                ('field', models.CharField(max_length=20)),
+                ('field_mixed_case', models.CharField(max_length=20, db_column='FiEld_MiXeD_CaSe')),
+                (
+                    'fake_initial_mode',
+                    models.ManyToManyField('migrations.FakeInitialModel', db_table='m2m_MiXeD_CaSe'),
+                ),
+            ],
+            options={
+                'db_table': 'migrations_MiXeD_CaSe_InItIaL_MoDel',
+            },
+        ),
+    ]
diff --git a/tests/migrations/test_fake_initial_case_insensitive/initial/__init__.py b/tests/migrations/test_fake_initial_case_insensitive/initial/__init__.py
new file mode 100644

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 1445, End line: 1468

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_column_name_clashes(cls):
        # Store a list of column names which have already been used by other fields.
        used_column_names = []
        errors = []

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Ensure the column name is not already in use.
            if column_name and column_name in used_column_names:
                errors.append(
                    checks.Error(
                        "Field '%s' has column name '%s' that is used by "
                        "another field." % (f.name, column_name),
                        hint="Specify a 'db_column' for the field.",
                        obj=cls,
                        id='models.E007'
                    )
                )
            else:
                used_column_names.append(column_name)

        return errors
```
### 2 - django/db/migrations/executor.py:

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
### 3 - django/db/models/base.py:

Start line: 1761, End line: 1832

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_long_column_names(cls):
        """
        Check that any auto-generated column names are shorter than the limits
        for each database in which the model will be created.
        """
        errors = []
        allowed_len = None
        db_alias = None

        # Find the minimum max allowed length among all specified db_aliases.
        for db in settings.DATABASES:
            # skip databases where the model won't be created
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            max_name_length = connection.ops.max_name_length()
            if max_name_length is None or connection.features.truncates_names:
                continue
            else:
                if allowed_len is None:
                    allowed_len = max_name_length
                    db_alias = db
                elif max_name_length < allowed_len:
                    allowed_len = max_name_length
                    db_alias = db

        if allowed_len is None:
            return errors

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Check if auto-generated name for the field is too long
            # for the database.
            if f.db_column is None and column_name is not None and len(column_name) > allowed_len:
                errors.append(
                    checks.Error(
                        'Autogenerated column name too long for field "%s". '
                        'Maximum length is "%s" for database "%s".'
                        % (column_name, allowed_len, db_alias),
                        hint="Set the column name manually using 'db_column'.",
                        obj=cls,
                        id='models.E018',
                    )
                )

        for f in cls._meta.local_many_to_many:
            # Skip nonexistent models.
            if isinstance(f.remote_field.through, str):
                continue

            # Check if auto-generated name for the M2M field is too long
            # for the database.
            for m2m in f.remote_field.through._meta.local_fields:
                _, rel_name = m2m.get_attname_column()
                if m2m.db_column is None and rel_name is not None and len(rel_name) > allowed_len:
                    errors.append(
                        checks.Error(
                            'Autogenerated column name too long for M2M field '
                            '"%s". Maximum length is "%s" for database "%s".'
                            % (rel_name, allowed_len, db_alias),
                            hint=(
                                "Use 'through' to create a separate model for "
                                "M2M and then set column_name using 'db_column'."
                            ),
                            obj=cls,
                            id='models.E019',
                        )
                    )

        return errors
```
### 4 - django/db/migrations/autodetector.py:

Start line: 1123, End line: 1144

```python
class MigrationAutodetector:

    def generate_altered_unique_together(self):
        self._generate_altered_foo_together(operations.AlterUniqueTogether)

    def generate_altered_index_together(self):
        self._generate_altered_foo_together(operations.AlterIndexTogether)

    def generate_altered_db_table(self):
        models_to_check = self.kept_model_keys.union(self.kept_proxy_keys, self.kept_unmanaged_keys)
        for app_label, model_name in sorted(models_to_check):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            old_db_table_name = old_model_state.options.get('db_table')
            new_db_table_name = new_model_state.options.get('db_table')
            if old_db_table_name != new_db_table_name:
                self.add_operation(
                    app_label,
                    operations.AlterModelTable(
                        name=model_name,
                        table=new_db_table_name,
                    )
                )
```
### 5 - django/db/migrations/exceptions.py:

Start line: 1, End line: 55

```python
from django.db.utils import DatabaseError


class AmbiguityError(Exception):
    """More than one migration matches a name prefix."""
    pass


class BadMigrationError(Exception):
    """There's a bad migration (unreadable/bad format/etc.)."""
    pass


class CircularDependencyError(Exception):
    """There's an impossible-to-resolve circular dependency."""
    pass


class InconsistentMigrationHistory(Exception):
    """An applied migration has some of its dependencies not applied."""
    pass


class InvalidBasesError(ValueError):
    """A model's base classes can't be resolved."""
    pass


class IrreversibleError(RuntimeError):
    """An irreversible migration is about to be reversed."""
    pass


class NodeNotFoundError(LookupError):
    """An attempt on a node is made that is not available in the graph."""

    def __init__(self, message, node, origin=None):
        self.message = message
        self.origin = origin
        self.node = node

    def __str__(self):
        return self.message

    def __repr__(self):
        return "NodeNotFoundError(%r)" % (self.node,)


class MigrationSchemaMissing(DatabaseError):
    pass


class InvalidMigrationPlan(ValueError):
    pass
```
### 6 - django/db/migrations/autodetector.py:

Start line: 1182, End line: 1207

```python
class MigrationAutodetector:

    def generate_altered_order_with_respect_to(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if (old_model_state.options.get("order_with_respect_to") !=
                    new_model_state.options.get("order_with_respect_to")):
                # Make sure it comes second if we're adding
                # (removal dependency is part of RemoveField)
                dependencies = []
                if new_model_state.options.get("order_with_respect_to"):
                    dependencies.append((
                        app_label,
                        model_name,
                        new_model_state.options["order_with_respect_to"],
                        True,
                    ))
                # Actually generate the operation
                self.add_operation(
                    app_label,
                    operations.AlterOrderWithRespectTo(
                        name=model_name,
                        order_with_respect_to=new_model_state.options.get('order_with_respect_to'),
                    ),
                    dependencies=dependencies,
                )
```
### 7 - django/db/migrations/autodetector.py:

Start line: 1027, End line: 1043

```python
class MigrationAutodetector:

    def create_altered_constraints(self):
        option_name = operations.AddConstraint.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_constraints = old_model_state.options[option_name]
            new_constraints = new_model_state.options[option_name]
            add_constraints = [c for c in new_constraints if c not in old_constraints]
            rem_constraints = [c for c in old_constraints if c not in new_constraints]

            self.altered_constraints.update({
                (app_label, model_name): {
                    'added_constraints': add_constraints, 'removed_constraints': rem_constraints,
                }
            })
```
### 8 - django/db/migrations/autodetector.py:

Start line: 1086, End line: 1121

```python
class MigrationAutodetector:

    def _generate_altered_foo_together(self, operation):
        option_name = operation.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            # We run the old version through the field renames to account for those
            old_value = old_model_state.options.get(option_name)
            old_value = {
                tuple(
                    self.renamed_fields.get((app_label, model_name, n), n)
                    for n in unique
                )
                for unique in old_value
            } if old_value else set()

            new_value = new_model_state.options.get(option_name)
            new_value = set(new_value) if new_value else set()

            if old_value != new_value:
                dependencies = []
                for foo_togethers in new_value:
                    for field_name in foo_togethers:
                        field = self.new_apps.get_model(app_label, model_name)._meta.get_field(field_name)
                        if field.remote_field and field.remote_field.model:
                            dependencies.extend(self._get_dependencies_for_foreign_key(field))

                self.add_operation(
                    app_label,
                    operation(
                        name=model_name,
                        **{option_name: new_value}
                    ),
                    dependencies=dependencies,
                )
```
### 9 - django/db/models/base.py:

Start line: 1470, End line: 1492

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model_name_db_lookup_clashes(cls):
        errors = []
        model_name = cls.__name__
        if model_name.startswith('_') or model_name.endswith('_'):
            errors.append(
                checks.Error(
                    "The model name '%s' cannot start or end with an underscore "
                    "as it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E023'
                )
            )
        elif LOOKUP_SEP in model_name:
            errors.append(
                checks.Error(
                    "The model name '%s' cannot contain double underscores as "
                    "it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E024'
                )
            )
        return errors
```
### 10 - django/db/migrations/autodetector.py:

Start line: 1209, End line: 1221

```python
class MigrationAutodetector:

    def generate_altered_managers(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if old_model_state.managers != new_model_state.managers:
                self.add_operation(
                    app_label,
                    operations.AlterModelManagers(
                        name=model_name,
                        managers=new_model_state.managers,
                    )
                )
```
### 75 - django/db/migrations/executor.py:

Start line: 281, End line: 296

```python
class MigrationExecutor:

    def check_replacements(self):
        """
        Mark replacement migrations applied if their replaced set all are.

        Do this unconditionally on every migrate, rather than just when
        migrations are applied or unapplied, to correctly handle the case
        when a new squash migration is pushed to a deployment that already had
        all its replaced migrations applied. In this case no new migration will
        be applied, but the applied state of the squashed migration must be
        maintained.
        """
        applied = self.recorder.applied_migrations()
        for key, migration in self.loader.replacements.items():
            all_applied = all(m in applied for m in migration.replaces)
            if all_applied and key not in applied:
                self.recorder.record_applied(*key)
```
