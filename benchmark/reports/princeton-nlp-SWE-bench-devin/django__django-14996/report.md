# django__django-14996

| **django/django** | `69b0736fad1d1f0197409ca025b7bcdf5666ae62` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 4089 |
| **Any found context length** | 4089 |
| **Avg pos** | 69.0 |
| **Min pos** | 11 |
| **Max pos** | 108 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -840,6 +840,20 @@ def generate_renamed_fields(self):
                             old_field_dec[0:2] == field_dec[0:2] and
                             dict(old_field_dec[2], db_column=old_db_column) == field_dec[2])):
                         if self.questioner.ask_rename(model_name, rem_field_name, field_name, field):
+                            # A db_column mismatch requires a prior noop
+                            # AlterField for the subsequent RenameField to be a
+                            # noop on attempts at preserving the old name.
+                            if old_field.db_column != field.db_column:
+                                altered_field = field.clone()
+                                altered_field.name = rem_field_name
+                                self.add_operation(
+                                    app_label,
+                                    operations.AlterField(
+                                        model_name=model_name,
+                                        name=rem_field_name,
+                                        field=altered_field,
+                                    ),
+                                )
                             self.add_operation(
                                 app_label,
                                 operations.RenameField(
@@ -970,7 +984,10 @@ def generate_altered_fields(self):
                     new_field.remote_field.through = old_field.remote_field.through
             old_field_dec = self.deep_deconstruct(old_field)
             new_field_dec = self.deep_deconstruct(new_field)
-            if old_field_dec != new_field_dec:
+            # If the field was confirmed to be renamed it means that only
+            # db_column was allowed to change which generate_renamed_fields()
+            # already accounts for by adding an AlterField operation.
+            if old_field_dec != new_field_dec and old_field_name == field_name:
                 both_m2m = old_field.many_to_many and new_field.many_to_many
                 neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                 if both_m2m or neither_m2m:
diff --git a/django/db/migrations/operations/fields.py b/django/db/migrations/operations/fields.py
--- a/django/db/migrations/operations/fields.py
+++ b/django/db/migrations/operations/fields.py
@@ -239,7 +239,11 @@ def migration_name_fragment(self):
     def reduce(self, operation, app_label):
         if isinstance(operation, RemoveField) and self.is_same_field_operation(operation):
             return [operation]
-        elif isinstance(operation, RenameField) and self.is_same_field_operation(operation):
+        elif (
+            isinstance(operation, RenameField) and
+            self.is_same_field_operation(operation) and
+            self.field.db_column is None
+        ):
             return [
                 operation,
                 AlterField(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/autodetector.py | 843 | 843 | 108 | 9 | 37304
| django/db/migrations/autodetector.py | 973 | 973 | 19 | 9 | 7191
| django/db/migrations/operations/fields.py | 242 | 242 | 11 | 1 | 4089


## Problem Statement

```
Renaming field and providing prior field name to db_column should be an SQL noop
Description
	 
		(last modified by Jacob Walls)
	 
Renaming a field and setting the prior implicit field name as the db_column to avoid db operations creates a migration emitting unnecessary SQL. Similar to #31826, which handled a very similar scenario but where there is no field rename, I would expect a SQL noop. I tested with SQLite and MySQL 5.7.31. 
class Apple(models.Model):
	core = models.BooleanField()
class Apple(models.Model):
	core_renamed = models.BooleanField(db_column='core')
Was apple.core renamed to apple.core_renamed (a BooleanField)? [y/N] y
Migrations for 'renamez':
 renamez/migrations/0002_rename_core_apple_core_renamed_and_more.py
	- Rename field core on apple to core_renamed
	- Alter field core_renamed on apple
python manage.py sqlmigrate renamez 0002 showing unnecessary SQL:
BEGIN;
--
-- Rename field core on apple to core_renamed
--
ALTER TABLE "renamez_apple" RENAME COLUMN "core" TO "core_renamed";
--
-- Alter field core_renamed on apple
--
ALTER TABLE "renamez_apple" RENAME COLUMN "core_renamed" TO "core";
COMMIT;
Without renaming the field, follow the same flow and get an AlterField migration without SQL, which is what #31826 intended:
BEGIN;
--
-- Alter field core on apple
--
COMMIT;

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/migrations/operations/fields.py** | 254 | 320| 519 | 519 | 2477 | 
| 2 | 2 django/db/migrations/operations/models.py | 319 | 368| 493 | 1012 | 8927 | 
| 3 | 3 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 1517 | 13130 | 
| 4 | 4 django/db/migrations/questioner.py | 198 | 216| 233 | 1750 | 15654 | 
| 5 | 4 django/db/migrations/operations/models.py | 370 | 390| 213 | 1963 | 15654 | 
| 6 | **4 django/db/migrations/operations/fields.py** | 322 | 342| 160 | 2123 | 15654 | 
| 7 | 4 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 2545 | 15654 | 
| 8 | 5 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 3023 | 18287 | 
| 9 | 6 django/db/migrations/state.py | 259 | 309| 468 | 3491 | 26150 | 
| 10 | 6 django/db/migrations/operations/models.py | 416 | 466| 410 | 3901 | 26150 | 
| **-> 11 <-** | **6 django/db/migrations/operations/fields.py** | 229 | 251| 188 | 4089 | 26150 | 
| 12 | 6 django/db/migrations/questioner.py | 172 | 196| 235 | 4324 | 26150 | 
| 13 | 7 django/db/backends/base/schema.py | 684 | 754| 799 | 5123 | 39032 | 
| 14 | 7 django/db/migrations/questioner.py | 56 | 86| 255 | 5378 | 39032 | 
| 15 | 8 django/db/backends/mysql/schema.py | 41 | 50| 134 | 5512 | 40606 | 
| 16 | 8 django/db/backends/mysql/schema.py | 1 | 39| 428 | 5940 | 40606 | 
| 17 | **8 django/db/migrations/operations/fields.py** | 217 | 227| 146 | 6086 | 40606 | 
| 18 | 8 django/db/backends/base/schema.py | 468 | 489| 234 | 6320 | 40606 | 
| **-> 19 <-** | **9 django/db/migrations/autodetector.py** | 921 | 1001| 871 | 7191 | 52291 | 
| 20 | 9 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 8011 | 52291 | 
| 21 | 9 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 8192 | 52291 | 
| 22 | 9 django/db/backends/base/schema.py | 1151 | 1170| 175 | 8367 | 52291 | 
| 23 | 9 django/db/backends/sqlite3/schema.py | 386 | 419| 358 | 8725 | 52291 | 
| 24 | 9 django/db/backends/base/schema.py | 755 | 833| 826 | 9551 | 52291 | 
| 25 | 9 django/db/migrations/operations/models.py | 392 | 413| 170 | 9721 | 52291 | 
| 26 | 9 django/db/backends/base/schema.py | 621 | 683| 700 | 10421 | 52291 | 
| 27 | **9 django/db/migrations/operations/fields.py** | 184 | 215| 191 | 10612 | 52291 | 
| 28 | 9 django/db/migrations/questioner.py | 260 | 305| 361 | 10973 | 52291 | 
| 29 | 9 django/db/migrations/operations/models.py | 580 | 596| 215 | 11188 | 52291 | 
| 30 | 10 django/db/backends/oracle/schema.py | 84 | 139| 715 | 11903 | 54434 | 
| 31 | 10 django/db/migrations/state.py | 232 | 257| 240 | 12143 | 54434 | 
| 32 | 10 django/db/backends/mysql/schema.py | 142 | 160| 192 | 12335 | 54434 | 
| 33 | 10 django/db/migrations/operations/models.py | 289 | 317| 162 | 12497 | 54434 | 
| 34 | 10 django/db/backends/base/schema.py | 834 | 874| 506 | 13003 | 54434 | 
| 35 | 10 django/db/backends/oracle/schema.py | 152 | 206| 493 | 13496 | 54434 | 
| 36 | 10 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 14227 | 54434 | 
| 37 | **10 django/db/migrations/operations/fields.py** | 107 | 117| 127 | 14354 | 54434 | 
| 38 | 10 django/db/backends/oracle/schema.py | 62 | 82| 249 | 14603 | 54434 | 
| 39 | 11 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 14798 | 54629 | 
| 40 | 11 django/db/backends/base/schema.py | 968 | 987| 296 | 15094 | 54629 | 
| 41 | **11 django/db/migrations/operations/fields.py** | 93 | 105| 130 | 15224 | 54629 | 
| 42 | **11 django/db/migrations/operations/fields.py** | 1 | 36| 227 | 15451 | 54629 | 
| 43 | 12 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 16098 | 56797 | 
| 44 | 13 django/db/backends/oracle/operations.py | 481 | 500| 240 | 16338 | 62790 | 
| 45 | 13 django/db/migrations/state.py | 133 | 168| 391 | 16729 | 62790 | 
| 46 | 13 django/db/migrations/operations/models.py | 598 | 615| 163 | 16892 | 62790 | 
| 47 | 14 django/db/models/sql/query.py | 840 | 877| 383 | 17275 | 85126 | 
| 48 | 14 django/db/migrations/questioner.py | 151 | 170| 185 | 17460 | 85126 | 
| 49 | 14 django/db/backends/mysql/schema.py | 96 | 106| 138 | 17598 | 85126 | 
| 50 | **14 django/db/migrations/autodetector.py** | 462 | 514| 465 | 18063 | 85126 | 
| 51 | 14 django/db/migrations/state.py | 207 | 230| 247 | 18310 | 85126 | 
| 52 | 14 django/db/backends/oracle/operations.py | 21 | 73| 574 | 18884 | 85126 | 
| 53 | 14 django/db/backends/oracle/operations.py | 465 | 479| 203 | 19087 | 85126 | 
| 54 | 14 django/db/backends/oracle/operations.py | 337 | 348| 226 | 19313 | 85126 | 
| 55 | 14 django/db/backends/base/schema.py | 1 | 29| 209 | 19522 | 85126 | 
| 56 | 14 django/db/backends/oracle/operations.py | 412 | 463| 516 | 20038 | 85126 | 
| 57 | 14 django/db/backends/oracle/operations.py | 585 | 600| 221 | 20259 | 85126 | 
| 58 | 14 django/db/migrations/operations/models.py | 618 | 674| 325 | 20584 | 85126 | 
| 59 | 15 django/db/models/base.py | 1497 | 1520| 176 | 20760 | 102456 | 
| 60 | 15 django/db/backends/oracle/schema.py | 1 | 44| 454 | 21214 | 102456 | 
| 61 | 15 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 21565 | 102456 | 
| 62 | **15 django/db/migrations/autodetector.py** | 900 | 919| 184 | 21749 | 102456 | 
| 63 | **15 django/db/migrations/operations/fields.py** | 38 | 60| 183 | 21932 | 102456 | 
| 64 | 16 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 22311 | 103089 | 
| 65 | 16 django/db/backends/oracle/schema.py | 141 | 150| 142 | 22453 | 103089 | 
| 66 | 16 django/db/models/base.py | 915 | 947| 385 | 22838 | 103089 | 
| 67 | 17 django/db/backends/sqlite3/operations.py | 180 | 205| 190 | 23028 | 106361 | 
| 68 | 17 django/db/migrations/operations/models.py | 511 | 528| 168 | 23196 | 106361 | 
| 69 | 17 django/db/backends/base/schema.py | 1189 | 1211| 199 | 23395 | 106361 | 
| 70 | 17 django/db/migrations/questioner.py | 238 | 257| 195 | 23590 | 106361 | 
| 71 | **17 django/db/migrations/operations/fields.py** | 142 | 181| 344 | 23934 | 106361 | 
| 72 | 17 django/db/models/base.py | 1875 | 1948| 572 | 24506 | 106361 | 
| 73 | 18 django/db/models/fields/related.py | 267 | 298| 284 | 24790 | 120357 | 
| 74 | **18 django/db/migrations/operations/fields.py** | 119 | 139| 129 | 24919 | 120357 | 
| 75 | 18 django/db/migrations/operations/models.py | 1 | 38| 235 | 25154 | 120357 | 
| 76 | 19 django/db/backends/sqlite3/features.py | 1 | 127| 1163 | 26317 | 121520 | 
| 77 | 20 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 26434 | 121637 | 
| 78 | 20 django/db/models/base.py | 1440 | 1495| 491 | 26925 | 121637 | 
| 79 | 20 django/db/models/sql/query.py | 1699 | 1741| 436 | 27361 | 121637 | 
| 80 | 20 django/db/models/sql/query.py | 365 | 415| 494 | 27855 | 121637 | 
| 81 | 20 django/db/backends/oracle/operations.py | 602 | 621| 303 | 28158 | 121637 | 
| 82 | 20 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 28417 | 121637 | 
| 83 | 20 django/db/models/sql/query.py | 2094 | 2116| 229 | 28646 | 121637 | 
| 84 | 20 django/db/backends/base/schema.py | 548 | 576| 289 | 28935 | 121637 | 
| 85 | 20 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 29153 | 121637 | 
| 86 | 20 django/db/backends/oracle/operations.py | 152 | 169| 309 | 29462 | 121637 | 
| 87 | 21 django/db/backends/oracle/features.py | 1 | 121| 1032 | 30494 | 122670 | 
| 88 | **21 django/db/migrations/autodetector.py** | 1114 | 1151| 317 | 30811 | 122670 | 
| 89 | 21 django/db/migrations/operations/models.py | 849 | 885| 331 | 31142 | 122670 | 
| 90 | 21 django/db/backends/sqlite3/schema.py | 332 | 348| 173 | 31315 | 122670 | 
| 91 | 21 django/db/backends/base/schema.py | 578 | 619| 489 | 31804 | 122670 | 
| 92 | 21 django/db/backends/oracle/operations.py | 92 | 102| 222 | 32026 | 122670 | 
| 93 | 21 django/db/backends/mysql/schema.py | 124 | 140| 205 | 32231 | 122670 | 
| 94 | 21 django/db/migrations/operations/models.py | 124 | 247| 853 | 33084 | 122670 | 
| 95 | 21 django/db/backends/oracle/operations.py | 171 | 182| 227 | 33311 | 122670 | 
| 96 | 22 django/db/backends/mysql/compiler.py | 42 | 72| 241 | 33552 | 123264 | 
| 97 | 23 django/contrib/admin/utils.py | 289 | 307| 175 | 33727 | 127422 | 
| 98 | 23 django/db/models/fields/related.py | 198 | 266| 687 | 34414 | 127422 | 
| 99 | 24 django/db/backends/mysql/operations.py | 222 | 279| 437 | 34851 | 131149 | 
| 100 | 24 django/db/backends/sqlite3/operations.py | 207 | 226| 209 | 35060 | 131149 | 
| 101 | 24 django/db/migrations/operations/models.py | 500 | 509| 129 | 35189 | 131149 | 
| 102 | 24 django/db/backends/oracle/operations.py | 277 | 291| 206 | 35395 | 131149 | 
| 103 | 24 django/db/models/fields/related.py | 139 | 166| 201 | 35596 | 131149 | 
| 104 | 25 django/db/models/sql/compiler.py | 1080 | 1120| 337 | 35933 | 145941 | 
| 105 | 25 django/db/models/sql/query.py | 2069 | 2092| 257 | 36190 | 145941 | 
| 106 | 25 django/db/models/sql/compiler.py | 1605 | 1645| 406 | 36596 | 145941 | 
| 107 | 25 django/db/backends/mysql/compiler.py | 1 | 15| 132 | 36728 | 145941 | 
| **-> 108 <-** | **25 django/db/migrations/autodetector.py** | 804 | 854| 576 | 37304 | 145941 | 
| 109 | 25 django/db/backends/base/schema.py | 876 | 898| 185 | 37489 | 145941 | 
| 110 | 25 django/db/backends/base/schema.py | 491 | 546| 613 | 38102 | 145941 | 
| 111 | 26 django/db/migrations/operations/special.py | 63 | 114| 390 | 38492 | 147499 | 
| 112 | 26 django/db/models/fields/related.py | 1631 | 1672| 497 | 38989 | 147499 | 
| 113 | 27 django/contrib/contenttypes/management/__init__.py | 1 | 43| 357 | 39346 | 148474 | 
| 114 | 27 django/db/backends/base/schema.py | 51 | 119| 785 | 40131 | 148474 | 
| 115 | 27 django/db/backends/oracle/operations.py | 350 | 371| 228 | 40359 | 148474 | 
| 116 | 27 django/db/backends/oracle/operations.py | 373 | 410| 369 | 40728 | 148474 | 
| 117 | 28 django/contrib/gis/db/backends/spatialite/schema.py | 128 | 169| 376 | 41104 | 149826 | 
| 118 | 28 django/db/migrations/questioner.py | 218 | 236| 177 | 41281 | 149826 | 
| 119 | 28 django/db/models/sql/compiler.py | 441 | 464| 234 | 41515 | 149826 | 
| 120 | 28 django/db/backends/oracle/operations.py | 117 | 131| 229 | 41744 | 149826 | 
| 121 | **28 django/db/migrations/autodetector.py** | 89 | 101| 116 | 41860 | 149826 | 
| 122 | 28 django/db/backends/base/schema.py | 213 | 259| 445 | 42305 | 149826 | 
| 123 | 28 django/db/models/sql/query.py | 1470 | 1555| 801 | 43106 | 149826 | 
| 124 | 29 django/core/management/commands/migrate.py | 71 | 160| 774 | 43880 | 153163 | 
| 125 | 30 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 44534 | 155036 | 
| 126 | **30 django/db/migrations/autodetector.py** | 856 | 898| 394 | 44928 | 155036 | 
| 127 | 31 django/db/models/fields/reverse_related.py | 160 | 178| 167 | 45095 | 157356 | 
| 128 | 31 django/db/migrations/operations/special.py | 181 | 204| 246 | 45341 | 157356 | 
| 129 | 31 django/db/backends/mysql/operations.py | 1 | 35| 282 | 45623 | 157356 | 
| 130 | 31 django/db/migrations/state.py | 397 | 413| 199 | 45822 | 157356 | 
| 131 | 31 django/db/backends/sqlite3/operations.py | 330 | 387| 545 | 46367 | 157356 | 
| 132 | 31 django/db/backends/mysql/schema.py | 108 | 122| 143 | 46510 | 157356 | 
| 133 | **31 django/db/migrations/operations/fields.py** | 63 | 91| 171 | 46681 | 157356 | 
| 134 | 31 django/core/management/commands/migrate.py | 162 | 227| 632 | 47313 | 157356 | 
| 135 | 31 django/core/management/commands/migrate.py | 228 | 279| 537 | 47850 | 157356 | 
| 136 | 32 django/db/migrations/operations/base.py | 1 | 109| 804 | 48654 | 158386 | 
| 137 | **32 django/db/migrations/autodetector.py** | 1153 | 1174| 231 | 48885 | 158386 | 
| 138 | 33 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 49102 | 158603 | 
| 139 | 34 django/core/management/commands/flush.py | 27 | 83| 486 | 49588 | 159290 | 
| 140 | 34 django/db/backends/base/schema.py | 278 | 306| 205 | 49793 | 159290 | 
| 141 | 34 django/db/models/fields/related.py | 300 | 334| 293 | 50086 | 159290 | 
| 142 | 34 django/db/backends/oracle/operations.py | 133 | 150| 291 | 50377 | 159290 | 
| 143 | 34 django/db/backends/base/schema.py | 1364 | 1396| 293 | 50670 | 159290 | 
| 144 | 34 django/db/backends/base/schema.py | 32 | 48| 166 | 50836 | 159290 | 
| 145 | 35 django/db/backends/ddl_references.py | 204 | 233| 247 | 51083 | 160889 | 
| 146 | 35 django/db/backends/mysql/operations.py | 195 | 220| 215 | 51298 | 160889 | 
| 147 | 36 django/db/backends/oracle/creation.py | 130 | 165| 399 | 51697 | 164782 | 
| 148 | 37 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 51697 | 164859 | 
| 149 | 37 django/db/models/fields/related.py | 1674 | 1690| 286 | 51983 | 164859 | 
| 150 | 37 django/db/backends/oracle/operations.py | 308 | 335| 271 | 52254 | 164859 | 
| 151 | 38 django/db/migrations/executor.py | 289 | 382| 843 | 53097 | 168213 | 
| 152 | 38 django/db/models/sql/compiler.py | 1262 | 1296| 332 | 53429 | 168213 | 
| 153 | 38 django/db/backends/oracle/operations.py | 213 | 261| 411 | 53840 | 168213 | 
| 154 | 39 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 53840 | 168281 | 
| 155 | 39 django/db/models/sql/compiler.py | 1371 | 1430| 617 | 54457 | 168281 | 
| 156 | 40 django/contrib/gis/db/backends/mysql/schema.py | 25 | 38| 146 | 54603 | 168902 | 
| 157 | 41 django/db/backends/base/features.py | 1 | 112| 895 | 55498 | 171909 | 
| 158 | 41 django/db/backends/oracle/operations.py | 502 | 518| 190 | 55688 | 171909 | 
| 159 | 42 django/db/backends/sqlite3/base.py | 312 | 401| 850 | 56538 | 177979 | 
| 160 | 42 django/db/backends/base/schema.py | 415 | 429| 183 | 56721 | 177979 | 
| 161 | 42 django/db/backends/mysql/schema.py | 52 | 94| 388 | 57109 | 177979 | 
| 162 | 43 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 57377 | 178247 | 
| 163 | 44 django/db/models/fields/__init__.py | 1 | 82| 636 | 58013 | 196415 | 
| 164 | 44 django/db/backends/oracle/schema.py | 46 | 60| 133 | 58146 | 196415 | 
| 165 | **44 django/db/migrations/autodetector.py** | 1212 | 1237| 245 | 58391 | 196415 | 
| 166 | 44 django/db/backends/base/schema.py | 900 | 935| 267 | 58658 | 196415 | 
| 167 | 44 django/db/backends/mysql/operations.py | 341 | 356| 217 | 58875 | 196415 | 
| 168 | 45 django/db/backends/postgresql/operations.py | 138 | 158| 221 | 59096 | 198976 | 
| 169 | 45 django/db/backends/oracle/operations.py | 104 | 115| 212 | 59308 | 198976 | 
| 170 | 45 django/db/models/fields/related.py | 772 | 790| 222 | 59530 | 198976 | 
| 171 | 45 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 60321 | 198976 | 
| 172 | 45 django/db/models/sql/query.py | 879 | 902| 203 | 60524 | 198976 | 
| 173 | 46 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 60674 | 199126 | 


### Hint

```
I think this is due to how the auto-detector ​generates field renames before field alterations. I assume the issue goes away if you swap the order of operations in your migration? ​As this test exemplifies the RenameField operation happens before the AlterField operation does so we'd need to adjust generate_renamed_fields to add an AlterField and prevents generate_altered_fields for doing so when this happens.
Thanks for having a look. I see now the scope of #31826 was just for flows where the field is not renamed. So that makes this ticket a request to extend this to field renames, which looks like was discussed as 3 and 4 here. I assume the issue goes away if you swap the order of operations in your migration? If I switch the order to have AlterField followed by RenameField, FieldDoesNotExist is raised when migrating. These are the operations: operations = [ migrations.RenameField( model_name='apple', old_name='core', new_name='core_renamed', ), migrations.AlterField( model_name='apple', name='core_renamed', field=models.BooleanField(db_column='core'), ), ]
You'll want to adjust AlterField.name accordingly if you swap the order of operations; change name='core_renamed' to name='core'. operations = [ migrations.AlterField( model_name='apple', name='core', field=models.BooleanField(db_column='core'), ), migrations.RenameField( model_name='apple', old_name='core', new_name='core_renamed', ), ]
```

## Patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -840,6 +840,20 @@ def generate_renamed_fields(self):
                             old_field_dec[0:2] == field_dec[0:2] and
                             dict(old_field_dec[2], db_column=old_db_column) == field_dec[2])):
                         if self.questioner.ask_rename(model_name, rem_field_name, field_name, field):
+                            # A db_column mismatch requires a prior noop
+                            # AlterField for the subsequent RenameField to be a
+                            # noop on attempts at preserving the old name.
+                            if old_field.db_column != field.db_column:
+                                altered_field = field.clone()
+                                altered_field.name = rem_field_name
+                                self.add_operation(
+                                    app_label,
+                                    operations.AlterField(
+                                        model_name=model_name,
+                                        name=rem_field_name,
+                                        field=altered_field,
+                                    ),
+                                )
                             self.add_operation(
                                 app_label,
                                 operations.RenameField(
@@ -970,7 +984,10 @@ def generate_altered_fields(self):
                     new_field.remote_field.through = old_field.remote_field.through
             old_field_dec = self.deep_deconstruct(old_field)
             new_field_dec = self.deep_deconstruct(new_field)
-            if old_field_dec != new_field_dec:
+            # If the field was confirmed to be renamed it means that only
+            # db_column was allowed to change which generate_renamed_fields()
+            # already accounts for by adding an AlterField operation.
+            if old_field_dec != new_field_dec and old_field_name == field_name:
                 both_m2m = old_field.many_to_many and new_field.many_to_many
                 neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                 if both_m2m or neither_m2m:
diff --git a/django/db/migrations/operations/fields.py b/django/db/migrations/operations/fields.py
--- a/django/db/migrations/operations/fields.py
+++ b/django/db/migrations/operations/fields.py
@@ -239,7 +239,11 @@ def migration_name_fragment(self):
     def reduce(self, operation, app_label):
         if isinstance(operation, RemoveField) and self.is_same_field_operation(operation):
             return [operation]
-        elif isinstance(operation, RenameField) and self.is_same_field_operation(operation):
+        elif (
+            isinstance(operation, RenameField) and
+            self.is_same_field_operation(operation) and
+            self.field.db_column is None
+        ):
             return [
                 operation,
                 AlterField(

```

## Test Patch

```diff
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -1001,14 +1001,17 @@ def test_rename_field_preserved_db_column(self):
         ]
         changes = self.get_changes(before, after, MigrationQuestioner({'ask_rename': True}))
         self.assertNumberMigrations(changes, 'app', 1)
-        self.assertOperationTypes(changes, 'app', 0, ['RenameField', 'AlterField'])
+        self.assertOperationTypes(changes, 'app', 0, ['AlterField', 'RenameField'])
         self.assertOperationAttributes(
-            changes, 'app', 0, 0, model_name='foo', old_name='field', new_name='renamed_field',
+            changes, 'app', 0, 0, model_name='foo', name='field',
         )
-        self.assertOperationAttributes(changes, 'app', 0, 1, model_name='foo', name='renamed_field')
-        self.assertEqual(changes['app'][0].operations[-1].field.deconstruct(), (
-            'renamed_field', 'django.db.models.IntegerField', [], {'db_column': 'field'},
+        self.assertEqual(changes['app'][0].operations[0].field.deconstruct(), (
+            'field', 'django.db.models.IntegerField', [], {'db_column': 'field'},
         ))
+        self.assertOperationAttributes(
+            changes, 'app', 0, 1, model_name='foo', old_name='field',
+            new_name='renamed_field',
+        )
 
     def test_rename_related_field_preserved_db_column(self):
         before = [
@@ -1031,17 +1034,20 @@ def test_rename_related_field_preserved_db_column(self):
         ]
         changes = self.get_changes(before, after, MigrationQuestioner({'ask_rename': True}))
         self.assertNumberMigrations(changes, 'app', 1)
-        self.assertOperationTypes(changes, 'app', 0, ['RenameField', 'AlterField'])
+        self.assertOperationTypes(changes, 'app', 0, ['AlterField', 'RenameField'])
         self.assertOperationAttributes(
-            changes, 'app', 0, 0, model_name='bar', old_name='foo', new_name='renamed_foo',
+            changes, 'app', 0, 0, model_name='bar', name='foo',
         )
-        self.assertOperationAttributes(changes, 'app', 0, 1, model_name='bar', name='renamed_foo')
-        self.assertEqual(changes['app'][0].operations[-1].field.deconstruct(), (
-            'renamed_foo',
+        self.assertEqual(changes['app'][0].operations[0].field.deconstruct(), (
+            'foo',
             'django.db.models.ForeignKey',
             [],
             {'to': 'app.foo', 'on_delete': models.CASCADE, 'db_column': 'foo_id'},
         ))
+        self.assertOperationAttributes(
+            changes, 'app', 0, 1, model_name='bar', old_name='foo',
+            new_name='renamed_foo',
+        )
 
     def test_rename_model(self):
         """Tests autodetection of renamed models."""

```


## Code snippets

### 1 - django/db/migrations/operations/fields.py:

Start line: 254, End line: 320

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
            'model_name': self.model_name,
            'old_name': self.old_name,
            'new_name': self.new_name,
        }
        return (
            self.__class__.__name__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.rename_field(app_label, self.model_name_lower, self.old_name, self.new_name)

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
        return "Rename field %s on %s to %s" % (self.old_name, self.model_name, self.new_name)

    @property
    def migration_name_fragment(self):
        return 'rename_%s_%s_%s' % (
            self.old_name_lower,
            self.model_name_lower,
            self.new_name_lower,
        )

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (
            name.lower() == self.old_name_lower or
            name.lower() == self.new_name_lower
        )
```
### 2 - django/db/migrations/operations/models.py:

Start line: 319, End line: 368

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
                to_field = to_state.apps.get_model(
                    *related_key
                )._meta.get_field(related_object.field.name)
                schema_editor.alter_field(
                    model,
                    related_object.field,
                    to_field,
                )
            # Rename M2M fields whose name is based on this model's name.
            fields = zip(old_model._meta.local_many_to_many, new_model._meta.local_many_to_many)
            for (old_field, new_field) in fields:
                # Skip self-referential fields as these are renamed above.
                if new_field.model == new_field.related_model or not new_field.remote_field.through._meta.auto_created:
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
### 3 - django/db/backends/sqlite3/schema.py:

Start line: 101, End line: 140

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        if not self._field_should_be_altered(old_field, new_field):
            return
        old_field_name = old_field.name
        table_name = model._meta.db_table
        _, old_column_name = old_field.get_attname_column()
        if (new_field.name != old_field_name and
                not self.connection.features.supports_atomic_references_rename and
                self._is_referenced_by_fk_constraint(table_name, old_column_name, ignore_self=True)):
            if self.connection.in_atomic_block:
                raise NotSupportedError((
                    'Renaming the %r.%r column while in a transaction is not '
                    'supported on SQLite < 3.26 because it would break referential '
                    'integrity. Try adding `atomic = False` to the Migration class.'
                ) % (model._meta.db_table, old_field_name))
            with atomic(self.connection.alias):
                super().alter_field(model, old_field, new_field, strict=strict)
                # Follow SQLite's documented procedure for performing changes
                # that don't affect the on-disk content.
                # https://sqlite.org/lang_altertable.html#otheralter
                with self.connection.cursor() as cursor:
                    schema_version = cursor.execute('PRAGMA schema_version').fetchone()[0]
                    cursor.execute('PRAGMA writable_schema = 1')
                    references_template = ' REFERENCES "%s" ("%%s") ' % table_name
                    new_column_name = new_field.get_attname_column()[1]
                    search = references_template % old_column_name
                    replacement = references_template % new_column_name
                    cursor.execute('UPDATE sqlite_master SET sql = replace(sql, %s, %s)', (search, replacement))
                    cursor.execute('PRAGMA schema_version = %d' % (schema_version + 1))
                    cursor.execute('PRAGMA writable_schema = 0')
                    # The integrity check will raise an exception and rollback
                    # the transaction if the sqlite_master updates corrupt the
                    # database.
                    cursor.execute('PRAGMA integrity_check')
            # Perform a VACUUM to refresh the database representation from
            # the sqlite_master table.
            with self.connection.cursor() as cursor:
                cursor.execute('VACUUM')
        else:
            super().alter_field(model, old_field, new_field, strict=strict)
```
### 4 - django/db/migrations/questioner.py:

Start line: 198, End line: 216

```python
class InteractiveMigrationQuestioner(MigrationQuestioner):

    def ask_rename(self, model_name, old_name, new_name, field_instance):
        """Was this field really renamed?"""
        msg = 'Was %s.%s renamed to %s.%s (a %s)? [y/N]'
        return self._boolean_input(msg % (model_name, old_name, model_name, new_name,
                                          field_instance.__class__.__name__), False)

    def ask_rename_model(self, old_model_state, new_model_state):
        """Was this model really renamed?"""
        msg = 'Was the model %s.%s renamed to %s? [y/N]'
        return self._boolean_input(msg % (old_model_state.app_label, old_model_state.name,
                                          new_model_state.name), False)

    def ask_merge(self, app_label):
        return self._boolean_input(
            "\nMerging will only work if the operations printed above do not conflict\n" +
            "with each other (working on different fields or models)\n" +
            'Should these migration branches be merged? [y/N]',
            False,
        )
```
### 5 - django/db/migrations/operations/models.py:

Start line: 370, End line: 390

```python
class RenameModel(ModelOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.new_name_lower, self.old_name_lower = self.old_name_lower, self.new_name_lower
        self.new_name, self.old_name = self.old_name, self.new_name

        self.database_forwards(app_label, schema_editor, from_state, to_state)

        self.new_name_lower, self.old_name_lower = self.old_name_lower, self.new_name_lower
        self.new_name, self.old_name = self.old_name, self.new_name

    def references_model(self, name, app_label):
        return (
            name.lower() == self.old_name_lower or
            name.lower() == self.new_name_lower
        )

    def describe(self):
        return "Rename model %s to %s" % (self.old_name, self.new_name)

    @property
    def migration_name_fragment(self):
        return 'rename_%s_%s' % (self.old_name_lower, self.new_name_lower)
```
### 6 - django/db/migrations/operations/fields.py:

Start line: 322, End line: 342

```python
class RenameField(FieldOperation):

    def reduce(self, operation, app_label):
        if (isinstance(operation, RenameField) and
                self.is_same_model_operation(operation) and
                self.new_name_lower == operation.old_name_lower):
            return [
                RenameField(
                    self.model_name,
                    self.old_name,
                    operation.new_name,
                ),
            ]
        # Skip `FieldOperation.reduce` as we want to run `references_field`
        # against self.old_name and self.new_name.
        return (
            super(FieldOperation, self).reduce(operation, app_label) or
            not (
                operation.references_field(self.model_name, self.old_name, app_label) or
                operation.references_field(self.model_name, self.new_name, app_label)
            )
        )
```
### 7 - django/db/backends/sqlite3/schema.py:

Start line: 350, End line: 384

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        """Perform a "physical" (non-ManyToMany) field update."""
        # Use "ALTER TABLE ... RENAME COLUMN" if only the column name
        # changed and there aren't any constraints.
        if (self.connection.features.can_alter_table_rename_column and
            old_field.column != new_field.column and
            self.column_sql(model, old_field) == self.column_sql(model, new_field) and
            not (old_field.remote_field and old_field.db_constraint or
                 new_field.remote_field and new_field.db_constraint)):
            return self.execute(self._rename_field_sql(model._meta.db_table, old_field, new_field, new_type))
        # Alter by remaking table
        self._remake_table(model, alter_field=(old_field, new_field))
        # Rebuild tables with FKs pointing to this field.
        if new_field.unique and old_type != new_type:
            related_models = set()
            opts = new_field.model._meta
            for remote_field in opts.related_objects:
                # Ignore self-relationship since the table was already rebuilt.
                if remote_field.related_model == model:
                    continue
                if not remote_field.many_to_many:
                    if remote_field.field_name == new_field.name:
                        related_models.add(remote_field.related_model)
                elif new_field.primary_key and remote_field.through._meta.auto_created:
                    related_models.add(remote_field.through)
            if new_field.primary_key:
                for many_to_many in opts.many_to_many:
                    # Ignore self-relationship since the table was already rebuilt.
                    if many_to_many.related_model == model:
                        continue
                    if many_to_many.remote_field.through._meta.auto_created:
                        related_models.add(many_to_many.remote_field.through)
            for related_model in related_models:
                self._remake_table(related_model)
```
### 8 - django/core/management/commands/inspectdb.py:

Start line: 175, End line: 229

```python
class Command(BaseCommand):

    def normalize_col_name(self, col_name, used_column_names, is_relation):
        """
        Modify the column name to make it Python-compatible as a field name
        """
        field_params = {}
        field_notes = []

        new_name = col_name.lower()
        if new_name != col_name:
            field_notes.append('Field name made lowercase.')

        if is_relation:
            if new_name.endswith('_id'):
                new_name = new_name[:-3]
            else:
                field_params['db_column'] = col_name

        new_name, num_repl = re.subn(r'\W', '_', new_name)
        if num_repl > 0:
            field_notes.append('Field renamed to remove unsuitable characters.')

        if new_name.find(LOOKUP_SEP) >= 0:
            while new_name.find(LOOKUP_SEP) >= 0:
                new_name = new_name.replace(LOOKUP_SEP, '_')
            if col_name.lower().find(LOOKUP_SEP) >= 0:
                # Only add the comment if the double underscore was in the original name
                field_notes.append("Field renamed because it contained more than one '_' in a row.")

        if new_name.startswith('_'):
            new_name = 'field%s' % new_name
            field_notes.append("Field renamed because it started with '_'.")

        if new_name.endswith('_'):
            new_name = '%sfield' % new_name
            field_notes.append("Field renamed because it ended with '_'.")

        if keyword.iskeyword(new_name):
            new_name += '_field'
            field_notes.append('Field renamed because it was a Python reserved word.')

        if new_name[0].isdigit():
            new_name = 'number_%s' % new_name
            field_notes.append("Field renamed because it wasn't a valid Python identifier.")

        if new_name in used_column_names:
            num = 0
            while '%s_%d' % (new_name, num) in used_column_names:
                num += 1
            new_name = '%s_%d' % (new_name, num)
            field_notes.append('Field renamed because of name conflict.')

        if col_name != new_name and field_notes:
            field_params['db_column'] = col_name

        return new_name, field_params, field_notes
```
### 9 - django/db/migrations/state.py:

Start line: 259, End line: 309

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
            from_fields = getattr(field, 'from_fields', None)
            if from_fields:
                field.from_fields = tuple([
                    new_name if from_field_name == old_name else from_field_name
                    for from_field_name in from_fields
                ])
        # Fix index/unique_together to refer to the new field.
        options = model_state.options
        for option in ('index_together', 'unique_together'):
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
                if getattr(remote_field, 'field_name', None) == old_name:
                    remote_field.field_name = new_name
                if to_fields:
                    field.to_fields = tuple([
                        new_name if to_field_name == old_name else to_field_name
                        for to_field_name in to_fields
                    ])
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
### 10 - django/db/migrations/operations/models.py:

Start line: 416, End line: 466

```python
class AlterModelTable(ModelOptionOperation):
    """Rename a model's table."""

    def __init__(self, name, table):
        self.table = table
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            'name': self.name,
            'table': self.table,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.alter_model_options(app_label, self.name_lower, {'db_table': self.table})

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
            for (old_field, new_field) in zip(old_model._meta.local_many_to_many, new_model._meta.local_many_to_many):
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
            self.table if self.table is not None else "(default)"
        )

    @property
    def migration_name_fragment(self):
        return 'alter_%s_table' % self.name_lower
```
### 11 - django/db/migrations/operations/fields.py:

Start line: 229, End line: 251

```python
class AlterField(FieldOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.database_forwards(app_label, schema_editor, from_state, to_state)

    def describe(self):
        return "Alter field %s on %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return 'alter_%s_%s' % (self.model_name_lower, self.name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, RemoveField) and self.is_same_field_operation(operation):
            return [operation]
        elif isinstance(operation, RenameField) and self.is_same_field_operation(operation):
            return [
                operation,
                AlterField(
                    model_name=self.model_name,
                    name=operation.new_name,
                    field=self.field,
                ),
            ]
        return super().reduce(operation, app_label)
```
### 17 - django/db/migrations/operations/fields.py:

Start line: 217, End line: 227

```python
class AlterField(FieldOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            from_field = from_model._meta.get_field(self.name)
            to_field = to_model._meta.get_field(self.name)
            if not self.preserve_default:
                to_field.default = self.field.default
            schema_editor.alter_field(from_model, from_field, to_field)
            if not self.preserve_default:
                to_field.default = NOT_PROVIDED
```
### 19 - django/db/migrations/autodetector.py:

Start line: 921, End line: 1001

```python
class MigrationAutodetector:

    def generate_altered_fields(self):
        """
        Make AlterField operations, or possibly RemovedField/AddField if alter
        isn't possible.
        """
        for app_label, model_name, field_name in sorted(self.old_field_keys & self.new_field_keys):
            # Did the field change?
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_field_name = self.renamed_fields.get((app_label, model_name, field_name), field_name)
            old_field = self.from_state.models[app_label, old_model_name].get_field(old_field_name)
            new_field = self.to_state.models[app_label, model_name].get_field(field_name)
            dependencies = []
            # Implement any model renames on relations; these are handled by RenameModel
            # so we need to exclude them from the comparison
            if hasattr(new_field, "remote_field") and getattr(new_field.remote_field, "model", None):
                rename_key = resolve_relation(new_field.remote_field.model, app_label, model_name)
                if rename_key in self.renamed_models:
                    new_field.remote_field.model = old_field.remote_field.model
                # Handle ForeignKey which can only have a single to_field.
                remote_field_name = getattr(new_field.remote_field, 'field_name', None)
                if remote_field_name:
                    to_field_rename_key = rename_key + (remote_field_name,)
                    if to_field_rename_key in self.renamed_fields:
                        # Repoint both model and field name because to_field
                        # inclusion in ForeignKey.deconstruct() is based on
                        # both.
                        new_field.remote_field.model = old_field.remote_field.model
                        new_field.remote_field.field_name = old_field.remote_field.field_name
                # Handle ForeignObjects which can have multiple from_fields/to_fields.
                from_fields = getattr(new_field, 'from_fields', None)
                if from_fields:
                    from_rename_key = (app_label, model_name)
                    new_field.from_fields = tuple([
                        self.renamed_fields.get(from_rename_key + (from_field,), from_field)
                        for from_field in from_fields
                    ])
                    new_field.to_fields = tuple([
                        self.renamed_fields.get(rename_key + (to_field,), to_field)
                        for to_field in new_field.to_fields
                    ])
                dependencies.extend(self._get_dependencies_for_foreign_key(
                    app_label, model_name, new_field, self.to_state,
                ))
            if (
                hasattr(new_field, 'remote_field') and
                getattr(new_field.remote_field, 'through', None)
            ):
                rename_key = resolve_relation(new_field.remote_field.through, app_label, model_name)
                if rename_key in self.renamed_models:
                    new_field.remote_field.through = old_field.remote_field.through
            old_field_dec = self.deep_deconstruct(old_field)
            new_field_dec = self.deep_deconstruct(new_field)
            if old_field_dec != new_field_dec:
                both_m2m = old_field.many_to_many and new_field.many_to_many
                neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                if both_m2m or neither_m2m:
                    # Either both fields are m2m or neither is
                    preserve_default = True
                    if (old_field.null and not new_field.null and not new_field.has_default() and
                            not new_field.many_to_many):
                        field = new_field.clone()
                        new_default = self.questioner.ask_not_null_alteration(field_name, model_name)
                        if new_default is not models.NOT_PROVIDED:
                            field.default = new_default
                            preserve_default = False
                    else:
                        field = new_field
                    self.add_operation(
                        app_label,
                        operations.AlterField(
                            model_name=model_name,
                            name=field_name,
                            field=field,
                            preserve_default=preserve_default,
                        ),
                        dependencies=dependencies,
                    )
                else:
                    # We cannot alter between m2m and concrete fields
                    self._generate_removed_field(app_label, model_name, field_name)
                    self._generate_added_field(app_label, model_name, field_name)
```
### 27 - django/db/migrations/operations/fields.py:

Start line: 184, End line: 215

```python
class AlterField(FieldOperation):
    """
    Alter a field's database column (e.g. null, max_length) to the provided
    new field.
    """

    def __init__(self, model_name, name, field, preserve_default=True):
        self.preserve_default = preserve_default
        super().__init__(model_name, name, field)

    def deconstruct(self):
        kwargs = {
            'model_name': self.model_name,
            'name': self.name,
            'field': self.field,
        }
        if self.preserve_default is not True:
            kwargs['preserve_default'] = self.preserve_default
        return (
            self.__class__.__name__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.alter_field(
            app_label,
            self.model_name_lower,
            self.name,
            self.field,
            self.preserve_default,
        )
```
### 37 - django/db/migrations/operations/fields.py:

Start line: 107, End line: 117

```python
class AddField(FieldOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        from_model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, from_model):
            schema_editor.remove_field(from_model, from_model._meta.get_field(self.name))

    def describe(self):
        return "Add field %s to %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return '%s_%s' % (self.model_name_lower, self.name_lower)
```
### 41 - django/db/migrations/operations/fields.py:

Start line: 93, End line: 105

```python
class AddField(FieldOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            field = to_model._meta.get_field(self.name)
            if not self.preserve_default:
                field.default = self.field.default
            schema_editor.add_field(
                from_model,
                field,
            )
            if not self.preserve_default:
                field.default = NOT_PROVIDED
```
### 42 - django/db/migrations/operations/fields.py:

Start line: 1, End line: 36

```python
from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property

from .base import Operation


class FieldOperation(Operation):
    def __init__(self, model_name, name, field=None):
        self.model_name = model_name
        self.name = name
        self.field = field

    @cached_property
    def model_name_lower(self):
        return self.model_name.lower()

    @cached_property
    def name_lower(self):
        return self.name.lower()

    def is_same_model_operation(self, operation):
        return self.model_name_lower == operation.model_name_lower

    def is_same_field_operation(self, operation):
        return self.is_same_model_operation(operation) and self.name_lower == operation.name_lower

    def references_model(self, name, app_label):
        name_lower = name.lower()
        if name_lower == self.model_name_lower:
            return True
        if self.field:
            return bool(field_references(
                (app_label, self.model_name_lower), self.field, (app_label, name_lower)
            ))
        return False
```
### 50 - django/db/migrations/autodetector.py:

Start line: 462, End line: 514

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
                    rem_model_state = self.from_state.models[rem_app_label, rem_model_name]
                    rem_model_fields_def = self.only_relation_agnostic_fields(rem_model_state.fields)
                    if model_fields_def == rem_model_fields_def:
                        if self.questioner.ask_rename_model(rem_model_state, model_state):
                            dependencies = []
                            fields = list(model_state.fields.values()) + [
                                field.remote_field
                                for relations in self.to_state.relations[app_label, model_name].values()
                                for field in relations.values()
                            ]
                            for field in fields:
                                if field.is_relation:
                                    dependencies.extend(
                                        self._get_dependencies_for_foreign_key(
                                            app_label, model_name, field, self.to_state,
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
                            renamed_models_rel_key = '%s.%s' % (
                                rem_model_state.app_label,
                                rem_model_state.name_lower,
                            )
                            self.renamed_models_rel[renamed_models_rel_key] = '%s.%s' % (
                                model_state.app_label,
                                model_state.name_lower,
                            )
                            self.old_model_keys.remove((rem_app_label, rem_model_name))
                            self.old_model_keys.add((app_label, model_name))
                            break
```
### 62 - django/db/migrations/autodetector.py:

Start line: 900, End line: 919

```python
class MigrationAutodetector:

    def generate_removed_fields(self):
        """Make RemoveField operations."""
        for app_label, model_name, field_name in sorted(self.old_field_keys - self.new_field_keys):
            self._generate_removed_field(app_label, model_name, field_name)

    def _generate_removed_field(self, app_label, model_name, field_name):
        self.add_operation(
            app_label,
            operations.RemoveField(
                model_name=model_name,
                name=field_name,
            ),
            # We might need to depend on the removal of an
            # order_with_respect_to or index/unique_together operation;
            # this is safely ignored if there isn't one
            dependencies=[
                (app_label, model_name, field_name, "order_wrt_unset"),
                (app_label, model_name, field_name, "foo_together_change"),
            ],
        )
```
### 63 - django/db/migrations/operations/fields.py:

Start line: 38, End line: 60

```python
class FieldOperation(Operation):

    def references_field(self, model_name, name, app_label):
        model_name_lower = model_name.lower()
        # Check if this operation locally references the field.
        if model_name_lower == self.model_name_lower:
            if name == self.name:
                return True
            elif self.field and hasattr(self.field, 'from_fields') and name in self.field.from_fields:
                return True
        # Check if this operation remotely references the field.
        if self.field is None:
            return False
        return bool(field_references(
            (app_label, self.model_name_lower),
            self.field,
            (app_label, model_name_lower),
            name,
        ))

    def reduce(self, operation, app_label):
        return (
            super().reduce(operation, app_label) or
            not operation.references_field(self.model_name, self.name, app_label)
        )
```
### 71 - django/db/migrations/operations/fields.py:

Start line: 142, End line: 181

```python
class RemoveField(FieldOperation):
    """Remove a field from a model."""

    def deconstruct(self):
        kwargs = {
            'model_name': self.model_name,
            'name': self.name,
        }
        return (
            self.__class__.__name__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.remove_field(app_label, self.model_name_lower, self.name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        from_model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, from_model):
            schema_editor.remove_field(from_model, from_model._meta.get_field(self.name))

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            schema_editor.add_field(from_model, to_model._meta.get_field(self.name))

    def describe(self):
        return "Remove field %s from %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return 'remove_%s_%s' % (self.model_name_lower, self.name_lower)

    def reduce(self, operation, app_label):
        from .models import DeleteModel
        if isinstance(operation, DeleteModel) and operation.name_lower == self.model_name_lower:
            return [operation]
        return super().reduce(operation, app_label)
```
### 74 - django/db/migrations/operations/fields.py:

Start line: 119, End line: 139

```python
class AddField(FieldOperation):

    def reduce(self, operation, app_label):
        if isinstance(operation, FieldOperation) and self.is_same_field_operation(operation):
            if isinstance(operation, AlterField):
                return [
                    AddField(
                        model_name=self.model_name,
                        name=operation.name,
                        field=operation.field,
                    ),
                ]
            elif isinstance(operation, RemoveField):
                return []
            elif isinstance(operation, RenameField):
                return [
                    AddField(
                        model_name=self.model_name,
                        name=operation.new_name,
                        field=self.field,
                    ),
                ]
        return super().reduce(operation, app_label)
```
### 88 - django/db/migrations/autodetector.py:

Start line: 1114, End line: 1151

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
                        field = new_model_state.get_field(field_name)
                        if field.remote_field and field.remote_field.model:
                            dependencies.extend(self._get_dependencies_for_foreign_key(
                                app_label, model_name, field, self.to_state,
                            ))

                self.add_operation(
                    app_label,
                    operation(
                        name=model_name,
                        **{option_name: new_value}
                    ),
                    dependencies=dependencies,
                )
```
### 108 - django/db/migrations/autodetector.py:

Start line: 804, End line: 854

```python
class MigrationAutodetector:

    def generate_deleted_proxies(self):
        """Make DeleteModel options for proxy models."""
        deleted = self.old_proxy_keys - self.new_proxy_keys
        for app_label, model_name in sorted(deleted):
            model_state = self.from_state.models[app_label, model_name]
            assert model_state.options.get("proxy")
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
            )

    def generate_renamed_fields(self):
        """Work out renamed fields."""
        self.renamed_fields = {}
        for app_label, model_name, field_name in sorted(self.new_field_keys - self.old_field_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, old_model_name]
            field = new_model_state.get_field(field_name)
            # Scan to see if this is actually a rename!
            field_dec = self.deep_deconstruct(field)
            for rem_app_label, rem_model_name, rem_field_name in sorted(self.old_field_keys - self.new_field_keys):
                if rem_app_label == app_label and rem_model_name == model_name:
                    old_field = old_model_state.get_field(rem_field_name)
                    old_field_dec = self.deep_deconstruct(old_field)
                    if field.remote_field and field.remote_field.model and 'to' in old_field_dec[2]:
                        old_rel_to = old_field_dec[2]['to']
                        if old_rel_to in self.renamed_models_rel:
                            old_field_dec[2]['to'] = self.renamed_models_rel[old_rel_to]
                    old_field.set_attributes_from_name(rem_field_name)
                    old_db_column = old_field.get_attname_column()[1]
                    if (old_field_dec == field_dec or (
                            # Was the field renamed and db_column equal to the
                            # old field's column added?
                            old_field_dec[0:2] == field_dec[0:2] and
                            dict(old_field_dec[2], db_column=old_db_column) == field_dec[2])):
                        if self.questioner.ask_rename(model_name, rem_field_name, field_name, field):
                            self.add_operation(
                                app_label,
                                operations.RenameField(
                                    model_name=model_name,
                                    old_name=rem_field_name,
                                    new_name=field_name,
                                )
                            )
                            self.old_field_keys.remove((rem_app_label, rem_model_name, rem_field_name))
                            self.old_field_keys.add((app_label, model_name, field_name))
                            self.renamed_fields[app_label, model_name, field_name] = rem_field_name
                            break
```
### 121 - django/db/migrations/autodetector.py:

Start line: 89, End line: 101

```python
class MigrationAutodetector:

    def only_relation_agnostic_fields(self, fields):
        """
        Return a definition of the fields that ignores field names and
        what related fields actually relate to. Used for detecting renames (as
        the related fields change during renames).
        """
        fields_def = []
        for name, field in sorted(fields.items()):
            deconstruction = self.deep_deconstruct(field)
            if field.remote_field and field.remote_field.model:
                del deconstruction[2]['to']
            fields_def.append(deconstruction)
        return fields_def
```
### 126 - django/db/migrations/autodetector.py:

Start line: 856, End line: 898

```python
class MigrationAutodetector:

    def generate_added_fields(self):
        """Make AddField operations."""
        for app_label, model_name, field_name in sorted(self.new_field_keys - self.old_field_keys):
            self._generate_added_field(app_label, model_name, field_name)

    def _generate_added_field(self, app_label, model_name, field_name):
        field = self.to_state.models[app_label, model_name].get_field(field_name)
        # Fields that are foreignkeys/m2ms depend on stuff
        dependencies = []
        if field.remote_field and field.remote_field.model:
            dependencies.extend(self._get_dependencies_for_foreign_key(
                app_label, model_name, field, self.to_state,
            ))
        # You can't just add NOT NULL fields with no default or fields
        # which don't allow empty strings as default.
        time_fields = (models.DateField, models.DateTimeField, models.TimeField)
        preserve_default = (
            field.null or field.has_default() or field.many_to_many or
            (field.blank and field.empty_strings_allowed) or
            (isinstance(field, time_fields) and field.auto_now)
        )
        if not preserve_default:
            field = field.clone()
            if isinstance(field, time_fields) and field.auto_now_add:
                field.default = self.questioner.ask_auto_now_add_addition(field_name, model_name)
            else:
                field.default = self.questioner.ask_not_null_addition(field_name, model_name)
        if (
            field.unique and
            field.default is not models.NOT_PROVIDED and
            callable(field.default)
        ):
            self.questioner.ask_unique_callable_default_addition(field_name, model_name)
        self.add_operation(
            app_label,
            operations.AddField(
                model_name=model_name,
                name=field_name,
                field=field,
                preserve_default=preserve_default,
            ),
            dependencies=dependencies,
        )
```
### 133 - django/db/migrations/operations/fields.py:

Start line: 63, End line: 91

```python
class AddField(FieldOperation):
    """Add a field to a model."""

    def __init__(self, model_name, name, field, preserve_default=True):
        self.preserve_default = preserve_default
        super().__init__(model_name, name, field)

    def deconstruct(self):
        kwargs = {
            'model_name': self.model_name,
            'name': self.name,
            'field': self.field,
        }
        if self.preserve_default is not True:
            kwargs['preserve_default'] = self.preserve_default
        return (
            self.__class__.__name__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.add_field(
            app_label,
            self.model_name_lower,
            self.name,
            self.field,
            self.preserve_default,
        )
```
### 137 - django/db/migrations/autodetector.py:

Start line: 1153, End line: 1174

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
### 165 - django/db/migrations/autodetector.py:

Start line: 1212, End line: 1237

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
