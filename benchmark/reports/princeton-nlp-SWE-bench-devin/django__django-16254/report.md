# django__django-16254

| **django/django** | `e580b891cb5ae31eb0571c88428afb9bf69e47f2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 55302 |
| **Any found context length** | 55302 |
| **Avg pos** | 139.0 |
| **Min pos** | 139 |
| **Max pos** | 139 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -379,7 +379,10 @@ def delete_model(self, model, handle_autom2m=True):
 
     def add_field(self, model, field):
         """Create a field on a model."""
-        if (
+        # Special-case implicit M2M tables.
+        if field.many_to_many and field.remote_field.through._meta.auto_created:
+            self.create_model(field.remote_field.through)
+        elif (
             # Primary keys and unique fields are not supported in ALTER TABLE
             # ADD COLUMN.
             field.primary_key

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/sqlite3/schema.py | 382 | 382 | 139 | 1 | 55302


## Problem Statement

```
Adding ManyToManyField on SQLite rebuilds table.
Description
	
Hey there,
While updating the ​django-migration-linter for Django 4.1 (yeah, a bit late to the party, but I was busy eh :P), I bumped into what seems to be a regression.
On SQLite, when adding a ManyToManyField to a table, it seems to rebuild the table - whereas it did not in Django 4.0.
From my understanding, rebuilding the table is not necessary in this case. Or perhaps I'm missing something.
Steps to reproduce:
1/ Before models:
class A(models.Model):
	pass
class B(models.Model):
	pass
(with it's boring migration)
2/ After models:
class A(models.Model):
	pass
class B(models.Model):
	many_to_many = models.ManyToManyField(A)
Which, expectedly, generates the migration:
from django.db import migrations, models
class Migration(migrations.Migration):
	dependencies = [("app_add_manytomany_field", "0001_initial")]
	operations = [
		migrations.AddField(
			model_name="b",
			name="many_to_many",
			field=models.ManyToManyField(to="app_add_manytomany_field.A"),
		)
	]
All good up until here.
Now the "regression", in Django 4.0, a sqlmigrate generates:
BEGIN;
--
-- Add field many_to_many to b
--
CREATE TABLE "app_add_manytomany_field_b_many_to_many" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "b_id" integer NOT NULL REFERENCES "app_add_manytomany_field_b" ("id") DEFERRABLE INITIALLY DEFERRED, "a_id" integer NOT NULL REFERENCES "app_add_manytomany_field_a" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE UNIQUE INDEX "app_add_manytomany_field_b_many_to_many_b_id_a_id_3e15251d_uniq" ON "app_add_manytomany_field_b_many_to_many" ("b_id", "a_id");
CREATE INDEX "app_add_manytomany_field_b_many_to_many_b_id_953b185b" ON "app_add_manytomany_field_b_many_to_many" ("b_id");
CREATE INDEX "app_add_manytomany_field_b_many_to_many_a_id_4b44832a" ON "app_add_manytomany_field_b_many_to_many" ("a_id");
COMMIT;
whereas in Django 4.1:
BEGIN;
--
-- Add field many_to_many to b
--
CREATE TABLE "new__app_add_manytomany_field_b" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "x" integer NOT NULL);
CREATE TABLE "app_add_manytomany_field_b_many_to_many" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "b_id" integer NOT NULL REFERENCES "app_add_manytomany_field_b" ("id") DEFERRABLE INITIALLY DEFERRED, "a_id" integer NOT NULL REFERENCES "app_add_manytomany_field_a" ("id") DEFERRABLE INITIALLY DEFERRED);
INSERT INTO "new__app_add_manytomany_field_b" ("id", "x") SELECT "id", "x" FROM "app_add_manytomany_field_b";
DROP TABLE "app_add_manytomany_field_b";
ALTER TABLE "new__app_add_manytomany_field_b" RENAME TO "app_add_manytomany_field_b";
CREATE UNIQUE INDEX "app_add_manytomany_field_b_many_to_many_b_id_a_id_3e15251d_uniq" ON "app_add_manytomany_field_b_many_to_many" ("b_id", "a_id");
CREATE INDEX "app_add_manytomany_field_b_many_to_many_b_id_953b185b" ON "app_add_manytomany_field_b_many_to_many" ("b_id");
CREATE INDEX "app_add_manytomany_field_b_many_to_many_a_id_4b44832a" ON "app_add_manytomany_field_b_many_to_many" ("a_id");
COMMIT;
I could bisect it down to this commit 2f73e5406d54cb8945e187eff302a3a3373350be (from #32502 and this ​PR).
In the diff we see that the # Special-case implicit M2M tables comment and its code were removed.
That's potentially a lead for a fix here I guess :)
(On a side note, this commit introduced another regression #33408. But that's not related to the issue at hand)
Thank you!

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/backends/sqlite3/schema.py** | 485 | 535| 395 | 395 | 4583 | 
| 2 | **1 django/db/backends/sqlite3/schema.py** | 123 | 174| 527 | 922 | 4583 | 
| 3 | 2 django/db/models/fields/related.py | 1882 | 1929| 505 | 1427 | 19190 | 
| 4 | 2 django/db/models/fields/related.py | 1450 | 1579| 984 | 2411 | 19190 | 
| 5 | 2 django/db/models/fields/related.py | 1931 | 1958| 305 | 2716 | 19190 | 
| 6 | **2 django/db/backends/sqlite3/schema.py** | 100 | 121| 195 | 2911 | 19190 | 
| 7 | 3 django/db/backends/base/schema.py | 1249 | 1279| 318 | 3229 | 33054 | 
| 8 | 3 django/db/models/fields/related.py | 1681 | 1731| 431 | 3660 | 33054 | 
| 9 | **3 django/db/backends/sqlite3/schema.py** | 265 | 360| 807 | 4467 | 33054 | 
| 10 | 3 django/db/backends/base/schema.py | 39 | 72| 214 | 4681 | 33054 | 
| 11 | **3 django/db/backends/sqlite3/schema.py** | 176 | 264| 822 | 5503 | 33054 | 
| 12 | 4 django/db/migrations/autodetector.py | 1101 | 1218| 982 | 6485 | 46515 | 
| 13 | **4 django/db/backends/sqlite3/schema.py** | 426 | 483| 488 | 6973 | 46515 | 
| 14 | 4 django/db/backends/base/schema.py | 890 | 974| 778 | 7751 | 46515 | 
| 15 | 5 django/db/migrations/questioner.py | 57 | 87| 255 | 8006 | 49211 | 
| 16 | 5 django/db/backends/base/schema.py | 1063 | 1143| 779 | 8785 | 49211 | 
| 17 | 5 django/db/models/fields/related.py | 1733 | 1771| 413 | 9198 | 49211 | 
| 18 | 6 django/db/backends/postgresql/schema.py | 140 | 246| 866 | 10064 | 51952 | 
| 19 | 6 django/db/migrations/autodetector.py | 1504 | 1532| 235 | 10299 | 51952 | 
| 20 | 6 django/db/models/fields/related.py | 302 | 339| 296 | 10595 | 51952 | 
| 21 | 6 django/db/migrations/questioner.py | 291 | 342| 367 | 10962 | 51952 | 
| 22 | 6 django/db/migrations/autodetector.py | 1027 | 1076| 384 | 11346 | 51952 | 
| 23 | 6 django/db/backends/base/schema.py | 1457 | 1476| 175 | 11521 | 51952 | 
| 24 | 7 django/db/models/sql/compiler.py | 1964 | 2025| 588 | 12109 | 68523 | 
| 25 | 7 django/db/backends/base/schema.py | 975 | 1062| 822 | 12931 | 68523 | 
| 26 | 8 django/db/migrations/operations/models.py | 370 | 424| 501 | 13432 | 76236 | 
| 27 | 9 django/db/backends/mysql/schema.py | 1 | 43| 471 | 13903 | 78190 | 
| 28 | 9 django/db/migrations/autodetector.py | 600 | 776| 1231 | 15134 | 78190 | 
| 29 | 10 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 15361 | 78417 | 
| 30 | 11 django/db/backends/sqlite3/features.py | 68 | 115| 383 | 15744 | 79703 | 
| 31 | 11 django/db/migrations/autodetector.py | 1534 | 1553| 187 | 15931 | 79703 | 
| 32 | 12 django/db/backends/sqlite3/operations.py | 216 | 241| 218 | 16149 | 83190 | 
| 33 | 12 django/db/backends/base/schema.py | 799 | 889| 773 | 16922 | 83190 | 
| 34 | 12 django/db/backends/postgresql/schema.py | 267 | 301| 277 | 17199 | 83190 | 
| 35 | 12 django/db/backends/sqlite3/features.py | 117 | 151| 245 | 17444 | 83190 | 
| 36 | 13 django/db/backends/oracle/schema.py | 75 | 103| 344 | 17788 | 85532 | 
| 37 | 13 django/db/backends/sqlite3/features.py | 1 | 66| 671 | 18459 | 85532 | 
| 38 | 13 django/db/migrations/autodetector.py | 1595 | 1627| 258 | 18717 | 85532 | 
| 39 | 13 django/db/backends/mysql/schema.py | 105 | 119| 144 | 18861 | 85532 | 
| 40 | **13 django/db/backends/sqlite3/schema.py** | 25 | 41| 168 | 19029 | 85532 | 
| 41 | 14 django/db/models/fields/related_descriptors.py | 786 | 861| 592 | 19621 | 96610 | 
| 42 | 15 django/db/backends/mysql/operations.py | 436 | 465| 274 | 19895 | 100788 | 
| 43 | 15 django/db/migrations/autodetector.py | 1369 | 1395| 144 | 20039 | 100788 | 
| 44 | 16 django/db/migrations/state.py | 437 | 458| 204 | 20243 | 108956 | 
| 45 | 17 django/db/migrations/operations/fields.py | 115 | 127| 131 | 20374 | 111468 | 
| 46 | 17 django/db/migrations/autodetector.py | 1432 | 1477| 318 | 20692 | 111468 | 
| 47 | 18 django/db/models/options.py | 1 | 57| 347 | 21039 | 119151 | 
| 48 | 19 django/db/backends/mysql/compiler.py | 51 | 81| 240 | 21279 | 119787 | 
| 49 | 19 django/db/backends/sqlite3/operations.py | 417 | 437| 148 | 21427 | 119787 | 
| 50 | 20 django/db/backends/mysql/features.py | 84 | 158| 597 | 22024 | 122211 | 
| 51 | 21 django/db/migrations/loader.py | 222 | 305| 791 | 22815 | 125364 | 
| 52 | 21 django/db/models/fields/related.py | 1581 | 1679| 655 | 23470 | 125364 | 
| 53 | 22 django/contrib/redirects/migrations/0001_initial.py | 1 | 66| 309 | 23779 | 125673 | 
| 54 | 22 django/db/migrations/autodetector.py | 1220 | 1307| 719 | 24498 | 125673 | 
| 55 | 22 django/db/backends/base/schema.py | 1 | 36| 223 | 24721 | 125673 | 
| 56 | 22 django/db/migrations/autodetector.py | 1479 | 1502| 161 | 24882 | 125673 | 
| 57 | 23 django/contrib/admin/migrations/0001_initial.py | 1 | 77| 363 | 25245 | 126036 | 
| 58 | 23 django/db/models/fields/related.py | 341 | 378| 294 | 25539 | 126036 | 
| 59 | 23 django/db/migrations/operations/fields.py | 101 | 113| 130 | 25669 | 126036 | 
| 60 | 24 django/db/models/fields/related_lookups.py | 170 | 210| 250 | 25919 | 127636 | 
| 61 | 24 django/db/backends/base/schema.py | 75 | 152| 798 | 26717 | 127636 | 
| 62 | 24 django/db/migrations/autodetector.py | 1346 | 1367| 197 | 26914 | 127636 | 
| 63 | 24 django/db/backends/oracle/schema.py | 1 | 29| 289 | 27203 | 127636 | 
| 64 | 24 django/db/models/fields/related_descriptors.py | 1184 | 1218| 341 | 27544 | 127636 | 
| 65 | 24 django/db/models/fields/related.py | 871 | 898| 237 | 27781 | 127636 | 
| 66 | 24 django/db/migrations/autodetector.py | 811 | 906| 712 | 28493 | 127636 | 
| 67 | 24 django/db/backends/postgresql/schema.py | 119 | 138| 234 | 28727 | 127636 | 
| 68 | 24 django/db/models/fields/related.py | 154 | 185| 209 | 28936 | 127636 | 
| 69 | 25 django/contrib/flatpages/migrations/0001_initial.py | 1 | 70| 355 | 29291 | 127991 | 
| 70 | 26 django/contrib/gis/db/backends/spatialite/schema.py | 137 | 192| 404 | 29695 | 129385 | 
| 71 | 26 django/db/migrations/state.py | 240 | 263| 247 | 29942 | 129385 | 
| 72 | 26 django/db/migrations/autodetector.py | 1309 | 1344| 232 | 30174 | 129385 | 
| 73 | 26 django/db/backends/mysql/features.py | 160 | 267| 728 | 30902 | 129385 | 
| 74 | 26 django/db/models/fields/related_lookups.py | 110 | 148| 324 | 31226 | 129385 | 
| 75 | 26 django/db/models/fields/related.py | 1418 | 1448| 172 | 31398 | 129385 | 
| 76 | 27 django/db/models/base.py | 1 | 66| 357 | 31755 | 147916 | 
| 77 | 27 django/db/models/fields/related_descriptors.py | 1304 | 1373| 522 | 32277 | 147916 | 
| 78 | 27 django/db/migrations/questioner.py | 217 | 247| 252 | 32529 | 147916 | 
| 79 | 27 django/db/models/fields/related.py | 1018 | 1054| 261 | 32790 | 147916 | 
| 80 | 28 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 33185 | 148582 | 
| 81 | 28 django/db/migrations/operations/fields.py | 270 | 337| 521 | 33706 | 148582 | 
| 82 | **28 django/db/backends/sqlite3/schema.py** | 537 | 561| 162 | 33868 | 148582 | 
| 83 | 29 django/db/backends/oracle/operations.py | 407 | 453| 385 | 34253 | 154802 | 
| 84 | 29 django/db/models/base.py | 459 | 572| 953 | 35206 | 154802 | 
| 85 | 29 django/db/models/fields/related.py | 208 | 224| 142 | 35348 | 154802 | 
| 86 | 29 django/db/models/fields/related.py | 1960 | 1994| 266 | 35614 | 154802 | 
| 87 | 29 django/db/migrations/state.py | 291 | 345| 476 | 36090 | 154802 | 
| 88 | **29 django/db/backends/sqlite3/schema.py** | 362 | 378| 132 | 36222 | 154802 | 
| 89 | 29 django/db/models/fields/related_lookups.py | 1 | 40| 221 | 36443 | 154802 | 
| 90 | 29 django/db/models/fields/related.py | 901 | 990| 583 | 37026 | 154802 | 
| 91 | 29 django/db/models/fields/related_descriptors.py | 1096 | 1122| 220 | 37246 | 154802 | 
| 92 | 29 django/db/migrations/autodetector.py | 1078 | 1099| 188 | 37434 | 154802 | 
| 93 | 29 django/db/migrations/autodetector.py | 516 | 581| 482 | 37916 | 154802 | 
| 94 | 29 django/db/migrations/autodetector.py | 983 | 1025| 297 | 38213 | 154802 | 
| 95 | 29 django/db/migrations/autodetector.py | 908 | 981| 623 | 38836 | 154802 | 
| 96 | 29 django/db/models/base.py | 250 | 367| 874 | 39710 | 154802 | 
| 97 | 29 django/db/backends/mysql/schema.py | 121 | 136| 116 | 39826 | 154802 | 
| 98 | 29 django/db/backends/sqlite3/operations.py | 356 | 415| 577 | 40403 | 154802 | 
| 99 | 29 django/db/backends/base/schema.py | 1499 | 1521| 199 | 40602 | 154802 | 
| 100 | 29 django/db/models/fields/related.py | 226 | 301| 696 | 41298 | 154802 | 
| 101 | 29 django/db/backends/postgresql/schema.py | 303 | 328| 235 | 41533 | 154802 | 
| 102 | 30 django/core/management/commands/migrate.py | 96 | 189| 765 | 42298 | 158728 | 
| 103 | **30 django/db/backends/sqlite3/schema.py** | 398 | 424| 256 | 42554 | 158728 | 
| 104 | 30 django/db/models/base.py | 1064 | 1116| 516 | 43070 | 158728 | 
| 105 | 30 django/db/migrations/state.py | 265 | 289| 238 | 43308 | 158728 | 
| 106 | 30 django/db/models/base.py | 1521 | 1556| 273 | 43581 | 158728 | 
| 107 | 30 django/db/migrations/operations/fields.py | 227 | 237| 146 | 43727 | 158728 | 
| 108 | 30 django/core/management/commands/migrate.py | 270 | 368| 813 | 44540 | 158728 | 
| 109 | 30 django/db/backends/postgresql/schema.py | 1 | 81| 711 | 45251 | 158728 | 
| 110 | 30 django/db/migrations/autodetector.py | 403 | 419| 141 | 45392 | 158728 | 
| 111 | 30 django/db/backends/mysql/schema.py | 138 | 154| 144 | 45536 | 158728 | 
| 112 | 31 django/db/migrations/utils.py | 1 | 24| 134 | 45670 | 159599 | 
| 113 | 32 django/db/models/__init__.py | 1 | 116| 682 | 46352 | 160281 | 
| 114 | 32 django/db/migrations/operations/models.py | 644 | 668| 231 | 46583 | 160281 | 
| 115 | 32 django/db/migrations/autodetector.py | 280 | 378| 806 | 47389 | 160281 | 
| 116 | 32 django/db/backends/postgresql/schema.py | 248 | 265| 170 | 47559 | 160281 | 
| 117 | 32 django/db/backends/mysql/features.py | 1 | 62| 443 | 48002 | 160281 | 
| 118 | 33 django/db/backends/postgresql/operations.py | 338 | 357| 150 | 48152 | 163270 | 
| 119 | 33 django/db/migrations/questioner.py | 166 | 187| 188 | 48340 | 163270 | 
| 120 | 33 django/db/models/base.py | 2254 | 2445| 1302 | 49642 | 163270 | 
| 121 | 34 django/db/backends/sqlite3/introspection.py | 265 | 302| 294 | 49936 | 166595 | 
| 122 | 35 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 47| 225 | 50161 | 166820 | 
| 123 | 35 django/db/migrations/operations/fields.py | 239 | 267| 206 | 50367 | 166820 | 
| 124 | 35 django/db/models/fields/related.py | 187 | 206| 155 | 50522 | 166820 | 
| 125 | 35 django/db/models/fields/related.py | 1 | 40| 251 | 50773 | 166820 | 
| 126 | 35 django/db/backends/mysql/operations.py | 1 | 42| 354 | 51127 | 166820 | 
| 127 | 36 django/db/backends/oracle/creation.py | 159 | 201| 411 | 51538 | 170835 | 
| 128 | 36 django/db/models/base.py | 1807 | 1842| 242 | 51780 | 170835 | 
| 129 | 36 django/db/migrations/operations/models.py | 570 | 594| 213 | 51993 | 170835 | 
| 130 | 36 django/db/migrations/questioner.py | 249 | 267| 177 | 52170 | 170835 | 
| 131 | 36 django/db/backends/base/schema.py | 523 | 542| 196 | 52366 | 170835 | 
| 132 | 37 django/contrib/auth/migrations/0001_initial.py | 1 | 206| 1007 | 53373 | 171842 | 
| 133 | 37 django/db/backends/base/schema.py | 1478 | 1497| 176 | 53549 | 171842 | 
| 134 | 37 django/db/models/fields/related.py | 1103 | 1120| 133 | 53682 | 171842 | 
| 135 | 37 django/db/backends/base/schema.py | 599 | 627| 245 | 53927 | 171842 | 
| 136 | 37 django/db/models/fields/related_descriptors.py | 863 | 891| 229 | 54156 | 171842 | 
| 137 | 38 django/db/backends/oracle/features.py | 88 | 152| 604 | 54760 | 173178 | 
| 138 | 39 django/core/serializers/xml_serializer.py | 127 | 179| 404 | 55164 | 176812 | 
| **-> 139 <-** | **39 django/db/backends/sqlite3/schema.py** | 380 | 396| 138 | 55302 | 176812 | 
| 140 | 39 django/db/backends/oracle/schema.py | 172 | 181| 142 | 55444 | 176812 | 
| 141 | 39 django/db/migrations/operations/models.py | 670 | 686| 159 | 55603 | 176812 | 
| 142 | 39 django/db/migrations/operations/models.py | 1027 | 1065| 283 | 55886 | 176812 | 
| 143 | 39 django/db/backends/sqlite3/operations.py | 1 | 42| 314 | 56200 | 176812 | 
| 144 | 39 django/db/backends/postgresql/schema.py | 330 | 366| 210 | 56410 | 176812 | 
| 145 | 40 django/contrib/gis/db/backends/mysql/schema.py | 57 | 88| 247 | 56657 | 177469 | 
| 146 | 40 django/db/models/fields/related_descriptors.py | 404 | 425| 158 | 56815 | 177469 | 
| 147 | 41 django/db/models/fields/__init__.py | 443 | 468| 198 | 57013 | 196204 | 
| 148 | 42 django/core/management/commands/squashmigrations.py | 162 | 254| 766 | 57779 | 198244 | 


## Patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -379,7 +379,10 @@ def delete_model(self, model, handle_autom2m=True):
 
     def add_field(self, model, field):
         """Create a field on a model."""
-        if (
+        # Special-case implicit M2M tables.
+        if field.many_to_many and field.remote_field.through._meta.auto_created:
+            self.create_model(field.remote_field.through)
+        elif (
             # Primary keys and unique fields are not supported in ALTER TABLE
             # ADD COLUMN.
             field.primary_key

```

## Test Patch

```diff
diff --git a/tests/schema/tests.py b/tests/schema/tests.py
--- a/tests/schema/tests.py
+++ b/tests/schema/tests.py
@@ -2200,8 +2200,25 @@ class Meta:
         with self.assertRaises(DatabaseError):
             self.column_classes(new_field.remote_field.through)
         # Add the field
-        with connection.schema_editor() as editor:
+        with CaptureQueriesContext(
+            connection
+        ) as ctx, connection.schema_editor() as editor:
             editor.add_field(LocalAuthorWithM2M, new_field)
+        # Table is not rebuilt.
+        self.assertEqual(
+            len(
+                [
+                    query["sql"]
+                    for query in ctx.captured_queries
+                    if "CREATE TABLE" in query["sql"]
+                ]
+            ),
+            1,
+        )
+        self.assertIs(
+            any("DROP TABLE" in query["sql"] for query in ctx.captured_queries),
+            False,
+        )
         # Ensure there is now an m2m table there
         columns = self.column_classes(new_field.remote_field.through)
         self.assertEqual(

```


## Code snippets

### 1 - django/db/backends/sqlite3/schema.py:

Start line: 485, End line: 535

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
                alter_field=(
                    # The field that points to the target model is needed, so
                    # we can tell alter_field to change it - this is
                    # m2m_reverse_field_name() (as opposed to m2m_field_name(),
                    # which points to our model).
                    old_field.remote_field.through._meta.get_field(
                        old_field.m2m_reverse_field_name()
                    ),
                    new_field.remote_field.through._meta.get_field(
                        new_field.m2m_reverse_field_name()
                    ),
                ),
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
### 2 - django/db/backends/sqlite3/schema.py:

Start line: 123, End line: 174

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        if not self._field_should_be_altered(old_field, new_field):
            return
        old_field_name = old_field.name
        table_name = model._meta.db_table
        _, old_column_name = old_field.get_attname_column()
        if (
            new_field.name != old_field_name
            and not self.connection.features.supports_atomic_references_rename
            and self._is_referenced_by_fk_constraint(
                table_name, old_column_name, ignore_self=True
            )
        ):
            if self.connection.in_atomic_block:
                raise NotSupportedError(
                    (
                        "Renaming the %r.%r column while in a transaction is not "
                        "supported on SQLite < 3.26 because it would break referential "
                        "integrity. Try adding `atomic = False` to the Migration class."
                    )
                    % (model._meta.db_table, old_field_name)
                )
            with atomic(self.connection.alias):
                super().alter_field(model, old_field, new_field, strict=strict)
                # Follow SQLite's documented procedure for performing changes
                # that don't affect the on-disk content.
                # https://sqlite.org/lang_altertable.html#otheralter
                with self.connection.cursor() as cursor:
                    schema_version = cursor.execute("PRAGMA schema_version").fetchone()[
                        0
                    ]
                    cursor.execute("PRAGMA writable_schema = 1")
                    references_template = ' REFERENCES "%s" ("%%s") ' % table_name
                    new_column_name = new_field.get_attname_column()[1]
                    search = references_template % old_column_name
                    replacement = references_template % new_column_name
                    cursor.execute(
                        "UPDATE sqlite_master SET sql = replace(sql, %s, %s)",
                        (search, replacement),
                    )
                    cursor.execute("PRAGMA schema_version = %d" % (schema_version + 1))
                    cursor.execute("PRAGMA writable_schema = 0")
                    # The integrity check will raise an exception and rollback
                    # the transaction if the sqlite_master updates corrupt the
                    # database.
                    cursor.execute("PRAGMA integrity_check")
            # Perform a VACUUM to refresh the database representation from
            # the sqlite_master table.
            with self.connection.cursor() as cursor:
                cursor.execute("VACUUM")
        else:
            super().alter_field(model, old_field, new_field, strict=strict)
```
### 3 - django/db/models/fields/related.py:

Start line: 1882, End line: 1929

```python
class ManyToManyField(RelatedField):

    def contribute_to_class(self, cls, name, **kwargs):
        # To support multiple relations to self, it's useful to have a non-None
        # related name on symmetrical relations for internal reasons. The
        # concept doesn't make a lot of sense externally ("you want me to
        # specify *what* on my non-reversible relation?!"), so we set it up
        # automatically. The funky name reduces the chance of an accidental
        # clash.
        if self.remote_field.symmetrical and (
            self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT
            or self.remote_field.model == cls._meta.object_name
        ):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = "_%s_%s_%s_+" % (
                cls._meta.app_label,
                cls.__name__.lower(),
                name,
            )

        super().contribute_to_class(cls, name, **kwargs)

        # The intermediate m2m model is not auto created if:
        #  1) There is a manually specified intermediate, or
        #  2) The class owning the m2m field is abstract.
        #  3) The class owning the m2m field has been swapped out.
        if not cls._meta.abstract:
            if self.remote_field.through:

                def resolve_through_model(_, model, field):
                    field.remote_field.through = model

                lazy_related_operation(
                    resolve_through_model, cls, self.remote_field.through, field=self
                )
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(
                    self, cls
                )

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 4 - django/db/models/fields/related.py:

Start line: 1450, End line: 1579

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, "_meta"):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label,
                self.remote_field.through.__name__,
            )
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(
            include_auto_created=True
        ):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id="fields.E331",
                )
            )

        else:
            assert from_model is not None, (
                "ManyToManyField with intermediate "
                "tables cannot be checked if you don't pass the model "
                "where the field is attached to."
            )
            # Set some useful local variables
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            # Count foreign keys in intermediate model
            if self_referential:
                seen_self = sum(
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument."
                            % (self, from_model_name),
                            hint=(
                                "Use through_fields to specify which two foreign keys "
                                "Django should use."
                            ),
                            obj=self.remote_field.through,
                            id="fields.E333",
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            (
                                "The model is used as an intermediate model by "
                                "'%s', but it has more than one foreign key "
                                "from '%s', which is ambiguous. You must specify "
                                "which foreign key Django should use via the "
                                "through_fields keyword argument."
                            )
                            % (self, from_model_name),
                            hint=(
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E334",
                        )
                    )

                if seen_to > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than one foreign key "
                            "to '%s', which is ambiguous. You must specify "
                            "which foreign key Django should use via the "
                            "through_fields keyword argument." % (self, to_model_name),
                            hint=(
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E335",
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'."
                            % (self, from_model_name, to_model_name),
                            obj=self.remote_field.through,
                            id="fields.E336",
                        )
                    )
        # ... other code
```
### 5 - django/db/models/fields/related.py:

Start line: 1931, End line: 1958

```python
class ManyToManyField(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal M2Ms (i.e., those with a related name ending with '+')
        # and swapped models don't get a related descriptor.
        if (
            not self.remote_field.is_hidden()
            and not related.related_model._meta.swapped
        ):
            setattr(
                cls,
                related.get_accessor_name(),
                ManyToManyDescriptor(self.remote_field, reverse=True),
            )

        # Set up the accessors for the column names on the m2m table.
        self.m2m_column_name = partial(self._get_m2m_attr, related, "column")
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, "column")

        self.m2m_field_name = partial(self._get_m2m_attr, related, "name")
        self.m2m_reverse_field_name = partial(
            self._get_m2m_reverse_attr, related, "name"
        )

        get_m2m_rel = partial(self._get_m2m_attr, related, "remote_field")
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(
            self._get_m2m_reverse_attr, related, "remote_field"
        )
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name
```
### 6 - django/db/backends/sqlite3/schema.py:

Start line: 100, End line: 121

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_db_table(
        self, model, old_db_table, new_db_table, disable_constraints=True
    ):
        if (
            not self.connection.features.supports_atomic_references_rename
            and disable_constraints
            and self._is_referenced_by_fk_constraint(old_db_table)
        ):
            if self.connection.in_atomic_block:
                raise NotSupportedError(
                    (
                        "Renaming the %r table while in a transaction is not "
                        "supported on SQLite < 3.26 because it would break referential "
                        "integrity. Try adding `atomic = False` to the Migration class."
                    )
                    % old_db_table
                )
            self.connection.enable_constraint_checking()
            super().alter_db_table(model, old_db_table, new_db_table)
            self.connection.disable_constraint_checking()
        else:
            super().alter_db_table(model, old_db_table, new_db_table)
```
### 7 - django/db/backends/base/schema.py:

Start line: 1249, End line: 1279

```python
class BaseDatabaseSchemaEditor:

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        # Rename the through table
        if (
            old_field.remote_field.through._meta.db_table
            != new_field.remote_field.through._meta.db_table
        ):
            self.alter_db_table(
                old_field.remote_field.through,
                old_field.remote_field.through._meta.db_table,
                new_field.remote_field.through._meta.db_table,
            )
        # Repoint the FK to the other side
        self.alter_field(
            new_field.remote_field.through,
            # The field that points to the target model is needed, so we can
            # tell alter_field to change it - this is m2m_reverse_field_name()
            # (as opposed to m2m_field_name(), which points to our model).
            old_field.remote_field.through._meta.get_field(
                old_field.m2m_reverse_field_name()
            ),
            new_field.remote_field.through._meta.get_field(
                new_field.m2m_reverse_field_name()
            ),
        )
        self.alter_field(
            new_field.remote_field.through,
            # for self-referential models we need to alter field from the other end too
            old_field.remote_field.through._meta.get_field(old_field.m2m_field_name()),
            new_field.remote_field.through._meta.get_field(new_field.m2m_field_name()),
        )
```
### 8 - django/db/models/fields/related.py:

Start line: 1681, End line: 1731

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if (
            isinstance(self.remote_field.through, str)
            or not self.remote_field.through._meta.managed
        ):
            return []
        registered_tables = {
            model._meta.db_table: model
            for model in self.opts.apps.get_models(include_auto_created=True)
            if model != self.remote_field.through and model._meta.managed
        }
        m2m_db_table = self.m2m_db_table()
        model = registered_tables.get(m2m_db_table)
        # The second condition allows multiple m2m relations on a model if
        # some point to a through model that proxies another through model.
        if (
            model
            and model._meta.concrete_model
            != self.remote_field.through._meta.concrete_model
        ):
            if model._meta.auto_created:

                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name

                opts = model._meta.auto_created._meta
                clashing_obj = "%s.%s" % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, "fields.W344"
                error_hint = (
                    "You have configured settings.DATABASE_ROUTERS. Verify "
                    "that the table of %r is correctly routed to a separate "
                    "database." % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, "fields.E340"
                error_hint = None
            return [
                error_class(
                    "The field's intermediary table '%s' clashes with the "
                    "table name of '%s'." % (m2m_db_table, clashing_obj),
                    obj=self,
                    hint=error_hint,
                    id=error_id,
                )
            ]
        return []
```
### 9 - django/db/backends/sqlite3/schema.py:

Start line: 265, End line: 360

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_field=None
    ):
        # ... other code
        apps = Apps()

        # Work out the new value of unique_together, taking renames into
        # account
        unique_together = [
            [rename_mapping.get(n, n) for n in unique]
            for unique in model._meta.unique_together
        ]

        # Work out the new value for index_together, taking renames into
        # account
        index_together = [
            [rename_mapping.get(n, n) for n in index]
            for index in model._meta.index_together
        ]

        indexes = model._meta.indexes
        if delete_field:
            indexes = [
                index for index in indexes if delete_field.name not in index.fields
            ]

        constraints = list(model._meta.constraints)

        # Provide isolated instances of the fields to the new model body so
        # that the existing model's internals aren't interfered with when
        # the dummy model is constructed.
        body_copy = copy.deepcopy(body)

        # Construct a new model with the new fields to allow self referential
        # primary key to resolve to. This model won't ever be materialized as a
        # table and solely exists for foreign key reference resolution purposes.
        # This wouldn't be required if the schema editor was operating on model
        # states instead of rendered models.
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": model._meta.db_table,
            "unique_together": unique_together,
            "index_together": index_together,
            "indexes": indexes,
            "constraints": constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        type(model._meta.object_name, model.__bases__, body_copy)

        # Construct a model with a renamed table name.
        body_copy = copy.deepcopy(body)
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": "new__%s" % strip_quotes(model._meta.db_table),
            "unique_together": unique_together,
            "index_together": index_together,
            "indexes": indexes,
            "constraints": constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        new_model = type("New%s" % model._meta.object_name, model.__bases__, body_copy)

        # Create a new table with the updated schema.
        self.create_model(new_model)

        # Copy data from the old table into the new table
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_model._meta.db_table),
                ", ".join(self.quote_name(x) for x in mapping),
                ", ".join(mapping.values()),
                self.quote_name(model._meta.db_table),
            )
        )

        # Delete the old table to make way for the new
        self.delete_model(model, handle_autom2m=False)

        # Rename the new table to take way for the old
        self.alter_db_table(
            new_model,
            new_model._meta.db_table,
            model._meta.db_table,
            disable_constraints=False,
        )

        # Run deferred SQL on correct table
        for sql in self.deferred_sql:
            self.execute(sql)
        self.deferred_sql = []
        # Fix any PK-removed field
        if restore_pk_field:
            restore_pk_field.primary_key = True
```
### 10 - django/db/backends/base/schema.py:

Start line: 39, End line: 72

```python
def _all_related_fields(model):
    # Related fields must be returned in a deterministic order.
    return sorted(
        model._meta._get_fields(
            forward=False,
            reverse=True,
            include_hidden=True,
            include_parents=False,
        ),
        key=operator.attrgetter("name"),
    )


def _related_non_m2m_objects(old_field, new_field):
    # Filter out m2m objects from reverse relations.
    # Return (old_relation, new_relation) tuples.
    related_fields = zip(
        (
            obj
            for obj in _all_related_fields(old_field.model)
            if _is_relevant_relation(obj, old_field)
        ),
        (
            obj
            for obj in _all_related_fields(new_field.model)
            if _is_relevant_relation(obj, new_field)
        ),
    )
    for old_rel, new_rel in related_fields:
        yield old_rel, new_rel
        yield from _related_non_m2m_objects(
            old_rel.remote_field,
            new_rel.remote_field,
        )
```
### 11 - django/db/backends/sqlite3/schema.py:

Start line: 176, End line: 264

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_field=None
    ):
        """
        Shortcut to transform a model from old_model into new_model

        This follows the correct procedure to perform non-rename or column
        addition operations based on SQLite's documentation

        https://www.sqlite.org/lang_altertable.html#caution

        The essential steps are:
          1. Create a table with the updated definition called "new__app_model"
          2. Copy the data from the existing "app_model" table to the new table
          3. Drop the "app_model" table
          4. Rename the "new__app_model" table to "app_model"
          5. Restore any index of the previous "app_model" table.
        """
        # Self-referential fields must be recreated rather than copied from
        # the old model to ensure their remote_field.field_name doesn't refer
        # to an altered field.
        def is_self_referential(f):
            return f.is_relation and f.remote_field.model is model

        # Work out the new fields dict / mapping
        body = {
            f.name: f.clone() if is_self_referential(f) else f
            for f in model._meta.local_concrete_fields
        }
        # Since mapping might mix column names and default values,
        # its values must be already quoted.
        mapping = {
            f.column: self.quote_name(f.column)
            for f in model._meta.local_concrete_fields
        }
        # This maps field names (not columns) for things like unique_together
        rename_mapping = {}
        # If any of the new or altered fields is introducing a new PK,
        # remove the old one
        restore_pk_field = None
        if getattr(create_field, "primary_key", False) or (
            alter_field and getattr(alter_field[1], "primary_key", False)
        ):
            for name, field in list(body.items()):
                if field.primary_key and not (
                    # Do not remove the old primary key when an altered field
                    # that introduces a primary key is the same field.
                    alter_field
                    and name == alter_field[1].name
                ):
                    field.primary_key = False
                    restore_pk_field = field
                    if field.auto_created:
                        del body[name]
                        del mapping[field.column]
        # Add in any created fields
        if create_field:
            body[create_field.name] = create_field
            # Choose a default and insert it into the copy map
            if not create_field.many_to_many and create_field.concrete:
                mapping[create_field.column] = self.prepare_default(
                    self.effective_default(create_field),
                )
        # Add in any altered fields
        if alter_field:
            old_field, new_field = alter_field
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            if old_field.null and not new_field.null:
                case_sql = "coalesce(%(col)s, %(default)s)" % {
                    "col": self.quote_name(old_field.column),
                    "default": self.prepare_default(self.effective_default(new_field)),
                }
                mapping[new_field.column] = case_sql
            else:
                mapping[new_field.column] = self.quote_name(old_field.column)
            rename_mapping[old_field.name] = new_field.name
        # Remove any deleted fields
        if delete_field:
            del body[delete_field.name]
            del mapping[delete_field.column]
            # Remove any implicit M2M tables
            if (
                delete_field.many_to_many
                and delete_field.remote_field.through._meta.auto_created
            ):
                return self.delete_model(delete_field.remote_field.through)
        # Work inside a new app registry
        # ... other code
```
### 13 - django/db/backends/sqlite3/schema.py:

Start line: 426, End line: 483

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        """Perform a "physical" (non-ManyToMany) field update."""
        # Use "ALTER TABLE ... RENAME COLUMN" if only the column name
        # changed and there aren't any constraints.
        if (
            self.connection.features.can_alter_table_rename_column
            and old_field.column != new_field.column
            and self.column_sql(model, old_field) == self.column_sql(model, new_field)
            and not (
                old_field.remote_field
                and old_field.db_constraint
                or new_field.remote_field
                and new_field.db_constraint
            )
        ):
            return self.execute(
                self._rename_field_sql(
                    model._meta.db_table, old_field, new_field, new_type
                )
            )
        # Alter by remaking table
        self._remake_table(model, alter_field=(old_field, new_field))
        # Rebuild tables with FKs pointing to this field.
        old_collation = old_db_params.get("collation")
        new_collation = new_db_params.get("collation")
        if new_field.unique and (
            old_type != new_type or old_collation != new_collation
        ):
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
### 40 - django/db/backends/sqlite3/schema.py:

Start line: 25, End line: 41

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the schema edition.
        if not self.connection.disable_constraint_checking():
            raise NotSupportedError(
                "SQLite schema editor cannot be used while foreign key "
                "constraint checks are enabled. Make sure to disable them "
                "before entering a transaction.atomic() context because "
                "SQLite does not support disabling them in the middle of "
                "a multi-statement transaction."
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.check_constraints()
        super().__exit__(exc_type, exc_value, traceback)
        self.connection.enable_constraint_checking()
```
### 82 - django/db/backends/sqlite3/schema.py:

Start line: 537, End line: 561

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def add_constraint(self, model, constraint):
        if isinstance(constraint, UniqueConstraint) and (
            constraint.condition
            or constraint.contains_expressions
            or constraint.include
            or constraint.deferrable
        ):
            super().add_constraint(model, constraint)
        else:
            self._remake_table(model)

    def remove_constraint(self, model, constraint):
        if isinstance(constraint, UniqueConstraint) and (
            constraint.condition
            or constraint.contains_expressions
            or constraint.include
            or constraint.deferrable
        ):
            super().remove_constraint(model, constraint)
        else:
            self._remake_table(model)

    def _collate_sql(self, collation):
        return "COLLATE " + collation
```
### 88 - django/db/backends/sqlite3/schema.py:

Start line: 362, End line: 378

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def delete_model(self, model, handle_autom2m=True):
        if handle_autom2m:
            super().delete_model(model)
        else:
            # Delete the table (and only that)
            self.execute(
                self.sql_delete_table
                % {
                    "table": self.quote_name(model._meta.db_table),
                }
            )
            # Remove all deferred statements referencing the deleted table.
            for sql in list(self.deferred_sql):
                if isinstance(sql, Statement) and sql.references_table(
                    model._meta.db_table
                ):
                    self.deferred_sql.remove(sql)
```
### 103 - django/db/backends/sqlite3/schema.py:

Start line: 398, End line: 424

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def remove_field(self, model, field):
        """
        Remove a field from a model. Usually involves deleting a column,
        but for M2Ms may involve deleting a table.
        """
        # M2M fields are a special case
        if field.many_to_many:
            # For implicit M2M tables, delete the auto-created table
            if field.remote_field.through._meta.auto_created:
                self.delete_model(field.remote_field.through)
            # For explicit "through" M2M fields, do nothing
        elif (
            self.connection.features.can_alter_table_drop_column
            # Primary keys, unique fields, indexed fields, and foreign keys are
            # not supported in ALTER TABLE DROP COLUMN.
            and not field.primary_key
            and not field.unique
            and not field.db_index
            and not (field.remote_field and field.db_constraint)
        ):
            super().remove_field(model, field)
        # For everything else, remake.
        else:
            # It might not actually have a column behind it
            if field.db_parameters(connection=self.connection)["type"] is None:
                return
            self._remake_table(model, delete_field=field)
```
### 139 - django/db/backends/sqlite3/schema.py:

Start line: 380, End line: 396

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def add_field(self, model, field):
        """Create a field on a model."""
        if (
            # Primary keys and unique fields are not supported in ALTER TABLE
            # ADD COLUMN.
            field.primary_key
            or field.unique
            or
            # Fields with default values cannot by handled by ALTER TABLE ADD
            # COLUMN statement because DROP DEFAULT is not supported in
            # ALTER TABLE.
            not field.null
            or self.effective_default(field) is not None
        ):
            self._remake_table(model, create_field=field)
        else:
            super().add_field(model, field)
```
