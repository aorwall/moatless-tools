# django__django-15993

| **django/django** | `71902e0d9f93670c4f93ff9d66095b0e571be74b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1193 |
| **Any found context length** | 1193 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -371,13 +371,12 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
         new_model = to_state.apps.get_model(app_label, self.new_name)
         if self.allow_migrate_model(schema_editor.connection.alias, new_model):
             old_model = from_state.apps.get_model(app_label, self.old_name)
-            old_db_table = old_model._meta.db_table
-            new_db_table = new_model._meta.db_table
-            # Don't alter when a table name is not changed.
-            if old_db_table == new_db_table:
-                return
             # Move the main table
-            schema_editor.alter_db_table(new_model, old_db_table, new_db_table)
+            schema_editor.alter_db_table(
+                new_model,
+                old_model._meta.db_table,
+                new_model._meta.db_table,
+            )
             # Alter the fields pointing to us
             for related_object in old_model._meta.related_objects:
                 if related_object.related_model == old_model:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/operations/models.py | 374 | 380 | 3 | 1 | 1193


## Problem Statement

```
RenameModel with db_table should be a noop.
Description
	
A RenameModel operation that already has db_table defined must be a noop.
In Postgres, it drops and recreates foreign key constraints. In sqlite it recreates the table (as expected for a table renaming).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/migrations/operations/models.py** | 482 | 530| 409 | 409 | 7751 | 
| 2 | 2 django/db/backends/base/schema.py | 597 | 625| 245 | 654 | 21612 | 
| **-> 3 <-** | **2 django/db/migrations/operations/models.py** | 370 | 425| 539 | 1193 | 21612 | 
| 4 | **2 django/db/migrations/operations/models.py** | 427 | 440| 127 | 1320 | 21612 | 
| 5 | 3 django/db/backends/sqlite3/schema.py | 100 | 121| 195 | 1515 | 26195 | 
| 6 | **3 django/db/migrations/operations/models.py** | 442 | 479| 267 | 1782 | 26195 | 
| 7 | 3 django/db/backends/sqlite3/schema.py | 176 | 264| 822 | 2604 | 26195 | 
| 8 | **3 django/db/migrations/operations/models.py** | 344 | 368| 157 | 2761 | 26195 | 
| 9 | 4 django/db/migrations/operations/fields.py | 270 | 337| 521 | 3282 | 28707 | 
| 10 | **4 django/db/migrations/operations/models.py** | 309 | 341| 249 | 3531 | 28707 | 
| 11 | 5 django/db/backends/mysql/schema.py | 1 | 42| 456 | 3987 | 30330 | 
| 12 | 5 django/db/backends/sqlite3/schema.py | 265 | 360| 807 | 4794 | 30330 | 
| 13 | 6 django/db/backends/oracle/schema.py | 52 | 73| 146 | 4940 | 32672 | 
| 14 | 6 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 5467 | 32672 | 
| 15 | **6 django/db/migrations/operations/models.py** | 41 | 111| 524 | 5991 | 32672 | 
| 16 | 6 django/db/backends/sqlite3/schema.py | 362 | 378| 132 | 6123 | 32672 | 
| 17 | 6 django/db/backends/mysql/schema.py | 44 | 53| 134 | 6257 | 32672 | 
| 18 | 6 django/db/backends/base/schema.py | 437 | 454| 160 | 6417 | 32672 | 
| 19 | 6 django/db/backends/base/schema.py | 1614 | 1651| 243 | 6660 | 32672 | 
| 20 | 6 django/db/backends/base/schema.py | 1390 | 1406| 153 | 6813 | 32672 | 
| 21 | **6 django/db/migrations/operations/models.py** | 934 | 969| 319 | 7132 | 32672 | 
| 22 | **6 django/db/migrations/operations/models.py** | 971 | 988| 149 | 7281 | 32672 | 
| 23 | 6 django/db/backends/base/schema.py | 1455 | 1474| 175 | 7456 | 32672 | 
| 24 | 6 django/db/backends/oracle/schema.py | 105 | 170| 740 | 8196 | 32672 | 
| 25 | 7 django/contrib/postgres/operations.py | 289 | 330| 278 | 8474 | 35020 | 
| 26 | 8 django/db/backends/postgresql/schema.py | 219 | 265| 408 | 8882 | 37467 | 
| 27 | 8 django/db/backends/base/schema.py | 888 | 972| 778 | 9660 | 37467 | 
| 28 | 8 django/db/backends/postgresql/schema.py | 133 | 198| 517 | 10177 | 37467 | 
| 29 | **8 django/db/migrations/operations/models.py** | 879 | 932| 348 | 10525 | 37467 | 
| 30 | 8 django/db/backends/base/schema.py | 1497 | 1519| 199 | 10724 | 37467 | 
| 31 | 8 django/db/backends/base/schema.py | 521 | 540| 196 | 10920 | 37467 | 
| 32 | 8 django/db/backends/base/schema.py | 1718 | 1755| 303 | 11223 | 37467 | 
| 33 | 8 django/db/backends/base/schema.py | 456 | 475| 151 | 11374 | 37467 | 
| 34 | 8 django/contrib/postgres/operations.py | 255 | 286| 248 | 11622 | 37467 | 
| 35 | 8 django/db/backends/oracle/schema.py | 75 | 103| 344 | 11966 | 37467 | 
| 36 | 9 django/db/migrations/autodetector.py | 516 | 581| 482 | 12448 | 50928 | 
| 37 | 10 django/db/models/base.py | 1785 | 1807| 171 | 12619 | 69479 | 
| 38 | **10 django/db/migrations/operations/models.py** | 990 | 1025| 249 | 12868 | 69479 | 
| 39 | 10 django/db/migrations/autodetector.py | 908 | 981| 623 | 13491 | 69479 | 
| 40 | 10 django/db/backends/postgresql/schema.py | 267 | 292| 235 | 13726 | 69479 | 
| 41 | 10 django/db/backends/sqlite3/schema.py | 426 | 483| 488 | 14214 | 69479 | 
| 42 | 10 django/db/models/base.py | 2157 | 2238| 588 | 14802 | 69479 | 
| 43 | 11 django/contrib/gis/db/backends/spatialite/schema.py | 137 | 192| 404 | 15206 | 70873 | 
| 44 | 11 django/db/backends/oracle/schema.py | 226 | 253| 212 | 15418 | 70873 | 
| 45 | 11 django/db/backends/postgresql/schema.py | 200 | 217| 170 | 15588 | 70873 | 
| 46 | **11 django/db/migrations/operations/models.py** | 136 | 306| 968 | 16556 | 70873 | 
| 47 | 11 django/db/backends/base/schema.py | 563 | 595| 268 | 16824 | 70873 | 
| 48 | 12 django/db/backends/oracle/operations.py | 455 | 518| 532 | 17356 | 77093 | 
| 49 | 12 django/db/backends/sqlite3/schema.py | 485 | 535| 395 | 17751 | 77093 | 
| 50 | 12 django/db/backends/mysql/schema.py | 158 | 176| 192 | 17943 | 77093 | 
| 51 | 13 django/db/models/constraints.py | 251 | 265| 117 | 18060 | 80021 | 
| 52 | 13 django/db/models/base.py | 2256 | 2447| 1302 | 19362 | 80021 | 
| 53 | 13 django/db/backends/base/schema.py | 1061 | 1141| 779 | 20141 | 80021 | 
| 54 | 13 django/db/backends/sqlite3/schema.py | 398 | 424| 256 | 20397 | 80021 | 
| 55 | 13 django/db/models/base.py | 1128 | 1147| 188 | 20585 | 80021 | 
| 56 | 14 django/db/migrations/state.py | 142 | 179| 395 | 20980 | 88189 | 
| 57 | 15 django/db/backends/sqlite3/base.py | 233 | 352| 887 | 21867 | 91278 | 
| 58 | **15 django/db/migrations/operations/models.py** | 1069 | 1109| 337 | 22204 | 91278 | 
| 59 | 15 django/db/migrations/autodetector.py | 1397 | 1430| 289 | 22493 | 91278 | 
| 60 | 15 django/db/backends/base/schema.py | 973 | 1060| 824 | 23317 | 91278 | 
| 61 | 16 django/db/migrations/questioner.py | 217 | 247| 252 | 23569 | 93974 | 
| 62 | 16 django/db/backends/oracle/schema.py | 172 | 181| 142 | 23711 | 93974 | 
| 63 | **16 django/db/migrations/operations/models.py** | 690 | 742| 320 | 24031 | 93974 | 
| 64 | 16 django/db/backends/base/schema.py | 204 | 281| 646 | 24677 | 93974 | 
| 65 | 16 django/db/backends/oracle/operations.py | 538 | 562| 250 | 24927 | 93974 | 
| 66 | 17 django/db/backends/postgresql/operations.py | 187 | 210| 227 | 25154 | 96963 | 
| 67 | 17 django/db/backends/postgresql/schema.py | 294 | 330| 210 | 25364 | 96963 | 
| 68 | 17 django/db/migrations/operations/fields.py | 339 | 358| 153 | 25517 | 96963 | 
| 69 | 17 django/db/backends/oracle/schema.py | 183 | 209| 217 | 25734 | 96963 | 
| 70 | 17 django/db/migrations/state.py | 291 | 345| 476 | 26210 | 96963 | 
| 71 | 17 django/db/backends/base/schema.py | 701 | 731| 293 | 26503 | 96963 | 
| 72 | 17 django/db/backends/postgresql/operations.py | 212 | 244| 314 | 26817 | 96963 | 
| 73 | **17 django/db/migrations/operations/models.py** | 113 | 134| 164 | 26981 | 96963 | 
| 74 | 17 django/db/backends/oracle/operations.py | 407 | 453| 385 | 27366 | 96963 | 
| 75 | 17 django/db/backends/base/schema.py | 542 | 561| 199 | 27565 | 96963 | 
| 76 | 17 django/db/backends/base/schema.py | 1247 | 1277| 318 | 27883 | 96963 | 
| 77 | 17 django/db/backends/postgresql/schema.py | 119 | 131| 179 | 28062 | 96963 | 
| 78 | 17 django/db/backends/base/schema.py | 797 | 887| 773 | 28835 | 96963 | 
| 79 | 17 django/db/migrations/autodetector.py | 983 | 1025| 297 | 29132 | 96963 | 
| 80 | 18 django/db/models/sql/query.py | 805 | 838| 300 | 29432 | 120134 | 
| 81 | 18 django/db/backends/base/schema.py | 1476 | 1495| 176 | 29608 | 120134 | 
| 82 | 18 django/db/backends/base/schema.py | 1 | 36| 223 | 29831 | 120134 | 
| 83 | 18 django/db/backends/sqlite3/schema.py | 537 | 561| 162 | 29993 | 120134 | 
| 84 | 18 django/db/backends/base/schema.py | 1653 | 1675| 173 | 30166 | 120134 | 
| 85 | 18 django/db/models/base.py | 1760 | 1783| 176 | 30342 | 120134 | 
| 86 | 18 django/db/backends/base/schema.py | 75 | 152| 798 | 31140 | 120134 | 
| 87 | 18 django/db/backends/oracle/operations.py | 651 | 669| 225 | 31365 | 120134 | 
| 88 | 19 django/db/backends/sqlite3/operations.py | 189 | 214| 190 | 31555 | 123621 | 
| 89 | 19 django/db/migrations/autodetector.py | 1534 | 1553| 187 | 31742 | 123621 | 
| 90 | **19 django/db/migrations/operations/models.py** | 645 | 669| 231 | 31973 | 123621 | 
| 91 | 19 django/db/backends/base/schema.py | 1561 | 1612| 343 | 32316 | 123621 | 
| 92 | 19 django/db/backends/sqlite3/schema.py | 25 | 41| 168 | 32484 | 123621 | 
| 93 | 20 django/db/backends/ddl_references.py | 223 | 255| 252 | 32736 | 125250 | 
| 94 | 20 django/db/models/base.py | 1074 | 1126| 516 | 33252 | 125250 | 
| 95 | 20 django/db/backends/oracle/operations.py | 520 | 536| 207 | 33459 | 125250 | 
| 96 | 20 django/db/models/base.py | 943 | 1031| 686 | 34145 | 125250 | 
| 97 | 21 django/db/backends/oracle/creation.py | 159 | 201| 411 | 34556 | 129265 | 
| 98 | 21 django/db/models/base.py | 1523 | 1558| 273 | 34829 | 129265 | 
| 99 | 21 django/db/backends/postgresql/operations.py | 338 | 357| 150 | 34979 | 129265 | 
| 100 | 21 django/db/backends/mysql/schema.py | 104 | 118| 144 | 35123 | 129265 | 
| 101 | 21 django/db/backends/mysql/schema.py | 138 | 156| 209 | 35332 | 129265 | 
| 102 | 22 django/core/management/commands/createcachetable.py | 42 | 54| 121 | 35453 | 130156 | 
| 103 | 22 django/db/backends/sqlite3/schema.py | 380 | 396| 138 | 35591 | 130156 | 
| 104 | 23 django/db/models/sql/datastructures.py | 126 | 144| 141 | 35732 | 131649 | 
| 105 | **23 django/db/migrations/operations/models.py** | 21 | 38| 116 | 35848 | 131649 | 
| 106 | 23 django/db/models/constraints.py | 233 | 249| 139 | 35987 | 131649 | 
| 107 | 23 django/db/models/sql/datastructures.py | 179 | 221| 273 | 36260 | 131649 | 
| 108 | 23 django/db/migrations/questioner.py | 189 | 215| 238 | 36498 | 131649 | 
| 109 | 23 django/core/management/commands/createcachetable.py | 56 | 131| 552 | 37050 | 131649 | 
| 110 | 24 django/db/backends/mysql/operations.py | 203 | 232| 218 | 37268 | 135827 | 
| 111 | 24 django/db/backends/base/schema.py | 627 | 699| 665 | 37933 | 135827 | 
| 112 | 24 django/db/models/base.py | 1705 | 1758| 490 | 38423 | 135827 | 
| 113 | 25 django/contrib/postgres/constraints.py | 129 | 153| 184 | 38607 | 137784 | 
| 114 | 26 django/db/backends/mysql/base.py | 314 | 368| 421 | 39028 | 141296 | 
| 115 | 26 django/db/backends/ddl_references.py | 138 | 181| 297 | 39325 | 141296 | 
| 116 | 27 django/db/backends/postgresql/creation.py | 1 | 39| 266 | 39591 | 141979 | 
| 117 | 27 django/db/backends/postgresql/schema.py | 1 | 81| 690 | 40281 | 141979 | 
| 118 | 27 django/contrib/gis/db/backends/spatialite/schema.py | 91 | 110| 155 | 40436 | 141979 | 
| 119 | 27 django/db/backends/sqlite3/operations.py | 243 | 260| 143 | 40579 | 141979 | 
| 120 | 27 django/db/backends/base/schema.py | 1521 | 1559| 240 | 40819 | 141979 | 
| 121 | 27 django/db/models/base.py | 248 | 365| 874 | 41693 | 141979 | 
| 122 | **27 django/db/migrations/operations/models.py** | 671 | 687| 159 | 41852 | 141979 | 
| 123 | 27 django/db/models/sql/query.py | 905 | 948| 445 | 42297 | 141979 | 
| 124 | 27 django/db/backends/base/schema.py | 1279 | 1309| 331 | 42628 | 141979 | 
| 125 | 27 django/db/backends/oracle/creation.py | 302 | 311| 114 | 42742 | 141979 | 
| 126 | 28 django/db/models/sql/compiler.py | 1764 | 1796| 254 | 42996 | 158132 | 
| 127 | 28 django/db/backends/sqlite3/operations.py | 216 | 241| 218 | 43214 | 158132 | 
| 128 | 28 django/db/models/constraints.py | 215 | 231| 138 | 43352 | 158132 | 
| 129 | 28 django/db/backends/base/schema.py | 477 | 519| 316 | 43668 | 158132 | 
| 130 | 28 django/db/migrations/questioner.py | 57 | 87| 255 | 43923 | 158132 | 
| 131 | 28 django/db/backends/mysql/schema.py | 120 | 136| 144 | 44067 | 158132 | 
| 132 | 28 django/db/models/base.py | 1395 | 1425| 216 | 44283 | 158132 | 
| 133 | 28 django/db/backends/mysql/operations.py | 234 | 253| 163 | 44446 | 158132 | 
| 134 | 29 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 44673 | 158359 | 
| 135 | 29 django/db/backends/sqlite3/schema.py | 1 | 23| 191 | 44864 | 158359 | 
| 136 | 29 django/db/backends/postgresql/schema.py | 83 | 117| 332 | 45196 | 158359 | 
| 137 | 29 django/db/models/base.py | 1560 | 1590| 243 | 45439 | 158359 | 
| 138 | 29 django/db/backends/postgresql/creation.py | 58 | 88| 255 | 45694 | 158359 | 
| 139 | 29 django/db/models/sql/query.py | 1077 | 1108| 307 | 46001 | 158359 | 
| 140 | 30 django/contrib/contenttypes/management/__init__.py | 1 | 43| 358 | 46359 | 159347 | 
| 141 | 30 django/db/models/constraints.py | 16 | 44| 232 | 46591 | 159347 | 
| 142 | 30 django/db/backends/oracle/schema.py | 211 | 224| 115 | 46706 | 159347 | 
| 143 | **30 django/db/migrations/operations/models.py** | 836 | 876| 344 | 47050 | 159347 | 
| 144 | 30 django/db/models/base.py | 1298 | 1345| 409 | 47459 | 159347 | 
| 145 | 30 django/db/backends/base/schema.py | 1143 | 1170| 195 | 47654 | 159347 | 
| 146 | 30 django/db/backends/oracle/operations.py | 368 | 379| 226 | 47880 | 159347 | 
| 147 | 30 django/db/backends/oracle/operations.py | 21 | 81| 630 | 48510 | 159347 | 
| 148 | 30 django/db/models/base.py | 195 | 247| 458 | 48968 | 159347 | 
| 149 | 30 django/db/backends/base/schema.py | 283 | 338| 461 | 49429 | 159347 | 
| 150 | 30 django/db/backends/oracle/creation.py | 313 | 334| 180 | 49609 | 159347 | 
| 151 | **30 django/db/migrations/operations/models.py** | 560 | 569| 129 | 49738 | 159347 | 
| 152 | 30 django/db/backends/base/schema.py | 733 | 795| 524 | 50262 | 159347 | 
| 153 | 30 django/db/models/base.py | 1451 | 1496| 311 | 50573 | 159347 | 
| 154 | 31 django/contrib/gis/db/backends/oracle/schema.py | 74 | 122| 323 | 50896 | 160258 | 
| 155 | 32 django/db/models/options.py | 412 | 438| 204 | 51100 | 167846 | 
| 156 | 32 django/db/backends/ddl_references.py | 1 | 41| 222 | 51322 | 167846 | 
| 157 | 32 django/db/models/base.py | 477 | 590| 953 | 52275 | 167846 | 
| 158 | 32 django/contrib/postgres/constraints.py | 107 | 127| 202 | 52477 | 167846 | 
| 159 | 32 django/db/models/base.py | 592 | 630| 322 | 52799 | 167846 | 
| 160 | 33 django/db/backends/base/creation.py | 314 | 325| 117 | 52916 | 170787 | 
| 161 | 34 django/db/models/indexes.py | 156 | 189| 334 | 53250 | 173173 | 
| 162 | 34 django/db/models/sql/query.py | 1931 | 1979| 445 | 53695 | 173173 | 
| 163 | **34 django/db/migrations/operations/models.py** | 1028 | 1066| 283 | 53978 | 173173 | 
| 164 | 35 django/core/management/commands/inspectdb.py | 54 | 253| 1562 | 55540 | 176145 | 
| 165 | 35 django/db/backends/sqlite3/operations.py | 417 | 437| 148 | 55688 | 176145 | 
| 166 | 35 django/db/backends/oracle/schema.py | 1 | 29| 289 | 55977 | 176145 | 
| 167 | 35 django/core/management/commands/inspectdb.py | 255 | 313| 487 | 56464 | 176145 | 
| 168 | 35 django/contrib/postgres/constraints.py | 196 | 236| 367 | 56831 | 176145 | 
| 169 | **35 django/db/migrations/operations/models.py** | 1 | 18| 137 | 56968 | 176145 | 
| 170 | 35 django/db/models/base.py | 1427 | 1449| 193 | 57161 | 176145 | 
| 171 | 35 django/db/backends/postgresql/creation.py | 41 | 56| 174 | 57335 | 176145 | 
| 172 | **35 django/db/migrations/operations/models.py** | 571 | 595| 213 | 57548 | 176145 | 
| 173 | 36 django/db/backends/sqlite3/features.py | 66 | 113| 383 | 57931 | 177392 | 
| 174 | 36 django/db/backends/mysql/operations.py | 436 | 465| 274 | 58205 | 177392 | 
| 175 | 36 django/db/backends/base/schema.py | 39 | 72| 214 | 58419 | 177392 | 
| 176 | 37 django/forms/models.py | 461 | 491| 244 | 58663 | 189586 | 
| 177 | 38 django/db/backends/mysql/creation.py | 1 | 29| 221 | 58884 | 190255 | 
| 178 | 38 django/contrib/postgres/operations.py | 232 | 252| 160 | 59044 | 190255 | 
| 179 | 38 django/contrib/gis/db/backends/spatialite/schema.py | 68 | 89| 137 | 59181 | 190255 | 
| 180 | 38 django/db/backends/oracle/schema.py | 31 | 50| 206 | 59387 | 190255 | 
| 181 | 38 django/db/models/base.py | 1995 | 2048| 355 | 59742 | 190255 | 
| 182 | 38 django/db/models/base.py | 1193 | 1233| 306 | 60048 | 190255 | 
| 183 | 38 django/db/migrations/operations/fields.py | 154 | 195| 348 | 60396 | 190255 | 
| 184 | 38 django/db/migrations/state.py | 126 | 140| 163 | 60559 | 190255 | 
| 185 | 38 django/db/backends/ddl_references.py | 44 | 76| 208 | 60767 | 190255 | 
| 186 | 38 django/db/migrations/state.py | 181 | 238| 598 | 61365 | 190255 | 
| 187 | 38 django/contrib/postgres/operations.py | 39 | 63| 197 | 61562 | 190255 | 
| 188 | 38 django/db/backends/mysql/creation.py | 31 | 60| 261 | 61823 | 190255 | 
| 189 | 38 django/db/backends/oracle/creation.py | 29 | 124| 768 | 62591 | 190255 | 
| 190 | 38 django/db/backends/base/creation.py | 353 | 381| 216 | 62807 | 190255 | 
| 191 | 39 django/db/models/sql/__init__.py | 1 | 7| 0 | 62807 | 190321 | 
| 192 | 40 django/db/backends/sqlite3/creation.py | 25 | 52| 242 | 63049 | 191613 | 
| 193 | 40 django/db/migrations/autodetector.py | 1101 | 1218| 982 | 64031 | 191613 | 
| 194 | 40 django/db/models/base.py | 2450 | 2502| 341 | 64372 | 191613 | 
| 195 | 40 django/db/models/base.py | 1592 | 1617| 184 | 64556 | 191613 | 


### Hint

```
​PR
I fixed the patch and it is waiting for review.
In afeafd60: Fixed #33201 -- Made RenameModel operation a noop for models with db_table.
```

## Patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -371,13 +371,12 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
         new_model = to_state.apps.get_model(app_label, self.new_name)
         if self.allow_migrate_model(schema_editor.connection.alias, new_model):
             old_model = from_state.apps.get_model(app_label, self.old_name)
-            old_db_table = old_model._meta.db_table
-            new_db_table = new_model._meta.db_table
-            # Don't alter when a table name is not changed.
-            if old_db_table == new_db_table:
-                return
             # Move the main table
-            schema_editor.alter_db_table(new_model, old_db_table, new_db_table)
+            schema_editor.alter_db_table(
+                new_model,
+                old_model._meta.db_table,
+                new_model._meta.db_table,
+            )
             # Alter the fields pointing to us
             for related_object in old_model._meta.related_objects:
                 if related_object.related_model == old_model:

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -1058,8 +1058,8 @@ def test_rename_model_with_m2m(self):
             Pony._meta.get_field("riders").remote_field.through.objects.count(), 2
         )
 
-    def test_rename_model_with_db_table_noop(self):
-        app_label = "test_rmwdbtn"
+    def test_rename_model_with_db_table_rename_m2m(self):
+        app_label = "test_rmwdbrm2m"
         project_state = self.apply_operations(
             app_label,
             ProjectState(),
@@ -1069,32 +1069,28 @@ def test_rename_model_with_db_table_noop(self):
                     fields=[
                         ("id", models.AutoField(primary_key=True)),
                     ],
-                    options={"db_table": "rider"},
                 ),
                 migrations.CreateModel(
                     "Pony",
                     fields=[
                         ("id", models.AutoField(primary_key=True)),
-                        (
-                            "rider",
-                            models.ForeignKey("%s.Rider" % app_label, models.CASCADE),
-                        ),
+                        ("riders", models.ManyToManyField("Rider")),
                     ],
+                    options={"db_table": "pony"},
                 ),
             ],
         )
         new_state = project_state.clone()
-        operation = migrations.RenameModel("Rider", "Runner")
+        operation = migrations.RenameModel("Pony", "PinkPony")
         operation.state_forwards(app_label, new_state)
-
-        with connection.schema_editor() as editor:
-            with self.assertNumQueries(0):
-                operation.database_forwards(app_label, editor, project_state, new_state)
         with connection.schema_editor() as editor:
-            with self.assertNumQueries(0):
-                operation.database_backwards(
-                    app_label, editor, new_state, project_state
-                )
+            operation.database_forwards(app_label, editor, project_state, new_state)
+
+        Pony = new_state.apps.get_model(app_label, "PinkPony")
+        Rider = new_state.apps.get_model(app_label, "Rider")
+        pony = Pony.objects.create()
+        rider = Rider.objects.create()
+        pony.riders.add(rider)
 
     def test_rename_m2m_target_model(self):
         app_label = "test_rename_m2m_target_model"

```


## Code snippets

### 1 - django/db/migrations/operations/models.py:

Start line: 482, End line: 530

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
            for (old_field, new_field) in zip(
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
### 2 - django/db/backends/base/schema.py:

Start line: 597, End line: 625

```python
class BaseDatabaseSchemaEditor:

    def alter_db_table(self, model, old_db_table, new_db_table):
        """Rename the table a model points to."""
        if old_db_table == new_db_table or (
            self.connection.features.ignores_table_name_case
            and old_db_table.lower() == new_db_table.lower()
        ):
            return
        self.execute(
            self.sql_rename_table
            % {
                "old_table": self.quote_name(old_db_table),
                "new_table": self.quote_name(new_db_table),
            }
        )
        # Rename all references to the old table name.
        for sql in self.deferred_sql:
            if isinstance(sql, Statement):
                sql.rename_table_references(old_db_table, new_db_table)

    def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
        """Move a model's table between tablespaces."""
        self.execute(
            self.sql_retablespace_table
            % {
                "table": self.quote_name(model._meta.db_table),
                "old_tablespace": self.quote_name(old_db_tablespace),
                "new_tablespace": self.quote_name(new_db_tablespace),
            }
        )
```
### 3 - django/db/migrations/operations/models.py:

Start line: 370, End line: 425

```python
class RenameModel(ModelOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.new_name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.old_name)
            old_db_table = old_model._meta.db_table
            new_db_table = new_model._meta.db_table
            # Don't alter when a table name is not changed.
            if old_db_table == new_db_table:
                return
            # Move the main table
            schema_editor.alter_db_table(new_model, old_db_table, new_db_table)
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
            for (old_field, new_field) in fields:
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
### 4 - django/db/migrations/operations/models.py:

Start line: 427, End line: 440

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
### 5 - django/db/backends/sqlite3/schema.py:

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
### 6 - django/db/migrations/operations/models.py:

Start line: 442, End line: 479

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
### 7 - django/db/backends/sqlite3/schema.py:

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
### 8 - django/db/migrations/operations/models.py:

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
### 15 - django/db/migrations/operations/models.py:

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
### 21 - django/db/migrations/operations/models.py:

Start line: 934, End line: 969

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
### 22 - django/db/migrations/operations/models.py:

Start line: 971, End line: 988

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
### 29 - django/db/migrations/operations/models.py:

Start line: 879, End line: 932

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
### 38 - django/db/migrations/operations/models.py:

Start line: 990, End line: 1025

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
### 46 - django/db/migrations/operations/models.py:

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
### 58 - django/db/migrations/operations/models.py:

Start line: 1069, End line: 1109

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
### 63 - django/db/migrations/operations/models.py:

Start line: 690, End line: 742

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
### 73 - django/db/migrations/operations/models.py:

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
### 90 - django/db/migrations/operations/models.py:

Start line: 645, End line: 669

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
### 105 - django/db/migrations/operations/models.py:

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
### 122 - django/db/migrations/operations/models.py:

Start line: 671, End line: 687

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
### 143 - django/db/migrations/operations/models.py:

Start line: 836, End line: 876

```python
class RemoveIndex(IndexOperation):
    """Remove an index from a model."""

    def __init__(self, model_name, name):
        self.model_name = model_name
        self.name = name

    def state_forwards(self, app_label, state):
        state.remove_index(app_label, self.model_name_lower, self.name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            from_model_state = from_state.models[app_label, self.model_name_lower]
            index = from_model_state.get_index_by_name(self.name)
            schema_editor.remove_index(model, index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            to_model_state = to_state.models[app_label, self.model_name_lower]
            index = to_model_state.get_index_by_name(self.name)
            schema_editor.add_index(model, index)

    def deconstruct(self):
        kwargs = {
            "model_name": self.model_name,
            "name": self.name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs,
        )

    def describe(self):
        return "Remove index %s from %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return "remove_%s_%s" % (self.model_name_lower, self.name.lower())
```
### 151 - django/db/migrations/operations/models.py:

Start line: 560, End line: 569

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
### 163 - django/db/migrations/operations/models.py:

Start line: 1028, End line: 1066

```python
class AddConstraint(IndexOperation):
    option_name = "constraints"

    def __init__(self, model_name, constraint):
        self.model_name = model_name
        self.constraint = constraint

    def state_forwards(self, app_label, state):
        state.add_constraint(app_label, self.model_name_lower, self.constraint)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.add_constraint(model, self.constraint)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.remove_constraint(model, self.constraint)

    def deconstruct(self):
        return (
            self.__class__.__name__,
            [],
            {
                "model_name": self.model_name,
                "constraint": self.constraint,
            },
        )

    def describe(self):
        return "Create constraint %s on model %s" % (
            self.constraint.name,
            self.model_name,
        )

    @property
    def migration_name_fragment(self):
        return "%s_%s" % (self.model_name_lower, self.constraint.name.lower())
```
### 169 - django/db/migrations/operations/models.py:

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
### 172 - django/db/migrations/operations/models.py:

Start line: 571, End line: 595

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
