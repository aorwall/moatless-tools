# django__django-14999

| **django/django** | `a754b82dac511475b6276039471ccd17cc64aeb8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1137 |
| **Any found context length** | 1137 |
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
@@ -320,12 +320,13 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
         new_model = to_state.apps.get_model(app_label, self.new_name)
         if self.allow_migrate_model(schema_editor.connection.alias, new_model):
             old_model = from_state.apps.get_model(app_label, self.old_name)
+            old_db_table = old_model._meta.db_table
+            new_db_table = new_model._meta.db_table
+            # Don't alter when a table name is not changed.
+            if old_db_table == new_db_table:
+                return
             # Move the main table
-            schema_editor.alter_db_table(
-                new_model,
-                old_model._meta.db_table,
-                new_model._meta.db_table,
-            )
+            schema_editor.alter_db_table(new_model, old_db_table, new_db_table)
             # Alter the fields pointing to us
             for related_object in old_model._meta.related_objects:
                 if related_object.related_model == old_model:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/operations/models.py | 323 | 327 | 3 | 1 | 1137


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
| 1 | **1 django/db/migrations/operations/models.py** | 416 | 466| 410 | 410 | 6450 | 
| 2 | 2 django/db/backends/base/schema.py | 468 | 489| 234 | 644 | 19332 | 
| **-> 3 <-** | **2 django/db/migrations/operations/models.py** | 319 | 368| 493 | 1137 | 19332 | 
| 4 | **2 django/db/migrations/operations/models.py** | 370 | 390| 213 | 1350 | 19332 | 
| 5 | 3 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 2170 | 23535 | 
| 6 | 3 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 2351 | 23535 | 
| 7 | 3 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 3082 | 23535 | 
| 8 | **3 django/db/migrations/operations/models.py** | 289 | 317| 162 | 3244 | 23535 | 
| 9 | 4 django/db/migrations/operations/fields.py | 258 | 324| 519 | 3763 | 26028 | 
| 10 | **4 django/db/migrations/operations/models.py** | 250 | 286| 254 | 4017 | 26028 | 
| 11 | **4 django/db/migrations/operations/models.py** | 392 | 413| 170 | 4187 | 26028 | 
| 12 | 5 django/db/backends/mysql/schema.py | 1 | 39| 428 | 4615 | 27602 | 
| 13 | 6 django/db/backends/oracle/schema.py | 46 | 60| 133 | 4748 | 29745 | 
| 14 | 6 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 5253 | 29745 | 
| 15 | **6 django/db/migrations/operations/models.py** | 41 | 104| 513 | 5766 | 29745 | 
| 16 | 6 django/db/backends/mysql/schema.py | 41 | 50| 134 | 5900 | 29745 | 
| 17 | 6 django/db/backends/base/schema.py | 348 | 363| 154 | 6054 | 29745 | 
| 18 | 6 django/db/backends/base/schema.py | 684 | 754| 799 | 6853 | 29745 | 
| 19 | 6 django/db/backends/base/schema.py | 1151 | 1170| 175 | 7028 | 29745 | 
| 20 | 7 django/contrib/postgres/operations.py | 296 | 330| 267 | 7295 | 32131 | 
| 21 | 7 django/db/backends/base/schema.py | 1289 | 1308| 163 | 7458 | 32131 | 
| 22 | 7 django/db/backends/oracle/schema.py | 84 | 139| 715 | 8173 | 32131 | 
| 23 | 8 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 8524 | 34299 | 
| 24 | 8 django/db/backends/base/schema.py | 1364 | 1396| 293 | 8817 | 34299 | 
| 25 | 9 django/db/migrations/autodetector.py | 808 | 872| 676 | 9493 | 46411 | 
| 26 | 9 django/db/backends/base/schema.py | 415 | 429| 183 | 9676 | 46411 | 
| 27 | 9 django/db/backends/base/schema.py | 1189 | 1211| 199 | 9875 | 46411 | 
| 28 | 9 django/db/backends/base/schema.py | 365 | 379| 141 | 10016 | 46411 | 
| 29 | 9 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 10438 | 46411 | 
| 30 | 9 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 10656 | 46411 | 
| 31 | 9 django/db/backends/oracle/schema.py | 62 | 82| 249 | 10905 | 46411 | 
| 32 | 9 django/contrib/postgres/operations.py | 262 | 293| 248 | 11153 | 46411 | 
| 33 | 9 django/db/backends/base/schema.py | 755 | 833| 826 | 11979 | 46411 | 
| 34 | 9 django/db/migrations/autodetector.py | 466 | 518| 465 | 12444 | 46411 | 
| 35 | 9 django/db/backends/base/schema.py | 834 | 874| 506 | 12950 | 46411 | 
| 36 | 9 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 13597 | 46411 | 
| 37 | 10 django/db/models/base.py | 1522 | 1544| 171 | 13768 | 63741 | 
| 38 | 11 django/contrib/gis/db/backends/spatialite/schema.py | 128 | 169| 376 | 14144 | 65093 | 
| 39 | **11 django/db/migrations/operations/models.py** | 124 | 247| 853 | 14997 | 65093 | 
| 40 | 11 django/db/models/base.py | 1875 | 1948| 572 | 15569 | 65093 | 
| 41 | 11 django/db/backends/sqlite3/schema.py | 386 | 419| 358 | 15927 | 65093 | 
| 42 | 11 django/db/backends/base/schema.py | 452 | 466| 174 | 16101 | 65093 | 
| 43 | 11 django/db/backends/mysql/schema.py | 142 | 160| 192 | 16293 | 65093 | 
| 44 | 12 django/db/migrations/questioner.py | 198 | 216| 233 | 16526 | 67617 | 
| 45 | 12 django/db/models/base.py | 949 | 966| 181 | 16707 | 67617 | 
| 46 | 13 django/db/migrations/state.py | 133 | 168| 391 | 17098 | 75480 | 
| 47 | **13 django/db/migrations/operations/models.py** | 849 | 885| 331 | 17429 | 75480 | 
| 48 | 14 django/db/backends/oracle/operations.py | 412 | 463| 516 | 17945 | 81473 | 
| 49 | 14 django/db/backends/sqlite3/schema.py | 332 | 348| 173 | 18118 | 81473 | 
| 50 | 15 django/db/backends/sqlite3/base.py | 312 | 401| 850 | 18968 | 87543 | 
| 51 | 15 django/db/backends/oracle/schema.py | 152 | 206| 493 | 19461 | 87543 | 
| 52 | 15 django/db/backends/oracle/schema.py | 141 | 150| 142 | 19603 | 87543 | 
| 53 | 16 django/db/backends/postgresql/operations.py | 160 | 187| 311 | 19914 | 90104 | 
| 54 | 16 django/db/backends/base/schema.py | 156 | 211| 608 | 20522 | 90104 | 
| 55 | **16 django/db/migrations/operations/models.py** | 618 | 674| 325 | 20847 | 90104 | 
| 56 | 16 django/db/backends/oracle/operations.py | 481 | 500| 240 | 21087 | 90104 | 
| 57 | 16 django/db/backends/postgresql/operations.py | 138 | 158| 221 | 21308 | 90104 | 
| 58 | 16 django/db/migrations/operations/fields.py | 326 | 346| 160 | 21468 | 90104 | 
| 59 | 16 django/db/backends/base/schema.py | 548 | 576| 289 | 21757 | 90104 | 
| 60 | 16 django/db/migrations/state.py | 259 | 309| 468 | 22225 | 90104 | 
| 61 | 16 django/db/models/base.py | 915 | 947| 385 | 22610 | 90104 | 
| 62 | **16 django/db/migrations/operations/models.py** | 106 | 122| 156 | 22766 | 90104 | 
| 63 | 16 django/db/models/base.py | 1966 | 2124| 1178 | 23944 | 90104 | 
| 64 | 16 django/db/backends/base/schema.py | 1246 | 1287| 357 | 24301 | 90104 | 
| 65 | 16 django/db/backends/base/schema.py | 621 | 683| 700 | 25001 | 90104 | 
| 66 | 16 django/db/backends/base/schema.py | 431 | 450| 199 | 25200 | 90104 | 
| 67 | 17 django/db/models/constraints.py | 198 | 216| 234 | 25434 | 92165 | 
| 68 | 17 django/db/backends/oracle/operations.py | 373 | 410| 369 | 25803 | 92165 | 
| 69 | 17 django/db/backends/base/schema.py | 968 | 987| 296 | 26099 | 92165 | 
| 70 | 17 django/db/backends/oracle/schema.py | 1 | 44| 454 | 26553 | 92165 | 
| 71 | 17 django/db/backends/base/schema.py | 1172 | 1187| 170 | 26723 | 92165 | 
| 72 | 18 django/db/models/sql/query.py | 745 | 776| 296 | 27019 | 114501 | 
| 73 | 18 django/db/backends/sqlite3/schema.py | 421 | 445| 162 | 27181 | 114501 | 
| 74 | 18 django/db/backends/oracle/operations.py | 585 | 600| 221 | 27402 | 114501 | 
| 75 | 18 django/db/backends/base/schema.py | 1310 | 1332| 173 | 27575 | 114501 | 
| 76 | 18 django/db/models/base.py | 1497 | 1520| 176 | 27751 | 114501 | 
| 77 | 19 django/db/backends/sqlite3/operations.py | 180 | 205| 190 | 27941 | 117773 | 
| 78 | 19 django/db/backends/postgresql/schema.py | 212 | 225| 182 | 28123 | 117773 | 
| 79 | 19 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 28440 | 117773 | 
| 80 | **19 django/db/migrations/operations/models.py** | 580 | 596| 215 | 28655 | 117773 | 
| 81 | 19 django/db/backends/oracle/operations.py | 465 | 479| 203 | 28858 | 117773 | 
| 82 | 20 django/db/backends/oracle/creation.py | 130 | 165| 399 | 29257 | 121666 | 
| 83 | 20 django/db/backends/mysql/schema.py | 96 | 106| 138 | 29395 | 121666 | 
| 84 | 20 django/db/models/base.py | 1269 | 1300| 267 | 29662 | 121666 | 
| 85 | 21 django/db/backends/ddl_references.py | 204 | 233| 247 | 29909 | 123265 | 
| 86 | 21 django/db/backends/base/schema.py | 1 | 29| 209 | 30118 | 123265 | 
| 87 | 21 django/db/migrations/autodetector.py | 1215 | 1230| 179 | 30297 | 123265 | 
| 88 | 21 django/db/backends/base/schema.py | 51 | 119| 785 | 31082 | 123265 | 
| 89 | 21 django/db/backends/ddl_references.py | 132 | 163| 283 | 31365 | 123265 | 
| 90 | 22 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 31486 | 124121 | 
| 91 | 23 django/db/models/sql/datastructures.py | 103 | 114| 133 | 31619 | 125571 | 
| 92 | 23 django/db/backends/mysql/schema.py | 124 | 140| 205 | 31824 | 125571 | 
| 93 | 23 django/db/backends/base/schema.py | 1213 | 1244| 233 | 32057 | 125571 | 
| 94 | 23 django/db/migrations/questioner.py | 172 | 196| 235 | 32292 | 125571 | 
| 95 | 23 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 32824 | 125571 | 
| 96 | 23 django/db/models/base.py | 1440 | 1495| 491 | 33315 | 125571 | 
| 97 | 23 django/db/backends/base/schema.py | 491 | 546| 613 | 33928 | 125571 | 
| 98 | 23 django/db/models/sql/datastructures.py | 149 | 186| 266 | 34194 | 125571 | 
| 99 | 23 django/db/backends/postgresql/schema.py | 227 | 239| 152 | 34346 | 125571 | 
| 100 | 23 django/db/models/sql/query.py | 840 | 877| 383 | 34729 | 125571 | 
| 101 | 23 django/contrib/gis/db/backends/spatialite/schema.py | 84 | 102| 153 | 34882 | 125571 | 
| 102 | 23 django/db/models/base.py | 212 | 322| 866 | 35748 | 125571 | 
| 103 | 23 django/db/backends/sqlite3/operations.py | 228 | 244| 142 | 35890 | 125571 | 
| 104 | 23 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 36516 | 125571 | 
| 105 | 24 django/db/backends/postgresql/creation.py | 1 | 37| 261 | 36777 | 126240 | 
| 106 | 24 django/db/backends/base/schema.py | 989 | 1016| 327 | 37104 | 126240 | 
| 107 | 25 django/db/backends/mysql/operations.py | 195 | 220| 215 | 37319 | 129967 | 
| 108 | 25 django/db/backends/oracle/creation.py | 253 | 281| 277 | 37596 | 129967 | 
| 109 | 25 django/db/models/constraints.py | 187 | 196| 130 | 37726 | 129967 | 
| 110 | 26 django/contrib/postgres/constraints.py | 107 | 140| 250 | 37976 | 131489 | 
| 111 | 27 django/db/backends/mysql/base.py | 296 | 334| 402 | 38378 | 134938 | 
| 112 | 27 django/db/models/base.py | 813 | 876| 644 | 39022 | 134938 | 
| 113 | **27 django/db/migrations/operations/models.py** | 1 | 38| 235 | 39257 | 134938 | 
| 114 | 28 django/db/models/sql/compiler.py | 1462 | 1494| 254 | 39511 | 149730 | 
| 115 | 28 django/db/migrations/questioner.py | 56 | 86| 255 | 39766 | 149730 | 
| 116 | 28 django/db/backends/mysql/schema.py | 108 | 122| 143 | 39909 | 149730 | 
| 117 | 28 django/db/backends/sqlite3/operations.py | 207 | 226| 209 | 40118 | 149730 | 
| 118 | **28 django/db/migrations/operations/models.py** | 598 | 615| 163 | 40281 | 149730 | 
| 119 | 29 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 40476 | 149925 | 
| 120 | 29 django/db/backends/oracle/operations.py | 21 | 73| 574 | 41050 | 149925 | 
| 121 | 29 django/db/backends/postgresql/creation.py | 56 | 81| 247 | 41297 | 149925 | 
| 122 | 29 django/db/models/sql/query.py | 987 | 1018| 307 | 41604 | 149925 | 
| 123 | 29 django/db/models/base.py | 1178 | 1206| 213 | 41817 | 149925 | 
| 124 | 29 django/db/models/base.py | 1087 | 1130| 404 | 42221 | 149925 | 
| 125 | 30 django/contrib/contenttypes/management/__init__.py | 1 | 43| 357 | 42578 | 150900 | 
| 126 | 30 django/db/models/base.py | 1302 | 1331| 244 | 42822 | 150900 | 
| 127 | 30 django/db/backends/base/schema.py | 381 | 413| 246 | 43068 | 150900 | 
| 128 | **30 django/db/migrations/operations/models.py** | 772 | 812| 344 | 43412 | 150900 | 
| 129 | 30 django/db/models/base.py | 1208 | 1242| 230 | 43642 | 150900 | 
| 130 | 30 django/db/backends/oracle/operations.py | 337 | 348| 226 | 43868 | 150900 | 
| 131 | 31 django/db/models/options.py | 365 | 388| 198 | 44066 | 158267 | 
| 132 | 31 django/db/backends/base/schema.py | 213 | 259| 445 | 44511 | 158267 | 
| 133 | 31 django/contrib/postgres/constraints.py | 92 | 105| 179 | 44690 | 158267 | 
| 134 | 31 django/db/backends/base/schema.py | 876 | 898| 185 | 44875 | 158267 | 
| 135 | 31 django/db/models/base.py | 999 | 1027| 230 | 45105 | 158267 | 
| 136 | 31 django/db/backends/sqlite3/schema.py | 67 | 84| 196 | 45301 | 158267 | 
| 137 | 31 django/db/models/base.py | 169 | 211| 413 | 45714 | 158267 | 
| 138 | 31 django/db/backends/base/schema.py | 578 | 619| 489 | 46203 | 158267 | 
| 139 | **31 django/db/migrations/operations/models.py** | 500 | 509| 129 | 46332 | 158267 | 
| 140 | 31 django/db/backends/ddl_references.py | 1 | 39| 218 | 46550 | 158267 | 
| 141 | 32 django/contrib/gis/db/backends/oracle/schema.py | 60 | 95| 297 | 46847 | 159099 | 
| 142 | **32 django/db/migrations/operations/models.py** | 815 | 846| 273 | 47120 | 159099 | 
| 143 | 33 django/db/models/indexes.py | 142 | 170| 326 | 47446 | 161422 | 
| 144 | 33 django/db/migrations/operations/fields.py | 142 | 181| 344 | 47790 | 161422 | 
| 145 | 33 django/db/models/base.py | 404 | 509| 913 | 48703 | 161422 | 
| 146 | 33 django/db/models/sql/query.py | 1699 | 1741| 436 | 49139 | 161422 | 
| 147 | **33 django/db/migrations/operations/models.py** | 511 | 528| 168 | 49307 | 161422 | 
| 148 | 33 django/db/backends/mysql/operations.py | 222 | 279| 437 | 49744 | 161422 | 
| 149 | 34 django/db/backends/mysql/creation.py | 1 | 30| 221 | 49965 | 162061 | 
| 150 | 34 django/contrib/gis/db/backends/spatialite/schema.py | 63 | 82| 133 | 50098 | 162061 | 
| 151 | 35 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 50576 | 164694 | 
| 152 | 36 django/forms/models.py | 389 | 417| 240 | 50816 | 176612 | 
| 153 | 36 django/contrib/postgres/operations.py | 239 | 259| 163 | 50979 | 176612 | 
| 154 | 36 django/db/models/base.py | 1723 | 1771| 348 | 51327 | 176612 | 
| 155 | 36 django/db/backends/postgresql/creation.py | 39 | 54| 173 | 51500 | 176612 | 
| 156 | 37 django/db/backends/sqlite3/features.py | 1 | 127| 1163 | 52663 | 177775 | 
| 157 | 38 django/db/backends/base/creation.py | 324 | 343| 121 | 52784 | 180563 | 
| 158 | 39 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 52974 | 181184 | 
| 159 | 39 django/db/backends/oracle/creation.py | 30 | 100| 722 | 53696 | 181184 | 
| 160 | 39 django/db/backends/ddl_references.py | 42 | 74| 208 | 53904 | 181184 | 
| 161 | 39 django/db/backends/postgresql/operations.py | 189 | 276| 696 | 54600 | 181184 | 
| 162 | 39 django/db/models/base.py | 2127 | 2178| 351 | 54951 | 181184 | 
| 163 | 39 django/contrib/postgres/operations.py | 37 | 61| 197 | 55148 | 181184 | 
| 164 | 39 django/db/backends/mysql/creation.py | 32 | 56| 253 | 55401 | 181184 | 
| 165 | 39 django/db/migrations/state.py | 117 | 131| 163 | 55564 | 181184 | 
| 166 | 40 django/db/models/sql/__init__.py | 1 | 7| 0 | 55564 | 181244 | 
| 167 | 40 django/db/backends/base/schema.py | 32 | 48| 166 | 55730 | 181244 | 
| 168 | 41 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 55969 | 182095 | 
| 169 | 41 django/db/models/sql/query.py | 365 | 415| 494 | 56463 | 182095 | 
| 170 | 41 django/db/models/base.py | 1333 | 1358| 184 | 56647 | 182095 | 
| 171 | 41 django/db/backends/mysql/base.py | 258 | 294| 259 | 56906 | 182095 | 
| 172 | 42 django/db/migrations/operations/base.py | 111 | 141| 229 | 57135 | 183125 | 
| 173 | 42 django/db/migrations/autodetector.py | 1064 | 1080| 188 | 57323 | 183125 | 
| 174 | 43 django/db/models/fields/related.py | 866 | 887| 169 | 57492 | 197121 | 
| 175 | 43 django/db/backends/oracle/operations.py | 350 | 371| 228 | 57720 | 197121 | 
| 176 | 43 django/core/management/commands/inspectdb.py | 38 | 173| 1291 | 59011 | 197121 | 


## Patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -320,12 +320,13 @@ def database_forwards(self, app_label, schema_editor, from_state, to_state):
         new_model = to_state.apps.get_model(app_label, self.new_name)
         if self.allow_migrate_model(schema_editor.connection.alias, new_model):
             old_model = from_state.apps.get_model(app_label, self.old_name)
+            old_db_table = old_model._meta.db_table
+            new_db_table = new_model._meta.db_table
+            # Don't alter when a table name is not changed.
+            if old_db_table == new_db_table:
+                return
             # Move the main table
-            schema_editor.alter_db_table(
-                new_model,
-                old_model._meta.db_table,
-                new_model._meta.db_table,
-            )
+            schema_editor.alter_db_table(new_model, old_db_table, new_db_table)
             # Alter the fields pointing to us
             for related_object in old_model._meta.related_objects:
                 if related_object.related_model == old_model:

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -793,6 +793,28 @@ def test_rename_model_with_m2m(self):
         self.assertEqual(Rider.objects.count(), 2)
         self.assertEqual(Pony._meta.get_field('riders').remote_field.through.objects.count(), 2)
 
+    def test_rename_model_with_db_table_noop(self):
+        app_label = 'test_rmwdbtn'
+        project_state = self.apply_operations(app_label, ProjectState(), operations=[
+            migrations.CreateModel('Rider', fields=[
+                ('id', models.AutoField(primary_key=True)),
+            ], options={'db_table': 'rider'}),
+            migrations.CreateModel('Pony', fields=[
+                ('id', models.AutoField(primary_key=True)),
+                ('rider', models.ForeignKey('%s.Rider' % app_label, models.CASCADE)),
+            ]),
+        ])
+        new_state = project_state.clone()
+        operation = migrations.RenameModel('Rider', 'Runner')
+        operation.state_forwards(app_label, new_state)
+
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_forwards(app_label, editor, project_state, new_state)
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_backwards(app_label, editor, new_state, project_state)
+
     def test_rename_m2m_target_model(self):
         app_label = "test_rename_m2m_target_model"
         project_state = self.apply_operations(app_label, ProjectState(), operations=[

```


## Code snippets

### 1 - django/db/migrations/operations/models.py:

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
### 2 - django/db/backends/base/schema.py:

Start line: 468, End line: 489

```python
class BaseDatabaseSchemaEditor:

    def alter_db_table(self, model, old_db_table, new_db_table):
        """Rename the table a model points to."""
        if (old_db_table == new_db_table or
            (self.connection.features.ignores_table_name_case and
                old_db_table.lower() == new_db_table.lower())):
            return
        self.execute(self.sql_rename_table % {
            "old_table": self.quote_name(old_db_table),
            "new_table": self.quote_name(new_db_table),
        })
        # Rename all references to the old table name.
        for sql in self.deferred_sql:
            if isinstance(sql, Statement):
                sql.rename_table_references(old_db_table, new_db_table)

    def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
        """Move a model's table between tablespaces."""
        self.execute(self.sql_retablespace_table % {
            "table": self.quote_name(model._meta.db_table),
            "old_tablespace": self.quote_name(old_db_tablespace),
            "new_tablespace": self.quote_name(new_db_tablespace),
        })
```
### 3 - django/db/migrations/operations/models.py:

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
### 4 - django/db/migrations/operations/models.py:

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
### 5 - django/db/backends/sqlite3/schema.py:

Start line: 142, End line: 223

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(self, model, create_field=None, delete_field=None, alter_field=None):
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
        mapping = {f.column: self.quote_name(f.column) for f in model._meta.local_concrete_fields}
        # This maps field names (not columns) for things like unique_together
        rename_mapping = {}
        # If any of the new or altered fields is introducing a new PK,
        # remove the old one
        restore_pk_field = None
        if getattr(create_field, 'primary_key', False) or (
                alter_field and getattr(alter_field[1], 'primary_key', False)):
            for name, field in list(body.items()):
                if field.primary_key:
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
                mapping[create_field.column] = self.quote_value(
                    self.effective_default(create_field)
                )
        # Add in any altered fields
        if alter_field:
            old_field, new_field = alter_field
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            if old_field.null and not new_field.null:
                case_sql = "coalesce(%(col)s, %(default)s)" % {
                    'col': self.quote_name(old_field.column),
                    'default': self.quote_value(self.effective_default(new_field))
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
            if delete_field.many_to_many and delete_field.remote_field.through._meta.auto_created:
                return self.delete_model(delete_field.remote_field.through)
        # Work inside a new app registry
        apps = Apps()

        # Work out the new value of unique_together, taking renames into
        # account
        unique_together = [
            [rename_mapping.get(n, n) for n in unique]
            for unique in model._meta.unique_together
        ]
        # ... other code
```
### 6 - django/db/backends/sqlite3/schema.py:

Start line: 86, End line: 99

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_db_table(self, model, old_db_table, new_db_table, disable_constraints=True):
        if (not self.connection.features.supports_atomic_references_rename and
                disable_constraints and self._is_referenced_by_fk_constraint(old_db_table)):
            if self.connection.in_atomic_block:
                raise NotSupportedError((
                    'Renaming the %r table while in a transaction is not '
                    'supported on SQLite < 3.26 because it would break referential '
                    'integrity. Try adding `atomic = False` to the Migration class.'
                ) % old_db_table)
            self.connection.enable_constraint_checking()
            super().alter_db_table(model, old_db_table, new_db_table)
            self.connection.disable_constraint_checking()
        else:
            super().alter_db_table(model, old_db_table, new_db_table)
```
### 7 - django/db/backends/sqlite3/schema.py:

Start line: 225, End line: 307

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(self, model, create_field=None, delete_field=None, alter_field=None):

        # Work out the new value for index_together, taking renames into
        # account
        index_together = [
            [rename_mapping.get(n, n) for n in index]
            for index in model._meta.index_together
        ]

        indexes = model._meta.indexes
        if delete_field:
            indexes = [
                index for index in indexes
                if delete_field.name not in index.fields
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
            'app_label': model._meta.app_label,
            'db_table': model._meta.db_table,
            'unique_together': unique_together,
            'index_together': index_together,
            'indexes': indexes,
            'constraints': constraints,
            'apps': apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy['Meta'] = meta
        body_copy['__module__'] = model.__module__
        type(model._meta.object_name, model.__bases__, body_copy)

        # Construct a model with a renamed table name.
        body_copy = copy.deepcopy(body)
        meta_contents = {
            'app_label': model._meta.app_label,
            'db_table': 'new__%s' % strip_quotes(model._meta.db_table),
            'unique_together': unique_together,
            'index_together': index_together,
            'indexes': indexes,
            'constraints': constraints,
            'apps': apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy['Meta'] = meta
        body_copy['__module__'] = model.__module__
        new_model = type('New%s' % model._meta.object_name, model.__bases__, body_copy)

        # Create a new table with the updated schema.
        self.create_model(new_model)

        # Copy data from the old table into the new table
        self.execute("INSERT INTO %s (%s) SELECT %s FROM %s" % (
            self.quote_name(new_model._meta.db_table),
            ', '.join(self.quote_name(x) for x in mapping),
            ', '.join(mapping.values()),
            self.quote_name(model._meta.db_table),
        ))

        # Delete the old table to make way for the new
        self.delete_model(model, handle_autom2m=False)

        # Rename the new table to take way for the old
        self.alter_db_table(
            new_model, new_model._meta.db_table, model._meta.db_table,
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
### 8 - django/db/migrations/operations/models.py:

Start line: 289, End line: 317

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
            'old_name': self.old_name,
            'new_name': self.new_name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.rename_model(app_label, self.old_name, self.new_name)
```
### 9 - django/db/migrations/operations/fields.py:

Start line: 258, End line: 324

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
### 10 - django/db/migrations/operations/models.py:

Start line: 250, End line: 286

```python
class DeleteModel(ModelOperation):
    """Drop a model's table."""

    def deconstruct(self):
        kwargs = {
            'name': self.name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

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
        return 'delete_%s' % self.name_lower
```
### 11 - django/db/migrations/operations/models.py:

Start line: 392, End line: 413

```python
class RenameModel(ModelOperation):

    def reduce(self, operation, app_label):
        if (isinstance(operation, RenameModel) and
                self.new_name_lower == operation.old_name_lower):
            return [
                RenameModel(
                    self.old_name,
                    operation.new_name,
                ),
            ]
        # Skip `ModelOperation.reduce` as we want to run `references_model`
        # against self.new_name.
        return (
            super(ModelOperation, self).reduce(operation, app_label) or
            not operation.references_model(self.new_name, app_label)
        )


class ModelOptionOperation(ModelOperation):
    def reduce(self, operation, app_label):
        if isinstance(operation, (self.__class__, DeleteModel)) and self.name_lower == operation.name_lower:
            return [operation]
        return super().reduce(operation, app_label)
```
### 15 - django/db/migrations/operations/models.py:

Start line: 41, End line: 104

```python
class CreateModel(ModelOperation):
    """Create a model's table."""

    serialization_expand_args = ['fields', 'options', 'managers']

    def __init__(self, name, fields, options=None, bases=None, managers=None):
        self.fields = fields
        self.options = options or {}
        self.bases = bases or (models.Model,)
        self.managers = managers or []
        super().__init__(name)
        # Sanity-check that there are no duplicated field names, bases, or
        # manager names
        _check_for_duplicates('fields', (name for name, _ in self.fields))
        _check_for_duplicates('bases', (
            base._meta.label_lower if hasattr(base, '_meta') else
            base.lower() if isinstance(base, str) else base
            for base in self.bases
        ))
        _check_for_duplicates('managers', (name for name, _ in self.managers))

    def deconstruct(self):
        kwargs = {
            'name': self.name,
            'fields': self.fields,
        }
        if self.options:
            kwargs['options'] = self.options
        if self.bases and self.bases != (models.Model,):
            kwargs['bases'] = self.bases
        if self.managers and self.managers != [('objects', models.Manager())]:
            kwargs['managers'] = self.managers
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.add_model(ModelState(
            app_label,
            self.name,
            list(self.fields),
            dict(self.options),
            tuple(self.bases),
            list(self.managers),
        ))

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def describe(self):
        return "Create %smodel %s" % ("proxy " if self.options.get("proxy", False) else "", self.name)

    @property
    def migration_name_fragment(self):
        return self.name_lower
```
### 39 - django/db/migrations/operations/models.py:

Start line: 124, End line: 247

```python
class CreateModel(ModelOperation):

    def reduce(self, operation, app_label):
        if (isinstance(operation, DeleteModel) and
                self.name_lower == operation.name_lower and
                not self.options.get("proxy", False)):
            return []
        elif isinstance(operation, RenameModel) and self.name_lower == operation.old_name_lower:
            return [
                CreateModel(
                    operation.new_name,
                    fields=self.fields,
                    options=self.options,
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif isinstance(operation, AlterModelOptions) and self.name_lower == operation.name_lower:
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
        elif isinstance(operation, AlterTogetherOptionOperation) and self.name_lower == operation.name_lower:
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options={**self.options, **{operation.option_name: operation.option_value}},
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif isinstance(operation, AlterOrderWithRespectTo) and self.name_lower == operation.name_lower:
            return [
                CreateModel(
                    self.name,
                    fields=self.fields,
                    options={**self.options, 'order_with_respect_to': operation.order_with_respect_to},
                    bases=self.bases,
                    managers=self.managers,
                ),
            ]
        elif isinstance(operation, FieldOperation) and self.name_lower == operation.model_name_lower:
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
                for option_name in ('unique_together', 'index_together'):
                    option = options.pop(option_name, None)
                    if option:
                        option = set(filter(bool, (
                            tuple(f for f in fields if f != operation.name_lower) for fields in option
                        )))
                        if option:
                            options[option_name] = option
                order_with_respect_to = options.get('order_with_respect_to')
                if order_with_respect_to == operation.name_lower:
                    del options['order_with_respect_to']
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
                for option_name in ('unique_together', 'index_together'):
                    option = options.get(option_name)
                    if option:
                        options[option_name] = {
                            tuple(operation.new_name if f == operation.old_name else f for f in fields)
                            for fields in option
                        }
                order_with_respect_to = options.get('order_with_respect_to')
                if order_with_respect_to == operation.old_name:
                    options['order_with_respect_to'] = operation.new_name
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
### 47 - django/db/migrations/operations/models.py:

Start line: 849, End line: 885

```python
class RemoveConstraint(IndexOperation):
    option_name = 'constraints'

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
        return self.__class__.__name__, [], {
            'model_name': self.model_name,
            'name': self.name,
        }

    def describe(self):
        return 'Remove constraint %s from model %s' % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return 'remove_%s_%s' % (self.model_name_lower, self.name.lower())
```
### 55 - django/db/migrations/operations/models.py:

Start line: 618, End line: 674

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
            'name': self.name,
            'options': self.options,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

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
        return 'alter_%s_options' % self.name_lower
```
### 62 - django/db/migrations/operations/models.py:

Start line: 106, End line: 122

```python
class CreateModel(ModelOperation):

    def references_model(self, name, app_label):
        name_lower = name.lower()
        if name_lower == self.name_lower:
            return True

        # Check we didn't inherit from the model
        reference_model_tuple = (app_label, name_lower)
        for base in self.bases:
            if (base is not models.Model and isinstance(base, (models.base.ModelBase, str)) and
                    resolve_relation(base, app_label) == reference_model_tuple):
                return True

        # Check we have no FKs/M2Ms with it
        for _name, field in self.fields:
            if field_references((app_label, self.name_lower), field, reference_model_tuple):
                return True
        return False
```
### 80 - django/db/migrations/operations/models.py:

Start line: 580, End line: 596

```python
class AlterOrderWithRespectTo(ModelOptionOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.name)
            # Remove a field if we need to
            if from_model._meta.order_with_respect_to and not to_model._meta.order_with_respect_to:
                schema_editor.remove_field(from_model, from_model._meta.get_field("_order"))
            # Add a field if we need to (altering the column is untouched as
            # it's likely a rename)
            elif to_model._meta.order_with_respect_to and not from_model._meta.order_with_respect_to:
                field = to_model._meta.get_field("_order")
                if not field.has_default():
                    field.default = 0
                schema_editor.add_field(
                    from_model,
                    field,
                )
```
### 113 - django/db/migrations/operations/models.py:

Start line: 1, End line: 38

```python
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property

from .fields import (
    AddField, AlterField, FieldOperation, RemoveField, RenameField,
)


def _check_for_duplicates(arg_name, objs):
    used_vals = set()
    for val in objs:
        if val in used_vals:
            raise ValueError(
                "Found duplicate value %s in CreateModel %s argument." % (val, arg_name)
            )
        used_vals.add(val)


class ModelOperation(Operation):
    def __init__(self, name):
        self.name = name

    @cached_property
    def name_lower(self):
        return self.name.lower()

    def references_model(self, name, app_label):
        return name.lower() == self.name_lower

    def reduce(self, operation, app_label):
        return (
            super().reduce(operation, app_label) or
            not operation.references_model(self.name, app_label)
        )
```
### 118 - django/db/migrations/operations/models.py:

Start line: 598, End line: 615

```python
class AlterOrderWithRespectTo(ModelOptionOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.database_forwards(app_label, schema_editor, from_state, to_state)

    def references_field(self, model_name, name, app_label):
        return (
            self.references_model(model_name, app_label) and
            (
                self.order_with_respect_to is None or
                name == self.order_with_respect_to
            )
        )

    def describe(self):
        return "Set order_with_respect_to on %s to %s" % (self.name, self.order_with_respect_to)

    @property
    def migration_name_fragment(self):
        return 'alter_%s_order_with_respect_to' % self.name_lower
```
### 128 - django/db/migrations/operations/models.py:

Start line: 772, End line: 812

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
            'model_name': self.model_name,
            'name': self.name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs,
        )

    def describe(self):
        return 'Remove index %s from %s' % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return 'remove_%s_%s' % (self.model_name_lower, self.name.lower())
```
### 139 - django/db/migrations/operations/models.py:

Start line: 500, End line: 509

```python
class AlterTogetherOptionOperation(ModelOptionOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.name)
            alter_together = getattr(schema_editor, 'alter_%s' % self.option_name)
            alter_together(
                new_model,
                getattr(old_model._meta, self.option_name, set()),
                getattr(new_model._meta, self.option_name, set()),
            )
```
### 142 - django/db/migrations/operations/models.py:

Start line: 815, End line: 846

```python
class AddConstraint(IndexOperation):
    option_name = 'constraints'

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
        return self.__class__.__name__, [], {
            'model_name': self.model_name,
            'constraint': self.constraint,
        }

    def describe(self):
        return 'Create constraint %s on model %s' % (self.constraint.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return '%s_%s' % (self.model_name_lower, self.constraint.name.lower())
```
### 147 - django/db/migrations/operations/models.py:

Start line: 511, End line: 528

```python
class AlterTogetherOptionOperation(ModelOptionOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        return self.database_forwards(app_label, schema_editor, from_state, to_state)

    def references_field(self, model_name, name, app_label):
        return (
            self.references_model(model_name, app_label) and
            (
                not self.option_value or
                any((name in fields) for fields in self.option_value)
            )
        )

    def describe(self):
        return "Alter %s for %s (%s constraint(s))" % (self.option_name, self.name, len(self.option_value or ''))

    @property
    def migration_name_fragment(self):
        return 'alter_%s_%s' % (self.name_lower, self.option_name)
```
