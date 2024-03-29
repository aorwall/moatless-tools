# django__django-13237

| **django/django** | `3a6fa1d962ad9bd5678290bc22dd35bff13eb1f5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -1037,10 +1037,16 @@ def _field_indexes_sql(self, model, field):
         return output
 
     def _field_should_be_altered(self, old_field, new_field):
-        # Don't alter when changing only a field name.
+        _, old_path, old_args, old_kwargs = old_field.deconstruct()
+        _, new_path, new_args, new_kwargs = new_field.deconstruct()
+        # Don't alter when:
+        # - changing only a field name
+        # - adding only a db_column and the column name is not changed
+        old_kwargs.pop('db_column', None)
+        new_kwargs.pop('db_column', None)
         return (
             old_field.column != new_field.column or
-            old_field.deconstruct()[1:] != new_field.deconstruct()[1:]
+            (old_path, old_args, old_kwargs) != (new_path, new_args, new_kwargs)
         )
 
     def _field_should_be_indexed(self, model, field):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/schema.py | 1040 | 1043 | - | 1 | -


## Problem Statement

```
AlterField with db_column addition should be a noop.
Description
	 
		(last modified by Iuri de Silvio)
	 
When I change pink = models.Integer(default=0) to pink = models.Integer(default=0, db_column="pink") the migration drop/create the same constraints when it is an FK or even reconstruct the table (SQLite), but nothing really changed. The constraint drop/create is a blocking operation for PostgreSQL, so it is an undesirable and unexpected behavior.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/backends/base/schema.py** | 639 | 711| 796 | 796 | 11920 | 
| 2 | 2 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 1301 | 16055 | 
| 3 | 3 django/db/migrations/operations/fields.py | 216 | 234| 185 | 1486 | 19153 | 
| 4 | **3 django/db/backends/base/schema.py** | 576 | 638| 700 | 2186 | 19153 | 
| 5 | 4 django/db/backends/oracle/schema.py | 57 | 77| 249 | 2435 | 20908 | 
| 6 | 5 django/db/backends/postgresql/schema.py | 161 | 187| 351 | 2786 | 22940 | 
| 7 | **5 django/db/backends/base/schema.py** | 712 | 781| 740 | 3526 | 22940 | 
| 8 | 5 django/db/backends/oracle/schema.py | 79 | 123| 583 | 4109 | 22940 | 
| 9 | **5 django/db/backends/base/schema.py** | 782 | 822| 519 | 4628 | 22940 | 
| 10 | 5 django/db/migrations/operations/fields.py | 236 | 246| 146 | 4774 | 22940 | 
| 11 | 5 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 5196 | 22940 | 
| 12 | 6 django/db/migrations/questioner.py | 162 | 185| 246 | 5442 | 25013 | 
| 13 | **6 django/db/backends/base/schema.py** | 1 | 28| 198 | 5640 | 25013 | 
| 14 | 6 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 5821 | 25013 | 
| 15 | 7 django/db/migrations/autodetector.py | 913 | 994| 876 | 6697 | 36632 | 
| 16 | 7 django/db/migrations/operations/fields.py | 248 | 270| 188 | 6885 | 36632 | 
| 17 | 8 django/db/backends/mysql/schema.py | 88 | 98| 138 | 7023 | 38128 | 
| 18 | 8 django/db/backends/mysql/schema.py | 100 | 113| 148 | 7171 | 38128 | 
| 19 | 8 django/db/backends/mysql/schema.py | 115 | 129| 201 | 7372 | 38128 | 
| 20 | 9 django/db/migrations/operations/models.py | 597 | 613| 215 | 7587 | 45023 | 
| 21 | 9 django/db/migrations/operations/fields.py | 85 | 95| 124 | 7711 | 45023 | 
| 22 | 9 django/db/backends/postgresql/schema.py | 91 | 159| 539 | 8250 | 45023 | 
| 23 | 9 django/db/backends/mysql/schema.py | 1 | 37| 387 | 8637 | 45023 | 
| 24 | **9 django/db/backends/base/schema.py** | 1082 | 1104| 199 | 8836 | 45023 | 
| 25 | 9 django/db/backends/mysql/schema.py | 131 | 149| 192 | 9028 | 45023 | 
| 26 | **9 django/db/backends/base/schema.py** | 533 | 574| 489 | 9517 | 45023 | 
| 27 | 10 django/contrib/postgres/constraints.py | 92 | 105| 180 | 9697 | 46437 | 
| 28 | 10 django/db/backends/sqlite3/schema.py | 386 | 432| 444 | 10141 | 46437 | 
| 29 | 10 django/contrib/postgres/constraints.py | 107 | 126| 155 | 10296 | 46437 | 
| 30 | 10 django/db/migrations/questioner.py | 143 | 160| 183 | 10479 | 46437 | 
| 31 | 11 django/db/models/fields/__init__.py | 308 | 336| 205 | 10684 | 64126 | 
| 32 | 11 django/db/migrations/operations/fields.py | 97 | 109| 130 | 10814 | 64126 | 
| 33 | 11 django/db/migrations/operations/fields.py | 192 | 214| 147 | 10961 | 64126 | 
| 34 | **11 django/db/backends/base/schema.py** | 1248 | 1277| 268 | 11229 | 64126 | 
| 35 | 11 django/db/migrations/operations/fields.py | 273 | 299| 158 | 11387 | 64126 | 
| 36 | 11 django/db/migrations/operations/fields.py | 346 | 381| 335 | 11722 | 64126 | 
| 37 | 11 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 11940 | 64126 | 
| 38 | 12 django/db/backends/oracle/operations.py | 21 | 73| 574 | 12514 | 70067 | 
| 39 | 12 django/db/backends/oracle/schema.py | 1 | 39| 405 | 12919 | 70067 | 
| 40 | 12 django/db/migrations/operations/models.py | 833 | 866| 308 | 13227 | 70067 | 
| 41 | 13 django/db/models/constraints.py | 79 | 161| 729 | 13956 | 71682 | 
| 42 | 14 django/db/models/base.py | 1904 | 2035| 976 | 14932 | 88326 | 
| 43 | **14 django/db/backends/base/schema.py** | 407 | 421| 174 | 15106 | 88326 | 
| 44 | 14 django/db/migrations/operations/models.py | 869 | 908| 378 | 15484 | 88326 | 
| 45 | 14 django/db/migrations/questioner.py | 56 | 81| 220 | 15704 | 88326 | 
| 46 | 14 django/db/backends/oracle/operations.py | 370 | 407| 369 | 16073 | 88326 | 
| 47 | **14 django/db/backends/base/schema.py** | 446 | 501| 610 | 16683 | 88326 | 
| 48 | 15 django/db/models/fields/related.py | 841 | 862| 169 | 16852 | 102202 | 
| 49 | 16 django/contrib/gis/db/backends/spatialite/schema.py | 128 | 169| 376 | 17228 | 103554 | 
| 50 | 16 django/db/migrations/operations/models.py | 458 | 487| 302 | 17530 | 103554 | 
| 51 | 16 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 18156 | 103554 | 
| 52 | 16 django/db/backends/oracle/operations.py | 478 | 497| 240 | 18396 | 103554 | 
| 53 | 16 django/db/migrations/questioner.py | 227 | 240| 123 | 18519 | 103554 | 
| 54 | 16 django/db/migrations/operations/fields.py | 301 | 344| 410 | 18929 | 103554 | 
| 55 | 16 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 19660 | 103554 | 
| 56 | **16 django/db/backends/base/schema.py** | 1065 | 1080| 170 | 19830 | 103554 | 
| 57 | **16 django/db/backends/base/schema.py** | 278 | 299| 173 | 20003 | 103554 | 
| 58 | 16 django/db/backends/oracle/operations.py | 409 | 460| 516 | 20519 | 103554 | 
| 59 | 16 django/db/migrations/operations/fields.py | 111 | 121| 127 | 20646 | 103554 | 
| 60 | **16 django/db/backends/base/schema.py** | 386 | 405| 197 | 20843 | 103554 | 
| 61 | 16 django/db/models/fields/__init__.py | 2352 | 2401| 311 | 21154 | 103554 | 
| 62 | **16 django/db/backends/base/schema.py** | 898 | 917| 296 | 21450 | 103554 | 
| 63 | 16 django/db/models/fields/__init__.py | 1740 | 1767| 215 | 21665 | 103554 | 
| 64 | 16 django/db/backends/oracle/operations.py | 599 | 618| 303 | 21968 | 103554 | 
| 65 | 16 django/db/backends/oracle/operations.py | 344 | 368| 244 | 22212 | 103554 | 
| 66 | 16 django/db/backends/mysql/schema.py | 39 | 48| 134 | 22346 | 103554 | 
| 67 | 16 django/db/migrations/operations/models.py | 530 | 547| 168 | 22514 | 103554 | 
| 68 | 17 django/contrib/postgres/fields/ranges.py | 115 | 162| 262 | 22776 | 105646 | 
| 69 | 18 django/contrib/gis/db/backends/postgis/schema.py | 51 | 74| 195 | 22971 | 106311 | 
| 70 | 18 django/db/models/fields/related.py | 997 | 1024| 215 | 23186 | 106311 | 
| 71 | 19 django/db/models/fields/json.py | 368 | 378| 131 | 23317 | 110249 | 
| 72 | 19 django/db/backends/postgresql/schema.py | 189 | 202| 182 | 23499 | 110249 | 
| 73 | 19 django/contrib/postgres/constraints.py | 156 | 166| 132 | 23631 | 110249 | 
| 74 | 20 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 23826 | 110444 | 
| 75 | 20 django/db/models/fields/__init__.py | 1715 | 1738| 146 | 23972 | 110444 | 
| 76 | 21 django/db/models/lookups.py | 358 | 390| 294 | 24266 | 115397 | 
| 77 | 21 django/db/backends/postgresql/schema.py | 204 | 215| 140 | 24406 | 115397 | 
| 78 | 21 django/db/migrations/autodetector.py | 1036 | 1052| 188 | 24594 | 115397 | 
| 79 | 21 django/db/migrations/operations/fields.py | 1 | 37| 241 | 24835 | 115397 | 
| 80 | **21 django/db/backends/base/schema.py** | 255 | 276| 154 | 24989 | 115397 | 
| 81 | 21 django/db/backends/oracle/operations.py | 271 | 285| 206 | 25195 | 115397 | 
| 82 | **21 django/db/backends/base/schema.py** | 846 | 875| 238 | 25433 | 115397 | 
| 83 | **21 django/db/backends/base/schema.py** | 824 | 844| 191 | 25624 | 115397 | 
| 84 | **21 django/db/backends/base/schema.py** | 1106 | 1136| 214 | 25838 | 115397 | 
| 85 | 21 django/db/models/fields/__init__.py | 1045 | 1058| 104 | 25942 | 115397 | 
| 86 | **21 django/db/backends/base/schema.py** | 370 | 384| 182 | 26124 | 115397 | 
| 87 | 22 django/db/models/deletion.py | 1 | 76| 566 | 26690 | 119223 | 
| 88 | 22 django/contrib/postgres/fields/ranges.py | 43 | 91| 362 | 27052 | 119223 | 
| 89 | 23 django/db/backends/postgresql/operations.py | 1 | 27| 245 | 27297 | 121762 | 
| 90 | 23 django/db/backends/sqlite3/schema.py | 332 | 348| 173 | 27470 | 121762 | 
| 91 | 23 django/db/migrations/operations/models.py | 615 | 632| 163 | 27633 | 121762 | 
| 92 | 24 django/db/models/sql/compiler.py | 1535 | 1575| 409 | 28042 | 135958 | 
| 93 | 24 django/db/backends/oracle/operations.py | 205 | 253| 411 | 28453 | 135958 | 
| 94 | 24 django/db/models/fields/__init__.py | 1928 | 1955| 234 | 28687 | 135958 | 
| 95 | 24 django/db/models/fields/related.py | 750 | 768| 222 | 28909 | 135958 | 
| 96 | **24 django/db/backends/base/schema.py** | 503 | 531| 289 | 29198 | 135958 | 
| 97 | **24 django/db/backends/base/schema.py** | 1138 | 1173| 291 | 29489 | 135958 | 
| 98 | 24 django/db/models/lookups.py | 491 | 512| 172 | 29661 | 135958 | 
| 99 | **24 django/db/backends/base/schema.py** | 1175 | 1192| 142 | 29803 | 135958 | 
| 100 | 25 django/db/models/sql/query.py | 2335 | 2351| 177 | 29980 | 158448 | 
| 101 | 25 django/db/models/fields/__init__.py | 367 | 393| 199 | 30179 | 158448 | 
| 102 | 25 django/db/models/fields/related.py | 864 | 890| 240 | 30419 | 158448 | 
| 103 | 25 django/db/models/fields/__init__.py | 2432 | 2457| 143 | 30562 | 158448 | 
| 104 | 26 django/db/models/sql/where.py | 230 | 246| 130 | 30692 | 160252 | 
| 105 | 26 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 31009 | 160252 | 
| 106 | 26 django/db/backends/oracle/operations.py | 164 | 174| 206 | 31215 | 160252 | 
| 107 | 26 django/contrib/postgres/constraints.py | 1 | 67| 550 | 31765 | 160252 | 
| 108 | 26 django/db/models/base.py | 1401 | 1456| 491 | 32256 | 160252 | 
| 109 | 27 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 32734 | 162855 | 
| 110 | 27 django/db/migrations/operations/models.py | 519 | 528| 129 | 32863 | 162855 | 
| 111 | 27 django/db/models/fields/related.py | 913 | 933| 178 | 33041 | 162855 | 
| 112 | 28 django/db/backends/postgresql/base.py | 272 | 293| 156 | 33197 | 165667 | 
| 113 | 29 django/db/backends/base/features.py | 1 | 113| 899 | 34096 | 168391 | 
| 114 | 29 django/db/models/fields/related.py | 255 | 282| 269 | 34365 | 168391 | 
| 115 | 29 django/db/migrations/questioner.py | 207 | 224| 171 | 34536 | 168391 | 
| 116 | 29 django/contrib/postgres/constraints.py | 69 | 90| 201 | 34737 | 168391 | 
| 117 | 30 django/db/backends/sqlite3/operations.py | 312 | 357| 453 | 35190 | 171422 | 
| 118 | 30 django/db/backends/oracle/operations.py | 92 | 101| 201 | 35391 | 171422 | 
| 119 | 31 django/db/backends/mysql/compiler.py | 1 | 14| 123 | 35514 | 171924 | 
| 120 | 31 django/db/backends/oracle/operations.py | 255 | 269| 231 | 35745 | 171924 | 
| 121 | 32 django/db/backends/oracle/features.py | 1 | 81| 637 | 36382 | 172561 | 
| 122 | **32 django/db/backends/base/schema.py** | 877 | 896| 158 | 36540 | 172561 | 
| 123 | 32 django/db/models/constraints.py | 163 | 170| 124 | 36664 | 172561 | 
| 124 | 32 django/db/models/fields/related.py | 771 | 839| 521 | 37185 | 172561 | 
| 125 | 32 django/db/migrations/operations/fields.py | 123 | 143| 129 | 37314 | 172561 | 
| 126 | 32 django/db/backends/oracle/operations.py | 145 | 162| 309 | 37623 | 172561 | 
| 127 | 32 django/db/migrations/operations/fields.py | 383 | 400| 135 | 37758 | 172561 | 
| 128 | 32 django/db/models/fields/__init__.py | 1 | 81| 633 | 38391 | 172561 | 
| 129 | 33 django/db/backends/mysql/operations.py | 1 | 34| 273 | 38664 | 176224 | 
| 130 | 33 django/db/backends/oracle/operations.py | 116 | 130| 225 | 38889 | 176224 | 
| 131 | 33 django/db/models/sql/compiler.py | 1038 | 1078| 337 | 39226 | 176224 | 
| 132 | 33 django/db/backends/oracle/schema.py | 41 | 55| 133 | 39359 | 176224 | 
| 133 | 34 django/db/backends/sqlite3/introspection.py | 224 | 238| 146 | 39505 | 180073 | 
| 134 | 35 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 39654 | 180750 | 
| 135 | 35 django/db/models/fields/related.py | 284 | 318| 293 | 39947 | 180750 | 
| 136 | 35 django/db/models/fields/__init__.py | 1975 | 2011| 198 | 40145 | 180750 | 
| 137 | 35 django/db/models/constraints.py | 32 | 76| 372 | 40517 | 180750 | 
| 138 | 35 django/db/backends/oracle/operations.py | 582 | 597| 221 | 40738 | 180750 | 
| 139 | 35 django/db/models/fields/related.py | 984 | 995| 128 | 40866 | 180750 | 
| 140 | 35 django/db/backends/oracle/operations.py | 176 | 203| 319 | 41185 | 180750 | 
| 141 | 35 django/db/backends/oracle/operations.py | 543 | 562| 209 | 41394 | 180750 | 
| 142 | 35 django/db/backends/oracle/operations.py | 462 | 476| 203 | 41597 | 180750 | 
| 143 | 35 django/db/models/sql/query.py | 364 | 414| 494 | 42091 | 180750 | 
| 144 | 36 django/db/backends/mysql/base.py | 252 | 288| 259 | 42350 | 184180 | 
| 145 | 36 django/db/backends/sqlite3/operations.py | 1 | 38| 258 | 42608 | 184180 | 
| 146 | 36 django/db/backends/mysql/schema.py | 50 | 86| 349 | 42957 | 184180 | 
| 147 | 36 django/db/models/lookups.py | 305 | 355| 306 | 43263 | 184180 | 
| 148 | 36 django/db/migrations/operations/models.py | 339 | 388| 493 | 43756 | 184180 | 
| 149 | 36 django/db/migrations/autodetector.py | 892 | 911| 184 | 43940 | 184180 | 
| 150 | 37 django/db/backends/postgresql/introspection.py | 210 | 226| 179 | 44119 | 186428 | 
| 151 | 37 django/db/migrations/autodetector.py | 856 | 890| 339 | 44458 | 186428 | 
| 152 | 37 django/db/backends/postgresql/base.py | 65 | 132| 698 | 45156 | 186428 | 
| 153 | 37 django/db/backends/oracle/operations.py | 302 | 329| 271 | 45427 | 186428 | 
| 154 | 37 django/db/models/base.py | 1458 | 1481| 176 | 45603 | 186428 | 
| 155 | 38 django/contrib/gis/db/backends/oracle/schema.py | 34 | 58| 229 | 45832 | 187260 | 
| 156 | 38 django/db/models/fields/__init__.py | 2047 | 2077| 199 | 46031 | 187260 | 
| 157 | 39 django/db/backends/oracle/base.py | 102 | 155| 626 | 46657 | 192490 | 
| 158 | 39 django/db/models/sql/query.py | 703 | 738| 389 | 47046 | 192490 | 
| 159 | 40 django/db/backends/postgresql/features.py | 1 | 95| 758 | 47804 | 193248 | 
| 160 | 40 django/contrib/postgres/fields/ranges.py | 231 | 321| 479 | 48283 | 193248 | 
| 161 | 40 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 49103 | 193248 | 
| 162 | 40 django/db/backends/postgresql/introspection.py | 138 | 209| 751 | 49854 | 193248 | 
| 163 | 40 django/db/models/fields/related.py | 1202 | 1233| 180 | 50034 | 193248 | 
| 164 | 40 django/contrib/postgres/constraints.py | 128 | 154| 231 | 50265 | 193248 | 
| 165 | 41 django/db/backends/sqlite3/features.py | 1 | 80| 725 | 50990 | 193973 | 
| 166 | 41 django/db/backends/oracle/operations.py | 103 | 114| 212 | 51202 | 193973 | 
| 167 | 41 django/db/migrations/operations/fields.py | 64 | 83| 128 | 51330 | 193973 | 
| 168 | 41 django/db/migrations/operations/fields.py | 146 | 189| 394 | 51724 | 193973 | 
| 169 | 41 django/db/models/fields/__init__.py | 1769 | 1814| 279 | 52003 | 193973 | 
| 170 | 41 django/contrib/gis/db/backends/postgis/schema.py | 1 | 19| 206 | 52209 | 193973 | 
| 171 | 41 django/db/backends/postgresql/operations.py | 161 | 188| 311 | 52520 | 193973 | 
| 172 | 41 django/db/models/fields/related.py | 509 | 574| 492 | 53012 | 193973 | 
| 173 | 41 django/db/backends/sqlite3/operations.py | 228 | 257| 198 | 53210 | 193973 | 
| 174 | 41 django/db/models/constraints.py | 172 | 196| 188 | 53398 | 193973 | 


## Patch

```diff
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -1037,10 +1037,16 @@ def _field_indexes_sql(self, model, field):
         return output
 
     def _field_should_be_altered(self, old_field, new_field):
-        # Don't alter when changing only a field name.
+        _, old_path, old_args, old_kwargs = old_field.deconstruct()
+        _, new_path, new_args, new_kwargs = new_field.deconstruct()
+        # Don't alter when:
+        # - changing only a field name
+        # - adding only a db_column and the column name is not changed
+        old_kwargs.pop('db_column', None)
+        new_kwargs.pop('db_column', None)
         return (
             old_field.column != new_field.column or
-            old_field.deconstruct()[1:] != new_field.deconstruct()[1:]
+            (old_path, old_args, old_kwargs) != (new_path, new_args, new_kwargs)
         )
 
     def _field_should_be_indexed(self, model, field):

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -1354,6 +1354,59 @@ def test_alter_field(self):
         self.assertEqual(definition[1], [])
         self.assertEqual(sorted(definition[2]), ["field", "model_name", "name"])
 
+    def test_alter_field_add_db_column_noop(self):
+        """
+        AlterField operation is a noop when adding only a db_column and the
+        column name is not changed.
+        """
+        app_label = 'test_afadbn'
+        project_state = self.set_up_test_model(app_label, related_model=True)
+        pony_table = '%s_pony' % app_label
+        new_state = project_state.clone()
+        operation = migrations.AlterField('Pony', 'weight', models.FloatField(db_column='weight'))
+        operation.state_forwards(app_label, new_state)
+        self.assertIsNone(
+            project_state.models[app_label, 'pony'].fields['weight'].db_column,
+        )
+        self.assertEqual(
+            new_state.models[app_label, 'pony'].fields['weight'].db_column,
+            'weight',
+        )
+        self.assertColumnExists(pony_table, 'weight')
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertColumnExists(pony_table, 'weight')
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_backwards(app_label, editor, new_state, project_state)
+        self.assertColumnExists(pony_table, 'weight')
+
+        rider_table = '%s_rider' % app_label
+        new_state = project_state.clone()
+        operation = migrations.AlterField(
+            'Rider',
+            'pony',
+            models.ForeignKey('Pony', models.CASCADE, db_column='pony_id'),
+        )
+        operation.state_forwards(app_label, new_state)
+        self.assertIsNone(
+            project_state.models[app_label, 'rider'].fields['pony'].db_column,
+        )
+        self.assertIs(
+            new_state.models[app_label, 'rider'].fields['pony'].db_column,
+            'pony_id',
+        )
+        self.assertColumnExists(rider_table, 'pony_id')
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertColumnExists(rider_table, 'pony_id')
+        with connection.schema_editor() as editor:
+            with self.assertNumQueries(0):
+                operation.database_forwards(app_label, editor, new_state, project_state)
+        self.assertColumnExists(rider_table, 'pony_id')
+
     def test_alter_field_pk(self):
         """
         Tests the AlterField operation on primary keys (for things like PostgreSQL's SERIAL weirdness)

```


## Code snippets

### 1 - django/db/backends/base/schema.py:

Start line: 639, End line: 711

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        # ... other code
        if old_field.db_index and not old_field.unique and (not new_field.db_index or new_field.unique):
            # Find the index for this field
            meta_index_names = {index.name for index in model._meta.indexes}
            # Retrieve only BTREE indexes since this is what's created with
            # db_index=True.
            index_names = self._constraint_names(
                model, [old_field.column], index=True, type_=Index.suffix,
                exclude=meta_index_names,
            )
            for index_name in index_names:
                # The only way to check if an index was created with
                # db_index=True or with Index(['field'], name='foo')
                # is to look at its name (refs #28053).
                self.execute(self._delete_index_sql(model, index_name))
        # Change check constraints?
        if old_db_params['check'] != new_db_params['check'] and old_db_params['check']:
            meta_constraint_names = {constraint.name for constraint in model._meta.constraints}
            constraint_names = self._constraint_names(
                model, [old_field.column], check=True,
                exclude=meta_constraint_names,
            )
            if strict and len(constraint_names) != 1:
                raise ValueError("Found wrong number (%s) of check constraints for %s.%s" % (
                    len(constraint_names),
                    model._meta.db_table,
                    old_field.column,
                ))
            for constraint_name in constraint_names:
                self.execute(self._delete_check_sql(model, constraint_name))
        # Have they renamed the column?
        if old_field.column != new_field.column:
            self.execute(self._rename_field_sql(model._meta.db_table, old_field, new_field, new_type))
            # Rename all references to the renamed column.
            for sql in self.deferred_sql:
                if isinstance(sql, Statement):
                    sql.rename_column_references(model._meta.db_table, old_field.column, new_field.column)
        # Next, start accumulating actions to do
        actions = []
        null_actions = []
        post_actions = []
        # Type change?
        if old_type != new_type:
            fragment, other_actions = self._alter_column_type_sql(model, old_field, new_field, new_type)
            actions.append(fragment)
            post_actions.extend(other_actions)
        # When changing a column NULL constraint to NOT NULL with a given
        # default value, we need to perform 4 steps:
        #  1. Add a default for new incoming writes
        #  2. Update existing NULL rows with new default
        #  3. Replace NULL constraint with NOT NULL
        #  4. Drop the default again.
        # Default change?
        needs_database_default = False
        if old_field.null and not new_field.null:
            old_default = self.effective_default(old_field)
            new_default = self.effective_default(new_field)
            if (
                not self.skip_default(new_field) and
                old_default != new_default and
                new_default is not None
            ):
                needs_database_default = True
                actions.append(self._alter_column_default_sql(model, old_field, new_field))
        # Nullability change?
        if old_field.null != new_field.null:
            fragment = self._alter_column_null_sql(model, old_field, new_field)
            if fragment:
                null_actions.append(fragment)
        # Only if we have a default and there is a change from NULL to NOT NULL
        four_way_default_alteration = (
            new_field.has_default() and
            (old_field.null and not new_field.null)
        )
        # ... other code
```
### 2 - django/db/backends/sqlite3/schema.py:

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
### 3 - django/db/migrations/operations/fields.py:

Start line: 216, End line: 234

```python
class AlterField(FieldOperation):

    def state_forwards(self, app_label, state):
        if not self.preserve_default:
            field = self.field.clone()
            field.default = NOT_PROVIDED
        else:
            field = self.field
        model_state = state.models[app_label, self.model_name_lower]
        model_state.fields[self.name] = field
        # TODO: investigate if old relational fields must be reloaded or if it's
        # sufficient if the new field is (#27737).
        # Delay rendering of relationships if it's not a relational field and
        # not referenced by a foreign key.
        delay = (
            not field.is_relation and
            not field_is_referenced(
                state, (app_label, self.model_name_lower), (self.name, field),
            )
        )
        state.reload_model(app_label, self.model_name_lower, delay=delay)
```
### 4 - django/db/backends/base/schema.py:

Start line: 576, End line: 638

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        """Perform a "physical" (non-ManyToMany) field update."""
        # Drop any FK constraints, we'll remake them later
        fks_dropped = set()
        if (
            self.connection.features.supports_foreign_keys and
            old_field.remote_field and
            old_field.db_constraint
        ):
            fk_names = self._constraint_names(model, [old_field.column], foreign_key=True)
            if strict and len(fk_names) != 1:
                raise ValueError("Found wrong number (%s) of foreign key constraints for %s.%s" % (
                    len(fk_names),
                    model._meta.db_table,
                    old_field.column,
                ))
            for fk_name in fk_names:
                fks_dropped.add((old_field.column,))
                self.execute(self._delete_fk_sql(model, fk_name))
        # Has unique been removed?
        if old_field.unique and (not new_field.unique or self._field_became_primary_key(old_field, new_field)):
            # Find the unique constraint for this field
            meta_constraint_names = {constraint.name for constraint in model._meta.constraints}
            constraint_names = self._constraint_names(
                model, [old_field.column], unique=True, primary_key=False,
                exclude=meta_constraint_names,
            )
            if strict and len(constraint_names) != 1:
                raise ValueError("Found wrong number (%s) of unique constraints for %s.%s" % (
                    len(constraint_names),
                    model._meta.db_table,
                    old_field.column,
                ))
            for constraint_name in constraint_names:
                self.execute(self._delete_unique_sql(model, constraint_name))
        # Drop incoming FK constraints if the field is a primary key or unique,
        # which might be a to_field target, and things are going to change.
        drop_foreign_keys = (
            self.connection.features.supports_foreign_keys and (
                (old_field.primary_key and new_field.primary_key) or
                (old_field.unique and new_field.unique)
            ) and old_type != new_type
        )
        if drop_foreign_keys:
            # '_meta.related_field' also contains M2M reverse fields, these
            # will be filtered out
            for _old_rel, new_rel in _related_non_m2m_objects(old_field, new_field):
                rel_fk_names = self._constraint_names(
                    new_rel.related_model, [new_rel.field.column], foreign_key=True
                )
                for fk_name in rel_fk_names:
                    self.execute(self._delete_fk_sql(new_rel.related_model, fk_name))
        # Removed an index? (no strict check, as multiple indexes are possible)
        # Remove indexes if db_index switched to False or a unique constraint
        # will now be used in lieu of an index. The following lines from the
        # truth table show all True cases; the rest are False:
        #
        # old_field.db_index | old_field.unique | new_field.db_index | new_field.unique
        # ------------------------------------------------------------------------------
        # True               | False            | False              | False
        # True               | False            | False              | True
        # True               | False            | True               | True
        # ... other code
```
### 5 - django/db/backends/oracle/schema.py:

Start line: 57, End line: 77

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        try:
            super().alter_field(model, old_field, new_field, strict)
        except DatabaseError as e:
            description = str(e)
            # If we're changing type to an unsupported type we need a
            # SQLite-ish workaround
            if 'ORA-22858' in description or 'ORA-22859' in description:
                self._alter_field_type_workaround(model, old_field, new_field)
            # If an identity column is changing to a non-numeric type, drop the
            # identity first.
            elif 'ORA-30675' in description:
                self._drop_identity(model._meta.db_table, old_field.column)
                self.alter_field(model, old_field, new_field, strict)
            # If a primary key column is changing to an identity column, drop
            # the primary key first.
            elif 'ORA-30673' in description and old_field.primary_key:
                self._delete_primary_key(model, strict=True)
                self._alter_field_type_workaround(model, old_field, new_field)
            else:
                raise
```
### 6 - django/db/backends/postgresql/schema.py:

Start line: 161, End line: 187

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        # Drop indexes on varchar/text/citext columns that are changing to a
        # different type.
        if (old_field.db_index or old_field.unique) and (
            (old_type.startswith('varchar') and not new_type.startswith('varchar')) or
            (old_type.startswith('text') and not new_type.startswith('text')) or
            (old_type.startswith('citext') and not new_type.startswith('citext'))
        ):
            index_name = self._create_index_name(model._meta.db_table, [old_field.column], suffix='_like')
            self.execute(self._delete_index_sql(model, index_name))

        super()._alter_field(
            model, old_field, new_field, old_type, new_type, old_db_params,
            new_db_params, strict,
        )
        # Added an index? Create any PostgreSQL-specific indexes.
        if ((not (old_field.db_index or old_field.unique) and new_field.db_index) or
                (not old_field.unique and new_field.unique)):
            like_index_statement = self._create_like_index_sql(model, new_field)
            if like_index_statement is not None:
                self.execute(like_index_statement)

        # Removed an index? Drop any PostgreSQL-specific indexes.
        if old_field.unique and not (new_field.db_index or new_field.unique):
            index_to_remove = self._create_index_name(model._meta.db_table, [old_field.column], suffix='_like')
            self.execute(self._delete_index_sql(model, index_to_remove))
```
### 7 - django/db/backends/base/schema.py:

Start line: 712, End line: 781

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        # ... other code
        if actions or null_actions:
            if not four_way_default_alteration:
                # If we don't have to do a 4-way default alteration we can
                # directly run a (NOT) NULL alteration
                actions = actions + null_actions
            # Combine actions together if we can (e.g. postgres)
            if self.connection.features.supports_combined_alters and actions:
                sql, params = tuple(zip(*actions))
                actions = [(", ".join(sql), sum(params, []))]
            # Apply those actions
            for sql, params in actions:
                self.execute(
                    self.sql_alter_column % {
                        "table": self.quote_name(model._meta.db_table),
                        "changes": sql,
                    },
                    params,
                )
            if four_way_default_alteration:
                # Update existing rows with default value
                self.execute(
                    self.sql_update_with_default % {
                        "table": self.quote_name(model._meta.db_table),
                        "column": self.quote_name(new_field.column),
                        "default": "%s",
                    },
                    [new_default],
                )
                # Since we didn't run a NOT NULL change before we need to do it
                # now
                for sql, params in null_actions:
                    self.execute(
                        self.sql_alter_column % {
                            "table": self.quote_name(model._meta.db_table),
                            "changes": sql,
                        },
                        params,
                    )
        if post_actions:
            for sql, params in post_actions:
                self.execute(sql, params)
        # If primary_key changed to False, delete the primary key constraint.
        if old_field.primary_key and not new_field.primary_key:
            self._delete_primary_key(model, strict)
        # Added a unique?
        if self._unique_should_be_added(old_field, new_field):
            self.execute(self._create_unique_sql(model, [new_field.column]))
        # Added an index? Add an index if db_index switched to True or a unique
        # constraint will no longer be used in lieu of an index. The following
        # lines from the truth table show all True cases; the rest are False:
        #
        # old_field.db_index | old_field.unique | new_field.db_index | new_field.unique
        # ------------------------------------------------------------------------------
        # False              | False            | True               | False
        # False              | True             | True               | False
        # True               | True             | True               | False
        if (not old_field.db_index or old_field.unique) and new_field.db_index and not new_field.unique:
            self.execute(self._create_index_sql(model, [new_field]))
        # Type alteration on primary key? Then we need to alter the column
        # referring to us.
        rels_to_update = []
        if drop_foreign_keys:
            rels_to_update.extend(_related_non_m2m_objects(old_field, new_field))
        # Changed to become primary key?
        if self._field_became_primary_key(old_field, new_field):
            # Make the new one
            self.execute(self._create_primary_key_sql(model, new_field))
            # Update all referencing columns
            rels_to_update.extend(_related_non_m2m_objects(old_field, new_field))
        # Handle our type alters on the other end of rels from the PK stuff above
        # ... other code
```
### 8 - django/db/backends/oracle/schema.py:

Start line: 79, End line: 123

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field_type_workaround(self, model, old_field, new_field):
        """
        Oracle refuses to change from some type to other type.
        What we need to do instead is:
        - Add a nullable version of the desired field with a temporary name. If
          the new column is an auto field, then the temporary column can't be
          nullable.
        - Update the table to transfer values from old to new
        - Drop old column
        - Rename the new column and possibly drop the nullable property
        """
        # Make a new field that's like the new one but with a temporary
        # column name.
        new_temp_field = copy.deepcopy(new_field)
        new_temp_field.null = (new_field.get_internal_type() not in ('AutoField', 'BigAutoField', 'SmallAutoField'))
        new_temp_field.column = self._generate_temp_name(new_field.column)
        # Add it
        self.add_field(model, new_temp_field)
        # Explicit data type conversion
        # https://docs.oracle.com/en/database/oracle/oracle-database/18/sqlrf
        # /Data-Type-Comparison-Rules.html#GUID-D0C5A47E-6F93-4C2D-9E49-4F2B86B359DD
        new_value = self.quote_name(old_field.column)
        old_type = old_field.db_type(self.connection)
        if re.match('^N?CLOB', old_type):
            new_value = "TO_CHAR(%s)" % new_value
            old_type = 'VARCHAR2'
        if re.match('^N?VARCHAR2', old_type):
            new_internal_type = new_field.get_internal_type()
            if new_internal_type == 'DateField':
                new_value = "TO_DATE(%s, 'YYYY-MM-DD')" % new_value
            elif new_internal_type == 'DateTimeField':
                new_value = "TO_TIMESTAMP(%s, 'YYYY-MM-DD HH24:MI:SS.FF')" % new_value
            elif new_internal_type == 'TimeField':
                # TimeField are stored as TIMESTAMP with a 1900-01-01 date part.
                new_value = "TO_TIMESTAMP(CONCAT('1900-01-01 ', %s), 'YYYY-MM-DD HH24:MI:SS.FF')" % new_value
        # Transfer values across
        self.execute("UPDATE %s set %s=%s" % (
            self.quote_name(model._meta.db_table),
            self.quote_name(new_temp_field.column),
            new_value,
        ))
        # Drop the old field
        self.remove_field(model, old_field)
        # Rename and possibly make the new field NOT NULL
        super().alter_field(model, new_temp_field, new_field)
```
### 9 - django/db/backends/base/schema.py:

Start line: 782, End line: 822

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        # ... other code
        for old_rel, new_rel in rels_to_update:
            rel_db_params = new_rel.field.db_parameters(connection=self.connection)
            rel_type = rel_db_params['type']
            fragment, other_actions = self._alter_column_type_sql(
                new_rel.related_model, old_rel.field, new_rel.field, rel_type
            )
            self.execute(
                self.sql_alter_column % {
                    "table": self.quote_name(new_rel.related_model._meta.db_table),
                    "changes": fragment[0],
                },
                fragment[1],
            )
            for sql, params in other_actions:
                self.execute(sql, params)
        # Does it have a foreign key?
        if (self.connection.features.supports_foreign_keys and new_field.remote_field and
                (fks_dropped or not old_field.remote_field or not old_field.db_constraint) and
                new_field.db_constraint):
            self.execute(self._create_fk_sql(model, new_field, "_fk_%(to_table)s_%(to_column)s"))
        # Rebuild FKs that pointed to us if we previously had to drop them
        if drop_foreign_keys:
            for rel in new_field.model._meta.related_objects:
                if _is_relevant_relation(rel, new_field) and rel.field.db_constraint:
                    self.execute(self._create_fk_sql(rel.related_model, rel.field, "_fk"))
        # Does it have check constraints we need to add?
        if old_db_params['check'] != new_db_params['check'] and new_db_params['check']:
            constraint_name = self._create_index_name(model._meta.db_table, [new_field.column], suffix='_check')
            self.execute(self._create_check_sql(model, constraint_name, new_db_params['check']))
        # Drop the default if we need to
        # (Django usually does not use in-database defaults)
        if needs_database_default:
            changes_sql, params = self._alter_column_default_sql(model, old_field, new_field, drop=True)
            sql = self.sql_alter_column % {
                "table": self.quote_name(model._meta.db_table),
                "changes": changes_sql,
            }
            self.execute(sql, params)
        # Reset connection if required
        if self.connection.features.connection_persists_old_columns:
            self.connection.close()
```
### 10 - django/db/migrations/operations/fields.py:

Start line: 236, End line: 246

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
### 13 - django/db/backends/base/schema.py:

Start line: 1, End line: 28

```python
import logging
from datetime import datetime

from django.db.backends.ddl_references import (
    Columns, ForeignKeyName, IndexName, Statement, Table,
)
from django.db.backends.utils import names_digest, split_identifier
from django.db.models import Deferrable, Index
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone

logger = logging.getLogger('django.db.backends.schema')


def _is_relevant_relation(relation, altered_field):
    """
    When altering the given field, must constraints on its model from the given
    relation be temporarily dropped?
    """
    field = relation.field
    if field.many_to_many:
        # M2M reverse field
        return False
    if altered_field.primary_key and field.to_fields == [None]:
        # Foreign key constraint on the primary key, which is being altered.
        return True
    # Is the constraint targeting the field being altered?
    return altered_field.name in field.to_fields
```
### 24 - django/db/backends/base/schema.py:

Start line: 1082, End line: 1104

```python
class BaseDatabaseSchemaEditor:

    def _fk_constraint_name(self, model, field, suffix):
        def create_fk_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        return ForeignKeyName(
            model._meta.db_table,
            [field.column],
            split_identifier(field.target_field.model._meta.db_table)[1],
            [field.target_field.column],
            suffix,
            create_fk_name,
        )

    def _delete_fk_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_fk, model, name)

    def _deferrable_constraint_sql(self, deferrable):
        if deferrable is None:
            return ''
        if deferrable == Deferrable.DEFERRED:
            return ' DEFERRABLE INITIALLY DEFERRED'
        if deferrable == Deferrable.IMMEDIATE:
            return ' DEFERRABLE INITIALLY IMMEDIATE'
```
### 26 - django/db/backends/base/schema.py:

Start line: 533, End line: 574

```python
class BaseDatabaseSchemaEditor:

    def alter_field(self, model, old_field, new_field, strict=False):
        """
        Allow a field's type, uniqueness, nullability, default, column,
        constraints, etc. to be modified.
        `old_field` is required to compute the necessary changes.
        If `strict` is True, raise errors if the old column does not match
        `old_field` precisely.
        """
        if not self._field_should_be_altered(old_field, new_field):
            return
        # Ensure this field is even column-based
        old_db_params = old_field.db_parameters(connection=self.connection)
        old_type = old_db_params['type']
        new_db_params = new_field.db_parameters(connection=self.connection)
        new_type = new_db_params['type']
        if ((old_type is None and old_field.remote_field is None) or
                (new_type is None and new_field.remote_field is None)):
            raise ValueError(
                "Cannot alter field %s into %s - they do not properly define "
                "db_type (are you using a badly-written custom field?)" %
                (old_field, new_field),
            )
        elif old_type is None and new_type is None and (
                old_field.remote_field.through and new_field.remote_field.through and
                old_field.remote_field.through._meta.auto_created and
                new_field.remote_field.through._meta.auto_created):
            return self._alter_many_to_many(model, old_field, new_field, strict)
        elif old_type is None and new_type is None and (
                old_field.remote_field.through and new_field.remote_field.through and
                not old_field.remote_field.through._meta.auto_created and
                not new_field.remote_field.through._meta.auto_created):
            # Both sides have through models; this is a no-op.
            return
        elif old_type is None or new_type is None:
            raise ValueError(
                "Cannot alter field %s into %s - they are not compatible types "
                "(you cannot alter to or from M2M fields, or add or remove "
                "through= on M2M fields)" % (old_field, new_field)
            )

        self._alter_field(model, old_field, new_field, old_type, new_type,
                          old_db_params, new_db_params, strict)
```
### 34 - django/db/backends/base/schema.py:

Start line: 1248, End line: 1277

```python
class BaseDatabaseSchemaEditor:

    def _delete_primary_key(self, model, strict=False):
        constraint_names = self._constraint_names(model, primary_key=True)
        if strict and len(constraint_names) != 1:
            raise ValueError('Found wrong number (%s) of PK constraints for %s' % (
                len(constraint_names),
                model._meta.db_table,
            ))
        for constraint_name in constraint_names:
            self.execute(self._delete_primary_key_sql(model, constraint_name))

    def _create_primary_key_sql(self, model, field):
        return Statement(
            self.sql_create_pk,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(
                self._create_index_name(model._meta.db_table, [field.column], suffix="_pk")
            ),
            columns=Columns(model._meta.db_table, [field.column], self.quote_name),
        )

    def _delete_primary_key_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_pk, model, name)

    def remove_procedure(self, procedure_name, param_types=()):
        sql = self.sql_delete_procedure % {
            'procedure': self.quote_name(procedure_name),
            'param_types': ','.join(param_types),
        }
        self.execute(sql)
```
### 43 - django/db/backends/base/schema.py:

Start line: 407, End line: 421

```python
class BaseDatabaseSchemaEditor:

    def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
        meta_constraint_names = {constraint.name for constraint in model._meta.constraints}
        meta_index_names = {constraint.name for constraint in model._meta.indexes}
        columns = [model._meta.get_field(field).column for field in fields]
        constraint_names = self._constraint_names(
            model, columns, exclude=meta_constraint_names | meta_index_names,
            **constraint_kwargs
        )
        if len(constraint_names) != 1:
            raise ValueError("Found wrong number (%s) of constraints for %s(%s)" % (
                len(constraint_names),
                model._meta.db_table,
                ", ".join(columns),
            ))
        self.execute(self._delete_constraint_sql(sql, model, constraint_names[0]))
```
### 47 - django/db/backends/base/schema.py:

Start line: 446, End line: 501

```python
class BaseDatabaseSchemaEditor:

    def add_field(self, model, field):
        """
        Create a field on a model. Usually involves adding a column, but may
        involve adding a table instead (for M2M fields).
        """
        # Special-case implicit M2M tables
        if field.many_to_many and field.remote_field.through._meta.auto_created:
            return self.create_model(field.remote_field.through)
        # Get the column's definition
        definition, params = self.column_sql(model, field, include_default=True)
        # It might not actually have a column behind it
        if definition is None:
            return
        # Check constraints can go on the column SQL here
        db_params = field.db_parameters(connection=self.connection)
        if db_params['check']:
            definition += " " + self.sql_check_constraint % db_params
        if field.remote_field and self.connection.features.supports_foreign_keys and field.db_constraint:
            constraint_suffix = '_fk_%(to_table)s_%(to_column)s'
            # Add FK constraint inline, if supported.
            if self.sql_create_column_inline_fk:
                to_table = field.remote_field.model._meta.db_table
                to_column = field.remote_field.model._meta.get_field(field.remote_field.field_name).column
                namespace, _ = split_identifier(model._meta.db_table)
                definition += " " + self.sql_create_column_inline_fk % {
                    'name': self._fk_constraint_name(model, field, constraint_suffix),
                    'namespace': '%s.' % self.quote_name(namespace) if namespace else '',
                    'column': self.quote_name(field.column),
                    'to_table': self.quote_name(to_table),
                    'to_column': self.quote_name(to_column),
                    'deferrable': self.connection.ops.deferrable_sql()
                }
            # Otherwise, add FK constraints later.
            else:
                self.deferred_sql.append(self._create_fk_sql(model, field, constraint_suffix))
        # Build the SQL and run it
        sql = self.sql_create_column % {
            "table": self.quote_name(model._meta.db_table),
            "column": self.quote_name(field.column),
            "definition": definition,
        }
        self.execute(sql, params)
        # Drop the default if we need to
        # (Django usually does not use in-database defaults)
        if not self.skip_default(field) and self.effective_default(field) is not None:
            changes_sql, params = self._alter_column_default_sql(model, None, field, drop=True)
            sql = self.sql_alter_column % {
                "table": self.quote_name(model._meta.db_table),
                "changes": changes_sql,
            }
            self.execute(sql, params)
        # Add an index, if required
        self.deferred_sql.extend(self._field_indexes_sql(model, field))
        # Reset connection if required
        if self.connection.features.connection_persists_old_columns:
            self.connection.close()
```
### 56 - django/db/backends/base/schema.py:

Start line: 1065, End line: 1080

```python
class BaseDatabaseSchemaEditor:

    def _create_fk_sql(self, model, field, suffix):
        table = Table(model._meta.db_table, self.quote_name)
        name = self._fk_constraint_name(model, field, suffix)
        column = Columns(model._meta.db_table, [field.column], self.quote_name)
        to_table = Table(field.target_field.model._meta.db_table, self.quote_name)
        to_column = Columns(field.target_field.model._meta.db_table, [field.target_field.column], self.quote_name)
        deferrable = self.connection.ops.deferrable_sql()
        return Statement(
            self.sql_create_fk,
            table=table,
            name=name,
            column=column,
            to_table=to_table,
            to_column=to_column,
            deferrable=deferrable,
        )
```
### 57 - django/db/backends/base/schema.py:

Start line: 278, End line: 299

```python
class BaseDatabaseSchemaEditor:

    @staticmethod
    def _effective_default(field):
        # This method allows testing its logic without a connection.
        if field.has_default():
            default = field.get_default()
        elif not field.null and field.blank and field.empty_strings_allowed:
            if field.get_internal_type() == "BinaryField":
                default = b''
            else:
                default = ''
        elif getattr(field, 'auto_now', False) or getattr(field, 'auto_now_add', False):
            default = datetime.now()
            internal_type = field.get_internal_type()
            if internal_type == 'DateField':
                default = default.date()
            elif internal_type == 'TimeField':
                default = default.time()
            elif internal_type == 'DateTimeField':
                default = timezone.now()
        else:
            default = None
        return default
```
### 60 - django/db/backends/base/schema.py:

Start line: 386, End line: 405

```python
class BaseDatabaseSchemaEditor:

    def alter_index_together(self, model, old_index_together, new_index_together):
        """
        Deal with a model changing its index_together. The input
        index_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_index_together}
        news = {tuple(fields) for fields in new_index_together}
        # Deleted indexes
        for fields in olds.difference(news):
            self._delete_composed_index(
                model,
                fields,
                {'index': True, 'unique': False},
                self.sql_delete_index,
            )
        # Created indexes
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_index_sql(model, fields, suffix="_idx"))
```
### 62 - django/db/backends/base/schema.py:

Start line: 898, End line: 917

```python
class BaseDatabaseSchemaEditor:

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        # Rename the through table
        if old_field.remote_field.through._meta.db_table != new_field.remote_field.through._meta.db_table:
            self.alter_db_table(old_field.remote_field.through, old_field.remote_field.through._meta.db_table,
                                new_field.remote_field.through._meta.db_table)
        # Repoint the FK to the other side
        self.alter_field(
            new_field.remote_field.through,
            # We need the field that points to the target model, so we can tell alter_field to change it -
            # this is m2m_reverse_field_name() (as opposed to m2m_field_name, which points to our model)
            old_field.remote_field.through._meta.get_field(old_field.m2m_reverse_field_name()),
            new_field.remote_field.through._meta.get_field(new_field.m2m_reverse_field_name()),
        )
        self.alter_field(
            new_field.remote_field.through,
            # for self-referential models we need to alter field from the other end too
            old_field.remote_field.through._meta.get_field(old_field.m2m_field_name()),
            new_field.remote_field.through._meta.get_field(new_field.m2m_field_name()),
        )
```
### 80 - django/db/backends/base/schema.py:

Start line: 255, End line: 276

```python
class BaseDatabaseSchemaEditor:

    def skip_default(self, field):
        """
        Some backends don't accept default values for certain columns types
        (i.e. MySQL longtext and longblob).
        """
        return False

    def prepare_default(self, value):
        """
        Only used for backends which have requires_literal_defaults feature
        """
        raise NotImplementedError(
            'subclasses of BaseDatabaseSchemaEditor for backends which have '
            'requires_literal_defaults must provide a prepare_default() method'
        )

    def _column_default_sql(self, field):
        """
        Return the SQL to use in a DEFAULT clause. The resulting string should
        contain a '%s' placeholder for a default value.
        """
        return '%s'
```
### 82 - django/db/backends/base/schema.py:

Start line: 846, End line: 875

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_default_sql(self, model, old_field, new_field, drop=False):
        """
        Hook to specialize column default alteration.

        Return a (sql, params) fragment to add or drop (depending on the drop
        argument) a default to new_field's column.
        """
        new_default = self.effective_default(new_field)
        default = self._column_default_sql(new_field)
        params = [new_default]

        if drop:
            params = []
        elif self.connection.features.requires_literal_defaults:
            # Some databases (Oracle) can't take defaults as a parameter
            # If this is the case, the SchemaEditor for that database should
            # implement prepare_default().
            default = self.prepare_default(new_default)
            params = []

        new_db_params = new_field.db_parameters(connection=self.connection)
        sql = self.sql_alter_column_no_default if drop else self.sql_alter_column_default
        return (
            sql % {
                'column': self.quote_name(new_field.column),
                'type': new_db_params['type'],
                'default': default,
            },
            params,
        )
```
### 83 - django/db/backends/base/schema.py:

Start line: 824, End line: 844

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_null_sql(self, model, old_field, new_field):
        """
        Hook to specialize column null alteration.

        Return a (sql, params) fragment to set a column to null or non-null
        as required by new_field, or None if no changes are required.
        """
        if (self.connection.features.interprets_empty_strings_as_nulls and
                new_field.get_internal_type() in ("CharField", "TextField")):
            # The field is nullable in the database anyway, leave it alone.
            return
        else:
            new_db_params = new_field.db_parameters(connection=self.connection)
            sql = self.sql_alter_column_null if new_field.null else self.sql_alter_column_not_null
            return (
                sql % {
                    'column': self.quote_name(new_field.column),
                    'type': new_db_params['type'],
                },
                [],
            )
```
### 84 - django/db/backends/base/schema.py:

Start line: 1106, End line: 1136

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(
        self, model, fields, name, condition=None, deferrable=None,
        include=None, opclasses=None,
    ):
        if (
            deferrable and
            not self.connection.features.supports_deferrable_unique_constraints
        ):
            return None
        if condition or include or opclasses:
            # Databases support conditional and covering unique constraints via
            # a unique index.
            sql = self._create_unique_sql(
                model,
                fields,
                name=name,
                condition=condition,
                include=include,
                opclasses=opclasses,
            )
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            'columns': ', '.join(map(self.quote_name, fields)),
            'deferrable': self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }
```
### 86 - django/db/backends/base/schema.py:

Start line: 370, End line: 384

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))
```
### 96 - django/db/backends/base/schema.py:

Start line: 503, End line: 531

```python
class BaseDatabaseSchemaEditor:

    def remove_field(self, model, field):
        """
        Remove a field from a model. Usually involves deleting a column,
        but for M2Ms may involve deleting a table.
        """
        # Special-case implicit M2M tables
        if field.many_to_many and field.remote_field.through._meta.auto_created:
            return self.delete_model(field.remote_field.through)
        # It might not actually have a column behind it
        if field.db_parameters(connection=self.connection)['type'] is None:
            return
        # Drop any FK constraints, MySQL requires explicit deletion
        if field.remote_field:
            fk_names = self._constraint_names(model, [field.column], foreign_key=True)
            for fk_name in fk_names:
                self.execute(self._delete_fk_sql(model, fk_name))
        # Delete the column
        sql = self.sql_delete_column % {
            "table": self.quote_name(model._meta.db_table),
            "column": self.quote_name(field.column),
        }
        self.execute(sql)
        # Reset connection if required
        if self.connection.features.connection_persists_old_columns:
            self.connection.close()
        # Remove all deferred statements referencing the deleted column.
        for sql in list(self.deferred_sql):
            if isinstance(sql, Statement) and sql.references_column(model._meta.db_table, field.column):
                self.deferred_sql.remove(sql)
```
### 97 - django/db/backends/base/schema.py:

Start line: 1138, End line: 1173

```python
class BaseDatabaseSchemaEditor:

    def _create_unique_sql(
        self, model, columns, name=None, condition=None, deferrable=None,
        include=None, opclasses=None,
    ):
        if (
            (
                deferrable and
                not self.connection.features.supports_deferrable_unique_constraints
            ) or
            (condition and not self.connection.features.supports_partial_indexes) or
            (include and not self.connection.features.supports_covering_indexes)
        ):
            return None

        def create_unique_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        table = Table(model._meta.db_table, self.quote_name)
        if name is None:
            name = IndexName(model._meta.db_table, columns, '_uniq', create_unique_name)
        else:
            name = self.quote_name(name)
        columns = self._index_columns(table, columns, col_suffixes=(), opclasses=opclasses)
        if condition or include or opclasses:
            sql = self.sql_create_unique_index
        else:
            sql = self.sql_create_unique
        return Statement(
            sql,
            table=table,
            name=name,
            columns=columns,
            condition=self._index_condition_sql(condition),
            deferrable=self._deferrable_constraint_sql(deferrable),
            include=self._index_include_sql(model, include),
        )
```
### 99 - django/db/backends/base/schema.py:

Start line: 1175, End line: 1192

```python
class BaseDatabaseSchemaEditor:

    def _delete_unique_sql(
        self, model, name, condition=None, deferrable=None, include=None,
        opclasses=None,
    ):
        if (
            (
                deferrable and
                not self.connection.features.supports_deferrable_unique_constraints
            ) or
            (condition and not self.connection.features.supports_partial_indexes) or
            (include and not self.connection.features.supports_covering_indexes)
        ):
            return None
        if condition or include or opclasses:
            sql = self.sql_delete_index
        else:
            sql = self.sql_delete_unique
        return self._delete_constraint_sql(sql, model, name)
```
### 122 - django/db/backends/base/schema.py:

Start line: 877, End line: 896

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_type_sql(self, model, old_field, new_field, new_type):
        """
        Hook to specialize column type alteration for different backends,
        for cases when a creation type is different to an alteration type
        (e.g. SERIAL in PostgreSQL, PostGIS fields).

        Return a two-tuple of: an SQL fragment of (sql, params) to insert into
        an ALTER TABLE statement and a list of extra (sql, params) tuples to
        run once the field is altered.
        """
        return (
            (
                self.sql_alter_column_type % {
                    "column": self.quote_name(new_field.column),
                    "type": new_type,
                },
                [],
            ),
            [],
        )
```
