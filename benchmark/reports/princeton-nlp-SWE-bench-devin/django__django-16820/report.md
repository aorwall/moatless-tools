# django__django-16820

| **django/django** | `c61219a7ae051d2baab53f041e00592011fc550c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 6 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -303,6 +303,71 @@ def reduce(self, operation, app_label):
                         managers=self.managers,
                     ),
                 ]
+        elif (
+            isinstance(operation, IndexOperation)
+            and self.name_lower == operation.model_name_lower
+        ):
+            if isinstance(operation, AddIndex):
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": [
+                                *self.options.get("indexes", []),
+                                operation.index,
+                            ],
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
+            elif isinstance(operation, RemoveIndex):
+                options_indexes = [
+                    index
+                    for index in self.options.get("indexes", [])
+                    if index.name != operation.name
+                ]
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": options_indexes,
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
+            elif isinstance(operation, RenameIndex) and operation.old_fields:
+                options_index_together = {
+                    fields
+                    for fields in self.options.get("index_together", [])
+                    if fields != operation.old_fields
+                }
+                if options_index_together:
+                    self.options["index_together"] = options_index_together
+                else:
+                    self.options.pop("index_together", None)
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": [
+                                *self.options.get("indexes", []),
+                                models.Index(
+                                    fields=operation.old_fields, name=operation.new_name
+                                ),
+                            ],
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
         return super().reduce(operation, app_label)
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/operations/models.py | 306 | 306 | - | 6 | -


## Problem Statement

```
Squashing migrations with Meta.index_together -> indexes transition should remove deprecation warnings.
Description
	
Squashing migrations with Meta.index_together -> Meta.indexes transition should remove deprecation warnings. As far as I'm aware, it's a 4.2 release blocker because you cannot get rid of the index_together deprecation warnings without rewriting migrations, see comment.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/utils/deprecation.py | 1 | 38| 235 | 235 | 1058 | 
| 2 | 2 django/db/migrations/autodetector.py | 1520 | 1548| 235 | 470 | 14809 | 
| 3 | 3 django/core/management/commands/squashmigrations.py | 161 | 253| 766 | 1236 | 16849 | 
| 4 | 3 django/db/migrations/autodetector.py | 1216 | 1303| 719 | 1955 | 16849 | 
| 5 | 3 django/core/management/commands/squashmigrations.py | 62 | 159| 809 | 2764 | 16849 | 
| 6 | 3 django/db/migrations/autodetector.py | 1495 | 1518| 161 | 2925 | 16849 | 
| 7 | 4 django/db/backends/base/schema.py | 565 | 584| 199 | 3124 | 31431 | 
| 8 | 4 django/utils/deprecation.py | 41 | 83| 339 | 3463 | 31431 | 
| 9 | 4 django/utils/deprecation.py | 86 | 137| 362 | 3825 | 31431 | 
| 10 | 5 django/core/management/commands/migrate.py | 191 | 269| 678 | 4503 | 35357 | 
| 11 | **6 django/db/migrations/operations/models.py** | 1005 | 1022| 149 | 4652 | 43276 | 
| 12 | 6 django/db/migrations/autodetector.py | 1305 | 1342| 252 | 4904 | 43276 | 
| 13 | 6 django/db/migrations/autodetector.py | 399 | 415| 141 | 5045 | 43276 | 
| 14 | 7 django/db/migrations/executor.py | 290 | 305| 165 | 5210 | 46717 | 
| 15 | 8 django/core/management/commands/makemigrations.py | 1 | 23| 185 | 5395 | 50665 | 
| 16 | 9 django/core/management/commands/optimizemigration.py | 1 | 130| 940 | 6335 | 51605 | 
| 17 | 10 django/db/models/options.py | 1 | 58| 353 | 6688 | 59301 | 
| 18 | 11 django/db/models/fields/__init__.py | 468 | 493| 198 | 6886 | 78305 | 
| 19 | 11 django/core/management/commands/squashmigrations.py | 1 | 60| 387 | 7273 | 78305 | 
| 20 | **11 django/db/migrations/operations/models.py** | 968 | 1003| 319 | 7592 | 78305 | 
| 21 | 11 django/db/migrations/autodetector.py | 1448 | 1493| 318 | 7910 | 78305 | 
| 22 | 11 django/core/management/commands/migrate.py | 270 | 368| 813 | 8723 | 78305 | 
| 23 | 12 django/core/management/base.py | 566 | 606| 293 | 9016 | 83165 | 
| 24 | 12 django/core/management/commands/makemigrations.py | 405 | 515| 927 | 9943 | 83165 | 
| 25 | **12 django/db/migrations/operations/models.py** | 600 | 624| 213 | 10156 | 83165 | 
| 26 | 12 django/core/management/commands/migrate.py | 96 | 189| 765 | 10921 | 83165 | 
| 27 | **12 django/db/migrations/operations/models.py** | 1024 | 1059| 249 | 11170 | 83165 | 
| 28 | **12 django/db/migrations/operations/models.py** | 1103 | 1143| 337 | 11507 | 83165 | 
| 29 | 13 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 43| 232 | 11739 | 83397 | 
| 30 | 14 django/db/migrations/loader.py | 307 | 336| 212 | 11951 | 86550 | 
| 31 | 15 django/contrib/postgres/fields/citext.py | 1 | 24| 164 | 12115 | 87136 | 
| 32 | 16 django/db/models/base.py | 1875 | 1902| 191 | 12306 | 105863 | 
| 33 | 17 django/db/migrations/exceptions.py | 1 | 61| 249 | 12555 | 106113 | 
| 34 | **17 django/db/migrations/operations/models.py** | 589 | 598| 129 | 12684 | 106113 | 
| 35 | 17 django/core/management/commands/makemigrations.py | 261 | 331| 572 | 13256 | 106113 | 
| 36 | 18 django/core/files/storage/__init__.py | 1 | 43| 235 | 13491 | 106348 | 
| 37 | 18 django/core/management/commands/makemigrations.py | 104 | 194| 791 | 14282 | 106348 | 
| 38 | 19 django/db/backends/mysql/features.py | 86 | 169| 670 | 14952 | 108862 | 
| 39 | 19 django/db/migrations/loader.py | 222 | 305| 791 | 15743 | 108862 | 
| 40 | 19 django/db/backends/base/schema.py | 52 | 72| 152 | 15895 | 108862 | 
| 41 | 19 django/db/migrations/autodetector.py | 1571 | 1591| 192 | 16087 | 108862 | 
| 42 | 20 django/db/backends/mysql/schema.py | 122 | 140| 133 | 16220 | 111053 | 
| 43 | 20 django/db/migrations/autodetector.py | 1633 | 1665| 258 | 16478 | 111053 | 
| 44 | **20 django/db/migrations/operations/models.py** | 913 | 966| 348 | 16826 | 111053 | 
| 45 | 20 django/db/migrations/autodetector.py | 281 | 379| 806 | 17632 | 111053 | 
| 46 | 20 django/db/backends/base/schema.py | 586 | 618| 268 | 17900 | 111053 | 
| 47 | 21 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 59 | 77| 113 | 18013 | 111618 | 
| 48 | 21 django/core/management/commands/squashmigrations.py | 255 | 268| 112 | 18125 | 111618 | 
| 49 | 22 django/contrib/postgres/operations.py | 144 | 166| 261 | 18386 | 113998 | 
| 50 | 23 django/db/models/constraints.py | 1 | 16| 136 | 18522 | 117511 | 
| 51 | 23 django/db/backends/mysql/schema.py | 1 | 44| 484 | 19006 | 117511 | 
| 52 | 23 django/db/migrations/autodetector.py | 267 | 280| 178 | 19184 | 117511 | 
| 53 | 24 django/db/migrations/graph.py | 124 | 157| 308 | 19492 | 120134 | 
| 54 | 25 django/db/migrations/questioner.py | 291 | 342| 367 | 19859 | 122830 | 
| 55 | 26 django/contrib/postgres/indexes.py | 1 | 42| 287 | 20146 | 124686 | 
| 56 | 26 django/core/management/commands/migrate.py | 369 | 390| 204 | 20350 | 124686 | 
| 57 | 26 django/db/migrations/autodetector.py | 381 | 397| 161 | 20511 | 124686 | 
| 58 | 27 django/db/models/fields/related.py | 1 | 42| 267 | 20778 | 139390 | 
| 59 | 27 django/db/backends/base/schema.py | 1313 | 1357| 412 | 21190 | 139390 | 
| 60 | 27 django/core/management/commands/makemigrations.py | 196 | 259| 458 | 21648 | 139390 | 
| 61 | 27 django/core/management/commands/migrate.py | 17 | 94| 487 | 22135 | 139390 | 
| 62 | 27 django/db/models/base.py | 1 | 66| 361 | 22496 | 139390 | 
| 63 | 27 django/db/migrations/autodetector.py | 1593 | 1631| 304 | 22800 | 139390 | 
| 64 | 28 django/db/models/__init__.py | 1 | 116| 682 | 23482 | 140072 | 
| 65 | 29 django/db/models/sql/compiler.py | 1999 | 2060| 588 | 24070 | 156831 | 
| 66 | 29 django/db/migrations/autodetector.py | 1074 | 1095| 188 | 24258 | 156831 | 
| 67 | 29 django/db/migrations/autodetector.py | 1683 | 1733| 439 | 24697 | 156831 | 
| 68 | 30 django/db/backends/sqlite3/schema.py | 489 | 551| 472 | 25169 | 161546 | 
| 69 | 30 django/db/migrations/autodetector.py | 1367 | 1395| 164 | 25333 | 161546 | 
| 70 | 30 django/db/migrations/autodetector.py | 90 | 102| 119 | 25452 | 161546 | 
| 71 | 31 django/db/migrations/recorder.py | 1 | 22| 148 | 25600 | 162233 | 
| 72 | 31 django/db/backends/sqlite3/schema.py | 122 | 173| 527 | 26127 | 162233 | 
| 73 | 32 django/contrib/redirects/migrations/0001_initial.py | 1 | 65| 309 | 26436 | 162542 | 
| 74 | 32 django/db/migrations/graph.py | 269 | 292| 183 | 26619 | 162542 | 
| 75 | 32 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 56| 452 | 27071 | 162542 | 
| 76 | 32 django/db/migrations/autodetector.py | 1344 | 1365| 197 | 27268 | 162542 | 
| 77 | 32 django/db/backends/sqlite3/schema.py | 99 | 120| 195 | 27463 | 162542 | 
| 78 | 32 django/db/backends/base/schema.py | 544 | 563| 196 | 27659 | 162542 | 
| 79 | 32 django/db/migrations/autodetector.py | 596 | 772| 1231 | 28890 | 162542 | 
| 80 | 32 django/core/management/commands/migrate.py | 432 | 487| 409 | 29299 | 162542 | 
| 81 | 32 django/db/backends/base/schema.py | 1470 | 1486| 153 | 29452 | 162542 | 
| 82 | 32 django/db/backends/base/schema.py | 948 | 1035| 795 | 30247 | 162542 | 
| 83 | 33 django/db/models/fields/json.py | 1 | 21| 131 | 30378 | 167313 | 
| 84 | **33 django/db/migrations/operations/models.py** | 870 | 910| 344 | 30722 | 167313 | 
| 85 | 33 django/db/migrations/autodetector.py | 1 | 18| 113 | 30835 | 167313 | 
| 86 | 33 django/db/backends/base/schema.py | 852 | 947| 799 | 31634 | 167313 | 
| 87 | 34 django/db/backends/postgresql/schema.py | 278 | 312| 277 | 31911 | 170171 | 
| 88 | 34 django/db/backends/postgresql/schema.py | 314 | 339| 235 | 32146 | 170171 | 
| 89 | 34 django/utils/deprecation.py | 139 | 157| 122 | 32268 | 170171 | 
| 90 | 34 django/db/backends/mysql/schema.py | 210 | 231| 205 | 32473 | 170171 | 
| 91 | 34 django/core/management/commands/migrate.py | 1 | 14| 134 | 32607 | 170171 | 
| 92 | 35 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 33002 | 170837 | 
| 93 | 36 django/contrib/postgres/aggregates/general.py | 1 | 51| 263 | 33265 | 171493 | 
| 94 | **36 django/db/migrations/operations/models.py** | 674 | 698| 231 | 33496 | 171493 | 
| 95 | 36 django/db/backends/mysql/schema.py | 160 | 208| 374 | 33870 | 171493 | 
| 96 | 37 django/db/migrations/writer.py | 1 | 115| 888 | 34758 | 173794 | 
| 97 | 37 django/db/migrations/autodetector.py | 211 | 232| 238 | 34996 | 173794 | 
| 98 | 37 django/db/migrations/autodetector.py | 807 | 902| 712 | 35708 | 173794 | 
| 99 | 38 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 27| 143 | 35851 | 173937 | 
| 100 | 39 django/contrib/gis/db/backends/spatialite/schema.py | 137 | 192| 404 | 36255 | 175331 | 
| 101 | 39 django/db/backends/base/schema.py | 1520 | 1535| 206 | 36461 | 175331 | 
| 102 | 39 django/core/management/commands/makemigrations.py | 26 | 102| 446 | 36907 | 175331 | 
| 103 | 39 django/db/migrations/graph.py | 101 | 122| 192 | 37099 | 175331 | 
| 104 | 40 django/contrib/gis/db/backends/mysql/schema.py | 57 | 88| 247 | 37346 | 175988 | 
| 105 | 40 django/db/models/base.py | 2286 | 2477| 1306 | 38652 | 175988 | 
| 106 | 40 django/db/backends/postgresql/schema.py | 144 | 257| 920 | 39572 | 175988 | 
| 107 | 40 django/db/backends/base/schema.py | 1537 | 1556| 175 | 39747 | 175988 | 
| 108 | 40 django/core/management/commands/migrate.py | 392 | 430| 361 | 40108 | 175988 | 
| 109 | 41 django/contrib/sitemaps/views.py | 1 | 27| 162 | 40270 | 177060 | 
| 110 | 42 django/db/backends/mysql/base.py | 276 | 312| 259 | 40529 | 180572 | 
| 111 | 43 django/contrib/flatpages/migrations/0001_initial.py | 1 | 69| 355 | 40884 | 180927 | 
| 112 | **43 django/db/migrations/operations/models.py** | 370 | 416| 380 | 41264 | 180927 | 
| 113 | 43 django/db/backends/postgresql/schema.py | 341 | 377| 212 | 41476 | 180927 | 
| 114 | 43 django/db/models/base.py | 1599 | 1620| 150 | 41626 | 180927 | 
| 115 | 44 django/contrib/auth/migrations/0001_initial.py | 1 | 205| 1007 | 42633 | 181934 | 
| 116 | **44 django/db/migrations/operations/models.py** | 627 | 648| 148 | 42781 | 181934 | 
| 117 | 45 django/db/backends/sqlite3/features.py | 63 | 130| 528 | 43309 | 183278 | 
| 118 | 45 django/db/migrations/autodetector.py | 1735 | 1760| 243 | 43552 | 183278 | 
| 119 | **45 django/db/migrations/operations/models.py** | 1062 | 1100| 283 | 43835 | 183278 | 
| 120 | 46 django/db/backends/postgresql/features.py | 1 | 112| 895 | 44730 | 184323 | 
| 121 | 46 django/db/backends/mysql/schema.py | 142 | 158| 144 | 44874 | 184323 | 
| 122 | 47 django/core/management/sql.py | 42 | 60| 132 | 45006 | 184690 | 
| 123 | 47 django/db/migrations/executor.py | 307 | 411| 862 | 45868 | 184690 | 
| 124 | 47 django/db/models/fields/related.py | 1423 | 1461| 213 | 46081 | 184690 | 
| 125 | 47 django/db/migrations/autodetector.py | 904 | 977| 623 | 46704 | 184690 | 
| 126 | 48 django/db/backends/mysql/compiler.py | 55 | 85| 240 | 46944 | 185337 | 
| 127 | 48 django/db/migrations/autodetector.py | 1550 | 1569| 187 | 47131 | 185337 | 
| 128 | **48 django/db/migrations/operations/models.py** | 700 | 716| 159 | 47290 | 185337 | 
| 129 | 48 django/db/migrations/recorder.py | 24 | 46| 145 | 47435 | 185337 | 
| 130 | 48 django/contrib/postgres/operations.py | 1 | 41| 303 | 47738 | 185337 | 
| 131 | 48 django/db/backends/sqlite3/schema.py | 256 | 361| 885 | 48623 | 185337 | 
| 132 | 48 django/db/migrations/graph.py | 63 | 99| 337 | 48960 | 185337 | 
| 133 | 48 django/db/models/fields/__init__.py | 382 | 414| 210 | 49170 | 185337 | 
| 134 | 48 django/db/migrations/autodetector.py | 1667 | 1681| 135 | 49305 | 185337 | 
| 135 | 48 django/db/migrations/autodetector.py | 480 | 510| 267 | 49572 | 185337 | 
| 136 | 49 django/contrib/admin/migrations/0001_initial.py | 1 | 76| 363 | 49935 | 185700 | 
| 137 | **49 django/db/migrations/operations/models.py** | 562 | 587| 163 | 50098 | 185700 | 
| 138 | **49 django/db/migrations/operations/models.py** | 418 | 431| 127 | 50225 | 185700 | 
| 139 | 49 django/db/migrations/writer.py | 209 | 315| 632 | 50857 | 185700 | 
| 140 | 49 django/db/models/constraints.py | 19 | 73| 456 | 51313 | 185700 | 
| 141 | 49 django/db/migrations/recorder.py | 48 | 104| 400 | 51713 | 185700 | 


## Patch

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -303,6 +303,71 @@ def reduce(self, operation, app_label):
                         managers=self.managers,
                     ),
                 ]
+        elif (
+            isinstance(operation, IndexOperation)
+            and self.name_lower == operation.model_name_lower
+        ):
+            if isinstance(operation, AddIndex):
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": [
+                                *self.options.get("indexes", []),
+                                operation.index,
+                            ],
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
+            elif isinstance(operation, RemoveIndex):
+                options_indexes = [
+                    index
+                    for index in self.options.get("indexes", [])
+                    if index.name != operation.name
+                ]
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": options_indexes,
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
+            elif isinstance(operation, RenameIndex) and operation.old_fields:
+                options_index_together = {
+                    fields
+                    for fields in self.options.get("index_together", [])
+                    if fields != operation.old_fields
+                }
+                if options_index_together:
+                    self.options["index_together"] = options_index_together
+                else:
+                    self.options.pop("index_together", None)
+                return [
+                    CreateModel(
+                        self.name,
+                        fields=self.fields,
+                        options={
+                            **self.options,
+                            "indexes": [
+                                *self.options.get("indexes", []),
+                                models.Index(
+                                    fields=operation.old_fields, name=operation.new_name
+                                ),
+                            ],
+                        },
+                        bases=self.bases,
+                        managers=self.managers,
+                    ),
+                ]
         return super().reduce(operation, app_label)
 
 

```

## Test Patch

```diff
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -2266,10 +2266,9 @@ def test_same_app_circular_fk_dependency_with_unique_together_and_indexes(self):
             changes,
             "eggs",
             0,
-            ["CreateModel", "CreateModel", "AddIndex", "AlterUniqueTogether"],
+            ["CreateModel", "CreateModel"],
         )
         self.assertNotIn("unique_together", changes["eggs"][0].operations[0].options)
-        self.assertNotIn("unique_together", changes["eggs"][0].operations[1].options)
         self.assertMigrationDependencies(changes, "eggs", 0, [])
 
     def test_alter_db_table_add(self):
@@ -2565,6 +2564,9 @@ def test(from_state, to_state, msg):
 
     def test_create_model_with_indexes(self):
         """Test creation of new model with indexes already defined."""
+        added_index = models.Index(
+            fields=["name"], name="create_model_with_indexes_idx"
+        )
         author = ModelState(
             "otherapp",
             "Author",
@@ -2573,25 +2575,25 @@ def test_create_model_with_indexes(self):
                 ("name", models.CharField(max_length=200)),
             ],
             {
-                "indexes": [
-                    models.Index(fields=["name"], name="create_model_with_indexes_idx")
-                ]
+                "indexes": [added_index],
             },
         )
         changes = self.get_changes([], [author])
-        added_index = models.Index(
-            fields=["name"], name="create_model_with_indexes_idx"
-        )
         # Right number of migrations?
         self.assertEqual(len(changes["otherapp"]), 1)
         # Right number of actions?
         migration = changes["otherapp"][0]
-        self.assertEqual(len(migration.operations), 2)
+        self.assertEqual(len(migration.operations), 1)
         # Right actions order?
-        self.assertOperationTypes(changes, "otherapp", 0, ["CreateModel", "AddIndex"])
+        self.assertOperationTypes(changes, "otherapp", 0, ["CreateModel"])
         self.assertOperationAttributes(changes, "otherapp", 0, 0, name="Author")
         self.assertOperationAttributes(
-            changes, "otherapp", 0, 1, model_name="author", index=added_index
+            changes,
+            "otherapp",
+            0,
+            0,
+            name="Author",
+            options={"indexes": [added_index]},
         )
 
     def test_add_indexes(self):
@@ -4043,62 +4045,69 @@ def test_add_model_order_with_respect_to_unique_together(self):
             },
         )
 
-    def test_add_model_order_with_respect_to_index_constraint(self):
-        tests = [
-            (
-                "AddIndex",
-                {
-                    "indexes": [
-                        models.Index(fields=["_order"], name="book_order_idx"),
-                    ]
-                },
-            ),
-            (
-                "AddConstraint",
-                {
-                    "constraints": [
-                        models.CheckConstraint(
-                            check=models.Q(_order__gt=1),
-                            name="book_order_gt_1",
-                        ),
-                    ]
-                },
-            ),
-        ]
-        for operation, extra_option in tests:
-            with self.subTest(operation=operation):
-                after = ModelState(
-                    "testapp",
-                    "Author",
-                    [
-                        ("id", models.AutoField(primary_key=True)),
-                        ("name", models.CharField(max_length=200)),
-                        ("book", models.ForeignKey("otherapp.Book", models.CASCADE)),
-                    ],
-                    options={
-                        "order_with_respect_to": "book",
-                        **extra_option,
-                    },
-                )
-                changes = self.get_changes([], [self.book, after])
-                self.assertNumberMigrations(changes, "testapp", 1)
-                self.assertOperationTypes(
-                    changes,
-                    "testapp",
-                    0,
-                    [
-                        "CreateModel",
-                        operation,
-                    ],
-                )
-                self.assertOperationAttributes(
-                    changes,
-                    "testapp",
-                    0,
-                    0,
-                    name="Author",
-                    options={"order_with_respect_to": "book"},
-                )
+    def test_add_model_order_with_respect_to_constraint(self):
+        after = ModelState(
+            "testapp",
+            "Author",
+            [
+                ("id", models.AutoField(primary_key=True)),
+                ("name", models.CharField(max_length=200)),
+                ("book", models.ForeignKey("otherapp.Book", models.CASCADE)),
+            ],
+            options={
+                "order_with_respect_to": "book",
+                "constraints": [
+                    models.CheckConstraint(
+                        check=models.Q(_order__gt=1), name="book_order_gt_1"
+                    ),
+                ],
+            },
+        )
+        changes = self.get_changes([], [self.book, after])
+        self.assertNumberMigrations(changes, "testapp", 1)
+        self.assertOperationTypes(
+            changes,
+            "testapp",
+            0,
+            ["CreateModel", "AddConstraint"],
+        )
+        self.assertOperationAttributes(
+            changes,
+            "testapp",
+            0,
+            0,
+            name="Author",
+            options={"order_with_respect_to": "book"},
+        )
+
+    def test_add_model_order_with_respect_to_index(self):
+        after = ModelState(
+            "testapp",
+            "Author",
+            [
+                ("id", models.AutoField(primary_key=True)),
+                ("name", models.CharField(max_length=200)),
+                ("book", models.ForeignKey("otherapp.Book", models.CASCADE)),
+            ],
+            options={
+                "order_with_respect_to": "book",
+                "indexes": [models.Index(fields=["_order"], name="book_order_idx")],
+            },
+        )
+        changes = self.get_changes([], [self.book, after])
+        self.assertNumberMigrations(changes, "testapp", 1)
+        self.assertOperationTypes(changes, "testapp", 0, ["CreateModel"])
+        self.assertOperationAttributes(
+            changes,
+            "testapp",
+            0,
+            0,
+            name="Author",
+            options={
+                "order_with_respect_to": "book",
+                "indexes": [models.Index(fields=["_order"], name="book_order_idx")],
+            },
+        )
 
     def test_set_alter_order_with_respect_to_index_constraint_unique_together(self):
         tests = [
diff --git a/tests/migrations/test_optimizer.py b/tests/migrations/test_optimizer.py
--- a/tests/migrations/test_optimizer.py
+++ b/tests/migrations/test_optimizer.py
@@ -1172,3 +1172,181 @@ def test_add_remove_index(self):
             ],
             [],
         )
+
+    def test_create_model_add_index(self):
+        self.assertOptimizesTo(
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [models.Index(fields=["age"], name="idx_pony_age")],
+                    },
+                ),
+                migrations.AddIndex(
+                    "Pony",
+                    models.Index(fields=["weight"], name="idx_pony_weight"),
+                ),
+            ],
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [
+                            models.Index(fields=["age"], name="idx_pony_age"),
+                            models.Index(fields=["weight"], name="idx_pony_weight"),
+                        ],
+                    },
+                ),
+            ],
+        )
+
+    def test_create_model_remove_index(self):
+        self.assertOptimizesTo(
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [
+                            models.Index(fields=["age"], name="idx_pony_age"),
+                            models.Index(fields=["weight"], name="idx_pony_weight"),
+                        ],
+                    },
+                ),
+                migrations.RemoveIndex("Pony", "idx_pony_age"),
+            ],
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [
+                            models.Index(fields=["weight"], name="idx_pony_weight"),
+                        ],
+                    },
+                ),
+            ],
+        )
+
+    def test_create_model_remove_index_together_rename_index(self):
+        self.assertOptimizesTo(
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "index_together": [("age", "weight")],
+                    },
+                ),
+                migrations.RenameIndex(
+                    "Pony", new_name="idx_pony_age_weight", old_fields=("age", "weight")
+                ),
+            ],
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [
+                            models.Index(
+                                fields=["age", "weight"], name="idx_pony_age_weight"
+                            ),
+                        ],
+                    },
+                ),
+            ],
+        )
+
+    def test_create_model_index_together_rename_index(self):
+        self.assertOptimizesTo(
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                        ("height", models.IntegerField()),
+                        ("rank", models.IntegerField()),
+                    ],
+                    options={
+                        "index_together": [("age", "weight"), ("height", "rank")],
+                    },
+                ),
+                migrations.RenameIndex(
+                    "Pony", new_name="idx_pony_age_weight", old_fields=("age", "weight")
+                ),
+            ],
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                        ("height", models.IntegerField()),
+                        ("rank", models.IntegerField()),
+                    ],
+                    options={
+                        "index_together": {("height", "rank")},
+                        "indexes": [
+                            models.Index(
+                                fields=["age", "weight"], name="idx_pony_age_weight"
+                            ),
+                        ],
+                    },
+                ),
+            ],
+        )
+
+    def test_create_model_rename_index_no_old_fields(self):
+        self.assertOptimizesTo(
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [models.Index(fields=["age"], name="idx_pony_age")],
+                    },
+                ),
+                migrations.RenameIndex(
+                    "Pony", new_name="idx_pony_age_new", old_name="idx_pony_age"
+                ),
+            ],
+            [
+                migrations.CreateModel(
+                    name="Pony",
+                    fields=[
+                        ("weight", models.IntegerField()),
+                        ("age", models.IntegerField()),
+                    ],
+                    options={
+                        "indexes": [models.Index(fields=["age"], name="idx_pony_age")],
+                    },
+                ),
+                migrations.RenameIndex(
+                    "Pony", new_name="idx_pony_age_new", old_name="idx_pony_age"
+                ),
+            ],
+        )

```


## Code snippets

### 1 - django/utils/deprecation.py:

Start line: 1, End line: 38

```python
import inspect
import warnings

from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async


class RemovedInDjango51Warning(DeprecationWarning):
    pass


class RemovedInDjango60Warning(PendingDeprecationWarning):
    pass


RemovedInNextVersionWarning = RemovedInDjango51Warning
RemovedAfterNextVersionWarning = RemovedInDjango60Warning


class warn_about_renamed_method:
    def __init__(
        self, class_name, old_method_name, new_method_name, deprecation_warning
    ):
        self.class_name = class_name
        self.old_method_name = old_method_name
        self.new_method_name = new_method_name
        self.deprecation_warning = deprecation_warning

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            warnings.warn(
                "`%s.%s` is deprecated, use `%s` instead."
                % (self.class_name, self.old_method_name, self.new_method_name),
                self.deprecation_warning,
                2,
            )
            return f(*args, **kwargs)

        return wrapper
```
### 2 - django/db/migrations/autodetector.py:

Start line: 1520, End line: 1548

```python
class MigrationAutodetector:

    def generate_removed_altered_unique_together(self):
        self._generate_removed_altered_foo_together(operations.AlterUniqueTogether)

    # RemovedInDjango51Warning.
    def generate_removed_altered_index_together(self):
        self._generate_removed_altered_foo_together(operations.AlterIndexTogether)

    def _generate_altered_foo_together(self, operation):
        for (
            old_value,
            new_value,
            app_label,
            model_name,
            dependencies,
        ) in self._get_altered_foo_together_operations(operation.option_name):
            removal_value = new_value.intersection(old_value)
            if new_value != removal_value:
                self.add_operation(
                    app_label,
                    operation(name=model_name, **{operation.option_name: new_value}),
                    dependencies=dependencies,
                )

    def generate_altered_unique_together(self):
        self._generate_altered_foo_together(operations.AlterUniqueTogether)

    # RemovedInDjango51Warning.
    def generate_altered_index_together(self):
        self._generate_altered_foo_together(operations.AlterIndexTogether)
```
### 3 - django/core/management/commands/squashmigrations.py:

Start line: 161, End line: 253

```python
class Command(BaseCommand):

    def handle(self, **options):
        # ... other code

        if no_optimize:
            if self.verbosity > 0:
                self.stdout.write(
                    self.style.MIGRATE_HEADING("(Skipping optimization.)")
                )
            new_operations = operations
        else:
            if self.verbosity > 0:
                self.stdout.write(self.style.MIGRATE_HEADING("Optimizing..."))

            optimizer = MigrationOptimizer()
            new_operations = optimizer.optimize(operations, migration.app_label)

            if self.verbosity > 0:
                if len(new_operations) == len(operations):
                    self.stdout.write("  No optimizations possible.")
                else:
                    self.stdout.write(
                        "  Optimized from %s operations to %s operations."
                        % (len(operations), len(new_operations))
                    )

        # Work out the value of replaces (any squashed ones we're re-squashing)
        # need to feed their replaces into ours
        replaces = []
        for migration in migrations_to_squash:
            if migration.replaces:
                replaces.extend(migration.replaces)
            else:
                replaces.append((migration.app_label, migration.name))

        # Make a new migration with those operations
        subclass = type(
            "Migration",
            (migrations.Migration,),
            {
                "dependencies": dependencies,
                "operations": new_operations,
                "replaces": replaces,
            },
        )
        if start_migration_name:
            if squashed_name:
                # Use the name from --squashed-name.
                prefix, _ = start_migration.name.split("_", 1)
                name = "%s_%s" % (prefix, squashed_name)
            else:
                # Generate a name.
                name = "%s_squashed_%s" % (start_migration.name, migration.name)
            new_migration = subclass(name, app_label)
        else:
            name = "0001_%s" % (squashed_name or "squashed_%s" % migration.name)
            new_migration = subclass(name, app_label)
            new_migration.initial = True

        # Write out the new migration file
        writer = MigrationWriter(new_migration, include_header)
        if os.path.exists(writer.path):
            raise CommandError(
                f"Migration {new_migration.name} already exists. Use a different name."
            )
        with open(writer.path, "w", encoding="utf-8") as fh:
            fh.write(writer.as_string())
        run_formatters([writer.path])

        if self.verbosity > 0:
            self.stdout.write(
                self.style.MIGRATE_HEADING(
                    "Created new squashed migration %s" % writer.path
                )
                + "\n"
                "  You should commit this migration but leave the old ones in place;\n"
                "  the new migration will be used for new installs. Once you are sure\n"
                "  all instances of the codebase have applied the migrations you "
                "squashed,\n"
                "  you can delete them."
            )
            if writer.needs_manual_porting:
                self.stdout.write(
                    self.style.MIGRATE_HEADING("Manual porting required") + "\n"
                    "  Your migrations contained functions that must be manually "
                    "copied over,\n"
                    "  as we could not safely copy their implementation.\n"
                    "  See the comment at the top of the squashed migration for "
                    "details."
                )
                if shutil.which("black"):
                    self.stdout.write(
                        self.style.WARNING(
                            "Squashed migration couldn't be formatted using the "
                            '"black" command. You can call it manually.'
                        )
                    )
```
### 4 - django/db/migrations/autodetector.py:

Start line: 1216, End line: 1303

```python
class MigrationAutodetector:

    def create_altered_indexes(self):
        option_name = operations.AddIndex.option_name
        self.renamed_index_together_values = defaultdict(list)

        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_indexes = old_model_state.options[option_name]
            new_indexes = new_model_state.options[option_name]
            added_indexes = [idx for idx in new_indexes if idx not in old_indexes]
            removed_indexes = [idx for idx in old_indexes if idx not in new_indexes]
            renamed_indexes = []
            # Find renamed indexes.
            remove_from_added = []
            remove_from_removed = []
            for new_index in added_indexes:
                new_index_dec = new_index.deconstruct()
                new_index_name = new_index_dec[2].pop("name")
                for old_index in removed_indexes:
                    old_index_dec = old_index.deconstruct()
                    old_index_name = old_index_dec[2].pop("name")
                    # Indexes are the same except for the names.
                    if (
                        new_index_dec == old_index_dec
                        and new_index_name != old_index_name
                    ):
                        renamed_indexes.append((old_index_name, new_index_name, None))
                        remove_from_added.append(new_index)
                        remove_from_removed.append(old_index)
            # Find index_together changed to indexes.
            for (
                old_value,
                new_value,
                index_together_app_label,
                index_together_model_name,
                dependencies,
            ) in self._get_altered_foo_together_operations(
                operations.AlterIndexTogether.option_name
            ):
                if (
                    app_label != index_together_app_label
                    or model_name != index_together_model_name
                ):
                    continue
                removed_values = old_value.difference(new_value)
                for removed_index_together in removed_values:
                    renamed_index_together_indexes = []
                    for new_index in added_indexes:
                        _, args, kwargs = new_index.deconstruct()
                        # Ensure only 'fields' are defined in the Index.
                        if (
                            not args
                            and new_index.fields == list(removed_index_together)
                            and set(kwargs) == {"name", "fields"}
                        ):
                            renamed_index_together_indexes.append(new_index)

                    if len(renamed_index_together_indexes) == 1:
                        renamed_index = renamed_index_together_indexes[0]
                        remove_from_added.append(renamed_index)
                        renamed_indexes.append(
                            (None, renamed_index.name, removed_index_together)
                        )
                        self.renamed_index_together_values[
                            index_together_app_label, index_together_model_name
                        ].append(removed_index_together)
            # Remove renamed indexes from the lists of added and removed
            # indexes.
            added_indexes = [
                idx for idx in added_indexes if idx not in remove_from_added
            ]
            removed_indexes = [
                idx for idx in removed_indexes if idx not in remove_from_removed
            ]

            self.altered_indexes.update(
                {
                    (app_label, model_name): {
                        "added_indexes": added_indexes,
                        "removed_indexes": removed_indexes,
                        "renamed_indexes": renamed_indexes,
                    }
                }
            )
```
### 5 - django/core/management/commands/squashmigrations.py:

Start line: 62, End line: 159

```python
class Command(BaseCommand):

    def handle(self, **options):
        self.verbosity = options["verbosity"]
        self.interactive = options["interactive"]
        app_label = options["app_label"]
        start_migration_name = options["start_migration_name"]
        migration_name = options["migration_name"]
        no_optimize = options["no_optimize"]
        squashed_name = options["squashed_name"]
        include_header = options["include_header"]
        # Validate app_label.
        try:
            apps.get_app_config(app_label)
        except LookupError as err:
            raise CommandError(str(err))
        # Load the current graph state, check the app and migration they asked
        # for exists.
        loader = MigrationLoader(connections[DEFAULT_DB_ALIAS])
        if app_label not in loader.migrated_apps:
            raise CommandError(
                "App '%s' does not have migrations (so squashmigrations on "
                "it makes no sense)" % app_label
            )

        migration = self.find_migration(loader, app_label, migration_name)

        # Work out the list of predecessor migrations
        migrations_to_squash = [
            loader.get_migration(al, mn)
            for al, mn in loader.graph.forwards_plan(
                (migration.app_label, migration.name)
            )
            if al == migration.app_label
        ]

        if start_migration_name:
            start_migration = self.find_migration(
                loader, app_label, start_migration_name
            )
            start = loader.get_migration(
                start_migration.app_label, start_migration.name
            )
            try:
                start_index = migrations_to_squash.index(start)
                migrations_to_squash = migrations_to_squash[start_index:]
            except ValueError:
                raise CommandError(
                    "The migration '%s' cannot be found. Maybe it comes after "
                    "the migration '%s'?\n"
                    "Have a look at:\n"
                    "  python manage.py showmigrations %s\n"
                    "to debug this issue." % (start_migration, migration, app_label)
                )

        # Tell them what we're doing and optionally ask if we should proceed
        if self.verbosity > 0 or self.interactive:
            self.stdout.write(
                self.style.MIGRATE_HEADING("Will squash the following migrations:")
            )
            for migration in migrations_to_squash:
                self.stdout.write(" - %s" % migration.name)

            if self.interactive:
                answer = None
                while not answer or answer not in "yn":
                    answer = input("Do you wish to proceed? [yN] ")
                    if not answer:
                        answer = "n"
                        break
                    else:
                        answer = answer[0].lower()
                if answer != "y":
                    return

        # Load the operations from all those migrations and concat together,
        # along with collecting external dependencies and detecting
        # double-squashing
        operations = []
        dependencies = set()
        # We need to take all dependencies from the first migration in the list
        # as it may be 0002 depending on 0001
        first_migration = True
        for smigration in migrations_to_squash:
            if smigration.replaces:
                raise CommandError(
                    "You cannot squash squashed migrations! Please transition it to a "
                    "normal migration first: https://docs.djangoproject.com/en/%s/"
                    "topics/migrations/#squashing-migrations" % get_docs_version()
                )
            operations.extend(smigration.operations)
            for dependency in smigration.dependencies:
                if isinstance(dependency, SwappableTuple):
                    if settings.AUTH_USER_MODEL == dependency.setting:
                        dependencies.add(("__setting__", "AUTH_USER_MODEL"))
                    else:
                        dependencies.add(dependency)
                elif dependency[0] != smigration.app_label or first_migration:
                    dependencies.add(dependency)
            first_migration = False
        # ... other code
```
### 6 - django/db/migrations/autodetector.py:

Start line: 1495, End line: 1518

```python
class MigrationAutodetector:

    def _generate_removed_altered_foo_together(self, operation):
        for (
            old_value,
            new_value,
            app_label,
            model_name,
            dependencies,
        ) in self._get_altered_foo_together_operations(operation.option_name):
            if operation == operations.AlterIndexTogether:
                old_value = {
                    value
                    for value in old_value
                    if value
                    not in self.renamed_index_together_values[app_label, model_name]
                }
            removal_value = new_value.intersection(old_value)
            if removal_value or old_value:
                self.add_operation(
                    app_label,
                    operation(
                        name=model_name, **{operation.option_name: removal_value}
                    ),
                    dependencies=dependencies,
                )
```
### 7 - django/db/backends/base/schema.py:

Start line: 565, End line: 584

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
                {"index": True, "unique": False},
                self.sql_delete_index,
            )
        # Created indexes
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_index_sql(model, fields=fields, suffix="_idx"))
```
### 8 - django/utils/deprecation.py:

Start line: 41, End line: 83

```python
class RenameMethodsBase(type):
    """
    Handles the deprecation paths when renaming a method.

    It does the following:
        1) Define the new method if missing and complain about it.
        2) Define the old method if missing.
        3) Complain whenever an old method is called.

    See #15363 for more details.
    """

    renamed_methods = ()

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        for base in inspect.getmro(new_class):
            class_name = base.__name__
            for renamed_method in cls.renamed_methods:
                old_method_name = renamed_method[0]
                old_method = base.__dict__.get(old_method_name)
                new_method_name = renamed_method[1]
                new_method = base.__dict__.get(new_method_name)
                deprecation_warning = renamed_method[2]
                wrapper = warn_about_renamed_method(class_name, *renamed_method)

                # Define the new method if missing and complain about it
                if not new_method and old_method:
                    warnings.warn(
                        "`%s.%s` method should be renamed `%s`."
                        % (class_name, old_method_name, new_method_name),
                        deprecation_warning,
                        2,
                    )
                    setattr(base, new_method_name, old_method)
                    setattr(base, old_method_name, wrapper(old_method))

                # Define the old method as a wrapped call to the new method.
                if not old_method and new_method:
                    setattr(base, old_method_name, wrapper(new_method))

        return new_class
```
### 9 - django/utils/deprecation.py:

Start line: 86, End line: 137

```python
class DeprecationInstanceCheck(type):
    def __instancecheck__(self, instance):
        warnings.warn(
            "`%s` is deprecated, use `%s` instead." % (self.__name__, self.alternative),
            self.deprecation_warning,
            2,
        )
        return super().__instancecheck__(instance)


class MiddlewareMixin:
    sync_capable = True
    async_capable = True

    def __init__(self, get_response):
        if get_response is None:
            raise ValueError("get_response must be provided.")
        self.get_response = get_response
        self._async_check()
        super().__init__()

    def __repr__(self):
        return "<%s get_response=%s>" % (
            self.__class__.__qualname__,
            getattr(
                self.get_response,
                "__qualname__",
                self.get_response.__class__.__name__,
            ),
        )

    def _async_check(self):
        """
        If get_response is a coroutine function, turns us into async mode so
        a thread is not consumed during a whole request.
        """
        if iscoroutinefunction(self.get_response):
            # Mark the class as async-capable, but do the actual switch
            # inside __call__ to avoid swapping out dunder methods
            markcoroutinefunction(self)

    def __call__(self, request):
        # Exit out to async mode, if needed
        if iscoroutinefunction(self):
            return self.__acall__(request)
        response = None
        if hasattr(self, "process_request"):
            response = self.process_request(request)
        response = response or self.get_response(request)
        if hasattr(self, "process_response"):
            response = self.process_response(request, response)
        return response
```
### 10 - django/core/management/commands/migrate.py:

Start line: 191, End line: 269

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        # ... other code

        if options["prune"]:
            if not options["app_label"]:
                raise CommandError(
                    "Migrations can be pruned only when an app is specified."
                )
            if self.verbosity > 0:
                self.stdout.write("Pruning migrations:", self.style.MIGRATE_HEADING)
            to_prune = set(executor.loader.applied_migrations) - set(
                executor.loader.disk_migrations
            )
            squashed_migrations_with_deleted_replaced_migrations = [
                migration_key
                for migration_key, migration_obj in executor.loader.replacements.items()
                if any(replaced in to_prune for replaced in migration_obj.replaces)
            ]
            if squashed_migrations_with_deleted_replaced_migrations:
                self.stdout.write(
                    self.style.NOTICE(
                        "  Cannot use --prune because the following squashed "
                        "migrations have their 'replaces' attributes and may not "
                        "be recorded as applied:"
                    )
                )
                for migration in squashed_migrations_with_deleted_replaced_migrations:
                    app, name = migration
                    self.stdout.write(f"    {app}.{name}")
                self.stdout.write(
                    self.style.NOTICE(
                        "  Re-run 'manage.py migrate' if they are not marked as "
                        "applied, and remove 'replaces' attributes in their "
                        "Migration classes."
                    )
                )
            else:
                to_prune = sorted(
                    migration for migration in to_prune if migration[0] == app_label
                )
                if to_prune:
                    for migration in to_prune:
                        app, name = migration
                        if self.verbosity > 0:
                            self.stdout.write(
                                self.style.MIGRATE_LABEL(f"  Pruning {app}.{name}"),
                                ending="",
                            )
                        executor.recorder.record_unapplied(app, name)
                        if self.verbosity > 0:
                            self.stdout.write(self.style.SUCCESS(" OK"))
                elif self.verbosity > 0:
                    self.stdout.write("  No migrations to prune.")

        plan = executor.migration_plan(targets)

        if options["plan"]:
            self.stdout.write("Planned operations:", self.style.MIGRATE_LABEL)
            if not plan:
                self.stdout.write("  No planned migration operations.")
            else:
                for migration, backwards in plan:
                    self.stdout.write(str(migration), self.style.MIGRATE_HEADING)
                    for operation in migration.operations:
                        message, is_error = self.describe_operation(
                            operation, backwards
                        )
                        style = self.style.WARNING if is_error else None
                        self.stdout.write("    " + message, style)
                if options["check_unapplied"]:
                    sys.exit(1)
            return
        if options["check_unapplied"]:
            if plan:
                sys.exit(1)
            return
        if options["prune"]:
            return

        # At this point, ignore run_syncdb if there aren't any apps to sync.
        run_syncdb = options["run_syncdb"] and executor.loader.unmigrated_apps
        # Print some useful info
        # ... other code
```
### 11 - django/db/migrations/operations/models.py:

Start line: 1005, End line: 1022

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
### 20 - django/db/migrations/operations/models.py:

Start line: 968, End line: 1003

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
### 25 - django/db/migrations/operations/models.py:

Start line: 600, End line: 624

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
### 27 - django/db/migrations/operations/models.py:

Start line: 1024, End line: 1059

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
### 28 - django/db/migrations/operations/models.py:

Start line: 1103, End line: 1143

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
### 34 - django/db/migrations/operations/models.py:

Start line: 589, End line: 598

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
### 44 - django/db/migrations/operations/models.py:

Start line: 913, End line: 966

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
### 84 - django/db/migrations/operations/models.py:

Start line: 870, End line: 910

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
### 94 - django/db/migrations/operations/models.py:

Start line: 674, End line: 698

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
### 112 - django/db/migrations/operations/models.py:

Start line: 370, End line: 416

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
                # Rename columns and the M2M table.
                schema_editor._alter_many_to_many(
                    new_model,
                    old_field,
                    new_field,
                    strict=False,
                )
```
### 116 - django/db/migrations/operations/models.py:

Start line: 627, End line: 648

```python
class AlterUniqueTogether(AlterTogetherOptionOperation):
    """
    Change the value of unique_together to the target one.
    Input value of unique_together must be a set of tuples.
    """

    option_name = "unique_together"

    def __init__(self, name, unique_together):
        super().__init__(name, unique_together)


class AlterIndexTogether(AlterTogetherOptionOperation):
    """
    Change the value of index_together to the target one.
    Input value of index_together must be a set of tuples.
    """

    option_name = "index_together"

    def __init__(self, name, index_together):
        super().__init__(name, index_together)
```
### 119 - django/db/migrations/operations/models.py:

Start line: 1062, End line: 1100

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
### 128 - django/db/migrations/operations/models.py:

Start line: 700, End line: 716

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
### 137 - django/db/migrations/operations/models.py:

Start line: 562, End line: 587

```python
class AlterTogetherOptionOperation(ModelOptionOperation):
    option_name = None

    def __init__(self, name, option_value):
        if option_value:
            option_value = set(normalize_together(option_value))
        setattr(self, self.option_name, option_value)
        super().__init__(name)

    @cached_property
    def option_value(self):
        return getattr(self, self.option_name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            self.option_name: self.option_value,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(
            app_label,
            self.name_lower,
            {self.option_name: self.option_value},
        )
```
### 138 - django/db/migrations/operations/models.py:

Start line: 418, End line: 431

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
