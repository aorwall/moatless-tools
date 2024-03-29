# django__django-13924

| **django/django** | `7b3ec6bcc8309d5b2003d355fe6f78af89cfeb52` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 13839 |
| **Any found context length** | 13839 |
| **Avg pos** | 36.0 |
| **Min pos** | 36 |
| **Max pos** | 36 |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/executor.py b/django/db/migrations/executor.py
--- a/django/db/migrations/executor.py
+++ b/django/db/migrations/executor.py
@@ -225,8 +225,9 @@ def apply_migration(self, state, migration, fake=False, fake_initial=False):
                 # Alright, do it normally
                 with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
                     state = migration.apply(state, schema_editor)
-                    self.record_migration(migration)
-                    migration_recorded = True
+                    if not schema_editor.deferred_sql:
+                        self.record_migration(migration)
+                        migration_recorded = True
         if not migration_recorded:
             self.record_migration(migration)
         # Report progress

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/executor.py | 228 | 229 | 36 | 5 | 13839


## Problem Statement

```
Migrations are marked applied even if deferred SQL fails to execute
Description
	
The changes introduced in c86a3d80a25acd1887319198ca21a84c451014ad to address #29721 fail to account for the possibility of the schema editor accumulation of deferred SQL which is run at SchemaEditor.__exit__ time.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/backends/base/schema.py | 45 | 112| 769 | 769 | 12498 | 
| 2 | 1 django/db/backends/base/schema.py | 655 | 725| 796 | 1565 | 12498 | 
| 3 | 2 django/db/migrations/operations/special.py | 116 | 130| 139 | 1704 | 14056 | 
| 4 | 3 django/db/backends/mysql/schema.py | 1 | 38| 409 | 2113 | 15578 | 
| 5 | 4 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 2618 | 19734 | 
| 6 | 4 django/db/backends/base/schema.py | 726 | 804| 827 | 3445 | 19734 | 
| 7 | 4 django/db/backends/base/schema.py | 114 | 147| 322 | 3767 | 19734 | 
| 8 | 4 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 3948 | 19734 | 
| 9 | **5 django/db/migrations/executor.py** | 280 | 373| 843 | 4791 | 23007 | 
| 10 | 6 django/db/migrations/recorder.py | 1 | 21| 148 | 4939 | 23684 | 
| 11 | 7 django/db/backends/oracle/schema.py | 59 | 79| 249 | 5188 | 25719 | 
| 12 | 7 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 5919 | 25719 | 
| 13 | 7 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 6236 | 25719 | 
| 14 | 8 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 6883 | 27887 | 
| 15 | 9 django/db/migrations/autodetector.py | 1155 | 1189| 296 | 7179 | 39506 | 
| 16 | 9 django/db/backends/base/schema.py | 805 | 845| 519 | 7698 | 39506 | 
| 17 | 9 django/db/backends/sqlite3/schema.py | 386 | 435| 464 | 8162 | 39506 | 
| 18 | 10 django/db/migrations/migration.py | 92 | 127| 338 | 8500 | 41316 | 
| 19 | 10 django/db/backends/oracle/schema.py | 81 | 125| 583 | 9083 | 41316 | 
| 20 | 10 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 9434 | 41316 | 
| 21 | **10 django/db/migrations/executor.py** | 263 | 278| 165 | 9599 | 41316 | 
| 22 | 11 django/db/migrations/exceptions.py | 1 | 55| 249 | 9848 | 41566 | 
| 23 | 11 django/db/backends/oracle/schema.py | 127 | 136| 142 | 9990 | 41566 | 
| 24 | 11 django/db/backends/base/schema.py | 931 | 950| 296 | 10286 | 41566 | 
| 25 | 11 django/db/backends/mysql/schema.py | 40 | 49| 134 | 10420 | 41566 | 
| 26 | 12 django/db/migrations/loader.py | 288 | 312| 205 | 10625 | 44671 | 
| 27 | 12 django/db/backends/base/schema.py | 1133 | 1148| 170 | 10795 | 44671 | 
| 28 | 12 django/db/migrations/autodetector.py | 1191 | 1216| 245 | 11040 | 44671 | 
| 29 | 13 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 11419 | 45304 | 
| 30 | 13 django/db/migrations/migration.py | 129 | 178| 481 | 11900 | 45304 | 
| 31 | 14 django/contrib/gis/db/backends/spatialite/schema.py | 128 | 169| 376 | 12276 | 46656 | 
| 32 | 15 django/contrib/gis/db/backends/postgis/schema.py | 51 | 74| 195 | 12471 | 47329 | 
| 33 | 16 django/db/migrations/operations/fields.py | 236 | 246| 146 | 12617 | 50427 | 
| 34 | 16 django/db/migrations/autodetector.py | 913 | 994| 876 | 13493 | 50427 | 
| 35 | 16 django/db/backends/postgresql/schema.py | 227 | 239| 152 | 13645 | 50427 | 
| **-> 36 <-** | **16 django/db/migrations/executor.py** | 213 | 235| 194 | 13839 | 50427 | 
| 37 | 16 django/db/migrations/autodetector.py | 1232 | 1280| 436 | 14275 | 50427 | 
| 38 | 17 django/db/migrations/operations/models.py | 462 | 491| 302 | 14577 | 57404 | 
| 39 | 17 django/db/migrations/operations/models.py | 619 | 636| 163 | 14740 | 57404 | 
| 40 | 18 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 15060 | 57724 | 
| 41 | 18 django/db/migrations/autodetector.py | 1095 | 1130| 312 | 15372 | 57724 | 
| 42 | 18 django/db/migrations/autodetector.py | 262 | 333| 748 | 16120 | 57724 | 
| 43 | **18 django/db/migrations/executor.py** | 237 | 261| 227 | 16347 | 57724 | 
| 44 | 19 django/contrib/postgres/operations.py | 36 | 60| 197 | 16544 | 59583 | 
| 45 | 19 django/db/migrations/recorder.py | 46 | 97| 390 | 16934 | 59583 | 
| 46 | 19 django/db/migrations/loader.py | 156 | 182| 291 | 17225 | 59583 | 
| 47 | 19 django/db/migrations/operations/models.py | 523 | 532| 129 | 17354 | 59583 | 
| 48 | 19 django/db/migrations/operations/special.py | 181 | 204| 246 | 17600 | 59583 | 
| 49 | 19 django/db/backends/postgresql/schema.py | 212 | 225| 182 | 17782 | 59583 | 
| 50 | 19 django/db/migrations/operations/models.py | 534 | 551| 168 | 17950 | 59583 | 
| 51 | 19 django/db/migrations/autodetector.py | 1132 | 1153| 231 | 18181 | 59583 | 
| 52 | 19 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 18807 | 59583 | 
| 53 | 19 django/db/backends/oracle/schema.py | 1 | 41| 427 | 19234 | 59583 | 
| 54 | 19 django/db/migrations/operations/models.py | 601 | 617| 215 | 19449 | 59583 | 
| 55 | 19 django/db/backends/mysql/schema.py | 89 | 99| 138 | 19587 | 59583 | 
| 56 | 19 django/db/backends/base/schema.py | 1150 | 1172| 199 | 19786 | 59583 | 
| 57 | 20 django/core/management/commands/migrate.py | 169 | 251| 808 | 20594 | 62839 | 
| 58 | 21 django/db/models/sql/query.py | 716 | 751| 389 | 20983 | 85230 | 
| 59 | 21 django/db/migrations/autodetector.py | 356 | 370| 138 | 21121 | 85230 | 
| 60 | 21 django/db/backends/base/schema.py | 1262 | 1284| 173 | 21294 | 85230 | 
| 61 | 21 django/db/backends/base/schema.py | 1174 | 1204| 214 | 21508 | 85230 | 
| 62 | 21 django/db/migrations/operations/fields.py | 97 | 109| 130 | 21638 | 85230 | 
| 63 | 21 django/db/backends/base/schema.py | 284 | 305| 173 | 21811 | 85230 | 
| 64 | 21 django/db/migrations/loader.py | 207 | 286| 783 | 22594 | 85230 | 
| 65 | 21 django/db/migrations/autodetector.py | 1036 | 1052| 188 | 22782 | 85230 | 
| 66 | 21 django/contrib/gis/db/backends/postgis/schema.py | 1 | 19| 206 | 22988 | 85230 | 
| 67 | 21 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 23247 | 85230 | 
| 68 | 21 django/db/backends/oracle/schema.py | 43 | 57| 133 | 23380 | 85230 | 
| 69 | 21 django/db/backends/base/schema.py | 1243 | 1260| 142 | 23522 | 85230 | 
| 70 | 21 django/db/migrations/operations/fields.py | 111 | 121| 127 | 23649 | 85230 | 
| 71 | 22 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 23844 | 85425 | 
| 72 | 22 django/db/backends/base/schema.py | 1 | 29| 209 | 24053 | 85425 | 
| 73 | 22 django/core/management/commands/migrate.py | 1 | 18| 140 | 24193 | 85425 | 
| 74 | 22 django/db/migrations/autodetector.py | 1054 | 1074| 136 | 24329 | 85425 | 
| 75 | 23 django/contrib/gis/db/backends/mysql/schema.py | 25 | 38| 146 | 24475 | 86056 | 
| 76 | 23 django/db/migrations/operations/fields.py | 248 | 270| 188 | 24663 | 86056 | 
| 77 | 23 django/db/backends/mysql/schema.py | 51 | 87| 349 | 25012 | 86056 | 
| 78 | 23 django/core/management/commands/migrate.py | 71 | 167| 834 | 25846 | 86056 | 
| 79 | 23 django/db/migrations/autodetector.py | 1218 | 1230| 131 | 25977 | 86056 | 
| 80 | 24 django/db/migrations/graph.py | 259 | 280| 179 | 26156 | 88659 | 
| 81 | 24 django/db/migrations/loader.py | 337 | 354| 152 | 26308 | 88659 | 
| 82 | 25 django/conf/__init__.py | 1 | 30| 179 | 26487 | 90390 | 
| 83 | 25 django/db/backends/base/schema.py | 423 | 437| 174 | 26661 | 90390 | 
| 84 | 25 django/core/management/commands/migrate.py | 272 | 304| 349 | 27010 | 90390 | 
| 85 | 26 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 27859 | 91239 | 
| 86 | 27 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 28021 | 91401 | 
| 87 | 28 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 28843 | 94240 | 
| 88 | 29 django/contrib/gis/db/backends/oracle/schema.py | 34 | 58| 229 | 29072 | 95072 | 
| 89 | 29 django/db/migrations/autodetector.py | 237 | 261| 267 | 29339 | 95072 | 
| 90 | 30 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 30130 | 96945 | 
| 91 | 31 django/db/models/sql/compiler.py | 1549 | 1589| 409 | 30539 | 111336 | 
| 92 | 31 django/core/management/commands/makemigrations.py | 239 | 326| 873 | 31412 | 111336 | 
| 93 | 31 django/db/migrations/autodetector.py | 996 | 1012| 188 | 31600 | 111336 | 
| 94 | 31 django/db/migrations/operations/fields.py | 216 | 234| 185 | 31785 | 111336 | 
| 95 | 32 django/core/management/sql.py | 35 | 50| 116 | 31901 | 111665 | 
| 96 | 32 django/contrib/gis/db/backends/spatialite/schema.py | 1 | 35| 316 | 32217 | 111665 | 
| 97 | 33 django/db/backends/ddl_references.py | 1 | 39| 218 | 32435 | 113287 | 
| 98 | 34 django/db/migrations/questioner.py | 227 | 240| 123 | 32558 | 115360 | 
| 99 | 34 django/db/backends/base/schema.py | 592 | 654| 700 | 33258 | 115360 | 
| 100 | 34 django/db/backends/base/schema.py | 1114 | 1131| 176 | 33434 | 115360 | 
| 101 | 34 django/db/backends/base/schema.py | 402 | 421| 199 | 33633 | 115360 | 
| 102 | 34 django/db/migrations/operations/special.py | 63 | 114| 390 | 34023 | 115360 | 
| 103 | 34 django/db/models/sql/compiler.py | 1089 | 1118| 253 | 34276 | 115360 | 
| 104 | 34 django/contrib/gis/db/backends/mysql/schema.py | 1 | 23| 203 | 34479 | 115360 | 
| 105 | 34 django/contrib/postgres/operations.py | 1 | 34| 258 | 34737 | 115360 | 
| 106 | 34 django/db/backends/mysql/schema.py | 101 | 114| 148 | 34885 | 115360 | 
| 107 | 34 django/db/backends/mysql/schema.py | 134 | 152| 192 | 35077 | 115360 | 
| 108 | 34 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 35267 | 115360 | 
| 109 | 34 django/db/backends/base/schema.py | 1206 | 1241| 291 | 35558 | 115360 | 
| 110 | 34 django/db/migrations/autodetector.py | 35 | 45| 120 | 35678 | 115360 | 
| 111 | 34 django/db/migrations/operations/models.py | 343 | 392| 493 | 36171 | 115360 | 
| 112 | 34 django/contrib/gis/db/backends/postgis/schema.py | 21 | 49| 292 | 36463 | 115360 | 
| 113 | 35 django/db/models/base.py | 1 | 50| 328 | 36791 | 132376 | 
| 114 | 35 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 37611 | 132376 | 
| 115 | 35 django/db/migrations/operations/special.py | 44 | 60| 180 | 37791 | 132376 | 
| 116 | 35 django/db/migrations/autodetector.py | 1282 | 1317| 311 | 38102 | 132376 | 
| 117 | 36 django/db/migrations/__init__.py | 1 | 3| 0 | 38102 | 132400 | 
| 118 | 36 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 38524 | 132400 | 
| 119 | 36 django/db/migrations/autodetector.py | 1014 | 1034| 134 | 38658 | 132400 | 
| 120 | 37 django/db/migrations/utils.py | 1 | 18| 0 | 38658 | 132488 | 
| 121 | 37 django/db/migrations/autodetector.py | 335 | 354| 196 | 38854 | 132488 | 
| 122 | 37 django/db/migrations/autodetector.py | 1076 | 1093| 180 | 39034 | 132488 | 
| 123 | 37 django/db/migrations/autodetector.py | 526 | 681| 1240 | 40274 | 132488 | 
| 124 | 37 django/db/migrations/graph.py | 99 | 120| 192 | 40466 | 132488 | 
| 125 | 37 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 40621 | 132488 | 
| 126 | 37 django/db/migrations/autodetector.py | 435 | 461| 256 | 40877 | 132488 | 
| 127 | 37 django/db/backends/base/schema.py | 386 | 400| 182 | 41059 | 132488 | 
| 128 | 37 django/db/backends/base/schema.py | 900 | 929| 237 | 41296 | 132488 | 
| 129 | 37 django/contrib/gis/db/backends/oracle/schema.py | 60 | 95| 297 | 41593 | 132488 | 
| 130 | 37 django/contrib/gis/db/backends/spatialite/schema.py | 37 | 61| 206 | 41799 | 132488 | 
| 131 | 37 django/contrib/postgres/operations.py | 191 | 212| 207 | 42006 | 132488 | 
| 132 | 37 django/contrib/postgres/operations.py | 63 | 118| 250 | 42256 | 132488 | 
| 133 | 37 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 42910 | 132488 | 
| 134 | 38 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 43021 | 132599 | 
| 135 | 39 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 43238 | 132816 | 
| 136 | 39 django/db/backends/mysql/schema.py | 116 | 132| 205 | 43443 | 132816 | 
| 137 | 39 django/db/migrations/operations/fields.py | 346 | 381| 335 | 43778 | 132816 | 
| 138 | 39 django/db/migrations/operations/models.py | 639 | 695| 366 | 44144 | 132816 | 
| 139 | 40 django/db/migrations/state.py | 153 | 163| 132 | 44276 | 137919 | 
| 140 | 40 django/contrib/gis/db/backends/oracle/schema.py | 1 | 32| 325 | 44601 | 137919 | 
| 141 | 40 django/db/backends/base/schema.py | 439 | 460| 234 | 44835 | 137919 | 
| 142 | 41 django/core/serializers/base.py | 232 | 249| 208 | 45043 | 140344 | 
| 143 | **41 django/db/migrations/executor.py** | 127 | 150| 235 | 45278 | 140344 | 
| 144 | 41 django/db/backends/base/schema.py | 869 | 898| 238 | 45516 | 140344 | 
| 145 | 42 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 45823 | 140651 | 
| 146 | 42 django/core/management/commands/migrate.py | 21 | 69| 407 | 46230 | 140651 | 
| 147 | 43 django/core/management/base.py | 479 | 511| 281 | 46511 | 145288 | 
| 148 | 43 django/core/management/commands/migrate.py | 253 | 270| 208 | 46719 | 145288 | 
| 149 | 43 django/db/backends/base/schema.py | 1316 | 1348| 292 | 47011 | 145288 | 
| 150 | 44 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 47148 | 145425 | 
| 151 | 44 django/db/migrations/autodetector.py | 101 | 196| 769 | 47917 | 145425 | 
| 152 | 45 django/db/migrations/writer.py | 2 | 115| 886 | 48803 | 147672 | 
| 153 | 45 django/db/migrations/graph.py | 61 | 97| 337 | 49140 | 147672 | 
| 154 | 45 django/db/models/sql/query.py | 667 | 714| 511 | 49651 | 147672 | 
| 155 | 46 django/contrib/postgres/constraints.py | 108 | 127| 155 | 49806 | 149097 | 
| 156 | 46 django/db/backends/base/schema.py | 549 | 590| 489 | 50295 | 149097 | 
| 157 | 47 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 50502 | 149304 | 
| 158 | 47 django/db/backends/base/schema.py | 847 | 867| 191 | 50693 | 149304 | 
| 159 | 47 django/db/migrations/operations/fields.py | 85 | 95| 124 | 50817 | 149304 | 
| 160 | 47 django/db/backends/base/schema.py | 981 | 1002| 196 | 51013 | 149304 | 
| 161 | 47 django/db/migrations/writer.py | 201 | 301| 619 | 51632 | 149304 | 
| 162 | 48 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 51790 | 150490 | 
| 163 | 49 django/contrib/gis/db/models/functions.py | 462 | 490| 225 | 52015 | 154435 | 
| 164 | 50 django/db/migrations/operations/base.py | 1 | 109| 804 | 52819 | 155465 | 
| 165 | 50 django/db/backends/base/schema.py | 261 | 282| 154 | 52973 | 155465 | 
| 166 | 50 django/db/migrations/graph.py | 282 | 298| 159 | 53132 | 155465 | 
| 167 | 50 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 53375 | 155465 | 
| 168 | 51 django/db/backends/mysql/compiler.py | 1 | 14| 123 | 53498 | 155985 | 
| 169 | 51 django/core/management/commands/makemigrations.py | 154 | 192| 313 | 53811 | 155985 | 
| 170 | 51 django/db/migrations/state.py | 589 | 605| 146 | 53957 | 155985 | 
| 171 | 51 django/db/migrations/loader.py | 68 | 132| 551 | 54508 | 155985 | 
| 172 | 51 django/db/migrations/recorder.py | 23 | 44| 145 | 54653 | 155985 | 
| 173 | 52 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 54 | 70| 109 | 54762 | 156538 | 
| 174 | 52 django/db/migrations/loader.py | 184 | 205| 213 | 54975 | 156538 | 
| 175 | 52 django/db/migrations/state.py | 490 | 522| 250 | 55225 | 156538 | 
| 176 | 53 django/db/backends/oracle/operations.py | 408 | 459| 516 | 55741 | 162504 | 
| 177 | 53 django/db/backends/mysql/compiler.py | 41 | 63| 176 | 55917 | 162504 | 
| 178 | **53 django/db/migrations/executor.py** | 152 | 211| 567 | 56484 | 162504 | 
| 179 | 53 django/db/models/sql/compiler.py | 1341 | 1400| 617 | 57101 | 162504 | 
| 180 | 54 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 57101 | 162581 | 
| 181 | 54 django/db/migrations/state.py | 165 | 189| 213 | 57314 | 162581 | 
| 182 | 54 django/db/migrations/operations/models.py | 394 | 414| 213 | 57527 | 162581 | 
| 183 | 55 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 57527 | 162656 | 
| 184 | 56 django/db/migrations/serializer.py | 143 | 162| 223 | 57750 | 165327 | 
| 185 | 57 django/contrib/postgres/apps.py | 43 | 70| 250 | 58000 | 165923 | 
| 186 | 57 django/db/migrations/loader.py | 1 | 53| 409 | 58409 | 165923 | 
| 187 | 57 django/db/models/sql/compiler.py | 1047 | 1087| 337 | 58746 | 165923 | 
| 188 | 57 django/contrib/gis/db/backends/spatialite/schema.py | 104 | 126| 212 | 58958 | 165923 | 
| 189 | 57 django/db/models/sql/query.py | 848 | 885| 383 | 59341 | 165923 | 
| 190 | 57 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 59559 | 165923 | 
| 191 | 57 django/contrib/gis/db/backends/spatialite/schema.py | 63 | 82| 133 | 59692 | 165923 | 
| 192 | 57 django/db/models/sql/compiler.py | 1402 | 1420| 203 | 59895 | 165923 | 
| 193 | 57 django/db/backends/oracle/schema.py | 138 | 198| 544 | 60439 | 165923 | 
| 194 | 57 django/db/models/sql/compiler.py | 49 | 61| 155 | 60594 | 165923 | 
| 195 | 57 django/db/migrations/operations/models.py | 879 | 918| 378 | 60972 | 165923 | 
| 196 | 57 django/db/migrations/operations/base.py | 111 | 141| 229 | 61201 | 165923 | 
| 197 | 57 django/db/migrations/migration.py | 180 | 214| 263 | 61464 | 165923 | 
| 198 | 58 django/db/backends/oracle/features.py | 1 | 124| 1048 | 62512 | 166971 | 
| 199 | 59 django/db/backends/mysql/features.py | 56 | 104| 453 | 62965 | 168943 | 
| 200 | 59 django/core/management/commands/makemigrations.py | 194 | 237| 434 | 63399 | 168943 | 
| 201 | **59 django/db/migrations/executor.py** | 64 | 80| 168 | 63567 | 168943 | 
| 202 | 59 django/contrib/gis/db/backends/mysql/schema.py | 65 | 78| 121 | 63688 | 168943 | 
| 203 | 59 django/contrib/postgres/apps.py | 1 | 20| 158 | 63846 | 168943 | 
| 204 | 59 django/db/migrations/migration.py | 1 | 90| 734 | 64580 | 168943 | 
| 205 | 59 django/db/migrations/state.py | 105 | 151| 367 | 64947 | 168943 | 
| 206 | 59 django/db/migrations/operations/models.py | 843 | 876| 308 | 65255 | 168943 | 
| 207 | 59 django/db/migrations/autodetector.py | 222 | 235| 197 | 65452 | 168943 | 
| 208 | 59 django/db/migrations/loader.py | 314 | 335| 208 | 65660 | 168943 | 


## Patch

```diff
diff --git a/django/db/migrations/executor.py b/django/db/migrations/executor.py
--- a/django/db/migrations/executor.py
+++ b/django/db/migrations/executor.py
@@ -225,8 +225,9 @@ def apply_migration(self, state, migration, fake=False, fake_initial=False):
                 # Alright, do it normally
                 with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
                     state = migration.apply(state, schema_editor)
-                    self.record_migration(migration)
-                    migration_recorded = True
+                    if not schema_editor.deferred_sql:
+                        self.record_migration(migration)
+                        migration_recorded = True
         if not migration_recorded:
             self.record_migration(migration)
         # Report progress

```

## Test Patch

```diff
diff --git a/tests/migrations/test_executor.py b/tests/migrations/test_executor.py
--- a/tests/migrations/test_executor.py
+++ b/tests/migrations/test_executor.py
@@ -1,11 +1,12 @@
 from unittest import mock
 
 from django.apps.registry import apps as global_apps
-from django.db import DatabaseError, connection
+from django.db import DatabaseError, connection, migrations, models
 from django.db.migrations.exceptions import InvalidMigrationPlan
 from django.db.migrations.executor import MigrationExecutor
 from django.db.migrations.graph import MigrationGraph
 from django.db.migrations.recorder import MigrationRecorder
+from django.db.migrations.state import ProjectState
 from django.test import (
     SimpleTestCase, modify_settings, override_settings, skipUnlessDBFeature,
 )
@@ -655,18 +656,60 @@ def test_migrate_marks_replacement_applied_even_if_it_did_nothing(self):
     # When the feature is False, the operation and the record won't be
     # performed in a transaction and the test will systematically pass.
     @skipUnlessDBFeature('can_rollback_ddl')
-    @override_settings(MIGRATION_MODULES={'migrations': 'migrations.test_migrations'})
     def test_migrations_applied_and_recorded_atomically(self):
         """Migrations are applied and recorded atomically."""
+        class Migration(migrations.Migration):
+            operations = [
+                migrations.CreateModel('model', [
+                    ('id', models.AutoField(primary_key=True)),
+                ]),
+            ]
+
         executor = MigrationExecutor(connection)
         with mock.patch('django.db.migrations.executor.MigrationExecutor.record_migration') as record_migration:
             record_migration.side_effect = RuntimeError('Recording migration failed.')
             with self.assertRaisesMessage(RuntimeError, 'Recording migration failed.'):
+                executor.apply_migration(
+                    ProjectState(),
+                    Migration('0001_initial', 'record_migration'),
+                )
                 executor.migrate([('migrations', '0001_initial')])
         # The migration isn't recorded as applied since it failed.
         migration_recorder = MigrationRecorder(connection)
-        self.assertFalse(migration_recorder.migration_qs.filter(app='migrations', name='0001_initial').exists())
-        self.assertTableNotExists('migrations_author')
+        self.assertIs(
+            migration_recorder.migration_qs.filter(
+                app='record_migration', name='0001_initial',
+            ).exists(),
+            False,
+        )
+        self.assertTableNotExists('record_migration_model')
+
+    def test_migrations_not_applied_on_deferred_sql_failure(self):
+        """Migrations are not recorded if deferred SQL application fails."""
+        class DeferredSQL:
+            def __str__(self):
+                raise DatabaseError('Failed to apply deferred SQL')
+
+        class Migration(migrations.Migration):
+            atomic = False
+
+            def apply(self, project_state, schema_editor, collect_sql=False):
+                schema_editor.deferred_sql.append(DeferredSQL())
+
+        executor = MigrationExecutor(connection)
+        with self.assertRaisesMessage(DatabaseError, 'Failed to apply deferred SQL'):
+            executor.apply_migration(
+                ProjectState(),
+                Migration('0001_initial', 'deferred_sql'),
+            )
+        # The migration isn't recorded as applied since it failed.
+        migration_recorder = MigrationRecorder(connection)
+        self.assertIs(
+            migration_recorder.migration_qs.filter(
+                app='deferred_sql', name='0001_initial',
+            ).exists(),
+            False,
+        )
 
 
 class FakeLoader:

```


## Code snippets

### 1 - django/db/backends/base/schema.py:

Start line: 45, End line: 112

```python
class BaseDatabaseSchemaEditor:
    """
    This class and its subclasses are responsible for emitting schema-changing
    statements to the databases - model creation/removal/alteration, field
    renaming, index fiddling, and so on.
    """

    # Overrideable SQL templates
    sql_create_table = "CREATE TABLE %(table)s (%(definition)s)"
    sql_rename_table = "ALTER TABLE %(old_table)s RENAME TO %(new_table)s"
    sql_retablespace_table = "ALTER TABLE %(table)s SET TABLESPACE %(new_tablespace)s"
    sql_delete_table = "DROP TABLE %(table)s CASCADE"

    sql_create_column = "ALTER TABLE %(table)s ADD COLUMN %(column)s %(definition)s"
    sql_alter_column = "ALTER TABLE %(table)s %(changes)s"
    sql_alter_column_type = "ALTER COLUMN %(column)s TYPE %(type)s"
    sql_alter_column_null = "ALTER COLUMN %(column)s DROP NOT NULL"
    sql_alter_column_not_null = "ALTER COLUMN %(column)s SET NOT NULL"
    sql_alter_column_default = "ALTER COLUMN %(column)s SET DEFAULT %(default)s"
    sql_alter_column_no_default = "ALTER COLUMN %(column)s DROP DEFAULT"
    sql_alter_column_collate = "ALTER COLUMN %(column)s TYPE %(type)s%(collation)s"
    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s CASCADE"
    sql_rename_column = "ALTER TABLE %(table)s RENAME COLUMN %(old_column)s TO %(new_column)s"
    sql_update_with_default = "UPDATE %(table)s SET %(column)s = %(default)s WHERE %(column)s IS NULL"

    sql_unique_constraint = "UNIQUE (%(columns)s)%(deferrable)s"
    sql_check_constraint = "CHECK (%(check)s)"
    sql_delete_constraint = "ALTER TABLE %(table)s DROP CONSTRAINT %(name)s"
    sql_constraint = "CONSTRAINT %(name)s %(constraint)s"

    sql_create_check = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s CHECK (%(check)s)"
    sql_delete_check = sql_delete_constraint

    sql_create_unique = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s UNIQUE (%(columns)s)%(deferrable)s"
    sql_delete_unique = sql_delete_constraint

    sql_create_fk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s FOREIGN KEY (%(column)s) "
        "REFERENCES %(to_table)s (%(to_column)s)%(deferrable)s"
    )
    sql_create_inline_fk = None
    sql_create_column_inline_fk = None
    sql_delete_fk = sql_delete_constraint

    sql_create_index = "CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(include)s%(extra)s%(condition)s"
    sql_create_unique_index = "CREATE UNIQUE INDEX %(name)s ON %(table)s (%(columns)s)%(include)s%(condition)s"
    sql_delete_index = "DROP INDEX %(name)s"

    sql_create_pk = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s PRIMARY KEY (%(columns)s)"
    sql_delete_pk = sql_delete_constraint

    sql_delete_procedure = 'DROP PROCEDURE %(procedure)s'

    def __init__(self, connection, collect_sql=False, atomic=True):
        self.connection = connection
        self.collect_sql = collect_sql
        if self.collect_sql:
            self.collected_sql = []
        self.atomic_migration = self.connection.features.can_rollback_ddl and atomic

    # State-managing methods

    def __enter__(self):
        self.deferred_sql = []
        if self.atomic_migration:
            self.atomic = atomic(self.connection.alias)
            self.atomic.__enter__()
        return self
```
### 2 - django/db/backends/base/schema.py:

Start line: 655, End line: 725

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
        # Collation change?
        old_collation = getattr(old_field, 'db_collation', None)
        new_collation = getattr(new_field, 'db_collation', None)
        if old_collation != new_collation:
            # Collation change handles also a type change.
            fragment = self._alter_column_collation_sql(model, new_field, new_type, new_collation)
            actions.append(fragment)
        # Type change?
        elif old_type != new_type:
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
        # ... other code
```
### 3 - django/db/migrations/operations/special.py:

Start line: 116, End line: 130

```python
class RunSQL(Operation):

    def _run_sql(self, schema_editor, sqls):
        if isinstance(sqls, (list, tuple)):
            for sql in sqls:
                params = None
                if isinstance(sql, (list, tuple)):
                    elements = len(sql)
                    if elements == 2:
                        sql, params = sql
                    else:
                        raise ValueError("Expected a 2-tuple but got %d" % elements)
                schema_editor.execute(sql, params=params)
        elif sqls != RunSQL.noop:
            statements = schema_editor.connection.ops.prepare_sql_script(sqls)
            for statement in statements:
                schema_editor.execute(statement, params=None)
```
### 4 - django/db/backends/mysql/schema.py:

Start line: 1, End line: 38

```python
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    sql_rename_table = "RENAME TABLE %(old_table)s TO %(new_table)s"

    sql_alter_column_null = "MODIFY %(column)s %(type)s NULL"
    sql_alter_column_not_null = "MODIFY %(column)s %(type)s NOT NULL"
    sql_alter_column_type = "MODIFY %(column)s %(type)s"
    sql_alter_column_collate = "MODIFY %(column)s %(type)s%(collation)s"

    # No 'CASCADE' which works as a no-op in MySQL but is undocumented
    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s"

    sql_delete_unique = "ALTER TABLE %(table)s DROP INDEX %(name)s"
    sql_create_column_inline_fk = (
        ', ADD CONSTRAINT %(name)s FOREIGN KEY (%(column)s) '
        'REFERENCES %(to_table)s(%(to_column)s)'
    )
    sql_delete_fk = "ALTER TABLE %(table)s DROP FOREIGN KEY %(name)s"

    sql_delete_index = "DROP INDEX %(name)s ON %(table)s"

    sql_create_pk = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s PRIMARY KEY (%(columns)s)"
    sql_delete_pk = "ALTER TABLE %(table)s DROP PRIMARY KEY"

    sql_create_index = 'CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(extra)s'

    @property
    def sql_delete_check(self):
        if self.connection.mysql_is_mariadb:
            # The name of the column check constraint is the same as the field
            # name on MariaDB. Adding IF EXISTS clause prevents migrations
            # crash. Constraint is removed during a "MODIFY" column statement.
            return 'ALTER TABLE %(table)s DROP CONSTRAINT IF EXISTS %(name)s'
        return 'ALTER TABLE %(table)s DROP CHECK %(name)s'
```
### 5 - django/db/backends/sqlite3/schema.py:

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
### 6 - django/db/backends/base/schema.py:

Start line: 726, End line: 804

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        # ... other code
        if old_field.null != new_field.null:
            fragment = self._alter_column_null_sql(model, old_field, new_field)
            if fragment:
                null_actions.append(fragment)
        # Only if we have a default and there is a change from NULL to NOT NULL
        four_way_default_alteration = (
            new_field.has_default() and
            (old_field.null and not new_field.null)
        )
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
            self.execute(self._create_index_sql(model, fields=[new_field]))
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
### 7 - django/db/backends/base/schema.py:

Start line: 114, End line: 147

```python
class BaseDatabaseSchemaEditor:

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for sql in self.deferred_sql:
                self.execute(sql)
        if self.atomic_migration:
            self.atomic.__exit__(exc_type, exc_value, traceback)

    # Core utility functions

    def execute(self, sql, params=()):
        """Execute the given SQL statement, with optional parameters."""
        # Don't perform the transactional DDL check if SQL is being collected
        # as it's not going to be executed anyway.
        if not self.collect_sql and self.connection.in_atomic_block and not self.connection.features.can_rollback_ddl:
            raise TransactionManagementError(
                "Executing DDL statements while in a transaction on databases "
                "that can't perform a rollback is prohibited."
            )
        # Account for non-string statement objects.
        sql = str(sql)
        # Log the command we're running, then run it
        logger.debug("%s; (params %r)", sql, params, extra={'params': params, 'sql': sql})
        if self.collect_sql:
            ending = "" if sql.endswith(";") else ";"
            if params is not None:
                self.collected_sql.append((sql % tuple(map(self.quote_value, params))) + ending)
            else:
                self.collected_sql.append(sql + ending)
        else:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)

    def quote_name(self, name):
        return self.connection.ops.quote_name(name)
```
### 8 - django/db/backends/sqlite3/schema.py:

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
### 9 - django/db/migrations/executor.py:

Start line: 280, End line: 373

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
        fold_identifier_case = self.connection.features.ignores_table_name_case
        with self.connection.cursor() as cursor:
            existing_table_names = set(self.connection.introspection.table_names(cursor))
            if fold_identifier_case:
                existing_table_names = {name.casefold() for name in existing_table_names}
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
                db_table = model._meta.db_table
                if fold_identifier_case:
                    db_table = db_table.casefold()
                if db_table not in existing_table_names:
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
                    through_db_table = field.remote_field.through._meta.db_table
                    if fold_identifier_case:
                        through_db_table = through_db_table.casefold()
                    if through_db_table not in existing_table_names:
                        return False, project_state
                    else:
                        found_add_field_migration = True
                        continue
                with self.connection.cursor() as cursor:
                    columns = self.connection.introspection.get_table_description(cursor, table)
                for column in columns:
                    field_column = field.column
                    column_name = column.name
                    if fold_identifier_case:
                        column_name = column_name.casefold()
                        field_column = field_column.casefold()
                    if column_name == field_column:
                        found_add_field_migration = True
                        break
                else:
                    return False, project_state
        # If we get this far and we found at least one CreateModel or AddField migration,
        # the migration is considered implicitly applied.
        return (found_create_model_migration or found_add_field_migration), after_state
```
### 10 - django/db/migrations/recorder.py:

Start line: 1, End line: 21

```python
from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now

from .exceptions import MigrationSchemaMissing


class MigrationRecorder:
    """
    Deal with storing migration records in the database.

    Because this table is actually itself used for dealing with model
    creation, it's the one thing we can't do normally via migrations.
    We manually handle table creation/schema updating (using schema backend)
    and then have a floating model to do queries with.

    If a migration is unapplied its row is removed from the table. Having
    a row in the table always means a migration is applied.
    """
    _migration_class = None
```
### 21 - django/db/migrations/executor.py:

Start line: 263, End line: 278

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
### 36 - django/db/migrations/executor.py:

Start line: 213, End line: 235

```python
class MigrationExecutor:

    def apply_migration(self, state, migration, fake=False, fake_initial=False):
        """Run a migration forwards."""
        migration_recorded = False
        if self.progress_callback:
            self.progress_callback("apply_start", migration, fake)
        if not fake:
            if fake_initial:
                # Test to see if this is an already-applied initial migration
                applied, state = self.detect_soft_applied(state, migration)
                if applied:
                    fake = True
            if not fake:
                # Alright, do it normally
                with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
                    state = migration.apply(state, schema_editor)
                    self.record_migration(migration)
                    migration_recorded = True
        if not migration_recorded:
            self.record_migration(migration)
        # Report progress
        if self.progress_callback:
            self.progress_callback("apply_success", migration, fake)
        return state
```
### 43 - django/db/migrations/executor.py:

Start line: 237, End line: 261

```python
class MigrationExecutor:

    def record_migration(self, migration):
        # For replacement migrations, record individual statuses
        if migration.replaces:
            for app_label, name in migration.replaces:
                self.recorder.record_applied(app_label, name)
        else:
            self.recorder.record_applied(migration.app_label, migration.name)

    def unapply_migration(self, state, migration, fake=False):
        """Run a migration backwards."""
        if self.progress_callback:
            self.progress_callback("unapply_start", migration, fake)
        if not fake:
            with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
                state = migration.unapply(state, schema_editor)
        # For replacement migrations, record individual statuses
        if migration.replaces:
            for app_label, name in migration.replaces:
                self.recorder.record_unapplied(app_label, name)
        else:
            self.recorder.record_unapplied(migration.app_label, migration.name)
        # Report progress
        if self.progress_callback:
            self.progress_callback("unapply_success", migration, fake)
        return state
```
### 143 - django/db/migrations/executor.py:

Start line: 127, End line: 150

```python
class MigrationExecutor:

    def _migrate_all_forwards(self, state, plan, full_plan, fake, fake_initial):
        """
        Take a list of 2-tuples of the form (migration instance, False) and
        apply them in the order they occur in the full_plan.
        """
        migrations_to_run = {m[0] for m in plan}
        for migration, _ in full_plan:
            if not migrations_to_run:
                # We remove every migration that we applied from these sets so
                # that we can bail out once the last migration has been applied
                # and don't always run until the very end of the migration
                # process.
                break
            if migration in migrations_to_run:
                if 'apps' not in state.__dict__:
                    if self.progress_callback:
                        self.progress_callback("render_start")
                    state.apps  # Render all -- performance critical
                    if self.progress_callback:
                        self.progress_callback("render_success")
                state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial)
                migrations_to_run.remove(migration)

        return state
```
### 178 - django/db/migrations/executor.py:

Start line: 152, End line: 211

```python
class MigrationExecutor:

    def _migrate_all_backwards(self, plan, full_plan, fake):
        """
        Take a list of 2-tuples of the form (migration instance, True) and
        unapply them in reverse order they occur in the full_plan.

        Since unapplying a migration requires the project state prior to that
        migration, Django will compute the migration states before each of them
        in a first run over the plan and then unapply them in a second run over
        the plan.
        """
        migrations_to_run = {m[0] for m in plan}
        # Holds all migration states prior to the migrations being unapplied
        states = {}
        state = self._create_project_state()
        applied_migrations = {
            self.loader.graph.nodes[key] for key in self.loader.applied_migrations
            if key in self.loader.graph.nodes
        }
        if self.progress_callback:
            self.progress_callback("render_start")
        for migration, _ in full_plan:
            if not migrations_to_run:
                # We remove every migration that we applied from this set so
                # that we can bail out once the last migration has been applied
                # and don't always run until the very end of the migration
                # process.
                break
            if migration in migrations_to_run:
                if 'apps' not in state.__dict__:
                    state.apps  # Render all -- performance critical
                # The state before this migration
                states[migration] = state
                # The old state keeps as-is, we continue with the new state
                state = migration.mutate_state(state, preserve=True)
                migrations_to_run.remove(migration)
            elif migration in applied_migrations:
                # Only mutate the state if the migration is actually applied
                # to make sure the resulting state doesn't include changes
                # from unrelated migrations.
                migration.mutate_state(state, preserve=False)
        if self.progress_callback:
            self.progress_callback("render_success")

        for migration, _ in plan:
            self.unapply_migration(states[migration], migration, fake=fake)
            applied_migrations.remove(migration)

        # Generate the post migration state by starting from the state before
        # the last migration is unapplied and mutating it to include all the
        # remaining applied migrations.
        last_unapplied_migration = plan[-1][0]
        state = states[last_unapplied_migration]
        for index, (migration, _) in enumerate(full_plan):
            if migration == last_unapplied_migration:
                for migration, _ in full_plan[index:]:
                    if migration in applied_migrations:
                        migration.mutate_state(state, preserve=False)
                break

        return state
```
### 201 - django/db/migrations/executor.py:

Start line: 64, End line: 80

```python
class MigrationExecutor:

    def _create_project_state(self, with_applied_migrations=False):
        """
        Create a project state including all the applications without
        migrations and applied migrations if with_applied_migrations=True.
        """
        state = ProjectState(real_apps=list(self.loader.unmigrated_apps))
        if with_applied_migrations:
            # Create the forwards plan Django would follow on an empty database
            full_plan = self.migration_plan(self.loader.graph.leaf_nodes(), clean_start=True)
            applied_migrations = {
                self.loader.graph.nodes[key] for key in self.loader.applied_migrations
                if key in self.loader.graph.nodes
            }
            for migration, _ in full_plan:
                if migration in applied_migrations:
                    migration.mutate_state(state, preserve=False)
        return state
```
