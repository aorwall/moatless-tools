# django__django-10087

| **django/django** | `02cd16a7a04529c726e5bb5a13d5979119f25c7d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 562 |
| **Any found context length** | 265 |
| **Avg pos** | 3.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/sqlmigrate.py b/django/core/management/commands/sqlmigrate.py
--- a/django/core/management/commands/sqlmigrate.py
+++ b/django/core/management/commands/sqlmigrate.py
@@ -1,3 +1,4 @@
+from django.apps import apps
 from django.core.management.base import BaseCommand, CommandError
 from django.db import DEFAULT_DB_ALIAS, connections
 from django.db.migrations.executor import MigrationExecutor
@@ -37,6 +38,11 @@ def handle(self, *args, **options):
 
         # Resolve command-line arguments into a migration
         app_label, migration_name = options['app_label'], options['migration_name']
+        # Validate app_label
+        try:
+            apps.get_app_config(app_label)
+        except LookupError as err:
+            raise CommandError(str(err))
         if app_label not in executor.loader.migrated_apps:
             raise CommandError("App '%s' does not have migrations" % app_label)
         try:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/sqlmigrate.py | 1 | 1 | 1 | 1 | 265
| django/core/management/commands/sqlmigrate.py | 40 | 40 | 2 | 1 | 562


## Problem Statement

```
Misleading sqlmigrate "App 'apps.somethings' does not have migrations." error message
Description
	
This ticket is very similar to https://code.djangoproject.com/ticket/29506
As shown above, validation should be added sqlmigrate.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/management/commands/sqlmigrate.py** | 1 | 29| 265 | 265 | 557 | 
| **-> 2 <-** | **1 django/core/management/commands/sqlmigrate.py** | 31 | 60| 297 | 562 | 557 | 
| 3 | 2 django/core/management/commands/showmigrations.py | 35 | 53| 158 | 720 | 1649 | 
| 4 | 3 django/core/management/sql.py | 38 | 53| 116 | 836 | 2039 | 
| 5 | 3 django/core/management/sql.py | 21 | 35| 116 | 952 | 2039 | 
| 6 | 4 django/core/management/commands/migrate.py | 21 | 61| 361 | 1313 | 4754 | 
| 7 | 4 django/core/management/commands/migrate.py | 63 | 139| 644 | 1957 | 4754 | 
| 8 | 4 django/core/management/commands/migrate.py | 140 | 228| 871 | 2828 | 4754 | 
| 9 | 5 django/db/migrations/exceptions.py | 1 | 55| 250 | 3078 | 5005 | 
| 10 | 6 django/core/management/commands/makemigrations.py | 56 | 139| 748 | 3826 | 7699 | 
| 11 | 7 django/core/management/commands/squashmigrations.py | 197 | 210| 112 | 3938 | 9535 | 
| 12 | 7 django/core/management/commands/migrate.py | 1 | 18| 144 | 4082 | 9535 | 
| 13 | 7 django/core/management/commands/makemigrations.py | 140 | 177| 302 | 4384 | 9535 | 
| 14 | 8 django/db/migrations/recorder.py | 1 | 81| 592 | 4976 | 10127 | 
| 15 | 9 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 5113 | 10264 | 
| 16 | 9 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 5262 | 10264 | 
| 17 | 10 django/db/backends/mysql/validation.py | 1 | 27| 248 | 5510 | 10746 | 
| 18 | 10 django/core/management/commands/migrate.py | 264 | 312| 404 | 5914 | 10746 | 
| 19 | 10 django/core/management/commands/showmigrations.py | 55 | 89| 360 | 6274 | 10746 | 
| 20 | 11 django/db/migrations/loader.py | 145 | 171| 291 | 6565 | 13629 | 
| 21 | 11 django/core/management/commands/makemigrations.py | 23 | 54| 271 | 6836 | 13629 | 
| 22 | 12 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 192 | 7028 | 13821 | 
| 23 | 13 django/db/migrations/state.py | 591 | 602| 136 | 7164 | 18949 | 
| 24 | 14 django/db/migrations/executor.py | 291 | 370| 712 | 7876 | 22192 | 
| 25 | 15 django/db/migrations/autodetector.py | 1115 | 1140| 245 | 8121 | 33444 | 
| 26 | 15 django/db/migrations/state.py | 1 | 25| 196 | 8317 | 33444 | 
| 27 | 15 django/db/migrations/autodetector.py | 344 | 358| 141 | 8458 | 33444 | 
| 28 | 16 django/db/migrations/operations/special.py | 116 | 130| 139 | 8597 | 35002 | 
| 29 | 16 django/db/migrations/autodetector.py | 1142 | 1154| 131 | 8728 | 35002 | 
| 30 | 17 django/db/backends/sqlite3/schema.py | 1 | 28| 231 | 8959 | 38485 | 
| 31 | 17 django/db/migrations/state.py | 292 | 316| 266 | 9225 | 38485 | 
| 32 | 18 django/core/management/base.py | 436 | 468| 282 | 9507 | 42745 | 
| 33 | 19 django/db/migrations/questioner.py | 226 | 239| 123 | 9630 | 44820 | 
| 34 | 20 django/db/utils.py | 267 | 309| 322 | 9952 | 46838 | 
| 35 | 20 django/db/migrations/loader.py | 272 | 296| 205 | 10157 | 46838 | 
| 36 | 21 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 11006 | 47687 | 
| 37 | 21 django/core/management/commands/squashmigrations.py | 1 | 39| 327 | 11333 | 47687 | 
| 38 | 22 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 11640 | 47994 | 
| 39 | 22 django/core/management/commands/migrate.py | 230 | 262| 337 | 11977 | 47994 | 
| 40 | 23 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 12299 | 48316 | 
| 41 | 23 django/core/management/commands/squashmigrations.py | 41 | 129| 782 | 13081 | 48316 | 
| 42 | 23 django/db/migrations/autodetector.py | 250 | 324| 810 | 13891 | 48316 | 
| 43 | 24 django/apps/registry.py | 209 | 229| 237 | 14128 | 51714 | 
| 44 | 24 django/db/migrations/autodetector.py | 1056 | 1077| 231 | 14359 | 51714 | 
| 45 | 24 django/db/migrations/questioner.py | 1 | 53| 468 | 14827 | 51714 | 
| 46 | 24 django/db/migrations/operations/special.py | 181 | 204| 246 | 15073 | 51714 | 
| 47 | 25 django/contrib/contenttypes/apps.py | 1 | 23| 150 | 15223 | 51864 | 
| 48 | 26 django/core/exceptions.py | 94 | 184| 623 | 15846 | 52869 | 
| 49 | 27 django/core/checks/model_checks.py | 73 | 97| 268 | 16114 | 54185 | 
| 50 | 27 django/db/migrations/state.py | 246 | 290| 432 | 16546 | 54185 | 
| 51 | 27 django/db/migrations/autodetector.py | 1019 | 1054| 312 | 16858 | 54185 | 
| 52 | 27 django/db/migrations/autodetector.py | 38 | 48| 120 | 16978 | 54185 | 
| 53 | 28 django/db/migrations/utils.py | 1 | 18| 0 | 16978 | 54273 | 
| 54 | 28 django/core/management/commands/showmigrations.py | 91 | 132| 323 | 17301 | 54273 | 
| 55 | 29 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 17518 | 54490 | 
| 56 | 29 django/db/migrations/loader.py | 298 | 320| 206 | 17724 | 54490 | 
| 57 | 30 django/contrib/messages/apps.py | 1 | 8| 0 | 17724 | 54527 | 
| 58 | 30 django/core/checks/model_checks.py | 99 | 120| 263 | 17987 | 54527 | 
| 59 | 31 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 273 | 18260 | 54800 | 
| 60 | 32 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 18422 | 54962 | 
| 61 | 32 django/db/migrations/executor.py | 274 | 289| 165 | 18587 | 54962 | 
| 62 | 32 django/db/migrations/autodetector.py | 423 | 449| 256 | 18843 | 54962 | 
| 63 | 33 django/db/migrations/graph.py | 144 | 166| 197 | 19040 | 58133 | 
| 64 | 34 django/db/models/base.py | 1208 | 1233| 184 | 19224 | 72306 | 
| 65 | 34 django/db/migrations/autodetector.py | 978 | 998| 134 | 19358 | 72306 | 
| 66 | 35 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 19358 | 72382 | 
| 67 | 35 django/db/migrations/loader.py | 64 | 121| 535 | 19893 | 72382 | 
| 68 | 35 django/db/migrations/autodetector.py | 1205 | 1228| 240 | 20133 | 72382 | 
| 69 | 36 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 20244 | 72493 | 
| 70 | 36 django/core/management/commands/showmigrations.py | 1 | 33| 266 | 20510 | 72493 | 
| 71 | 37 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 20510 | 72594 | 
| 72 | 37 django/db/models/base.py | 1 | 44| 277 | 20787 | 72594 | 
| 73 | 38 django/db/migrations/__init__.py | 1 | 3| 0 | 20787 | 72618 | 
| 74 | 38 django/db/migrations/state.py | 155 | 165| 132 | 20919 | 72618 | 
| 75 | 39 django/contrib/auth/apps.py | 1 | 29| 213 | 21132 | 72831 | 
| 76 | 39 django/db/migrations/autodetector.py | 236 | 249| 178 | 21310 | 72831 | 
| 77 | 40 django/db/models/signals.py | 37 | 54| 231 | 21541 | 73318 | 
| 78 | 40 django/db/migrations/loader.py | 51 | 62| 116 | 21657 | 73318 | 
| 79 | 40 django/apps/registry.py | 124 | 142| 166 | 21823 | 73318 | 
| 80 | 40 django/db/migrations/autodetector.py | 960 | 976| 188 | 22011 | 73318 | 
| 81 | 40 django/db/migrations/state.py | 553 | 574| 229 | 22240 | 73318 | 
| 82 | 41 django/contrib/sitemaps/apps.py | 1 | 8| 0 | 22240 | 73359 | 
| 83 | 42 django/db/migrations/operations/__init__.py | 1 | 16| 179 | 22419 | 73538 | 
| 84 | 43 django/db/migrations/operations/models.py | 628 | 641| 138 | 22557 | 79906 | 
| 85 | 43 django/core/checks/model_checks.py | 122 | 155| 332 | 22889 | 79906 | 
| 86 | 43 django/db/migrations/autodetector.py | 326 | 342| 166 | 23055 | 79906 | 
| 87 | 43 django/db/migrations/loader.py | 196 | 270| 746 | 23801 | 79906 | 
| 88 | 44 django/contrib/admin/sites.py | 1 | 28| 172 | 23973 | 84001 | 
| 89 | 45 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 24180 | 84208 | 
| 90 | 46 django/contrib/admin/checks.py | 1 | 27| 179 | 24359 | 92831 | 
| 91 | 46 django/db/migrations/operations/models.py | 430 | 446| 203 | 24562 | 92831 | 
| 92 | 46 django/db/migrations/state.py | 167 | 191| 213 | 24775 | 92831 | 
| 93 | 46 django/db/backends/sqlite3/schema.py | 77 | 89| 163 | 24938 | 92831 | 
| 94 | 46 django/core/management/commands/squashmigrations.py | 131 | 195| 649 | 25587 | 92831 | 
| 95 | 46 django/db/migrations/autodetector.py | 1156 | 1203| 429 | 26016 | 92831 | 
| 96 | 47 django/db/migrations/operations/fields.py | 75 | 95| 223 | 26239 | 95818 | 
| 97 | 47 django/db/migrations/autodetector.py | 511 | 647| 1058 | 27297 | 95818 | 
| 98 | 47 django/db/migrations/autodetector.py | 221 | 234| 199 | 27496 | 95818 | 
| 99 | 47 django/core/management/commands/makemigrations.py | 225 | 305| 820 | 28316 | 95818 | 
| 100 | 47 django/core/checks/model_checks.py | 31 | 52| 168 | 28484 | 95818 | 
| 101 | 47 django/db/migrations/state.py | 230 | 243| 133 | 28617 | 95818 | 
| 102 | 48 django/core/management/commands/sqlsequencereset.py | 1 | 24| 172 | 28789 | 95990 | 
| 103 | 49 django/db/backends/base/creation.py | 276 | 295| 121 | 28910 | 98298 | 
| 104 | 50 django/core/management/templates.py | 206 | 236| 235 | 29145 | 100923 | 
| 105 | 50 django/db/migrations/executor.py | 64 | 80| 168 | 29313 | 100923 | 
| 106 | 50 django/db/migrations/graph.py | 325 | 343| 158 | 29471 | 100923 | 
| 107 | 50 django/db/migrations/operations/special.py | 63 | 114| 390 | 29861 | 100923 | 
| 108 | 51 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 29983 | 101172 | 
| 109 | 51 django/db/backends/sqlite3/schema.py | 331 | 365| 358 | 30341 | 101172 | 
| 110 | 52 django/core/management/commands/check.py | 1 | 35| 241 | 30582 | 101622 | 
| 111 | 52 django/db/migrations/autodetector.py | 1079 | 1113| 296 | 30878 | 101622 | 
| 112 | 53 django/core/checks/security/sessions.py | 1 | 98| 572 | 31450 | 102195 | 
| 113 | 54 django/core/checks/templates.py | 1 | 36| 259 | 31709 | 102455 | 
| 114 | 55 django/core/management/commands/flush.py | 27 | 83| 496 | 32205 | 103160 | 
| 115 | 56 django/core/checks/security/csrf.py | 1 | 41| 299 | 32504 | 103459 | 
| 116 | 56 django/db/migrations/autodetector.py | 1000 | 1017| 180 | 32684 | 103459 | 
| 117 | 57 django/db/backends/base/features.py | 268 | 290| 178 | 32862 | 105750 | 
| 118 | 58 django/db/migrations/operations/base.py | 104 | 142| 303 | 33165 | 106833 | 
| 119 | 59 django/db/models/options.py | 1 | 38| 330 | 33495 | 113721 | 
| 120 | 59 django/db/migrations/questioner.py | 55 | 80| 220 | 33715 | 113721 | 
| 121 | 59 django/db/migrations/operations/models.py | 610 | 626| 216 | 33931 | 113721 | 
| 122 | 60 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 33931 | 113818 | 
| 123 | 60 django/db/migrations/autodetector.py | 451 | 492| 418 | 34349 | 113818 | 
| 124 | 61 django/contrib/gis/db/models/sql/__init__.py | 1 | 8| 0 | 34349 | 113853 | 
| 125 | 61 django/db/migrations/operations/models.py | 448 | 460| 139 | 34488 | 113853 | 
| 126 | 62 django/db/migrations/migration.py | 127 | 194| 585 | 35073 | 115486 | 
| 127 | 63 django/db/backends/base/schema.py | 1002 | 1015| 122 | 35195 | 125739 | 
| 128 | 63 django/db/migrations/state.py | 576 | 589| 137 | 35332 | 125739 | 
| 129 | 63 django/db/migrations/operations/fields.py | 208 | 218| 146 | 35478 | 125739 | 
| 130 | 63 django/core/checks/model_checks.py | 54 | 71| 210 | 35688 | 125739 | 
| 131 | 63 django/db/migrations/graph.py | 1 | 14| 105 | 35793 | 125739 | 
| 132 | 64 django/contrib/flatpages/sitemaps.py | 1 | 13| 112 | 35905 | 125851 | 
| 133 | 64 django/db/migrations/loader.py | 123 | 143| 211 | 36116 | 125851 | 
| 134 | 64 django/db/backends/sqlite3/schema.py | 213 | 277| 567 | 36683 | 125851 | 
| 135 | 65 django/conf/global_settings.py | 491 | 636| 869 | 37552 | 131448 | 
| 136 | 66 django/db/backends/mysql/creation.py | 1 | 39| 317 | 37869 | 132105 | 
| 137 | 66 django/db/migrations/operations/models.py | 322 | 371| 493 | 38362 | 132105 | 
| 138 | 66 django/db/migrations/operations/models.py | 1 | 37| 233 | 38595 | 132105 | 
| 139 | 67 django/core/management/commands/loaddata.py | 63 | 79| 187 | 38782 | 134951 | 
| 140 | 68 django/contrib/postgres/apps.py | 1 | 36| 291 | 39073 | 135242 | 
| 141 | 69 django/db/backends/mysql/schema.py | 1 | 39| 359 | 39432 | 136245 | 
| 142 | 69 django/db/models/base.py | 1178 | 1206| 235 | 39667 | 136245 | 
| 143 | 69 django/db/migrations/operations/models.py | 101 | 131| 266 | 39933 | 136245 | 
| 144 | 70 django/contrib/auth/management/__init__.py | 37 | 88| 465 | 40398 | 137322 | 
| 145 | 70 django/core/management/commands/flush.py | 1 | 25| 214 | 40612 | 137322 | 
| 146 | 70 django/db/migrations/state.py | 492 | 524| 250 | 40862 | 137322 | 
| 147 | 70 django/db/migrations/loader.py | 173 | 194| 213 | 41075 | 137322 | 
| 148 | 71 django/core/checks/security/base.py | 88 | 190| 747 | 41822 | 138946 | 
| 149 | 71 django/core/management/commands/loaddata.py | 32 | 61| 291 | 42113 | 138946 | 
| 150 | 71 django/core/checks/model_checks.py | 1 | 28| 162 | 42275 | 138946 | 
| 151 | 71 django/apps/registry.py | 259 | 271| 112 | 42387 | 138946 | 
| 152 | 72 django/db/migrations/serializer.py | 139 | 158| 223 | 42610 | 141644 | 
| 153 | 73 django/db/backends/oracle/creation.py | 130 | 163| 393 | 43003 | 145359 | 
| 154 | 73 django/db/models/options.py | 299 | 319| 158 | 43161 | 145359 | 
| 155 | 74 django/db/backends/postgresql_psycopg2/creation.py | 1 | 2| 0 | 43161 | 145370 | 
| 156 | 75 django/db/backends/mysql/operations.py | 169 | 212| 329 | 43490 | 148222 | 
| 157 | 76 django/contrib/humanize/apps.py | 1 | 8| 0 | 43490 | 148263 | 
| 158 | 77 django/db/backends/sqlite3/base.py | 227 | 280| 499 | 43989 | 152665 | 
| 159 | 77 django/db/migrations/autodetector.py | 494 | 510| 186 | 44175 | 152665 | 
| 160 | 78 django/db/backends/postgresql_psycopg2/schema.py | 1 | 2| 0 | 44175 | 152676 | 
| 161 | 79 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 44313 | 152814 | 
| 162 | 80 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 44313 | 152882 | 
| 163 | 80 django/db/migrations/state.py | 318 | 345| 256 | 44569 | 152882 | 
| 164 | 80 django/db/migrations/executor.py | 231 | 254| 216 | 44785 | 152882 | 
| 165 | 80 django/db/migrations/operations/fields.py | 63 | 73| 125 | 44910 | 152882 | 
| 166 | 80 django/db/migrations/operations/special.py | 44 | 60| 180 | 45090 | 152882 | 
| 167 | 80 django/db/backends/base/schema.py | 42 | 110| 728 | 45818 | 152882 | 
| 168 | 81 django/db/migrations/writer.py | 208 | 297| 571 | 46389 | 155099 | 
| 169 | 82 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 46539 | 155249 | 
| 170 | 83 django/contrib/postgres/utils.py | 1 | 30| 218 | 46757 | 155467 | 
| 171 | 84 django/contrib/sites/apps.py | 1 | 14| 0 | 46757 | 155539 | 
| 172 | 84 django/db/migrations/loader.py | 1 | 49| 390 | 47147 | 155539 | 
| 173 | 84 django/db/migrations/questioner.py | 142 | 159| 183 | 47330 | 155539 | 
| 174 | 84 django/db/migrations/state.py | 107 | 153| 368 | 47698 | 155539 | 
| 175 | 84 django/core/checks/security/base.py | 1 | 86| 750 | 48448 | 155539 | 
| 176 | 84 django/db/migrations/autodetector.py | 1230 | 1261| 314 | 48762 | 155539 | 
| 177 | 84 django/db/backends/sqlite3/schema.py | 91 | 127| 469 | 49231 | 155539 | 
| 178 | 84 django/core/management/commands/loaddata.py | 1 | 29| 151 | 49382 | 155539 | 
| 179 | 85 django/core/checks/database.py | 1 | 12| 0 | 49382 | 155592 | 
| 180 | 85 django/db/migrations/operations/fields.py | 147 | 159| 181 | 49563 | 155592 | 
| 181 | 85 django/db/models/base.py | 1629 | 1700| 565 | 50128 | 155592 | 
| 182 | 86 django/contrib/contenttypes/management/__init__.py | 1 | 42| 356 | 50484 | 156566 | 
| 183 | 87 django/contrib/postgres/validators.py | 1 | 21| 181 | 50665 | 157117 | 
| 184 | 87 django/db/migrations/autodetector.py | 683 | 774| 815 | 51480 | 157117 | 
| 185 | 88 django/db/backends/postgresql/creation.py | 31 | 42| 133 | 51613 | 157672 | 
| 186 | 88 django/db/migrations/autodetector.py | 884 | 958| 812 | 52425 | 157672 | 
| 187 | 89 django/contrib/redirects/apps.py | 1 | 8| 0 | 52425 | 157712 | 
| 188 | 89 django/db/migrations/autodetector.py | 1 | 36| 295 | 52720 | 157712 | 
| 189 | 89 django/db/migrations/autodetector.py | 827 | 861| 339 | 53059 | 157712 | 
| 190 | 89 django/db/migrations/operations/models.py | 373 | 389| 182 | 53241 | 157712 | 
| 191 | 89 django/db/migrations/autodetector.py | 90 | 102| 118 | 53359 | 157712 | 
| 192 | 89 django/db/migrations/questioner.py | 83 | 106| 187 | 53546 | 157712 | 
| 193 | 90 django/contrib/gis/db/backends/mysql/operations.py | 52 | 67| 174 | 53720 | 158529 | 
| 194 | 91 django/db/backends/postgresql_psycopg2/features.py | 1 | 2| 0 | 53720 | 158540 | 
| 195 | 92 django/db/models/fields/related.py | 155 | 168| 144 | 53864 | 172010 | 
| 196 | 92 django/db/backends/postgresql/creation.py | 1 | 29| 184 | 54048 | 172010 | 
| 197 | 92 django/db/migrations/autodetector.py | 104 | 195| 743 | 54791 | 172010 | 
| 198 | 93 django/db/backends/dummy/features.py | 1 | 6| 0 | 54791 | 172035 | 


### Hint

```
​https://github.com/django/django/pull/10087 I added validation to sqlmigrate
```

## Patch

```diff
diff --git a/django/core/management/commands/sqlmigrate.py b/django/core/management/commands/sqlmigrate.py
--- a/django/core/management/commands/sqlmigrate.py
+++ b/django/core/management/commands/sqlmigrate.py
@@ -1,3 +1,4 @@
+from django.apps import apps
 from django.core.management.base import BaseCommand, CommandError
 from django.db import DEFAULT_DB_ALIAS, connections
 from django.db.migrations.executor import MigrationExecutor
@@ -37,6 +38,11 @@ def handle(self, *args, **options):
 
         # Resolve command-line arguments into a migration
         app_label, migration_name = options['app_label'], options['migration_name']
+        # Validate app_label
+        try:
+            apps.get_app_config(app_label)
+        except LookupError as err:
+            raise CommandError(str(err))
         if app_label not in executor.loader.migrated_apps:
             raise CommandError("App '%s' does not have migrations" % app_label)
         try:

```

## Test Patch

```diff
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -1434,6 +1434,14 @@ def test_migrate_app_name_specified_as_label(self):
         with self.assertRaisesMessage(CommandError, self.did_you_mean_auth_error):
             call_command('migrate', 'django.contrib.auth')
 
+    def test_sqlmigrate_nonexistent_app_label(self):
+        with self.assertRaisesMessage(CommandError, self.nonexistent_app_error):
+            call_command('sqlmigrate', 'nonexistent_app', '0002')
+
+    def test_sqlmigrate_app_name_specified_as_label(self):
+        with self.assertRaisesMessage(CommandError, self.did_you_mean_auth_error):
+            call_command('sqlmigrate', 'django.contrib.auth', '0002')
+
     def test_squashmigrations_nonexistent_app_label(self):
         with self.assertRaisesMessage(CommandError, self.nonexistent_app_error):
             call_command('squashmigrations', 'nonexistent_app', '0002')

```


## Code snippets

### 1 - django/core/management/commands/sqlmigrate.py:

Start line: 1, End line: 29

```python
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.loader import AmbiguityError


class Command(BaseCommand):
    help = "Prints the SQL statements for the named migration."

    output_transaction = True

    def add_arguments(self, parser):
        parser.add_argument('app_label', help='App label of the application containing the migration.')
        parser.add_argument('migration_name', help='Migration name to print the SQL for.')
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database to create SQL for. Defaults to the "default" database.',
        )
        parser.add_argument(
            '--backwards', action='store_true', dest='backwards',
            help='Creates SQL to unapply the migration, rather than to apply it',
        )

    def execute(self, *args, **options):
        # sqlmigrate doesn't support coloring its output but we need to force
        # no_color=True so that the BEGIN/COMMIT statements added by
        # output_transaction don't get colored either.
        options['no_color'] = True
        return super().execute(*args, **options)
```
### 2 - django/core/management/commands/sqlmigrate.py:

Start line: 31, End line: 60

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        # Get the database we're operating from
        connection = connections[options['database']]

        # Load up an executor to get all the migration data
        executor = MigrationExecutor(connection)

        # Resolve command-line arguments into a migration
        app_label, migration_name = options['app_label'], options['migration_name']
        if app_label not in executor.loader.migrated_apps:
            raise CommandError("App '%s' does not have migrations" % app_label)
        try:
            migration = executor.loader.get_migration_by_prefix(app_label, migration_name)
        except AmbiguityError:
            raise CommandError("More than one migration matches '%s' in app '%s'. Please be more specific." % (
                migration_name, app_label))
        except KeyError:
            raise CommandError("Cannot find a migration matching '%s' from app '%s'. Is it in INSTALLED_APPS?" % (
                migration_name, app_label))
        targets = [(app_label, migration.name)]

        # Show begin/end around output only for atomic migrations
        self.output_transaction = migration.atomic

        # Make a plan that represents just the requested migrations and show SQL
        # for it
        plan = [(executor.loader.graph.nodes[targets[0]], options['backwards'])]
        sql_statements = executor.collect_sql(plan)
        return '\n'.join(sql_statements)
```
### 3 - django/core/management/commands/showmigrations.py:

Start line: 35, End line: 53

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        self.verbosity = options['verbosity']

        # Get the database we're operating from
        db = options['database']
        connection = connections[db]

        if options['format'] == "plan":
            return self.show_plan(connection, options['app_label'])
        else:
            return self.show_list(connection, options['app_label'])

    def _validate_app_names(self, loader, app_names):
        invalid_apps = []
        for app_name in app_names:
            if app_name not in loader.migrated_apps:
                invalid_apps.append(app_name)
        if invalid_apps:
            raise CommandError('No migrations present for: %s' % (', '.join(sorted(invalid_apps))))
```
### 4 - django/core/management/sql.py:

Start line: 38, End line: 53

```python
def emit_post_migrate_signal(verbosity, interactive, db, **kwargs):
    # Emit the post_migrate signal for every application.
    for app_config in apps.get_app_configs():
        if app_config.models_module is None:
            continue
        if verbosity >= 2:
            print("Running post-migrate handlers for application %s" % app_config.label)
        models.signals.post_migrate.send(
            sender=app_config,
            app_config=app_config,
            verbosity=verbosity,
            interactive=interactive,
            using=db,
            **kwargs
        )
```
### 5 - django/core/management/sql.py:

Start line: 21, End line: 35

```python
def emit_pre_migrate_signal(verbosity, interactive, db, **kwargs):
    # Emit the pre_migrate signal for every application.
    for app_config in apps.get_app_configs():
        if app_config.models_module is None:
            continue
        if verbosity >= 2:
            print("Running pre-migrate handlers for application %s" % app_config.label)
        models.signals.pre_migrate.send(
            sender=app_config,
            app_config=app_config,
            verbosity=verbosity,
            interactive=interactive,
            using=db,
            **kwargs
        )
```
### 6 - django/core/management/commands/migrate.py:

Start line: 21, End line: 61

```python
class Command(BaseCommand):
    help = "Updates database schema. Manages both apps with migrations and those without."

    def add_arguments(self, parser):
        parser.add_argument(
            'app_label', nargs='?',
            help='App label of an application to synchronize the state.',
        )
        parser.add_argument(
            'migration_name', nargs='?',
            help='Database state will be brought to the state after that '
                 'migration. Use the name "zero" to unapply all migrations.',
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--database', action='store', dest='database',
            default=DEFAULT_DB_ALIAS,
            help='Nominates a database to synchronize. Defaults to the "default" database.',
        )
        parser.add_argument(
            '--fake', action='store_true', dest='fake',
            help='Mark migrations as run without actually running them.',
        )
        parser.add_argument(
            '--fake-initial', action='store_true', dest='fake_initial',
            help='Detect if tables already exist and fake-apply initial migrations if so. Make sure '
                 'that the current database schema matches your initial migration before using this '
                 'flag. Django will only check for an existing table name.',
        )
        parser.add_argument(
            '--run-syncdb', action='store_true', dest='run_syncdb',
            help='Creates tables for apps without migrations.',
        )

    def _run_checks(self, **kwargs):
        issues = run_checks(tags=[Tags.database])
        issues.extend(super()._run_checks(**kwargs))
        return issues
```
### 7 - django/core/management/commands/migrate.py:

Start line: 63, End line: 139

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):

        self.verbosity = options['verbosity']
        self.interactive = options['interactive']

        # Import the 'management' module within each installed app, to register
        # dispatcher events.
        for app_config in apps.get_app_configs():
            if module_has_submodule(app_config.module, "management"):
                import_module('.management', app_config.name)

        # Get the database we're operating from
        db = options['database']
        connection = connections[db]

        # Hook for backends needing any database preparation
        connection.prepare_database()
        # Work out which apps have migrations and which do not
        executor = MigrationExecutor(connection, self.migration_progress_callback)

        # Raise an error if any migrations are applied before their dependencies.
        executor.loader.check_consistent_history(connection)

        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any
        conflicts = executor.loader.detect_conflicts()
        if conflicts:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they supplied command line arguments, work out what they mean.
        target_app_labels_only = True
        if options['app_label']:
            # Validate app_label.
            app_label = options['app_label']
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                raise CommandError(str(err))
            if app_label not in executor.loader.migrated_apps:
                raise CommandError("App '%s' does not have migrations." % app_label)

        if options['app_label'] and options['migration_name']:
            migration_name = options['migration_name']
            if migration_name == "zero":
                targets = [(app_label, None)]
            else:
                try:
                    migration = executor.loader.get_migration_by_prefix(app_label, migration_name)
                except AmbiguityError:
                    raise CommandError(
                        "More than one migration matches '%s' in app '%s'. "
                        "Please be more specific." %
                        (migration_name, app_label)
                    )
                except KeyError:
                    raise CommandError("Cannot find a migration matching '%s' from app '%s'." % (
                        migration_name, app_label))
                targets = [(app_label, migration.name)]
            target_app_labels_only = False
        elif options['app_label']:
            targets = [key for key in executor.loader.graph.leaf_nodes() if key[0] == app_label]
        else:
            targets = executor.loader.graph.leaf_nodes()

        plan = executor.migration_plan(targets)
        run_syncdb = options['run_syncdb'] and executor.loader.unmigrated_apps

        # Print some useful info
        # ... other code
```
### 8 - django/core/management/commands/migrate.py:

Start line: 140, End line: 228

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        # ... other code
        if self.verbosity >= 1:
            self.stdout.write(self.style.MIGRATE_HEADING("Operations to perform:"))
            if run_syncdb:
                self.stdout.write(
                    self.style.MIGRATE_LABEL("  Synchronize unmigrated apps: ") +
                    (", ".join(sorted(executor.loader.unmigrated_apps)))
                )
            if target_app_labels_only:
                self.stdout.write(
                    self.style.MIGRATE_LABEL("  Apply all migrations: ") +
                    (", ".join(sorted({a for a, n in targets})) or "(none)")
                )
            else:
                if targets[0][1] is None:
                    self.stdout.write(self.style.MIGRATE_LABEL(
                        "  Unapply all migrations: ") + "%s" % (targets[0][0],)
                    )
                else:
                    self.stdout.write(self.style.MIGRATE_LABEL(
                        "  Target specific migration: ") + "%s, from %s"
                        % (targets[0][1], targets[0][0])
                    )

        pre_migrate_state = executor._create_project_state(with_applied_migrations=True)
        pre_migrate_apps = pre_migrate_state.apps
        emit_pre_migrate_signal(
            self.verbosity, self.interactive, connection.alias, apps=pre_migrate_apps, plan=plan,
        )

        # Run the syncdb phase.
        if run_syncdb:
            if self.verbosity >= 1:
                self.stdout.write(self.style.MIGRATE_HEADING("Synchronizing apps without migrations:"))
            self.sync_apps(connection, executor.loader.unmigrated_apps)

        # Migrate!
        if self.verbosity >= 1:
            self.stdout.write(self.style.MIGRATE_HEADING("Running migrations:"))
        if not plan:
            if self.verbosity >= 1:
                self.stdout.write("  No migrations to apply.")
                # If there's changes that aren't in migrations yet, tell them how to fix it.
                autodetector = MigrationAutodetector(
                    executor.loader.project_state(),
                    ProjectState.from_apps(apps),
                )
                changes = autodetector.changes(graph=executor.loader.graph)
                if changes:
                    self.stdout.write(self.style.NOTICE(
                        "  Your models have changes that are not yet reflected "
                        "in a migration, and so won't be applied."
                    ))
                    self.stdout.write(self.style.NOTICE(
                        "  Run 'manage.py makemigrations' to make new "
                        "migrations, and then re-run 'manage.py migrate' to "
                        "apply them."
                    ))
            fake = False
            fake_initial = False
        else:
            fake = options['fake']
            fake_initial = options['fake_initial']
        post_migrate_state = executor.migrate(
            targets, plan=plan, state=pre_migrate_state.clone(), fake=fake,
            fake_initial=fake_initial,
        )
        # post_migrate signals have access to all models. Ensure that all models
        # are reloaded in case any are delayed.
        post_migrate_state.clear_delayed_apps_cache()
        post_migrate_apps = post_migrate_state.apps

        # Re-render models of real apps to include relationships now that
        # we've got a final state. This wouldn't be necessary if real apps
        # models were rendered with relationships in the first place.
        with post_migrate_apps.bulk_update():
            model_keys = []
            for model_state in post_migrate_apps.real_models:
                model_key = model_state.app_label, model_state.name_lower
                model_keys.append(model_key)
                post_migrate_apps.unregister_model(*model_key)
        post_migrate_apps.render_multiple([
            ModelState.from_model(apps.get_model(*model)) for model in model_keys
        ])

        # Send the post_migrate signal, so individual apps can do whatever they need
        # to do at this point.
        emit_post_migrate_signal(
            self.verbosity, self.interactive, connection.alias, apps=post_migrate_apps, plan=plan,
        )
```
### 9 - django/db/migrations/exceptions.py:

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
### 10 - django/core/management/commands/makemigrations.py:

Start line: 56, End line: 139

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        self.verbosity = options['verbosity']
        self.interactive = options['interactive']
        self.dry_run = options['dry_run']
        self.merge = options['merge']
        self.empty = options['empty']
        self.migration_name = options['name']
        check_changes = options['check_changes']

        # Make sure the app they asked for exists
        app_labels = set(app_labels)
        has_bad_labels = False
        for app_label in app_labels:
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                self.stderr.write(str(err))
                has_bad_labels = True
        if has_bad_labels:
            sys.exit(2)

        # Load the current graph state. Pass in None for the connection so
        # the loader doesn't try to resolve replaced migrations from DB.
        loader = MigrationLoader(None, ignore_no_migrations=True)

        # Raise an error if any migrations are applied before their dependencies.
        consistency_check_labels = {config.label for config in apps.get_app_configs()}
        # Non-default databases are only checked if database routers used.
        aliases_to_check = connections if settings.DATABASE_ROUTERS else [DEFAULT_DB_ALIAS]
        for alias in sorted(aliases_to_check):
            connection = connections[alias]
            if (connection.settings_dict['ENGINE'] != 'django.db.backends.dummy' and any(
                    # At least one model must be migrated to the database.
                    router.allow_migrate(connection.alias, app_label, model_name=model._meta.object_name)
                    for app_label in consistency_check_labels
                    for model in apps.get_app_config(app_label).get_models()
            )):
                loader.check_consistent_history(connection)

        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any and they don't want to merge
        conflicts = loader.detect_conflicts()

        # If app_labels is specified, filter out conflicting migrations for unspecified apps
        if app_labels:
            conflicts = {
                app_label: conflict for app_label, conflict in conflicts.items()
                if app_label in app_labels
            }

        if conflicts and not self.merge:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they want to merge and there's nothing to merge, then politely exit
        if self.merge and not conflicts:
            self.stdout.write("No conflicts detected to merge.")
            return

        # If they want to merge and there is something to merge, then
        # divert into the merge code
        if self.merge and conflicts:
            return self.handle_merge(loader, conflicts)

        if self.interactive:
            questioner = InteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        else:
            questioner = NonInteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        # Set up autodetector
        autodetector = MigrationAutodetector(
            loader.project_state(),
            ProjectState.from_apps(apps),
            questioner,
        )

        # If they want to make an empty migration, make one for each app
        # ... other code
```
