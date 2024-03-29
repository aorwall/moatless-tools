# django__django-16116

| **django/django** | `5d36a8266c7d5d1994d7a7eeb4016f80d9cb0401` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1682 |
| **Any found context length** | 458 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/makemigrations.py b/django/core/management/commands/makemigrations.py
--- a/django/core/management/commands/makemigrations.py
+++ b/django/core/management/commands/makemigrations.py
@@ -70,7 +70,10 @@ def add_arguments(self, parser):
             "--check",
             action="store_true",
             dest="check_changes",
-            help="Exit with a non-zero status if model changes are missing migrations.",
+            help=(
+                "Exit with a non-zero status if model changes are missing migrations "
+                "and don't actually write them."
+            ),
         )
         parser.add_argument(
             "--scriptable",
@@ -248,12 +251,12 @@ def handle(self, *app_labels, **options):
                 else:
                     self.log("No changes detected")
         else:
+            if check_changes:
+                sys.exit(1)
             if self.update:
                 self.write_to_last_migration_files(changes)
             else:
                 self.write_migration_files(changes)
-            if check_changes:
-                sys.exit(1)
 
     def write_to_last_migration_files(self, changes):
         loader = MigrationLoader(connections[DEFAULT_DB_ALIAS])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/makemigrations.py | 73 | 73 | 3 | 1 | 1682
| django/core/management/commands/makemigrations.py | 251 | 252 | 1 | 1 | 458


## Problem Statement

```
makemigrations --check generating migrations is inconsistent with other uses of --check
Description
	
To script a check for missing migrations but without actually intending to create the migrations, it is necessary to use both --check and --dry-run, which is inconsistent with migrate --check and optimizemigration --check, which just exit (after possibly logging a bit).
I'm suggesting that makemigrations --check should just exit without making migrations.
The choice to write the migrations anyway was not discussed AFAICT on ticket:25604 or ​https://groups.google.com/g/django-developers/c/zczdY6c9KSg/m/ZXCXQsGDDAAJ.
Noticed when reading ​PR to adjust the documentation of migrate --check. I think the current documentation is silent on this question.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/management/commands/makemigrations.py** | 193 | 256| 458 | 458 | 3935 | 
| 2 | **1 django/core/management/commands/makemigrations.py** | 101 | 191| 791 | 1249 | 3935 | 
| **-> 3 <-** | **1 django/core/management/commands/makemigrations.py** | 26 | 99| 433 | 1682 | 3935 | 
| 4 | **1 django/core/management/commands/makemigrations.py** | 402 | 512| 927 | 2609 | 3935 | 
| 5 | 2 django/core/management/commands/migrate.py | 191 | 269| 678 | 3287 | 7861 | 
| 6 | 3 django/db/migrations/questioner.py | 291 | 342| 367 | 3654 | 10557 | 
| 7 | 4 django/db/migrations/loader.py | 307 | 336| 212 | 3866 | 13710 | 
| 8 | 5 django/db/migrations/executor.py | 290 | 305| 165 | 4031 | 17151 | 
| 9 | 5 django/core/management/commands/migrate.py | 96 | 189| 765 | 4796 | 17151 | 
| 10 | 5 django/core/management/commands/migrate.py | 17 | 94| 487 | 5283 | 17151 | 
| 11 | **5 django/core/management/commands/makemigrations.py** | 1 | 23| 185 | 5468 | 17151 | 
| 12 | **5 django/core/management/commands/makemigrations.py** | 258 | 328| 572 | 6040 | 17151 | 
| 13 | 6 django/core/management/commands/optimizemigration.py | 1 | 130| 940 | 6980 | 18091 | 
| 14 | 6 django/db/migrations/executor.py | 307 | 411| 862 | 7842 | 18091 | 
| 15 | 6 django/core/management/commands/migrate.py | 270 | 368| 813 | 8655 | 18091 | 
| 16 | 7 django/core/management/commands/squashmigrations.py | 162 | 254| 766 | 9421 | 20131 | 
| 17 | **7 django/core/management/commands/makemigrations.py** | 330 | 400| 621 | 10042 | 20131 | 
| 18 | 8 django/core/management/base.py | 556 | 596| 293 | 10335 | 24917 | 
| 19 | 9 django/db/migrations/autodetector.py | 280 | 378| 806 | 11141 | 38378 | 
| 20 | 9 django/db/migrations/autodetector.py | 403 | 419| 141 | 11282 | 38378 | 
| 21 | 9 django/core/management/commands/squashmigrations.py | 62 | 160| 809 | 12091 | 38378 | 
| 22 | 9 django/db/migrations/questioner.py | 1 | 55| 469 | 12560 | 38378 | 
| 23 | 9 django/db/migrations/autodetector.py | 1555 | 1593| 304 | 12864 | 38378 | 
| 24 | 9 django/db/migrations/loader.py | 169 | 197| 295 | 13159 | 38378 | 
| 25 | 9 django/core/management/commands/migrate.py | 369 | 390| 204 | 13363 | 38378 | 
| 26 | 9 django/db/migrations/questioner.py | 90 | 107| 163 | 13526 | 38378 | 
| 27 | 9 django/db/migrations/autodetector.py | 40 | 50| 120 | 13646 | 38378 | 
| 28 | 9 django/db/migrations/autodetector.py | 266 | 279| 178 | 13824 | 38378 | 
| 29 | 9 django/core/management/commands/migrate.py | 392 | 430| 361 | 14185 | 38378 | 
| 30 | 10 django/db/migrations/recorder.py | 48 | 104| 400 | 14585 | 39065 | 
| 31 | 11 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 14980 | 39731 | 
| 32 | 12 django/db/migrations/exceptions.py | 1 | 61| 249 | 15229 | 39981 | 
| 33 | 12 django/db/migrations/loader.py | 222 | 305| 791 | 16020 | 39981 | 
| 34 | 13 django/core/management/commands/showmigrations.py | 56 | 77| 158 | 16178 | 41277 | 
| 35 | 13 django/db/migrations/autodetector.py | 104 | 208| 909 | 17087 | 41277 | 
| 36 | 13 django/db/migrations/autodetector.py | 1595 | 1627| 258 | 17345 | 41277 | 
| 37 | 13 django/db/migrations/autodetector.py | 1479 | 1502| 161 | 17506 | 41277 | 
| 38 | 13 django/core/management/commands/squashmigrations.py | 1 | 60| 387 | 17893 | 41277 | 
| 39 | 13 django/core/management/commands/squashmigrations.py | 256 | 269| 112 | 18005 | 41277 | 
| 40 | 14 django/db/migrations/graph.py | 196 | 218| 239 | 18244 | 43900 | 
| 41 | 14 django/db/migrations/autodetector.py | 1432 | 1477| 318 | 18562 | 43900 | 
| 42 | 14 django/db/migrations/autodetector.py | 1504 | 1532| 235 | 18797 | 43900 | 
| 43 | 14 django/core/management/commands/showmigrations.py | 79 | 132| 492 | 19289 | 43900 | 
| 44 | 14 django/db/migrations/autodetector.py | 1346 | 1367| 197 | 19486 | 43900 | 
| 45 | 14 django/db/migrations/autodetector.py | 1645 | 1695| 439 | 19925 | 43900 | 
| 46 | 15 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 112 | 20037 | 44012 | 
| 47 | 15 django/db/migrations/autodetector.py | 811 | 906| 712 | 20749 | 44012 | 
| 48 | 15 django/db/migrations/executor.py | 263 | 288| 228 | 20977 | 44012 | 
| 49 | 15 django/db/migrations/autodetector.py | 1534 | 1553| 187 | 21164 | 44012 | 
| 50 | 15 django/db/migrations/autodetector.py | 1697 | 1722| 243 | 21407 | 44012 | 
| 51 | 16 django/contrib/redirects/migrations/0001_initial.py | 1 | 66| 309 | 21716 | 44321 | 
| 52 | 16 django/db/migrations/autodetector.py | 1629 | 1643| 135 | 21851 | 44321 | 
| 53 | 16 django/db/migrations/questioner.py | 109 | 124| 135 | 21986 | 44321 | 
| 54 | 16 django/core/management/commands/sqlmigrate.py | 1 | 38| 276 | 22262 | 44321 | 
| 55 | 16 django/db/migrations/autodetector.py | 1369 | 1395| 144 | 22406 | 44321 | 
| 56 | 16 django/core/management/commands/migrate.py | 489 | 512| 186 | 22592 | 44321 | 
| 57 | 16 django/core/management/commands/migrate.py | 1 | 14| 134 | 22726 | 44321 | 
| 58 | 17 django/db/migrations/writer.py | 118 | 204| 754 | 23480 | 46592 | 
| 59 | 17 django/db/migrations/graph.py | 101 | 122| 192 | 23672 | 46592 | 
| 60 | 18 django/contrib/sessions/migrations/0001_initial.py | 1 | 39| 173 | 23845 | 46765 | 
| 61 | 18 django/db/migrations/autodetector.py | 583 | 599| 186 | 24031 | 46765 | 
| 62 | 18 django/db/migrations/autodetector.py | 1 | 18| 115 | 24146 | 46765 | 
| 63 | 19 django/core/checks/model_checks.py | 187 | 228| 345 | 24491 | 48575 | 
| 64 | 20 django/contrib/auth/migrations/0001_initial.py | 1 | 206| 1007 | 25498 | 49582 | 
| 65 | 21 django/core/management/sql.py | 42 | 60| 132 | 25630 | 49949 | 
| 66 | 21 django/db/migrations/executor.py | 1 | 71| 571 | 26201 | 49949 | 
| 67 | 21 django/db/migrations/autodetector.py | 516 | 581| 482 | 26683 | 49949 | 
| 68 | 22 django/contrib/flatpages/migrations/0001_initial.py | 1 | 70| 355 | 27038 | 50304 | 
| 69 | 22 django/db/migrations/autodetector.py | 21 | 38| 185 | 27223 | 50304 | 
| 70 | 22 django/db/migrations/loader.py | 338 | 363| 214 | 27437 | 50304 | 
| 71 | 22 django/core/management/commands/showmigrations.py | 1 | 54| 321 | 27758 | 50304 | 
| 72 | 23 django/core/management/commands/check.py | 47 | 84| 233 | 27991 | 50795 | 
| 73 | 24 django/db/utils.py | 237 | 279| 322 | 28313 | 52694 | 
| 74 | 24 django/db/migrations/recorder.py | 1 | 22| 148 | 28461 | 52694 | 
| 75 | 24 django/db/migrations/loader.py | 1 | 58| 414 | 28875 | 52694 | 
| 76 | 24 django/db/migrations/graph.py | 45 | 60| 121 | 28996 | 52694 | 
| 77 | 25 django/contrib/admin/migrations/0001_initial.py | 1 | 77| 363 | 29359 | 53057 | 
| 78 | 25 django/db/migrations/graph.py | 63 | 99| 337 | 29696 | 53057 | 
| 79 | 25 django/db/migrations/graph.py | 269 | 292| 183 | 29879 | 53057 | 
| 80 | 25 django/db/migrations/autodetector.py | 600 | 776| 1231 | 31110 | 53057 | 
| 81 | 25 django/db/migrations/executor.py | 73 | 92| 172 | 31282 | 53057 | 
| 82 | 26 django/conf/global_settings.py | 645 | 669| 184 | 31466 | 58934 | 
| 83 | 26 django/db/migrations/questioner.py | 217 | 247| 252 | 31718 | 58934 | 
| 84 | 26 django/core/checks/model_checks.py | 135 | 159| 268 | 31986 | 58934 | 
| 85 | 26 django/db/migrations/autodetector.py | 1078 | 1099| 188 | 32174 | 58934 | 
| 86 | 26 django/core/management/commands/showmigrations.py | 134 | 177| 340 | 32514 | 58934 | 
| 87 | 26 django/core/management/commands/migrate.py | 432 | 487| 409 | 32923 | 58934 | 
| 88 | 27 django/db/migrations/state.py | 973 | 989| 140 | 33063 | 67102 | 
| 89 | 27 django/core/checks/model_checks.py | 93 | 116| 170 | 33233 | 67102 | 
| 90 | 27 django/db/migrations/autodetector.py | 1220 | 1307| 719 | 33952 | 67102 | 
| 91 | 27 django/db/migrations/autodetector.py | 1101 | 1218| 982 | 34934 | 67102 | 
| 92 | 27 django/core/checks/model_checks.py | 161 | 185| 267 | 35201 | 67102 | 
| 93 | 27 django/db/migrations/autodetector.py | 1309 | 1344| 232 | 35433 | 67102 | 
| 94 | 28 django/db/migrations/migration.py | 200 | 240| 292 | 35725 | 69009 | 
| 95 | 29 django/db/migrations/__init__.py | 1 | 3| 0 | 35725 | 69033 | 
| 96 | 29 django/db/migrations/executor.py | 174 | 234| 568 | 36293 | 69033 | 
| 97 | 30 django/db/migrations/operations/special.py | 182 | 209| 254 | 36547 | 70606 | 
| 98 | 31 django/db/backends/base/creation.py | 327 | 351| 282 | 36829 | 73547 | 
| 99 | 31 django/db/migrations/autodetector.py | 233 | 264| 256 | 37085 | 73547 | 
| 100 | 31 django/db/migrations/loader.py | 73 | 139| 553 | 37638 | 73547 | 
| 101 | 31 django/db/migrations/questioner.py | 57 | 87| 255 | 37893 | 73547 | 
| 102 | 31 django/db/migrations/executor.py | 236 | 261| 206 | 38099 | 73547 | 
| 103 | 32 django/db/models/base.py | 1523 | 1558| 273 | 38372 | 92098 | 
| 104 | 33 django/contrib/sites/migrations/0001_initial.py | 1 | 45| 210 | 38582 | 92308 | 
| 105 | 33 django/db/migrations/autodetector.py | 908 | 981| 623 | 39205 | 92308 | 
| 106 | 33 django/db/migrations/writer.py | 1 | 115| 888 | 40093 | 92308 | 
| 107 | 33 django/db/migrations/autodetector.py | 380 | 401| 199 | 40292 | 92308 | 
| 108 | 34 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 40292 | 92409 | 
| 109 | 35 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 40409 | 92526 | 
| 110 | 35 django/core/management/commands/check.py | 1 | 45| 263 | 40672 | 92526 | 
| 111 | 35 django/db/models/base.py | 2256 | 2447| 1302 | 41974 | 92526 | 
| 112 | 36 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 42201 | 92753 | 
| 113 | 36 django/db/migrations/graph.py | 294 | 312| 163 | 42364 | 92753 | 
| 114 | 36 django/db/migrations/loader.py | 365 | 386| 159 | 42523 | 92753 | 
| 115 | 36 django/db/migrations/executor.py | 147 | 172| 239 | 42762 | 92753 | 
| 116 | 36 django/db/migrations/autodetector.py | 90 | 102| 119 | 42881 | 92753 | 
| 117 | 37 django/db/backends/mysql/validation.py | 1 | 36| 246 | 43127 | 93284 | 
| 118 | 37 django/db/migrations/autodetector.py | 484 | 514| 267 | 43394 | 93284 | 
| 119 | 37 django/db/migrations/autodetector.py | 1724 | 1737| 132 | 43526 | 93284 | 
| 120 | 37 django/db/migrations/questioner.py | 126 | 164| 327 | 43853 | 93284 | 
| 121 | 37 django/db/migrations/state.py | 872 | 904| 250 | 44103 | 93284 | 
| 122 | 37 django/db/migrations/writer.py | 206 | 312| 632 | 44735 | 93284 | 
| 123 | 37 django/db/migrations/autodetector.py | 778 | 809| 278 | 45013 | 93284 | 
| 124 | 37 django/db/migrations/loader.py | 199 | 220| 213 | 45226 | 93284 | 
| 125 | 37 django/db/migrations/autodetector.py | 421 | 482| 538 | 45764 | 93284 | 
| 126 | 37 django/db/models/base.py | 1650 | 1684| 248 | 46012 | 93284 | 
| 127 | 37 django/core/checks/model_checks.py | 1 | 90| 671 | 46683 | 93284 | 
| 128 | 37 django/core/checks/model_checks.py | 118 | 133| 176 | 46859 | 93284 | 
| 129 | 37 django/db/migrations/migration.py | 139 | 198| 519 | 47378 | 93284 | 
| 130 | 38 django/core/checks/async_checks.py | 1 | 17| 0 | 47378 | 93378 | 
| 131 | 38 django/db/migrations/state.py | 411 | 435| 213 | 47591 | 93378 | 
| 132 | 39 django/contrib/admin/checks.py | 1244 | 1273| 196 | 47787 | 102910 | 
| 133 | 40 django/db/migrations/operations/models.py | 1 | 18| 137 | 47924 | 110623 | 
| 134 | 40 django/db/migrations/questioner.py | 189 | 215| 238 | 48162 | 110623 | 
| 135 | 41 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 28| 143 | 48305 | 110766 | 
| 136 | 41 django/db/migrations/recorder.py | 24 | 46| 145 | 48450 | 110766 | 
| 137 | 42 django/db/backends/mysql/base.py | 276 | 312| 259 | 48709 | 114278 | 
| 138 | 42 django/core/management/base.py | 460 | 554| 665 | 49374 | 114278 | 
| 139 | 43 django/core/checks/templates.py | 1 | 47| 311 | 49685 | 114758 | 
| 140 | 44 django/db/backends/base/base.py | 499 | 594| 604 | 50289 | 120341 | 
| 141 | 45 django/contrib/contenttypes/checks.py | 1 | 25| 130 | 50419 | 120602 | 
| 142 | 45 django/db/migrations/graph.py | 124 | 157| 308 | 50727 | 120602 | 
| 143 | 46 django/core/management/commands/flush.py | 31 | 93| 498 | 51225 | 121305 | 
| 144 | 47 django/contrib/auth/checks.py | 1 | 104| 728 | 51953 | 122821 | 
| 145 | 47 django/contrib/admin/checks.py | 55 | 173| 772 | 52725 | 122821 | 
| 146 | 47 django/db/migrations/autodetector.py | 1397 | 1430| 289 | 53014 | 122821 | 
| 147 | 48 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 47| 225 | 53239 | 123046 | 
| 148 | 48 django/core/management/base.py | 38 | 72| 233 | 53472 | 123046 | 
| 149 | 49 django/core/checks/security/base.py | 226 | 239| 127 | 53599 | 125235 | 
| 150 | 49 django/db/migrations/migration.py | 1 | 92| 730 | 54329 | 125235 | 
| 151 | 50 django/db/migrations/operations/base.py | 1 | 106| 749 | 55078 | 126277 | 
| 152 | 50 django/db/migrations/operations/special.py | 119 | 133| 139 | 55217 | 126277 | 
| 153 | 51 django/db/models/options.py | 1 | 57| 347 | 55564 | 133865 | 
| 154 | 51 django/db/models/options.py | 388 | 410| 161 | 55725 | 133865 | 
| 155 | 51 django/db/migrations/migration.py | 94 | 137| 372 | 56097 | 133865 | 
| 156 | 51 django/db/models/base.py | 2157 | 2238| 588 | 56685 | 133865 | 
| 157 | 52 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 19| 0 | 56685 | 133944 | 
| 158 | 53 django/db/backends/mysql/features.py | 86 | 160| 597 | 57282 | 136315 | 
| 159 | 54 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 57282 | 136391 | 
| 160 | 55 django/core/management/commands/makemessages.py | 326 | 407| 819 | 58101 | 142412 | 
| 161 | 55 django/contrib/auth/checks.py | 107 | 221| 786 | 58887 | 142412 | 
| 162 | 56 django/contrib/contenttypes/apps.py | 1 | 23| 159 | 59046 | 142571 | 
| 163 | 56 django/db/migrations/state.py | 181 | 238| 598 | 59644 | 142571 | 
| 164 | 57 django/core/checks/__init__.py | 1 | 48| 327 | 59971 | 142898 | 
| 165 | 58 django/db/backends/sqlite3/base.py | 233 | 352| 888 | 60859 | 145988 | 
| 166 | 59 django/db/models/fields/__init__.py | 1245 | 1282| 245 | 61104 | 164723 | 
| 167 | 59 django/db/migrations/operations/base.py | 108 | 147| 295 | 61399 | 164723 | 
| 168 | 59 django/db/migrations/state.py | 651 | 677| 270 | 61669 | 164723 | 
| 169 | 59 django/db/migrations/executor.py | 94 | 145| 454 | 62123 | 164723 | 
| 170 | 60 django/db/models/fields/mixins.py | 31 | 60| 178 | 62301 | 165071 | 
| 171 | 60 django/db/migrations/autodetector.py | 210 | 231| 238 | 62539 | 165071 | 
| 172 | 61 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 44| 232 | 62771 | 165303 | 
| 173 | 61 django/contrib/admin/checks.py | 1322 | 1351| 176 | 62947 | 165303 | 
| 174 | 62 django/db/backends/mysql/schema.py | 1 | 43| 471 | 63418 | 167257 | 
| 175 | 62 django/db/migrations/state.py | 126 | 140| 163 | 63581 | 167257 | 
| 176 | 62 django/db/migrations/loader.py | 141 | 167| 220 | 63801 | 167257 | 
| 177 | 63 django/core/checks/database.py | 1 | 15| 0 | 63801 | 167326 | 
| 178 | 64 django/core/checks/caches.py | 22 | 58| 294 | 64095 | 167853 | 
| 179 | 64 django/conf/global_settings.py | 513 | 644| 807 | 64902 | 167853 | 
| 180 | 64 django/db/migrations/operations/models.py | 570 | 594| 213 | 65115 | 167853 | 
| 181 | 64 django/db/models/base.py | 1995 | 2048| 355 | 65470 | 167853 | 
| 182 | 64 django/db/models/base.py | 1 | 64| 346 | 65816 | 167853 | 
| 183 | 64 django/db/migrations/autodetector.py | 1027 | 1076| 384 | 66200 | 167853 | 
| 184 | 64 django/db/migrations/operations/models.py | 689 | 741| 320 | 66520 | 167853 | 
| 185 | 64 django/db/migrations/state.py | 397 | 409| 136 | 66656 | 167853 | 
| 186 | 65 django/contrib/auth/apps.py | 1 | 31| 224 | 66880 | 168077 | 
| 187 | 66 django/db/migrations/optimizer.py | 1 | 38| 344 | 67224 | 168669 | 
| 188 | 66 django/core/management/sql.py | 1 | 39| 235 | 67459 | 168669 | 
| 189 | 66 django/contrib/admin/checks.py | 810 | 824| 127 | 67586 | 168669 | 
| 190 | 67 django/utils/deprecation.py | 87 | 140| 382 | 67968 | 169740 | 
| 191 | 67 django/db/migrations/questioner.py | 166 | 187| 188 | 68156 | 169740 | 
| 192 | 67 django/core/management/commands/makemessages.py | 408 | 476| 499 | 68655 | 169740 | 
| 193 | 67 django/db/migrations/operations/special.py | 63 | 117| 396 | 69051 | 169740 | 
| 194 | 67 django/db/models/base.py | 1592 | 1617| 184 | 69235 | 169740 | 
| 195 | 67 django/db/migrations/operations/models.py | 113 | 134| 164 | 69399 | 169740 | 
| 196 | 67 django/core/management/base.py | 421 | 458| 300 | 69699 | 169740 | 
| 197 | 67 django/db/migrations/state.py | 958 | 971| 138 | 69837 | 169740 | 
| 198 | 67 django/contrib/admin/checks.py | 460 | 478| 137 | 69974 | 169740 | 
| 199 | 67 django/db/migrations/loader.py | 60 | 71| 116 | 70090 | 169740 | 
| 200 | 68 django/db/backends/base/features.py | 362 | 380| 173 | 70263 | 172846 | 
| 201 | 69 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 29| 158 | 70421 | 173004 | 
| 202 | 69 django/db/migrations/graph.py | 314 | 334| 180 | 70601 | 173004 | 
| 203 | 70 django/db/models/constraints.py | 62 | 125| 522 | 71123 | 175932 | 
| 204 | 70 django/db/migrations/operations/models.py | 1068 | 1108| 337 | 71460 | 175932 | 
| 205 | 70 django/contrib/contenttypes/checks.py | 28 | 47| 129 | 71589 | 175932 | 
| 206 | 71 django/db/models/fields/related.py | 1418 | 1448| 172 | 71761 | 190542 | 
| 207 | 71 django/db/migrations/operations/models.py | 670 | 686| 159 | 71920 | 190542 | 
| 208 | 71 django/db/migrations/operations/models.py | 559 | 568| 129 | 72049 | 190542 | 


### Hint

```
I think this makes sense.
```

## Patch

```diff
diff --git a/django/core/management/commands/makemigrations.py b/django/core/management/commands/makemigrations.py
--- a/django/core/management/commands/makemigrations.py
+++ b/django/core/management/commands/makemigrations.py
@@ -70,7 +70,10 @@ def add_arguments(self, parser):
             "--check",
             action="store_true",
             dest="check_changes",
-            help="Exit with a non-zero status if model changes are missing migrations.",
+            help=(
+                "Exit with a non-zero status if model changes are missing migrations "
+                "and don't actually write them."
+            ),
         )
         parser.add_argument(
             "--scriptable",
@@ -248,12 +251,12 @@ def handle(self, *app_labels, **options):
                 else:
                     self.log("No changes detected")
         else:
+            if check_changes:
+                sys.exit(1)
             if self.update:
                 self.write_to_last_migration_files(changes)
             else:
                 self.write_migration_files(changes)
-            if check_changes:
-                sys.exit(1)
 
     def write_to_last_migration_files(self, changes):
         loader = MigrationLoader(connections[DEFAULT_DB_ALIAS])

```

## Test Patch

```diff
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -2391,9 +2391,10 @@ def test_makemigrations_check(self):
         makemigrations --check should exit with a non-zero status when
         there are changes to an app requiring migrations.
         """
-        with self.temporary_migration_module():
+        with self.temporary_migration_module() as tmpdir:
             with self.assertRaises(SystemExit):
                 call_command("makemigrations", "--check", "migrations", verbosity=0)
+            self.assertFalse(os.path.exists(tmpdir))
 
         with self.temporary_migration_module(
             module="migrations.test_migrations_no_changes"

```


## Code snippets

### 1 - django/core/management/commands/makemigrations.py:

Start line: 193, End line: 256

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        # ... other code

        if self.interactive:
            questioner = InteractiveMigrationQuestioner(
                specified_apps=app_labels,
                dry_run=self.dry_run,
                prompt_output=self.log_output,
            )
        else:
            questioner = NonInteractiveMigrationQuestioner(
                specified_apps=app_labels,
                dry_run=self.dry_run,
                verbosity=self.verbosity,
                log=self.log,
            )
        # Set up autodetector
        autodetector = MigrationAutodetector(
            loader.project_state(),
            ProjectState.from_apps(apps),
            questioner,
        )

        # If they want to make an empty migration, make one for each app
        if self.empty:
            if not app_labels:
                raise CommandError(
                    "You must supply at least one app label when using --empty."
                )
            # Make a fake changes() result we can pass to arrange_for_graph
            changes = {app: [Migration("custom", app)] for app in app_labels}
            changes = autodetector.arrange_for_graph(
                changes=changes,
                graph=loader.graph,
                migration_name=self.migration_name,
            )
            self.write_migration_files(changes)
            return

        # Detect changes
        changes = autodetector.changes(
            graph=loader.graph,
            trim_to_apps=app_labels or None,
            convert_apps=app_labels or None,
            migration_name=self.migration_name,
        )

        if not changes:
            # No changes? Tell them.
            if self.verbosity >= 1:
                if app_labels:
                    if len(app_labels) == 1:
                        self.log("No changes detected in app '%s'" % app_labels.pop())
                    else:
                        self.log(
                            "No changes detected in apps '%s'"
                            % ("', '".join(app_labels))
                        )
                else:
                    self.log("No changes detected")
        else:
            if self.update:
                self.write_to_last_migration_files(changes)
            else:
                self.write_migration_files(changes)
            if check_changes:
                sys.exit(1)
```
### 2 - django/core/management/commands/makemigrations.py:

Start line: 101, End line: 191

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        self.written_files = []
        self.verbosity = options["verbosity"]
        self.interactive = options["interactive"]
        self.dry_run = options["dry_run"]
        self.merge = options["merge"]
        self.empty = options["empty"]
        self.migration_name = options["name"]
        if self.migration_name and not self.migration_name.isidentifier():
            raise CommandError("The migration name must be a valid Python identifier.")
        self.include_header = options["include_header"]
        check_changes = options["check_changes"]
        self.scriptable = options["scriptable"]
        self.update = options["update"]
        # If logs and prompts are diverted to stderr, remove the ERROR style.
        if self.scriptable:
            self.stderr.style_func = None

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
        aliases_to_check = (
            connections if settings.DATABASE_ROUTERS else [DEFAULT_DB_ALIAS]
        )
        for alias in sorted(aliases_to_check):
            connection = connections[alias]
            if connection.settings_dict["ENGINE"] != "django.db.backends.dummy" and any(
                # At least one model must be migrated to the database.
                router.allow_migrate(
                    connection.alias, app_label, model_name=model._meta.object_name
                )
                for app_label in consistency_check_labels
                for model in apps.get_app_config(app_label).get_models()
            ):
                try:
                    loader.check_consistent_history(connection)
                except OperationalError as error:
                    warnings.warn(
                        "Got an error checking a consistent migration history "
                        "performed for database connection '%s': %s" % (alias, error),
                        RuntimeWarning,
                    )
        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any and they don't want to merge
        conflicts = loader.detect_conflicts()

        # If app_labels is specified, filter out conflicting migrations for
        # unspecified apps.
        if app_labels:
            conflicts = {
                app_label: conflict
                for app_label, conflict in conflicts.items()
                if app_label in app_labels
            }

        if conflicts and not self.merge:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app) for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they want to merge and there's nothing to merge, then politely exit
        if self.merge and not conflicts:
            self.log("No conflicts detected to merge.")
            return

        # If they want to merge and there is something to merge, then
        # divert into the merge code
        if self.merge and conflicts:
            return self.handle_merge(loader, conflicts)
        # ... other code
```
### 3 - django/core/management/commands/makemigrations.py:

Start line: 26, End line: 99

```python
class Command(BaseCommand):
    help = "Creates new migration(s) for apps."

    def add_arguments(self, parser):
        parser.add_argument(
            "args",
            metavar="app_label",
            nargs="*",
            help="Specify the app label(s) to create migrations for.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Just show what migrations would be made; don't actually write them.",
        )
        parser.add_argument(
            "--merge",
            action="store_true",
            help="Enable fixing of migration conflicts.",
        )
        parser.add_argument(
            "--empty",
            action="store_true",
            help="Create an empty migration.",
        )
        parser.add_argument(
            "--noinput",
            "--no-input",
            action="store_false",
            dest="interactive",
            help="Tells Django to NOT prompt the user for input of any kind.",
        )
        parser.add_argument(
            "-n",
            "--name",
            help="Use this name for migration file(s).",
        )
        parser.add_argument(
            "--no-header",
            action="store_false",
            dest="include_header",
            help="Do not add header comments to new migration file(s).",
        )
        parser.add_argument(
            "--check",
            action="store_true",
            dest="check_changes",
            help="Exit with a non-zero status if model changes are missing migrations.",
        )
        parser.add_argument(
            "--scriptable",
            action="store_true",
            dest="scriptable",
            help=(
                "Divert log output and input prompts to stderr, writing only "
                "paths of generated migration files to stdout."
            ),
        )
        parser.add_argument(
            "--update",
            action="store_true",
            dest="update",
            help=(
                "Merge model changes into the latest migration and optimize the "
                "resulting operations."
            ),
        )

    @property
    def log_output(self):
        return self.stderr if self.scriptable else self.stdout

    def log(self, msg):
        self.log_output.write(msg)
```
### 4 - django/core/management/commands/makemigrations.py:

Start line: 402, End line: 512

```python
class Command(BaseCommand):

    def handle_merge(self, loader, conflicts):
        """
        Handles merging together conflicted migrations interactively,
        if it's safe; otherwise, advises on how to fix it.
        """
        if self.interactive:
            questioner = InteractiveMigrationQuestioner(prompt_output=self.log_output)
        else:
            questioner = MigrationQuestioner(defaults={"ask_merge": True})

        for app_label, migration_names in conflicts.items():
            # Grab out the migrations in question, and work out their
            # common ancestor.
            merge_migrations = []
            for migration_name in migration_names:
                migration = loader.get_migration(app_label, migration_name)
                migration.ancestry = [
                    mig
                    for mig in loader.graph.forwards_plan((app_label, migration_name))
                    if mig[0] == migration.app_label
                ]
                merge_migrations.append(migration)

            def all_items_equal(seq):
                return all(item == seq[0] for item in seq[1:])

            merge_migrations_generations = zip(*(m.ancestry for m in merge_migrations))
            common_ancestor_count = sum(
                1
                for common_ancestor_generation in takewhile(
                    all_items_equal, merge_migrations_generations
                )
            )
            if not common_ancestor_count:
                raise ValueError(
                    "Could not find common ancestor of %s" % migration_names
                )
            # Now work out the operations along each divergent branch
            for migration in merge_migrations:
                migration.branch = migration.ancestry[common_ancestor_count:]
                migrations_ops = (
                    loader.get_migration(node_app, node_name).operations
                    for node_app, node_name in migration.branch
                )
                migration.merged_operations = sum(migrations_ops, [])
            # In future, this could use some of the Optimizer code
            # (can_optimize_through) to automatically see if they're
            # mergeable. For now, we always just prompt the user.
            if self.verbosity > 0:
                self.log(self.style.MIGRATE_HEADING("Merging %s" % app_label))
                for migration in merge_migrations:
                    self.log(self.style.MIGRATE_LABEL("  Branch %s" % migration.name))
                    for operation in migration.merged_operations:
                        self.log("    - %s" % operation.describe())
            if questioner.ask_merge(app_label):
                # If they still want to merge it, then write out an empty
                # file depending on the migrations needing merging.
                numbers = [
                    MigrationAutodetector.parse_number(migration.name)
                    for migration in merge_migrations
                ]
                try:
                    biggest_number = max(x for x in numbers if x is not None)
                except ValueError:
                    biggest_number = 1
                subclass = type(
                    "Migration",
                    (Migration,),
                    {
                        "dependencies": [
                            (app_label, migration.name)
                            for migration in merge_migrations
                        ],
                    },
                )
                parts = ["%04i" % (biggest_number + 1)]
                if self.migration_name:
                    parts.append(self.migration_name)
                else:
                    parts.append("merge")
                    leaf_names = "_".join(
                        sorted(migration.name for migration in merge_migrations)
                    )
                    if len(leaf_names) > 47:
                        parts.append(get_migration_name_timestamp())
                    else:
                        parts.append(leaf_names)
                migration_name = "_".join(parts)
                new_migration = subclass(migration_name, app_label)
                writer = MigrationWriter(new_migration, self.include_header)

                if not self.dry_run:
                    # Write the merge migrations file to the disk
                    with open(writer.path, "w", encoding="utf-8") as fh:
                        fh.write(writer.as_string())
                    run_formatters([writer.path])
                    if self.verbosity > 0:
                        self.log("\nCreated new merge migration %s" % writer.path)
                        if self.scriptable:
                            self.stdout.write(writer.path)
                elif self.verbosity == 3:
                    # Alternatively, makemigrations --merge --dry-run --verbosity 3
                    # will log the merge migrations rather than saving the file
                    # to the disk.
                    self.log(
                        self.style.MIGRATE_HEADING(
                            "Full merge migrations file '%s':" % writer.filename
                        )
                    )
                    self.log(writer.as_string())
```
### 5 - django/core/management/commands/migrate.py:

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
### 6 - django/db/migrations/questioner.py:

Start line: 291, End line: 342

```python
class NonInteractiveMigrationQuestioner(MigrationQuestioner):
    def __init__(
        self,
        defaults=None,
        specified_apps=None,
        dry_run=None,
        verbosity=1,
        log=None,
    ):
        self.verbosity = verbosity
        self.log = log
        super().__init__(
            defaults=defaults,
            specified_apps=specified_apps,
            dry_run=dry_run,
        )

    def log_lack_of_migration(self, field_name, model_name, reason):
        if self.verbosity > 0:
            self.log(
                f"Field '{field_name}' on model '{model_name}' not migrated: "
                f"{reason}."
            )

    def ask_not_null_addition(self, field_name, model_name):
        # We can't ask the user, so act like the user aborted.
        self.log_lack_of_migration(
            field_name,
            model_name,
            "it is impossible to add a non-nullable field without specifying "
            "a default",
        )
        sys.exit(3)

    def ask_not_null_alteration(self, field_name, model_name):
        # We can't ask the user, so set as not provided.
        self.log(
            f"Field '{field_name}' on model '{model_name}' given a default of "
            f"NOT PROVIDED and must be corrected."
        )
        return NOT_PROVIDED

    def ask_auto_now_add_addition(self, field_name, model_name):
        # We can't ask the user, so act like the user aborted.
        self.log_lack_of_migration(
            field_name,
            model_name,
            "it is impossible to add a field with 'auto_now_add=True' without "
            "specifying a default",
        )
        sys.exit(3)
```
### 7 - django/db/migrations/loader.py:

Start line: 307, End line: 336

```python
class MigrationLoader:

    def check_consistent_history(self, connection):
        """
        Raise InconsistentMigrationHistory if any applied migrations have
        unapplied dependencies.
        """
        recorder = MigrationRecorder(connection)
        applied = recorder.applied_migrations()
        for migration in applied:
            # If the migration is unknown, skip it.
            if migration not in self.graph.nodes:
                continue
            for parent in self.graph.node_map[migration].parents:
                if parent not in applied:
                    # Skip unapplied squashed migrations that have all of their
                    # `replaces` applied.
                    if parent in self.replacements:
                        if all(
                            m in applied for m in self.replacements[parent].replaces
                        ):
                            continue
                    raise InconsistentMigrationHistory(
                        "Migration {}.{} is applied before its dependency "
                        "{}.{} on database '{}'.".format(
                            migration[0],
                            migration[1],
                            parent[0],
                            parent[1],
                            connection.alias,
                        )
                    )
```
### 8 - django/db/migrations/executor.py:

Start line: 290, End line: 305

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
### 9 - django/core/management/commands/migrate.py:

Start line: 96, End line: 189

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        database = options["database"]
        if not options["skip_checks"]:
            self.check(databases=[database])

        self.verbosity = options["verbosity"]
        self.interactive = options["interactive"]

        # Import the 'management' module within each installed app, to register
        # dispatcher events.
        for app_config in apps.get_app_configs():
            if module_has_submodule(app_config.module, "management"):
                import_module(".management", app_config.name)

        # Get the database we're operating from
        connection = connections[database]

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
                "%s in %s" % (", ".join(names), app) for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they supplied command line arguments, work out what they mean.
        run_syncdb = options["run_syncdb"]
        target_app_labels_only = True
        if options["app_label"]:
            # Validate app_label.
            app_label = options["app_label"]
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                raise CommandError(str(err))
            if run_syncdb:
                if app_label in executor.loader.migrated_apps:
                    raise CommandError(
                        "Can't use run_syncdb with app '%s' as it has migrations."
                        % app_label
                    )
            elif app_label not in executor.loader.migrated_apps:
                raise CommandError("App '%s' does not have migrations." % app_label)

        if options["app_label"] and options["migration_name"]:
            migration_name = options["migration_name"]
            if migration_name == "zero":
                targets = [(app_label, None)]
            else:
                try:
                    migration = executor.loader.get_migration_by_prefix(
                        app_label, migration_name
                    )
                except AmbiguityError:
                    raise CommandError(
                        "More than one migration matches '%s' in app '%s'. "
                        "Please be more specific." % (migration_name, app_label)
                    )
                except KeyError:
                    raise CommandError(
                        "Cannot find a migration matching '%s' from app '%s'."
                        % (migration_name, app_label)
                    )
                target = (app_label, migration.name)
                # Partially applied squashed migrations are not included in the
                # graph, use the last replacement instead.
                if (
                    target not in executor.loader.graph.nodes
                    and target in executor.loader.replacements
                ):
                    incomplete_migration = executor.loader.replacements[target]
                    target = incomplete_migration.replaces[-1]
                targets = [target]
            target_app_labels_only = False
        elif options["app_label"]:
            targets = [
                key for key in executor.loader.graph.leaf_nodes() if key[0] == app_label
            ]
        else:
            targets = executor.loader.graph.leaf_nodes()
        # ... other code
```
### 10 - django/core/management/commands/migrate.py:

Start line: 17, End line: 94

```python
class Command(BaseCommand):
    help = (
        "Updates database schema. Manages both apps with migrations and those without."
    )
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-checks",
            action="store_true",
            help="Skip system checks.",
        )
        parser.add_argument(
            "app_label",
            nargs="?",
            help="App label of an application to synchronize the state.",
        )
        parser.add_argument(
            "migration_name",
            nargs="?",
            help="Database state will be brought to the state after that "
            'migration. Use the name "zero" to unapply all migrations.',
        )
        parser.add_argument(
            "--noinput",
            "--no-input",
            action="store_false",
            dest="interactive",
            help="Tells Django to NOT prompt the user for input of any kind.",
        )
        parser.add_argument(
            "--database",
            default=DEFAULT_DB_ALIAS,
            help=(
                'Nominates a database to synchronize. Defaults to the "default" '
                "database."
            ),
        )
        parser.add_argument(
            "--fake",
            action="store_true",
            help="Mark migrations as run without actually running them.",
        )
        parser.add_argument(
            "--fake-initial",
            action="store_true",
            help=(
                "Detect if tables already exist and fake-apply initial migrations if "
                "so. Make sure that the current database schema matches your initial "
                "migration before using this flag. Django will only check for an "
                "existing table name."
            ),
        )
        parser.add_argument(
            "--plan",
            action="store_true",
            help="Shows a list of the migration actions that will be performed.",
        )
        parser.add_argument(
            "--run-syncdb",
            action="store_true",
            help="Creates tables for apps without migrations.",
        )
        parser.add_argument(
            "--check",
            action="store_true",
            dest="check_unapplied",
            help=(
                "Exits with a non-zero status if unapplied migrations exist and does "
                "not actually apply migrations."
            ),
        )
        parser.add_argument(
            "--prune",
            action="store_true",
            dest="prune",
            help="Delete nonexistent migrations from the django_migrations table.",
        )
```
### 11 - django/core/management/commands/makemigrations.py:

Start line: 1, End line: 23

```python
import os
import sys
import warnings
from itertools import takewhile

from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.utils import run_formatters
from django.db import DEFAULT_DB_ALIAS, OperationalError, connections, router
from django.db.migrations import Migration
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.migration import SwappableTuple
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import (
    InteractiveMigrationQuestioner,
    MigrationQuestioner,
    NonInteractiveMigrationQuestioner,
)
from django.db.migrations.state import ProjectState
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.migrations.writer import MigrationWriter
```
### 12 - django/core/management/commands/makemigrations.py:

Start line: 258, End line: 328

```python
class Command(BaseCommand):

    def write_to_last_migration_files(self, changes):
        loader = MigrationLoader(connections[DEFAULT_DB_ALIAS])
        new_changes = {}
        update_previous_migration_paths = {}
        for app_label, app_migrations in changes.items():
            # Find last migration.
            leaf_migration_nodes = loader.graph.leaf_nodes(app=app_label)
            if len(leaf_migration_nodes) == 0:
                raise CommandError(
                    f"App {app_label} has no migration, cannot update last migration."
                )
            leaf_migration_node = leaf_migration_nodes[0]
            # Multiple leaf nodes have already been checked earlier in command.
            leaf_migration = loader.graph.nodes[leaf_migration_node]
            # Updated migration cannot be a squash migration, a dependency of
            # another migration, and cannot be already applied.
            if leaf_migration.replaces:
                raise CommandError(
                    f"Cannot update squash migration '{leaf_migration}'."
                )
            if leaf_migration_node in loader.applied_migrations:
                raise CommandError(
                    f"Cannot update applied migration '{leaf_migration}'."
                )
            depending_migrations = [
                migration
                for migration in loader.disk_migrations.values()
                if leaf_migration_node in migration.dependencies
            ]
            if depending_migrations:
                formatted_migrations = ", ".join(
                    [f"'{migration}'" for migration in depending_migrations]
                )
                raise CommandError(
                    f"Cannot update migration '{leaf_migration}' that migrations "
                    f"{formatted_migrations} depend on."
                )
            # Build new migration.
            for migration in app_migrations:
                leaf_migration.operations.extend(migration.operations)

                for dependency in migration.dependencies:
                    if isinstance(dependency, SwappableTuple):
                        if settings.AUTH_USER_MODEL == dependency.setting:
                            leaf_migration.dependencies.append(
                                ("__setting__", "AUTH_USER_MODEL")
                            )
                        else:
                            leaf_migration.dependencies.append(dependency)
                    elif dependency[0] != migration.app_label:
                        leaf_migration.dependencies.append(dependency)
            # Optimize migration.
            optimizer = MigrationOptimizer()
            leaf_migration.operations = optimizer.optimize(
                leaf_migration.operations, app_label
            )
            # Update name.
            previous_migration_path = MigrationWriter(leaf_migration).path
            suggested_name = (
                leaf_migration.name[:4] + "_" + leaf_migration.suggest_name()
            )
            if leaf_migration.name == suggested_name:
                new_name = leaf_migration.name + "_updated"
            else:
                new_name = suggested_name
            leaf_migration.name = new_name
            # Register overridden migration.
            new_changes[app_label] = [leaf_migration]
            update_previous_migration_paths[app_label] = previous_migration_path

        self.write_migration_files(new_changes, update_previous_migration_paths)
```
### 17 - django/core/management/commands/makemigrations.py:

Start line: 330, End line: 400

```python
class Command(BaseCommand):

    def write_migration_files(self, changes, update_previous_migration_paths=None):
        """
        Take a changes dict and write them out as migration files.
        """
        directory_created = {}
        for app_label, app_migrations in changes.items():
            if self.verbosity >= 1:
                self.log(self.style.MIGRATE_HEADING("Migrations for '%s':" % app_label))
            for migration in app_migrations:
                # Describe the migration
                writer = MigrationWriter(migration, self.include_header)
                if self.verbosity >= 1:
                    # Display a relative path if it's below the current working
                    # directory, or an absolute path otherwise.
                    migration_string = self.get_relative_path(writer.path)
                    self.log("  %s\n" % self.style.MIGRATE_LABEL(migration_string))
                    for operation in migration.operations:
                        self.log("    - %s" % operation.describe())
                    if self.scriptable:
                        self.stdout.write(migration_string)
                if not self.dry_run:
                    # Write the migrations file to the disk.
                    migrations_directory = os.path.dirname(writer.path)
                    if not directory_created.get(app_label):
                        os.makedirs(migrations_directory, exist_ok=True)
                        init_path = os.path.join(migrations_directory, "__init__.py")
                        if not os.path.isfile(init_path):
                            open(init_path, "w").close()
                        # We just do this once per app
                        directory_created[app_label] = True
                    migration_string = writer.as_string()
                    with open(writer.path, "w", encoding="utf-8") as fh:
                        fh.write(migration_string)
                        self.written_files.append(writer.path)
                    if update_previous_migration_paths:
                        prev_path = update_previous_migration_paths[app_label]
                        rel_prev_path = self.get_relative_path(prev_path)
                        if writer.needs_manual_porting:
                            migration_path = self.get_relative_path(writer.path)
                            self.log(
                                self.style.WARNING(
                                    f"Updated migration {migration_path} requires "
                                    f"manual porting.\n"
                                    f"Previous migration {rel_prev_path} was kept and "
                                    f"must be deleted after porting functions manually."
                                )
                            )
                        else:
                            os.remove(prev_path)
                            self.log(f"Deleted {rel_prev_path}")
                elif self.verbosity == 3:
                    # Alternatively, makemigrations --dry-run --verbosity 3
                    # will log the migrations rather than saving the file to
                    # the disk.
                    self.log(
                        self.style.MIGRATE_HEADING(
                            "Full migrations file '%s':" % writer.filename
                        )
                    )
                    self.log(writer.as_string())
        run_formatters(self.written_files)

    @staticmethod
    def get_relative_path(path):
        try:
            migration_string = os.path.relpath(path)
        except ValueError:
            migration_string = path
        if migration_string.startswith(".."):
            migration_string = path
        return migration_string
```
