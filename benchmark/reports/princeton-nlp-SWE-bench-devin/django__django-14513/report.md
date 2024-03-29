# django__django-14513

| **django/django** | `de4f6201835043ba664e8dcbdceffc4b0955ce29` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 8015 |
| **Any found context length** | 3468 |
| **Avg pos** | 32.0 |
| **Min pos** | 7 |
| **Max pos** | 18 |
| **Top file pos** | 5 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/showmigrations.py b/django/core/management/commands/showmigrations.py
--- a/django/core/management/commands/showmigrations.py
+++ b/django/core/management/commands/showmigrations.py
@@ -4,6 +4,7 @@
 from django.core.management.base import BaseCommand
 from django.db import DEFAULT_DB_ALIAS, connections
 from django.db.migrations.loader import MigrationLoader
+from django.db.migrations.recorder import MigrationRecorder
 
 
 class Command(BaseCommand):
@@ -69,6 +70,8 @@ def show_list(self, connection, app_names=None):
         """
         # Load migrations from disk/DB
         loader = MigrationLoader(connection, ignore_no_migrations=True)
+        recorder = MigrationRecorder(connection)
+        recorded_migrations = recorder.applied_migrations()
         graph = loader.graph
         # If we were passed a list of apps, validate it
         if app_names:
@@ -91,7 +94,11 @@ def show_list(self, connection, app_names=None):
                         applied_migration = loader.applied_migrations.get(plan_node)
                         # Mark it as applied/unapplied
                         if applied_migration:
-                            output = ' [X] %s' % title
+                            if plan_node in recorded_migrations:
+                                output = ' [X] %s' % title
+                            else:
+                                title += " Run 'manage.py migrate' to finish recording."
+                                output = ' [-] %s' % title
                             if self.verbosity >= 2 and hasattr(applied_migration, 'applied'):
                                 output += ' (applied at %s)' % applied_migration.applied.strftime('%Y-%m-%d %H:%M:%S')
                             self.stdout.write(output)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/showmigrations.py | 7 | 7 | 18 | 5 | 8015
| django/core/management/commands/showmigrations.py | 72 | 72 | 7 | 5 | 3468
| django/core/management/commands/showmigrations.py | 94 | 94 | 7 | 5 | 3468


## Problem Statement

```
Better Indication of Squash Migration State in showmigrations
Description
	
In the discussion of #25231 (​https://github.com/django/django/pull/5112) it became clear that there was a disconnect between the current output of showmigrations and the actual recorded applied state of squashed migrations.
Currently if all of the replaced/original migrations have been run, showmigrations will output that the related squashed migration has been applied with an [X] in the output even if that has not yet been recorded by the migration recorder. However, it is currently a requirement that migrate be run to record this applied state for the squashed migration before the original migrations are removed. If a deployment process is looking for an empty [ ] to know to run the migration then this may trip it up.
This case is to consider an output for showmigrations which can indicate that this migration has only been "soft" applied, that is applied but not recorded yet.
Changes to the planner for such an output may also impact #24900.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/migrations/executor.py | 263 | 278| 165 | 165 | 3280 | 
| 2 | 2 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 956 | 5153 | 
| 3 | 2 django/db/migrations/executor.py | 280 | 373| 843 | 1799 | 5153 | 
| 4 | 2 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 2453 | 5153 | 
| 5 | 3 django/db/migrations/recorder.py | 46 | 97| 390 | 2843 | 5830 | 
| 6 | 4 django/db/migrations/loader.py | 288 | 312| 205 | 3048 | 8935 | 
| **-> 7 <-** | **5 django/core/management/commands/showmigrations.py** | 65 | 103| 420 | 3468 | 10130 | 
| 8 | 5 django/db/migrations/loader.py | 314 | 335| 208 | 3676 | 10130 | 
| 9 | 6 django/db/migrations/autodetector.py | 103 | 197| 782 | 4458 | 21710 | 
| 10 | 6 django/core/management/commands/squashmigrations.py | 206 | 219| 112 | 4570 | 21710 | 
| 11 | 7 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 5392 | 24549 | 
| 12 | **7 django/core/management/commands/showmigrations.py** | 105 | 148| 340 | 5732 | 24549 | 
| 13 | 8 django/db/migrations/state.py | 170 | 194| 213 | 5945 | 30291 | 
| 14 | 8 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 6295 | 30291 | 
| 15 | 8 django/db/migrations/loader.py | 207 | 286| 783 | 7078 | 30291 | 
| 16 | 9 django/core/management/commands/migrate.py | 272 | 304| 349 | 7427 | 33555 | 
| 17 | 9 django/db/migrations/autodetector.py | 1167 | 1201| 296 | 7723 | 33555 | 
| **-> 18 <-** | **9 django/core/management/commands/showmigrations.py** | 1 | 40| 292 | 8015 | 33555 | 
| 19 | 9 django/db/migrations/executor.py | 238 | 261| 225 | 8240 | 33555 | 
| 20 | 10 django/db/migrations/questioner.py | 226 | 239| 123 | 8363 | 35612 | 
| 21 | 10 django/db/migrations/state.py | 352 | 376| 266 | 8629 | 35612 | 
| 22 | 10 django/core/management/commands/migrate.py | 253 | 270| 212 | 8841 | 35612 | 
| 23 | 10 django/core/management/commands/migrate.py | 169 | 251| 812 | 9653 | 35612 | 
| 24 | 11 django/db/migrations/migration.py | 91 | 126| 338 | 9991 | 37434 | 
| 25 | **11 django/core/management/commands/showmigrations.py** | 42 | 63| 158 | 10149 | 37434 | 
| 26 | 11 django/db/migrations/autodetector.py | 258 | 329| 748 | 10897 | 37434 | 
| 27 | 11 django/core/management/commands/makemigrations.py | 154 | 192| 313 | 11210 | 37434 | 
| 28 | 11 django/core/management/commands/makemigrations.py | 239 | 326| 873 | 12083 | 37434 | 
| 29 | 11 django/db/migrations/state.py | 110 | 156| 367 | 12450 | 37434 | 
| 30 | 11 django/core/management/commands/migrate.py | 355 | 378| 186 | 12636 | 37434 | 
| 31 | 11 django/db/migrations/executor.py | 64 | 80| 168 | 12804 | 37434 | 
| 32 | 11 django/db/migrations/recorder.py | 23 | 44| 145 | 12949 | 37434 | 
| 33 | 12 django/core/management/sql.py | 38 | 54| 128 | 13077 | 37790 | 
| 34 | 12 django/db/migrations/state.py | 158 | 168| 132 | 13209 | 37790 | 
| 35 | 12 django/db/migrations/migration.py | 128 | 177| 481 | 13690 | 37790 | 
| 36 | 12 django/db/migrations/recorder.py | 1 | 21| 148 | 13838 | 37790 | 
| 37 | 13 django/db/migrations/graph.py | 300 | 320| 180 | 14018 | 40393 | 
| 38 | 13 django/db/migrations/state.py | 81 | 108| 277 | 14295 | 40393 | 
| 39 | 13 django/db/migrations/executor.py | 213 | 236| 203 | 14498 | 40393 | 
| 40 | 13 django/db/migrations/autodetector.py | 1203 | 1228| 245 | 14743 | 40393 | 
| 41 | 13 django/db/migrations/loader.py | 156 | 182| 291 | 15034 | 40393 | 
| 42 | 13 django/db/migrations/executor.py | 152 | 211| 567 | 15601 | 40393 | 
| 43 | 13 django/core/management/commands/migrate.py | 71 | 167| 834 | 16435 | 40393 | 
| 44 | 13 django/db/migrations/graph.py | 61 | 97| 337 | 16772 | 40393 | 
| 45 | 13 django/db/migrations/autodetector.py | 1 | 35| 294 | 17066 | 40393 | 
| 46 | 13 django/db/migrations/state.py | 196 | 226| 335 | 17401 | 40393 | 
| 47 | 14 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 17660 | 41026 | 
| 48 | 14 django/db/migrations/autodetector.py | 244 | 257| 178 | 17838 | 41026 | 
| 49 | 14 django/db/migrations/state.py | 658 | 674| 146 | 17984 | 41026 | 
| 50 | 14 django/db/migrations/autodetector.py | 533 | 684| 1174 | 19158 | 41026 | 
| 51 | 14 django/core/management/commands/migrate.py | 21 | 69| 407 | 19565 | 41026 | 
| 52 | 14 django/db/migrations/autodetector.py | 1294 | 1329| 311 | 19876 | 41026 | 
| 53 | 15 django/db/migrations/operations/special.py | 44 | 60| 180 | 20056 | 42584 | 
| 54 | 15 django/db/migrations/graph.py | 44 | 58| 121 | 20177 | 42584 | 
| 55 | 15 django/db/migrations/autodetector.py | 1244 | 1292| 436 | 20613 | 42584 | 
| 56 | 16 django/db/migrations/operations/base.py | 1 | 109| 804 | 21417 | 43614 | 
| 57 | 16 django/db/migrations/state.py | 228 | 246| 173 | 21590 | 43614 | 
| 58 | 16 django/db/migrations/executor.py | 127 | 150| 235 | 21825 | 43614 | 
| 59 | 16 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 22204 | 43614 | 
| 60 | 16 django/db/migrations/autodetector.py | 352 | 366| 138 | 22342 | 43614 | 
| 61 | 16 django/db/migrations/questioner.py | 1 | 53| 451 | 22793 | 43614 | 
| 62 | 16 django/db/migrations/autodetector.py | 431 | 460| 265 | 23058 | 43614 | 
| 63 | 17 django/core/management/base.py | 479 | 511| 281 | 23339 | 48251 | 
| 64 | 17 django/db/migrations/state.py | 304 | 350| 444 | 23783 | 48251 | 
| 65 | 17 django/db/migrations/autodetector.py | 37 | 47| 120 | 23903 | 48251 | 
| 66 | 17 django/db/migrations/state.py | 559 | 591| 250 | 24153 | 48251 | 
| 67 | 17 django/db/migrations/graph.py | 282 | 298| 159 | 24312 | 48251 | 
| 68 | 17 django/db/migrations/autodetector.py | 223 | 242| 234 | 24546 | 48251 | 
| 69 | 17 django/db/migrations/loader.py | 337 | 354| 152 | 24698 | 48251 | 
| 70 | 17 django/db/migrations/autodetector.py | 1230 | 1242| 131 | 24829 | 48251 | 
| 71 | 17 django/db/migrations/operations/special.py | 181 | 204| 246 | 25075 | 48251 | 
| 72 | 17 django/core/management/commands/migrate.py | 1 | 18| 140 | 25215 | 48251 | 
| 73 | 17 django/db/migrations/migration.py | 179 | 219| 283 | 25498 | 48251 | 
| 74 | 17 django/db/migrations/operations/base.py | 111 | 141| 229 | 25727 | 48251 | 
| 75 | 17 django/core/management/commands/makemigrations.py | 24 | 59| 284 | 26011 | 48251 | 
| 76 | 17 django/db/migrations/migration.py | 1 | 89| 726 | 26737 | 48251 | 
| 77 | 17 django/db/migrations/state.py | 289 | 301| 125 | 26862 | 48251 | 
| 78 | 18 django/db/migrations/writer.py | 118 | 199| 744 | 27606 | 50498 | 
| 79 | 18 django/db/migrations/graph.py | 193 | 243| 450 | 28056 | 50498 | 
| 80 | 18 django/db/migrations/autodetector.py | 1105 | 1142| 317 | 28373 | 50498 | 
| 81 | 18 django/core/management/commands/migrate.py | 306 | 353| 396 | 28769 | 50498 | 
| 82 | 19 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 28931 | 50660 | 
| 83 | 19 django/db/migrations/loader.py | 55 | 66| 116 | 29047 | 50660 | 
| 84 | 19 django/core/management/commands/makemigrations.py | 194 | 237| 434 | 29481 | 50660 | 
| 85 | 19 django/db/migrations/autodetector.py | 719 | 802| 679 | 30160 | 50660 | 
| 86 | 19 django/db/migrations/loader.py | 68 | 132| 551 | 30711 | 50660 | 
| 87 | 19 django/db/migrations/state.py | 643 | 656| 138 | 30849 | 50660 | 
| 88 | 19 django/db/migrations/executor.py | 1 | 62| 496 | 31345 | 50660 | 
| 89 | 19 django/db/migrations/autodetector.py | 1144 | 1165| 231 | 31576 | 50660 | 
| 90 | 19 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 31731 | 50660 | 
| 91 | 19 django/db/migrations/state.py | 1 | 26| 197 | 31928 | 50660 | 
| 92 | 20 django/db/migrations/exceptions.py | 1 | 55| 249 | 32177 | 50910 | 
| 93 | 20 django/db/migrations/state.py | 248 | 286| 331 | 32508 | 50910 | 
| 94 | 20 django/db/migrations/graph.py | 99 | 120| 192 | 32700 | 50910 | 
| 95 | 20 django/db/migrations/state.py | 378 | 404| 244 | 32944 | 50910 | 
| 96 | 20 django/db/migrations/loader.py | 1 | 53| 409 | 33353 | 50910 | 
| 97 | 20 django/db/migrations/autodetector.py | 1037 | 1053| 188 | 33541 | 50910 | 
| 98 | 20 django/db/migrations/questioner.py | 186 | 204| 237 | 33778 | 50910 | 
| 99 | 20 django/db/migrations/autodetector.py | 331 | 350| 196 | 33974 | 50910 | 
| 100 | 20 django/core/management/sql.py | 1 | 35| 228 | 34202 | 50910 | 
| 101 | 20 django/db/migrations/autodetector.py | 997 | 1013| 188 | 34390 | 50910 | 
| 102 | 21 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 34658 | 51178 | 
| 103 | 21 django/db/migrations/graph.py | 122 | 155| 308 | 34966 | 51178 | 
| 104 | 21 django/db/migrations/autodetector.py | 804 | 854| 576 | 35542 | 51178 | 
| 105 | 21 django/db/migrations/autodetector.py | 199 | 221| 239 | 35781 | 51178 | 
| 106 | 21 django/db/migrations/executor.py | 82 | 125| 402 | 36183 | 51178 | 
| 107 | 22 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 36490 | 51485 | 
| 108 | 23 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 36601 | 51596 | 
| 109 | 23 django/db/migrations/state.py | 464 | 558| 809 | 37410 | 51596 | 
| 110 | 24 django/contrib/admin/migrations/0001_initial.py | 1 | 47| 314 | 37724 | 51910 | 
| 111 | 24 django/db/migrations/graph.py | 259 | 280| 179 | 37903 | 51910 | 
| 112 | 24 django/db/migrations/autodetector.py | 1015 | 1035| 134 | 38037 | 51910 | 
| 113 | 25 django/db/utils.py | 255 | 297| 322 | 38359 | 53917 | 
| 114 | 25 django/db/migrations/autodetector.py | 1055 | 1075| 136 | 38495 | 53917 | 
| 115 | 25 django/db/migrations/autodetector.py | 462 | 514| 465 | 38960 | 53917 | 
| 116 | 26 django/db/migrations/operations/models.py | 317 | 342| 290 | 39250 | 60901 | 
| 117 | 26 django/db/migrations/writer.py | 2 | 115| 886 | 40136 | 60901 | 
| 118 | 26 django/db/migrations/operations/models.py | 880 | 919| 378 | 40514 | 60901 | 
| 119 | 26 django/db/migrations/operations/models.py | 620 | 637| 163 | 40677 | 60901 | 
| 120 | 27 django/db/migrations/operations/fields.py | 85 | 95| 124 | 40801 | 64024 | 
| 121 | 27 django/db/migrations/autodetector.py | 1077 | 1103| 277 | 41078 | 64024 | 
| 122 | 27 django/db/migrations/graph.py | 245 | 257| 151 | 41229 | 64024 | 
| 123 | 27 django/db/migrations/loader.py | 184 | 205| 213 | 41442 | 64024 | 
| 124 | 27 django/db/migrations/loader.py | 134 | 154| 211 | 41653 | 64024 | 
| 125 | 27 django/db/migrations/operations/models.py | 844 | 877| 308 | 41961 | 64024 | 
| 126 | 27 django/db/migrations/state.py | 620 | 641| 227 | 42188 | 64024 | 
| 127 | 27 django/db/migrations/state.py | 29 | 56| 233 | 42421 | 64024 | 
| 128 | 27 django/db/migrations/autodetector.py | 686 | 717| 278 | 42699 | 64024 | 
| 129 | 28 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 54 | 70| 109 | 42808 | 64577 | 
| 130 | 28 django/db/migrations/operations/fields.py | 216 | 234| 185 | 42993 | 64577 | 
| 131 | 28 django/db/migrations/graph.py | 157 | 191| 328 | 43321 | 64577 | 
| 132 | 29 django/core/checks/model_checks.py | 89 | 110| 168 | 43489 | 66362 | 
| 133 | 29 django/db/migrations/writer.py | 201 | 301| 619 | 44108 | 66362 | 
| 134 | 29 django/db/migrations/operations/special.py | 63 | 114| 390 | 44498 | 66362 | 
| 135 | 29 django/db/migrations/operations/models.py | 463 | 492| 302 | 44800 | 66362 | 
| 136 | 29 django/db/migrations/questioner.py | 83 | 106| 187 | 44987 | 66362 | 
| 137 | 29 django/db/migrations/autodetector.py | 915 | 995| 871 | 45858 | 66362 | 
| 138 | 29 django/db/migrations/questioner.py | 55 | 80| 220 | 46078 | 66362 | 
| 139 | 29 django/db/migrations/operations/models.py | 395 | 415| 213 | 46291 | 66362 | 
| 140 | 29 django/db/migrations/autodetector.py | 516 | 532| 186 | 46477 | 66362 | 
| 141 | 30 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 46672 | 66557 | 
| 142 | 31 django/contrib/auth/migrations/0001_initial.py | 1 | 104| 843 | 47515 | 67400 | 
| 143 | 31 django/db/migrations/operations/models.py | 344 | 393| 493 | 48008 | 67400 | 
| 144 | 32 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 48199 | 67591 | 
| 145 | 32 django/db/migrations/operations/models.py | 524 | 533| 129 | 48328 | 67591 | 
| 146 | 33 django/db/models/base.py | 1332 | 1357| 184 | 48512 | 84903 | 
| 147 | 33 django/db/migrations/operations/special.py | 116 | 130| 139 | 48651 | 84903 | 
| 148 | 34 django/db/migrations/__init__.py | 1 | 3| 0 | 48651 | 84927 | 
| 149 | 35 django/db/migrations/optimizer.py | 40 | 70| 248 | 48899 | 85517 | 
| 150 | 35 django/db/migrations/operations/models.py | 535 | 552| 168 | 49067 | 85517 | 
| 151 | 35 django/db/migrations/autodetector.py | 894 | 913| 184 | 49251 | 85517 | 
| 152 | 35 django/core/checks/model_checks.py | 112 | 127| 176 | 49427 | 85517 | 
| 153 | 35 django/db/migrations/graph.py | 1 | 41| 227 | 49654 | 85517 | 
| 154 | 35 django/db/migrations/state.py | 407 | 462| 474 | 50128 | 85517 | 
| 155 | 36 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 50245 | 85634 | 
| 156 | 36 django/db/migrations/operations/special.py | 1 | 42| 313 | 50558 | 85634 | 
| 157 | 36 django/db/migrations/autodetector.py | 368 | 429| 552 | 51110 | 85634 | 
| 158 | 37 django/db/models/options.py | 343 | 363| 158 | 51268 | 93001 | 
| 159 | 38 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 51475 | 93208 | 
| 160 | 38 django/db/migrations/state.py | 593 | 618| 228 | 51703 | 93208 | 
| 161 | 38 django/db/migrations/operations/models.py | 602 | 618| 215 | 51918 | 93208 | 
| 162 | 38 django/db/migrations/autodetector.py | 89 | 101| 116 | 52034 | 93208 | 
| 163 | 38 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 51| 444 | 52478 | 93208 | 
| 164 | 38 django/db/migrations/operations/models.py | 577 | 600| 183 | 52661 | 93208 | 
| 165 | 39 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 52878 | 93425 | 
| 166 | 39 django/db/migrations/state.py | 59 | 78| 209 | 53087 | 93425 | 
| 167 | 39 django/db/migrations/operations/models.py | 640 | 696| 366 | 53453 | 93425 | 
| 168 | 39 django/db/models/options.py | 365 | 388| 198 | 53651 | 93425 | 
| 169 | 40 django/core/management/commands/dumpdata.py | 81 | 153| 624 | 54275 | 95343 | 
| 170 | 41 django/db/backends/base/creation.py | 301 | 322| 258 | 54533 | 98131 | 
| 171 | 41 django/db/migrations/questioner.py | 161 | 184| 246 | 54779 | 98131 | 
| 172 | 41 django/db/migrations/operations/fields.py | 97 | 109| 130 | 54909 | 98131 | 
| 173 | 41 django/db/migrations/operations/models.py | 1 | 39| 242 | 55151 | 98131 | 
| 174 | 41 django/db/migrations/operations/fields.py | 111 | 121| 127 | 55278 | 98131 | 
| 175 | 41 django/db/migrations/operations/models.py | 125 | 248| 853 | 56131 | 98131 | 
| 176 | 41 django/core/checks/model_checks.py | 155 | 176| 263 | 56394 | 98131 | 
| 177 | 42 django/core/management/commands/makemessages.py | 61 | 81| 146 | 56540 | 103718 | 
| 178 | 43 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 56540 | 103819 | 
| 179 | 44 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 56540 | 103895 | 
| 180 | 44 django/db/migrations/optimizer.py | 1 | 38| 344 | 56884 | 103895 | 
| 181 | 45 django/contrib/postgres/operations.py | 37 | 61| 197 | 57081 | 106281 | 
| 182 | 46 django/contrib/contenttypes/apps.py | 1 | 24| 161 | 57242 | 106442 | 
| 183 | 47 django/utils/deconstruct.py | 1 | 56| 385 | 57627 | 106827 | 
| 184 | 47 django/core/checks/model_checks.py | 129 | 153| 268 | 57895 | 106827 | 
| 185 | 48 django/core/management/commands/flush.py | 27 | 83| 486 | 58381 | 107514 | 
| 186 | 49 django/db/models/fields/related.py | 183 | 196| 140 | 58521 | 121483 | 
| 187 | 49 django/core/checks/model_checks.py | 178 | 211| 332 | 58853 | 121483 | 
| 188 | 49 django/db/models/options.py | 1 | 35| 300 | 59153 | 121483 | 
| 189 | 50 django/conf/global_settings.py | 496 | 646| 930 | 60083 | 127170 | 
| 190 | 50 django/db/migrations/operations/models.py | 107 | 123| 156 | 60239 | 127170 | 
| 191 | 50 django/db/migrations/operations/fields.py | 248 | 270| 188 | 60427 | 127170 | 


## Patch

```diff
diff --git a/django/core/management/commands/showmigrations.py b/django/core/management/commands/showmigrations.py
--- a/django/core/management/commands/showmigrations.py
+++ b/django/core/management/commands/showmigrations.py
@@ -4,6 +4,7 @@
 from django.core.management.base import BaseCommand
 from django.db import DEFAULT_DB_ALIAS, connections
 from django.db.migrations.loader import MigrationLoader
+from django.db.migrations.recorder import MigrationRecorder
 
 
 class Command(BaseCommand):
@@ -69,6 +70,8 @@ def show_list(self, connection, app_names=None):
         """
         # Load migrations from disk/DB
         loader = MigrationLoader(connection, ignore_no_migrations=True)
+        recorder = MigrationRecorder(connection)
+        recorded_migrations = recorder.applied_migrations()
         graph = loader.graph
         # If we were passed a list of apps, validate it
         if app_names:
@@ -91,7 +94,11 @@ def show_list(self, connection, app_names=None):
                         applied_migration = loader.applied_migrations.get(plan_node)
                         # Mark it as applied/unapplied
                         if applied_migration:
-                            output = ' [X] %s' % title
+                            if plan_node in recorded_migrations:
+                                output = ' [X] %s' % title
+                            else:
+                                title += " Run 'manage.py migrate' to finish recording."
+                                output = ' [-] %s' % title
                             if self.verbosity >= 2 and hasattr(applied_migration, 'applied'):
                                 output += ' (applied at %s)' % applied_migration.applied.strftime('%Y-%m-%d %H:%M:%S')
                             self.stdout.write(output)

```

## Test Patch

```diff
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -928,6 +928,15 @@ def test_migrate_record_squashed(self):
         recorder = MigrationRecorder(connection)
         recorder.record_applied("migrations", "0001_initial")
         recorder.record_applied("migrations", "0002_second")
+        out = io.StringIO()
+        call_command('showmigrations', 'migrations', stdout=out, no_color=True)
+        self.assertEqual(
+            "migrations\n"
+            " [-] 0001_squashed_0002 (2 squashed migrations) "
+            "run 'manage.py migrate' to finish recording.\n",
+            out.getvalue().lower(),
+        )
+
         out = io.StringIO()
         call_command("migrate", "migrations", verbosity=0)
         call_command("showmigrations", "migrations", stdout=out, no_color=True)

```


## Code snippets

### 1 - django/db/migrations/executor.py:

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
### 2 - django/core/management/commands/squashmigrations.py:

Start line: 45, End line: 134

```python
class Command(BaseCommand):

    def handle(self, **options):

        self.verbosity = options['verbosity']
        self.interactive = options['interactive']
        app_label = options['app_label']
        start_migration_name = options['start_migration_name']
        migration_name = options['migration_name']
        no_optimize = options['no_optimize']
        squashed_name = options['squashed_name']
        include_header = options['include_header']
        # Validate app_label.
        try:
            apps.get_app_config(app_label)
        except LookupError as err:
            raise CommandError(str(err))
        # Load the current graph state, check the app and migration they asked for exists
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
            for al, mn in loader.graph.forwards_plan((migration.app_label, migration.name))
            if al == migration.app_label
        ]

        if start_migration_name:
            start_migration = self.find_migration(loader, app_label, start_migration_name)
            start = loader.get_migration(start_migration.app_label, start_migration.name)
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
            self.stdout.write(self.style.MIGRATE_HEADING("Will squash the following migrations:"))
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
                    "You cannot squash squashed migrations! Please transition "
                    "it to a normal migration first: "
                    "https://docs.djangoproject.com/en/%s/topics/migrations/#squashing-migrations" % get_docs_version()
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
### 3 - django/db/migrations/executor.py:

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
### 4 - django/core/management/commands/squashmigrations.py:

Start line: 136, End line: 204

```python
class Command(BaseCommand):

    def handle(self, **options):
        # ... other code

        if no_optimize:
            if self.verbosity > 0:
                self.stdout.write(self.style.MIGRATE_HEADING("(Skipping optimization.)"))
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
                        "  Optimized from %s operations to %s operations." %
                        (len(operations), len(new_operations))
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
        subclass = type("Migration", (migrations.Migration,), {
            "dependencies": dependencies,
            "operations": new_operations,
            "replaces": replaces,
        })
        if start_migration_name:
            if squashed_name:
                # Use the name from --squashed-name.
                prefix, _ = start_migration.name.split('_', 1)
                name = '%s_%s' % (prefix, squashed_name)
            else:
                # Generate a name.
                name = '%s_squashed_%s' % (start_migration.name, migration.name)
            new_migration = subclass(name, app_label)
        else:
            name = '0001_%s' % (squashed_name or 'squashed_%s' % migration.name)
            new_migration = subclass(name, app_label)
            new_migration.initial = True

        # Write out the new migration file
        writer = MigrationWriter(new_migration, include_header)
        with open(writer.path, "w", encoding='utf-8') as fh:
            fh.write(writer.as_string())

        if self.verbosity > 0:
            self.stdout.write(
                self.style.MIGRATE_HEADING('Created new squashed migration %s' % writer.path) + '\n'
                '  You should commit this migration but leave the old ones in place;\n'
                '  the new migration will be used for new installs. Once you are sure\n'
                '  all instances of the codebase have applied the migrations you squashed,\n'
                '  you can delete them.'
            )
            if writer.needs_manual_porting:
                self.stdout.write(
                    self.style.MIGRATE_HEADING('Manual porting required') + '\n'
                    '  Your migrations contained functions that must be manually copied over,\n'
                    '  as we could not safely copy their implementation.\n'
                    '  See the comment at the top of the squashed migration for details.'
                )
```
### 5 - django/db/migrations/recorder.py:

Start line: 46, End line: 97

```python
class MigrationRecorder:

    def __init__(self, connection):
        self.connection = connection

    @property
    def migration_qs(self):
        return self.Migration.objects.using(self.connection.alias)

    def has_table(self):
        """Return True if the django_migrations table exists."""
        with self.connection.cursor() as cursor:
            tables = self.connection.introspection.table_names(cursor)
        return self.Migration._meta.db_table in tables

    def ensure_schema(self):
        """Ensure the table exists and has the correct schema."""
        # If the table's there, that's fine - we've never changed its schema
        # in the codebase.
        if self.has_table():
            return
        # Make the table
        try:
            with self.connection.schema_editor() as editor:
                editor.create_model(self.Migration)
        except DatabaseError as exc:
            raise MigrationSchemaMissing("Unable to create the django_migrations table (%s)" % exc)

    def applied_migrations(self):
        """
        Return a dict mapping (app_name, migration_name) to Migration instances
        for all applied migrations.
        """
        if self.has_table():
            return {(migration.app, migration.name): migration for migration in self.migration_qs}
        else:
            # If the django_migrations table doesn't exist, then no migrations
            # are applied.
            return {}

    def record_applied(self, app, name):
        """Record that a migration was applied."""
        self.ensure_schema()
        self.migration_qs.create(app=app, name=name)

    def record_unapplied(self, app, name):
        """Record that a migration was unapplied."""
        self.ensure_schema()
        self.migration_qs.filter(app=app, name=name).delete()

    def flush(self):
        """Delete all migration records. Useful for testing migrations."""
        self.migration_qs.all().delete()
```
### 6 - django/db/migrations/loader.py:

Start line: 288, End line: 312

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
                        if all(m in applied for m in self.replacements[parent].replaces):
                            continue
                    raise InconsistentMigrationHistory(
                        "Migration {}.{} is applied before its dependency "
                        "{}.{} on database '{}'.".format(
                            migration[0], migration[1], parent[0], parent[1],
                            connection.alias,
                        )
                    )
```
### 7 - django/core/management/commands/showmigrations.py:

Start line: 65, End line: 103

```python
class Command(BaseCommand):

    def show_list(self, connection, app_names=None):
        """
        Show a list of all migrations on the system, or only those of
        some named apps.
        """
        # Load migrations from disk/DB
        loader = MigrationLoader(connection, ignore_no_migrations=True)
        graph = loader.graph
        # If we were passed a list of apps, validate it
        if app_names:
            self._validate_app_names(loader, app_names)
        # Otherwise, show all apps in alphabetic order
        else:
            app_names = sorted(loader.migrated_apps)
        # For each app, print its migrations in order from oldest (roots) to
        # newest (leaves).
        for app_name in app_names:
            self.stdout.write(app_name, self.style.MIGRATE_LABEL)
            shown = set()
            for node in graph.leaf_nodes(app_name):
                for plan_node in graph.forwards_plan(node):
                    if plan_node not in shown and plan_node[0] == app_name:
                        # Give it a nice title if it's a squashed one
                        title = plan_node[1]
                        if graph.nodes[plan_node].replaces:
                            title += " (%s squashed migrations)" % len(graph.nodes[plan_node].replaces)
                        applied_migration = loader.applied_migrations.get(plan_node)
                        # Mark it as applied/unapplied
                        if applied_migration:
                            output = ' [X] %s' % title
                            if self.verbosity >= 2 and hasattr(applied_migration, 'applied'):
                                output += ' (applied at %s)' % applied_migration.applied.strftime('%Y-%m-%d %H:%M:%S')
                            self.stdout.write(output)
                        else:
                            self.stdout.write(" [ ] %s" % title)
                        shown.add(plan_node)
            # If we didn't print anything, then a small message
            if not shown:
                self.stdout.write(" (no migrations)", self.style.ERROR)
```
### 8 - django/db/migrations/loader.py:

Start line: 314, End line: 335

```python
class MigrationLoader:

    def detect_conflicts(self):
        """
        Look through the loaded graph and detect any conflicts - apps
        with more than one leaf migration. Return a dict of the app labels
        that conflict with the migration names that conflict.
        """
        seen_apps = {}
        conflicting_apps = set()
        for app_label, migration_name in self.graph.leaf_nodes():
            if app_label in seen_apps:
                conflicting_apps.add(app_label)
            seen_apps.setdefault(app_label, set()).add(migration_name)
        return {app_label: sorted(seen_apps[app_label]) for app_label in conflicting_apps}

    def project_state(self, nodes=None, at_end=True):
        """
        Return a ProjectState object representing the most recent state
        that the loaded migrations represent.

        See graph.make_state() for the meaning of "nodes" and "at_end".
        """
        return self.graph.make_state(nodes=nodes, at_end=at_end, real_apps=list(self.unmigrated_apps))
```
### 9 - django/db/migrations/autodetector.py:

Start line: 103, End line: 197

```python
class MigrationAutodetector:

    def _detect_changes(self, convert_apps=None, graph=None):
        """
        Return a dict of migration plans which will achieve the
        change from from_state to to_state. The dict has app labels
        as keys and a list of migrations as values.

        The resulting migrations aren't specially named, but the names
        do matter for dependencies inside the set.

        convert_apps is the list of apps to convert to use migrations
        (i.e. to make initial migrations for, in the usual case)

        graph is an optional argument that, if provided, can help improve
        dependency generation and avoid potential circular dependencies.
        """
        # The first phase is generating all the operations for each app
        # and gathering them into a big per-app list.
        # Then go through that list, order it, and split into migrations to
        # resolve dependencies caused by M2Ms and FKs.
        self.generated_operations = {}
        self.altered_indexes = {}
        self.altered_constraints = {}

        # Prepare some old/new state and model lists, separating
        # proxy models and ignoring unmigrated apps.
        self.old_model_keys = set()
        self.old_proxy_keys = set()
        self.old_unmanaged_keys = set()
        self.new_model_keys = set()
        self.new_proxy_keys = set()
        self.new_unmanaged_keys = set()
        for (app_label, model_name), model_state in self.from_state.models.items():
            if not model_state.options.get('managed', True):
                self.old_unmanaged_keys.add((app_label, model_name))
            elif app_label not in self.from_state.real_apps:
                if model_state.options.get('proxy'):
                    self.old_proxy_keys.add((app_label, model_name))
                else:
                    self.old_model_keys.add((app_label, model_name))

        for (app_label, model_name), model_state in self.to_state.models.items():
            if not model_state.options.get('managed', True):
                self.new_unmanaged_keys.add((app_label, model_name))
            elif (
                app_label not in self.from_state.real_apps or
                (convert_apps and app_label in convert_apps)
            ):
                if model_state.options.get('proxy'):
                    self.new_proxy_keys.add((app_label, model_name))
                else:
                    self.new_model_keys.add((app_label, model_name))

        self.from_state.resolve_fields_and_relations()
        self.to_state.resolve_fields_and_relations()

        # Renames have to come first
        self.generate_renamed_models()

        # Prepare lists of fields and generate through model map
        self._prepare_field_lists()
        self._generate_through_model_map()

        # Generate non-rename model operations
        self.generate_deleted_models()
        self.generate_created_models()
        self.generate_deleted_proxies()
        self.generate_created_proxies()
        self.generate_altered_options()
        self.generate_altered_managers()

        # Create the altered indexes and store them in self.altered_indexes.
        # This avoids the same computation in generate_removed_indexes()
        # and generate_added_indexes().
        self.create_altered_indexes()
        self.create_altered_constraints()
        # Generate index removal operations before field is removed
        self.generate_removed_constraints()
        self.generate_removed_indexes()
        # Generate field operations
        self.generate_renamed_fields()
        self.generate_removed_fields()
        self.generate_added_fields()
        self.generate_altered_fields()
        self.generate_altered_order_with_respect_to()
        self.generate_altered_unique_together()
        self.generate_altered_index_together()
        self.generate_added_indexes()
        self.generate_added_constraints()
        self.generate_altered_db_table()

        self._sort_migrations()
        self._build_migration_list(graph)
        self._optimize_migrations()

        return self.migrations
```
### 10 - django/core/management/commands/squashmigrations.py:

Start line: 206, End line: 219

```python
class Command(BaseCommand):

    def find_migration(self, loader, app_label, name):
        try:
            return loader.get_migration_by_prefix(app_label, name)
        except AmbiguityError:
            raise CommandError(
                "More than one migration matches '%s' in app '%s'. Please be "
                "more specific." % (name, app_label)
            )
        except KeyError:
            raise CommandError(
                "Cannot find a migration matching '%s' from app '%s'." %
                (name, app_label)
            )
```
### 12 - django/core/management/commands/showmigrations.py:

Start line: 105, End line: 148

```python
class Command(BaseCommand):

    def show_plan(self, connection, app_names=None):
        """
        Show all known migrations (or only those of the specified app_names)
        in the order they will be applied.
        """
        # Load migrations from disk/DB
        loader = MigrationLoader(connection)
        graph = loader.graph
        if app_names:
            self._validate_app_names(loader, app_names)
            targets = [key for key in graph.leaf_nodes() if key[0] in app_names]
        else:
            targets = graph.leaf_nodes()
        plan = []
        seen = set()

        # Generate the plan
        for target in targets:
            for migration in graph.forwards_plan(target):
                if migration not in seen:
                    node = graph.node_map[migration]
                    plan.append(node)
                    seen.add(migration)

        # Output
        def print_deps(node):
            out = []
            for parent in sorted(node.parents):
                out.append("%s.%s" % parent.key)
            if out:
                return " ... (%s)" % ", ".join(out)
            return ""

        for node in plan:
            deps = ""
            if self.verbosity >= 2:
                deps = print_deps(node)
            if node.key in loader.applied_migrations:
                self.stdout.write("[X]  %s.%s%s" % (node.key[0], node.key[1], deps))
            else:
                self.stdout.write("[ ]  %s.%s%s" % (node.key[0], node.key[1], deps))
        if not plan:
            self.stdout.write('(no migrations)', self.style.ERROR)
```
### 18 - django/core/management/commands/showmigrations.py:

Start line: 1, End line: 40

```python
import sys

from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import MigrationLoader


class Command(BaseCommand):
    help = "Shows all available migrations for the current project"

    def add_arguments(self, parser):
        parser.add_argument(
            'app_label', nargs='*',
            help='App labels of applications to limit the output to.',
        )
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database to synchronize. Defaults to the "default" database.',
        )

        formats = parser.add_mutually_exclusive_group()
        formats.add_argument(
            '--list', '-l', action='store_const', dest='format', const='list',
            help=(
                'Shows a list of all migrations and which are applied. '
                'With a verbosity level of 2 or above, the applied datetimes '
                'will be included.'
            ),
        )
        formats.add_argument(
            '--plan', '-p', action='store_const', dest='format', const='plan',
            help=(
                'Shows all migrations in the order they will be applied. '
                'With a verbosity level of 2 or above all direct migration dependencies '
                'and reverse dependencies (run_before) will be included.'
            )
        )

        parser.set_defaults(format='list')
```
### 25 - django/core/management/commands/showmigrations.py:

Start line: 42, End line: 63

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
        has_bad_names = False
        for app_name in app_names:
            try:
                apps.get_app_config(app_name)
            except LookupError as err:
                self.stderr.write(str(err))
                has_bad_names = True
        if has_bad_names:
            sys.exit(2)
```
