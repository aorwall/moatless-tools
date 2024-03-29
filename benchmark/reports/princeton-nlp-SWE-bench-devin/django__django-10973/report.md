# django__django-10973

| **django/django** | `ddb293685235fd09e932805771ae97f72e817181` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 472 |
| **Any found context length** | 472 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py
--- a/django/db/backends/postgresql/client.py
+++ b/django/db/backends/postgresql/client.py
@@ -2,17 +2,9 @@
 import signal
 import subprocess
 
-from django.core.files.temp import NamedTemporaryFile
 from django.db.backends.base.client import BaseDatabaseClient
 
 
-def _escape_pgpass(txt):
-    """
-    Escape a fragment of a PostgreSQL .pgpass file.
-    """
-    return txt.replace('\\', '\\\\').replace(':', '\\:')
-
-
 class DatabaseClient(BaseDatabaseClient):
     executable_name = 'psql'
 
@@ -34,38 +26,17 @@ def runshell_db(cls, conn_params):
             args += ['-p', str(port)]
         args += [dbname]
 
-        temp_pgpass = None
         sigint_handler = signal.getsignal(signal.SIGINT)
+        subprocess_env = os.environ.copy()
+        if passwd:
+            subprocess_env['PGPASSWORD'] = str(passwd)
         try:
-            if passwd:
-                # Create temporary .pgpass file.
-                temp_pgpass = NamedTemporaryFile(mode='w+')
-                try:
-                    print(
-                        _escape_pgpass(host) or '*',
-                        str(port) or '*',
-                        _escape_pgpass(dbname) or '*',
-                        _escape_pgpass(user) or '*',
-                        _escape_pgpass(passwd),
-                        file=temp_pgpass,
-                        sep=':',
-                        flush=True,
-                    )
-                    os.environ['PGPASSFILE'] = temp_pgpass.name
-                except UnicodeEncodeError:
-                    # If the current locale can't encode the data, let the
-                    # user input the password manually.
-                    pass
             # Allow SIGINT to pass to psql to abort queries.
             signal.signal(signal.SIGINT, signal.SIG_IGN)
-            subprocess.check_call(args)
+            subprocess.run(args, check=True, env=subprocess_env)
         finally:
             # Restore the original SIGINT handler.
             signal.signal(signal.SIGINT, sigint_handler)
-            if temp_pgpass:
-                temp_pgpass.close()
-                if 'PGPASSFILE' in os.environ:  # unit tests need cleanup
-                    del os.environ['PGPASSFILE']
 
     def runshell(self):
         DatabaseClient.runshell_db(self.connection.get_connection_params())

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/postgresql/client.py | 5 | 15 | 1 | 1 | 472
| django/db/backends/postgresql/client.py | 37 | 68 | 1 | 1 | 472


## Problem Statement

```
Use subprocess.run and PGPASSWORD for client in postgres backend
Description
	
​subprocess.run was added in python 3.5 (which is the minimum version since Django 2.1). This function allows you to pass a custom environment for the subprocess.
Using this in django.db.backends.postgres.client to set PGPASSWORD simplifies the code and makes it more reliable.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/backends/postgresql/client.py** | 1 | 72| 472 | 472 | 472 | 
| 2 | 2 django/db/backends/postgresql/base.py | 1 | 58| 469 | 941 | 2897 | 
| 3 | 3 django/db/backends/mysql/client.py | 1 | 49| 422 | 1363 | 3319 | 
| 4 | 4 django/db/backends/postgresql/utils.py | 1 | 8| 0 | 1363 | 3357 | 
| 5 | 5 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 1700 | 3896 | 
| 6 | 5 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 206 | 1906 | 3896 | 
| 7 | 6 django/contrib/postgres/__init__.py | 1 | 2| 0 | 1906 | 3910 | 
| 8 | 6 django/db/backends/postgresql/base.py | 138 | 175| 330 | 2236 | 3910 | 
| 9 | 6 django/db/backends/postgresql/base.py | 261 | 288| 224 | 2460 | 3910 | 
| 10 | 7 django/contrib/postgres/apps.py | 40 | 67| 249 | 2709 | 4476 | 
| 11 | 8 django/contrib/postgres/functions.py | 1 | 12| 0 | 2709 | 4529 | 
| 12 | 9 django/db/backends/mysql/creation.py | 1 | 29| 219 | 2928 | 5143 | 
| 13 | 10 django/db/backends/base/client.py | 1 | 13| 0 | 2928 | 5248 | 
| 14 | 10 django/db/backends/postgresql/base.py | 177 | 195| 186 | 3114 | 5248 | 
| 15 | 11 django/contrib/auth/management/commands/createsuperuser.py | 63 | 169| 962 | 4076 | 6963 | 
| 16 | 12 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 4314 | 7611 | 
| 17 | 12 django/db/backends/postgresql/base.py | 61 | 137| 778 | 5092 | 7611 | 
| 18 | 12 django/db/backends/postgresql/creation.py | 36 | 51| 174 | 5266 | 7611 | 
| 19 | 13 django/contrib/postgres/forms/__init__.py | 1 | 5| 0 | 5266 | 7653 | 
| 20 | 13 django/contrib/postgres/apps.py | 1 | 17| 129 | 5395 | 7653 | 
| 21 | 14 django/db/backends/sqlite3/client.py | 1 | 13| 0 | 5395 | 7711 | 
| 22 | 15 django/core/management/commands/dbshell.py | 1 | 32| 231 | 5626 | 7942 | 
| 23 | 16 django/contrib/gis/db/backends/postgis/base.py | 1 | 27| 197 | 5823 | 8139 | 
| 24 | 16 django/contrib/auth/management/commands/createsuperuser.py | 1 | 61| 440 | 6263 | 8139 | 
| 25 | 17 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 6263 | 8162 | 
| 26 | 18 django/db/backends/postgresql/operations.py | 1 | 27| 244 | 6507 | 10794 | 
| 27 | 19 django/contrib/auth/admin.py | 1 | 22| 188 | 6695 | 12520 | 
| 28 | 20 django/db/backends/postgresql/features.py | 1 | 70| 571 | 7266 | 13091 | 
| 29 | 21 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 7266 | 13144 | 
| 30 | 22 django/db/backends/mysql/base.py | 1 | 49| 437 | 7703 | 16069 | 
| 31 | 23 django/db/backends/oracle/client.py | 1 | 18| 0 | 7703 | 16171 | 
| 32 | 24 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 7703 | 16191 | 
| 33 | 25 django/contrib/auth/password_validation.py | 1 | 32| 206 | 7909 | 17680 | 
| 34 | 25 django/db/backends/postgresql/base.py | 197 | 259| 461 | 8370 | 17680 | 
| 35 | 25 django/db/backends/postgresql/creation.py | 53 | 78| 248 | 8618 | 17680 | 
| 36 | 25 django/contrib/auth/password_validation.py | 91 | 115| 189 | 8807 | 17680 | 
| 37 | 26 django/db/backends/utils.py | 48 | 64| 176 | 8983 | 19573 | 
| 38 | 27 django/db/utils.py | 1 | 48| 150 | 9133 | 21591 | 
| 39 | 28 django/core/management/commands/runserver.py | 23 | 52| 240 | 9373 | 23042 | 
| 40 | 28 django/core/management/commands/runserver.py | 106 | 162| 517 | 9890 | 23042 | 
| 41 | 29 django/contrib/gis/db/backends/postgis/features.py | 1 | 13| 0 | 9890 | 23132 | 
| 42 | 30 django/contrib/auth/hashers.py | 388 | 404| 127 | 10017 | 27919 | 
| 43 | 31 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 10264 | 28166 | 
| 44 | 32 django/contrib/gis/db/backends/mysql/base.py | 1 | 17| 0 | 10264 | 28275 | 
| 45 | 32 django/contrib/auth/hashers.py | 343 | 358| 146 | 10410 | 28275 | 
| 46 | 32 django/contrib/auth/hashers.py | 229 | 273| 414 | 10824 | 28275 | 
| 47 | 33 django/db/backends/postgresql/schema.py | 1 | 33| 331 | 11155 | 29564 | 
| 48 | 34 django/db/backends/base/base.py | 1 | 21| 125 | 11280 | 34271 | 
| 49 | 34 django/contrib/auth/hashers.py | 450 | 464| 126 | 11406 | 34271 | 
| 50 | 35 django/contrib/gis/db/backends/postgis/operations.py | 1 | 25| 217 | 11623 | 37878 | 
| 51 | 36 django/contrib/gis/db/backends/spatialite/client.py | 1 | 6| 0 | 11623 | 37907 | 
| 52 | 36 django/contrib/auth/admin.py | 128 | 189| 465 | 12088 | 37907 | 
| 53 | 36 django/contrib/auth/hashers.py | 64 | 77| 147 | 12235 | 37907 | 
| 54 | 36 django/contrib/auth/hashers.py | 406 | 416| 127 | 12362 | 37907 | 
| 55 | 37 django/conf/global_settings.py | 145 | 260| 856 | 13218 | 43502 | 
| 56 | 37 django/db/backends/postgresql/operations.py | 201 | 285| 700 | 13918 | 43502 | 
| 57 | 38 django/contrib/auth/models.py | 128 | 158| 273 | 14191 | 46430 | 
| 58 | 39 django/contrib/auth/views.py | 328 | 360| 239 | 14430 | 49082 | 
| 59 | 39 django/contrib/auth/hashers.py | 328 | 341| 120 | 14550 | 49082 | 
| 60 | 39 django/contrib/auth/password_validation.py | 160 | 206| 356 | 14906 | 49082 | 
| 61 | 40 django/contrib/admin/sites.py | 306 | 321| 158 | 15064 | 53198 | 
| 62 | 40 django/db/backends/postgresql/operations.py | 98 | 133| 267 | 15331 | 53198 | 
| 63 | 41 django/db/backends/oracle/creation.py | 220 | 251| 390 | 15721 | 57093 | 
| 64 | 42 django/contrib/auth/base_user.py | 1 | 44| 298 | 16019 | 57977 | 
| 65 | 42 django/contrib/auth/hashers.py | 418 | 447| 290 | 16309 | 57977 | 
| 66 | 42 django/conf/global_settings.py | 490 | 635| 869 | 17178 | 57977 | 
| 67 | 42 django/contrib/auth/hashers.py | 1 | 27| 187 | 17365 | 57977 | 
| 68 | 42 django/contrib/postgres/apps.py | 20 | 37| 188 | 17553 | 57977 | 
| 69 | 43 scripts/manage_translations.py | 1 | 29| 200 | 17753 | 59670 | 
| 70 | 44 django/core/mail/backends/__init__.py | 1 | 2| 0 | 17753 | 59678 | 
| 71 | 45 django/db/__init__.py | 45 | 66| 106 | 17859 | 60121 | 
| 72 | 45 django/db/utils.py | 100 | 128| 310 | 18169 | 60121 | 
| 73 | 46 django/contrib/auth/forms.py | 299 | 340| 276 | 18445 | 63056 | 
| 74 | 46 django/contrib/auth/views.py | 187 | 203| 122 | 18567 | 63056 | 
| 75 | 47 django/db/migrations/operations/special.py | 133 | 179| 304 | 18871 | 64614 | 
| 76 | 48 django/core/management/utils.py | 1 | 28| 185 | 19056 | 65742 | 
| 77 | 48 django/contrib/auth/forms.py | 373 | 425| 345 | 19401 | 65742 | 
| 78 | 49 django/contrib/postgres/indexes.py | 1 | 36| 264 | 19665 | 67133 | 
| 79 | 49 django/contrib/auth/password_validation.py | 35 | 51| 114 | 19779 | 67133 | 
| 80 | 49 django/db/migrations/operations/special.py | 181 | 204| 246 | 20025 | 67133 | 
| 81 | 49 django/contrib/auth/management/commands/createsuperuser.py | 171 | 195| 189 | 20214 | 67133 | 
| 82 | 50 django/contrib/postgres/serializers.py | 1 | 11| 0 | 20214 | 67234 | 
| 83 | 51 django/core/mail/backends/dummy.py | 1 | 11| 0 | 20214 | 67277 | 
| 84 | 52 django/db/backends/oracle/base.py | 37 | 59| 235 | 20449 | 72326 | 
| 85 | 53 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 20599 | 72476 | 
| 86 | 53 django/db/backends/base/base.py | 201 | 274| 519 | 21118 | 72476 | 
| 87 | 53 django/contrib/auth/forms.py | 343 | 370| 183 | 21301 | 72476 | 
| 88 | 53 django/contrib/auth/hashers.py | 215 | 226| 150 | 21451 | 72476 | 
| 89 | 53 django/contrib/auth/views.py | 246 | 282| 339 | 21790 | 72476 | 
| 90 | 53 django/contrib/auth/forms.py | 125 | 153| 235 | 22025 | 72476 | 
| 91 | 53 django/contrib/auth/management/commands/createsuperuser.py | 197 | 212| 139 | 22164 | 72476 | 
| 92 | 54 django/contrib/postgres/operations.py | 38 | 78| 162 | 22326 | 72921 | 
| 93 | 54 django/db/backends/mysql/creation.py | 57 | 68| 154 | 22480 | 72921 | 
| 94 | 55 django/contrib/gis/db/backends/oracle/base.py | 1 | 17| 0 | 22480 | 73024 | 
| 95 | 56 django/contrib/auth/__init__.py | 1 | 59| 402 | 22882 | 74638 | 
| 96 | 57 django/utils/version.py | 1 | 15| 129 | 23011 | 75432 | 
| 97 | 58 django/contrib/auth/tokens.py | 54 | 63| 134 | 23145 | 76216 | 
| 98 | 59 django/db/backends/sqlite3/base.py | 193 | 235| 733 | 23878 | 81514 | 
| 99 | 59 django/db/backends/oracle/base.py | 1 | 34| 232 | 24110 | 81514 | 
| 100 | 59 django/core/management/commands/runserver.py | 66 | 104| 397 | 24507 | 81514 | 
| 101 | 59 django/db/backends/postgresql/operations.py | 157 | 199| 454 | 24961 | 81514 | 
| 102 | 59 django/core/management/commands/runserver.py | 1 | 20| 191 | 25152 | 81514 | 
| 103 | 60 django/contrib/gis/geoip2/base.py | 1 | 21| 141 | 25293 | 83550 | 
| 104 | 61 django/contrib/gis/gdal/base.py | 1 | 7| 0 | 25293 | 83592 | 
| 105 | 62 django/contrib/gis/gdal/libgdal.py | 101 | 124| 158 | 25451 | 84544 | 
| 106 | 63 django/core/management/commands/shell.py | 42 | 81| 401 | 25852 | 85365 | 
| 107 | 64 django/contrib/auth/backends.py | 1 | 43| 308 | 26160 | 86802 | 
| 108 | 65 django/contrib/gis/db/backends/spatialite/adapter.py | 1 | 10| 0 | 26160 | 86868 | 
| 109 | 65 django/contrib/postgres/indexes.py | 160 | 178| 134 | 26294 | 86868 | 
| 110 | 66 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 26432 | 87006 | 
| 111 | 66 django/contrib/auth/hashers.py | 600 | 637| 276 | 26708 | 87006 | 
| 112 | 66 django/db/backends/oracle/creation.py | 1 | 28| 226 | 26934 | 87006 | 
| 113 | 67 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 27129 | 87201 | 
| 114 | 68 django/core/cache/backends/db.py | 97 | 110| 234 | 27363 | 89287 | 
| 115 | 69 django/db/backends/dummy/features.py | 1 | 7| 0 | 27363 | 89319 | 
| 116 | 70 django/contrib/gis/db/backends/postgis/adapter.py | 53 | 66| 131 | 27494 | 89808 | 
| 117 | 70 django/contrib/auth/forms.py | 44 | 62| 124 | 27618 | 89808 | 
| 118 | 71 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 27618 | 89886 | 
| 119 | 71 django/db/__init__.py | 1 | 18| 141 | 27759 | 89886 | 
| 120 | 71 django/db/backends/mysql/base.py | 172 | 224| 427 | 28186 | 89886 | 
| 121 | 71 django/contrib/auth/views.py | 206 | 220| 133 | 28319 | 89886 | 
| 122 | 72 django/db/backends/base/features.py | 116 | 217| 854 | 29173 | 92308 | 
| 123 | 72 django/db/backends/sqlite3/base.py | 165 | 191| 270 | 29443 | 92308 | 
| 124 | 72 django/contrib/gis/db/backends/postgis/operations.py | 98 | 156| 703 | 30146 | 92308 | 
| 125 | 72 django/contrib/auth/base_user.py | 47 | 140| 585 | 30731 | 92308 | 
| 126 | 72 django/db/backends/base/features.py | 1 | 115| 900 | 31631 | 92308 | 
| 127 | 73 django/core/management/commands/testserver.py | 1 | 27| 205 | 31836 | 92742 | 
| 128 | 73 django/contrib/gis/db/backends/postgis/operations.py | 297 | 332| 315 | 32151 | 92742 | 
| 129 | 73 django/contrib/auth/backends.py | 128 | 189| 444 | 32595 | 92742 | 
| 130 | 74 django/__init__.py | 1 | 25| 173 | 32768 | 92915 | 
| 131 | 75 django/contrib/postgres/validators.py | 1 | 21| 181 | 32949 | 93466 | 
| 132 | 75 django/contrib/auth/views.py | 284 | 325| 314 | 33263 | 93466 | 
| 133 | 75 django/core/management/commands/shell.py | 1 | 40| 268 | 33531 | 93466 | 
| 134 | 75 django/db/backends/base/base.py | 516 | 532| 113 | 33644 | 93466 | 
| 135 | 75 django/db/backends/base/base.py | 534 | 581| 300 | 33944 | 93466 | 
| 136 | 75 django/db/backends/oracle/creation.py | 30 | 100| 722 | 34666 | 93466 | 
| 137 | 75 django/contrib/gis/db/backends/postgis/adapter.py | 1 | 51| 362 | 35028 | 93466 | 
| 138 | 75 scripts/manage_translations.py | 176 | 186| 116 | 35144 | 93466 | 
| 139 | 75 django/db/backends/postgresql/operations.py | 79 | 96| 202 | 35346 | 93466 | 
| 140 | 76 django/db/backends/mysql/operations.py | 1 | 31| 236 | 35582 | 96498 | 
| 141 | 77 django/contrib/postgres/utils.py | 1 | 30| 218 | 35800 | 96716 | 
| 142 | 78 django/contrib/gis/db/backends/postgis/schema.py | 1 | 18| 196 | 35996 | 97350 | 
| 143 | 78 django/contrib/auth/admin.py | 40 | 99| 504 | 36500 | 97350 | 
| 144 | 78 django/contrib/gis/db/backends/postgis/operations.py | 158 | 186| 302 | 36802 | 97350 | 
| 145 | 79 django/contrib/gis/db/models/sql/__init__.py | 1 | 8| 0 | 36802 | 97385 | 
| 146 | 80 setup.py | 1 | 66| 508 | 37310 | 98402 | 
| 147 | 80 django/db/backends/oracle/creation.py | 300 | 315| 193 | 37503 | 98402 | 
| 148 | 80 django/conf/global_settings.py | 395 | 489| 787 | 38290 | 98402 | 
| 149 | 80 django/db/backends/mysql/operations.py | 178 | 221| 329 | 38619 | 98402 | 
| 150 | 81 django/db/backends/dummy/base.py | 1 | 47| 270 | 38889 | 98847 | 
| 151 | 82 django/core/management/commands/startproject.py | 1 | 21| 137 | 39026 | 98984 | 
| 152 | 82 django/contrib/postgres/indexes.py | 90 | 112| 218 | 39244 | 98984 | 
| 153 | 83 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 39244 | 99052 | 
| 154 | 84 django/contrib/gis/utils/srs.py | 1 | 77| 691 | 39935 | 99743 | 
| 155 | 85 django/contrib/gis/db/backends/oracle/features.py | 1 | 12| 0 | 39935 | 99823 | 
| 156 | 86 django/db/migrations/utils.py | 1 | 18| 0 | 39935 | 99911 | 
| 157 | 86 django/contrib/gis/db/backends/postgis/operations.py | 84 | 95| 110 | 40045 | 99911 | 
| 158 | 86 django/db/backends/oracle/base.py | 207 | 230| 207 | 40252 | 99911 | 
| 159 | 87 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 40438 | 100213 | 
| 160 | 88 django/db/backends/sqlite3/creation.py | 1 | 46| 356 | 40794 | 101037 | 
| 161 | 89 django/core/mail/backends/locmem.py | 1 | 31| 183 | 40977 | 101221 | 
| 162 | 90 django/contrib/postgres/fields/jsonb.py | 1 | 27| 163 | 41140 | 102455 | 
| 163 | 91 django/db/backends/base/operations.py | 654 | 674| 187 | 41327 | 107841 | 
| 164 | 92 django/core/management/commands/createcachetable.py | 1 | 30| 219 | 41546 | 108703 | 
| 165 | 93 django/core/cache/backends/memcached.py | 1 | 36| 248 | 41794 | 110342 | 
| 166 | 93 django/contrib/auth/views.py | 222 | 243| 172 | 41966 | 110342 | 
| 167 | 93 django/db/backends/mysql/operations.py | 223 | 233| 151 | 42117 | 110342 | 
| 168 | 94 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 42117 | 110420 | 
| 169 | 94 django/contrib/auth/password_validation.py | 54 | 88| 277 | 42394 | 110420 | 
| 170 | 94 django/contrib/postgres/operations.py | 1 | 35| 282 | 42676 | 110420 | 
| 171 | 94 django/contrib/auth/models.py | 200 | 249| 315 | 42991 | 110420 | 
| 172 | 94 django/contrib/postgres/indexes.py | 115 | 137| 194 | 43185 | 110420 | 
| 173 | 94 django/contrib/auth/__init__.py | 87 | 132| 392 | 43577 | 110420 | 
| 174 | 94 django/contrib/gis/db/backends/postgis/operations.py | 334 | 345| 139 | 43716 | 110420 | 
| 175 | 94 django/db/backends/base/features.py | 219 | 290| 556 | 44272 | 110420 | 
| 176 | 94 django/db/backends/sqlite3/base.py | 1 | 74| 510 | 44782 | 110420 | 
| 177 | 94 django/core/cache/backends/db.py | 112 | 197| 794 | 45576 | 110420 | 
| 178 | 94 django/db/backends/utils.py | 92 | 111| 149 | 45725 | 110420 | 
| 179 | 95 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 45725 | 110497 | 
| 180 | 96 django/core/management/commands/sqlmigrate.py | 32 | 66| 328 | 46053 | 111086 | 
| 181 | 96 django/contrib/auth/backends.py | 108 | 126| 146 | 46199 | 111086 | 
| 182 | 97 django/core/management/commands/test.py | 25 | 57| 260 | 46459 | 111495 | 
| 183 | 97 django/contrib/auth/password_validation.py | 135 | 157| 197 | 46656 | 111495 | 
| 184 | 97 django/db/backends/postgresql/operations.py | 135 | 155| 221 | 46877 | 111495 | 
| 185 | 98 django/core/cache/backends/base.py | 50 | 87| 306 | 47183 | 113652 | 
| 186 | 98 django/db/utils.py | 51 | 97| 312 | 47495 | 113652 | 
| 187 | 99 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 48344 | 114501 | 
| 188 | 100 django/db/backends/signals.py | 1 | 4| 0 | 48344 | 114518 | 
| 189 | 100 django/contrib/auth/forms.py | 232 | 250| 179 | 48523 | 114518 | 
| 190 | 100 django/contrib/postgres/indexes.py | 140 | 157| 131 | 48654 | 114518 | 
| 191 | 100 django/core/management/commands/runserver.py | 54 | 64| 120 | 48774 | 114518 | 
| 192 | 100 django/core/management/commands/shell.py | 83 | 103| 162 | 48936 | 114518 | 
| 193 | 101 django/contrib/postgres/search.py | 138 | 145| 121 | 49057 | 116395 | 
| 194 | 102 django/core/management/commands/migrate.py | 21 | 65| 369 | 49426 | 119535 | 
| 195 | 102 django/contrib/auth/hashers.py | 276 | 326| 391 | 49817 | 119535 | 
| 196 | 103 django/core/management/__init__.py | 301 | 382| 743 | 50560 | 122807 | 
| 197 | 104 django/contrib/admin/utils.py | 306 | 363| 468 | 51028 | 126675 | 
| 198 | 104 django/db/backends/dummy/base.py | 50 | 74| 173 | 51201 | 126675 | 
| 199 | 105 django/middleware/security.py | 30 | 47| 164 | 51365 | 127093 | 
| 200 | 105 django/contrib/gis/db/backends/postgis/operations.py | 28 | 46| 219 | 51584 | 127093 | 
| 201 | 105 django/db/backends/utils.py | 66 | 89| 238 | 51822 | 127093 | 
| 202 | 106 django/core/servers/basehttp.py | 197 | 214| 210 | 52032 | 128805 | 
| 203 | 107 django/contrib/gis/__init__.py | 1 | 2| 0 | 52032 | 128819 | 
| 204 | 108 django/contrib/postgres/forms/jsonb.py | 1 | 63| 344 | 52376 | 129163 | 
| 205 | 109 django/db/models/functions/mixins.py | 1 | 20| 161 | 52537 | 129586 | 
| 206 | 110 django/bin/django-admin.py | 1 | 6| 0 | 52537 | 129613 | 
| 207 | 111 django/contrib/gis/geos/base.py | 1 | 7| 0 | 52537 | 129655 | 
| 208 | 112 django/contrib/postgres/fields/mixins.py | 1 | 30| 179 | 52716 | 129835 | 
| 209 | 112 django/db/backends/sqlite3/base.py | 147 | 163| 189 | 52905 | 129835 | 
| 210 | 112 django/contrib/auth/hashers.py | 80 | 102| 167 | 53072 | 129835 | 
| 211 | 112 django/db/backends/oracle/creation.py | 102 | 128| 314 | 53386 | 129835 | 
| 212 | 112 django/core/cache/backends/memcached.py | 65 | 93| 296 | 53682 | 129835 | 
| 213 | 112 django/core/cache/backends/db.py | 228 | 251| 259 | 53941 | 129835 | 
| 214 | 113 django/contrib/gis/db/backends/spatialite/features.py | 1 | 14| 0 | 53941 | 129918 | 
| 215 | 113 django/db/backends/sqlite3/creation.py | 48 | 79| 317 | 54258 | 129918 | 
| 216 | 114 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 54542 | 130622 | 
| 217 | 115 django/contrib/postgres/fields/ranges.py | 1 | 62| 421 | 54963 | 132375 | 
| 218 | 116 django/contrib/auth/urls.py | 1 | 21| 225 | 55188 | 132600 | 


## Patch

```diff
diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py
--- a/django/db/backends/postgresql/client.py
+++ b/django/db/backends/postgresql/client.py
@@ -2,17 +2,9 @@
 import signal
 import subprocess
 
-from django.core.files.temp import NamedTemporaryFile
 from django.db.backends.base.client import BaseDatabaseClient
 
 
-def _escape_pgpass(txt):
-    """
-    Escape a fragment of a PostgreSQL .pgpass file.
-    """
-    return txt.replace('\\', '\\\\').replace(':', '\\:')
-
-
 class DatabaseClient(BaseDatabaseClient):
     executable_name = 'psql'
 
@@ -34,38 +26,17 @@ def runshell_db(cls, conn_params):
             args += ['-p', str(port)]
         args += [dbname]
 
-        temp_pgpass = None
         sigint_handler = signal.getsignal(signal.SIGINT)
+        subprocess_env = os.environ.copy()
+        if passwd:
+            subprocess_env['PGPASSWORD'] = str(passwd)
         try:
-            if passwd:
-                # Create temporary .pgpass file.
-                temp_pgpass = NamedTemporaryFile(mode='w+')
-                try:
-                    print(
-                        _escape_pgpass(host) or '*',
-                        str(port) or '*',
-                        _escape_pgpass(dbname) or '*',
-                        _escape_pgpass(user) or '*',
-                        _escape_pgpass(passwd),
-                        file=temp_pgpass,
-                        sep=':',
-                        flush=True,
-                    )
-                    os.environ['PGPASSFILE'] = temp_pgpass.name
-                except UnicodeEncodeError:
-                    # If the current locale can't encode the data, let the
-                    # user input the password manually.
-                    pass
             # Allow SIGINT to pass to psql to abort queries.
             signal.signal(signal.SIGINT, signal.SIG_IGN)
-            subprocess.check_call(args)
+            subprocess.run(args, check=True, env=subprocess_env)
         finally:
             # Restore the original SIGINT handler.
             signal.signal(signal.SIGINT, sigint_handler)
-            if temp_pgpass:
-                temp_pgpass.close()
-                if 'PGPASSFILE' in os.environ:  # unit tests need cleanup
-                    del os.environ['PGPASSFILE']
 
     def runshell(self):
         DatabaseClient.runshell_db(self.connection.get_connection_params())

```

## Test Patch

```diff
diff --git a/tests/dbshell/test_postgresql.py b/tests/dbshell/test_postgresql.py
--- a/tests/dbshell/test_postgresql.py
+++ b/tests/dbshell/test_postgresql.py
@@ -1,5 +1,6 @@
 import os
 import signal
+import subprocess
 from unittest import mock
 
 from django.db.backends.postgresql.client import DatabaseClient
@@ -11,23 +12,17 @@ class PostgreSqlDbshellCommandTestCase(SimpleTestCase):
     def _run_it(self, dbinfo):
         """
         That function invokes the runshell command, while mocking
-        subprocess.call. It returns a 2-tuple with:
+        subprocess.run(). It returns a 2-tuple with:
         - The command line list
-        - The content of the file pointed by environment PGPASSFILE, or None.
+        - The the value of the PGPASSWORD environment variable, or None.
         """
-        def _mock_subprocess_call(*args):
+        def _mock_subprocess_run(*args, env=os.environ, **kwargs):
             self.subprocess_args = list(*args)
-            if 'PGPASSFILE' in os.environ:
-                with open(os.environ['PGPASSFILE']) as f:
-                    self.pgpass = f.read().strip()  # ignore line endings
-            else:
-                self.pgpass = None
-            return 0
-        self.subprocess_args = None
-        self.pgpass = None
-        with mock.patch('subprocess.call', new=_mock_subprocess_call):
+            self.pgpassword = env.get('PGPASSWORD')
+            return subprocess.CompletedProcess(self.subprocess_args, 0)
+        with mock.patch('subprocess.run', new=_mock_subprocess_run):
             DatabaseClient.runshell_db(dbinfo)
-        return self.subprocess_args, self.pgpass
+        return self.subprocess_args, self.pgpassword
 
     def test_basic(self):
         self.assertEqual(
@@ -39,7 +34,7 @@ def test_basic(self):
                 'port': '444',
             }), (
                 ['psql', '-U', 'someuser', '-h', 'somehost', '-p', '444', 'dbname'],
-                'somehost:444:dbname:someuser:somepassword',
+                'somepassword',
             )
         )
 
@@ -66,28 +61,13 @@ def test_column(self):
                 'port': '444',
             }), (
                 ['psql', '-U', 'some:user', '-h', '::1', '-p', '444', 'dbname'],
-                '\\:\\:1:444:dbname:some\\:user:some\\:password',
-            )
-        )
-
-    def test_escape_characters(self):
-        self.assertEqual(
-            self._run_it({
-                'database': 'dbname',
-                'user': 'some\\user',
-                'password': 'some\\password',
-                'host': 'somehost',
-                'port': '444',
-            }), (
-                ['psql', '-U', 'some\\user', '-h', 'somehost', '-p', '444', 'dbname'],
-                'somehost:444:dbname:some\\\\user:some\\\\password',
+                'some:password',
             )
         )
 
     def test_accent(self):
         username = 'rôle'
         password = 'sésame'
-        pgpass_string = 'somehost:444:dbname:%s:%s' % (username, password)
         self.assertEqual(
             self._run_it({
                 'database': 'dbname',
@@ -97,20 +77,20 @@ def test_accent(self):
                 'port': '444',
             }), (
                 ['psql', '-U', username, '-h', 'somehost', '-p', '444', 'dbname'],
-                pgpass_string,
+                password,
             )
         )
 
     def test_sigint_handler(self):
         """SIGINT is ignored in Python and passed to psql to abort quries."""
-        def _mock_subprocess_call(*args):
+        def _mock_subprocess_run(*args, **kwargs):
             handler = signal.getsignal(signal.SIGINT)
             self.assertEqual(handler, signal.SIG_IGN)
 
         sigint_handler = signal.getsignal(signal.SIGINT)
         # The default handler isn't SIG_IGN.
         self.assertNotEqual(sigint_handler, signal.SIG_IGN)
-        with mock.patch('subprocess.check_call', new=_mock_subprocess_call):
+        with mock.patch('subprocess.run', new=_mock_subprocess_run):
             DatabaseClient.runshell_db({})
         # dbshell restores the original handler.
         self.assertEqual(sigint_handler, signal.getsignal(signal.SIGINT))

```


## Code snippets

### 1 - django/db/backends/postgresql/client.py:

Start line: 1, End line: 72

```python
import os
import signal
import subprocess

from django.core.files.temp import NamedTemporaryFile
from django.db.backends.base.client import BaseDatabaseClient


def _escape_pgpass(txt):
    """
    Escape a fragment of a PostgreSQL .pgpass file.
    """
    return txt.replace('\\', '\\\\').replace(':', '\\:')


class DatabaseClient(BaseDatabaseClient):
    executable_name = 'psql'

    @classmethod
    def runshell_db(cls, conn_params):
        args = [cls.executable_name]

        host = conn_params.get('host', '')
        port = conn_params.get('port', '')
        dbname = conn_params.get('database', '')
        user = conn_params.get('user', '')
        passwd = conn_params.get('password', '')

        if user:
            args += ['-U', user]
        if host:
            args += ['-h', host]
        if port:
            args += ['-p', str(port)]
        args += [dbname]

        temp_pgpass = None
        sigint_handler = signal.getsignal(signal.SIGINT)
        try:
            if passwd:
                # Create temporary .pgpass file.
                temp_pgpass = NamedTemporaryFile(mode='w+')
                try:
                    print(
                        _escape_pgpass(host) or '*',
                        str(port) or '*',
                        _escape_pgpass(dbname) or '*',
                        _escape_pgpass(user) or '*',
                        _escape_pgpass(passwd),
                        file=temp_pgpass,
                        sep=':',
                        flush=True,
                    )
                    os.environ['PGPASSFILE'] = temp_pgpass.name
                except UnicodeEncodeError:
                    # If the current locale can't encode the data, let the
                    # user input the password manually.
                    pass
            # Allow SIGINT to pass to psql to abort queries.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            subprocess.check_call(args)
        finally:
            # Restore the original SIGINT handler.
            signal.signal(signal.SIGINT, sigint_handler)
            if temp_pgpass:
                temp_pgpass.close()
                if 'PGPASSFILE' in os.environ:  # unit tests need cleanup
                    del os.environ['PGPASSFILE']

    def runshell(self):
        DatabaseClient.runshell_db(self.connection.get_connection_params())
```
### 2 - django/db/backends/postgresql/base.py:

Start line: 1, End line: 58

```python
"""
PostgreSQL database backend for Django.

Requires psycopg 2: http://initd.org/projects/psycopg2
"""

import threading
import warnings

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import connections
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.utils import DatabaseError as WrappedDatabaseError
from django.utils.functional import cached_property
from django.utils.safestring import SafeString
from django.utils.version import get_version_tuple

try:
    import psycopg2 as Database
    import psycopg2.extensions
    import psycopg2.extras
except ImportError as e:
    raise ImproperlyConfigured("Error loading psycopg2 module: %s" % e)


def psycopg2_version():
    version = psycopg2.__version__.split(' ', 1)[0]
    return get_version_tuple(version)


PSYCOPG2_VERSION = psycopg2_version()

if PSYCOPG2_VERSION < (2, 5, 4):
    raise ImproperlyConfigured("psycopg2_version 2.5.4 or newer is required; you have %s" % psycopg2.__version__)


# Some of these import psycopg2, so import them after checking if it's installed.
from .client import DatabaseClient                          # NOQA isort:skip
from .creation import DatabaseCreation                      # NOQA isort:skip
from .features import DatabaseFeatures                      # NOQA isort:skip
from .introspection import DatabaseIntrospection            # NOQA isort:skip
from .operations import DatabaseOperations                  # NOQA isort:skip
from .schema import DatabaseSchemaEditor                    # NOQA isort:skip
from .utils import utc_tzinfo_factory                       # NOQA isort:skip

psycopg2.extensions.register_adapter(SafeString, psycopg2.extensions.QuotedString)
psycopg2.extras.register_uuid()

# Register support for inet[] manually so we don't have to handle the Inet()
# object on load all the time.
INETARRAY_OID = 1041
INETARRAY = psycopg2.extensions.new_array_type(
    (INETARRAY_OID,),
    'INETARRAY',
    psycopg2.extensions.UNICODE,
)
psycopg2.extensions.register_type(INETARRAY)
```
### 3 - django/db/backends/mysql/client.py:

Start line: 1, End line: 49

```python
import subprocess

from django.db.backends.base.client import BaseDatabaseClient


class DatabaseClient(BaseDatabaseClient):
    executable_name = 'mysql'

    @classmethod
    def settings_to_cmd_args(cls, settings_dict):
        args = [cls.executable_name]
        db = settings_dict['OPTIONS'].get('db', settings_dict['NAME'])
        user = settings_dict['OPTIONS'].get('user', settings_dict['USER'])
        passwd = settings_dict['OPTIONS'].get('passwd', settings_dict['PASSWORD'])
        host = settings_dict['OPTIONS'].get('host', settings_dict['HOST'])
        port = settings_dict['OPTIONS'].get('port', settings_dict['PORT'])
        server_ca = settings_dict['OPTIONS'].get('ssl', {}).get('ca')
        client_cert = settings_dict['OPTIONS'].get('ssl', {}).get('cert')
        client_key = settings_dict['OPTIONS'].get('ssl', {}).get('key')
        defaults_file = settings_dict['OPTIONS'].get('read_default_file')
        # Seems to be no good way to set sql_mode with CLI.

        if defaults_file:
            args += ["--defaults-file=%s" % defaults_file]
        if user:
            args += ["--user=%s" % user]
        if passwd:
            args += ["--password=%s" % passwd]
        if host:
            if '/' in host:
                args += ["--socket=%s" % host]
            else:
                args += ["--host=%s" % host]
        if port:
            args += ["--port=%s" % port]
        if server_ca:
            args += ["--ssl-ca=%s" % server_ca]
        if client_cert:
            args += ["--ssl-cert=%s" % client_cert]
        if client_key:
            args += ["--ssl-key=%s" % client_key]
        if db:
            args += [db]
        return args

    def runshell(self):
        args = DatabaseClient.settings_to_cmd_args(self.connection.settings_dict)
        subprocess.check_call(args)
```
### 4 - django/db/backends/postgresql/utils.py:

Start line: 1, End line: 8

```python

```
### 5 - django/contrib/auth/management/commands/changepassword.py:

Start line: 34, End line: 76

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        if options['username']:
            username = options['username']
        else:
            username = getpass.getuser()

        try:
            u = UserModel._default_manager.using(options['database']).get(**{
                UserModel.USERNAME_FIELD: username
            })
        except UserModel.DoesNotExist:
            raise CommandError("user '%s' does not exist" % username)

        self.stdout.write("Changing password for user '%s'\n" % u)

        MAX_TRIES = 3
        count = 0
        p1, p2 = 1, 2  # To make them initially mismatch.
        password_validated = False
        while (p1 != p2 or not password_validated) and count < MAX_TRIES:
            p1 = self._get_pass()
            p2 = self._get_pass("Password (again): ")
            if p1 != p2:
                self.stdout.write("Passwords do not match. Please try again.\n")
                count += 1
                # Don't validate passwords that don't match.
                continue
            try:
                validate_password(p2, u)
            except ValidationError as err:
                self.stderr.write('\n'.join(err.messages))
                count += 1
            else:
                password_validated = True

        if count == MAX_TRIES:
            raise CommandError("Aborting password change for user '%s' after %s attempts" % (u, count))

        u.set_password(p1)
        u.save()

        return "Password changed successfully for user '%s'" % u
```
### 6 - django/contrib/auth/management/commands/changepassword.py:

Start line: 1, End line: 32

```python
import getpass

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS

UserModel = get_user_model()


class Command(BaseCommand):
    help = "Change a user's password for django.contrib.auth."
    requires_migrations_checks = True
    requires_system_checks = False

    def _get_pass(self, prompt="Password: "):
        p = getpass.getpass(prompt=prompt)
        if not p:
            raise CommandError("aborted")
        return p

    def add_arguments(self, parser):
        parser.add_argument(
            'username', nargs='?',
            help='Username to change password for; by default, it\'s the current username.',
        )
        parser.add_argument(
            '--database',
            default=DEFAULT_DB_ALIAS,
            help='Specifies the database to use. Default is "default".',
        )
```
### 7 - django/contrib/postgres/__init__.py:

Start line: 1, End line: 2

```python

```
### 8 - django/db/backends/postgresql/base.py:

Start line: 138, End line: 175

```python
class DatabaseWrapper(BaseDatabaseWrapper):
    creation_class = DatabaseCreation
    features_class = DatabaseFeatures
    introspection_class = DatabaseIntrospection
    ops_class = DatabaseOperations
    # PostgreSQL backend-specific attributes.
    _named_cursor_idx = 0

    def get_connection_params(self):
        settings_dict = self.settings_dict
        # None may be used to connect to the default 'postgres' db
        if settings_dict['NAME'] == '':
            raise ImproperlyConfigured(
                "settings.DATABASES is improperly configured. "
                "Please supply the NAME value.")
        if len(settings_dict['NAME'] or '') > self.ops.max_name_length():
            raise ImproperlyConfigured(
                "The database name '%s' (%d characters) is longer than "
                "PostgreSQL's limit of %d characters. Supply a shorter NAME "
                "in settings.DATABASES." % (
                    settings_dict['NAME'],
                    len(settings_dict['NAME']),
                    self.ops.max_name_length(),
                )
            )
        conn_params = {
            'database': settings_dict['NAME'] or 'postgres',
            **settings_dict['OPTIONS'],
        }
        conn_params.pop('isolation_level', None)
        if settings_dict['USER']:
            conn_params['user'] = settings_dict['USER']
        if settings_dict['PASSWORD']:
            conn_params['password'] = settings_dict['PASSWORD']
        if settings_dict['HOST']:
            conn_params['host'] = settings_dict['HOST']
        if settings_dict['PORT']:
            conn_params['port'] = settings_dict['PORT']
        return conn_params
```
### 9 - django/db/backends/postgresql/base.py:

Start line: 261, End line: 288

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    @property
    def _nodb_connection(self):
        nodb_connection = super()._nodb_connection
        try:
            nodb_connection.ensure_connection()
        except (Database.DatabaseError, WrappedDatabaseError):
            warnings.warn(
                "Normally Django will use a connection to the 'postgres' database "
                "to avoid running initialization queries against the production "
                "database when it's not needed (for example, when running tests). "
                "Django was unable to create a connection to the 'postgres' database "
                "and will use the first PostgreSQL database instead.",
                RuntimeWarning
            )
            for connection in connections.all():
                if connection.vendor == 'postgresql' and connection.settings_dict['NAME'] != 'postgres':
                    return self.__class__(
                        {**self.settings_dict, 'NAME': connection.settings_dict['NAME']},
                        alias=self.alias,
                        allow_thread_sharing=False,
                    )
        return nodb_connection

    @cached_property
    def pg_version(self):
        with self.temporary_connection():
            return self.connection.server_version
```
### 10 - django/contrib/postgres/apps.py:

Start line: 40, End line: 67

```python
class PostgresConfig(AppConfig):
    name = 'django.contrib.postgres'
    verbose_name = _('PostgreSQL extensions')

    def ready(self):
        setting_changed.connect(uninstall_if_needed)
        # Connections may already exist before we are called.
        for conn in connections.all():
            if conn.vendor == 'postgresql':
                conn.introspection.data_types_reverse.update({
                    3802: 'django.contrib.postgres.fields.JSONField',
                    3904: 'django.contrib.postgres.fields.IntegerRangeField',
                    3906: 'django.contrib.postgres.fields.DecimalRangeField',
                    3910: 'django.contrib.postgres.fields.DateTimeRangeField',
                    3912: 'django.contrib.postgres.fields.DateRangeField',
                    3926: 'django.contrib.postgres.fields.BigIntegerRangeField',
                })
                if conn.connection is not None:
                    register_type_handlers(conn)
        connection_created.connect(register_type_handlers)
        CharField.register_lookup(Unaccent)
        TextField.register_lookup(Unaccent)
        CharField.register_lookup(SearchLookup)
        TextField.register_lookup(SearchLookup)
        CharField.register_lookup(TrigramSimilar)
        TextField.register_lookup(TrigramSimilar)
        MigrationWriter.register_serializer(RANGE_TYPES, RangeSerializer)
```
