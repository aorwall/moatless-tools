# django__django-11239

| **django/django** | `d87bd29c4f8dfcdf3f4a4eb8340e6770a2416fe3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 269 |
| **Any found context length** | 269 |
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
@@ -17,6 +17,10 @@ def runshell_db(cls, conn_params):
         dbname = conn_params.get('database', '')
         user = conn_params.get('user', '')
         passwd = conn_params.get('password', '')
+        sslmode = conn_params.get('sslmode', '')
+        sslrootcert = conn_params.get('sslrootcert', '')
+        sslcert = conn_params.get('sslcert', '')
+        sslkey = conn_params.get('sslkey', '')
 
         if user:
             args += ['-U', user]
@@ -30,6 +34,14 @@ def runshell_db(cls, conn_params):
         subprocess_env = os.environ.copy()
         if passwd:
             subprocess_env['PGPASSWORD'] = str(passwd)
+        if sslmode:
+            subprocess_env['PGSSLMODE'] = str(sslmode)
+        if sslrootcert:
+            subprocess_env['PGSSLROOTCERT'] = str(sslrootcert)
+        if sslcert:
+            subprocess_env['PGSSLCERT'] = str(sslcert)
+        if sslkey:
+            subprocess_env['PGSSLKEY'] = str(sslkey)
         try:
             # Allow SIGINT to pass to psql to abort queries.
             signal.signal(signal.SIGINT, signal.SIG_IGN)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/postgresql/client.py | 20 | 20 | 1 | 1 | 269
| django/db/backends/postgresql/client.py | 33 | 33 | 1 | 1 | 269


## Problem Statement

```
Add support for postgresql client certificates and key to dbshell.
Description
	
This bug is very similar to the #28322
A common security procedure for DB access is to require mutual TLS for the DB connection.
This involves specifying a server certificate, client certificate, and client key when connecting.
Django already supports this configuration, it looks like this:
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.postgresql',
		'NAME': os.environ.get('POSTGRES_DB_NAME'),
		'USER': os.environ.get('POSTGRES_DB_USER'),
		'HOST': 'postgres',
		'PORT': '5432',
		'SCHEMA': os.environ.get('POSTGRES_DB_SCHEMA'),
		'OPTIONS': {
			 'sslmode': 'verify-ca',
			 'sslrootcert': os.environ.get('POSTGRES_CLI_SSL_CA', 'ca.crt'),
			 'sslcert': os.environ.get('POSTGRES_CLI_SSL_CRT', 'client_cert_chain.crt'),
			 'sslkey': os.environ.get('POSTGRES_CLI_SSL_KEY', 'client_key.key')
		}
	}
}
However the dbshell command does not support the client cert params.
Should be a trivial fix to add in support for the other 'ssl' parameters required here.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/backends/postgresql/client.py** | 1 | 43| 269 | 269 | 269 | 
| 2 | 2 django/db/backends/mysql/client.py | 1 | 49| 422 | 691 | 691 | 
| 3 | 3 django/db/backends/postgresql/base.py | 1 | 58| 469 | 1160 | 3092 | 
| 4 | 4 django/core/management/commands/dbshell.py | 1 | 32| 231 | 1391 | 3323 | 
| 5 | 5 django/db/backends/postgresql/features.py | 1 | 70| 571 | 1962 | 3894 | 
| 6 | 6 django/contrib/postgres/apps.py | 40 | 67| 249 | 2211 | 4460 | 
| 7 | 6 django/db/backends/postgresql/base.py | 61 | 137| 778 | 2989 | 4460 | 
| 8 | 7 django/db/backends/postgresql/utils.py | 1 | 8| 0 | 2989 | 4498 | 
| 9 | 7 django/db/backends/postgresql/base.py | 258 | 284| 217 | 3206 | 4498 | 
| 10 | 8 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 3444 | 5146 | 
| 11 | 8 django/db/backends/postgresql/base.py | 138 | 175| 330 | 3774 | 5146 | 
| 12 | 9 django/db/backends/postgresql/schema.py | 1 | 35| 376 | 4150 | 6524 | 
| 13 | 9 django/db/backends/postgresql/base.py | 197 | 256| 444 | 4594 | 6524 | 
| 14 | 9 django/db/backends/postgresql/creation.py | 36 | 51| 174 | 4768 | 6524 | 
| 15 | 10 django/contrib/postgres/functions.py | 1 | 12| 0 | 4768 | 6577 | 
| 16 | 11 django/contrib/postgres/__init__.py | 1 | 2| 0 | 4768 | 6591 | 
| 17 | 12 django/db/backends/mysql/creation.py | 1 | 29| 219 | 4987 | 7205 | 
| 18 | 13 django/db/backends/postgresql/operations.py | 98 | 133| 267 | 5254 | 9833 | 
| 19 | 14 django/core/management/commands/shell.py | 42 | 81| 401 | 5655 | 10654 | 
| 20 | 15 django/db/utils.py | 1 | 48| 150 | 5805 | 12672 | 
| 21 | 15 django/db/backends/postgresql/operations.py | 201 | 285| 700 | 6505 | 12672 | 
| 22 | 16 django/core/checks/security/base.py | 1 | 86| 752 | 7257 | 14298 | 
| 23 | 17 django/db/backends/base/features.py | 1 | 115| 900 | 8157 | 16751 | 
| 24 | 17 django/core/management/commands/shell.py | 83 | 103| 162 | 8319 | 16751 | 
| 25 | 17 django/db/backends/base/features.py | 219 | 294| 587 | 8906 | 16751 | 
| 26 | 18 django/contrib/gis/db/backends/postgis/base.py | 1 | 27| 197 | 9103 | 16948 | 
| 27 | 19 django/db/backends/base/client.py | 1 | 13| 0 | 9103 | 17053 | 
| 28 | 19 django/db/backends/postgresql/creation.py | 53 | 78| 248 | 9351 | 17053 | 
| 29 | 19 django/db/backends/postgresql/operations.py | 1 | 27| 243 | 9594 | 17053 | 
| 30 | 20 django/db/backends/sqlite3/client.py | 1 | 13| 0 | 9594 | 17111 | 
| 31 | 21 django/db/backends/oracle/features.py | 1 | 61| 494 | 10088 | 17606 | 
| 32 | 22 django/db/backends/mysql/features.py | 1 | 101| 843 | 10931 | 18586 | 
| 33 | 22 django/db/backends/base/features.py | 116 | 217| 854 | 11785 | 18586 | 
| 34 | 23 django/db/backends/sqlite3/base.py | 194 | 236| 793 | 12578 | 24138 | 
| 35 | 23 django/contrib/postgres/apps.py | 1 | 17| 129 | 12707 | 24138 | 
| 36 | 24 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 12707 | 24191 | 
| 37 | 25 django/db/backends/sqlite3/features.py | 1 | 44| 457 | 13164 | 24648 | 
| 38 | 25 django/core/checks/security/base.py | 88 | 190| 747 | 13911 | 24648 | 
| 39 | 26 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 13911 | 24671 | 
| 40 | 27 django/db/backends/sqlite3/schema.py | 1 | 36| 309 | 14220 | 28625 | 
| 41 | 27 django/db/backends/base/features.py | 296 | 309| 124 | 14344 | 28625 | 
| 42 | 28 django/db/backends/mysql/base.py | 1 | 49| 437 | 14781 | 31550 | 
| 43 | 28 django/db/backends/postgresql/base.py | 177 | 195| 186 | 14967 | 31550 | 
| 44 | 29 django/db/backends/oracle/client.py | 1 | 18| 0 | 14967 | 31652 | 
| 45 | 30 django/contrib/postgres/operations.py | 38 | 78| 162 | 15129 | 32097 | 
| 46 | 30 django/core/management/commands/shell.py | 1 | 40| 268 | 15397 | 32097 | 
| 47 | 31 django/core/management/commands/inspectdb.py | 172 | 226| 478 | 15875 | 34660 | 
| 48 | 32 django/db/backends/oracle/base.py | 37 | 59| 235 | 16110 | 39707 | 
| 49 | 33 django/db/backends/base/base.py | 202 | 275| 519 | 16629 | 44492 | 
| 50 | 34 django/contrib/postgres/indexes.py | 1 | 36| 264 | 16893 | 45883 | 
| 51 | 35 django/core/management/commands/sqlmigrate.py | 32 | 67| 347 | 17240 | 46491 | 
| 52 | 36 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 17240 | 46511 | 
| 53 | 36 django/db/backends/postgresql/schema.py | 118 | 150| 417 | 17657 | 46511 | 
| 54 | 37 django/contrib/postgres/forms/__init__.py | 1 | 5| 0 | 17657 | 46553 | 
| 55 | 38 django/contrib/gis/db/backends/postgis/schema.py | 1 | 18| 196 | 17853 | 47187 | 
| 56 | 39 django/db/backends/mysql/validation.py | 1 | 27| 248 | 18101 | 47669 | 
| 57 | 40 django/db/backends/oracle/creation.py | 30 | 100| 722 | 18823 | 51564 | 
| 58 | 40 django/db/backends/sqlite3/base.py | 378 | 412| 289 | 19112 | 51564 | 
| 59 | 40 django/contrib/postgres/operations.py | 1 | 35| 282 | 19394 | 51564 | 
| 60 | 40 django/core/checks/security/base.py | 193 | 211| 127 | 19521 | 51564 | 
| 61 | 40 django/db/utils.py | 181 | 224| 295 | 19816 | 51564 | 
| 62 | 40 django/db/backends/base/base.py | 1 | 22| 128 | 19944 | 51564 | 
| 63 | 40 django/db/backends/sqlite3/base.py | 1 | 75| 513 | 20457 | 51564 | 
| 64 | 41 django/db/__init__.py | 1 | 18| 141 | 20598 | 51957 | 
| 65 | 42 django/db/backends/mysql/schema.py | 1 | 44| 421 | 21019 | 53022 | 
| 66 | 42 django/core/management/commands/inspectdb.py | 1 | 36| 272 | 21291 | 53022 | 
| 67 | 42 django/db/backends/oracle/creation.py | 130 | 165| 399 | 21690 | 53022 | 
| 68 | 43 django/db/models/functions/mixins.py | 1 | 20| 161 | 21851 | 53445 | 
| 69 | 44 django/contrib/postgres/signals.py | 37 | 65| 257 | 22108 | 53946 | 
| 70 | 45 django/db/backends/base/operations.py | 654 | 674| 187 | 22295 | 59339 | 
| 71 | 45 django/contrib/postgres/indexes.py | 39 | 67| 311 | 22606 | 59339 | 
| 72 | 45 django/db/backends/postgresql/operations.py | 157 | 199| 454 | 23060 | 59339 | 
| 73 | 45 django/db/backends/oracle/creation.py | 220 | 251| 390 | 23450 | 59339 | 
| 74 | 46 django/core/management/commands/migrate.py | 21 | 65| 369 | 23819 | 62479 | 
| 75 | 46 django/db/backends/oracle/creation.py | 300 | 315| 193 | 24012 | 62479 | 
| 76 | 47 django/db/backends/base/schema.py | 40 | 115| 790 | 24802 | 73677 | 
| 77 | 48 django/contrib/gis/db/backends/postgis/features.py | 1 | 13| 0 | 24802 | 73767 | 
| 78 | 49 django/contrib/gis/db/backends/spatialite/base.py | 1 | 36| 331 | 25133 | 74386 | 
| 79 | 50 django/db/backends/base/creation.py | 278 | 297| 121 | 25254 | 76711 | 
| 80 | 51 django/db/backends/postgresql/introspection.py | 218 | 234| 179 | 25433 | 79070 | 
| 81 | 52 django/core/management/commands/check.py | 1 | 34| 226 | 25659 | 79505 | 
| 82 | 53 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 25996 | 80044 | 
| 83 | 53 django/db/backends/oracle/creation.py | 187 | 218| 319 | 26315 | 80044 | 
| 84 | 54 django/contrib/postgres/serializers.py | 1 | 11| 0 | 26315 | 80145 | 
| 85 | 55 django/db/backends/dummy/base.py | 1 | 47| 270 | 26585 | 80590 | 
| 86 | 55 django/db/backends/postgresql/introspection.py | 129 | 143| 196 | 26781 | 80590 | 
| 87 | 56 django/contrib/postgres/forms/jsonb.py | 1 | 63| 344 | 27125 | 80934 | 
| 88 | 56 django/db/backends/base/base.py | 550 | 597| 300 | 27425 | 80934 | 
| 89 | 57 django/contrib/postgres/fields/jsonb.py | 86 | 117| 273 | 27698 | 82168 | 
| 90 | 57 django/db/backends/sqlite3/schema.py | 85 | 98| 181 | 27879 | 82168 | 
| 91 | 58 django/core/checks/database.py | 1 | 12| 0 | 27879 | 82221 | 
| 92 | 59 django/db/backends/mysql/operations.py | 1 | 31| 236 | 28115 | 85266 | 
| 93 | 59 django/core/management/commands/check.py | 36 | 66| 214 | 28329 | 85266 | 
| 94 | 59 django/db/__init__.py | 40 | 62| 118 | 28447 | 85266 | 
| 95 | 60 django/db/models/functions/text.py | 24 | 55| 217 | 28664 | 87722 | 
| 96 | 60 django/db/backends/sqlite3/base.py | 291 | 375| 829 | 29493 | 87722 | 
| 97 | 61 django/contrib/gis/utils/srs.py | 1 | 77| 691 | 30184 | 88413 | 
| 98 | 61 django/contrib/postgres/indexes.py | 70 | 87| 133 | 30317 | 88413 | 
| 99 | 61 django/db/backends/sqlite3/base.py | 78 | 147| 710 | 31027 | 88413 | 
| 100 | 62 django/core/management/sql.py | 38 | 53| 116 | 31143 | 88803 | 
| 101 | 62 django/db/backends/sqlite3/base.py | 166 | 192| 270 | 31413 | 88803 | 
| 102 | 63 django/core/checks/security/csrf.py | 1 | 41| 299 | 31712 | 89102 | 
| 103 | 64 django/contrib/postgres/lookups.py | 1 | 77| 469 | 32181 | 89572 | 
| 104 | 65 django/conf/global_settings.py | 499 | 638| 853 | 33034 | 95176 | 
| 105 | 65 django/db/backends/postgresql/schema.py | 59 | 116| 372 | 33406 | 95176 | 
| 106 | 65 django/db/backends/base/base.py | 517 | 548| 227 | 33633 | 95176 | 
| 107 | 65 django/db/backends/postgresql/operations.py | 79 | 96| 202 | 33835 | 95176 | 
| 108 | 65 django/db/backends/postgresql/operations.py | 135 | 155| 221 | 34056 | 95176 | 
| 109 | 65 django/db/backends/sqlite3/base.py | 148 | 164| 189 | 34245 | 95176 | 
| 110 | 65 django/db/backends/oracle/creation.py | 1 | 28| 226 | 34471 | 95176 | 
| 111 | 66 django/contrib/postgres/fields/mixins.py | 1 | 30| 179 | 34650 | 95356 | 
| 112 | 66 django/db/backends/sqlite3/schema.py | 38 | 64| 243 | 34893 | 95356 | 
| 113 | 67 django/contrib/auth/hashers.py | 388 | 404| 127 | 35020 | 100143 | 
| 114 | 67 django/db/backends/oracle/base.py | 143 | 205| 780 | 35800 | 100143 | 
| 115 | 67 django/db/backends/oracle/creation.py | 253 | 281| 277 | 36077 | 100143 | 
| 116 | 68 django/db/backends/signals.py | 1 | 4| 0 | 36077 | 100160 | 
| 117 | 69 django/core/management/commands/createcachetable.py | 32 | 44| 121 | 36198 | 101022 | 
| 118 | 69 django/db/backends/postgresql/operations.py | 39 | 77| 446 | 36644 | 101022 | 
| 119 | 69 django/conf/global_settings.py | 145 | 263| 876 | 37520 | 101022 | 
| 120 | 69 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 206 | 37726 | 101022 | 
| 121 | 70 django/core/cache/backends/db.py | 112 | 197| 794 | 38520 | 103108 | 
| 122 | 71 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 38678 | 104294 | 
| 123 | 72 django/contrib/gis/db/backends/mysql/base.py | 1 | 17| 0 | 38678 | 104403 | 
| 124 | 72 django/db/backends/mysql/operations.py | 178 | 221| 329 | 39007 | 104403 | 
| 125 | 72 django/db/backends/oracle/base.py | 1 | 34| 232 | 39239 | 104403 | 
| 126 | 72 django/contrib/postgres/indexes.py | 140 | 157| 131 | 39370 | 104403 | 
| 127 | 72 django/db/backends/mysql/base.py | 245 | 278| 241 | 39611 | 104403 | 
| 128 | 73 django/contrib/gis/db/backends/spatialite/client.py | 1 | 6| 0 | 39611 | 104432 | 
| 129 | 73 django/core/cache/backends/db.py | 97 | 110| 234 | 39845 | 104432 | 
| 130 | 73 django/db/backends/mysql/base.py | 320 | 345| 173 | 40018 | 104432 | 
| 131 | 73 django/core/management/commands/migrate.py | 67 | 160| 825 | 40843 | 104432 | 
| 132 | 73 django/db/backends/base/base.py | 413 | 490| 525 | 41368 | 104432 | 
| 133 | 74 django/db/models/signals.py | 37 | 54| 231 | 41599 | 104919 | 
| 134 | 74 django/db/backends/oracle/creation.py | 102 | 128| 314 | 41913 | 104919 | 
| 135 | 74 django/contrib/postgres/indexes.py | 160 | 178| 134 | 42047 | 104919 | 
| 136 | 74 django/db/backends/mysql/features.py | 103 | 121| 143 | 42190 | 104919 | 
| 137 | 74 django/db/backends/sqlite3/base.py | 245 | 289| 407 | 42597 | 104919 | 
| 138 | 75 django/contrib/auth/password_validation.py | 160 | 206| 356 | 42953 | 106408 | 
| 139 | 76 django/core/management/commands/testserver.py | 29 | 55| 234 | 43187 | 106842 | 
| 140 | 76 django/db/backends/base/operations.py | 307 | 381| 557 | 43744 | 106842 | 
| 141 | 76 django/db/backends/mysql/creation.py | 57 | 68| 154 | 43898 | 106842 | 
| 142 | 76 django/core/management/commands/inspectdb.py | 38 | 170| 1244 | 45142 | 106842 | 
| 143 | 77 django/db/backends/sqlite3/operations.py | 294 | 336| 429 | 45571 | 109731 | 
| 144 | 77 django/contrib/postgres/fields/jsonb.py | 30 | 83| 361 | 45932 | 109731 | 
| 145 | 77 django/db/backends/oracle/creation.py | 317 | 401| 739 | 46671 | 109731 | 
| 146 | 77 django/db/backends/postgresql/introspection.py | 145 | 217| 756 | 47427 | 109731 | 
| 147 | 78 django/db/models/base.py | 1 | 45| 289 | 47716 | 124621 | 
| 148 | 79 django/contrib/postgres/search.py | 138 | 145| 121 | 47837 | 126498 | 
| 149 | 80 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 48121 | 127202 | 
| 150 | 81 django/db/backends/sqlite3/creation.py | 48 | 79| 317 | 48438 | 128026 | 
| 151 | 81 django/core/management/commands/createcachetable.py | 1 | 30| 219 | 48657 | 128026 | 
| 152 | 81 django/contrib/postgres/search.py | 147 | 179| 287 | 48944 | 128026 | 
| 153 | 81 django/core/management/commands/migrate.py | 161 | 242| 793 | 49737 | 128026 | 
| 154 | 81 django/core/management/sql.py | 21 | 35| 116 | 49853 | 128026 | 
| 155 | 81 django/contrib/gis/db/backends/spatialite/base.py | 38 | 75| 295 | 50148 | 128026 | 
| 156 | 81 django/db/backends/sqlite3/base.py | 237 | 243| 140 | 50288 | 128026 | 
| 157 | 81 django/db/backends/mysql/operations.py | 223 | 233| 151 | 50439 | 128026 | 
| 158 | 82 django/contrib/gis/db/backends/postgis/operations.py | 1 | 25| 217 | 50656 | 131633 | 
| 159 | 82 django/db/utils.py | 131 | 158| 202 | 50858 | 131633 | 
| 160 | 83 django/core/management/commands/sqlflush.py | 1 | 23| 161 | 51019 | 131794 | 
| 161 | 83 django/contrib/auth/hashers.py | 406 | 416| 127 | 51146 | 131794 | 
| 162 | 84 django/core/management/base.py | 379 | 444| 614 | 51760 | 136149 | 
| 163 | 85 django/contrib/postgres/utils.py | 1 | 30| 218 | 51978 | 136367 | 
| 164 | 85 django/contrib/postgres/fields/jsonb.py | 140 | 189| 284 | 52262 | 136367 | 
| 165 | 85 django/db/backends/sqlite3/base.py | 549 | 571| 143 | 52405 | 136367 | 
| 166 | 85 django/core/management/commands/sqlmigrate.py | 1 | 30| 266 | 52671 | 136367 | 
| 167 | 85 django/core/management/commands/testserver.py | 1 | 27| 205 | 52876 | 136367 | 
| 168 | 86 django/db/backends/mysql/compiler.py | 1 | 26| 163 | 53039 | 136531 | 
| 169 | 86 django/contrib/postgres/signals.py | 1 | 34| 243 | 53282 | 136531 | 
| 170 | 86 django/db/utils.py | 100 | 128| 310 | 53592 | 136531 | 
| 171 | 86 django/db/backends/sqlite3/schema.py | 100 | 137| 486 | 54078 | 136531 | 
| 172 | 86 django/contrib/postgres/indexes.py | 115 | 137| 194 | 54272 | 136531 | 
| 173 | 87 django/core/management/commands/flush.py | 27 | 83| 496 | 54768 | 137228 | 
| 174 | 88 django/contrib/postgres/fields/hstore.py | 73 | 113| 261 | 55029 | 137922 | 
| 175 | 88 django/db/backends/sqlite3/operations.py | 1 | 40| 269 | 55298 | 137922 | 
| 176 | 89 django/contrib/gis/management/commands/inspectdb.py | 1 | 17| 146 | 55444 | 138069 | 
| 177 | 89 django/db/backends/base/schema.py | 117 | 143| 267 | 55711 | 138069 | 
| 178 | 90 django/contrib/gis/db/backends/oracle/base.py | 1 | 17| 0 | 55711 | 138172 | 
| 179 | 91 django/db/backends/oracle/schema.py | 1 | 39| 400 | 56111 | 139917 | 
| 180 | 92 django/db/backends/oracle/operations.py | 371 | 417| 439 | 56550 | 145401 | 
| 181 | 92 django/db/backends/oracle/base.py | 207 | 230| 205 | 56755 | 145401 | 
| 182 | 93 django/contrib/postgres/fields/ranges.py | 167 | 180| 158 | 56913 | 147154 | 
| 183 | 93 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 57205 | 147154 | 
| 184 | 93 django/contrib/auth/hashers.py | 418 | 447| 290 | 57495 | 147154 | 
| 185 | 93 django/db/backends/sqlite3/creation.py | 1 | 46| 356 | 57851 | 147154 | 
| 186 | 93 django/contrib/postgres/apps.py | 20 | 37| 188 | 58039 | 147154 | 
| 187 | 93 django/db/backends/base/creation.py | 89 | 124| 307 | 58346 | 147154 | 
| 188 | 94 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 58346 | 147222 | 
| 189 | 94 django/db/backends/mysql/base.py | 172 | 224| 427 | 58773 | 147222 | 
| 190 | 95 django/core/management/commands/sqlsequencereset.py | 1 | 24| 172 | 58945 | 147394 | 
| 191 | 95 django/db/backends/base/creation.py | 1 | 87| 629 | 59574 | 147394 | 
| 192 | 96 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 59821 | 147641 | 
| 193 | 96 django/db/backends/oracle/operations.py | 531 | 545| 228 | 60049 | 147641 | 
| 194 | 97 django/db/backends/dummy/features.py | 1 | 7| 0 | 60049 | 147673 | 
| 195 | 97 django/db/backends/base/operations.py | 1 | 102| 797 | 60846 | 147673 | 
| 196 | 98 django/contrib/gis/db/backends/postgis/adapter.py | 1 | 51| 362 | 61208 | 148162 | 
| 197 | 98 django/db/backends/sqlite3/operations.py | 66 | 116| 523 | 61731 | 148162 | 
| 198 | 98 django/contrib/postgres/search.py | 122 | 136| 142 | 61873 | 148162 | 
| 199 | 98 django/db/backends/sqlite3/operations.py | 163 | 188| 190 | 62063 | 148162 | 
| 200 | 98 django/db/backends/oracle/operations.py | 248 | 262| 157 | 62220 | 148162 | 
| 201 | 98 django/db/backends/oracle/operations.py | 186 | 234| 370 | 62590 | 148162 | 
| 202 | 98 django/db/backends/base/schema.py | 753 | 793| 511 | 63101 | 148162 | 
| 203 | 98 django/contrib/gis/db/backends/postgis/operations.py | 98 | 156| 703 | 63804 | 148162 | 
| 204 | 98 django/db/models/base.py | 1230 | 1259| 242 | 64046 | 148162 | 
| 205 | 99 django/core/cache/backends/base.py | 239 | 256| 165 | 64211 | 150319 | 
| 206 | 100 django/db/migrations/operations/special.py | 116 | 130| 139 | 64350 | 151877 | 
| 207 | 101 django/contrib/auth/management/commands/createsuperuser.py | 63 | 169| 962 | 65312 | 153592 | 
| 208 | 101 django/db/backends/oracle/creation.py | 283 | 298| 183 | 65495 | 153592 | 
| 209 | 101 django/db/backends/base/schema.py | 610 | 682| 792 | 66287 | 153592 | 
| 210 | 102 django/contrib/auth/admin.py | 1 | 22| 188 | 66475 | 155318 | 
| 211 | 103 django/db/backends/utils.py | 216 | 253| 279 | 66754 | 157211 | 
| 212 | 103 django/db/utils.py | 51 | 97| 312 | 67066 | 157211 | 
| 213 | 104 django/core/management/commands/loaddata.py | 32 | 61| 261 | 67327 | 160087 | 
| 214 | 104 django/db/backends/oracle/operations.py | 332 | 369| 328 | 67655 | 160087 | 
| 215 | 104 django/db/backends/dummy/base.py | 50 | 74| 173 | 67828 | 160087 | 
| 216 | 105 django/contrib/postgres/fields/citext.py | 1 | 25| 113 | 67941 | 160201 | 
| 217 | 106 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 68075 | 160892 | 
| 218 | 106 django/db/backends/mysql/base.py | 96 | 170| 775 | 68850 | 160892 | 


## Patch

```diff
diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py
--- a/django/db/backends/postgresql/client.py
+++ b/django/db/backends/postgresql/client.py
@@ -17,6 +17,10 @@ def runshell_db(cls, conn_params):
         dbname = conn_params.get('database', '')
         user = conn_params.get('user', '')
         passwd = conn_params.get('password', '')
+        sslmode = conn_params.get('sslmode', '')
+        sslrootcert = conn_params.get('sslrootcert', '')
+        sslcert = conn_params.get('sslcert', '')
+        sslkey = conn_params.get('sslkey', '')
 
         if user:
             args += ['-U', user]
@@ -30,6 +34,14 @@ def runshell_db(cls, conn_params):
         subprocess_env = os.environ.copy()
         if passwd:
             subprocess_env['PGPASSWORD'] = str(passwd)
+        if sslmode:
+            subprocess_env['PGSSLMODE'] = str(sslmode)
+        if sslrootcert:
+            subprocess_env['PGSSLROOTCERT'] = str(sslrootcert)
+        if sslcert:
+            subprocess_env['PGSSLCERT'] = str(sslcert)
+        if sslkey:
+            subprocess_env['PGSSLKEY'] = str(sslkey)
         try:
             # Allow SIGINT to pass to psql to abort queries.
             signal.signal(signal.SIGINT, signal.SIG_IGN)

```

## Test Patch

```diff
diff --git a/tests/dbshell/test_postgresql.py b/tests/dbshell/test_postgresql.py
--- a/tests/dbshell/test_postgresql.py
+++ b/tests/dbshell/test_postgresql.py
@@ -14,15 +14,16 @@ def _run_it(self, dbinfo):
         That function invokes the runshell command, while mocking
         subprocess.run(). It returns a 2-tuple with:
         - The command line list
-        - The the value of the PGPASSWORD environment variable, or None.
+        - The dictionary of PG* environment variables, or {}.
         """
         def _mock_subprocess_run(*args, env=os.environ, **kwargs):
             self.subprocess_args = list(*args)
-            self.pgpassword = env.get('PGPASSWORD')
+            # PostgreSQL environment variables.
+            self.pg_env = {key: env[key] for key in env if key.startswith('PG')}
             return subprocess.CompletedProcess(self.subprocess_args, 0)
         with mock.patch('subprocess.run', new=_mock_subprocess_run):
             DatabaseClient.runshell_db(dbinfo)
-        return self.subprocess_args, self.pgpassword
+        return self.subprocess_args, self.pg_env
 
     def test_basic(self):
         self.assertEqual(
@@ -34,7 +35,7 @@ def test_basic(self):
                 'port': '444',
             }), (
                 ['psql', '-U', 'someuser', '-h', 'somehost', '-p', '444', 'dbname'],
-                'somepassword',
+                {'PGPASSWORD': 'somepassword'},
             )
         )
 
@@ -47,7 +48,29 @@ def test_nopass(self):
                 'port': '444',
             }), (
                 ['psql', '-U', 'someuser', '-h', 'somehost', '-p', '444', 'dbname'],
-                None,
+                {},
+            )
+        )
+
+    def test_ssl_certificate(self):
+        self.assertEqual(
+            self._run_it({
+                'database': 'dbname',
+                'user': 'someuser',
+                'host': 'somehost',
+                'port': '444',
+                'sslmode': 'verify-ca',
+                'sslrootcert': 'root.crt',
+                'sslcert': 'client.crt',
+                'sslkey': 'client.key',
+            }), (
+                ['psql', '-U', 'someuser', '-h', 'somehost', '-p', '444', 'dbname'],
+                {
+                    'PGSSLCERT': 'client.crt',
+                    'PGSSLKEY': 'client.key',
+                    'PGSSLMODE': 'verify-ca',
+                    'PGSSLROOTCERT': 'root.crt',
+                },
             )
         )
 
@@ -61,7 +84,7 @@ def test_column(self):
                 'port': '444',
             }), (
                 ['psql', '-U', 'some:user', '-h', '::1', '-p', '444', 'dbname'],
-                'some:password',
+                {'PGPASSWORD': 'some:password'},
             )
         )
 
@@ -77,7 +100,7 @@ def test_accent(self):
                 'port': '444',
             }), (
                 ['psql', '-U', username, '-h', 'somehost', '-p', '444', 'dbname'],
-                password,
+                {'PGPASSWORD': password},
             )
         )
 

```


## Code snippets

### 1 - django/db/backends/postgresql/client.py:

Start line: 1, End line: 43

```python
import os
import signal
import subprocess

from django.db.backends.base.client import BaseDatabaseClient


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

        sigint_handler = signal.getsignal(signal.SIGINT)
        subprocess_env = os.environ.copy()
        if passwd:
            subprocess_env['PGPASSWORD'] = str(passwd)
        try:
            # Allow SIGINT to pass to psql to abort queries.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            subprocess.run(args, check=True, env=subprocess_env)
        finally:
            # Restore the original SIGINT handler.
            signal.signal(signal.SIGINT, sigint_handler)

    def runshell(self):
        DatabaseClient.runshell_db(self.connection.get_connection_params())
```
### 2 - django/db/backends/mysql/client.py:

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
### 3 - django/db/backends/postgresql/base.py:

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
### 4 - django/core/management/commands/dbshell.py:

Start line: 1, End line: 32

```python
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections


class Command(BaseCommand):
    help = (
        "Runs the command-line client for specified database, or the "
        "default database if none is provided."
    )

    requires_system_checks = False

    def add_arguments(self, parser):
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database onto which to open a shell. Defaults to the "default" database.',
        )

    def handle(self, **options):
        connection = connections[options['database']]
        try:
            connection.client.runshell()
        except OSError:
            # Note that we're assuming OSError means that the client program
            # isn't installed. There's a possibility OSError would be raised
            # for some other reason, in which case this error message would be
            # inaccurate. Still, this message catches the common case.
            raise CommandError(
                'You appear not to have the %r program installed or on your path.' %
                connection.client.executable_name
            )
```
### 5 - django/db/backends/postgresql/features.py:

Start line: 1, End line: 70

```python
import operator

from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.utils import InterfaceError
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    allows_group_by_selected_pks = True
    can_return_columns_from_insert = True
    can_return_rows_from_bulk_insert = True
    has_real_datatype = True
    has_native_uuid_field = True
    has_native_duration_field = True
    can_defer_constraint_checks = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_of = True
    has_select_for_update_skip_locked = True
    can_release_savepoints = True
    supports_tablespaces = True
    supports_transactions = True
    can_introspect_autofield = True
    can_introspect_ip_address_field = True
    can_introspect_materialized_views = True
    can_introspect_small_integer_field = True
    can_distinct_on_fields = True
    can_rollback_ddl = True
    supports_combined_alters = True
    nulls_order_largest = True
    closed_cursor_error_class = InterfaceError
    has_case_insensitive_like = False
    greatest_least_ignores_nulls = True
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_slicing_ordering_in_compound = True
    create_test_procedure_without_params_sql = """
        CREATE FUNCTION test_procedure () RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := 1;
        END;
    $$ LANGUAGE plpgsql;"""
    create_test_procedure_with_int_param_sql = """
        CREATE FUNCTION test_procedure (P_I INTEGER) RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := P_I;
        END;
    $$ LANGUAGE plpgsql;"""
    requires_casted_case_in_updates = True
    supports_over_clause = True
    supports_aggregate_filter_clause = True
    supported_explain_formats = {'JSON', 'TEXT', 'XML', 'YAML'}
    validates_explain_options = False  # A query will error on invalid options.

    @cached_property
    def is_postgresql_9_6(self):
        return self.connection.pg_version >= 90600

    @cached_property
    def is_postgresql_10(self):
        return self.connection.pg_version >= 100000

    has_brin_autosummarize = property(operator.attrgetter('is_postgresql_10'))
    has_phraseto_tsquery = property(operator.attrgetter('is_postgresql_9_6'))
    supports_table_partitions = property(operator.attrgetter('is_postgresql_10'))
```
### 6 - django/contrib/postgres/apps.py:

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
### 7 - django/db/backends/postgresql/base.py:

Start line: 61, End line: 137

```python
class DatabaseWrapper(BaseDatabaseWrapper):
    vendor = 'postgresql'
    display_name = 'PostgreSQL'
    # This dictionary maps Field objects to their associated PostgreSQL column
    # types, as strings. Column-type strings can contain format strings; they'll
    # be interpolated against the values of Field.__dict__ before being output.
    # If a column type is set to None, it won't be included in the output.
    data_types = {
        'AutoField': 'serial',
        'BigAutoField': 'bigserial',
        'BinaryField': 'bytea',
        'BooleanField': 'boolean',
        'CharField': 'varchar(%(max_length)s)',
        'DateField': 'date',
        'DateTimeField': 'timestamp with time zone',
        'DecimalField': 'numeric(%(max_digits)s, %(decimal_places)s)',
        'DurationField': 'interval',
        'FileField': 'varchar(%(max_length)s)',
        'FilePathField': 'varchar(%(max_length)s)',
        'FloatField': 'double precision',
        'IntegerField': 'integer',
        'BigIntegerField': 'bigint',
        'IPAddressField': 'inet',
        'GenericIPAddressField': 'inet',
        'NullBooleanField': 'boolean',
        'OneToOneField': 'integer',
        'PositiveIntegerField': 'integer',
        'PositiveSmallIntegerField': 'smallint',
        'SlugField': 'varchar(%(max_length)s)',
        'SmallIntegerField': 'smallint',
        'TextField': 'text',
        'TimeField': 'time',
        'UUIDField': 'uuid',
    }
    data_type_check_constraints = {
        'PositiveIntegerField': '"%(column)s" >= 0',
        'PositiveSmallIntegerField': '"%(column)s" >= 0',
    }
    operators = {
        'exact': '= %s',
        'iexact': '= UPPER(%s)',
        'contains': 'LIKE %s',
        'icontains': 'LIKE UPPER(%s)',
        'regex': '~ %s',
        'iregex': '~* %s',
        'gt': '> %s',
        'gte': '>= %s',
        'lt': '< %s',
        'lte': '<= %s',
        'startswith': 'LIKE %s',
        'endswith': 'LIKE %s',
        'istartswith': 'LIKE UPPER(%s)',
        'iendswith': 'LIKE UPPER(%s)',
    }

    # The patterns below are used to generate SQL pattern lookup clauses when
    # the right-hand side of the lookup isn't a raw string (it might be an expression
    # or the result of a bilateral transformation).
    # In those cases, special characters for LIKE operators (e.g. \, *, _) should be
    # escaped on database side.
    #
    # Note: we use str.format() here for readability as '%' is used as a wildcard for
    # the LIKE operator.
    pattern_esc = r"REPLACE(REPLACE(REPLACE({}, E'\\', E'\\\\'), E'%%', E'\\%%'), E'_', E'\\_')"
    pattern_ops = {
        'contains': "LIKE '%%' || {} || '%%'",
        'icontains': "LIKE '%%' || UPPER({}) || '%%'",
        'startswith': "LIKE {} || '%%'",
        'istartswith': "LIKE UPPER({}) || '%%'",
        'endswith': "LIKE '%%' || {}",
        'iendswith': "LIKE '%%' || UPPER({})",
    }

    Database = Database
    SchemaEditorClass = DatabaseSchemaEditor
    # Classes instantiated in __init__().
    client_class = DatabaseClient
```
### 8 - django/db/backends/postgresql/utils.py:

Start line: 1, End line: 8

```python

```
### 9 - django/db/backends/postgresql/base.py:

Start line: 258, End line: 284

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
                    )
        return nodb_connection

    @cached_property
    def pg_version(self):
        with self.temporary_connection():
            return self.connection.server_version
```
### 10 - django/db/backends/postgresql/creation.py:

Start line: 1, End line: 34

```python
import sys

from psycopg2 import errorcodes

from django.db.backends.base.creation import BaseDatabaseCreation
from django.db.backends.utils import strip_quotes


class DatabaseCreation(BaseDatabaseCreation):

    def _quote_name(self, name):
        return self.connection.ops.quote_name(name)

    def _get_database_create_suffix(self, encoding=None, template=None):
        suffix = ""
        if encoding:
            suffix += " ENCODING '{}'".format(encoding)
        if template:
            suffix += " TEMPLATE {}".format(self._quote_name(template))
        return suffix and "WITH" + suffix

    def sql_table_creation_suffix(self):
        test_settings = self.connection.settings_dict['TEST']
        assert test_settings['COLLATION'] is None, (
            "PostgreSQL does not support collation setting at database creation time."
        )
        return self._get_database_create_suffix(
            encoding=test_settings['CHARSET'],
            template=test_settings.get('TEMPLATE'),
        )

    def _database_exists(self, cursor, database_name):
        cursor.execute('SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s', [strip_quotes(database_name)])
        return cursor.fetchone() is not None
```
