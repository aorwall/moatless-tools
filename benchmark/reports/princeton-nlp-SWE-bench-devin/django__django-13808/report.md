# django__django-13808

| **django/django** | `f054468cac325e8d8fa4d5934b939b93242a3730` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 989 |
| **Any found context length** | 989 |
| **Avg pos** | 8.0 |
| **Min pos** | 3 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/postgresql/base.py b/django/db/backends/postgresql/base.py
--- a/django/db/backends/postgresql/base.py
+++ b/django/db/backends/postgresql/base.py
@@ -152,10 +152,14 @@ class DatabaseWrapper(BaseDatabaseWrapper):
     def get_connection_params(self):
         settings_dict = self.settings_dict
         # None may be used to connect to the default 'postgres' db
-        if settings_dict['NAME'] == '':
+        if (
+            settings_dict['NAME'] == '' and
+            not settings_dict.get('OPTIONS', {}).get('service')
+        ):
             raise ImproperlyConfigured(
                 "settings.DATABASES is improperly configured. "
-                "Please supply the NAME value.")
+                "Please supply the NAME or OPTIONS['service'] value."
+            )
         if len(settings_dict['NAME'] or '') > self.ops.max_name_length():
             raise ImproperlyConfigured(
                 "The database name '%s' (%d characters) is longer than "
@@ -166,10 +170,19 @@ def get_connection_params(self):
                     self.ops.max_name_length(),
                 )
             )
-        conn_params = {
-            'database': settings_dict['NAME'] or 'postgres',
-            **settings_dict['OPTIONS'],
-        }
+        conn_params = {}
+        if settings_dict['NAME']:
+            conn_params = {
+                'database': settings_dict['NAME'],
+                **settings_dict['OPTIONS'],
+            }
+        elif settings_dict['NAME'] is None:
+            # Connect to the default 'postgres' db.
+            settings_dict.get('OPTIONS', {}).pop('service', None)
+            conn_params = {'database': 'postgres', **settings_dict['OPTIONS']}
+        else:
+            conn_params = {**settings_dict['OPTIONS']}
+
         conn_params.pop('isolation_level', None)
         if settings_dict['USER']:
             conn_params['user'] = settings_dict['USER']
diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py
--- a/django/db/backends/postgresql/client.py
+++ b/django/db/backends/postgresql/client.py
@@ -16,6 +16,7 @@ def settings_to_cmd_args_env(cls, settings_dict, parameters):
         dbname = settings_dict.get('NAME') or 'postgres'
         user = settings_dict.get('USER')
         passwd = settings_dict.get('PASSWORD')
+        service = options.get('service')
         sslmode = options.get('sslmode')
         sslrootcert = options.get('sslrootcert')
         sslcert = options.get('sslcert')
@@ -33,6 +34,8 @@ def settings_to_cmd_args_env(cls, settings_dict, parameters):
         env = {}
         if passwd:
             env['PGPASSWORD'] = str(passwd)
+        if service:
+            env['PGSERVICE'] = str(service)
         if sslmode:
             env['PGSSLMODE'] = str(sslmode)
         if sslrootcert:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/postgresql/base.py | 155 | 158 | 3 | 1 | 989
| django/db/backends/postgresql/base.py | 169 | 172 | 3 | 1 | 989
| django/db/backends/postgresql/client.py | 19 | 19 | 5 | 4 | 2143
| django/db/backends/postgresql/client.py | 36 | 36 | 5 | 4 | 2143


## Problem Statement

```
Allow postgresql database connections to use postgres services
Description
	 
		(last modified by levihb)
	 
Postgres offers a way to make database connections through the use of services, which are basically equivalent to MySQL's options files.
Server, database, username, etc information is stored by default in ~/.pg_service.conf and takes a very similar format to MySQL cnf files:
[my_alias]
host=10.0.19.10
user=postgres
dbname=postgres
port=5432
And password can be stored in ~/.pgpass under a different format.
I think being able to just add them to the DATABASES config would be useful, similar to how you can add MySQL cnf files. psycopg2 supports it just fine through the service argument/string connect(service='my_alias') connect('service=my_alias').
At the moment it can be added like this:
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.postgresql',
		'NAME': 'postgres',
		'OPTIONS': {'service': 'my_alias'}
	}
}
Which works, however it involves repeating the database name. I don't think the database name should be repeated twice because it couples the config and the service file together, and makes it harder to just move it between different environments. I think ideally you would just specify the service, either like this:
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.postgresql',
		'OPTIONS': {'service': 'my_alias'}
	}
}
Or maybe a better way would be?:
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.postgresql',
		'SERVICE': 'my_alias
	}
}
It seems like something that would be super easy to add. I don't mind creating a pull request for it, but would like to know why it hasn't been added, and how it would be recommended to add it.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/backends/postgresql/base.py** | 1 | 62| 456 | 456 | 2840 | 
| 2 | 2 django/contrib/postgres/apps.py | 43 | 70| 250 | 706 | 3436 | 
| **-> 3 <-** | **2 django/db/backends/postgresql/base.py** | 152 | 182| 283 | 989 | 3436 | 
| 4 | 3 django/db/backends/postgresql/features.py | 1 | 97| 767 | 1756 | 4203 | 
| **-> 5 <-** | **4 django/db/backends/postgresql/client.py** | 1 | 55| 387 | 2143 | 4590 | 
| 6 | 5 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 2381 | 5236 | 
| 7 | **5 django/db/backends/postgresql/base.py** | 297 | 340| 343 | 2724 | 5236 | 
| 8 | **5 django/db/backends/postgresql/base.py** | 184 | 206| 253 | 2977 | 5236 | 
| 9 | 6 django/contrib/gis/db/backends/postgis/base.py | 1 | 27| 197 | 3174 | 5433 | 
| 10 | 7 django/db/backends/postgresql/operations.py | 189 | 276| 696 | 3870 | 7994 | 
| 11 | 7 django/db/backends/postgresql/operations.py | 108 | 136| 235 | 4105 | 7994 | 
| 12 | **7 django/db/backends/postgresql/base.py** | 132 | 150| 177 | 4282 | 7994 | 
| 13 | 7 django/db/backends/postgresql/operations.py | 1 | 27| 245 | 4527 | 7994 | 
| 14 | 7 django/db/backends/postgresql/creation.py | 53 | 78| 247 | 4774 | 7994 | 
| 15 | 8 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 5400 | 10162 | 
| 16 | 8 django/db/backends/postgresql/creation.py | 36 | 51| 173 | 5573 | 10162 | 
| 17 | 8 django/contrib/postgres/apps.py | 1 | 20| 158 | 5731 | 10162 | 
| 18 | 9 django/db/utils.py | 134 | 157| 227 | 5958 | 12169 | 
| 19 | 9 django/db/backends/postgresql/schema.py | 212 | 225| 182 | 6140 | 12169 | 
| 20 | 10 django/contrib/postgres/indexes.py | 1 | 36| 276 | 6416 | 13996 | 
| 21 | **10 django/db/backends/postgresql/base.py** | 208 | 240| 260 | 6676 | 13996 | 
| 22 | 11 django/db/backends/base/features.py | 113 | 216| 833 | 7509 | 16956 | 
| 23 | **11 django/db/backends/postgresql/base.py** | 65 | 131| 689 | 8198 | 16956 | 
| 24 | 11 django/db/backends/postgresql/operations.py | 89 | 106| 202 | 8400 | 16956 | 
| 25 | 12 django/contrib/postgres/functions.py | 1 | 12| 0 | 8400 | 17009 | 
| 26 | 12 django/db/backends/base/features.py | 217 | 319| 879 | 9279 | 17009 | 
| 27 | 13 django/contrib/postgres/operations.py | 36 | 60| 197 | 9476 | 18868 | 
| 28 | 13 django/contrib/postgres/operations.py | 191 | 212| 207 | 9683 | 18868 | 
| 29 | 14 django/db/models/lookups.py | 272 | 299| 236 | 9919 | 23823 | 
| 30 | 15 django/db/backends/oracle/creation.py | 1 | 28| 225 | 10144 | 27716 | 
| 31 | 15 django/db/backends/base/features.py | 1 | 112| 895 | 11039 | 27716 | 
| 32 | 15 django/contrib/postgres/operations.py | 121 | 142| 223 | 11262 | 27716 | 
| 33 | 16 django/contrib/postgres/constraints.py | 108 | 127| 155 | 11417 | 29141 | 
| 34 | 17 django/contrib/postgres/signals.py | 37 | 65| 257 | 11674 | 29642 | 
| 35 | 17 django/db/backends/postgresql/schema.py | 227 | 239| 152 | 11826 | 29642 | 
| 36 | 18 django/db/backends/base/operations.py | 674 | 694| 187 | 12013 | 35238 | 
| 37 | 18 django/contrib/postgres/operations.py | 63 | 118| 250 | 12263 | 35238 | 
| 38 | 19 django/db/backends/mysql/features.py | 56 | 104| 453 | 12716 | 37210 | 
| 39 | 20 django/contrib/gis/db/backends/postgis/operations.py | 98 | 157| 720 | 13436 | 40813 | 
| 40 | 21 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 13436 | 40866 | 
| 41 | 21 django/db/backends/postgresql/operations.py | 138 | 158| 221 | 13657 | 40866 | 
| 42 | 21 django/contrib/postgres/operations.py | 1 | 34| 258 | 13915 | 40866 | 
| 43 | 21 django/db/backends/postgresql/operations.py | 160 | 187| 311 | 14226 | 40866 | 
| 44 | 21 django/db/backends/mysql/features.py | 193 | 243| 392 | 14618 | 40866 | 
| 45 | 22 django/contrib/postgres/lookups.py | 1 | 61| 337 | 14955 | 41203 | 
| 46 | 22 django/db/backends/mysql/features.py | 1 | 54| 406 | 15361 | 41203 | 
| 47 | 22 django/db/backends/mysql/features.py | 106 | 191| 741 | 16102 | 41203 | 
| 48 | 22 django/contrib/postgres/indexes.py | 155 | 181| 245 | 16347 | 41203 | 
| 49 | 22 django/db/backends/oracle/creation.py | 30 | 100| 722 | 17069 | 41203 | 
| 50 | 23 django/db/backends/postgresql/introspection.py | 214 | 235| 208 | 17277 | 43522 | 
| 51 | 24 django/contrib/gis/db/backends/postgis/schema.py | 1 | 19| 206 | 17483 | 44195 | 
| 52 | **24 django/db/backends/postgresql/base.py** | 274 | 295| 156 | 17639 | 44195 | 
| 53 | 25 django/contrib/gis/utils/srs.py | 1 | 77| 691 | 18330 | 44886 | 
| 54 | 26 django/db/backends/mysql/operations.py | 338 | 353| 217 | 18547 | 48585 | 
| 55 | 26 django/db/backends/base/features.py | 320 | 343| 209 | 18756 | 48585 | 
| 56 | 27 django/core/management/commands/dbshell.py | 1 | 21| 139 | 18895 | 48894 | 
| 57 | 28 django/db/backends/sqlite3/base.py | 206 | 250| 771 | 19666 | 54933 | 
| 58 | 29 django/db/backends/oracle/base.py | 155 | 217| 780 | 20446 | 60001 | 
| 59 | 30 django/contrib/postgres/search.py | 265 | 303| 248 | 20694 | 62223 | 
| 60 | 31 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 20694 | 62243 | 
| 61 | 31 django/contrib/postgres/operations.py | 215 | 235| 163 | 20857 | 62243 | 
| 62 | 31 django/db/backends/oracle/creation.py | 317 | 401| 738 | 21595 | 62243 | 
| 63 | 32 django/contrib/gis/db/backends/postgis/adapter.py | 1 | 55| 379 | 21974 | 62749 | 
| 64 | 32 django/db/backends/mysql/operations.py | 366 | 378| 132 | 22106 | 62749 | 
| 65 | 32 django/contrib/postgres/signals.py | 1 | 34| 243 | 22349 | 62749 | 
| 66 | 32 django/db/backends/postgresql/introspection.py | 47 | 58| 197 | 22546 | 62749 | 
| 67 | 33 django/db/backends/mysql/client.py | 1 | 58| 546 | 23092 | 63296 | 
| 68 | 34 django/contrib/postgres/forms/__init__.py | 1 | 4| 0 | 23092 | 63327 | 
| 69 | 35 django/contrib/postgres/serializers.py | 1 | 11| 0 | 23092 | 63428 | 
| 70 | **35 django/db/backends/postgresql/base.py** | 242 | 272| 267 | 23359 | 63428 | 
| 71 | 35 django/contrib/postgres/search.py | 160 | 195| 313 | 23672 | 63428 | 
| 72 | 36 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 23672 | 63451 | 
| 73 | 37 django/db/backends/sqlite3/features.py | 1 | 114| 1027 | 24699 | 64478 | 
| 74 | 37 django/db/backends/oracle/base.py | 219 | 272| 527 | 25226 | 64478 | 
| 75 | 38 django/db/backends/oracle/operations.py | 207 | 255| 411 | 25637 | 70444 | 
| 76 | 39 django/contrib/gis/db/backends/mysql/operations.py | 14 | 55| 353 | 25990 | 71332 | 
| 77 | 39 django/db/backends/postgresql/operations.py | 41 | 87| 528 | 26518 | 71332 | 
| 78 | 40 django/utils/connection.py | 34 | 77| 271 | 26789 | 71796 | 
| 79 | 40 django/db/backends/oracle/operations.py | 408 | 459| 516 | 27305 | 71796 | 
| 80 | 41 django/db/backends/base/creation.py | 1 | 100| 755 | 28060 | 74584 | 
| 81 | 41 django/db/backends/mysql/operations.py | 1 | 35| 282 | 28342 | 74584 | 
| 82 | 42 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 124 | 28466 | 75022 | 
| 83 | 42 django/db/backends/oracle/creation.py | 300 | 315| 193 | 28659 | 75022 | 
| 84 | 42 django/db/backends/base/creation.py | 241 | 257| 173 | 28832 | 75022 | 
| 85 | 42 django/db/utils.py | 159 | 178| 189 | 29021 | 75022 | 
| 86 | 42 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 29372 | 75022 | 
| 87 | 42 django/contrib/postgres/apps.py | 23 | 40| 188 | 29560 | 75022 | 
| 88 | 42 django/db/utils.py | 1 | 49| 177 | 29737 | 75022 | 
| 89 | 43 django/db/backends/base/base.py | 1 | 23| 138 | 29875 | 79925 | 
| 90 | 44 django/db/backends/mysql/creation.py | 1 | 30| 221 | 30096 | 80564 | 
| 91 | 45 django/db/backends/oracle/features.py | 1 | 124| 1048 | 31144 | 81612 | 
| 92 | 45 django/contrib/postgres/indexes.py | 204 | 229| 179 | 31323 | 81612 | 
| 93 | 45 django/contrib/postgres/operations.py | 169 | 189| 147 | 31470 | 81612 | 
| 94 | 46 django/db/models/fields/json.py | 152 | 164| 125 | 31595 | 85810 | 
| 95 | 46 django/contrib/postgres/search.py | 1 | 24| 205 | 31800 | 85810 | 
| 96 | 46 django/contrib/gis/db/backends/postgis/operations.py | 298 | 333| 313 | 32113 | 85810 | 
| 97 | 47 django/db/backends/dummy/base.py | 1 | 47| 270 | 32383 | 86255 | 
| 98 | 47 django/db/backends/base/base.py | 26 | 115| 789 | 33172 | 86255 | 
| 99 | 47 django/db/backends/oracle/creation.py | 102 | 128| 314 | 33486 | 86255 | 
| 100 | 47 django/db/backends/base/base.py | 207 | 280| 497 | 33983 | 86255 | 
| 101 | 48 django/db/__init__.py | 1 | 43| 272 | 34255 | 86527 | 
| 102 | 48 django/db/backends/mysql/creation.py | 32 | 56| 253 | 34508 | 86527 | 
| 103 | 48 django/db/backends/sqlite3/base.py | 176 | 204| 297 | 34805 | 86527 | 
| 104 | 49 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 35122 | 87378 | 
| 105 | 49 django/db/backends/postgresql/introspection.py | 100 | 117| 257 | 35379 | 87378 | 
| 106 | 50 django/db/backends/sqlite3/operations.py | 320 | 365| 453 | 35832 | 90463 | 
| 107 | 50 django/db/backends/oracle/operations.py | 333 | 344| 227 | 36059 | 90463 | 
| 108 | 50 django/contrib/gis/db/backends/postgis/operations.py | 1 | 25| 216 | 36275 | 90463 | 
| 109 | 50 django/db/backends/oracle/operations.py | 304 | 331| 271 | 36546 | 90463 | 
| 110 | 51 django/contrib/gis/db/backends/mysql/features.py | 1 | 44| 310 | 36856 | 90774 | 
| 111 | 51 django/contrib/postgres/constraints.py | 93 | 106| 179 | 37035 | 90774 | 
| 112 | 51 django/db/utils.py | 180 | 213| 224 | 37259 | 90774 | 
| 113 | 51 django/db/backends/mysql/operations.py | 278 | 289| 165 | 37424 | 90774 | 
| 114 | 51 django/contrib/postgres/search.py | 27 | 74| 281 | 37705 | 90774 | 
| 115 | 51 django/contrib/postgres/operations.py | 145 | 166| 261 | 37966 | 90774 | 
| 116 | 51 django/db/utils.py | 101 | 131| 323 | 38289 | 90774 | 
| 117 | 52 django/contrib/postgres/fields/jsonb.py | 1 | 15| 0 | 38289 | 90862 | 
| 118 | 52 django/contrib/postgres/constraints.py | 69 | 91| 213 | 38502 | 90862 | 
| 119 | 52 django/db/backends/sqlite3/base.py | 251 | 267| 374 | 38876 | 90862 | 
| 120 | 52 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 39523 | 90862 | 
| 121 | 53 django/db/backends/base/client.py | 1 | 27| 192 | 39715 | 91054 | 
| 122 | 54 django/contrib/postgres/aggregates/general.py | 1 | 69| 423 | 40138 | 91478 | 
| 123 | 55 django/contrib/postgres/fields/ranges.py | 231 | 321| 479 | 40617 | 93570 | 
| 124 | 56 django/contrib/gis/db/backends/spatialite/features.py | 1 | 25| 164 | 40781 | 93735 | 
| 125 | 56 django/db/backends/mysql/operations.py | 219 | 276| 431 | 41212 | 93735 | 
| 126 | 56 django/db/backends/oracle/base.py | 38 | 60| 210 | 41422 | 93735 | 
| 127 | 56 django/db/models/fields/json.py | 198 | 216| 232 | 41654 | 93735 | 
| 128 | 57 django/db/backends/oracle/client.py | 1 | 28| 163 | 41817 | 93899 | 
| 129 | 57 django/db/backends/postgresql/operations.py | 29 | 39| 170 | 41987 | 93899 | 
| 130 | 57 django/db/backends/oracle/operations.py | 563 | 579| 290 | 42277 | 93899 | 
| 131 | 57 django/db/backends/base/creation.py | 158 | 179| 203 | 42480 | 93899 | 
| 132 | 58 django/contrib/gis/db/backends/base/features.py | 1 | 112| 821 | 43301 | 94721 | 
| 133 | 58 django/db/backends/base/operations.py | 311 | 380| 529 | 43830 | 94721 | 
| 134 | 58 django/contrib/postgres/constraints.py | 1 | 67| 550 | 44380 | 94721 | 
| 135 | 58 django/contrib/postgres/search.py | 130 | 157| 248 | 44628 | 94721 | 
| 136 | 58 django/db/backends/base/creation.py | 102 | 137| 289 | 44917 | 94721 | 
| 137 | 58 django/db/backends/postgresql/introspection.py | 142 | 213| 755 | 45672 | 94721 | 
| 138 | 59 django/contrib/gis/db/backends/spatialite/base.py | 1 | 36| 331 | 46003 | 95340 | 
| 139 | 59 django/db/backends/sqlite3/operations.py | 170 | 195| 190 | 46193 | 95340 | 
| 140 | 60 django/db/backends/mysql/schema.py | 1 | 38| 409 | 46602 | 96862 | 
| 141 | 60 django/db/backends/base/creation.py | 324 | 343| 121 | 46723 | 96862 | 
| 142 | 60 django/db/backends/mysql/schema.py | 51 | 87| 349 | 47072 | 96862 | 
| 143 | 60 django/db/backends/oracle/operations.py | 369 | 406| 369 | 47441 | 96862 | 
| 144 | 61 django/db/models/functions/comparison.py | 45 | 56| 171 | 47612 | 98546 | 
| 145 | 62 django/db/backends/mysql/base.py | 194 | 229| 330 | 47942 | 101913 | 
| 146 | 62 django/db/backends/oracle/creation.py | 220 | 251| 390 | 48332 | 101913 | 
| 147 | 62 django/contrib/postgres/constraints.py | 157 | 167| 132 | 48464 | 101913 | 
| 148 | 62 django/db/backends/mysql/base.py | 168 | 192| 199 | 48663 | 101913 | 
| 149 | 62 django/contrib/gis/db/backends/mysql/operations.py | 57 | 77| 225 | 48888 | 101913 | 
| 150 | 63 django/contrib/gis/db/backends/postgis/features.py | 1 | 14| 0 | 48888 | 102009 | 
| 151 | 63 django/db/backends/oracle/operations.py | 346 | 367| 228 | 49116 | 102009 | 
| 152 | 63 django/db/backends/base/operations.py | 1 | 100| 829 | 49945 | 102009 | 
| 153 | 64 django/contrib/gis/db/backends/oracle/operations.py | 52 | 115| 766 | 50711 | 104091 | 
| 154 | 64 django/db/backends/oracle/operations.py | 289 | 302| 240 | 50951 | 104091 | 
| 155 | 64 django/contrib/gis/db/backends/spatialite/base.py | 38 | 75| 295 | 51246 | 104091 | 
| 156 | 64 django/contrib/postgres/indexes.py | 184 | 201| 137 | 51383 | 104091 | 
| 157 | 64 django/db/backends/base/base.py | 560 | 607| 300 | 51683 | 104091 | 
| 158 | 65 django/db/backends/base/schema.py | 45 | 112| 769 | 52452 | 116589 | 
| 159 | 65 django/db/backends/oracle/operations.py | 273 | 287| 206 | 52658 | 116589 | 
| 160 | 65 django/db/backends/mysql/creation.py | 58 | 69| 178 | 52836 | 116589 | 
| 161 | 65 django/contrib/postgres/indexes.py | 83 | 107| 266 | 53102 | 116589 | 
| 162 | 65 django/contrib/postgres/operations.py | 238 | 259| 163 | 53265 | 116589 | 
| 163 | 65 django/contrib/postgres/fields/ranges.py | 199 | 228| 279 | 53544 | 116589 | 
| 164 | 65 django/db/backends/postgresql/schema.py | 69 | 99| 254 | 53798 | 116589 | 
| 165 | 65 django/db/backends/oracle/creation.py | 130 | 165| 399 | 54197 | 116589 | 
| 166 | 65 django/db/backends/mysql/base.py | 329 | 354| 213 | 54410 | 116589 | 
| 167 | 65 django/db/backends/postgresql/introspection.py | 60 | 98| 428 | 54838 | 116589 | 
| 168 | 65 django/contrib/postgres/indexes.py | 130 | 152| 224 | 55062 | 116589 | 
| 169 | 65 django/contrib/gis/db/backends/postgis/adapter.py | 57 | 70| 131 | 55193 | 116589 | 
| 170 | 65 django/db/backends/dummy/base.py | 50 | 74| 173 | 55366 | 116589 | 
| 171 | 66 django/contrib/gis/db/backends/postgis/introspection.py | 1 | 27| 268 | 55634 | 117192 | 
| 172 | 66 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 55774 | 117192 | 
| 173 | 66 django/db/backends/mysql/base.py | 231 | 249| 165 | 55939 | 117192 | 
| 174 | 66 django/db/backends/sqlite3/operations.py | 197 | 216| 209 | 56148 | 117192 | 
| 175 | 66 django/db/backends/postgresql/introspection.py | 1 | 45| 353 | 56501 | 117192 | 
| 176 | 66 django/utils/connection.py | 1 | 31| 192 | 56693 | 117192 | 
| 177 | 66 django/db/backends/base/features.py | 345 | 363| 161 | 56854 | 117192 | 
| 178 | 66 django/db/backends/oracle/operations.py | 21 | 73| 574 | 57428 | 117192 | 
| 179 | 67 django/db/models/query.py | 1217 | 1245| 183 | 57611 | 134501 | 
| 180 | 67 django/contrib/postgres/search.py | 95 | 127| 277 | 57888 | 134501 | 
| 181 | 67 django/db/backends/oracle/creation.py | 187 | 218| 319 | 58207 | 134501 | 
| 182 | 68 django/core/cache/backends/db.py | 97 | 110| 234 | 58441 | 136623 | 
| 183 | 69 django/conf/global_settings.py | 401 | 495| 782 | 59223 | 142306 | 
| 184 | 69 django/db/backends/mysql/operations.py | 291 | 319| 243 | 59466 | 142306 | 
| 185 | 69 django/db/backends/oracle/creation.py | 253 | 281| 277 | 59743 | 142306 | 
| 186 | 69 django/db/backends/base/base.py | 527 | 558| 227 | 59970 | 142306 | 
| 187 | 69 django/db/backends/oracle/operations.py | 598 | 617| 303 | 60273 | 142306 | 
| 188 | 69 django/db/backends/base/base.py | 140 | 182| 335 | 60608 | 142306 | 
| 189 | 69 django/db/backends/oracle/operations.py | 104 | 115| 212 | 60820 | 142306 | 
| 190 | 69 django/conf/global_settings.py | 151 | 266| 859 | 61679 | 142306 | 
| 191 | 69 django/db/backends/base/base.py | 184 | 205| 208 | 61887 | 142306 | 


### Hint

```
Configuration without NAME already works for me, e.g.: 'default': { 'ENGINE': 'django.db.backends.postgresql', 'OPTIONS': { 'service': 'default_django_test' } }, so only setting PGSERVICE for ​the underlying command-line client and docs are missing. I don't mind creating a pull request for it, but would like to know why it hasn't been added, ... Because you're the first to suggest it.
Replying to Mariusz Felisiak: Configuration without NAME already works for me, e.g.: 'default': { 'ENGINE': 'django.db.backends.postgresql', 'OPTIONS': { 'service': 'default_django_test' } }, It doesn't work for me. E.g.: 'default': { 'ENGINE': 'django.db.backends.postgresql', 'OPTIONS': { 'service': 'test_service' } }, Throws the following when it tries to access the database: Traceback (most recent call last): File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/core/handlers/exception.py", line 47, in inner response = get_response(request) File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/core/handlers/base.py", line 179, in _get_response response = wrapped_callback(request, *callback_args, **callback_kwargs) File "/redacted_path/postgres_services/mysite/testapp/views.py", line 9, in index c = db_conn.cursor() File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/utils/asyncio.py", line 26, in inner return func(*args, **kwargs) File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/db/backends/base/base.py", line 259, in cursor return self._cursor() File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/db/backends/base/base.py", line 235, in _cursor self.ensure_connection() File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/utils/asyncio.py", line 26, in inner return func(*args, **kwargs) File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/db/backends/base/base.py", line 219, in ensure_connection self.connect() File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/utils/asyncio.py", line 26, in inner return func(*args, **kwargs) File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/db/backends/base/base.py", line 199, in connect conn_params = self.get_connection_params() File "/redacted_path/postgres_services/venv/lib/python3.8/site-packages/django/db/backends/postgresql/base.py", line 157, in get_connection_params raise ImproperlyConfigured( django.core.exceptions.ImproperlyConfigured: settings.DATABASES is improperly configured. Please supply the NAME value. While if I just add the NAME value it works just fine. This makes total sense as if you check the postgres base file it checks: if settings_dict['NAME'] == '': raise ImproperlyConfigured( "settings.DATABASES is improperly configured. " "Please supply the NAME value.") if len(settings_dict['NAME'] or '') > self.ops.max_name_length(): raise ImproperlyConfigured( "The database name '%s' (%d characters) is longer than " "PostgreSQL's limit of %d characters. Supply a shorter NAME " "in settings.DATABASES." % ( settings_dict['NAME'], len(settings_dict['NAME']), self.ops.max_name_length(), ) ) The first if evaluates to true. settings_dict['NAME'] must be getting defaulted to an empty string, because I'm not setting it to empty, I'm entirely leaving it out. so only setting PGSERVICE for ​the underlying command-line client and docs are missing. I don't mind adding support for those either. I don't mind creating a pull request for it, but would like to know why it hasn't been added, ... Because you're the first to suggest it. Oh I was sure I wouldn't have been. I've looked up how to use postgres services for many projects, and nearly always find people asking, even when it's for much much smaller projects. That's rather interesting.
It doesn't work for me. E.g.: You're right. I've only checked running tests which works fine.
```

## Patch

```diff
diff --git a/django/db/backends/postgresql/base.py b/django/db/backends/postgresql/base.py
--- a/django/db/backends/postgresql/base.py
+++ b/django/db/backends/postgresql/base.py
@@ -152,10 +152,14 @@ class DatabaseWrapper(BaseDatabaseWrapper):
     def get_connection_params(self):
         settings_dict = self.settings_dict
         # None may be used to connect to the default 'postgres' db
-        if settings_dict['NAME'] == '':
+        if (
+            settings_dict['NAME'] == '' and
+            not settings_dict.get('OPTIONS', {}).get('service')
+        ):
             raise ImproperlyConfigured(
                 "settings.DATABASES is improperly configured. "
-                "Please supply the NAME value.")
+                "Please supply the NAME or OPTIONS['service'] value."
+            )
         if len(settings_dict['NAME'] or '') > self.ops.max_name_length():
             raise ImproperlyConfigured(
                 "The database name '%s' (%d characters) is longer than "
@@ -166,10 +170,19 @@ def get_connection_params(self):
                     self.ops.max_name_length(),
                 )
             )
-        conn_params = {
-            'database': settings_dict['NAME'] or 'postgres',
-            **settings_dict['OPTIONS'],
-        }
+        conn_params = {}
+        if settings_dict['NAME']:
+            conn_params = {
+                'database': settings_dict['NAME'],
+                **settings_dict['OPTIONS'],
+            }
+        elif settings_dict['NAME'] is None:
+            # Connect to the default 'postgres' db.
+            settings_dict.get('OPTIONS', {}).pop('service', None)
+            conn_params = {'database': 'postgres', **settings_dict['OPTIONS']}
+        else:
+            conn_params = {**settings_dict['OPTIONS']}
+
         conn_params.pop('isolation_level', None)
         if settings_dict['USER']:
             conn_params['user'] = settings_dict['USER']
diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py
--- a/django/db/backends/postgresql/client.py
+++ b/django/db/backends/postgresql/client.py
@@ -16,6 +16,7 @@ def settings_to_cmd_args_env(cls, settings_dict, parameters):
         dbname = settings_dict.get('NAME') or 'postgres'
         user = settings_dict.get('USER')
         passwd = settings_dict.get('PASSWORD')
+        service = options.get('service')
         sslmode = options.get('sslmode')
         sslrootcert = options.get('sslrootcert')
         sslcert = options.get('sslcert')
@@ -33,6 +34,8 @@ def settings_to_cmd_args_env(cls, settings_dict, parameters):
         env = {}
         if passwd:
             env['PGPASSWORD'] = str(passwd)
+        if service:
+            env['PGSERVICE'] = str(service)
         if sslmode:
             env['PGSSLMODE'] = str(sslmode)
         if sslrootcert:

```

## Test Patch

```diff
diff --git a/tests/backends/postgresql/tests.py b/tests/backends/postgresql/tests.py
--- a/tests/backends/postgresql/tests.py
+++ b/tests/backends/postgresql/tests.py
@@ -68,6 +68,36 @@ def test_database_name_too_long(self):
         with self.assertRaisesMessage(ImproperlyConfigured, msg):
             DatabaseWrapper(settings).get_connection_params()
 
+    def test_database_name_empty(self):
+        from django.db.backends.postgresql.base import DatabaseWrapper
+        settings = connection.settings_dict.copy()
+        settings['NAME'] = ''
+        msg = (
+            "settings.DATABASES is improperly configured. Please supply the "
+            "NAME or OPTIONS['service'] value."
+        )
+        with self.assertRaisesMessage(ImproperlyConfigured, msg):
+            DatabaseWrapper(settings).get_connection_params()
+
+    def test_service_name(self):
+        from django.db.backends.postgresql.base import DatabaseWrapper
+        settings = connection.settings_dict.copy()
+        settings['OPTIONS'] = {'service': 'my_service'}
+        settings['NAME'] = ''
+        params = DatabaseWrapper(settings).get_connection_params()
+        self.assertEqual(params['service'], 'my_service')
+        self.assertNotIn('database', params)
+
+    def test_service_name_default_db(self):
+        # None is used to connect to the default 'postgres' db.
+        from django.db.backends.postgresql.base import DatabaseWrapper
+        settings = connection.settings_dict.copy()
+        settings['NAME'] = None
+        settings['OPTIONS'] = {'service': 'django_test'}
+        params = DatabaseWrapper(settings).get_connection_params()
+        self.assertEqual(params['database'], 'postgres')
+        self.assertNotIn('service', params)
+
     def test_connect_and_rollback(self):
         """
         PostgreSQL shouldn't roll back SET TIME ZONE, even if the first
diff --git a/tests/dbshell/test_postgresql.py b/tests/dbshell/test_postgresql.py
--- a/tests/dbshell/test_postgresql.py
+++ b/tests/dbshell/test_postgresql.py
@@ -67,6 +67,12 @@ def test_ssl_certificate(self):
             )
         )
 
+    def test_service(self):
+        self.assertEqual(
+            self.settings_to_cmd_args_env({'OPTIONS': {'service': 'django_test'}}),
+            (['psql', 'postgres'], {'PGSERVICE': 'django_test'}),
+        )
+
     def test_column(self):
         self.assertEqual(
             self.settings_to_cmd_args_env({

```


## Code snippets

### 1 - django/db/backends/postgresql/base.py:

Start line: 1, End line: 62

```python
"""
PostgreSQL database backend for Django.

Requires psycopg 2: https://www.psycopg.org/
"""

import asyncio
import threading
import warnings
from contextlib import contextmanager

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError as WrappedDatabaseError, connections
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.utils import (
    CursorDebugWrapper as BaseCursorDebugWrapper,
)
from django.utils.asyncio import async_unsafe
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
from .client import DatabaseClient  # NOQA
from .creation import DatabaseCreation  # NOQA
from .features import DatabaseFeatures  # NOQA
from .introspection import DatabaseIntrospection  # NOQA
from .operations import DatabaseOperations  # NOQA
from .schema import DatabaseSchemaEditor  # NOQA

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
### 2 - django/contrib/postgres/apps.py:

Start line: 43, End line: 70

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
        IndexExpression.register_wrappers(OrderBy, OpClass, Collate)
```
### 3 - django/db/backends/postgresql/base.py:

Start line: 152, End line: 182

```python
class DatabaseWrapper(BaseDatabaseWrapper):

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
### 4 - django/db/backends/postgresql/features.py:

Start line: 1, End line: 97

```python
import operator

from django.db import InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    allows_group_by_selected_pks = True
    can_return_columns_from_insert = True
    can_return_rows_from_bulk_insert = True
    has_real_datatype = True
    has_native_uuid_field = True
    has_native_duration_field = True
    has_native_json_field = True
    can_defer_constraint_checks = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_of = True
    has_select_for_update_skip_locked = True
    has_select_for_no_key_update = True
    can_release_savepoints = True
    supports_tablespaces = True
    supports_transactions = True
    can_introspect_materialized_views = True
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
    only_supports_unbounded_with_preceding_and_following = True
    supports_aggregate_filter_clause = True
    supported_explain_formats = {'JSON', 'TEXT', 'XML', 'YAML'}
    validates_explain_options = False  # A query will error on invalid options.
    supports_deferrable_unique_constraints = True
    has_json_operators = True
    json_key_contains_list_matching_requires_list = True
    test_collations = {
        'non_default': 'sv-x-icu',
        'swedish_ci': 'sv-x-icu',
    }

    django_test_skips = {
        'opclasses are PostgreSQL only.': {
            'indexes.tests.SchemaIndexesNotPostgreSQLTests.test_create_index_ignores_opclasses',
        },
    }

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            'PositiveBigIntegerField': 'BigIntegerField',
            'PositiveIntegerField': 'IntegerField',
            'PositiveSmallIntegerField': 'SmallIntegerField',
        }

    @cached_property
    def is_postgresql_11(self):
        return self.connection.pg_version >= 110000

    @cached_property
    def is_postgresql_12(self):
        return self.connection.pg_version >= 120000

    @cached_property
    def is_postgresql_13(self):
        return self.connection.pg_version >= 130000

    has_websearch_to_tsquery = property(operator.attrgetter('is_postgresql_11'))
    supports_covering_indexes = property(operator.attrgetter('is_postgresql_11'))
    supports_covering_gist_indexes = property(operator.attrgetter('is_postgresql_12'))
    supports_non_deterministic_collations = property(operator.attrgetter('is_postgresql_12'))
```
### 5 - django/db/backends/postgresql/client.py:

Start line: 1, End line: 55

```python
import signal

from django.db.backends.base.client import BaseDatabaseClient


class DatabaseClient(BaseDatabaseClient):
    executable_name = 'psql'

    @classmethod
    def settings_to_cmd_args_env(cls, settings_dict, parameters):
        args = [cls.executable_name]
        options = settings_dict.get('OPTIONS', {})

        host = settings_dict.get('HOST')
        port = settings_dict.get('PORT')
        dbname = settings_dict.get('NAME') or 'postgres'
        user = settings_dict.get('USER')
        passwd = settings_dict.get('PASSWORD')
        sslmode = options.get('sslmode')
        sslrootcert = options.get('sslrootcert')
        sslcert = options.get('sslcert')
        sslkey = options.get('sslkey')

        if user:
            args += ['-U', user]
        if host:
            args += ['-h', host]
        if port:
            args += ['-p', str(port)]
        args += [dbname]
        args.extend(parameters)

        env = {}
        if passwd:
            env['PGPASSWORD'] = str(passwd)
        if sslmode:
            env['PGSSLMODE'] = str(sslmode)
        if sslrootcert:
            env['PGSSLROOTCERT'] = str(sslrootcert)
        if sslcert:
            env['PGSSLCERT'] = str(sslcert)
        if sslkey:
            env['PGSSLKEY'] = str(sslkey)
        return args, env

    def runshell(self, parameters):
        sigint_handler = signal.getsignal(signal.SIGINT)
        try:
            # Allow SIGINT to pass to psql to abort queries.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            super().runshell(parameters)
        finally:
            # Restore the original SIGINT handler.
            signal.signal(signal.SIGINT, sigint_handler)
```
### 6 - django/db/backends/postgresql/creation.py:

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
### 7 - django/db/backends/postgresql/base.py:

Start line: 297, End line: 340

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    @contextmanager
    def _nodb_cursor(self):
        try:
            with super()._nodb_cursor() as cursor:
                yield cursor
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
                    conn = self.__class__(
                        {**self.settings_dict, 'NAME': connection.settings_dict['NAME']},
                        alias=self.alias,
                    )
                    try:
                        with conn.cursor() as cursor:
                            yield cursor
                    finally:
                        conn.close()

    @cached_property
    def pg_version(self):
        with self.temporary_connection():
            return self.connection.server_version

    def make_debug_cursor(self, cursor):
        return CursorDebugWrapper(cursor, self)


class CursorDebugWrapper(BaseCursorDebugWrapper):
    def copy_expert(self, sql, file, *args):
        with self.debug_sql(sql):
            return self.cursor.copy_expert(sql, file, *args)

    def copy_to(self, file, table, *args, **kwargs):
        with self.debug_sql(sql='COPY %s TO STDOUT' % table):
            return self.cursor.copy_to(file, table, *args, **kwargs)
```
### 8 - django/db/backends/postgresql/base.py:

Start line: 184, End line: 206

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    @async_unsafe
    def get_new_connection(self, conn_params):
        connection = Database.connect(**conn_params)

        # self.isolation_level must be set:
        # - after connecting to the database in order to obtain the database's
        #   default when no value is explicitly specified in options.
        # - before calling _set_autocommit() because if autocommit is on, that
        #   will set connection.isolation_level to ISOLATION_LEVEL_AUTOCOMMIT.
        options = self.settings_dict['OPTIONS']
        try:
            self.isolation_level = options['isolation_level']
        except KeyError:
            self.isolation_level = connection.isolation_level
        else:
            # Set the isolation level to the value from OPTIONS.
            if self.isolation_level != connection.isolation_level:
                connection.set_session(isolation_level=self.isolation_level)
        # Register dummy loads() to avoid a round trip from psycopg2's decode
        # to json.dumps() to json.loads(), when using a custom decoder in
        # JSONField.
        psycopg2.extras.register_default_jsonb(conn_or_curs=connection, loads=lambda x: x)
        return connection
```
### 9 - django/contrib/gis/db/backends/postgis/base.py:

Start line: 1, End line: 27

```python
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import (
    DatabaseWrapper as Psycopg2DatabaseWrapper,
)

from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor


class DatabaseWrapper(Psycopg2DatabaseWrapper):
    SchemaEditorClass = PostGISSchemaEditor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get('alias', '') != NO_DB_ALIAS:
            self.features = DatabaseFeatures(self)
            self.ops = PostGISOperations(self)
            self.introspection = PostGISIntrospection(self)

    def prepare_database(self):
        super().prepare_database()
        # Check that postgis extension is installed.
        with self.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis")
```
### 10 - django/db/backends/postgresql/operations.py:

Start line: 189, End line: 276

```python
class DatabaseOperations(BaseDatabaseOperations):

    def prep_for_iexact_query(self, x):
        return x

    def max_name_length(self):
        """
        Return the maximum length of an identifier.

        The maximum length of an identifier is 63 by default, but can be
        changed by recompiling PostgreSQL after editing the NAMEDATALEN
        macro in src/include/pg_config_manual.h.

        This implementation returns 63, but can be overridden by a custom
        database backend that inherits most of its behavior from this one.
        """
        return 63

    def distinct_sql(self, fields, params):
        if fields:
            params = [param for param_list in params for param in param_list]
            return (['DISTINCT ON (%s)' % ', '.join(fields)], params)
        else:
            return ['DISTINCT'], []

    def last_executed_query(self, cursor, sql, params):
        # https://www.psycopg.org/docs/cursor.html#cursor.query
        # The query attribute is a Psycopg extension to the DB API 2.0.
        if cursor.query is not None:
            return cursor.query.decode()
        return None

    def return_insert_columns(self, fields):
        if not fields:
            return '', ()
        columns = [
            '%s.%s' % (
                self.quote_name(field.model._meta.db_table),
                self.quote_name(field.column),
            ) for field in fields
        ]
        return 'RETURNING %s' % ', '.join(columns), ()

    def bulk_insert_sql(self, fields, placeholder_rows):
        placeholder_rows_sql = (", ".join(row) for row in placeholder_rows)
        values_sql = ", ".join("(%s)" % sql for sql in placeholder_rows_sql)
        return "VALUES " + values_sql

    def adapt_datefield_value(self, value):
        return value

    def adapt_datetimefield_value(self, value):
        return value

    def adapt_timefield_value(self, value):
        return value

    def adapt_decimalfield_value(self, value, max_digits=None, decimal_places=None):
        return value

    def adapt_ipaddressfield_value(self, value):
        if value:
            return Inet(value)
        return None

    def subtract_temporals(self, internal_type, lhs, rhs):
        if internal_type == 'DateField':
            lhs_sql, lhs_params = lhs
            rhs_sql, rhs_params = rhs
            params = (*lhs_params, *rhs_params)
            return "(interval '1 day' * (%s - %s))" % (lhs_sql, rhs_sql), params
        return super().subtract_temporals(internal_type, lhs, rhs)

    def explain_query_prefix(self, format=None, **options):
        prefix = super().explain_query_prefix(format)
        extra = {}
        if format:
            extra['FORMAT'] = format
        if options:
            extra.update({
                name.upper(): 'true' if value else 'false'
                for name, value in options.items()
            })
        if extra:
            prefix += ' (%s)' % ', '.join('%s %s' % i for i in extra.items())
        return prefix

    def ignore_conflicts_suffix_sql(self, ignore_conflicts=None):
        return 'ON CONFLICT DO NOTHING' if ignore_conflicts else super().ignore_conflicts_suffix_sql(ignore_conflicts)
```
### 12 - django/db/backends/postgresql/base.py:

Start line: 132, End line: 150

```python
class DatabaseWrapper(BaseDatabaseWrapper):
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
    creation_class = DatabaseCreation
    features_class = DatabaseFeatures
    introspection_class = DatabaseIntrospection
    ops_class = DatabaseOperations
    # PostgreSQL backend-specific attributes.
    _named_cursor_idx = 0
```
### 21 - django/db/backends/postgresql/base.py:

Start line: 208, End line: 240

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    def ensure_timezone(self):
        if self.connection is None:
            return False
        conn_timezone_name = self.connection.get_parameter_status('TimeZone')
        timezone_name = self.timezone_name
        if timezone_name and conn_timezone_name != timezone_name:
            with self.connection.cursor() as cursor:
                cursor.execute(self.ops.set_time_zone_sql(), [timezone_name])
            return True
        return False

    def init_connection_state(self):
        self.connection.set_client_encoding('UTF8')

        timezone_changed = self.ensure_timezone()
        if timezone_changed:
            # Commit after setting the time zone (see #17062)
            if not self.get_autocommit():
                self.connection.commit()

    @async_unsafe
    def create_cursor(self, name=None):
        if name:
            # In autocommit mode, the cursor will be used outside of a
            # transaction, hence use a holdable cursor.
            cursor = self.connection.cursor(name, scrollable=False, withhold=self.connection.autocommit)
        else:
            cursor = self.connection.cursor()
        cursor.tzinfo_factory = self.tzinfo_factory if settings.USE_TZ else None
        return cursor

    def tzinfo_factory(self, offset):
        return self.timezone
```
### 23 - django/db/backends/postgresql/base.py:

Start line: 65, End line: 131

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
        'JSONField': 'jsonb',
        'OneToOneField': 'integer',
        'PositiveBigIntegerField': 'bigint',
        'PositiveIntegerField': 'integer',
        'PositiveSmallIntegerField': 'smallint',
        'SlugField': 'varchar(%(max_length)s)',
        'SmallAutoField': 'smallserial',
        'SmallIntegerField': 'smallint',
        'TextField': 'text',
        'TimeField': 'time',
        'UUIDField': 'uuid',
    }
    data_type_check_constraints = {
        'PositiveBigIntegerField': '"%(column)s" >= 0',
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
```
### 52 - django/db/backends/postgresql/base.py:

Start line: 274, End line: 295

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    def _set_autocommit(self, autocommit):
        with self.wrap_database_errors:
            self.connection.autocommit = autocommit

    def check_constraints(self, table_names=None):
        """
        Check constraints by setting them to immediate. Return them to deferred
        afterward.
        """
        with self.cursor() as cursor:
            cursor.execute('SET CONSTRAINTS ALL IMMEDIATE')
            cursor.execute('SET CONSTRAINTS ALL DEFERRED')

    def is_usable(self):
        try:
            # Use a psycopg cursor directly, bypassing Django's utilities.
            with self.connection.cursor() as cursor:
                cursor.execute('SELECT 1')
        except Database.Error:
            return False
        else:
            return True
```
### 70 - django/db/backends/postgresql/base.py:

Start line: 242, End line: 272

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    @async_unsafe
    def chunked_cursor(self):
        self._named_cursor_idx += 1
        # Get the current async task
        # Note that right now this is behind @async_unsafe, so this is
        # unreachable, but in future we'll start loosening this restriction.
        # For now, it's here so that every use of "threading" is
        # also async-compatible.
        try:
            if hasattr(asyncio, 'current_task'):
                # Python 3.7 and up
                current_task = asyncio.current_task()
            else:
                # Python 3.6
                current_task = asyncio.Task.current_task()
        except RuntimeError:
            current_task = None
        # Current task can be none even if the current_task call didn't error
        if current_task:
            task_ident = str(id(current_task))
        else:
            task_ident = 'sync'
        # Use that and the thread ident to get a unique name
        return self._cursor(
            name='_django_curs_%d_%s_%d' % (
                # Avoid reusing name in other threads / tasks
                threading.current_thread().ident,
                task_ident,
                self._named_cursor_idx,
            )
        )
```
