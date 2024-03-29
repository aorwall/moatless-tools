# django__django-11584

| **django/django** | `fed5e19369f456e41f0768f4fb92602af027a46d` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 29301 |
| **Any found context length** | 29301 |
| **Avg pos** | 55.0 |
| **Min pos** | 110 |
| **Max pos** | 110 |
| **Top file pos** | 10 |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -240,8 +240,15 @@ def __init__(self):
 
     def watch_dir(self, path, glob):
         path = Path(path)
-        if not path.is_absolute():
-            raise ValueError('%s must be absolute.' % path)
+        try:
+            path = path.absolute()
+        except FileNotFoundError:
+            logger.debug(
+                'Unable to watch directory %s as it cannot be resolved.',
+                path,
+                exc_info=True,
+            )
+            return
         logger.debug('Watching dir %s with glob %s.', path, glob)
         self.directory_globs[path].add(glob)
 
diff --git a/django/utils/translation/reloader.py b/django/utils/translation/reloader.py
--- a/django/utils/translation/reloader.py
+++ b/django/utils/translation/reloader.py
@@ -14,8 +14,7 @@ def watch_for_translation_changes(sender, **kwargs):
         directories.extend(Path(config.path) / 'locale' for config in apps.get_app_configs())
         directories.extend(Path(p) for p in settings.LOCALE_PATHS)
         for path in directories:
-            absolute_path = path.absolute()
-            sender.watch_dir(absolute_path, '**/*.mo')
+            sender.watch_dir(path, '**/*.mo')
 
 
 def translation_file_changed(sender, file_path, **kwargs):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/autoreload.py | 243 | 244 | 110 | 10 | 29301
| django/utils/translation/reloader.py | 17 | 18 | - | - | -


## Problem Statement

```
[FATAL] FileNotFoundError with runserver command inside Docker container
Description
	
Summary
Trying to run the development server in a container with volume-mounted source is throwing a FileNotFoundError. I've verified that the issue is consistently reproducible with Django==2.2.3 and not present in Django==2.1.4.
Trace
**INFO** /code/publications/models.py changed, reloading.
**INFO** Watching for file changes with StatReloader
Performing system checks...
Traceback (most recent call last):
 File "manage.py", line 21, in <module>
	main()
 File "manage.py", line 17, in main
	execute_from_command_line(sys.argv)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/__init__.py", line 381, in execute_from_command_line
	utility.execute()
 File "/usr/local/lib/python3.6/site-packages/django/core/management/__init__.py", line 375, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/base.py", line 323, in run_from_argv
	self.execute(*args, **cmd_options)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 60, in execute
	super().execute(*args, **options)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/base.py", line 364, in execute
	output = self.handle(*args, **options)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 95, in handle
	self.run(**options)
 File "/usr/local/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 102, in run
	autoreload.run_with_reloader(self.inner_run, **options)
 File "/usr/local/lib/python3.6/site-packages/django/utils/autoreload.py", line 587, in run_with_reloader
	start_django(reloader, main_func, *args, **kwargs)
 File "/usr/local/lib/python3.6/site-packages/django/utils/autoreload.py", line 572, in start_django
	reloader.run(django_main_thread)
 File "/usr/local/lib/python3.6/site-packages/django/utils/autoreload.py", line 289, in run
	autoreload_started.send(sender=self)
 File "/usr/local/lib/python3.6/site-packages/django/dispatch/dispatcher.py", line 175, in send
	for receiver in self._live_receivers(sender)
 File "/usr/local/lib/python3.6/site-packages/django/dispatch/dispatcher.py", line 175, in <listcomp>
	for receiver in self._live_receivers(sender)
 File "/usr/local/lib/python3.6/site-packages/django/utils/translation/reloader.py", line 16, in watch_for_translation_changes
	absolute_path = path.absolute()
 File "/usr/local/lib/python3.6/pathlib.py", line 1129, in absolute
	obj = self._from_parts([os.getcwd()] + self._parts, init=False)
FileNotFoundError: [Errno 2] No such file or directory
Configuration
Dockerfile
FROM python:3.6.7-alpine3.7
RUN mkdir /code
WORKDIR /code
RUN apk add postgresql-dev libffi-dev build-base musl-dev
RUN apk add linux-headers
ADD requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 3031
ADD cs /code
docker-compose
version: '3.7'
services:
 db:
	image: postgres
	volumes:
	 - ./pg_data:/var/lib/postgresql/data
	ports:
	 - "5432:5432"
	environment:
	 POSTGRES_PASSWORD: postgres
	 POSTGRES_USER: postgres
	 POSTGRES_DB: postgres
 app:
	build:
	 context: .
	volumes:
	 - ./cs/:/code/
	ports:
	 - "8000:8000"
	env_file: .env
	command: ["python", "manage.py", "runserver", "0.0.0.0:8000"]

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/management/commands/runserver.py | 106 | 162| 517 | 517 | 1451 | 
| 2 | 1 django/core/management/commands/runserver.py | 66 | 104| 397 | 914 | 1451 | 
| 3 | 1 django/core/management/commands/runserver.py | 23 | 52| 240 | 1154 | 1451 | 
| 4 | 1 django/core/management/commands/runserver.py | 1 | 20| 191 | 1345 | 1451 | 
| 5 | 2 django/core/management/commands/testserver.py | 29 | 55| 234 | 1579 | 1885 | 
| 6 | 2 django/core/management/commands/testserver.py | 1 | 27| 205 | 1784 | 1885 | 
| 7 | 3 django/core/servers/basehttp.py | 200 | 217| 210 | 1994 | 3630 | 
| 8 | 4 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 2246 | 3883 | 
| 9 | 5 django/db/backends/postgresql/client.py | 1 | 55| 400 | 2646 | 4283 | 
| 10 | 6 django/db/backends/postgresql/base.py | 1 | 63| 500 | 3146 | 7033 | 
| 11 | 7 django/contrib/admin/checks.py | 718 | 749| 229 | 3375 | 16049 | 
| 12 | 8 django/db/backends/base/base.py | 1 | 23| 138 | 3513 | 20907 | 
| 13 | 9 django/db/utils.py | 1 | 49| 154 | 3667 | 23019 | 
| 14 | **10 django/utils/autoreload.py** | 567 | 583| 169 | 3836 | 27710 | 
| 15 | **10 django/utils/autoreload.py** | 586 | 598| 117 | 3953 | 27710 | 
| 16 | 10 django/core/management/commands/runserver.py | 54 | 64| 120 | 4073 | 27710 | 
| 17 | 11 django/core/management/commands/loaddata.py | 1 | 29| 151 | 4224 | 30578 | 
| 18 | 12 django/contrib/postgres/__init__.py | 1 | 2| 0 | 4224 | 30592 | 
| 19 | 13 django/db/__init__.py | 1 | 18| 141 | 4365 | 30985 | 
| 20 | 13 django/db/backends/postgresql/base.py | 287 | 326| 328 | 4693 | 30985 | 
| 21 | 14 django/db/backends/postgresql/utils.py | 1 | 8| 0 | 4693 | 31023 | 
| 22 | 15 django/db/backends/postgresql/creation.py | 36 | 51| 174 | 4867 | 31671 | 
| 23 | 16 django/core/management/__init__.py | 313 | 394| 743 | 5610 | 35019 | 
| 24 | 17 django/db/backends/mysql/client.py | 1 | 49| 422 | 6032 | 35441 | 
| 25 | 18 django/core/management/commands/startapp.py | 1 | 15| 0 | 6032 | 35542 | 
| 26 | 19 setup.py | 1 | 66| 508 | 6540 | 36565 | 
| 27 | 20 django/core/management/commands/dbshell.py | 1 | 32| 231 | 6771 | 36796 | 
| 28 | 21 django/contrib/postgres/apps.py | 40 | 67| 249 | 7020 | 37362 | 
| 29 | 21 django/db/backends/postgresql/creation.py | 53 | 78| 248 | 7268 | 37362 | 
| 30 | 22 django/core/management/commands/dumpdata.py | 170 | 194| 224 | 7492 | 38897 | 
| 31 | 23 django/core/management/commands/check.py | 1 | 34| 226 | 7718 | 39332 | 
| 32 | 24 django/core/management/commands/shell.py | 1 | 40| 268 | 7986 | 40153 | 
| 33 | 25 django/core/management/base.py | 347 | 382| 292 | 8278 | 44540 | 
| 34 | 26 django/core/management/commands/migrate.py | 21 | 65| 369 | 8647 | 47692 | 
| 35 | 26 django/core/management/commands/shell.py | 83 | 103| 162 | 8809 | 47692 | 
| 36 | 26 django/core/management/commands/migrate.py | 161 | 242| 793 | 9602 | 47692 | 
| 37 | 26 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 10109 | 47692 | 
| 38 | 27 django/utils/log.py | 1 | 76| 492 | 10601 | 49300 | 
| 39 | 28 django/db/backends/base/client.py | 1 | 13| 0 | 10601 | 49405 | 
| 40 | 28 django/core/management/commands/migrate.py | 67 | 160| 825 | 11426 | 49405 | 
| 41 | 29 django/core/management/commands/inspectdb.py | 1 | 36| 272 | 11698 | 52022 | 
| 42 | 30 django/db/models/base.py | 1 | 45| 289 | 11987 | 67112 | 
| 43 | 31 django/core/management/commands/test.py | 25 | 57| 260 | 12247 | 67521 | 
| 44 | 31 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 12873 | 67521 | 
| 45 | 31 django/core/servers/basehttp.py | 1 | 23| 164 | 13037 | 67521 | 
| 46 | 31 django/core/management/commands/shell.py | 42 | 81| 401 | 13438 | 67521 | 
| 47 | 31 django/core/management/commands/loaddata.py | 63 | 79| 187 | 13625 | 67521 | 
| 48 | 31 django/core/management/commands/migrate.py | 1 | 18| 148 | 13773 | 67521 | 
| 49 | 31 django/db/backends/postgresql/base.py | 143 | 180| 330 | 14103 | 67521 | 
| 50 | **31 django/utils/autoreload.py** | 1 | 45| 227 | 14330 | 67521 | 
| 51 | 31 django/core/management/commands/check.py | 36 | 66| 214 | 14544 | 67521 | 
| 52 | 32 django/core/management/commands/sqlmigrate.py | 32 | 69| 371 | 14915 | 68153 | 
| 53 | 33 django/core/management/commands/flush.py | 27 | 83| 496 | 15411 | 68850 | 
| 54 | 33 django/core/servers/basehttp.py | 122 | 157| 280 | 15691 | 68850 | 
| 55 | 34 django/db/backends/mysql/creation.py | 57 | 67| 149 | 15840 | 69459 | 
| 56 | 35 django/db/backends/sqlite3/client.py | 1 | 13| 0 | 15840 | 69517 | 
| 57 | 36 django/core/checks/security/base.py | 1 | 86| 752 | 16592 | 71143 | 
| 58 | 37 django/conf/global_settings.py | 499 | 638| 853 | 17445 | 76747 | 
| 59 | 37 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 17683 | 76747 | 
| 60 | 37 django/core/management/commands/loaddata.py | 32 | 61| 261 | 17944 | 76747 | 
| 61 | 38 django/contrib/gis/db/backends/postgis/base.py | 1 | 27| 197 | 18141 | 76944 | 
| 62 | **38 django/utils/autoreload.py** | 505 | 538| 228 | 18369 | 76944 | 
| 63 | 39 django/db/backends/sqlite3/creation.py | 48 | 79| 317 | 18686 | 77768 | 
| 64 | **39 django/utils/autoreload.py** | 278 | 292| 146 | 18832 | 77768 | 
| 65 | 39 django/core/servers/basehttp.py | 97 | 119| 211 | 19043 | 77768 | 
| 66 | 39 django/core/servers/basehttp.py | 53 | 73| 170 | 19213 | 77768 | 
| 67 | 39 django/core/management/commands/loaddata.py | 81 | 148| 593 | 19806 | 77768 | 
| 68 | 40 django/core/checks/security/csrf.py | 1 | 41| 299 | 20105 | 78067 | 
| 69 | 41 django/db/backends/signals.py | 1 | 4| 0 | 20105 | 78084 | 
| 70 | 41 django/core/servers/basehttp.py | 159 | 178| 170 | 20275 | 78084 | 
| 71 | 42 django/contrib/gis/geoip2/base.py | 1 | 21| 141 | 20416 | 80120 | 
| 72 | 43 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 20574 | 81306 | 
| 73 | 44 scripts/manage_translations.py | 1 | 29| 200 | 20774 | 82999 | 
| 74 | 44 django/conf/global_settings.py | 145 | 263| 876 | 21650 | 82999 | 
| 75 | 45 django/bin/django-admin.py | 1 | 6| 0 | 21650 | 83026 | 
| 76 | 45 django/contrib/postgres/apps.py | 1 | 17| 129 | 21779 | 83026 | 
| 77 | **45 django/utils/autoreload.py** | 332 | 371| 266 | 22045 | 83026 | 
| 78 | 46 django/__main__.py | 1 | 10| 0 | 22045 | 83071 | 
| 79 | 47 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 22194 | 85820 | 
| 80 | 48 django/contrib/postgres/forms/__init__.py | 1 | 5| 0 | 22194 | 85862 | 
| 81 | 49 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 22698 | 87128 | 
| 82 | 50 django/contrib/gis/db/backends/mysql/base.py | 1 | 17| 0 | 22698 | 87237 | 
| 83 | **50 django/utils/autoreload.py** | 48 | 76| 156 | 22854 | 87237 | 
| 84 | 51 django/core/checks/database.py | 1 | 12| 0 | 22854 | 87290 | 
| 85 | 52 django/contrib/auth/admin.py | 1 | 22| 188 | 23042 | 89016 | 
| 86 | 53 django/core/checks/messages.py | 53 | 76| 161 | 23203 | 89589 | 
| 87 | 54 django/contrib/gis/geos/error.py | 1 | 4| 0 | 23203 | 89613 | 
| 88 | 55 django/contrib/postgres/functions.py | 1 | 12| 0 | 23203 | 89666 | 
| 89 | 56 docs/conf.py | 1 | 95| 746 | 23949 | 92643 | 
| 90 | 56 django/core/management/base.py | 384 | 449| 614 | 24563 | 92643 | 
| 91 | 57 django/core/handlers/asgi.py | 1 | 21| 116 | 24679 | 95008 | 
| 92 | 58 django/db/models/sql/compiler.py | 1 | 20| 167 | 24846 | 108713 | 
| 93 | **58 django/utils/autoreload.py** | 482 | 503| 205 | 25051 | 108713 | 
| 94 | 58 django/db/backends/sqlite3/creation.py | 1 | 46| 356 | 25407 | 108713 | 
| 95 | 58 django/db/backends/base/base.py | 528 | 559| 227 | 25634 | 108713 | 
| 96 | 59 django/views/csrf.py | 15 | 100| 835 | 26469 | 110257 | 
| 97 | 59 django/db/backends/mysql/creation.py | 31 | 55| 254 | 26723 | 110257 | 
| 98 | 59 django/db/backends/mysql/creation.py | 1 | 29| 219 | 26942 | 110257 | 
| 99 | 60 django/contrib/postgres/serializers.py | 1 | 11| 0 | 26942 | 110358 | 
| 100 | 61 docs/_ext/djangodocs.py | 26 | 70| 385 | 27327 | 113431 | 
| 101 | 61 scripts/manage_translations.py | 176 | 186| 116 | 27443 | 113431 | 
| 102 | 62 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 27638 | 113626 | 
| 103 | 63 django/core/mail/backends/__init__.py | 1 | 2| 0 | 27638 | 113634 | 
| 104 | 64 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 27638 | 113654 | 
| 105 | 64 django/core/management/commands/sqlmigrate.py | 1 | 30| 266 | 27904 | 113654 | 
| 106 | 65 django/core/management/sql.py | 37 | 52| 116 | 28020 | 114039 | 
| 107 | **65 django/utils/autoreload.py** | 374 | 404| 349 | 28369 | 114039 | 
| 108 | 65 django/core/management/commands/compilemessages.py | 1 | 26| 157 | 28526 | 114039 | 
| 109 | 66 django/contrib/staticfiles/management/commands/collectstatic.py | 147 | 205| 503 | 29029 | 116883 | 
| **-> 110 <-** | **66 django/utils/autoreload.py** | 221 | 258| 272 | 29301 | 116883 | 
| 111 | 66 django/db/backends/postgresql/base.py | 203 | 232| 246 | 29547 | 116883 | 
| 112 | **66 django/utils/autoreload.py** | 423 | 435| 145 | 29692 | 116883 | 
| 113 | 66 django/core/management/commands/inspectdb.py | 38 | 174| 1298 | 30990 | 116883 | 
| 114 | 67 django/db/backends/dummy/features.py | 1 | 7| 0 | 30990 | 116915 | 
| 115 | 68 django/core/management/commands/makemessages.py | 283 | 362| 816 | 31806 | 122475 | 
| 116 | 68 django/contrib/staticfiles/management/commands/collectstatic.py | 37 | 68| 297 | 32103 | 122475 | 
| 117 | 69 django/http/request.py | 1 | 37| 251 | 32354 | 127305 | 
| 118 | 69 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 33142 | 127305 | 
| 119 | 70 django/db/backends/oracle/creation.py | 30 | 100| 722 | 33864 | 131200 | 
| 120 | 70 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 34148 | 131200 | 
| 121 | 70 django/core/management/base.py | 1 | 36| 223 | 34371 | 131200 | 
| 122 | 70 django/db/__init__.py | 40 | 62| 118 | 34489 | 131200 | 
| 123 | 70 django/db/backends/postgresql/base.py | 182 | 201| 192 | 34681 | 131200 | 
| 124 | 70 django/core/servers/basehttp.py | 76 | 95| 184 | 34865 | 131200 | 
| 125 | 70 django/core/management/commands/makemessages.py | 363 | 392| 231 | 35096 | 131200 | 
| 126 | 71 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 35096 | 131253 | 
| 127 | 72 django/core/files/storage.py | 1 | 22| 158 | 35254 | 134089 | 
| 128 | 73 django/contrib/gis/db/backends/spatialite/base.py | 1 | 36| 331 | 35585 | 134708 | 
| 129 | 74 django/db/backends/dummy/base.py | 1 | 47| 270 | 35855 | 135153 | 
| 130 | 74 django/db/backends/base/base.py | 561 | 608| 300 | 36155 | 135153 | 
| 131 | 75 django/core/wsgi.py | 1 | 14| 0 | 36155 | 135243 | 
| 132 | 76 django/views/debug.py | 72 | 95| 196 | 36351 | 139461 | 
| 133 | 76 django/core/checks/security/base.py | 193 | 211| 127 | 36478 | 139461 | 
| 134 | 76 django/db/backends/oracle/creation.py | 130 | 165| 399 | 36877 | 139461 | 
| 135 | 76 django/core/management/sql.py | 20 | 34| 116 | 36993 | 139461 | 
| 136 | 76 django/db/utils.py | 272 | 314| 322 | 37315 | 139461 | 
| 137 | 77 django/contrib/admin/exceptions.py | 1 | 12| 0 | 37315 | 139528 | 
| 138 | 78 django/core/checks/__init__.py | 1 | 25| 254 | 37569 | 139782 | 
| 139 | 78 django/core/management/commands/makemessages.py | 1 | 33| 247 | 37816 | 139782 | 
| 140 | 79 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 37937 | 140084 | 
| 141 | 80 django/core/asgi.py | 1 | 14| 0 | 37937 | 140169 | 
| 142 | 80 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 38229 | 140169 | 
| 143 | 81 django/core/checks/templates.py | 1 | 36| 259 | 38488 | 140429 | 
| 144 | 81 django/core/checks/security/base.py | 88 | 190| 747 | 39235 | 140429 | 
| 145 | 82 django/db/backends/base/creation.py | 199 | 216| 154 | 39389 | 142754 | 
| 146 | 83 django/contrib/gis/db/backends/spatialite/client.py | 1 | 6| 0 | 39389 | 142783 | 
| 147 | 83 django/conf/global_settings.py | 264 | 346| 800 | 40189 | 142783 | 
| 148 | 84 django/core/mail/backends/dummy.py | 1 | 11| 0 | 40189 | 142826 | 
| 149 | 85 django/dispatch/__init__.py | 1 | 10| 0 | 40189 | 142891 | 
| 150 | **85 django/utils/autoreload.py** | 540 | 564| 212 | 40401 | 142891 | 
| 151 | 86 django/contrib/gis/db/backends/spatialite/adapter.py | 1 | 10| 0 | 40401 | 142957 | 
| 152 | 87 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 40723 | 143279 | 
| 153 | 87 django/db/backends/postgresql/base.py | 234 | 264| 267 | 40990 | 143279 | 
| 154 | 88 django/contrib/sites/admin.py | 1 | 9| 0 | 40990 | 143325 | 
| 155 | 89 django/core/checks/urls.py | 1 | 27| 142 | 41132 | 144026 | 
| 156 | 90 django/core/management/utils.py | 1 | 28| 185 | 41317 | 145154 | 
| 157 | 91 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 41446 | 149070 | 
| 158 | 92 django/db/migrations/state.py | 1 | 24| 191 | 41637 | 154288 | 
| 159 | 92 django/utils/log.py | 162 | 196| 290 | 41927 | 154288 | 
| 160 | 93 django/contrib/gis/db/backends/postgis/features.py | 1 | 13| 0 | 41927 | 154378 | 
| 161 | 94 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 42061 | 155069 | 
| 162 | 95 django/db/migrations/__init__.py | 1 | 3| 0 | 42061 | 155093 | 
| 163 | 95 django/db/backends/postgresql/base.py | 66 | 142| 778 | 42839 | 155093 | 
| 164 | 95 django/core/management/commands/migrate.py | 259 | 291| 349 | 43188 | 155093 | 
| 165 | 96 django/core/handlers/wsgi.py | 64 | 119| 486 | 43674 | 156776 | 
| 166 | 97 django/views/__init__.py | 1 | 4| 0 | 43674 | 156791 | 
| 167 | 98 django/contrib/gis/__init__.py | 1 | 2| 0 | 43674 | 156805 | 
| 168 | 98 django/core/management/commands/compilemessages.py | 29 | 56| 231 | 43905 | 156805 | 
| 169 | 99 django/contrib/gis/geos/base.py | 1 | 7| 0 | 43905 | 156847 | 
| 170 | 100 django/contrib/auth/management/commands/createsuperuser.py | 64 | 178| 1076 | 44981 | 158679 | 
| 171 | 100 django/contrib/postgres/apps.py | 20 | 37| 188 | 45169 | 158679 | 
| 172 | 100 django/contrib/gis/db/backends/spatialite/base.py | 38 | 75| 295 | 45464 | 158679 | 
| 173 | 101 django/core/signals.py | 1 | 7| 0 | 45464 | 158734 | 
| 174 | 102 django/contrib/gis/gdal/base.py | 1 | 7| 0 | 45464 | 158776 | 
| 175 | 102 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 45766 | 158776 | 
| 176 | 103 django/core/files/__init__.py | 1 | 4| 0 | 45766 | 158791 | 
| 177 | 104 django/core/management/commands/startproject.py | 1 | 21| 137 | 45903 | 158928 | 
| 178 | 105 django/http/__init__.py | 1 | 22| 197 | 46100 | 159125 | 
| 179 | 106 django/db/models/options.py | 1 | 36| 304 | 46404 | 166144 | 
| 180 | 106 django/core/management/commands/loaddata.py | 217 | 273| 549 | 46953 | 166144 | 
| 181 | 107 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 47147 | 166338 | 
| 182 | 107 django/contrib/staticfiles/management/commands/collectstatic.py | 1 | 35| 215 | 47362 | 166338 | 
| 183 | 107 django/conf/global_settings.py | 398 | 497| 793 | 48155 | 166338 | 
| 184 | 108 django/contrib/admin/bin/compress.py | 1 | 64| 473 | 48628 | 166811 | 
| 185 | 108 django/core/management/commands/migrate.py | 243 | 257| 170 | 48798 | 166811 | 
| 186 | 108 django/db/backends/base/base.py | 204 | 281| 543 | 49341 | 166811 | 
| 187 | 108 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 49527 | 166811 | 
| 188 | 109 django/db/migrations/utils.py | 1 | 18| 0 | 49527 | 166899 | 
| 189 | **109 django/utils/autoreload.py** | 460 | 480| 268 | 49795 | 166899 | 
| 190 | 110 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 49795 | 166969 | 
| 191 | 110 django/views/csrf.py | 1 | 13| 132 | 49927 | 166969 | 
| 192 | 110 django/core/management/commands/flush.py | 1 | 25| 206 | 50133 | 166969 | 
| 193 | 111 django/apps/__init__.py | 1 | 5| 0 | 50133 | 166992 | 
| 194 | 111 django/contrib/staticfiles/management/commands/collectstatic.py | 294 | 328| 320 | 50453 | 166992 | 
| 195 | 112 django/contrib/gis/apps.py | 1 | 12| 0 | 50453 | 167063 | 
| 196 | 113 django/contrib/auth/urls.py | 1 | 21| 225 | 50678 | 167288 | 
| 197 | 114 django/db/backends/sqlite3/base.py | 247 | 292| 413 | 51091 | 172950 | 
| 198 | 115 django/db/backends/oracle/client.py | 1 | 18| 0 | 51091 | 173052 | 
| 199 | 116 django/db/migrations/autodetector.py | 1 | 15| 110 | 51201 | 184723 | 
| 200 | 116 django/db/backends/sqlite3/base.py | 167 | 193| 270 | 51471 | 184723 | 
| 201 | 116 django/db/backends/oracle/creation.py | 253 | 281| 277 | 51748 | 184723 | 
| 202 | 117 django/contrib/gis/management/commands/ogrinspect.py | 98 | 134| 411 | 52159 | 185937 | 
| 203 | 118 django/db/backends/mysql/base.py | 1 | 50| 447 | 52606 | 188967 | 
| 204 | 119 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 52811 | 189420 | 
| 205 | 120 django/contrib/gis/db/backends/postgis/operations.py | 1 | 25| 217 | 53028 | 193027 | 
| 206 | 121 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 53028 | 193050 | 
| 207 | 121 django/db/utils.py | 52 | 98| 312 | 53340 | 193050 | 
| 208 | 122 django/db/backends/utils.py | 1 | 47| 287 | 53627 | 194947 | 
| 209 | 123 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 53627 | 195040 | 


## Missing Patch Files

 * 1: django/utils/autoreload.py
 * 2: django/utils/translation/reloader.py

### Hint

```
First glance, this looks like some Docker weirdness: File "/usr/local/lib/python3.6/site-packages/django/utils/translation/reloader.py", line 16, in watch_for_translation_changes absolute_path = path.absolute() File "/usr/local/lib/python3.6/pathlib.py", line 1129, in absolute obj = self._from_parts([os.getcwd()] + self._parts, init=False) FileNotFoundError: [Errno 2] No such file or directory That's a standard library call raising the error, so why's that not working? @steinbachr it would be helpful if you could put a breakpoint in there and try to work out exactly what's going on. (That call should work. Why isn't it? Is there an obvious something that would?) Why the regression between 2.1 and 2.2? We were using os.path previously I guess... Still, this should be something that works, so a bit more digging is needed.
Is this while using Docker for Mac? Could it be related to this Docker issue, as it seems it’s being thrown in the cwd call: ​https://github.com/docker/for-mac/issues/1509 Can you confirm if this happens intermittently or happens all the time, and provide some more information on your machine (operating system, Docker version, filesystem type).
I'm thinking to close this as needsinfo for the moment. I'm very happy to re-open it if we can show that Django is at fault. Or if we want to provide a workaround in any case... (— Tom, happy to follow your lead there.) @steinbachr please do add the extra information. I do mean the "very happy to re-open" :) Thanks.
@Tom Forbes on my machine it is happening all the time. I can change my requirements to downgrade to Django==2.1.4 and rebuild the image, resulting in a working container. Then, I can upgrade to Django==2.2.3 and rebuild, resulting in a broken container, consistently. Some system information: Mac OS Version: 10.13.6 (High Sierra) Docker version 18.09.2, build 6247962 docker-compose version 1.23.2, build 1110ad01 Let me know if there's any additional info I can provide to help
```

## Patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -240,8 +240,15 @@ def __init__(self):
 
     def watch_dir(self, path, glob):
         path = Path(path)
-        if not path.is_absolute():
-            raise ValueError('%s must be absolute.' % path)
+        try:
+            path = path.absolute()
+        except FileNotFoundError:
+            logger.debug(
+                'Unable to watch directory %s as it cannot be resolved.',
+                path,
+                exc_info=True,
+            )
+            return
         logger.debug('Watching dir %s with glob %s.', path, glob)
         self.directory_globs[path].add(glob)
 
diff --git a/django/utils/translation/reloader.py b/django/utils/translation/reloader.py
--- a/django/utils/translation/reloader.py
+++ b/django/utils/translation/reloader.py
@@ -14,8 +14,7 @@ def watch_for_translation_changes(sender, **kwargs):
         directories.extend(Path(config.path) / 'locale' for config in apps.get_app_configs())
         directories.extend(Path(p) for p in settings.LOCALE_PATHS)
         for path in directories:
-            absolute_path = path.absolute()
-            sender.watch_dir(absolute_path, '**/*.mo')
+            sender.watch_dir(path, '**/*.mo')
 
 
 def translation_file_changed(sender, file_path, **kwargs):

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -499,6 +499,12 @@ def test_overlapping_glob_recursive(self, mocked_modules, notify_mock):
 class BaseReloaderTests(ReloaderTests):
     RELOADER_CLS = autoreload.BaseReloader
 
+    def test_watch_dir_with_unresolvable_path(self):
+        path = Path('unresolvable_directory')
+        with mock.patch.object(Path, 'absolute', side_effect=FileNotFoundError):
+            self.reloader.watch_dir(path, '**/*.mo')
+        self.assertEqual(list(self.reloader.directory_globs), [])
+
     def test_watch_with_glob(self):
         self.reloader.watch_dir(self.tempdir, '*.py')
         watched_files = list(self.reloader.watched_files())

```


## Code snippets

### 1 - django/core/management/commands/runserver.py:

Start line: 106, End line: 162

```python
class Command(BaseCommand):

    def inner_run(self, *args, **options):
        # If an exception was silenced in ManagementUtility.execute in order
        # to be raised in the child process, raise it now.
        autoreload.raise_last_exception()

        threading = options['use_threading']
        # 'shutdown_message' is a stealth option.
        shutdown_message = options.get('shutdown_message', '')
        quit_command = 'CTRL-BREAK' if sys.platform == 'win32' else 'CONTROL-C'

        self.stdout.write("Performing system checks...\n\n")
        self.check(display_num_errors=True)
        # Need to check migrations here, so can't use the
        # requires_migrations_check attribute.
        self.check_migrations()
        now = datetime.now().strftime('%B %d, %Y - %X')
        self.stdout.write(now)
        self.stdout.write((
            "Django version %(version)s, using settings %(settings)r\n"
            "Starting development server at %(protocol)s://%(addr)s:%(port)s/\n"
            "Quit the server with %(quit_command)s.\n"
        ) % {
            "version": self.get_version(),
            "settings": settings.SETTINGS_MODULE,
            "protocol": self.protocol,
            "addr": '[%s]' % self.addr if self._raw_ipv6 else self.addr,
            "port": self.port,
            "quit_command": quit_command,
        })

        try:
            handler = self.get_handler(*args, **options)
            run(self.addr, int(self.port), handler,
                ipv6=self.use_ipv6, threading=threading, server_cls=self.server_cls)
        except OSError as e:
            # Use helpful error messages instead of ugly tracebacks.
            ERRORS = {
                errno.EACCES: "You don't have permission to access that port.",
                errno.EADDRINUSE: "That port is already in use.",
                errno.EADDRNOTAVAIL: "That IP address can't be assigned to.",
            }
            try:
                error_text = ERRORS[e.errno]
            except KeyError:
                error_text = e
            self.stderr.write("Error: %s" % error_text)
            # Need to use an OS exit because sys.exit doesn't work in a thread
            os._exit(1)
        except KeyboardInterrupt:
            if shutdown_message:
                self.stdout.write(shutdown_message)
            sys.exit(0)


# Kept for backward compatibility
BaseRunserverCommand = Command
```
### 2 - django/core/management/commands/runserver.py:

Start line: 66, End line: 104

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        if not settings.DEBUG and not settings.ALLOWED_HOSTS:
            raise CommandError('You must set settings.ALLOWED_HOSTS if DEBUG is False.')

        self.use_ipv6 = options['use_ipv6']
        if self.use_ipv6 and not socket.has_ipv6:
            raise CommandError('Your Python does not support IPv6.')
        self._raw_ipv6 = False
        if not options['addrport']:
            self.addr = ''
            self.port = self.default_port
        else:
            m = re.match(naiveip_re, options['addrport'])
            if m is None:
                raise CommandError('"%s" is not a valid port number '
                                   'or address:port pair.' % options['addrport'])
            self.addr, _ipv4, _ipv6, _fqdn, self.port = m.groups()
            if not self.port.isdigit():
                raise CommandError("%r is not a valid port number." % self.port)
            if self.addr:
                if _ipv6:
                    self.addr = self.addr[1:-1]
                    self.use_ipv6 = True
                    self._raw_ipv6 = True
                elif self.use_ipv6 and not _fqdn:
                    raise CommandError('"%s" is not a valid IPv6 address.' % self.addr)
        if not self.addr:
            self.addr = self.default_addr_ipv6 if self.use_ipv6 else self.default_addr
            self._raw_ipv6 = self.use_ipv6
        self.run(**options)

    def run(self, **options):
        """Run the server, using the autoreloader if needed."""
        use_reloader = options['use_reloader']

        if use_reloader:
            autoreload.run_with_reloader(self.inner_run, **options)
        else:
            self.inner_run(None, **options)
```
### 3 - django/core/management/commands/runserver.py:

Start line: 23, End line: 52

```python
class Command(BaseCommand):
    help = "Starts a lightweight Web server for development."

    # Validation is called explicitly each time the server is reloaded.
    requires_system_checks = False
    stealth_options = ('shutdown_message',)

    default_addr = '127.0.0.1'
    default_addr_ipv6 = '::1'
    default_port = '8000'
    protocol = 'http'
    server_cls = WSGIServer

    def add_arguments(self, parser):
        parser.add_argument(
            'addrport', nargs='?',
            help='Optional port number, or ipaddr:port'
        )
        parser.add_argument(
            '--ipv6', '-6', action='store_true', dest='use_ipv6',
            help='Tells Django to use an IPv6 address.',
        )
        parser.add_argument(
            '--nothreading', action='store_false', dest='use_threading',
            help='Tells Django to NOT use threading.',
        )
        parser.add_argument(
            '--noreload', action='store_false', dest='use_reloader',
            help='Tells Django to NOT use the auto-reloader.',
        )
```
### 4 - django/core/management/commands/runserver.py:

Start line: 1, End line: 20

```python
import errno
import os
import re
import socket
import sys
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.servers.basehttp import (
    WSGIServer, get_internal_wsgi_application, run,
)
from django.utils import autoreload

naiveip_re = re.compile(r"""^(?:
(?P<addr>
    (?P<ipv4>\d{1,3}(?:\.\d{1,3}){3}) |         # IPv4 address
    (?P<ipv6>\[[a-fA-F0-9:]+\]) |               # IPv6 address
    (?P<fqdn>[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*) # FQDN
):)?(?P<port>\d+)$""", re.X)
```
### 5 - django/core/management/commands/testserver.py:

Start line: 29, End line: 55

```python
class Command(BaseCommand):

    def handle(self, *fixture_labels, **options):
        verbosity = options['verbosity']
        interactive = options['interactive']

        # Create a test database.
        db_name = connection.creation.create_test_db(verbosity=verbosity, autoclobber=not interactive, serialize=False)

        # Import the fixture data into the test database.
        call_command('loaddata', *fixture_labels, **{'verbosity': verbosity})

        # Run the development server. Turn off auto-reloading because it causes
        # a strange error -- it causes this handle() method to be called
        # multiple times.
        shutdown_message = (
            '\nServer stopped.\nNote that the test database, %r, has not been '
            'deleted. You can explore it on your own.' % db_name
        )
        use_threading = connection.features.test_db_allows_multiple_connections
        call_command(
            'runserver',
            addrport=options['addrport'],
            shutdown_message=shutdown_message,
            use_reloader=False,
            use_ipv6=options['use_ipv6'],
            use_threading=use_threading
        )
```
### 6 - django/core/management/commands/testserver.py:

Start line: 1, End line: 27

```python
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Runs a development server with data from the given fixture(s).'

    requires_system_checks = False

    def add_arguments(self, parser):
        parser.add_argument(
            'args', metavar='fixture', nargs='*',
            help='Path(s) to fixtures to load before running the server.',
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--addrport', default='',
            help='Port number or ipaddr:port to run the server on.',
        )
        parser.add_argument(
            '--ipv6', '-6', action='store_true', dest='use_ipv6',
            help='Tells Django to use an IPv6 address.',
        )
```
### 7 - django/core/servers/basehttp.py:

Start line: 200, End line: 217

```python
def run(addr, port, wsgi_handler, ipv6=False, threading=False, server_cls=WSGIServer):
    server_address = (addr, port)
    if threading:
        httpd_cls = type('WSGIServer', (socketserver.ThreadingMixIn, server_cls), {})
    else:
        httpd_cls = server_cls
    httpd = httpd_cls(server_address, WSGIRequestHandler, ipv6=ipv6)
    if threading:
        # ThreadingMixIn.daemon_threads indicates how threads will behave on an
        # abrupt shutdown; like quitting the server by the user or restarting
        # by the auto-reloader. True means the server will not wait for thread
        # termination before it quits. This will make auto-reloader faster
        # and will prevent the need to kill the server manually if a thread
        # isn't terminating correctly.
        httpd.daemon_threads = True
    httpd.set_app(wsgi_handler)
    httpd.serve_forever()
```
### 8 - django/contrib/staticfiles/management/commands/runserver.py:

Start line: 1, End line: 33

```python
from django.conf import settings
from django.contrib.staticfiles.handlers import StaticFilesHandler
from django.core.management.commands.runserver import (
    Command as RunserverCommand,
)


class Command(RunserverCommand):
    help = "Starts a lightweight Web server for development and also serves static files."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            '--nostatic', action="store_false", dest='use_static_handler',
            help='Tells Django to NOT automatically serve static files at STATIC_URL.',
        )
        parser.add_argument(
            '--insecure', action="store_true", dest='insecure_serving',
            help='Allows serving static files even if DEBUG is False.',
        )

    def get_handler(self, *args, **options):
        """
        Return the static files serving handler wrapping the default handler,
        if static files should be served. Otherwise return the default handler.
        """
        handler = super().get_handler(*args, **options)
        use_static_handler = options['use_static_handler']
        insecure_serving = options['insecure_serving']
        if use_static_handler and (settings.DEBUG or insecure_serving):
            return StaticFilesHandler(handler)
        return handler
```
### 9 - django/db/backends/postgresql/client.py:

Start line: 1, End line: 55

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
        sslmode = conn_params.get('sslmode', '')
        sslrootcert = conn_params.get('sslrootcert', '')
        sslcert = conn_params.get('sslcert', '')
        sslkey = conn_params.get('sslkey', '')

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
        if sslmode:
            subprocess_env['PGSSLMODE'] = str(sslmode)
        if sslrootcert:
            subprocess_env['PGSSLROOTCERT'] = str(sslrootcert)
        if sslcert:
            subprocess_env['PGSSLCERT'] = str(sslcert)
        if sslkey:
            subprocess_env['PGSSLKEY'] = str(sslkey)
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
### 10 - django/db/backends/postgresql/base.py:

Start line: 1, End line: 63

```python
"""
PostgreSQL database backend for Django.

Requires psycopg 2: http://initd.org/projects/psycopg2
"""

import asyncio
import threading
import warnings

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import connections
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.utils import (
    CursorDebugWrapper as BaseCursorDebugWrapper,
)
from django.db.utils import DatabaseError as WrappedDatabaseError
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
### 14 - django/utils/autoreload.py:

Start line: 567, End line: 583

```python
def start_django(reloader, main_func, *args, **kwargs):
    ensure_echo_on()

    main_func = check_errors(main_func)
    django_main_thread = threading.Thread(target=main_func, args=args, kwargs=kwargs, name='django-main-thread')
    django_main_thread.setDaemon(True)
    django_main_thread.start()

    while not reloader.should_stop:
        try:
            reloader.run(django_main_thread)
        except WatchmanUnavailable as ex:
            # It's possible that the watchman service shuts down or otherwise
            # becomes unavailable. In that case, use the StatReloader.
            reloader = StatReloader()
            logger.error('Error connecting to Watchman: %s', ex)
            logger.info('Watching for file changes with %s', reloader.__class__.__name__)
```
### 15 - django/utils/autoreload.py:

Start line: 586, End line: 598

```python
def run_with_reloader(main_func, *args, **kwargs):
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    try:
        if os.environ.get(DJANGO_AUTORELOAD_ENV) == 'true':
            reloader = get_reloader()
            logger.info('Watching for file changes with %s', reloader.__class__.__name__)
            start_django(reloader, main_func, *args, **kwargs)
        else:
            exit_code = restart_with_reloader()
            sys.exit(exit_code)
    except KeyboardInterrupt:
        pass
```
### 50 - django/utils/autoreload.py:

Start line: 1, End line: 45

```python
import functools
import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter

from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple

autoreload_started = Signal()
file_changed = Signal(providing_args=['file_path', 'kind'])

DJANGO_AUTORELOAD_ENV = 'RUN_MAIN'

logger = logging.getLogger('django.utils.autoreload')

# If an error is raised while importing a file, it's not placed in sys.modules.
# This means that any future modifications aren't caught. Keep a list of these
# file paths to allow watching them in the future.
_error_files = []
_exception = None

try:
    import termios
except ImportError:
    termios = None


try:
    import pywatchman
except ImportError:
    pywatchman = None
```
### 62 - django/utils/autoreload.py:

Start line: 505, End line: 538

```python
class WatchmanReloader(BaseReloader):

    def request_processed(self, **kwargs):
        logger.debug('Request processed. Setting update_watches event.')
        self.processed_request.set()

    def tick(self):
        request_finished.connect(self.request_processed)
        self.update_watches()
        while True:
            if self.processed_request.is_set():
                self.update_watches()
                self.processed_request.clear()
            try:
                self.client.receive()
            except pywatchman.SocketTimeout:
                pass
            except pywatchman.WatchmanError as ex:
                logger.debug('Watchman error: %s, checking server status.', ex)
                self.check_server_status(ex)
            else:
                for sub in list(self.client.subs.keys()):
                    self._check_subscription(sub)
            yield

    def stop(self):
        self.client.close()
        super().stop()

    def check_server_status(self, inner_ex=None):
        """Return True if the server is available."""
        try:
            self.client.query('version')
        except Exception:
            raise WatchmanUnavailable(str(inner_ex)) from inner_ex
        return True
```
### 64 - django/utils/autoreload.py:

Start line: 278, End line: 292

```python
class BaseReloader:

    def run(self, django_main_thread):
        logger.debug('Waiting for apps ready_event.')
        self.wait_for_apps_ready(apps, django_main_thread)
        from django.urls import get_resolver
        # Prevent a race condition where URL modules aren't loaded when the
        # reloader starts by accessing the urlconf_module property.
        try:
            get_resolver().urlconf_module
        except Exception:
            # Loading the urlconf can result in errors during development.
            # If this occurs then swallow the error and continue.
            pass
        logger.debug('Apps ready_event triggered. Sending autoreload_started signal.')
        autoreload_started.send(sender=self)
        self.run_loop()
```
### 77 - django/utils/autoreload.py:

Start line: 332, End line: 371

```python
class StatReloader(BaseReloader):
    SLEEP_TIME = 1  # Check for changes once per second.

    def tick(self):
        mtimes = {}
        while True:
            for filepath, mtime in self.snapshot_files():
                old_time = mtimes.get(filepath)
                mtimes[filepath] = mtime
                if old_time is None:
                    logger.debug('File %s first seen with mtime %s', filepath, mtime)
                    continue
                elif mtime > old_time:
                    logger.debug('File %s previous mtime: %s, current mtime: %s', filepath, old_time, mtime)
                    self.notify_file_changed(filepath)

            time.sleep(self.SLEEP_TIME)
            yield

    def snapshot_files(self):
        # watched_files may produce duplicate paths if globs overlap.
        seen_files = set()
        for file in self.watched_files():
            if file in seen_files:
                continue
            try:
                mtime = file.stat().st_mtime
            except OSError:
                # This is thrown when the file does not exist.
                continue
            seen_files.add(file)
            yield file, mtime

    @classmethod
    def check_availability(cls):
        return True


class WatchmanUnavailable(RuntimeError):
    pass
```
### 83 - django/utils/autoreload.py:

Start line: 48, End line: 76

```python
def check_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _exception
        try:
            fn(*args, **kwargs)
        except Exception:
            _exception = sys.exc_info()

            et, ev, tb = _exception

            if getattr(ev, 'filename', None) is None:
                # get the filename from the last item in the stack
                filename = traceback.extract_tb(tb)[-1][0]
            else:
                filename = ev.filename

            if filename not in _error_files:
                _error_files.append(filename)

            raise

    return wrapper


def raise_last_exception():
    global _exception
    if _exception is not None:
        raise _exception[1]
```
### 93 - django/utils/autoreload.py:

Start line: 482, End line: 503

```python
class WatchmanReloader(BaseReloader):

    def update_watches(self):
        try:
            self._update_watches()
        except Exception as ex:
            # If the service is still available, raise the original exception.
            if self.check_server_status(ex):
                raise

    def _check_subscription(self, sub):
        subscription = self.client.getSubscription(sub)
        if not subscription:
            return
        logger.debug('Watchman subscription %s has results.', sub)
        for result in subscription:
            # When using watch-project, it's not simple to get the relative
            # directory without storing some specific state. Store the full
            # path to the directory in the subscription name, prefixed by its
            # type (glob, files).
            root_directory = Path(result['subscription'].split(':', 1)[1])
            logger.debug('Found root directory %s', root_directory)
            for file in result.get('files', []):
                self.notify_file_changed(root_directory / file)
```
### 107 - django/utils/autoreload.py:

Start line: 374, End line: 404

```python
class WatchmanReloader(BaseReloader):
    def __init__(self):
        self.roots = defaultdict(set)
        self.processed_request = threading.Event()
        self.client_timeout = int(os.environ.get('DJANGO_WATCHMAN_TIMEOUT', 5))
        super().__init__()

    @cached_property
    def client(self):
        return pywatchman.client(timeout=self.client_timeout)

    def _watch_root(self, root):
        # In practice this shouldn't occur, however, it's possible that a
        # directory that doesn't exist yet is being watched. If it's outside of
        # sys.path then this will end up a new root. How to handle this isn't
        # clear: Not adding the root will likely break when subscribing to the
        # changes, however, as this is currently an internal API,  no files
        # will be being watched outside of sys.path. Fixing this by checking
        # inside watch_glob() and watch_dir() is expensive, instead this could
        # could fall back to the StatReloader if this case is detected? For
        # now, watching its parent, if possible, is sufficient.
        if not root.exists():
            if not root.parent.exists():
                logger.warning('Unable to watch root dir %s as neither it or its parent exist.', root)
                return
            root = root.parent
        result = self.client.query('watch-project', str(root.absolute()))
        if 'warning' in result:
            logger.warning('Watchman warning: %s', result['warning'])
        logger.debug('Watchman watch-project result: %s', result)
        return result['watch'], result.get('relative_path')
```
### 110 - django/utils/autoreload.py:

Start line: 221, End line: 258

```python
def trigger_reload(filename):
    logger.info('%s changed, reloading.', filename)
    sys.exit(3)


def restart_with_reloader():
    new_environ = {**os.environ, DJANGO_AUTORELOAD_ENV: 'true'}
    args = get_child_arguments()
    while True:
        exit_code = subprocess.call(args, env=new_environ, close_fds=False)
        if exit_code != 3:
            return exit_code


class BaseReloader:
    def __init__(self):
        self.extra_files = set()
        self.directory_globs = defaultdict(set)
        self._stop_condition = threading.Event()

    def watch_dir(self, path, glob):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('%s must be absolute.' % path)
        logger.debug('Watching dir %s with glob %s.', path, glob)
        self.directory_globs[path].add(glob)

    def watched_files(self, include_globs=True):
        """
        Yield all files that need to be watched, including module files and
        files within globs.
        """
        yield from iter_all_python_module_files()
        yield from self.extra_files
        if include_globs:
            for directory, patterns in self.directory_globs.items():
                for pattern in patterns:
                    yield from directory.glob(pattern)
```
### 112 - django/utils/autoreload.py:

Start line: 423, End line: 435

```python
class WatchmanReloader(BaseReloader):

    def _subscribe_dir(self, directory, filenames):
        if not directory.exists():
            if not directory.parent.exists():
                logger.warning('Unable to watch directory %s as neither it or its parent exist.', directory)
                return
            prefix = 'files-parent-%s' % directory.name
            filenames = ['%s/%s' % (directory.name, filename) for filename in filenames]
            directory = directory.parent
            expression = ['name', filenames, 'wholename']
        else:
            prefix = 'files'
            expression = ['name', filenames]
        self._subscribe(directory, '%s:%s' % (prefix, directory), expression)
```
### 150 - django/utils/autoreload.py:

Start line: 540, End line: 564

```python
class WatchmanReloader(BaseReloader):

    @classmethod
    def check_availability(cls):
        if not pywatchman:
            raise WatchmanUnavailable('pywatchman not installed.')
        client = pywatchman.client(timeout=0.1)
        try:
            result = client.capabilityCheck()
        except Exception:
            # The service is down?
            raise WatchmanUnavailable('Cannot connect to the watchman service.')
        version = get_version_tuple(result['version'])
        # Watchman 4.9 includes multiple improvements to watching project
        # directories as well as case insensitive filesystems.
        logger.debug('Watchman version %s', version)
        if version < (4, 9):
            raise WatchmanUnavailable('Watchman 4.9 or later is required.')


def get_reloader():
    """Return the most suitable reloader for this environment."""
    try:
        WatchmanReloader.check_availability()
    except WatchmanUnavailable:
        return StatReloader()
    return WatchmanReloader()
```
### 189 - django/utils/autoreload.py:

Start line: 460, End line: 480

```python
class WatchmanReloader(BaseReloader):

    def watched_roots(self, watched_files):
        extra_directories = self.directory_globs.keys()
        watched_file_dirs = [f.parent for f in watched_files]
        sys_paths = list(sys_path_directories())
        return frozenset((*extra_directories, *watched_file_dirs, *sys_paths))

    def _update_watches(self):
        watched_files = list(self.watched_files(include_globs=False))
        found_roots = common_roots(self.watched_roots(watched_files))
        logger.debug('Watching %s files', len(watched_files))
        logger.debug('Found common roots: %s', found_roots)
        # Setup initial roots for performance, shortest roots first.
        for root in sorted(found_roots):
            self._watch_root(root)
        for directory, patterns in self.directory_globs.items():
            self._watch_glob(directory, patterns)
        # Group sorted watched_files by their parent directory.
        sorted_files = sorted(watched_files, key=lambda p: p.parent)
        for directory, group in itertools.groupby(sorted_files, key=lambda p: p.parent):
            # These paths need to be relative to the parent directory.
            self._subscribe_dir(directory, [str(p.relative_to(directory)) for p in group])
```
