# django__django-13660

| **django/django** | `50c3ac6fa9b7c8a94a6d1dc87edf775e3bc4d575` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 563 |
| **Any found context length** | 563 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/shell.py b/django/core/management/commands/shell.py
--- a/django/core/management/commands/shell.py
+++ b/django/core/management/commands/shell.py
@@ -84,13 +84,13 @@ def python(self, options):
     def handle(self, **options):
         # Execute the command and exit.
         if options['command']:
-            exec(options['command'])
+            exec(options['command'], globals())
             return
 
         # Execute stdin if it has anything to read and exit.
         # Not supported on Windows due to select.select() limitations.
         if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
-            exec(sys.stdin.read())
+            exec(sys.stdin.read(), globals())
             return
 
         available_shells = [options['interface']] if options['interface'] else self.shells

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/shell.py | 87 | 93 | 2 | 1 | 563


## Problem Statement

```
shell command crashes when passing (with -c) the python code with functions.
Description
	
The examples below use Python 3.7 and Django 2.2.16, but I checked that the code is the same on master and works the same in Python 3.8.
Here's how ​python -c works:
$ python -c <<EOF " 
import django
def f():
		print(django.__version__)
f()"
EOF
2.2.16
Here's how ​python -m django shell -c works (paths shortened for clarify):
$ python -m django shell -c <<EOF "
import django
def f():
		print(django.__version__)
f()"
EOF
Traceback (most recent call last):
 File "{sys.base_prefix}/lib/python3.7/runpy.py", line 193, in _run_module_as_main
	"__main__", mod_spec)
 File "{sys.base_prefix}/lib/python3.7/runpy.py", line 85, in _run_code
	exec(code, run_globals)
 File "{sys.prefix}/lib/python3.7/site-packages/django/__main__.py", line 9, in <module>
	management.execute_from_command_line()
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/__init__.py", line 381, in execute_from_command_line
	utility.execute()
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/__init__.py", line 375, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/base.py", line 323, in run_from_argv
	self.execute(*args, **cmd_options)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/base.py", line 364, in execute
	output = self.handle(*args, **options)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/commands/shell.py", line 86, in handle
	exec(options['command'])
 File "<string>", line 5, in <module>
 File "<string>", line 4, in f
NameError: name 'django' is not defined
The problem is in the ​usage of ​exec:
	def handle(self, **options):
		# Execute the command and exit.
		if options['command']:
			exec(options['command'])
			return
		# Execute stdin if it has anything to read and exit.
		# Not supported on Windows due to select.select() limitations.
		if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
			exec(sys.stdin.read())
			return
exec should be passed a dictionary containing a minimal set of globals. This can be done by just passing a new, empty dictionary as the second argument of exec.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/core/management/commands/shell.py** | 42 | 82| 401 | 401 | 820 | 
| **-> 2 <-** | **1 django/core/management/commands/shell.py** | 84 | 104| 162 | 563 | 820 | 
| 3 | **1 django/core/management/commands/shell.py** | 1 | 40| 267 | 830 | 820 | 
| 4 | 2 django/core/management/commands/dbshell.py | 23 | 44| 175 | 1005 | 1129 | 
| 5 | 3 django/core/management/__init__.py | 78 | 181| 847 | 1852 | 4637 | 
| 6 | 4 django/core/management/commands/runserver.py | 107 | 159| 502 | 2354 | 6084 | 
| 7 | 5 django/core/management/base.py | 373 | 408| 296 | 2650 | 10721 | 
| 8 | 5 django/core/management/__init__.py | 334 | 420| 755 | 3405 | 10721 | 
| 9 | 5 django/core/management/commands/dbshell.py | 1 | 21| 139 | 3544 | 10721 | 
| 10 | 5 django/core/management/base.py | 21 | 42| 165 | 3709 | 10721 | 
| 11 | 6 django/core/management/commands/migrate.py | 71 | 167| 834 | 4543 | 13977 | 
| 12 | 6 django/core/management/commands/runserver.py | 67 | 105| 397 | 4940 | 13977 | 
| 13 | 7 django/utils/autoreload.py | 214 | 243| 298 | 5238 | 18999 | 
| 14 | 8 django/core/management/commands/testserver.py | 29 | 55| 234 | 5472 | 19432 | 
| 15 | 9 django/core/management/commands/flush.py | 27 | 83| 486 | 5958 | 20119 | 
| 16 | 9 django/core/management/base.py | 239 | 273| 313 | 6271 | 20119 | 
| 17 | 10 django/core/management/commands/dumpdata.py | 67 | 139| 624 | 6895 | 21729 | 
| 18 | 11 django/core/management/commands/check.py | 40 | 71| 221 | 7116 | 22201 | 
| 19 | 11 django/core/management/commands/dumpdata.py | 179 | 203| 224 | 7340 | 22201 | 
| 20 | 11 django/core/management/__init__.py | 260 | 332| 721 | 8061 | 22201 | 
| 21 | 12 django/contrib/admin/sites.py | 515 | 548| 281 | 8342 | 26399 | 
| 22 | 13 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 8721 | 27032 | 
| 23 | 14 django/bin/django-admin.py | 1 | 22| 138 | 8859 | 27170 | 
| 24 | 14 django/core/management/base.py | 410 | 477| 622 | 9481 | 27170 | 
| 25 | 15 django/core/management/commands/test.py | 25 | 59| 296 | 9777 | 27621 | 
| 26 | 16 django/core/management/commands/compilemessages.py | 59 | 116| 504 | 10281 | 28970 | 
| 27 | 16 django/core/management/commands/migrate.py | 169 | 251| 808 | 11089 | 28970 | 
| 28 | 17 django/core/management/utils.py | 1 | 27| 173 | 11262 | 30084 | 
| 29 | 17 django/core/management/commands/runserver.py | 1 | 21| 204 | 11466 | 30084 | 
| 30 | 18 django/contrib/auth/management/commands/createsuperuser.py | 81 | 202| 1158 | 12624 | 32147 | 
| 31 | 19 django/core/management/templates.py | 120 | 183| 560 | 13184 | 34822 | 
| 32 | 20 django/core/management/commands/makemessages.py | 283 | 362| 814 | 13998 | 40371 | 
| 33 | 20 django/core/management/templates.py | 58 | 118| 526 | 14524 | 40371 | 
| 34 | 21 django/db/backends/postgresql/client.py | 1 | 55| 387 | 14911 | 40758 | 
| 35 | 21 django/core/management/commands/testserver.py | 1 | 27| 204 | 15115 | 40758 | 
| 36 | 21 django/core/management/commands/runserver.py | 24 | 53| 239 | 15354 | 40758 | 
| 37 | 21 django/core/management/commands/check.py | 1 | 38| 256 | 15610 | 40758 | 
| 38 | 22 django/contrib/staticfiles/management/commands/collectstatic.py | 148 | 205| 497 | 16107 | 43606 | 
| 39 | 23 scripts/manage_translations.py | 1 | 29| 197 | 16304 | 45262 | 
| 40 | 24 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 16425 | 45564 | 
| 41 | 24 django/core/management/commands/makemessages.py | 363 | 392| 230 | 16655 | 45564 | 
| 42 | 24 django/utils/autoreload.py | 628 | 640| 117 | 16772 | 45564 | 
| 43 | 25 django/contrib/gis/db/models/proxy.py | 1 | 47| 363 | 17135 | 46233 | 
| 44 | 26 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 17926 | 48106 | 
| 45 | 27 docs/conf.py | 1 | 101| 799 | 18725 | 51146 | 
| 46 | 27 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 19075 | 51146 | 
| 47 | 28 django/contrib/sessions/management/commands/clearsessions.py | 1 | 22| 124 | 19199 | 51270 | 
| 48 | 28 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 19853 | 51270 | 
| 49 | 28 django/core/management/commands/makemessages.py | 1 | 34| 260 | 20113 | 51270 | 
| 50 | 28 django/core/management/commands/migrate.py | 21 | 69| 407 | 20520 | 51270 | 
| 51 | 29 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 21342 | 54109 | 
| 52 | 29 django/core/management/base.py | 70 | 94| 161 | 21503 | 54109 | 
| 53 | 30 django/core/management/commands/loaddata.py | 1 | 35| 177 | 21680 | 57046 | 
| 54 | 31 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 21814 | 57736 | 
| 55 | 32 django/db/backends/base/client.py | 1 | 27| 192 | 22006 | 57928 | 
| 56 | 32 django/core/management/commands/compilemessages.py | 1 | 27| 161 | 22167 | 57928 | 
| 57 | 32 django/contrib/auth/management/commands/createsuperuser.py | 1 | 79| 577 | 22744 | 57928 | 
| 58 | 32 django/core/management/commands/diffsettings.py | 1 | 39| 298 | 23042 | 57928 | 
| 59 | 33 django/core/management/commands/inspectdb.py | 1 | 36| 266 | 23308 | 60561 | 
| 60 | 34 django/db/utils.py | 1 | 49| 154 | 23462 | 62707 | 
| 61 | 34 django/core/management/utils.py | 52 | 74| 204 | 23666 | 62707 | 
| 62 | 34 django/core/management/base.py | 1 | 18| 115 | 23781 | 62707 | 
| 63 | 34 django/core/management/commands/migrate.py | 1 | 18| 140 | 23921 | 62707 | 
| 64 | 34 django/core/management/commands/loaddata.py | 69 | 85| 187 | 24108 | 62707 | 
| 65 | 34 django/core/management/base.py | 323 | 371| 376 | 24484 | 62707 | 
| 66 | 34 django/core/management/commands/test.py | 1 | 23| 160 | 24644 | 62707 | 
| 67 | 34 django/core/management/templates.py | 211 | 242| 236 | 24880 | 62707 | 
| 68 | 34 django/core/management/__init__.py | 228 | 258| 309 | 25189 | 62707 | 
| 69 | 35 docs/_ext/djangodocs.py | 274 | 366| 741 | 25930 | 65863 | 
| 70 | 35 django/core/management/base.py | 158 | 238| 762 | 26692 | 65863 | 
| 71 | 36 django/utils/log.py | 1 | 75| 484 | 27176 | 67505 | 
| 72 | 36 django/core/management/commands/makemigrations.py | 154 | 192| 313 | 27489 | 67505 | 
| 73 | 37 django/db/backends/mysql/client.py | 1 | 58| 546 | 28035 | 68052 | 
| 74 | 37 django/core/management/commands/migrate.py | 253 | 270| 208 | 28243 | 68052 | 
| 75 | 38 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 28401 | 69238 | 
| 76 | 38 django/core/management/base.py | 45 | 67| 205 | 28606 | 69238 | 
| 77 | 38 django/core/management/commands/inspectdb.py | 38 | 173| 1291 | 29897 | 69238 | 
| 78 | 38 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 30404 | 69238 | 
| 79 | 39 django/views/debug.py | 1 | 47| 296 | 30700 | 73829 | 
| 80 | 39 docs/_ext/djangodocs.py | 26 | 71| 398 | 31098 | 73829 | 
| 81 | 39 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 31284 | 73829 | 
| 82 | 39 django/core/management/commands/dumpdata.py | 141 | 177| 316 | 31600 | 73829 | 
| 83 | 39 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 31859 | 73829 | 
| 84 | 39 django/core/management/__init__.py | 184 | 226| 343 | 32202 | 73829 | 
| 85 | 39 django/core/management/commands/makemigrations.py | 24 | 59| 284 | 32486 | 73829 | 
| 86 | 40 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 333 | 32819 | 74363 | 
| 87 | 40 django/core/management/__init__.py | 1 | 40| 278 | 33097 | 74363 | 
| 88 | 41 django/conf/global_settings.py | 151 | 266| 859 | 33956 | 80123 | 
| 89 | 41 django/core/management/commands/makemessages.py | 197 | 214| 176 | 34132 | 80123 | 
| 90 | 42 django/__main__.py | 1 | 10| 0 | 34132 | 80168 | 
| 91 | 43 django/db/backends/sqlite3/client.py | 1 | 17| 104 | 34236 | 80273 | 
| 92 | 44 django/core/management/commands/startapp.py | 1 | 15| 0 | 34236 | 80374 | 
| 93 | 44 django/core/management/utils.py | 112 | 125| 119 | 34355 | 80374 | 
| 94 | 44 django/contrib/staticfiles/management/commands/collectstatic.py | 38 | 69| 297 | 34652 | 80374 | 
| 95 | 45 django/conf/__init__.py | 1 | 46| 298 | 34950 | 82561 | 
| 96 | 45 scripts/manage_translations.py | 176 | 186| 116 | 35066 | 82561 | 
| 97 | 46 django/db/models/base.py | 1 | 50| 328 | 35394 | 99234 | 
| 98 | 47 django/db/migrations/operations/special.py | 133 | 179| 304 | 35698 | 100792 | 
| 99 | 48 django/core/checks/templates.py | 1 | 36| 259 | 35957 | 101052 | 
| 100 | 48 django/utils/autoreload.py | 1 | 56| 287 | 36244 | 101052 | 
| 101 | 48 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 36536 | 101052 | 
| 102 | 49 django/views/csrf.py | 15 | 100| 835 | 37371 | 102596 | 
| 103 | 49 django/core/management/commands/flush.py | 1 | 25| 206 | 37577 | 102596 | 
| 104 | 49 django/core/management/base.py | 514 | 547| 291 | 37868 | 102596 | 
| 105 | 49 django/core/management/commands/makemessages.py | 216 | 281| 633 | 38501 | 102596 | 
| 106 | 49 django/core/management/commands/loaddata.py | 87 | 157| 640 | 39141 | 102596 | 
| 107 | 49 django/contrib/staticfiles/management/commands/collectstatic.py | 71 | 84| 136 | 39277 | 102596 | 
| 108 | 49 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 39432 | 102596 | 
| 109 | 49 django/core/management/base.py | 550 | 582| 240 | 39672 | 102596 | 
| 110 | 49 django/db/migrations/operations/special.py | 181 | 204| 246 | 39918 | 102596 | 
| 111 | 49 django/core/management/commands/compilemessages.py | 30 | 57| 230 | 40148 | 102596 | 
| 112 | 50 django/core/management/commands/startproject.py | 1 | 21| 137 | 40285 | 102733 | 
| 113 | 51 django/db/models/options.py | 1 | 34| 285 | 40570 | 109839 | 
| 114 | 52 django/db/backends/utils.py | 92 | 129| 297 | 40867 | 111705 | 
| 115 | 52 django/core/management/commands/runserver.py | 55 | 65| 120 | 40987 | 111705 | 
| 116 | 53 django/core/checks/security/base.py | 1 | 84| 743 | 41730 | 113597 | 
| 117 | 53 django/utils/autoreload.py | 59 | 87| 156 | 41886 | 113597 | 
| 118 | 53 django/contrib/staticfiles/management/commands/collectstatic.py | 1 | 36| 226 | 42112 | 113597 | 
| 119 | 54 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 42364 | 113850 | 
| 120 | 55 django/core/mail/backends/dummy.py | 1 | 11| 0 | 42364 | 113893 | 
| 121 | 56 django/db/models/functions/__init__.py | 1 | 45| 671 | 43035 | 114564 | 
| 122 | 57 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 43156 | 115420 | 
| 123 | 58 django/core/files/base.py | 75 | 118| 303 | 43459 | 116472 | 
| 124 | 59 django/contrib/gis/management/commands/ogrinspect.py | 98 | 135| 407 | 43866 | 117681 | 
| 125 | 59 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 44344 | 117681 | 
| 126 | 60 django/contrib/postgres/functions.py | 1 | 12| 0 | 44344 | 117734 | 
| 127 | 60 django/core/management/commands/makemessages.py | 394 | 416| 200 | 44544 | 117734 | 
| 128 | 61 django/template/backends/dummy.py | 1 | 53| 325 | 44869 | 118059 | 
| 129 | 62 django/core/checks/messages.py | 53 | 76| 161 | 45030 | 118632 | 
| 130 | 63 django/template/context.py | 1 | 24| 128 | 45158 | 120513 | 
| 131 | 64 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 45352 | 120707 | 
| 132 | 65 django/http/request.py | 1 | 50| 397 | 45749 | 126179 | 
| 133 | 66 django/contrib/gis/gdal/libgdal.py | 1 | 121| 913 | 46662 | 127092 | 
| 134 | 67 django/contrib/admin/exceptions.py | 1 | 12| 0 | 46662 | 127159 | 
| 135 | 68 django/core/servers/basehttp.py | 200 | 217| 210 | 46872 | 128904 | 
| 136 | 68 docs/conf.py | 102 | 206| 902 | 47774 | 128904 | 
| 137 | 68 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 47917 | 128904 | 
| 138 | 69 django/utils/version.py | 1 | 15| 129 | 48046 | 129702 | 
| 139 | 69 django/db/backends/utils.py | 65 | 89| 247 | 48293 | 129702 | 
| 140 | 70 django/template/__init__.py | 1 | 72| 390 | 48683 | 130092 | 
| 141 | 70 django/conf/__init__.py | 167 | 226| 546 | 49229 | 130092 | 
| 142 | 71 django/urls/__init__.py | 1 | 24| 239 | 49468 | 130331 | 
| 143 | 71 django/contrib/staticfiles/management/commands/collectstatic.py | 330 | 350| 205 | 49673 | 130331 | 
| 144 | 71 django/contrib/staticfiles/management/commands/collectstatic.py | 86 | 146| 480 | 50153 | 130331 | 
| 145 | 72 django/http/__init__.py | 1 | 22| 197 | 50350 | 130528 | 
| 146 | 72 django/utils/autoreload.py | 109 | 116| 114 | 50464 | 130528 | 
| 147 | 72 django/db/backends/utils.py | 47 | 63| 176 | 50640 | 130528 | 
| 148 | 72 django/core/management/__init__.py | 43 | 75| 265 | 50905 | 130528 | 
| 149 | 72 django/contrib/staticfiles/management/commands/collectstatic.py | 294 | 328| 320 | 51225 | 130528 | 
| 150 | 73 django/core/mail/backends/console.py | 1 | 43| 281 | 51506 | 130810 | 
| 151 | 73 django/db/backends/utils.py | 1 | 45| 273 | 51779 | 130810 | 
| 152 | 73 django/core/management/base.py | 120 | 155| 241 | 52020 | 130810 | 
| 153 | 74 django/db/models/functions/window.py | 52 | 79| 182 | 52202 | 131453 | 
| 154 | 75 django/utils/itercompat.py | 1 | 9| 0 | 52202 | 131493 | 
| 155 | 75 django/core/management/base.py | 479 | 511| 281 | 52483 | 131493 | 
| 156 | 76 django/core/cache/backends/base.py | 1 | 51| 254 | 52737 | 133676 | 
| 157 | 76 django/core/management/commands/migrate.py | 272 | 304| 349 | 53086 | 133676 | 
| 158 | 77 django/http/response.py | 279 | 317| 282 | 53368 | 138251 | 
| 159 | 78 django/db/models/functions/text.py | 1 | 39| 266 | 53634 | 140587 | 
| 160 | 78 django/db/models/functions/window.py | 28 | 49| 154 | 53788 | 140587 | 
| 161 | 79 django/core/exceptions.py | 1 | 104| 436 | 54224 | 141776 | 
| 162 | 80 django/template/engine.py | 1 | 53| 388 | 54612 | 143086 | 
| 163 | 81 django/conf/urls/__init__.py | 1 | 23| 152 | 54764 | 143238 | 
| 164 | 82 django/db/__init__.py | 1 | 18| 141 | 54905 | 143631 | 
| 165 | 82 django/core/management/commands/loaddata.py | 38 | 67| 261 | 55166 | 143631 | 
| 166 | 83 django/template/base.py | 1 | 94| 779 | 55945 | 151509 | 
| 167 | 84 django/core/files/locks.py | 19 | 119| 779 | 56724 | 152466 | 
| 168 | 84 django/conf/global_settings.py | 51 | 150| 1160 | 57884 | 152466 | 
| 169 | 84 django/contrib/gis/management/commands/ogrinspect.py | 33 | 96| 591 | 58475 | 152466 | 
| 170 | 84 django/conf/global_settings.py | 401 | 491| 794 | 59269 | 152466 | 
| 171 | 85 django/db/backends/dummy/base.py | 1 | 47| 270 | 59539 | 152911 | 
| 172 | 85 django/core/management/commands/migrate.py | 306 | 353| 396 | 59935 | 152911 | 
| 173 | 86 django/core/signals.py | 1 | 7| 0 | 59935 | 152938 | 
| 174 | 86 django/template/context.py | 133 | 167| 288 | 60223 | 152938 | 
| 175 | 86 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 60351 | 152938 | 
| 176 | 87 django/db/backends/postgresql/base.py | 1 | 62| 480 | 60831 | 155811 | 
| 177 | 88 django/core/mail/backends/__init__.py | 1 | 2| 0 | 60831 | 155819 | 
| 178 | 89 django/utils/module_loading.py | 1 | 24| 165 | 60996 | 156562 | 
| 179 | 90 django/contrib/admin/utils.py | 309 | 366| 466 | 61462 | 160724 | 
| 180 | 90 django/conf/global_settings.py | 492 | 627| 812 | 62274 | 160724 | 
| 181 | 91 django/core/handlers/exception.py | 54 | 126| 588 | 62862 | 161823 | 
| 182 | 92 django/template/backends/jinja2.py | 1 | 51| 341 | 63203 | 162645 | 
| 183 | 93 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 63398 | 162840 | 
| 184 | 94 django/db/backends/base/schema.py | 113 | 146| 322 | 63720 | 175186 | 
| 185 | 95 django/utils/encoding.py | 102 | 115| 130 | 63850 | 177548 | 
| 186 | 96 django/views/i18n.py | 88 | 191| 702 | 64552 | 180085 | 
| 187 | 96 django/conf/global_settings.py | 267 | 349| 800 | 65352 | 180085 | 
| 188 | 97 django/db/models/functions/mixins.py | 1 | 20| 161 | 65513 | 180503 | 
| 189 | 97 django/core/management/templates.py | 244 | 296| 404 | 65917 | 180503 | 
| 190 | 98 django/db/backends/base/operations.py | 674 | 694| 187 | 66104 | 186099 | 
| 191 | 99 django/core/management/commands/sqlflush.py | 1 | 26| 189 | 66293 | 186288 | 
| 192 | 99 django/contrib/staticfiles/management/commands/collectstatic.py | 207 | 242| 248 | 66541 | 186288 | 
| 193 | 99 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 66680 | 186288 | 
| 194 | 99 django/core/files/base.py | 1 | 29| 174 | 66854 | 186288 | 
| 195 | 100 django/urls/exceptions.py | 1 | 10| 0 | 66854 | 186313 | 
| 196 | 100 django/core/management/commands/squashmigrations.py | 206 | 219| 112 | 66966 | 186313 | 
| 197 | 101 django/db/backends/sqlite3/base.py | 1 | 78| 535 | 67501 | 192366 | 
| 198 | 101 django/core/management/templates.py | 40 | 56| 181 | 67682 | 192366 | 


### Hint

```
​PR includes tests and documents the new feature in the release notes (but not in the main docs since it seems more like a bug fix than a new feature to me).
```

## Patch

```diff
diff --git a/django/core/management/commands/shell.py b/django/core/management/commands/shell.py
--- a/django/core/management/commands/shell.py
+++ b/django/core/management/commands/shell.py
@@ -84,13 +84,13 @@ def python(self, options):
     def handle(self, **options):
         # Execute the command and exit.
         if options['command']:
-            exec(options['command'])
+            exec(options['command'], globals())
             return
 
         # Execute stdin if it has anything to read and exit.
         # Not supported on Windows due to select.select() limitations.
         if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
-            exec(sys.stdin.read())
+            exec(sys.stdin.read(), globals())
             return
 
         available_shells = [options['interface']] if options['interface'] else self.shells

```

## Test Patch

```diff
diff --git a/tests/shell/tests.py b/tests/shell/tests.py
--- a/tests/shell/tests.py
+++ b/tests/shell/tests.py
@@ -9,6 +9,13 @@
 
 
 class ShellCommandTestCase(SimpleTestCase):
+    script_globals = 'print("__name__" in globals())'
+    script_with_inline_function = (
+        'import django\n'
+        'def f():\n'
+        '    print(django.__version__)\n'
+        'f()'
+    )
 
     def test_command_option(self):
         with self.assertLogs('test', 'INFO') as cm:
@@ -21,6 +28,16 @@ def test_command_option(self):
             )
         self.assertEqual(cm.records[0].getMessage(), __version__)
 
+    def test_command_option_globals(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_globals)
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    def test_command_option_inline_function_call(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_with_inline_function)
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @unittest.skipIf(sys.platform == 'win32', "Windows select() doesn't support file descriptors.")
     @mock.patch('django.core.management.commands.shell.select')
     def test_stdin_read(self, select):
@@ -30,6 +47,30 @@ def test_stdin_read(self, select):
             call_command('shell')
         self.assertEqual(stdout.getvalue().strip(), '100')
 
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_globals(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_globals)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_inline_function_call(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_with_inline_function)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @mock.patch('django.core.management.commands.shell.select.select')  # [1]
     @mock.patch.dict('sys.modules', {'IPython': None})
     def test_shell_with_ipython_not_installed(self, select):

```


## Code snippets

### 1 - django/core/management/commands/shell.py:

Start line: 42, End line: 82

```python
class Command(BaseCommand):

    def python(self, options):
        import code

        # Set up a dictionary to serve as the environment for the shell, so
        # that tab completion works on objects that are imported at runtime.
        imported_objects = {}
        try:  # Try activating rlcompleter, because it's handy.
            import readline
        except ImportError:
            pass
        else:
            # We don't have to wrap the following import in a 'try', because
            # we already know 'readline' was imported successfully.
            import rlcompleter
            readline.set_completer(rlcompleter.Completer(imported_objects).complete)
            # Enable tab completion on systems using libedit (e.g. macOS).
            # These lines are copied from Python's Lib/site.py.
            readline_doc = getattr(readline, '__doc__', '')
            if readline_doc is not None and 'libedit' in readline_doc:
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab:complete")

        # We want to honor both $PYTHONSTARTUP and .pythonrc.py, so follow system
        # conventions and get $PYTHONSTARTUP first then .pythonrc.py.
        if not options['no_startup']:
            for pythonrc in OrderedSet([os.environ.get("PYTHONSTARTUP"), os.path.expanduser('~/.pythonrc.py')]):
                if not pythonrc:
                    continue
                if not os.path.isfile(pythonrc):
                    continue
                with open(pythonrc) as handle:
                    pythonrc_code = handle.read()
                # Match the behavior of the cpython shell where an error in
                # PYTHONSTARTUP prints an exception and continues.
                try:
                    exec(compile(pythonrc_code, pythonrc, 'exec'), imported_objects)
                except Exception:
                    traceback.print_exc()

        code.interact(local=imported_objects)
```
### 2 - django/core/management/commands/shell.py:

Start line: 84, End line: 104

```python
class Command(BaseCommand):

    def handle(self, **options):
        # Execute the command and exit.
        if options['command']:
            exec(options['command'])
            return

        # Execute stdin if it has anything to read and exit.
        # Not supported on Windows due to select.select() limitations.
        if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            exec(sys.stdin.read())
            return

        available_shells = [options['interface']] if options['interface'] else self.shells

        for shell in available_shells:
            try:
                return getattr(self, shell)(options)
            except ImportError:
                pass
        raise CommandError("Couldn't import {} interface.".format(shell))
```
### 3 - django/core/management/commands/shell.py:

Start line: 1, End line: 40

```python
import os
import select
import sys
import traceback

from django.core.management import BaseCommand, CommandError
from django.utils.datastructures import OrderedSet


class Command(BaseCommand):
    help = (
        "Runs a Python interactive interpreter. Tries to use IPython or "
        "bpython, if one of them is available. Any standard input is executed "
        "as code."
    )

    requires_system_checks = []
    shells = ['ipython', 'bpython', 'python']

    def add_arguments(self, parser):
        parser.add_argument(
            '--no-startup', action='store_true',
            help='When using plain Python, ignore the PYTHONSTARTUP environment variable and ~/.pythonrc.py script.',
        )
        parser.add_argument(
            '-i', '--interface', choices=self.shells,
            help='Specify an interactive interpreter interface. Available options: "ipython", "bpython", and "python"',
        )
        parser.add_argument(
            '-c', '--command',
            help='Instead of opening an interactive shell, run a command as Django and exit.',
        )

    def ipython(self, options):
        from IPython import start_ipython
        start_ipython(argv=[])

    def bpython(self, options):
        import bpython
        bpython.embed()
```
### 4 - django/core/management/commands/dbshell.py:

Start line: 23, End line: 44

```python
class Command(BaseCommand):

    def handle(self, **options):
        connection = connections[options['database']]
        try:
            connection.client.runshell(options['parameters'])
        except FileNotFoundError:
            # Note that we're assuming the FileNotFoundError relates to the
            # command missing. It could be raised for some other reason, in
            # which case this error message would be inaccurate. Still, this
            # message catches the common case.
            raise CommandError(
                'You appear not to have the %r program installed or on your path.' %
                connection.client.executable_name
            )
        except subprocess.CalledProcessError as e:
            raise CommandError(
                '"%s" returned non-zero exit status %s.' % (
                    ' '.join(e.cmd),
                    e.returncode,
                ),
                returncode=e.returncode,
            )
```
### 5 - django/core/management/__init__.py:

Start line: 78, End line: 181

```python
def call_command(command_name, *args, **options):
    """
    Call the given command, with the given options and args/kwargs.

    This is the primary API you should use for calling specific commands.

    `command_name` may be a string or a command object. Using a string is
    preferred unless the command object is required for further processing or
    testing.

    Some examples:
        call_command('migrate')
        call_command('shell', plain=True)
        call_command('sqlmigrate', 'myapp')

        from django.core.management.commands import flush
        cmd = flush.Command()
        call_command(cmd, verbosity=0, interactive=False)
        # Do something with cmd ...
    """
    if isinstance(command_name, BaseCommand):
        # Command object passed in.
        command = command_name
        command_name = command.__class__.__module__.split('.')[-1]
    else:
        # Load the command object by name.
        try:
            app_name = get_commands()[command_name]
        except KeyError:
            raise CommandError("Unknown command: %r" % command_name)

        if isinstance(app_name, BaseCommand):
            # If the command is already loaded, use it directly.
            command = app_name
        else:
            command = load_command_class(app_name, command_name)

    # Simulate argument parsing to get the option defaults (see #10080 for details).
    parser = command.create_parser('', command_name)
    # Use the `dest` option name from the parser option
    opt_mapping = {
        min(s_opt.option_strings).lstrip('-').replace('-', '_'): s_opt.dest
        for s_opt in parser._actions if s_opt.option_strings
    }
    arg_options = {opt_mapping.get(key, key): value for key, value in options.items()}
    parse_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            parse_args += map(str, arg)
        else:
            parse_args.append(str(arg))

    def get_actions(parser):
        # Parser actions and actions from sub-parser choices.
        for opt in parser._actions:
            if isinstance(opt, _SubParsersAction):
                for sub_opt in opt.choices.values():
                    yield from get_actions(sub_opt)
            else:
                yield opt

    parser_actions = list(get_actions(parser))
    mutually_exclusive_required_options = {
        opt
        for group in parser._mutually_exclusive_groups
        for opt in group._group_actions if group.required
    }
    # Any required arguments which are passed in via **options must be passed
    # to parse_args().
    for opt in parser_actions:
        if (
            opt.dest in options and
            (opt.required or opt in mutually_exclusive_required_options)
        ):
            parse_args.append(min(opt.option_strings))
            if isinstance(opt, (_AppendConstAction, _CountAction, _StoreConstAction)):
                continue
            value = arg_options[opt.dest]
            if isinstance(value, (list, tuple)):
                parse_args += map(str, value)
            else:
                parse_args.append(str(value))
    defaults = parser.parse_args(args=parse_args)
    defaults = dict(defaults._get_kwargs(), **arg_options)
    # Raise an error if any unknown options were passed.
    stealth_options = set(command.base_stealth_options + command.stealth_options)
    dest_parameters = {action.dest for action in parser_actions}
    valid_options = (dest_parameters | stealth_options).union(opt_mapping)
    unknown_options = set(options) - valid_options
    if unknown_options:
        raise TypeError(
            "Unknown option(s) for %s command: %s. "
            "Valid options are: %s." % (
                command_name,
                ', '.join(sorted(unknown_options)),
                ', '.join(sorted(valid_options)),
            )
        )
    # Move positional args out of options to mimic legacy optparse
    args = defaults.pop('args', ())
    if 'skip_checks' not in options:
        defaults['skip_checks'] = True

    return command.execute(*args, **defaults)
```
### 6 - django/core/management/commands/runserver.py:

Start line: 107, End line: 159

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
            "Quit the server with %(quit_command)s."
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
```
### 7 - django/core/management/base.py:

Start line: 373, End line: 408

```python
class BaseCommand:

    def execute(self, *args, **options):
        """
        Try to execute this command, performing system checks if needed (as
        controlled by the ``requires_system_checks`` attribute, except if
        force-skipped).
        """
        if options['force_color'] and options['no_color']:
            raise CommandError("The --no-color and --force-color options can't be used together.")
        if options['force_color']:
            self.style = color_style(force_color=True)
        elif options['no_color']:
            self.style = no_style()
            self.stderr.style_func = None
        if options.get('stdout'):
            self.stdout = OutputWrapper(options['stdout'])
        if options.get('stderr'):
            self.stderr = OutputWrapper(options['stderr'])

        if self.requires_system_checks and not options['skip_checks']:
            if self.requires_system_checks == ALL_CHECKS:
                self.check()
            else:
                self.check(tags=self.requires_system_checks)
        if self.requires_migrations_checks:
            self.check_migrations()
        output = self.handle(*args, **options)
        if output:
            if self.output_transaction:
                connection = connections[options.get('database', DEFAULT_DB_ALIAS)]
                output = '%s\n%s\n%s' % (
                    self.style.SQL_KEYWORD(connection.ops.start_transaction_sql()),
                    output,
                    self.style.SQL_KEYWORD(connection.ops.end_transaction_sql()),
                )
            self.stdout.write(output)
        return output
```
### 8 - django/core/management/__init__.py:

Start line: 334, End line: 420

```python
class ManagementUtility:

    def execute(self):
        """
        Given the command-line arguments, figure out which subcommand is being
        run, create a parser appropriate to that command, and run it.
        """
        try:
            subcommand = self.argv[1]
        except IndexError:
            subcommand = 'help'  # Display help if no arguments were given.

        # Preprocess options to extract --settings and --pythonpath.
        # These options could affect the commands that are available, so they
        # must be processed early.
        parser = CommandParser(
            prog=self.prog_name,
            usage='%(prog)s subcommand [options] [args]',
            add_help=False,
            allow_abbrev=False,
        )
        parser.add_argument('--settings')
        parser.add_argument('--pythonpath')
        parser.add_argument('args', nargs='*')  # catch-all
        try:
            options, args = parser.parse_known_args(self.argv[2:])
            handle_default_options(options)
        except CommandError:
            pass  # Ignore any option errors at this point.

        try:
            settings.INSTALLED_APPS
        except ImproperlyConfigured as exc:
            self.settings_exception = exc
        except ImportError as exc:
            self.settings_exception = exc

        if settings.configured:
            # Start the auto-reloading dev server even if the code is broken.
            # The hardcoded condition is a code smell but we can't rely on a
            # flag on the command class because we haven't located it yet.
            if subcommand == 'runserver' and '--noreload' not in self.argv:
                try:
                    autoreload.check_errors(django.setup)()
                except Exception:
                    # The exception will be raised later in the child process
                    # started by the autoreloader. Pretend it didn't happen by
                    # loading an empty list of applications.
                    apps.all_models = defaultdict(dict)
                    apps.app_configs = {}
                    apps.apps_ready = apps.models_ready = apps.ready = True

                    # Remove options not compatible with the built-in runserver
                    # (e.g. options for the contrib.staticfiles' runserver).
                    # Changes here require manually testing as described in
                    # #27522.
                    _parser = self.fetch_command('runserver').create_parser('django', 'runserver')
                    _options, _args = _parser.parse_known_args(self.argv[2:])
                    for _arg in _args:
                        self.argv.remove(_arg)

            # In all other cases, django.setup() is required to succeed.
            else:
                django.setup()

        self.autocomplete()

        if subcommand == 'help':
            if '--commands' in args:
                sys.stdout.write(self.main_help_text(commands_only=True) + '\n')
            elif not options.args:
                sys.stdout.write(self.main_help_text() + '\n')
            else:
                self.fetch_command(options.args[0]).print_help(self.prog_name, options.args[0])
        # Special-cases: We want 'django-admin --version' and
        # 'django-admin --help' to work, for backwards compatibility.
        elif subcommand == 'version' or self.argv[1:] == ['--version']:
            sys.stdout.write(django.get_version() + '\n')
        elif self.argv[1:] in (['--help'], ['-h']):
            sys.stdout.write(self.main_help_text() + '\n')
        else:
            self.fetch_command(subcommand).run_from_argv(self.argv)


def execute_from_command_line(argv=None):
    """Run a ManagementUtility."""
    utility = ManagementUtility(argv)
    utility.execute()
```
### 9 - django/core/management/commands/dbshell.py:

Start line: 1, End line: 21

```python
import subprocess

from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections


class Command(BaseCommand):
    help = (
        "Runs the command-line client for specified database, or the "
        "default database if none is provided."
    )

    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database onto which to open a shell. Defaults to the "default" database.',
        )
        parameters = parser.add_argument_group('parameters', prefix_chars='--')
        parameters.add_argument('parameters', nargs='*')
```
### 10 - django/core/management/base.py:

Start line: 21, End line: 42

```python
class CommandError(Exception):
    """
    Exception class indicating a problem while executing a management
    command.

    If this exception is raised during the execution of a management
    command, it will be caught and turned into a nicely-printed error
    message to the appropriate output stream (i.e., stderr); as a
    result, raising this exception (with a sensible description of the
    error) is the preferred way to indicate that something has gone
    wrong in the execution of a command.
    """
    def __init__(self, *args, returncode=1, **kwargs):
        self.returncode = returncode
        super().__init__(*args, **kwargs)


class SystemCheckError(CommandError):
    """
    The system check framework detected unrecoverable errors.
    """
    pass
```
