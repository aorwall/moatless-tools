# django__django-11583

| **django/django** | `60dc957a825232fdda9138e2f8878b2ca407a7c9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 14336 |
| **Any found context length** | 14336 |
| **Avg pos** | 62.0 |
| **Min pos** | 62 |
| **Max pos** | 62 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -143,6 +143,10 @@ def iter_modules_and_files(modules, extra_files):
             # The module could have been removed, don't fail loudly if this
             # is the case.
             continue
+        except ValueError as e:
+            # Network filesystems may return null bytes in file paths.
+            logger.debug('"%s" raised when resolving path: "%s"' % (str(e), path))
+            continue
         results.add(resolved_path)
     return frozenset(results)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/autoreload.py | 146 | 146 | 62 | 1 | 14336


## Problem Statement

```
Auto-reloading with StatReloader very intermittently throws "ValueError: embedded null byte".
Description
	
Raising this mainly so that it's tracked, as I have no idea how to reproduce it, nor why it's happening. It ultimately looks like a problem with Pathlib, which wasn't used prior to 2.2.
Stacktrace:
Traceback (most recent call last):
 File "manage.py" ...
	execute_from_command_line(sys.argv)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/__init__.py", line 381, in execute_from_command_line
	utility.execute()
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/__init__.py", line 375, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/base.py", line 323, in run_from_argv
	self.execute(*args, **cmd_options)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 60, in execute
	super().execute(*args, **options)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/base.py", line 364, in execute
	output = self.handle(*args, **options)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 95, in handle
	self.run(**options)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/core/management/commands/runserver.py", line 102, in run
	autoreload.run_with_reloader(self.inner_run, **options)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 577, in run_with_reloader
	start_django(reloader, main_func, *args, **kwargs)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 562, in start_django
	reloader.run(django_main_thread)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 280, in run
	self.run_loop()
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 286, in run_loop
	next(ticker)
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 326, in tick
	for filepath, mtime in self.snapshot_files():
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 342, in snapshot_files
	for file in self.watched_files():
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 241, in watched_files
	yield from iter_all_python_module_files()
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 103, in iter_all_python_module_files
	return iter_modules_and_files(modules, frozenset(_error_files))
 File "/Userz/kez/path/to/venv/lib/python3.6/site-packages/django/utils/autoreload.py", line 132, in iter_modules_and_files
	results.add(path.resolve().absolute())
 File "/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/pathlib.py", line 1120, in resolve
	s = self._flavour.resolve(self, strict=strict)
 File "/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/pathlib.py", line 346, in resolve
	return _resolve(base, str(path)) or sep
 File "/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/pathlib.py", line 330, in _resolve
	target = accessor.readlink(newpath)
 File "/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/pathlib.py", line 441, in readlink
	return os.readlink(path)
ValueError: embedded null byte
I did print(path) before os.readlink(path) in pathlib and ended up with:
/Users/kez
/Users/kez/.pyenv
/Users/kez/.pyenv/versions
/Users/kez/.pyenv/versions/3.6.2
/Users/kez/.pyenv/versions/3.6.2/lib
/Users/kez/.pyenv/versions/3.6.2/lib/python3.6
/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/asyncio
/Users/kez/.pyenv/versions/3.6.2/lib/python3.6/asyncio/selector_events.py
/Users
It always seems to be /Users which is last
It may have already printed /Users as part of another .resolve() multiple times (that is, the order is not deterministic, and it may have traversed beyond /Users successfully many times during startup.
I don't know where to begin looking for the rogue null byte, nor why it only exists sometimes.
Best guess I have is that there's a mountpoint in /Users to a samba share which may not have been connected to yet? I dunno.
I have no idea if it's fixable without removing the use of pathlib (which tbh I think should happen anyway, because it's slow) and reverting to using os.path.join and friends. 
I have no idea if it's fixed in a later Python version, but with no easy way to reproduce ... dunno how I'd check.
I have no idea if it's something specific to my system (pyenv, OSX 10.11, etc)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/autoreload.py** | 335 | 374| 266 | 266 | 4698 | 
| 2 | **1 django/utils/autoreload.py** | 1 | 45| 227 | 493 | 4698 | 
| 3 | **1 django/utils/autoreload.py** | 98 | 105| 114 | 607 | 4698 | 
| 4 | **1 django/utils/autoreload.py** | 48 | 76| 156 | 763 | 4698 | 
| 5 | **1 django/utils/autoreload.py** | 589 | 601| 117 | 880 | 4698 | 
| 6 | **1 django/utils/autoreload.py** | 377 | 407| 349 | 1229 | 4698 | 
| 7 | **1 django/utils/autoreload.py** | 281 | 295| 146 | 1375 | 4698 | 
| 8 | **1 django/utils/autoreload.py** | 297 | 332| 259 | 1634 | 4698 | 
| 9 | **1 django/utils/autoreload.py** | 181 | 228| 320 | 1954 | 4698 | 
| 10 | **1 django/utils/autoreload.py** | 543 | 567| 212 | 2166 | 4698 | 
| 11 | **1 django/utils/autoreload.py** | 570 | 586| 169 | 2335 | 4698 | 
| 12 | **1 django/utils/autoreload.py** | 463 | 483| 268 | 2603 | 4698 | 
| 13 | **1 django/utils/autoreload.py** | 426 | 438| 145 | 2748 | 4698 | 
| 14 | 2 django/core/files/storage.py | 226 | 287| 524 | 3272 | 7534 | 
| 15 | **2 django/utils/autoreload.py** | 231 | 261| 234 | 3506 | 7534 | 
| 16 | 3 django/db/migrations/loader.py | 148 | 174| 291 | 3797 | 10451 | 
| 17 | 4 django/core/management/commands/runserver.py | 66 | 104| 397 | 4194 | 11902 | 
| 18 | 5 django/core/checks/security/base.py | 1 | 86| 752 | 4946 | 13528 | 
| 19 | 5 django/core/files/storage.py | 198 | 224| 215 | 5161 | 13528 | 
| 20 | **5 django/utils/autoreload.py** | 485 | 506| 205 | 5366 | 13528 | 
| 21 | 6 django/contrib/staticfiles/finders.py | 70 | 93| 202 | 5568 | 15569 | 
| 22 | 7 django/core/files/base.py | 75 | 118| 303 | 5871 | 16621 | 
| 23 | 8 django/contrib/admin/checks.py | 718 | 749| 229 | 6100 | 25637 | 
| 24 | 9 django/core/files/__init__.py | 1 | 4| 0 | 6100 | 25652 | 
| 25 | 9 django/core/management/commands/runserver.py | 106 | 162| 517 | 6617 | 25652 | 
| 26 | 9 django/core/files/storage.py | 1 | 22| 158 | 6775 | 25652 | 
| 27 | 10 setup.py | 1 | 66| 508 | 7283 | 26675 | 
| 28 | 11 django/urls/base.py | 1 | 25| 177 | 7460 | 27861 | 
| 29 | 11 django/core/files/base.py | 1 | 29| 174 | 7634 | 27861 | 
| 30 | 12 django/db/models/options.py | 1 | 36| 304 | 7938 | 34880 | 
| 31 | **12 django/utils/autoreload.py** | 409 | 424| 156 | 8094 | 34880 | 
| 32 | 13 django/contrib/staticfiles/checks.py | 1 | 15| 0 | 8094 | 34956 | 
| 33 | 14 django/template/loaders/app_directories.py | 1 | 15| 0 | 8094 | 35015 | 
| 34 | 15 django/utils/itercompat.py | 1 | 9| 0 | 8094 | 35055 | 
| 35 | **15 django/utils/autoreload.py** | 150 | 178| 277 | 8371 | 35055 | 
| 36 | 16 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 8500 | 38971 | 
| 37 | 17 django/core/checks/templates.py | 1 | 36| 259 | 8759 | 39231 | 
| 38 | 18 django/conf/global_settings.py | 145 | 263| 876 | 9635 | 44835 | 
| 39 | 19 django/conf/__init__.py | 132 | 185| 472 | 10107 | 46621 | 
| 40 | **19 django/utils/autoreload.py** | 79 | 95| 161 | 10268 | 46621 | 
| 41 | 20 django/views/debug.py | 388 | 456| 575 | 10843 | 50839 | 
| 42 | 21 django/core/mail/backends/dummy.py | 1 | 11| 0 | 10843 | 50882 | 
| 43 | 22 django/template/backends/utils.py | 1 | 15| 0 | 10843 | 50971 | 
| 44 | 23 django/utils/translation/__init__.py | 55 | 65| 127 | 10970 | 53297 | 
| 45 | 23 django/core/management/commands/runserver.py | 1 | 20| 191 | 11161 | 53297 | 
| 46 | 24 django/__main__.py | 1 | 10| 0 | 11161 | 53342 | 
| 47 | 24 django/contrib/staticfiles/finders.py | 1 | 17| 110 | 11271 | 53342 | 
| 48 | 25 django/core/cache/utils.py | 1 | 13| 0 | 11271 | 53429 | 
| 49 | 26 django/apps/config.py | 54 | 79| 271 | 11542 | 55161 | 
| 50 | 27 django/db/utils.py | 1 | 49| 154 | 11696 | 57273 | 
| 51 | 28 django/db/models/base.py | 1 | 45| 289 | 11985 | 72363 | 
| 52 | 29 django/core/checks/messages.py | 53 | 76| 161 | 12146 | 72936 | 
| 53 | 30 django/core/management/commands/loaddata.py | 1 | 29| 151 | 12297 | 75804 | 
| 54 | 31 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 12297 | 75897 | 
| 55 | 31 django/conf/global_settings.py | 499 | 638| 853 | 13150 | 75897 | 
| 56 | 32 django/db/migrations/utils.py | 1 | 18| 0 | 13150 | 75985 | 
| 57 | 33 django/contrib/staticfiles/management/commands/collectstatic.py | 294 | 328| 320 | 13470 | 78829 | 
| 58 | 34 django/core/management/commands/compilemessages.py | 1 | 26| 157 | 13627 | 80095 | 
| 59 | 35 django/db/backends/dummy/features.py | 1 | 7| 0 | 13627 | 80127 | 
| 60 | **35 django/utils/autoreload.py** | 508 | 541| 228 | 13855 | 80127 | 
| 61 | 36 django/db/migrations/autodetector.py | 1 | 15| 110 | 13965 | 91798 | 
| **-> 62 <-** | **36 django/utils/autoreload.py** | 108 | 147| 371 | 14336 | 91798 | 
| 63 | 37 django/core/signals.py | 1 | 7| 0 | 14336 | 91853 | 
| 64 | 38 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 14336 | 91954 | 
| 65 | 38 django/core/management/commands/loaddata.py | 81 | 148| 593 | 14929 | 91954 | 
| 66 | 39 django/core/files/locks.py | 19 | 114| 773 | 15702 | 92905 | 
| 67 | 40 django/utils/archive.py | 143 | 182| 290 | 15992 | 94408 | 
| 68 | 41 django/db/models/fields/related.py | 1202 | 1313| 939 | 16931 | 107908 | 
| 69 | 42 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 16931 | 107983 | 
| 70 | 42 django/conf/global_settings.py | 264 | 346| 800 | 17731 | 107983 | 
| 71 | 43 django/contrib/sites/middleware.py | 1 | 13| 0 | 17731 | 108042 | 
| 72 | 44 django/core/validators.py | 1 | 26| 199 | 17930 | 112359 | 
| 73 | 45 django/utils/translation/trans_null.py | 1 | 68| 269 | 18199 | 112628 | 
| 74 | 46 django/utils/version.py | 1 | 15| 129 | 18328 | 113422 | 
| 75 | 47 django/template/base.py | 92 | 114| 153 | 18481 | 121283 | 
| 76 | 48 django/core/checks/caches.py | 1 | 17| 0 | 18481 | 121383 | 
| 77 | 48 django/contrib/staticfiles/storage.py | 257 | 327| 574 | 19055 | 121383 | 
| 78 | 48 django/core/validators.py | 111 | 150| 398 | 19453 | 121383 | 
| 79 | 49 django/contrib/staticfiles/__init__.py | 1 | 2| 0 | 19453 | 121397 | 
| 80 | 50 django/core/mail/backends/__init__.py | 1 | 2| 0 | 19453 | 121405 | 
| 81 | 51 django/core/management/commands/shell.py | 42 | 81| 401 | 19854 | 122226 | 
| 82 | 51 django/db/models/fields/related.py | 1 | 34| 240 | 20094 | 122226 | 
| 83 | 52 django/urls/resolvers.py | 438 | 497| 548 | 20642 | 127670 | 
| 84 | 52 django/db/migrations/autodetector.py | 49 | 87| 322 | 20964 | 127670 | 
| 85 | 53 django/core/checks/model_checks.py | 166 | 199| 332 | 21296 | 129351 | 
| 86 | 54 django/utils/_os.py | 34 | 50| 127 | 21423 | 129789 | 
| 87 | 54 django/core/checks/security/base.py | 88 | 190| 747 | 22170 | 129789 | 
| 88 | 55 django/dispatch/__init__.py | 1 | 10| 0 | 22170 | 129854 | 
| 89 | 55 django/template/base.py | 117 | 138| 127 | 22297 | 129854 | 
| 90 | 55 django/core/files/storage.py | 289 | 361| 483 | 22780 | 129854 | 
| 91 | 56 docs/conf.py | 1 | 95| 746 | 23526 | 132831 | 
| 92 | 57 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 23731 | 133284 | 
| 93 | 58 django/db/models/fields/__init__.py | 1688 | 1702| 144 | 23875 | 150295 | 
| 94 | 58 django/core/checks/model_checks.py | 143 | 164| 263 | 24138 | 150295 | 
| 95 | 59 django/contrib/messages/utils.py | 1 | 13| 0 | 24138 | 150345 | 
| 96 | 60 django/apps/__init__.py | 1 | 5| 0 | 24138 | 150368 | 
| 97 | 61 django/db/backends/base/base.py | 1 | 23| 138 | 24276 | 155226 | 
| 98 | 62 django/utils/encoding.py | 150 | 165| 182 | 24458 | 157558 | 
| 99 | 62 django/core/files/storage.py | 63 | 92| 343 | 24801 | 157558 | 
| 100 | 63 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 24801 | 157634 | 
| 101 | 64 django/contrib/staticfiles/apps.py | 1 | 14| 0 | 24801 | 157722 | 
| 102 | 64 django/urls/resolvers.py | 534 | 572| 355 | 25156 | 157722 | 
| 103 | 64 django/db/migrations/autodetector.py | 264 | 335| 748 | 25904 | 157722 | 
| 104 | 65 django/contrib/auth/__init__.py | 1 | 58| 393 | 26297 | 159289 | 
| 105 | 66 django/template/loaders/filesystem.py | 1 | 47| 287 | 26584 | 159576 | 
| 106 | 67 django/views/__init__.py | 1 | 4| 0 | 26584 | 159591 | 
| 107 | 68 django/template/loader_tags.py | 1 | 38| 182 | 26766 | 162142 | 
| 108 | 69 django/db/migrations/questioner.py | 1 | 54| 468 | 27234 | 164216 | 
| 109 | 70 django/utils/module_loading.py | 82 | 98| 128 | 27362 | 164959 | 
| 110 | 71 django/db/models/constants.py | 1 | 7| 0 | 27362 | 164984 | 
| 111 | 72 django/contrib/admin/utils.py | 306 | 363| 468 | 27830 | 169057 | 
| 112 | 73 django/db/models/fields/files.py | 150 | 209| 645 | 28475 | 172778 | 
| 113 | 74 django/db/migrations/__init__.py | 1 | 3| 0 | 28475 | 172802 | 
| 114 | 75 django/core/checks/urls.py | 53 | 68| 128 | 28603 | 173503 | 
| 115 | 76 django/db/backends/postgresql/utils.py | 1 | 8| 0 | 28603 | 173541 | 
| 116 | 77 django/contrib/syndication/__init__.py | 1 | 2| 0 | 28603 | 173558 | 
| 117 | 78 django/contrib/gis/gdal/prototypes/errcheck.py | 1 | 35| 221 | 28824 | 174541 | 
| 118 | 79 django/contrib/humanize/__init__.py | 1 | 2| 0 | 28824 | 174557 | 
| 119 | **79 django/utils/autoreload.py** | 263 | 279| 162 | 28986 | 174557 | 
| 120 | 79 django/core/checks/urls.py | 1 | 27| 142 | 29128 | 174557 | 
| 121 | 80 django/urls/exceptions.py | 1 | 10| 0 | 29128 | 174582 | 
| 122 | 81 django/db/migrations/serializer.py | 71 | 98| 233 | 29361 | 177120 | 
| 123 | 82 django/db/backends/signals.py | 1 | 4| 0 | 29361 | 177137 | 
| 124 | 82 django/core/files/base.py | 31 | 46| 129 | 29490 | 177137 | 
| 125 | 83 django/http/request.py | 1 | 37| 251 | 29741 | 181967 | 
| 126 | 84 django/contrib/sites/__init__.py | 1 | 2| 0 | 29741 | 181981 | 
| 127 | 85 django/urls/__init__.py | 1 | 24| 239 | 29980 | 182220 | 
| 128 | 86 django/views/i18n.py | 77 | 180| 711 | 30691 | 184727 | 
| 129 | 86 django/views/debug.py | 1 | 45| 306 | 30997 | 184727 | 
| 130 | 87 django/views/decorators/gzip.py | 1 | 6| 0 | 30997 | 184778 | 
| 131 | 87 django/template/base.py | 571 | 606| 359 | 31356 | 184778 | 
| 132 | 88 django/contrib/auth/backends.py | 1 | 31| 178 | 31534 | 186353 | 
| 133 | 89 django/contrib/sessions/exceptions.py | 1 | 12| 0 | 31534 | 186404 | 
| 134 | 90 django/template/defaultfilters.py | 804 | 848| 378 | 31912 | 192469 | 
| 135 | 90 django/db/migrations/autodetector.py | 1079 | 1114| 312 | 32224 | 192469 | 
| 136 | 91 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 32224 | 192492 | 


### Hint

```
Thanks for the report, however as you've admitted there is too many unknowns to accept this ticket. I don't believe that it is related with pathlib, maybe samba connection is unstable it's hard to tell.
I don't believe that it is related with pathlib Well ... it definitely is, you can see that from the stacktrace. The difference between 2.2 and 2.1 (and every version prior) for the purposes of this report is that AFAIK 2.2 is using pathlib.resolve() which deals with symlinks where under <2.2 I don't think the equivalent (os.path.realpath rather than os.path.abspath) was used. But yes, there's no path forward to fix the ticket as it stands, short of not using pathlib (or at least .resolve()).
Hey Keryn, Have you tried removing resolve() yourself, and did it fix the issue? I chose to use resolve() to try and work around a corner case with symlinks, and to generally just normalize the paths to prevent duplication. Also, regarding your comment above, you would need to use print(repr(path)), as I think the print machinery stops at the first null byte found (hence just /Users, which should never be monitored by itself). If you can provide me some more information I'm more than willing to look into this, or consider removing the resolve() call.
Replying to Tom Forbes: Hey Keryn, Have you tried removing resolve() yourself, and did it fix the issue? I chose to use resolve() to try and work around a corner case with symlinks, and to generally just normalize the paths to prevent duplication. Also, regarding your comment above, you would need to use print(repr(path)), as I think the print machinery stops at the first null byte found (hence just /Users, which should never be monitored by itself). If you can provide me some more information I'm more than willing to look into this, or consider removing the resolve() call. Hi Tom, I am also getting this error, see here for the stackoverflow question which I have attempted to answer: ​https://stackoverflow.com/questions/56406965/django-valueerror-embedded-null-byte/56685648#56685648 What is really odd is that it doesn't error every time and looks to error on a random file each time. I believe the issue is caused by having a venv within the top level directory but might be wrong. Bug is on all versions of django >= 2.2.0
Felix, I'm going to re-open this ticket if that's OK. While this is clearly something "funky" going on at a lower level than we handle, it used to work (at least, the error was swallowed). I think this is a fairly simple fix.
```

## Patch

```diff
diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py
--- a/django/utils/autoreload.py
+++ b/django/utils/autoreload.py
@@ -143,6 +143,10 @@ def iter_modules_and_files(modules, extra_files):
             # The module could have been removed, don't fail loudly if this
             # is the case.
             continue
+        except ValueError as e:
+            # Network filesystems may return null bytes in file paths.
+            logger.debug('"%s" raised when resolving path: "%s"' % (str(e), path))
+            continue
         results.add(resolved_path)
     return frozenset(results)
 

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -140,6 +140,17 @@ def test_main_module_without_file_is_not_resolved(self):
         fake_main = types.ModuleType('__main__')
         self.assertEqual(autoreload.iter_modules_and_files((fake_main,), frozenset()), frozenset())
 
+    def test_path_with_embedded_null_bytes(self):
+        for path in (
+            'embedded_null_byte\x00.py',
+            'di\x00rectory/embedded_null_byte.py',
+        ):
+            with self.subTest(path=path):
+                self.assertEqual(
+                    autoreload.iter_modules_and_files((), frozenset([path])),
+                    frozenset(),
+                )
+
 
 class TestCommonRoots(SimpleTestCase):
     def test_common_roots(self):

```


## Code snippets

### 1 - django/utils/autoreload.py:

Start line: 335, End line: 374

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
### 2 - django/utils/autoreload.py:

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
### 3 - django/utils/autoreload.py:

Start line: 98, End line: 105

```python
def iter_all_python_module_files():
    # This is a hot path during reloading. Create a stable sorted list of
    # modules based on the module name and pass it to iter_modules_and_files().
    # This ensures cached results are returned in the usual case that modules
    # aren't loaded on the fly.
    keys = sorted(sys.modules)
    modules = tuple(m for m in map(sys.modules.__getitem__, keys) if not isinstance(m, weakref.ProxyTypes))
    return iter_modules_and_files(modules, frozenset(_error_files))
```
### 4 - django/utils/autoreload.py:

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
### 5 - django/utils/autoreload.py:

Start line: 589, End line: 601

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
### 6 - django/utils/autoreload.py:

Start line: 377, End line: 407

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
### 7 - django/utils/autoreload.py:

Start line: 281, End line: 295

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
### 8 - django/utils/autoreload.py:

Start line: 297, End line: 332

```python
class BaseReloader:

    def run_loop(self):
        ticker = self.tick()
        while not self.should_stop:
            try:
                next(ticker)
            except StopIteration:
                break
        self.stop()

    def tick(self):
        """
        This generator is called in a loop from run_loop. It's important that
        the method takes care of pausing or otherwise waiting for a period of
        time. This split between run_loop() and tick() is to improve the
        testability of the reloader implementations by decoupling the work they
        do from the loop.
        """
        raise NotImplementedError('subclasses must implement tick().')

    @classmethod
    def check_availability(cls):
        raise NotImplementedError('subclasses must implement check_availability().')

    def notify_file_changed(self, path):
        results = file_changed.send(sender=self, file_path=path)
        logger.debug('%s notified as changed. Signal results: %s.', path, results)
        if not any(res[1] for res in results):
            trigger_reload(path)

    # These are primarily used for testing.
    @property
    def should_stop(self):
        return self._stop_condition.is_set()

    def stop(self):
        self._stop_condition.set()
```
### 9 - django/utils/autoreload.py:

Start line: 181, End line: 228

```python
def sys_path_directories():
    """
    Yield absolute directories from sys.path, ignoring entries that don't
    exist.
    """
    for path in sys.path:
        path = Path(path)
        try:
            resolved_path = path.resolve(strict=True).absolute()
        except FileNotFoundError:
            continue
        # If the path is a file (like a zip file), watch the parent directory.
        if resolved_path.is_file():
            yield resolved_path.parent
        else:
            yield resolved_path


def get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.
    """
    import django.__main__

    args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
    if sys.argv[0] == django.__main__.__file__:
        # The server was started with `python -m django runserver`.
        args += ['-m', 'django']
        args += sys.argv[1:]
    else:
        args += sys.argv
    return args


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
```
### 10 - django/utils/autoreload.py:

Start line: 543, End line: 567

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
### 11 - django/utils/autoreload.py:

Start line: 570, End line: 586

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
### 12 - django/utils/autoreload.py:

Start line: 463, End line: 483

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
### 13 - django/utils/autoreload.py:

Start line: 426, End line: 438

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
### 15 - django/utils/autoreload.py:

Start line: 231, End line: 261

```python
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

    def watch_file(self, path):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('%s must be absolute.' % path)
        logger.debug('Watching file %s.', path)
        self.extra_files.add(path)

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
### 20 - django/utils/autoreload.py:

Start line: 485, End line: 506

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
### 31 - django/utils/autoreload.py:

Start line: 409, End line: 424

```python
class WatchmanReloader(BaseReloader):

    @functools.lru_cache()
    def _get_clock(self, root):
        return self.client.query('clock', root)['clock']

    def _subscribe(self, directory, name, expression):
        root, rel_path = self._watch_root(directory)
        query = {
            'expression': expression,
            'fields': ['name'],
            'since': self._get_clock(root),
            'dedup_results': True,
        }
        if rel_path:
            query['relative_root'] = rel_path
        logger.debug('Issuing watchman subscription %s, for root %s. Query: %s', name, root, query)
        self.client.query('subscribe', root, name, query)
```
### 35 - django/utils/autoreload.py:

Start line: 150, End line: 178

```python
@functools.lru_cache(maxsize=1)
def common_roots(paths):
    """
    Return a tuple of common roots that are shared between the given paths.
    File system watchers operate on directories and aren't cheap to create.
    Try to find the minimum set of directories to watch that encompass all of
    the files that need to be watched.
    """
    # Inspired from Werkzeug:
    # https://github.com/pallets/werkzeug/blob/7477be2853df70a022d9613e765581b9411c3c39/werkzeug/_reloader.py
    # Create a sorted list of the path components, longest first.
    path_parts = sorted([x.parts for x in paths], key=len, reverse=True)
    tree = {}
    for chunks in path_parts:
        node = tree
        # Add each part of the path to the tree.
        for chunk in chunks:
            node = node.setdefault(chunk, {})
        # Clear the last leaf in the tree.
        node.clear()

    # Turn the tree into a list of Path instances.
    def _walk(node, path):
        for prefix, child in node.items():
            yield from _walk(child, path + (prefix,))
        if not node:
            yield Path(*path)

    return tuple(_walk(tree, ()))
```
### 40 - django/utils/autoreload.py:

Start line: 79, End line: 95

```python
def ensure_echo_on():
    """
    Ensure that echo mode is enabled. Some tools such as PDB disable
    it which causes usability issues after reload.
    """
    if not termios or not sys.stdin.isatty():
        return
    attr_list = termios.tcgetattr(sys.stdin)
    if not attr_list[3] & termios.ECHO:
        attr_list[3] |= termios.ECHO
        if hasattr(signal, 'SIGTTOU'):
            old_handler = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        else:
            old_handler = None
        termios.tcsetattr(sys.stdin, termios.TCSANOW, attr_list)
        if old_handler is not None:
            signal.signal(signal.SIGTTOU, old_handler)
```
### 60 - django/utils/autoreload.py:

Start line: 508, End line: 541

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
### 62 - django/utils/autoreload.py:

Start line: 108, End line: 147

```python
@functools.lru_cache(maxsize=1)
def iter_modules_and_files(modules, extra_files):
    """Iterate through all modules needed to be watched."""
    sys_file_paths = []
    for module in modules:
        # During debugging (with PyDev) the 'typing.io' and 'typing.re' objects
        # are added to sys.modules, however they are types not modules and so
        # cause issues here.
        if not isinstance(module, ModuleType):
            continue
        if module.__name__ == '__main__':
            # __main__ (usually manage.py) doesn't always have a __spec__ set.
            # Handle this by falling back to using __file__, resolved below.
            # See https://docs.python.org/reference/import.html#main-spec
            # __file__ may not exists, e.g. when running ipdb debugger.
            if hasattr(module, '__file__'):
                sys_file_paths.append(module.__file__)
            continue
        if getattr(module, '__spec__', None) is None:
            continue
        spec = module.__spec__
        # Modules could be loaded from places without a concrete location. If
        # this is the case, skip them.
        if spec.has_location:
            origin = spec.loader.archive if isinstance(spec.loader, zipimporter) else spec.origin
            sys_file_paths.append(origin)

    results = set()
    for filename in itertools.chain(sys_file_paths, extra_files):
        if not filename:
            continue
        path = Path(filename)
        try:
            resolved_path = path.resolve(strict=True).absolute()
        except FileNotFoundError:
            # The module could have been removed, don't fail loudly if this
            # is the case.
            continue
        results.add(resolved_path)
    return frozenset(results)
```
### 119 - django/utils/autoreload.py:

Start line: 263, End line: 279

```python
class BaseReloader:

    def wait_for_apps_ready(self, app_reg, django_main_thread):
        """
        Wait until Django reports that the apps have been loaded. If the given
        thread has terminated before the apps are ready, then a SyntaxError or
        other non-recoverable error has been raised. In that case, stop waiting
        for the apps_ready event and continue processing.

        Return True if the thread is alive and the ready event has been
        triggered, or False if the thread is terminated while waiting for the
        event.
        """
        while django_main_thread.is_alive():
            if app_reg.ready_event.wait(timeout=0.1):
                return True
        else:
            logger.debug('Main Django thread has terminated before apps are ready.')
            return False
```
