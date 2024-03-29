# django__django-11903

| **django/django** | `dee687e93a2d45e9fac404be2098cc4707d31c1f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 305 |
| **Any found context length** | 305 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/__init__.py b/django/core/management/__init__.py
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -229,7 +229,7 @@ def fetch_command(self, subcommand):
                 # (get_commands() swallows the original one) so the user is
                 # informed about it.
                 settings.INSTALLED_APPS
-            else:
+            elif not settings.configured:
                 sys.stderr.write("No Django settings specified.\n")
             possible_matches = get_close_matches(subcommand, commands)
             sys.stderr.write('Unknown command: %r' % subcommand)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/__init__.py | 232 | 232 | 1 | 1 | 305


## Problem Statement

```
ManagementUtility.fetch_command prints "No Django settings specified." even if they are.
Description
	
fetch_command(...) currently ​does the following:
if os.environ.get('DJANGO_SETTINGS_MODULE'):
	# If `subcommand` is missing due to misconfigured settings, the
	# following line will retrigger an ImproperlyConfigured exception
	# (get_commands() swallows the original one) so the user is
	# informed about it.
	settings.INSTALLED_APPS
else:
	sys.stderr.write("No Django settings specified.\n")
which doesn't account for settings being set via a UserSettingsHolder by doing settings.configure(...)
But the parent execute method ​correctly checks if settings.configured:
I've not checked deeply, but I don't think the intent or outcome depends specifically on the LazySettings having been configured via a Settings through a named module import, and it would seem that if settings.configured: could/should apply here too.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/management/__init__.py** | 215 | 245| 305 | 305 | 3400 | 
| 2 | 2 django/conf/__init__.py | 43 | 64| 199 | 504 | 5459 | 
| 3 | 3 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 638 | 6150 | 
| 4 | 3 django/conf/__init__.py | 98 | 111| 131 | 769 | 6150 | 
| 5 | 3 django/conf/__init__.py | 223 | 273| 378 | 1147 | 6150 | 
| 6 | 3 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 1290 | 6150 | 
| 7 | 3 django/conf/__init__.py | 161 | 220| 541 | 1831 | 6150 | 
| 8 | **3 django/core/management/__init__.py** | 321 | 402| 743 | 2574 | 6150 | 
| 9 | 3 django/conf/__init__.py | 113 | 136| 168 | 2742 | 6150 | 
| 10 | 3 django/conf/__init__.py | 66 | 96| 260 | 3002 | 6150 | 
| 11 | 3 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 3130 | 6150 | 
| 12 | 3 django/conf/__init__.py | 138 | 158| 167 | 3297 | 6150 | 
| 13 | 4 django/conf/global_settings.py | 1 | 50| 366 | 3663 | 11778 | 
| 14 | 5 django/core/management/base.py | 64 | 88| 161 | 3824 | 16165 | 
| 15 | 5 django/conf/__init__.py | 1 | 40| 240 | 4064 | 16165 | 
| 16 | 6 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 4269 | 16618 | 
| 17 | 7 django/core/management/commands/loaddata.py | 81 | 148| 593 | 4862 | 19486 | 
| 18 | 8 django/core/management/commands/makemessages.py | 393 | 415| 200 | 5062 | 25031 | 
| 19 | 9 django/core/checks/translation.py | 1 | 62| 447 | 5509 | 25478 | 
| 20 | 10 django/views/debug.py | 1 | 45| 306 | 5815 | 29696 | 
| 21 | **10 django/core/management/__init__.py** | 247 | 319| 721 | 6536 | 29696 | 
| 22 | 10 django/conf/global_settings.py | 145 | 260| 854 | 7390 | 29696 | 
| 23 | 10 django/core/management/base.py | 347 | 382| 292 | 7682 | 29696 | 
| 24 | 11 setup.py | 1 | 66| 508 | 8190 | 30719 | 
| 25 | 12 django/core/management/commands/migrate.py | 67 | 160| 825 | 9015 | 33877 | 
| 26 | 12 django/conf/global_settings.py | 496 | 640| 900 | 9915 | 33877 | 
| 27 | 13 django/db/utils.py | 187 | 230| 295 | 10210 | 35989 | 
| 28 | 14 django/core/checks/templates.py | 1 | 36| 259 | 10469 | 36249 | 
| 29 | 15 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 11257 | 38998 | 
| 30 | 15 django/core/management/commands/makemessages.py | 1 | 33| 247 | 11504 | 38998 | 
| 31 | 15 django/core/management/commands/loaddata.py | 63 | 79| 187 | 11691 | 38998 | 
| 32 | 16 django/utils/log.py | 1 | 75| 484 | 12175 | 40623 | 
| 33 | 16 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 12477 | 40623 | 
| 34 | 17 django/db/models/options.py | 1 | 36| 304 | 12781 | 47722 | 
| 35 | 17 django/core/management/commands/loaddata.py | 1 | 29| 151 | 12932 | 47722 | 
| 36 | 18 django/core/management/commands/check.py | 36 | 66| 214 | 13146 | 48157 | 
| 37 | 19 django/contrib/auth/management/commands/createsuperuser.py | 81 | 202| 1155 | 14301 | 50217 | 
| 38 | 19 django/core/management/commands/diffsettings.py | 1 | 39| 299 | 14600 | 50217 | 
| 39 | 20 django/core/checks/security/base.py | 1 | 83| 732 | 15332 | 51998 | 
| 40 | 20 django/core/management/commands/loaddata.py | 32 | 61| 261 | 15593 | 51998 | 
| 41 | 21 django/core/management/utils.py | 52 | 74| 204 | 15797 | 53112 | 
| 42 | 21 django/core/management/commands/migrate.py | 161 | 242| 793 | 16590 | 53112 | 
| 43 | 22 django/contrib/admin/checks.py | 718 | 749| 229 | 16819 | 62128 | 
| 44 | 22 setup.py | 69 | 138| 536 | 17355 | 62128 | 
| 45 | 22 django/contrib/auth/management/commands/createsuperuser.py | 1 | 79| 577 | 17932 | 62128 | 
| 46 | 23 django/core/management/commands/runserver.py | 106 | 158| 504 | 18436 | 63565 | 
| 47 | 23 django/core/management/base.py | 384 | 449| 614 | 19050 | 63565 | 
| 48 | 24 django/core/management/commands/squashmigrations.py | 136 | 200| 652 | 19702 | 65436 | 
| 49 | 25 django/core/management/commands/shell.py | 83 | 103| 162 | 19864 | 66257 | 
| 50 | 26 django/core/management/commands/flush.py | 27 | 83| 496 | 20360 | 66954 | 
| 51 | 26 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 21151 | 66954 | 
| 52 | 26 django/core/management/base.py | 451 | 483| 282 | 21433 | 66954 | 
| 53 | **26 django/core/management/__init__.py** | 1 | 38| 261 | 21694 | 66954 | 
| 54 | 26 django/core/management/commands/shell.py | 1 | 40| 268 | 21962 | 66954 | 
| 55 | 26 django/core/management/commands/loaddata.py | 275 | 303| 246 | 22208 | 66954 | 
| 56 | 27 django/apps/registry.py | 276 | 296| 229 | 22437 | 70361 | 
| 57 | 28 django/contrib/sites/managers.py | 1 | 61| 385 | 22822 | 70746 | 
| 58 | 28 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 23172 | 70746 | 
| 59 | 29 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 23330 | 71932 | 
| 60 | 29 django/core/management/commands/squashmigrations.py | 202 | 215| 112 | 23442 | 71932 | 
| 61 | 29 django/core/management/commands/migrate.py | 21 | 65| 369 | 23811 | 71932 | 
| 62 | 29 django/conf/global_settings.py | 261 | 343| 800 | 24611 | 71932 | 
| 63 | 29 django/core/management/base.py | 230 | 247| 159 | 24770 | 71932 | 
| 64 | 29 django/core/management/commands/migrate.py | 259 | 291| 349 | 25119 | 71932 | 
| 65 | 30 django/contrib/auth/checks.py | 1 | 94| 646 | 25765 | 73105 | 
| 66 | 31 django/core/management/templates.py | 58 | 117| 527 | 26292 | 75786 | 
| 67 | 31 django/core/management/commands/runserver.py | 66 | 104| 397 | 26689 | 75786 | 
| 68 | 31 django/core/management/commands/shell.py | 42 | 81| 401 | 27090 | 75786 | 
| 69 | 32 django/db/models/fields/related.py | 362 | 399| 292 | 27382 | 89298 | 
| 70 | 32 django/db/utils.py | 1 | 49| 154 | 27536 | 89298 | 
| 71 | 33 django/db/backends/mysql/client.py | 1 | 49| 424 | 27960 | 89722 | 
| 72 | 34 django/contrib/admin/models.py | 1 | 20| 118 | 28078 | 90847 | 
| 73 | 34 django/core/management/base.py | 1 | 36| 223 | 28301 | 90847 | 
| 74 | 34 django/core/management/commands/makemessages.py | 282 | 361| 816 | 29117 | 90847 | 
| 75 | 35 django/utils/termcolors.py | 134 | 216| 653 | 29770 | 92734 | 
| 76 | 35 django/core/management/templates.py | 210 | 241| 236 | 30006 | 92734 | 
| 77 | 35 django/core/management/commands/check.py | 1 | 34| 226 | 30232 | 92734 | 
| 78 | 36 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 30858 | 94269 | 
| 79 | 37 django/contrib/auth/backends.py | 31 | 49| 142 | 31000 | 96031 | 
| 80 | 38 django/utils/module_loading.py | 27 | 60| 300 | 31300 | 96774 | 
| 81 | 39 docs/_ext/djangodocs.py | 26 | 70| 385 | 31685 | 99847 | 
| 82 | 40 django/contrib/staticfiles/finders.py | 70 | 93| 202 | 31887 | 101888 | 
| 83 | 41 django/middleware/common.py | 63 | 74| 117 | 32004 | 103399 | 
| 84 | 42 django/db/models/base.py | 1832 | 1856| 175 | 32179 | 118688 | 
| 85 | 43 django/core/handlers/base.py | 1 | 62| 436 | 32615 | 119866 | 
| 86 | 44 scripts/manage_translations.py | 1 | 29| 197 | 32812 | 121515 | 
| 87 | 45 django/core/management/commands/createcachetable.py | 1 | 30| 219 | 33031 | 122377 | 
| 88 | 46 django/core/management/commands/testserver.py | 29 | 55| 234 | 33265 | 122811 | 
| 89 | 47 django/core/checks/model_checks.py | 129 | 153| 268 | 33533 | 124598 | 
| 90 | 48 django/core/management/commands/test.py | 25 | 57| 260 | 33793 | 125007 | 
| 91 | 48 django/core/management/base.py | 148 | 228| 751 | 34544 | 125007 | 
| 92 | 48 django/core/checks/model_checks.py | 178 | 211| 332 | 34876 | 125007 | 
| 93 | 49 django/contrib/sessions/management/commands/clearsessions.py | 1 | 20| 122 | 34998 | 125129 | 
| 94 | 49 django/core/management/commands/testserver.py | 1 | 27| 205 | 35203 | 125129 | 
| 95 | 49 django/conf/global_settings.py | 395 | 494| 792 | 35995 | 125129 | 
| 96 | 50 django/db/backends/dummy/base.py | 1 | 47| 270 | 36265 | 125574 | 
| 97 | 50 django/core/management/templates.py | 119 | 182| 563 | 36828 | 125574 | 
| 98 | 50 django/core/management/commands/makemessages.py | 362 | 391| 231 | 37059 | 125574 | 
| 99 | 50 django/core/checks/security/base.py | 85 | 180| 710 | 37769 | 125574 | 
| 100 | 51 django/contrib/staticfiles/management/commands/collectstatic.py | 147 | 205| 503 | 38272 | 128418 | 
| 101 | 52 django/core/management/commands/sqlmigrate.py | 32 | 69| 371 | 38643 | 129050 | 
| 102 | 52 django/contrib/staticfiles/finders.py | 1 | 17| 110 | 38753 | 129050 | 
| 103 | 53 django/contrib/admin/__init__.py | 1 | 30| 286 | 39039 | 129336 | 
| 104 | 54 django/utils/formats.py | 1 | 57| 377 | 39416 | 131428 | 
| 105 | 54 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 39700 | 131428 | 
| 106 | 54 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 39839 | 131428 | 
| 107 | 54 django/core/management/commands/loaddata.py | 150 | 215| 584 | 40423 | 131428 | 
| 108 | 55 django/apps/config.py | 204 | 217| 117 | 40540 | 133160 | 
| 109 | 55 django/apps/registry.py | 61 | 125| 438 | 40978 | 133160 | 
| 110 | 55 django/conf/global_settings.py | 344 | 394| 826 | 41804 | 133160 | 
| 111 | **55 django/core/management/__init__.py** | 76 | 168| 772 | 42576 | 133160 | 
| 112 | 56 django/db/migrations/loader.py | 275 | 299| 205 | 42781 | 136051 | 
| 113 | 57 django/core/management/commands/dbshell.py | 1 | 32| 231 | 43012 | 136282 | 
| 114 | 57 django/core/management/commands/migrate.py | 243 | 257| 170 | 43182 | 136282 | 
| 115 | 57 django/db/utils.py | 166 | 185| 188 | 43370 | 136282 | 
| 116 | 57 django/views/debug.py | 48 | 69| 160 | 43530 | 136282 | 
| 117 | 58 django/contrib/admindocs/views.py | 1 | 29| 216 | 43746 | 139592 | 
| 118 | 59 django/contrib/postgres/apps.py | 20 | 37| 188 | 43934 | 140158 | 
| 119 | 59 django/core/management/commands/test.py | 1 | 23| 154 | 44088 | 140158 | 
| 120 | 60 django/templatetags/l10n.py | 41 | 64| 190 | 44278 | 140600 | 
| 121 | 60 django/core/management/commands/loaddata.py | 217 | 273| 549 | 44827 | 140600 | 
| 122 | 60 django/db/models/options.py | 415 | 437| 154 | 44981 | 140600 | 
| 123 | 60 django/contrib/auth/checks.py | 97 | 167| 525 | 45506 | 140600 | 
| 124 | 60 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 46013 | 140600 | 
| 125 | **60 django/core/management/__init__.py** | 171 | 213| 343 | 46356 | 140600 | 
| 126 | 61 django/contrib/admin/utils.py | 285 | 303| 175 | 46531 | 144688 | 
| 127 | 62 django/utils/autoreload.py | 79 | 95| 161 | 46692 | 149405 | 
| 128 | 63 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 47014 | 149727 | 
| 129 | 63 django/contrib/staticfiles/management/commands/collectstatic.py | 1 | 35| 215 | 47229 | 149727 | 
| 130 | 63 django/db/models/base.py | 1859 | 1910| 351 | 47580 | 149727 | 
| 131 | 63 django/db/models/base.py | 1 | 49| 320 | 47900 | 149727 | 
| 132 | 64 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 48404 | 150993 | 
| 133 | 65 django/db/migrations/questioner.py | 1 | 54| 468 | 48872 | 153067 | 
| 134 | 65 django/core/management/base.py | 39 | 61| 205 | 49077 | 153067 | 
| 135 | 65 django/db/utils.py | 150 | 164| 145 | 49222 | 153067 | 
| 136 | 65 django/contrib/auth/management/commands/createsuperuser.py | 204 | 228| 204 | 49426 | 153067 | 
| 137 | 66 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 1 | 81| 598 | 50024 | 153685 | 
| 138 | 67 django/views/csrf.py | 1 | 13| 132 | 50156 | 155229 | 
| 139 | 68 django/contrib/auth/hashers.py | 1 | 27| 187 | 50343 | 160016 | 
| 140 | 69 django/db/models/__init__.py | 1 | 51| 576 | 50919 | 160592 | 
| 141 | 69 django/core/management/commands/runserver.py | 23 | 52| 240 | 51159 | 160592 | 
| 142 | 69 django/db/models/options.py | 387 | 413| 175 | 51334 | 160592 | 
| 143 | 69 django/core/checks/model_checks.py | 89 | 110| 168 | 51502 | 160592 | 
| 144 | 69 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 51651 | 160592 | 
| 145 | 70 django/db/backends/oracle/base.py | 38 | 60| 235 | 51886 | 165681 | 
| 146 | 71 django/contrib/auth/models.py | 187 | 224| 249 | 52135 | 168872 | 
| 147 | 71 django/core/management/commands/createcachetable.py | 32 | 44| 121 | 52256 | 168872 | 
| 148 | 71 django/core/management/commands/makemessages.py | 215 | 280| 633 | 52889 | 168872 | 
| 149 | **71 django/core/management/__init__.py** | 41 | 73| 265 | 53154 | 168872 | 
| 150 | 71 django/contrib/staticfiles/management/commands/collectstatic.py | 37 | 68| 297 | 53451 | 168872 | 
| 151 | 71 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 53743 | 168872 | 
| 152 | 71 django/apps/registry.py | 127 | 145| 166 | 53909 | 168872 | 
| 153 | 72 django/contrib/auth/middleware.py | 1 | 24| 193 | 54102 | 169888 | 
| 154 | 72 django/core/checks/model_checks.py | 155 | 176| 263 | 54365 | 169888 | 
| 155 | 73 django/contrib/admin/options.py | 1 | 96| 769 | 55134 | 188254 | 
| 156 | 73 django/core/management/commands/migrate.py | 1 | 18| 148 | 55282 | 188254 | 
| 157 | 73 django/core/management/commands/runserver.py | 1 | 20| 191 | 55473 | 188254 | 
| 158 | 73 django/core/management/commands/dumpdata.py | 170 | 194| 224 | 55697 | 188254 | 
| 159 | 73 django/core/management/base.py | 486 | 519| 291 | 55988 | 188254 | 
| 160 | 73 django/core/management/base.py | 297 | 345| 381 | 56369 | 188254 | 
| 161 | 74 django/core/management/commands/sqlflush.py | 1 | 26| 194 | 56563 | 188448 | 
| 162 | 75 django/core/management/commands/inspectdb.py | 1 | 36| 272 | 56835 | 191065 | 
| 163 | 76 django/contrib/auth/admin.py | 1 | 22| 188 | 57023 | 192791 | 
| 164 | 77 django/db/backends/base/base.py | 204 | 281| 543 | 57566 | 197649 | 
| 165 | 78 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 57818 | 197902 | 
| 166 | 78 django/core/management/commands/flush.py | 1 | 25| 206 | 58024 | 197902 | 
| 167 | 78 django/contrib/admin/checks.py | 57 | 126| 588 | 58612 | 197902 | 


### Hint

```
This is some sensible part of code which was altered a few times already. I guess the test suite should tell you if you can use settings.configured in that line.
Hi Keryn. startproject and startapp (at least) can bypass that if settings.configured check in execute() that you link, so we can get to fetch_command() without them in play. As it is, I take re-triggering of the You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() error to be targeting the usual case. But, I think you're right, the message is wrong: No Django settings specified. — you might well have manually called settings.configure(). (So a small clean up there.) ​PR.
Apologies, I've clearly not explained well enough, because whilst the PR makes more sense now for what the code does, it's not what I was driving at. Whether or not the if settings.configured: check has been performed previously (wrt to having been run via django-admin to do things like startproject or whatever), that check seems correct, where the check for the enrivonment variable may not be (ignoring that passing --settings may ultimately set the env var anyway via handle_default_options etc?). Consider, the default manage.py, trimmed down to the minimum: import os import sys os.environ.setdefault('DJANGO_SETTINGS_MODULE', 't30872.settings') from django.core.management import execute_from_command_line execute_from_command_line(sys.argv) using that to do something like python manage.py invalid_command should give back something like: Unknown command: 'invalid_command' Type 'manage.py help' for usage. However, a simple change: import os import sys from django.conf import settings settings.configure( DEBUG=True, INSTALLED_APPS=(), SECRET_KEY='30872' ) from django.core.management import execute_from_command_line execute_from_command_line(sys.argv) and re-running, gets this output: No Django settings specified. Unknown command: 'invalid_command' Type 'manage.py help' for usage. As far as I'm aware, ignoring TZ shenanigans, whether settings come via a module and are internally a Settings, or are manually configured and come as UserSettingsHolder is an implementation choice, and ultimately if settings.configured: ought to suffice for testing whether some form of bootstrapping has been provided (because ultimately it just checks _wrapped, which is one of the aforementioned class instances). Ultimately what I'm trying to suggest is that the occurance of the stdout No Django settings specified at all is the problem, and not the messaging itself, which is currently vague enough to be correct. As a corollary, the only other time I can see DJANGO_SETTINGS_MODULE being referenced in a similar way (searching only django.core.management) is in compilemessages, where again it's used as a stand-in for testing if settings are usable.
As I read it, the check is to catch the (I'd guess) common enough, error of providing a DJANGO_SETTINGS_MODULE, maybe via --settings, but misspelling it or such. Then we want to re-trigger the settings error (which is swallowed twice before). (This is also why we can't just check settings.configured, because that'll be False but not re-trigger the error.) The else is for the other (I'd guess) common error of having an app command, but forgetting to provide settings, so it doesn't get found. Then we have the (how common?) case of using settings.configure() with execute_from_command_line() and misspelling a command (despite autocomplete()) and then getting a slightly misleading message. For that we could do something like this: diff --git a/django/core/management/__init__.py b/django/core/management/__init__.py index adc7d173eb..cf6b60c93e 100644 --- a/django/core/management/__init__.py +++ b/django/core/management/__init__.py @@ -229,7 +229,7 @@ class ManagementUtility: # (get_commands() swallows the original one) so the user is # informed about it. settings.INSTALLED_APPS - else: + elif not settings.configured: sys.stderr.write("No Django settings specified.\n") possible_matches = get_close_matches(subcommand, commands) sys.stderr.write('Unknown command: %r' % subcommand) That passes the existing tests. Is it what you're after? If so I can write a test case for it on Tuesday.
This is also why we can't just check settings.configured, because that'll be False but not re-trigger the error. I think that's the crux of what I wasn't understanding. I'm guessing that always testing for settings.INSTALLED_APPS (ie: without being a in conditional branch) is some other use-case that won't work for all scenarios. Then we have the (how common?) case of using settings.configure() with execute_from_command_line() and misspelling a command (despite autocomplete()) and then getting a slightly misleading message [...] Yeah, I fully appreciate that it's an edge-case, and I'm certainly not going to push hard on it being done (it's been there for however many years already), and you definitely don't need to commit to a specific day(!) on which to do it, but it would seem that what you've proposed would quash my quibble :) [minor aside: I'd wager the usage of autocomplete is also vanishingly small, given it requires additional steps]
```

## Patch

```diff
diff --git a/django/core/management/__init__.py b/django/core/management/__init__.py
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -229,7 +229,7 @@ def fetch_command(self, subcommand):
                 # (get_commands() swallows the original one) so the user is
                 # informed about it.
                 settings.INSTALLED_APPS
-            else:
+            elif not settings.configured:
                 sys.stderr.write("No Django settings specified.\n")
             possible_matches = get_close_matches(subcommand, commands)
             sys.stderr.write('Unknown command: %r' % subcommand)

```

## Test Patch

```diff
diff --git a/tests/admin_scripts/tests.py b/tests/admin_scripts/tests.py
--- a/tests/admin_scripts/tests.py
+++ b/tests/admin_scripts/tests.py
@@ -634,6 +634,15 @@ def test_builtin_with_environment(self):
 # of the generated manage.py script
 ##########################################################################
 
+class ManageManullyConfiguredSettings(AdminScriptTestCase):
+    """Customized manage.py calling settings.configure()."""
+    def test_non_existent_command_output(self):
+        out, err = self.run_manage(['invalid_command'], manage_py='configured_settings_manage.py')
+        self.assertNoOutput(out)
+        self.assertOutput(err, "Unknown command: 'invalid_command'")
+        self.assertNotInOutput(err, 'No Django settings specified')
+
+
 class ManageNoSettings(AdminScriptTestCase):
     "A series of tests for manage.py when there is no settings.py file."
 

```


## Code snippets

### 1 - django/core/management/__init__.py:

Start line: 215, End line: 245

```python
class ManagementUtility:

    def fetch_command(self, subcommand):
        """
        Try to fetch the given subcommand, printing a message with the
        appropriate command called from the command line (usually
        "django-admin" or "manage.py") if it can't be found.
        """
        # Get commands outside of try block to prevent swallowing exceptions
        commands = get_commands()
        try:
            app_name = commands[subcommand]
        except KeyError:
            if os.environ.get('DJANGO_SETTINGS_MODULE'):
                # If `subcommand` is missing due to misconfigured settings, the
                # following line will retrigger an ImproperlyConfigured exception
                # (get_commands() swallows the original one) so the user is
                # informed about it.
                settings.INSTALLED_APPS
            else:
                sys.stderr.write("No Django settings specified.\n")
            possible_matches = get_close_matches(subcommand, commands)
            sys.stderr.write('Unknown command: %r' % subcommand)
            if possible_matches:
                sys.stderr.write('. Did you mean %s?' % possible_matches[0])
            sys.stderr.write("\nType '%s help' for usage.\n" % self.prog_name)
            sys.exit(1)
        if isinstance(app_name, BaseCommand):
            # If the command is already loaded, use it directly.
            klass = app_name
        else:
            klass = load_command_class(app_name, subcommand)
        return klass
```
### 2 - django/conf/__init__.py:

Start line: 43, End line: 64

```python
class LazySettings(LazyObject):
    """
    A lazy proxy for either global Django settings or a custom settings object.
    The user can manually configure settings prior to using them. Otherwise,
    Django uses the settings module pointed to by DJANGO_SETTINGS_MODULE.
    """
    def _setup(self, name=None):
        """
        Load the settings module pointed to by the environment variable. This
        is used the first time settings are needed, if the user hasn't
        configured settings manually.
        """
        settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
        if not settings_module:
            desc = ("setting %s" % name) if name else "settings"
            raise ImproperlyConfigured(
                "Requested %s, but settings are not configured. "
                "You must either define the environment variable %s "
                "or call settings.configure() before accessing settings."
                % (desc, ENVIRONMENT_VARIABLE))

        self._wrapped = Settings(settings_module)
```
### 3 - django/core/management/commands/diffsettings.py:

Start line: 41, End line: 55

```python
class Command(BaseCommand):

    def handle(self, **options):
        from django.conf import settings, Settings, global_settings

        # Because settings are imported lazily, we need to explicitly load them.
        if not settings.configured:
            settings._setup()

        user_settings = module_to_dict(settings._wrapped)
        default = options['default']
        default_settings = module_to_dict(Settings(default) if default else global_settings)
        output_func = {
            'hash': self.output_hash,
            'unified': self.output_unified,
        }[options['output']]
        return '\n'.join(output_func(user_settings, default_settings, **options))
```
### 4 - django/conf/__init__.py:

Start line: 98, End line: 111

```python
class LazySettings(LazyObject):

    def configure(self, default_settings=global_settings, **options):
        """
        Called to manually configure the settings. The 'default_settings'
        parameter sets where to retrieve any unspecified values from (its
        argument must support attribute access (__getattr__)).
        """
        if self._wrapped is not empty:
            raise RuntimeError('Settings already configured.')
        holder = UserSettingsHolder(default_settings)
        for name, value in options.items():
            if not name.isupper():
                raise TypeError('Setting %r must be uppercase.' % name)
            setattr(holder, name, value)
        self._wrapped = holder
```
### 5 - django/conf/__init__.py:

Start line: 223, End line: 273

```python
class UserSettingsHolder:
    """Holder for user configured settings."""
    # SETTINGS_MODULE doesn't make much sense in the manually configured
    # (standalone) case.
    SETTINGS_MODULE = None

    def __init__(self, default_settings):
        """
        Requests for configuration variables not in this class are satisfied
        from the module specified in default_settings (if possible).
        """
        self.__dict__['_deleted'] = set()
        self.default_settings = default_settings

    def __getattr__(self, name):
        if not name.isupper() or name in self._deleted:
            raise AttributeError
        return getattr(self.default_settings, name)

    def __setattr__(self, name, value):
        self._deleted.discard(name)
        if name == 'PASSWORD_RESET_TIMEOUT_DAYS':
            setattr(self, 'PASSWORD_RESET_TIMEOUT', value * 60 * 60 * 24)
            warnings.warn(PASSWORD_RESET_TIMEOUT_DAYS_DEPRECATED_MSG, RemovedInDjango40Warning)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self._deleted.add(name)
        if hasattr(self, name):
            super().__delattr__(name)

    def __dir__(self):
        return sorted(
            s for s in [*self.__dict__, *dir(self.default_settings)]
            if s not in self._deleted
        )

    def is_overridden(self, setting):
        deleted = (setting in self._deleted)
        set_locally = (setting in self.__dict__)
        set_on_default = getattr(self.default_settings, 'is_overridden', lambda s: False)(setting)
        return deleted or set_locally or set_on_default

    def __repr__(self):
        return '<%(cls)s>' % {
            'cls': self.__class__.__name__,
        }


settings = LazySettings()
```
### 6 - django/core/management/commands/diffsettings.py:

Start line: 69, End line: 80

```python
class Command(BaseCommand):

    def output_unified(self, user_settings, default_settings, **options):
        output = []
        for key in sorted(user_settings):
            if key not in default_settings:
                output.append(self.style.SUCCESS("+ %s = %s" % (key, user_settings[key])))
            elif user_settings[key] != default_settings[key]:
                output.append(self.style.ERROR("- %s = %s" % (key, default_settings[key])))
                output.append(self.style.SUCCESS("+ %s = %s" % (key, user_settings[key])))
            elif options['all']:
                output.append("  %s = %s" % (key, user_settings[key]))
        return output
```
### 7 - django/conf/__init__.py:

Start line: 161, End line: 220

```python
class Settings:
    def __init__(self, settings_module):
        # update this dict from global settings (but only for ALL_CAPS settings)
        for setting in dir(global_settings):
            if setting.isupper():
                setattr(self, setting, getattr(global_settings, setting))

        # store the settings module in case someone later cares
        self.SETTINGS_MODULE = settings_module

        mod = importlib.import_module(self.SETTINGS_MODULE)

        tuple_settings = (
            "INSTALLED_APPS",
            "TEMPLATE_DIRS",
            "LOCALE_PATHS",
        )
        self._explicit_settings = set()
        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)

                if (setting in tuple_settings and
                        not isinstance(setting_value, (list, tuple))):
                    raise ImproperlyConfigured("The %s setting must be a list or a tuple. " % setting)
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

        if not self.SECRET_KEY:
            raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")

        if self.is_overridden('PASSWORD_RESET_TIMEOUT_DAYS'):
            if self.is_overridden('PASSWORD_RESET_TIMEOUT'):
                raise ImproperlyConfigured(
                    'PASSWORD_RESET_TIMEOUT_DAYS/PASSWORD_RESET_TIMEOUT are '
                    'mutually exclusive.'
                )
            setattr(self, 'PASSWORD_RESET_TIMEOUT', self.PASSWORD_RESET_TIMEOUT_DAYS * 60 * 60 * 24)
            warnings.warn(PASSWORD_RESET_TIMEOUT_DAYS_DEPRECATED_MSG, RemovedInDjango40Warning)

        if hasattr(time, 'tzset') and self.TIME_ZONE:
            # When we can, attempt to validate the timezone. If we can't find
            # this file, no check happens and it's harmless.
            zoneinfo_root = Path('/usr/share/zoneinfo')
            zone_info_file = zoneinfo_root.joinpath(*self.TIME_ZONE.split('/'))
            if zoneinfo_root.exists() and not zone_info_file.exists():
                raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
            # Move the time zone info into os.environ. See ticket #2315 for why
            # we don't do this unconditionally (breaks Windows).
            os.environ['TZ'] = self.TIME_ZONE
            time.tzset()

    def is_overridden(self, setting):
        return setting in self._explicit_settings

    def __repr__(self):
        return '<%(cls)s "%(settings_module)s">' % {
            'cls': self.__class__.__name__,
            'settings_module': self.SETTINGS_MODULE,
        }
```
### 8 - django/core/management/__init__.py:

Start line: 321, End line: 402

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
        parser = CommandParser(usage='%(prog)s subcommand [options] [args]', add_help=False, allow_abbrev=False)
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
### 9 - django/conf/__init__.py:

Start line: 113, End line: 136

```python
class LazySettings(LazyObject):

    @staticmethod
    def _add_script_prefix(value):
        """
        Add SCRIPT_NAME prefix to relative paths.

        Useful when the app is being served at a subpath and manually prefixing
        subpath to STATIC_URL and MEDIA_URL in settings is inconvenient.
        """
        # Don't apply prefix to valid URLs.
        try:
            URLValidator()(value)
            return value
        except (ValidationError, AttributeError):
            pass
        # Don't apply prefix to absolute paths.
        if value.startswith('/'):
            return value
        from django.urls import get_script_prefix
        return '%s%s' % (get_script_prefix(), value)

    @property
    def configured(self):
        """Return True if the settings have already been configured."""
        return self._wrapped is not empty
```
### 10 - django/conf/__init__.py:

Start line: 66, End line: 96

```python
class LazySettings(LazyObject):

    def __repr__(self):
        # Hardcode the class name as otherwise it yields 'Settings'.
        if self._wrapped is empty:
            return '<LazySettings [Unevaluated]>'
        return '<LazySettings "%(settings_module)s">' % {
            'settings_module': self._wrapped.SETTINGS_MODULE,
        }

    def __getattr__(self, name):
        """Return the value of a setting and cache it in self.__dict__."""
        if self._wrapped is empty:
            self._setup(name)
        val = getattr(self._wrapped, name)
        self.__dict__[name] = val
        return val

    def __setattr__(self, name, value):
        """
        Set the value of setting. Clear all cached values if _wrapped changes
        (@override_settings does this) or clear single values when set.
        """
        if name == '_wrapped':
            self.__dict__.clear()
        else:
            self.__dict__.pop(name, None)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        """Delete a setting and clear it from cache if needed."""
        super().__delattr__(name)
        self.__dict__.pop(name, None)
```
### 21 - django/core/management/__init__.py:

Start line: 247, End line: 319

```python
class ManagementUtility:

    def autocomplete(self):
        """
        Output completion suggestions for BASH.

        The output of this function is passed to BASH's `COMREPLY` variable and
        treated as completion suggestions. `COMREPLY` expects a space
        separated string as the result.

        The `COMP_WORDS` and `COMP_CWORD` BASH environment variables are used
        to get information about the cli input. Please refer to the BASH
        man-page for more information about this variables.

        Subcommand options are saved as pairs. A pair consists of
        the long option string (e.g. '--exclude') and a boolean
        value indicating if the option requires arguments. When printing to
        stdout, an equal sign is appended to options which require arguments.

        Note: If debugging this function, it is recommended to write the debug
        output in a separate file. Otherwise the debug output will be treated
        and formatted as potential completion suggestions.
        """
        # Don't complete if user hasn't sourced bash_completion file.
        if 'DJANGO_AUTO_COMPLETE' not in os.environ:
            return

        cwords = os.environ['COMP_WORDS'].split()[1:]
        cword = int(os.environ['COMP_CWORD'])

        try:
            curr = cwords[cword - 1]
        except IndexError:
            curr = ''

        subcommands = [*get_commands(), 'help']
        options = [('--help', False)]

        # subcommand
        if cword == 1:
            print(' '.join(sorted(filter(lambda x: x.startswith(curr), subcommands))))
        # subcommand options
        # special case: the 'help' subcommand has no options
        elif cwords[0] in subcommands and cwords[0] != 'help':
            subcommand_cls = self.fetch_command(cwords[0])
            # special case: add the names of installed apps to options
            if cwords[0] in ('dumpdata', 'sqlmigrate', 'sqlsequencereset', 'test'):
                try:
                    app_configs = apps.get_app_configs()
                    # Get the last part of the dotted path as the app name.
                    options.extend((app_config.label, 0) for app_config in app_configs)
                except ImportError:
                    # Fail silently if DJANGO_SETTINGS_MODULE isn't set. The
                    # user will find out once they execute the command.
                    pass
            parser = subcommand_cls.create_parser('', cwords[0])
            options.extend(
                (min(s_opt.option_strings), s_opt.nargs != 0)
                for s_opt in parser._actions if s_opt.option_strings
            )
            # filter out previously specified options from available options
            prev_opts = {x.split('=')[0] for x in cwords[1:cword - 1]}
            options = (opt for opt in options if opt[0] not in prev_opts)

            # filter options by current input
            options = sorted((k, v) for k, v in options if k.startswith(curr))
            for opt_label, require_arg in options:
                # append '=' to options which require args
                if require_arg:
                    opt_label += '='
                print(opt_label)
        # Exit code of the bash completion function is never passed back to
        # the user, so it's safe to always exit with 0.
        # For more details see #25420.
        sys.exit(0)
```
### 53 - django/core/management/__init__.py:

Start line: 1, End line: 38

```python
import functools
import os
import pkgutil
import sys
from argparse import _SubParsersAction
from collections import defaultdict
from difflib import get_close_matches
from importlib import import_module

import django
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import (
    BaseCommand, CommandError, CommandParser, handle_default_options,
)
from django.core.management.color import color_style
from django.utils import autoreload


def find_commands(management_dir):
    """
    Given a path to a management directory, return a list of all the command
    names that are available.
    """
    command_dir = os.path.join(management_dir, 'commands')
    return [name for _, name, is_pkg in pkgutil.iter_modules([command_dir])
            if not is_pkg and not name.startswith('_')]


def load_command_class(app_name, name):
    """
    Given a command name and an application name, return the Command
    class instance. Allow all errors raised by the import process
    (ImportError, AttributeError) to propagate.
    """
    module = import_module('%s.management.commands.%s' % (app_name, name))
    return module.Command()
```
### 111 - django/core/management/__init__.py:

Start line: 76, End line: 168

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
    parse_args = [str(a) for a in args]

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
    parse_args += [
        '{}={}'.format(min(opt.option_strings), arg_options[opt.dest])
        for opt in parser_actions if (
            opt.dest in options and
            (opt.required or opt in mutually_exclusive_required_options)
        )
    ]
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
### 125 - django/core/management/__init__.py:

Start line: 171, End line: 213

```python
class ManagementUtility:
    """
    Encapsulate the logic of the django-admin and manage.py utilities.
    """
    def __init__(self, argv=None):
        self.argv = argv or sys.argv[:]
        self.prog_name = os.path.basename(self.argv[0])
        if self.prog_name == '__main__.py':
            self.prog_name = 'python -m django'
        self.settings_exception = None

    def main_help_text(self, commands_only=False):
        """Return the script's main help text, as a string."""
        if commands_only:
            usage = sorted(get_commands())
        else:
            usage = [
                "",
                "Type '%s help <subcommand>' for help on a specific subcommand." % self.prog_name,
                "",
                "Available subcommands:",
            ]
            commands_dict = defaultdict(lambda: [])
            for name, app in get_commands().items():
                if app == 'django.core':
                    app = 'django'
                else:
                    app = app.rpartition('.')[-1]
                commands_dict[app].append(name)
            style = color_style()
            for app in sorted(commands_dict):
                usage.append("")
                usage.append(style.NOTICE("[%s]" % app))
                for name in sorted(commands_dict[app]):
                    usage.append("    %s" % name)
            # Output an extra note if settings are not properly configured
            if self.settings_exception is not None:
                usage.append(style.NOTICE(
                    "Note that only Django core commands are listed "
                    "as settings are not properly configured (error: %s)."
                    % self.settings_exception))

        return '\n'.join(usage)
```
### 149 - django/core/management/__init__.py:

Start line: 41, End line: 73

```python
@functools.lru_cache(maxsize=None)
def get_commands():
    """
    Return a dictionary mapping command names to their callback applications.

    Look for a management.commands package in django.core, and in each
    installed application -- if a commands package exists, register all
    commands in that package.

    Core commands are always included. If a settings module has been
    specified, also include user-defined commands.

    The dictionary is in the format {command_name: app_name}. Key-value
    pairs from this dictionary can then be used in calls to
    load_command_class(app_name, command_name)

    If a specific version of a command must be loaded (e.g., with the
    startapp command), the instantiated module can be placed in the
    dictionary in place of the application name.

    The dictionary is cached on the first call and reused on subsequent
    calls.
    """
    commands = {name: 'django.core' for name in find_commands(__path__[0])}

    if not settings.configured:
        return commands

    for app_config in reversed(list(apps.get_app_configs())):
        path = os.path.join(app_config.path, 'management')
        commands.update({name: app_config.name for name in find_commands(path)})

    return commands
```
