# django__django-5470

| **django/django** | `9dcfecb7c6c8285630ad271888a9ec4ba9140e3a` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 496 |
| **Any found context length** | 496 |
| **Avg pos** | 53.0 |
| **Min pos** | 2 |
| **Max pos** | 102 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/__init__.py b/django/__init__.py
--- a/django/__init__.py
+++ b/django/__init__.py
@@ -1,3 +1,5 @@
+from __future__ import unicode_literals
+
 from django.utils.version import get_version
 
 VERSION = (1, 10, 0, 'alpha', 0)
@@ -5,14 +7,21 @@
 __version__ = get_version(VERSION)
 
 
-def setup():
+def setup(set_prefix=True):
     """
     Configure the settings (this happens as a side effect of accessing the
     first setting), configure logging and populate the app registry.
+    Set the thread-local urlresolvers script prefix if `set_prefix` is True.
     """
     from django.apps import apps
     from django.conf import settings
+    from django.core.urlresolvers import set_script_prefix
+    from django.utils.encoding import force_text
     from django.utils.log import configure_logging
 
     configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
+    if set_prefix:
+        set_script_prefix(
+            '/' if settings.FORCE_SCRIPT_NAME is None else force_text(settings.FORCE_SCRIPT_NAME)
+        )
     apps.populate(settings.INSTALLED_APPS)
diff --git a/django/core/wsgi.py b/django/core/wsgi.py
--- a/django/core/wsgi.py
+++ b/django/core/wsgi.py
@@ -10,5 +10,5 @@ def get_wsgi_application():
     Allows us to avoid making django.core.handlers.WSGIHandler public API, in
     case the internal WSGI implementation changes or moves in the future.
     """
-    django.setup()
+    django.setup(set_prefix=False)
     return WSGIHandler()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/__init__.py | 1 | 1 | 2 | 2 | 496
| django/__init__.py | 8 | 8 | 2 | 2 | 496
| django/core/wsgi.py | 13 | 13 | 102 | 60 | 34927


## Problem Statement

```
Set script prefix in django.setup() to allow its usage outside of requests
Description
	
The script prefix for django.core.urlresolvers doesn't get set to anything when being called through manage.py, because of course it doesn't know what that value should be. This is a problem if you're rendering views (or otherwise reversing urls) from a manage.py command (as one of my sites does to send emails).
This is solvable by calling set_script_prefix from settings.py, but that feels kind of dirty since it's then about to be rewritten in the WSGI handler.
I don't know what a good solution to this would be. Perhaps it would be nice to be able to set a global default script path somewhere that would then get incorporated into the default values of things like LOGIN_URL.
Maybe just a note in the documentation would be good. It took me a while to figure out, because I haven't been able to find anything else about this online. (I guess that non-/ script paths are uncommon and reversing urls from manage.py is also uncommon, so both together are very uncommon.)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/urlresolvers.py | 548 | 617| 384 | 384 | 4946 | 
| **-> 2 <-** | **2 django/__init__.py** | 1 | 19| 112 | 496 | 5058 | 
| 3 | 3 django/conf/global_settings.py | 137 | 248| 826 | 1322 | 10494 | 
| 4 | 3 django/conf/global_settings.py | 385 | 491| 805 | 2127 | 10494 | 
| 5 | 3 django/conf/global_settings.py | 1 | 48| 391 | 2518 | 10494 | 
| 6 | 4 django/core/handlers/wsgi.py | 1 | 25| 152 | 2670 | 12579 | 
| 7 | 5 django/conf/__init__.py | 87 | 138| 456 | 3126 | 13981 | 
| 8 | 6 setup.py | 1 | 95| 693 | 3819 | 14661 | 
| 9 | 6 django/conf/global_settings.py | 492 | 617| 804 | 4623 | 14661 | 
| 10 | 6 django/core/handlers/wsgi.py | 80 | 146| 577 | 5200 | 14661 | 
| 11 | 7 django/contrib/auth/admin.py | 1 | 24| 204 | 5404 | 16401 | 
| 12 | 7 django/conf/global_settings.py | 249 | 334| 807 | 6211 | 16401 | 
| 13 | 7 django/conf/__init__.py | 1 | 41| 300 | 6511 | 16401 | 
| 14 | 7 django/core/handlers/wsgi.py | 203 | 233| 342 | 6853 | 16401 | 
| 15 | 8 django/contrib/admin/templatetags/admin_urls.py | 1 | 17| 109 | 6962 | 16810 | 
| 16 | 9 django/middleware/common.py | 1 | 39| 300 | 7262 | 18312 | 
| 17 | 10 django/contrib/auth/urls.py | 1 | 20| 250 | 7512 | 18562 | 
| 18 | 11 django/conf/project_template/project_name/settings.py | 1 | 121| 663 | 8175 | 19225 | 
| 19 | 12 django/views/debug.py | 504 | 1041| 211 | 8386 | 30543 | 
| 20 | 13 django/contrib/auth/views.py | 1 | 27| 234 | 8620 | 32835 | 
| 21 | 14 django/contrib/admin/utils.py | 59 | 73| 139 | 8759 | 36468 | 
| 22 | 15 django/contrib/admin/sites.py | 1 | 27| 164 | 8923 | 40699 | 
| 23 | 15 django/core/urlresolvers.py | 483 | 546| 446 | 9369 | 40699 | 
| 24 | 16 django/conf/urls/__init__.py | 1 | 16| 131 | 9500 | 41445 | 
| 25 | 17 django/__main__.py | 1 | 10| 45 | 9545 | 41490 | 
| 26 | 17 django/contrib/admin/sites.py | 232 | 251| 222 | 9767 | 41490 | 
| 27 | 17 django/views/debug.py | 1203 | 1253| 478 | 10245 | 41490 | 
| 28 | 17 django/conf/__init__.py | 43 | 84| 331 | 10576 | 41490 | 
| 29 | 18 django/core/handlers/base.py | 1 | 22| 129 | 10705 | 43894 | 
| 30 | 19 django/http/request.py | 186 | 270| 625 | 11330 | 48253 | 
| 31 | 20 django/core/management/templates.py | 1 | 22| 130 | 11460 | 50934 | 
| 32 | 20 django/middleware/common.py | 41 | 66| 233 | 11693 | 50934 | 
| 33 | 20 django/contrib/admin/utils.py | 1 | 19| 142 | 11835 | 50934 | 
| 34 | 21 django/core/servers/basehttp.py | 138 | 159| 245 | 12080 | 52661 | 
| 35 | 22 django/core/management/__init__.py | 120 | 165| 349 | 12429 | 55587 | 
| 36 | 23 docs/_ext/djangodocs.py | 22 | 69| 424 | 12853 | 58372 | 
| 37 | 24 django/contrib/admindocs/urls.py | 1 | 33| 282 | 13135 | 58654 | 
| 38 | 24 django/contrib/admin/sites.py | 253 | 299| 483 | 13618 | 58654 | 
| 39 | 25 docs/conf.py | 1 | 112| 844 | 14462 | 61486 | 
| 40 | 25 django/middleware/common.py | 68 | 79| 123 | 14585 | 61486 | 
| 41 | 26 django/core/management/commands/runserver.py | 1 | 26| 247 | 14832 | 63135 | 
| 42 | 27 django/conf/project_template/manage.py | 1 | 11| 56 | 14888 | 63191 | 
| 43 | 28 django/utils/log.py | 1 | 51| 339 | 15227 | 64278 | 
| 44 | 29 django/conf/urls/i18n.py | 1 | 21| 127 | 15354 | 64405 | 
| 45 | 29 django/core/handlers/wsgi.py | 149 | 200| 400 | 15754 | 64405 | 
| 46 | 29 django/contrib/admin/sites.py | 517 | 541| 223 | 15977 | 64405 | 
| 47 | 29 django/core/servers/basehttp.py | 95 | 136| 397 | 16374 | 64405 | 
| 48 | 30 django/contrib/admin/__init__.py | 1 | 30| 286 | 16660 | 64691 | 
| 49 | 30 django/core/management/commands/runserver.py | 62 | 104| 412 | 17072 | 64691 | 
| 50 | 30 django/core/urlresolvers.py | 400 | 454| 590 | 17662 | 64691 | 
| 51 | 30 django/conf/global_settings.py | 335 | 384| 791 | 18453 | 64691 | 
| 52 | 31 django/views/csrf.py | 1 | 14| 127 | 18580 | 65969 | 
| 53 | 31 django/middleware/common.py | 81 | 100| 204 | 18784 | 65969 | 
| 54 | 32 django/contrib/sites/managers.py | 1 | 50| 302 | 19086 | 66388 | 
| 55 | 33 scripts/manage_translations.py | 1 | 27| 191 | 19277 | 68050 | 
| 56 | 34 django/core/management/commands/shell.py | 63 | 102| 348 | 19625 | 68861 | 
| 57 | 35 django/middleware/locale.py | 29 | 72| 367 | 19992 | 69414 | 
| 58 | 36 django/contrib/admin/helpers.py | 145 | 185| 339 | 20331 | 72363 | 
| 59 | 37 django/contrib/redirects/__init__.py | 1 | 2| 14 | 20345 | 72377 | 
| 60 | 38 django/utils/six.py | 816 | 855| 409 | 20754 | 79734 | 
| 61 | 38 django/contrib/admin/utils.py | 315 | 367| 442 | 21196 | 79734 | 
| 62 | 39 django/contrib/admindocs/views.py | 1 | 25| 183 | 21379 | 83067 | 
| 63 | 40 django/core/checks/security/csrf.py | 1 | 57| 428 | 21807 | 83495 | 
| 64 | 41 django/contrib/auth/management/__init__.py | 60 | 129| 574 | 22381 | 84938 | 
| 65 | 42 django/contrib/admin/options.py | 536 | 558| 270 | 22651 | 101142 | 
| 66 | 42 django/contrib/auth/views.py | 275 | 334| 366 | 23017 | 101142 | 
| 67 | 43 django/templatetags/static.py | 54 | 87| 161 | 23178 | 102027 | 
| 68 | 43 docs/conf.py | 113 | 222| 786 | 23964 | 102027 | 
| 69 | 44 django/contrib/staticfiles/handlers.py | 1 | 65| 483 | 24447 | 102510 | 
| 70 | 44 django/core/urlresolvers.py | 36 | 77| 352 | 24799 | 102510 | 
| 71 | 44 django/views/debug.py | 1043 | 1123| 909 | 25708 | 102510 | 
| 72 | 45 django/contrib/sites/__init__.py | 1 | 2| 14 | 25722 | 102524 | 
| 73 | 46 django/core/management/base.py | 65 | 108| 297 | 26019 | 106613 | 
| 74 | 46 django/core/management/commands/runserver.py | 48 | 60| 129 | 26148 | 106613 | 
| 75 | 46 django/core/management/commands/runserver.py | 106 | 155| 476 | 26624 | 106613 | 
| 76 | 47 django/db/models/options.py | 1 | 37| 299 | 26923 | 112637 | 
| 77 | 47 django/core/management/__init__.py | 268 | 350| 727 | 27650 | 112637 | 
| 78 | 47 django/views/csrf.py | 16 | 97| 767 | 28417 | 112637 | 
| 79 | 48 django/contrib/sites/middleware.py | 1 | 11| 48 | 28465 | 112685 | 
| 80 | 49 django/core/checks/security/base.py | 1 | 86| 752 | 29217 | 114197 | 
| 81 | 50 django/contrib/sites/management.py | 1 | 39| 349 | 29566 | 114546 | 
| 82 | 51 django/contrib/sites/models.py | 30 | 53| 200 | 29766 | 115344 | 
| 83 | 52 django/shortcuts.py | 118 | 158| 292 | 30058 | 116508 | 
| 84 | 52 django/core/management/base.py | 259 | 302| 347 | 30405 | 116508 | 
| 85 | 52 django/contrib/admin/options.py | 560 | 577| 146 | 30551 | 116508 | 
| 86 | 53 django/bin/django-admin.py | 1 | 6| 27 | 30578 | 116535 | 
| 87 | 54 django/contrib/admindocs/__init__.py | 1 | 2| 15 | 30593 | 116550 | 
| 88 | 54 django/core/urlresolvers.py | 128 | 152| 199 | 30792 | 116550 | 
| 89 | 55 django/views/defaults.py | 1 | 48| 378 | 31170 | 117310 | 
| 90 | 55 django/contrib/admin/options.py | 1 | 90| 749 | 31919 | 117310 | 
| 91 | 56 django/template/backends/utils.py | 1 | 18| 118 | 32037 | 117428 | 
| 92 | 56 django/utils/log.py | 54 | 70| 132 | 32169 | 117428 | 
| 93 | 56 django/core/urlresolvers.py | 369 | 398| 264 | 32433 | 117428 | 
| 94 | 56 django/core/management/base.py | 111 | 193| 781 | 33214 | 117428 | 
| 95 | 56 django/contrib/admin/options.py | 1708 | 1734| 252 | 33466 | 117428 | 
| 96 | 56 django/core/urlresolvers.py | 1 | 33| 273 | 33739 | 117428 | 
| 97 | 56 django/core/management/commands/runserver.py | 29 | 46| 195 | 33934 | 117428 | 
| 98 | 57 django/contrib/redirects/middleware.py | 1 | 52| 337 | 34271 | 117766 | 
| 99 | 58 django/conf/project_template/project_name/urls.py | 1 | 23| 212 | 34483 | 117978 | 
| 100 | 58 django/contrib/admin/sites.py | 323 | 338| 156 | 34639 | 117978 | 
| 101 | 59 django/contrib/admin/models.py | 1 | 28| 197 | 34836 | 118670 | 
| **-> 102 <-** | **60 django/core/wsgi.py** | 1 | 15| 91 | 34927 | 118761 | 
| 103 | 61 django/views/i18n.py | 81 | 184| 703 | 35630 | 121489 | 
| 104 | 61 django/core/management/templates.py | 62 | 120| 524 | 36154 | 121489 | 
| 105 | 61 django/core/urlresolvers.py | 304 | 328| 168 | 36322 | 121489 | 
| 106 | 61 django/core/handlers/base.py | 25 | 75| 426 | 36748 | 121489 | 
| 107 | 61 django/contrib/auth/views.py | 159 | 220| 428 | 37176 | 121489 | 
| 108 | 62 django/contrib/sessions/management/commands/clearsessions.py | 1 | 20| 122 | 37298 | 121611 | 
| 109 | 62 django/core/handlers/base.py | 119 | 230| 909 | 38207 | 121611 | 
| 110 | 62 django/core/management/base.py | 194 | 230| 276 | 38483 | 121611 | 
| 111 | 62 django/contrib/admin/options.py | 262 | 326| 454 | 38937 | 121611 | 
| 112 | 63 django/core/mail/backends/__init__.py | 1 | 2| 8 | 38945 | 121619 | 
| 113 | 64 django/contrib/redirects/migrations/0001_initial.py | 1 | 44| 289 | 39234 | 121908 | 
| 114 | 65 django/contrib/admindocs/utils.py | 1 | 33| 244 | 39478 | 122877 | 
| 115 | 66 django/contrib/auth/__init__.py | 1 | 59| 402 | 39880 | 124352 | 
| 116 | 67 django/contrib/sites/apps.py | 1 | 14| 73 | 39953 | 124425 | 
| 117 | 68 django/contrib/sessions/__init__.py | 1 | 2| 13 | 39966 | 124438 | 
| 118 | 68 django/core/management/templates.py | 308 | 329| 161 | 40127 | 124438 | 
| 119 | 69 django/core/management/commands/loaddata.py | 1 | 30| 168 | 40295 | 127030 | 
| 120 | 69 django/contrib/admin/options.py | 1174 | 1190| 167 | 40462 | 127030 | 
| 121 | 70 django/middleware/security.py | 1 | 25| 221 | 40683 | 127408 | 
| 122 | 71 django/contrib/sessions/models.py | 1 | 38| 258 | 40941 | 127666 | 
| 123 | 71 django/contrib/admin/options.py | 1157 | 1172| 167 | 41108 | 127666 | 
| 124 | 71 django/views/debug.py | 1125 | 1201| 688 | 41796 | 127666 | 
| 125 | 72 django/contrib/auth/middleware.py | 1 | 35| 259 | 42055 | 128746 | 
| 126 | 72 django/shortcuts.py | 1 | 34| 270 | 42325 | 128746 | 
| 127 | 73 django/utils/autoreload.py | 1 | 79| 293 | 42618 | 131087 | 
| 128 | 73 django/contrib/auth/admin.py | 43 | 97| 477 | 43095 | 131087 | 
| 129 | 74 django/contrib/auth/management/commands/changepassword.py | 31 | 75| 346 | 43441 | 131638 | 
| 130 | 74 django/core/urlresolvers.py | 260 | 302| 380 | 43821 | 131638 | 
| 131 | 74 django/middleware/locale.py | 1 | 27| 190 | 44011 | 131638 | 
| 132 | 75 django/contrib/admindocs/middleware.py | 1 | 25| 235 | 44246 | 131874 | 
| 133 | 75 django/contrib/auth/admin.py | 130 | 189| 463 | 44709 | 131874 | 
| 134 | 76 django/core/management/commands/startproject.py | 1 | 34| 255 | 44964 | 132129 | 
| 135 | 76 django/contrib/auth/middleware.py | 58 | 94| 357 | 45321 | 132129 | 
| 136 | 77 django/contrib/auth/management/commands/createsuperuser.py | 1 | 21| 114 | 45435 | 133665 | 
| 137 | 77 django/core/management/templates.py | 122 | 178| 499 | 45934 | 133665 | 
| 138 | 78 django/contrib/humanize/__init__.py | 1 | 2| 16 | 45950 | 133681 | 
| 139 | 79 django/contrib/sessions/middleware.py | 1 | 58| 448 | 46398 | 134130 | 
| 140 | 80 django/db/utils.py | 1 | 50| 186 | 46584 | 136242 | 
| 141 | 80 django/core/management/templates.py | 227 | 279| 403 | 46987 | 136242 | 
| 142 | 80 django/contrib/admin/sites.py | 175 | 207| 321 | 47308 | 136242 | 
| 143 | 80 django/contrib/admin/sites.py | 30 | 65| 296 | 47604 | 136242 | 
| 144 | 80 docs/conf.py | 224 | 365| 987 | 48591 | 136242 | 
| 145 | 81 django/middleware/csrf.py | 1 | 40| 279 | 48870 | 138277 | 
| 146 | 81 django/contrib/auth/management/commands/createsuperuser.py | 54 | 192| 1080 | 49950 | 138277 | 
| 147 | 82 django/contrib/staticfiles/utils.py | 44 | 61| 153 | 50103 | 138671 | 
| 148 | 83 django/contrib/admin/templatetags/admin_list.py | 1 | 29| 203 | 50306 | 142323 | 
| 149 | 84 django/db/backends/base/base.py | 1 | 23| 134 | 50440 | 146826 | 
| 150 | 84 django/views/debug.py | 1 | 37| 259 | 50699 | 146826 | 
| 151 | 85 django/contrib/admin/views/main.py | 1 | 34| 232 | 50931 | 150093 | 
| 152 | 85 django/core/urlresolvers.py | 231 | 248| 175 | 51106 | 150093 | 
| 153 | 86 django/contrib/admin/migrations/0001_initial.py | 1 | 51| 338 | 51444 | 150431 | 
| 154 | 86 django/core/management/commands/shell.py | 1 | 17| 186 | 51630 | 150431 | 
| 155 | 87 django/forms/models.py | 898 | 917| 218 | 51848 | 161754 | 
| 156 | 88 django/core/management/commands/startapp.py | 1 | 28| 197 | 52045 | 161951 | 
| 157 | 89 django/core/management/commands/makemessages.py | 1 | 35| 250 | 52295 | 167385 | 
| 158 | 90 django/contrib/admin/exceptions.py | 1 | 12| 66 | 52361 | 167452 | 
| 159 | 90 django/shortcuts.py | 37 | 59| 156 | 52517 | 167452 | 


### Hint

```
Based on code inspection, I confirm that this bug exists. Possible fix: django.core.management.setup_environ could do something along the lines of: from django.conf import settings from django.core.management.base import set_script_prefix from django.utils.encoding import force_unicode set_script_prefix(u'/' if settings.FORCE_SCRIPT_NAME is None else force_unicode(settings.FORCE_SCRIPT_NAME))
If it is not possible to figure out the value of script_prefix in all cases (like in manage.py) why not just store its value in a settings variable? Then reverse could just prepend this value to all paths deduced from urlconf.
Would it make sense to call set_script_prefix() in ManagementUtility's execute() method, ​once settings have been configured?
django.setup() seems to be a natural place for it.
I thought so initially, but this issue is limited to management commands, that's why I wonder about ManagementUtility.execute(). Once the right location for this is confirmed, the fix should be trivial.
I imagine it would also affect standalone scripts that invoke django.setup() but don't use ManagementUtility.
```

## Patch

```diff
diff --git a/django/__init__.py b/django/__init__.py
--- a/django/__init__.py
+++ b/django/__init__.py
@@ -1,3 +1,5 @@
+from __future__ import unicode_literals
+
 from django.utils.version import get_version
 
 VERSION = (1, 10, 0, 'alpha', 0)
@@ -5,14 +7,21 @@
 __version__ = get_version(VERSION)
 
 
-def setup():
+def setup(set_prefix=True):
     """
     Configure the settings (this happens as a side effect of accessing the
     first setting), configure logging and populate the app registry.
+    Set the thread-local urlresolvers script prefix if `set_prefix` is True.
     """
     from django.apps import apps
     from django.conf import settings
+    from django.core.urlresolvers import set_script_prefix
+    from django.utils.encoding import force_text
     from django.utils.log import configure_logging
 
     configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
+    if set_prefix:
+        set_script_prefix(
+            '/' if settings.FORCE_SCRIPT_NAME is None else force_text(settings.FORCE_SCRIPT_NAME)
+        )
     apps.populate(settings.INSTALLED_APPS)
diff --git a/django/core/wsgi.py b/django/core/wsgi.py
--- a/django/core/wsgi.py
+++ b/django/core/wsgi.py
@@ -10,5 +10,5 @@ def get_wsgi_application():
     Allows us to avoid making django.core.handlers.WSGIHandler public API, in
     case the internal WSGI implementation changes or moves in the future.
     """
-    django.setup()
+    django.setup(set_prefix=False)
     return WSGIHandler()

```

## Test Patch

```diff
diff --git a/tests/user_commands/management/commands/reverse_url.py b/tests/user_commands/management/commands/reverse_url.py
new file mode 100644
--- /dev/null
+++ b/tests/user_commands/management/commands/reverse_url.py
@@ -0,0 +1,10 @@
+from django.core.management.base import BaseCommand
+from django.core.urlresolvers import reverse
+
+
+class Command(BaseCommand):
+    """
+    This command returns a URL from a reverse() call.
+    """
+    def handle(self, *args, **options):
+        return reverse('some_url')
diff --git a/tests/user_commands/tests.py b/tests/user_commands/tests.py
--- a/tests/user_commands/tests.py
+++ b/tests/user_commands/tests.py
@@ -1,5 +1,7 @@
 import os
 
+from admin_scripts.tests import AdminScriptTestCase
+
 from django.apps import apps
 from django.core import management
 from django.core.management import BaseCommand, CommandError, find_commands
@@ -159,6 +161,23 @@ def patched_check(self_, **kwargs):
             BaseCommand.check = saved_check
 
 
+class CommandRunTests(AdminScriptTestCase):
+    """
+    Tests that need to run by simulating the command line, not by call_command.
+    """
+    def tearDown(self):
+        self.remove_settings('settings.py')
+
+    def test_script_prefix_set_in_commands(self):
+        self.write_settings('settings.py', apps=['user_commands'], sdict={
+            'ROOT_URLCONF': '"user_commands.urls"',
+            'FORCE_SCRIPT_NAME': '"/PREFIX/"',
+        })
+        out, err = self.run_manage(['reverse_url'])
+        self.assertNoOutput(err)
+        self.assertEqual(out.strip(), '/PREFIX/some/url/')
+
+
 class UtilsTests(SimpleTestCase):
 
     def test_no_existent_external_program(self):
diff --git a/tests/user_commands/urls.py b/tests/user_commands/urls.py
new file mode 100644
--- /dev/null
+++ b/tests/user_commands/urls.py
@@ -0,0 +1,5 @@
+from django.conf.urls import url
+
+urlpatterns = [
+    url(r'^some/url/$', lambda req:req, name='some_url'),
+]

```


## Code snippets

### 1 - django/core/urlresolvers.py:

Start line: 548, End line: 617

```python
reverse_lazy = lazy(reverse, six.text_type)


def clear_url_caches():
    get_callable.cache_clear()
    get_resolver.cache_clear()
    get_ns_resolver.cache_clear()


def set_script_prefix(prefix):
    """
    Sets the script prefix for the current thread.
    """
    if not prefix.endswith('/'):
        prefix += '/'
    _prefixes.value = prefix


def get_script_prefix():
    """
    Returns the currently active script prefix. Useful for client code that
    wishes to construct their own URLs manually (although accessing the request
    instance is normally going to be a lot cleaner).
    """
    return getattr(_prefixes, "value", '/')


def clear_script_prefix():
    """
    Unsets the script prefix for the current thread.
    """
    try:
        del _prefixes.value
    except AttributeError:
        pass


def set_urlconf(urlconf_name):
    """
    Sets the URLconf for the current thread (overriding the default one in
    settings). Set to None to revert back to the default.
    """
    if urlconf_name:
        _urlconfs.value = urlconf_name
    else:
        if hasattr(_urlconfs, "value"):
            del _urlconfs.value


def get_urlconf(default=None):
    """
    Returns the root URLconf to use for the current thread if it has been
    changed from the default one.
    """
    return getattr(_urlconfs, "value", default)


def is_valid_path(path, urlconf=None):
    """
    Returns True if the given path resolves against the default URL resolver,
    False otherwise.

    This is a convenience method to make working with "is this a match?" cases
    easier, avoiding unnecessarily indented try...except blocks.
    """
    try:
        resolve(path, urlconf)
        return True
    except Resolver404:
        return False
```
### 2 - django/__init__.py:

Start line: 1, End line: 19

```python
from django.utils.version import get_version

VERSION = (1, 10, 0, 'alpha', 0)

__version__ = get_version(VERSION)


def setup():
    """
    Configure the settings (this happens as a side effect of accessing the
    first setting), configure logging and populate the app registry.
    """
    from django.apps import apps
    from django.conf import settings
    from django.utils.log import configure_logging

    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
    apps.populate(settings.INSTALLED_APPS)
```
### 3 - django/conf/global_settings.py:

Start line: 137, End line: 248

```python
LANGUAGES_BIDI = ["he", "ar", "fa", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = 'django_language'
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = '/'


# If you set this to True, Django will format dates, numbers and calendars
# according to user current locale.
USE_L10N = False

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default content type and charset to use for all HttpResponse objects, if a
# MIME type isn't manually specified. These are used to construct the
# Content-Type header.
DEFAULT_CONTENT_TYPE = 'text/html'
DEFAULT_CHARSET = 'utf-8'

# Encoding of files read from disk (template and initial SQL files).
FILE_CHARSET = 'utf-8'

# Email address that error messages come from.
SERVER_EMAIL = 'root@localhost'

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'

# Host for sending email.
EMAIL_HOST = 'localhost'

# Port for sending email.
EMAIL_PORT = 25

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ''
EMAIL_HOST_PASSWORD = ''
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = 'webmaster@localhost'

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = '[Django] '

# Whether to append trailing slashes to URLs.
APPEND_SLASH = True

# Whether to prepend the "www." subdomain to URLs that don't have it.
PREPEND_WWW = False

# Override the server-derived value of SCRIPT_NAME
FORCE_SCRIPT_NAME = None

# List of compiled regular expression objects representing User-Agent strings
# that are not allowed to visit any page, systemwide. Use this for bad
# robots/crawlers. Here are a few examples:
#     import re
#     DISALLOWED_USER_AGENTS = [
#         re.compile(r'^NaverBot.*'),
#         re.compile(r'^EmailSiphon.*'),
#         re.compile(r'^SiteSucker.*'),
#         re.compile(r'^sohu-search')
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$),
#        re.compile(r'^/robots.txt$),
#        re.compile(r'^/phpmyadmin/),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
```
### 4 - django/conf/global_settings.py:

Start line: 385, End line: 491

```python
NUMBER_GROUPING = 0

# Thousand separator symbol
THOUSAND_SEPARATOR = ','

# The tablespaces to use for each model when not specified otherwise.
DEFAULT_TABLESPACE = ''
DEFAULT_INDEX_TABLESPACE = ''

# Default X-Frame-Options header value
X_FRAME_OPTIONS = 'SAMEORIGIN'

USE_X_FORWARDED_HOST = False
USE_X_FORWARDED_PORT = False

# The Python dotted path to the WSGI application that Django's internal server
# (runserver) will use. If `None`, the return value of
# 'django.core.wsgi.get_wsgi_application' is used, thus preserving the same
# behavior as previous versions of Django. Otherwise this should point to an
# actual WSGI application object.
WSGI_APPLICATION = None

# If your Django app is behind a proxy that sets a header to specify secure
# connections, AND that proxy ensures that user-submitted headers with the
# same name are ignored (so that people can't spoof it), set this value to
# a tuple of (header_name, header_value). For any requests that come in with
# that header/value, request.is_secure() will return True.
# WARNING! Only set this if you fully understand what you're doing. Otherwise,
# you may be opening yourself up to a security risk.
SECURE_PROXY_SSL_HEADER = None

##############
# MIDDLEWARE #
##############

# List of middleware classes to use.  Order is important; in the request phase,
# this middleware classes will be applied in the order given, and in the
# response phase the middleware will be applied in reverse order.
MIDDLEWARE_CLASSES = [
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
]

############
# SESSIONS #
############

# Cache to store session data if using the cache session backend.
SESSION_CACHE_ALIAS = 'default'
# Cookie name. This can be whatever you want.
SESSION_COOKIE_NAME = 'sessionid'
# Age of cookie, in seconds (default: 2 weeks).
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7 * 2
# A string like ".example.com", or None for standard domain cookie.
SESSION_COOKIE_DOMAIN = None
# Whether the session cookie should be secure (https:// only).
SESSION_COOKIE_SECURE = False
# The path of the session cookie.
SESSION_COOKIE_PATH = '/'
# Whether to use the non-RFC standard httpOnly flag (IE, FF3+, others)
SESSION_COOKIE_HTTPONLY = True
# Whether to save the session data on every request.
SESSION_SAVE_EVERY_REQUEST = False
# Whether a user's session cookie expires when the Web browser is closed.
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
# The module to store session data
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
# Directory to store session files if using the file session module. If None,
# the backend will use a sensible default.
SESSION_FILE_PATH = None
# class to serialize session data
SESSION_SERIALIZER = 'django.contrib.sessions.serializers.JSONSerializer'

#########
# CACHE #
#########

# The cache backends to use.
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}
CACHE_MIDDLEWARE_KEY_PREFIX = ''
CACHE_MIDDLEWARE_SECONDS = 600
CACHE_MIDDLEWARE_ALIAS = 'default'

##################
# AUTHENTICATION #
##################

AUTH_USER_MODEL = 'auth.User'

AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend']

LOGIN_URL = '/accounts/login/'

LOGOUT_URL = '/accounts/logout/'

LOGIN_REDIRECT_URL = '/accounts/profile/'

# The number of days a password reset link is valid for
PASSWORD_RESET_TIMEOUT_DAYS = 3

# the first hasher in this list is the preferred algorithm.  any
# password using different algorithms will be converted automatically
# upon login
```
### 5 - django/conf/global_settings.py:

Start line: 1, End line: 48

```python
# Default Django settings. Override these with settings in the module
# pointed-to by the DJANGO_SETTINGS_MODULE environment variable.

# This is defined here as a do-nothing function because we can't import
# django.utils.translation -- that module depends on the settings.
gettext_noop = lambda s: s

####################
# CORE             #
####################

DEBUG = False

# Whether the framework should propagate raw exceptions rather than catching
# them. This is useful under some testing situations and should never be used
# on a live site.
DEBUG_PROPAGATE_EXCEPTIONS = False

# Whether to use the "Etag" header. This saves bandwidth but slows down performance.
USE_ETAGS = False

# People who get code error notifications.
# In the format [('Full Name', 'email@example.com'), ('Full Name', 'anotheremail@example.com')]
ADMINS = []

# List of IP addresses, as strings, that:
#   * See debug comments, when DEBUG is true
#   * Receive x-headers
INTERNAL_IPS = []

# Hosts/domain names that are valid for this site.
# "*" matches anything, ".example.com" matches example.com and all subdomains
ALLOWED_HOSTS = []

# Local time zone for this installation. All choices can be found here:
# https://en.wikipedia.org/wiki/List_of_tz_zones_by_name (although not all
# systems may support all possibilities). When USE_TZ is True, this is
# interpreted as the default user time zone.
TIME_ZONE = 'America/Chicago'

# If you set this to True, Django will use timezone-aware datetimes.
USE_TZ = False

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

# Languages we provide translations for, out of the box.
```
### 6 - django/core/handlers/wsgi.py:

Start line: 1, End line: 25

```python
from __future__ import unicode_literals

import cgi
import codecs
import logging
import re
import sys
from io import BytesIO
from threading import Lock

from django import http
from django.conf import settings
from django.core import signals
from django.core.handlers import base
from django.core.urlresolvers import set_script_prefix
from django.utils import six
from django.utils.encoding import force_str, force_text
from django.utils.functional import cached_property

logger = logging.getLogger('django.request')

# encode() and decode() expect the charset to be a native string.
ISO_8859_1, UTF_8 = str('iso-8859-1'), str('utf-8')

_slashes_re = re.compile(br'/+')
```
### 7 - django/conf/__init__.py:

Start line: 87, End line: 138

```python
class Settings(BaseSettings):
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
                    raise ImproperlyConfigured("The %s setting must be a list or a tuple. "
                            "Please fix your settings." % setting)
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

        if not self.SECRET_KEY:
            raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")

        if hasattr(time, 'tzset') and self.TIME_ZONE:
            # When we can, attempt to validate the timezone. If we can't find
            # this file, no check happens and it's harmless.
            zoneinfo_root = '/usr/share/zoneinfo'
            if (os.path.exists(zoneinfo_root) and not
                    os.path.exists(os.path.join(zoneinfo_root, *(self.TIME_ZONE.split('/'))))):
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
### 8 - setup.py:

Start line: 1, End line: 95

```python
import os
import sys
from distutils.sysconfig import get_python_lib

from setuptools import find_packages, setup

# Warn if we are installing over top of an existing installation. This can
# cause issues where files that were deleted from a more recent Django are
# still present in site-packages. See #18115.
overlay_warning = False
if "install" in sys.argv:
    lib_paths = [get_python_lib()]
    if lib_paths[0].startswith("/usr/lib/"):
        # We have to try also with an explicit prefix of /usr/local in order to
        # catch Debian's custom user site-packages directory.
        lib_paths.append(get_python_lib(prefix="/usr/local"))
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "django"))
        if os.path.exists(existing_path):
            # We note the need for the warning here, but present it after the
            # command is run, so it's more likely to be seen.
            overlay_warning = True
            break


EXCLUDE_FROM_PACKAGES = ['django.conf.project_template',
                         'django.conf.app_template',
                         'django.bin']


# Dynamically calculate the version based on django.VERSION.
version = __import__('django').get_version()


setup(
    name='Django',
    version=version,
    url='http://www.djangoproject.com/',
    author='Django Software Foundation',
    author_email='foundation@djangoproject.com',
    description=('A high-level Python Web framework that encourages '
                 'rapid development and clean, pragmatic design.'),
    license='BSD',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    scripts=['django/bin/django-admin.py'],
    entry_points={'console_scripts': [
        'django-admin = django.core.management:execute_from_command_line',
    ]},
    extras_require={
        "bcrypt": ["bcrypt"],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Internet :: WWW/HTTP :: WSGI',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)


if overlay_warning:
    sys.stderr.write("""

    ====
    ING!
    ====

    have just installed Django over top of an existing
    allation, without removing it first. Because of this,
     install may now include extraneous files from a
    ious version that have since been removed from
    go. This is known to cause a variety of problems. You
    ld manually remove the

    isting_path)s

    ctory and re-install Django.

    % {"existing_path": existing_path})
```
### 9 - django/conf/global_settings.py:

Start line: 492, End line: 617

```python
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
    'django.contrib.auth.hashers.BCryptPasswordHasher',
    'django.contrib.auth.hashers.SHA1PasswordHasher',
    'django.contrib.auth.hashers.MD5PasswordHasher',
    'django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher',
    'django.contrib.auth.hashers.UnsaltedMD5PasswordHasher',
    'django.contrib.auth.hashers.CryptPasswordHasher',
]

AUTH_PASSWORD_VALIDATORS = []

###########
# SIGNING #
###########

SIGNING_BACKEND = 'django.core.signing.TimestampSigner'

########
# CSRF #
########

# Dotted path to callable to be used as view when a request is
# rejected by the CSRF middleware.
CSRF_FAILURE_VIEW = 'django.views.csrf.csrf_failure'

# Settings for CSRF cookie.
CSRF_COOKIE_NAME = 'csrftoken'
CSRF_COOKIE_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_DOMAIN = None
CSRF_COOKIE_PATH = '/'
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_HEADER_NAME = 'HTTP_X_CSRFTOKEN'
CSRF_TRUSTED_ORIGINS = []

############
# MESSAGES #
############

# Class to use as messages backend
MESSAGE_STORAGE = 'django.contrib.messages.storage.fallback.FallbackStorage'

# Default values of MESSAGE_LEVEL and MESSAGE_TAGS are defined within
# django.contrib.messages to avoid imports in this settings file.

###########
# LOGGING #
###########

# The callable to use to configure logging
LOGGING_CONFIG = 'logging.config.dictConfig'

# Custom logging configuration.
LOGGING = {}

# Default exception reporter filter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER_FILTER = 'django.views.debug.SafeExceptionReporterFilter'

###########
# TESTING #
###########

# The name of the class to use to run the test suite
TEST_RUNNER = 'django.test.runner.DiscoverRunner'

# Apps that don't need to be serialized at test database creation time
# (only apps with migrations are to start with)
TEST_NON_SERIALIZED_APPS = []

############
# FIXTURES #
############

# The list of directories to search for fixtures
FIXTURE_DIRS = []

###############
# STATICFILES #
###############

# A list of locations of additional static files
STATICFILES_DIRS = []

# The default file storage backend used during the build process
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
]

##############
# MIGRATIONS #
##############

# Migration module overrides for apps, by app label.
MIGRATION_MODULES = {}

#################
# SYSTEM CHECKS #
#################

# List of all issues generated by system checks that should be silenced. Light
# issues like warnings, infos or debugs will not generate a message. Silencing
# serious issues like errors and criticals does not result in hiding the
# message, but Django will not stop you from e.g. running server.
SILENCED_SYSTEM_CHECKS = []

#######################
# SECURITY MIDDLEWARE #
#######################
SECURE_BROWSER_XSS_FILTER = False
SECURE_CONTENT_TYPE_NOSNIFF = False
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_SECONDS = 0
SECURE_REDIRECT_EXEMPT = []
SECURE_SSL_HOST = None
SECURE_SSL_REDIRECT = False
```
### 10 - django/core/handlers/wsgi.py:

Start line: 80, End line: 146

```python
class WSGIRequest(http.HttpRequest):
    def __init__(self, environ):
        script_name = get_script_name(environ)
        path_info = get_path_info(environ)
        if not path_info:
            # Sometimes PATH_INFO exists, but is empty (e.g. accessing
            # the SCRIPT_NAME URL without a trailing slash). We really need to
            # operate as if they'd requested '/'. Not amazingly nice to force
            # the path like this, but should be harmless.
            path_info = '/'
        self.environ = environ
        self.path_info = path_info
        # be careful to only replace the first slash in the path because of
        # http://test/something and http://test//something being different as
        # stated in http://www.ietf.org/rfc/rfc2396.txt
        self.path = '%s/%s' % (script_name.rstrip('/'),
                               path_info.replace('/', '', 1))
        self.META = environ
        self.META['PATH_INFO'] = path_info
        self.META['SCRIPT_NAME'] = script_name
        self.method = environ['REQUEST_METHOD'].upper()
        _, content_params = cgi.parse_header(environ.get('CONTENT_TYPE', ''))
        if 'charset' in content_params:
            try:
                codecs.lookup(content_params['charset'])
            except LookupError:
                pass
            else:
                self.encoding = content_params['charset']
        self._post_parse_error = False
        try:
            content_length = int(environ.get('CONTENT_LENGTH'))
        except (ValueError, TypeError):
            content_length = 0
        self._stream = LimitedStream(self.environ['wsgi.input'], content_length)
        self._read_started = False
        self.resolver_match = None

    def _get_scheme(self):
        return self.environ.get('wsgi.url_scheme')

    @cached_property
    def GET(self):
        # The WSGI spec says 'QUERY_STRING' may be absent.
        raw_query_string = get_bytes_from_wsgi(self.environ, 'QUERY_STRING', '')
        return http.QueryDict(raw_query_string, encoding=self._encoding)

    def _get_post(self):
        if not hasattr(self, '_post'):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    @cached_property
    def COOKIES(self):
        raw_cookie = get_str_from_wsgi(self.environ, 'HTTP_COOKIE', '')
        return http.parse_cookie(raw_cookie)

    def _get_files(self):
        if not hasattr(self, '_files'):
            self._load_post_and_files()
        return self._files

    POST = property(_get_post, _set_post)
    FILES = property(_get_files)
```
### 102 - django/core/wsgi.py:

Start line: 1, End line: 15

```python
import django
from django.core.handlers.wsgi import WSGIHandler


def get_wsgi_application():
    """
    The public interface to Django's WSGI support. Should return a WSGI
    callable.

    Allows us to avoid making django.core.handlers.WSGIHandler public API, in
    case the internal WSGI implementation changes or moves in the future.
    """
    django.setup()
    return WSGIHandler()
```
