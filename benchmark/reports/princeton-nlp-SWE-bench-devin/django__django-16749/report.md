# django__django-16749

| **django/django** | `c3d7a71f836f7cfe8fa90dd9ae95b37b660d5aae` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 9301 |
| **Any found context length** | 1717 |
| **Avg pos** | 70.0 |
| **Min pos** | 5 |
| **Max pos** | 27 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/handlers/asgi.py b/django/core/handlers/asgi.py
--- a/django/core/handlers/asgi.py
+++ b/django/core/handlers/asgi.py
@@ -26,6 +26,15 @@
 logger = logging.getLogger("django.request")
 
 
+def get_script_prefix(scope):
+    """
+    Return the script prefix to use from either the scope or a setting.
+    """
+    if settings.FORCE_SCRIPT_NAME:
+        return settings.FORCE_SCRIPT_NAME
+    return scope.get("root_path", "") or ""
+
+
 class ASGIRequest(HttpRequest):
     """
     Custom request subclass that decodes from an ASGI-standard request dict
@@ -41,7 +50,7 @@ def __init__(self, scope, body_file):
         self._post_parse_error = False
         self._read_started = False
         self.resolver_match = None
-        self.script_name = self.scope.get("root_path", "")
+        self.script_name = get_script_prefix(scope)
         if self.script_name:
             # TODO: Better is-prefix checking, slash handling?
             self.path_info = scope["path"].removeprefix(self.script_name)
@@ -170,7 +179,7 @@ async def handle(self, scope, receive, send):
         except RequestAborted:
             return
         # Request is complete and can be served.
-        set_script_prefix(self.get_script_prefix(scope))
+        set_script_prefix(get_script_prefix(scope))
         await signals.request_started.asend(sender=self.__class__, scope=scope)
         # Get the request and check for basic issues.
         request, error_response = self.create_request(scope, body_file)
@@ -344,11 +353,3 @@ def chunk_bytes(cls, data):
                 (position + cls.chunk_size) >= len(data),
             )
             position += cls.chunk_size
-
-    def get_script_prefix(self, scope):
-        """
-        Return the script prefix to use from either the scope or a setting.
-        """
-        if settings.FORCE_SCRIPT_NAME:
-            return settings.FORCE_SCRIPT_NAME
-        return scope.get("root_path", "") or ""

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/handlers/asgi.py | 29 | 29 | 27 | 2 | 9301
| django/core/handlers/asgi.py | 44 | 44 | 27 | 2 | 9301
| django/core/handlers/asgi.py | 173 | 173 | 5 | 2 | 1717
| django/core/handlers/asgi.py | 347 | 354 | 11 | 2 | 4489


## Problem Statement

```
ASGIRequest doesn't respect settings.FORCE_SCRIPT_NAME.
Description
	
For example, I have settings.FORCE_SCRIPT_NAME = '/some-prefix'
I start a django server with command: daphne django_project.asgi:application
And I navigate to the ​http://localhost:8000/admin/login, and see the login form action url is "/admin/login" which is wrong, which should be "/some-prefix/admin/login"

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/conf/__init__.py | 129 | 142| 117 | 117 | 2365 | 
| 2 | **2 django/core/handlers/asgi.py** | 1 | 26| 126 | 243 | 5020 | 
| 3 | 3 django/conf/global_settings.py | 153 | 263| 832 | 1075 | 10858 | 
| 4 | 4 django/core/handlers/wsgi.py | 164 | 194| 323 | 1398 | 12635 | 
| **-> 5 <-** | **4 django/core/handlers/asgi.py** | 163 | 204| 319 | 1717 | 12635 | 
| 6 | 5 django/core/asgi.py | 1 | 14| 0 | 1717 | 12720 | 
| 7 | 5 django/conf/global_settings.py | 405 | 480| 790 | 2507 | 12720 | 
| 8 | 5 django/conf/global_settings.py | 481 | 596| 797 | 3304 | 12720 | 
| 9 | 5 django/conf/__init__.py | 175 | 241| 563 | 3867 | 12720 | 
| 10 | 5 django/conf/global_settings.py | 597 | 668| 453 | 4320 | 12720 | 
| **-> 11 <-** | **5 django/core/handlers/asgi.py** | 331 | 355| 169 | 4489 | 12720 | 
| 12 | 6 django/middleware/common.py | 62 | 74| 136 | 4625 | 14266 | 
| 13 | 7 django/core/checks/security/base.py | 1 | 79| 691 | 5316 | 16455 | 
| 14 | 8 django/core/servers/basehttp.py | 214 | 233| 170 | 5486 | 18596 | 
| 15 | 9 django/http/request.py | 183 | 192| 113 | 5599 | 24174 | 
| 16 | **9 django/core/handlers/asgi.py** | 138 | 161| 171 | 5770 | 24174 | 
| 17 | 10 django/contrib/admin/sites.py | 443 | 459| 136 | 5906 | 28671 | 
| 18 | 10 django/core/handlers/wsgi.py | 56 | 120| 568 | 6474 | 28671 | 
| 19 | 10 django/middleware/common.py | 34 | 60| 265 | 6739 | 28671 | 
| 20 | 10 django/http/request.py | 441 | 489| 338 | 7077 | 28671 | 
| 21 | 11 django/urls/base.py | 91 | 157| 383 | 7460 | 29867 | 
| 22 | 12 django/contrib/staticfiles/handlers.py | 84 | 116| 264 | 7724 | 30695 | 
| 23 | 13 django/contrib/auth/admin.py | 1 | 25| 195 | 7919 | 32466 | 
| 24 | 14 django/__init__.py | 1 | 25| 173 | 8092 | 32639 | 
| 25 | 15 django/utils/http.py | 303 | 324| 179 | 8271 | 35841 | 
| 26 | 16 django/views/defaults.py | 1 | 26| 151 | 8422 | 36831 | 
| **-> 27 <-** | **16 django/core/handlers/asgi.py** | 29 | 135| 879 | 9301 | 36831 | 
| 28 | 16 django/conf/global_settings.py | 264 | 355| 832 | 10133 | 36831 | 
| 29 | 16 django/core/servers/basehttp.py | 176 | 212| 279 | 10412 | 36831 | 
| 30 | 16 django/middleware/common.py | 76 | 98| 228 | 10640 | 36831 | 
| 31 | 17 django/contrib/admin/tests.py | 132 | 147| 169 | 10809 | 38473 | 
| 32 | 17 django/conf/global_settings.py | 356 | 404| 575 | 11384 | 38473 | 
| 33 | 17 django/http/request.py | 257 | 282| 191 | 11575 | 38473 | 
| 34 | 17 django/conf/global_settings.py | 1 | 50| 367 | 11942 | 38473 | 
| 35 | 17 django/views/defaults.py | 102 | 121| 144 | 12086 | 38473 | 
| 36 | 17 django/urls/base.py | 1 | 24| 170 | 12256 | 38473 | 
| 37 | 18 django/contrib/auth/views.py | 1 | 32| 255 | 12511 | 41187 | 
| 38 | 18 django/contrib/admin/sites.py | 228 | 249| 221 | 12732 | 41187 | 
| 39 | 18 django/contrib/admin/tests.py | 1 | 37| 265 | 12997 | 41187 | 
| 40 | 19 django/contrib/staticfiles/utils.py | 42 | 72| 217 | 13214 | 41649 | 
| 41 | 20 django/views/csrf.py | 30 | 88| 587 | 13801 | 42449 | 
| 42 | 20 django/urls/base.py | 27 | 88| 440 | 14241 | 42449 | 
| 43 | 21 django/views/debug.py | 196 | 221| 181 | 14422 | 47505 | 
| 44 | 22 django/core/management/commands/runserver.py | 1 | 22| 204 | 14626 | 49032 | 
| 45 | 23 django/contrib/auth/urls.py | 1 | 37| 253 | 14879 | 49285 | 
| 46 | 23 django/views/defaults.py | 124 | 150| 197 | 15076 | 49285 | 
| 47 | 24 django/middleware/security.py | 1 | 31| 281 | 15357 | 49811 | 
| 48 | 25 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 15480 | 55639 | 
| 49 | 25 django/middleware/common.py | 1 | 32| 247 | 15727 | 55639 | 
| 50 | 25 django/contrib/admin/sites.py | 405 | 441| 300 | 16027 | 55639 | 
| 51 | 25 django/conf/__init__.py | 264 | 286| 209 | 16236 | 55639 | 
| 52 | 26 django/contrib/admin/views/main.py | 1 | 61| 357 | 16593 | 60397 | 
| 53 | 26 django/contrib/admin/views/main.py | 276 | 292| 125 | 16718 | 60397 | 
| 54 | 26 django/contrib/admin/tests.py | 184 | 202| 177 | 16895 | 60397 | 
| 55 | 27 django/utils/log.py | 1 | 76| 501 | 17396 | 62071 | 
| 56 | 28 django/core/management/commands/diffsettings.py | 45 | 61| 138 | 17534 | 62781 | 
| 57 | 28 django/contrib/admin/sites.py | 1 | 33| 224 | 17758 | 62781 | 
| 58 | 28 django/conf/__init__.py | 1 | 45| 265 | 18023 | 62781 | 
| 59 | 28 django/contrib/admin/sites.py | 570 | 597| 196 | 18219 | 62781 | 
| 60 | 29 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 16| 0 | 18219 | 62849 | 
| 61 | 30 django/middleware/csrf.py | 1 | 55| 480 | 18699 | 66960 | 
| 62 | 31 django/contrib/auth/__init__.py | 1 | 38| 240 | 18939 | 68674 | 
| 63 | 32 django/contrib/auth/middleware.py | 25 | 36| 107 | 19046 | 69753 | 
| 64 | 32 django/contrib/auth/middleware.py | 1 | 22| 143 | 19189 | 69753 | 
| 65 | 32 django/conf/__init__.py | 288 | 315| 177 | 19366 | 69753 | 
| 66 | 33 django/core/checks/security/csrf.py | 1 | 42| 305 | 19671 | 70218 | 
| 67 | 33 django/middleware/security.py | 33 | 67| 251 | 19922 | 70218 | 
| 68 | 33 django/core/checks/security/base.py | 81 | 180| 732 | 20654 | 70218 | 
| 69 | 34 django/contrib/redirects/admin.py | 1 | 11| 0 | 20654 | 70287 | 
| 70 | 35 django/http/response.py | 630 | 653| 195 | 20849 | 75690 | 
| 71 | **35 django/core/handlers/asgi.py** | 275 | 329| 458 | 21307 | 75690 | 
| 72 | 36 django/core/handlers/exception.py | 1 | 21| 118 | 21425 | 76820 | 
| 73 | 36 django/views/csrf.py | 1 | 27| 212 | 21637 | 76820 | 
| 74 | 36 django/core/handlers/wsgi.py | 123 | 161| 334 | 21971 | 76820 | 
| 75 | 36 django/http/response.py | 1 | 29| 186 | 22157 | 76820 | 
| 76 | 37 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 16| 0 | 22157 | 76897 | 
| 77 | 38 django/contrib/auth/management/commands/createsuperuser.py | 250 | 279| 211 | 22368 | 79188 | 
| 78 | 39 django/contrib/sitemaps/__init__.py | 30 | 56| 243 | 22611 | 81016 | 
| 79 | 39 django/conf/__init__.py | 73 | 96| 232 | 22843 | 81016 | 
| 80 | 39 django/core/servers/basehttp.py | 28 | 53| 228 | 23071 | 81016 | 
| 81 | 39 django/middleware/common.py | 100 | 115| 165 | 23236 | 81016 | 
| 82 | 39 django/views/debug.py | 146 | 180| 283 | 23519 | 81016 | 
| 83 | **39 django/core/handlers/asgi.py** | 246 | 273| 237 | 23756 | 81016 | 
| 84 | 40 django/contrib/admin/helpers.py | 1 | 36| 209 | 23965 | 84628 | 
| 85 | 41 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 24384 | 85047 | 
| 86 | 42 django/contrib/admin/options.py | 1 | 119| 796 | 25180 | 104342 | 
| 87 | 42 django/conf/__init__.py | 48 | 71| 201 | 25381 | 104342 | 
| 88 | 43 docs/_ext/djangodocs.py | 26 | 71| 398 | 25779 | 107566 | 
| 89 | 44 docs/conf.py | 1 | 53| 485 | 26264 | 111119 | 
| 90 | 44 django/core/management/commands/runserver.py | 80 | 120| 401 | 26665 | 111119 | 
| 91 | 44 django/conf/__init__.py | 114 | 127| 131 | 26796 | 111119 | 
| 92 | 44 django/http/response.py | 656 | 689| 157 | 26953 | 111119 | 
| 93 | 44 django/http/request.py | 169 | 181| 133 | 27086 | 111119 | 
| 94 | 45 django/db/backends/postgresql/client.py | 1 | 65| 467 | 27553 | 111586 | 
| 95 | 45 django/middleware/common.py | 153 | 179| 255 | 27808 | 111586 | 
| 96 | 45 django/contrib/admin/sites.py | 251 | 313| 527 | 28335 | 111586 | 
| 97 | 46 django/conf/locale/id/formats.py | 5 | 50| 678 | 29013 | 112309 | 
| 98 | 46 django/http/request.py | 284 | 336| 351 | 29364 | 112309 | 
| 99 | 47 django/views/generic/base.py | 145 | 179| 204 | 29568 | 114215 | 
| 100 | 47 django/middleware/csrf.py | 252 | 268| 186 | 29754 | 114215 | 
| 101 | 48 django/urls/conf.py | 1 | 58| 451 | 30205 | 114932 | 
| 102 | 48 django/middleware/csrf.py | 296 | 346| 450 | 30655 | 114932 | 
| 103 | 49 django/conf/locale/en_GB/formats.py | 5 | 42| 673 | 31328 | 115650 | 
| 104 | 49 django/http/request.py | 1 | 48| 286 | 31614 | 115650 | 
| 105 | 50 django/views/i18n.py | 30 | 74| 382 | 31996 | 117519 | 
| 106 | 51 django/conf/locale/en_AU/formats.py | 5 | 42| 673 | 32669 | 118237 | 
| 107 | 51 django/core/management/commands/diffsettings.py | 63 | 73| 128 | 32797 | 118237 | 
| 108 | 52 django/core/management/commands/startapp.py | 1 | 15| 0 | 32797 | 118338 | 
| 109 | 52 django/contrib/staticfiles/handlers.py | 67 | 81| 127 | 32924 | 118338 | 
| 110 | 52 django/conf/__init__.py | 98 | 112| 129 | 33053 | 118338 | 
| 111 | 52 django/contrib/admin/options.py | 1866 | 1897| 303 | 33356 | 118338 | 
| 112 | 52 django/contrib/staticfiles/handlers.py | 1 | 64| 436 | 33792 | 118338 | 
| 113 | 53 django/contrib/sitemaps/views.py | 91 | 141| 369 | 34161 | 119410 | 
| 114 | 53 django/contrib/admin/sites.py | 600 | 614| 116 | 34277 | 119410 | 
| 115 | 53 django/core/checks/security/base.py | 259 | 284| 211 | 34488 | 119410 | 
| 116 | 54 django/urls/__init__.py | 1 | 54| 269 | 34757 | 119679 | 
| 117 | 55 django/conf/locale/ig/formats.py | 5 | 33| 387 | 35144 | 120111 | 
| 118 | 56 django/core/management/templates.py | 86 | 158| 606 | 35750 | 123138 | 
| 119 | 56 docs/conf.py | 54 | 131| 711 | 36461 | 123138 | 
| 120 | 57 django/core/management/commands/startproject.py | 1 | 22| 159 | 36620 | 123297 | 
| 121 | 58 django/conf/locale/ms/formats.py | 5 | 39| 607 | 37227 | 123949 | 
| 122 | 59 django/contrib/auth/migrations/0012_alter_user_first_name_max_length.py | 1 | 18| 0 | 37227 | 124028 | 
| 123 | 60 django/conf/locale/sl/formats.py | 5 | 45| 713 | 37940 | 124786 | 
| 124 | 60 django/contrib/admin/options.py | 1730 | 1761| 293 | 38233 | 124786 | 
| 125 | 61 django/contrib/auth/password_validation.py | 217 | 267| 386 | 38619 | 126680 | 
| 126 | 61 django/views/generic/base.py | 256 | 286| 246 | 38865 | 126680 | 
| 127 | 61 django/contrib/auth/admin.py | 149 | 214| 477 | 39342 | 126680 | 
| 128 | 62 django/contrib/sites/admin.py | 1 | 9| 0 | 39342 | 126726 | 
| 129 | 63 django/template/backends/jinja2.py | 1 | 52| 356 | 39698 | 127541 | 
| 130 | 64 django/contrib/auth/decorators.py | 1 | 40| 315 | 40013 | 128133 | 
| 131 | 64 docs/conf.py | 133 | 236| 934 | 40947 | 128133 | 
| 132 | 64 django/core/management/commands/runserver.py | 25 | 66| 284 | 41231 | 128133 | 
| 133 | 65 django/contrib/admin/utils.py | 311 | 337| 189 | 41420 | 132523 | 
| 134 | 65 django/contrib/sitemaps/views.py | 42 | 88| 423 | 41843 | 132523 | 
| 135 | 66 django/core/management/base.py | 395 | 429| 297 | 42140 | 137383 | 
| 136 | 67 django/conf/locale/sr/formats.py | 5 | 45| 742 | 42882 | 138170 | 
| 137 | 68 django/conf/locale/sr_Latn/formats.py | 5 | 45| 741 | 43623 | 138956 | 
| 138 | 69 django/conf/locale/cs/formats.py | 5 | 44| 609 | 44232 | 139610 | 
| 139 | 70 django/template/defaulttags.py | 544 | 568| 191 | 44423 | 150383 | 
| 140 | 71 django/db/backends/mysql/client.py | 1 | 73| 642 | 45065 | 151025 | 
| 141 | 72 django/template/backends/dummy.py | 1 | 52| 327 | 45392 | 151352 | 
| 142 | 73 django/conf/locale/bg/formats.py | 5 | 22| 130 | 45522 | 151526 | 
| 143 | 74 django/conf/locale/tg/formats.py | 5 | 33| 401 | 45923 | 151972 | 
| 144 | 74 django/contrib/auth/middleware.py | 59 | 96| 362 | 46285 | 151972 | 
| 145 | 75 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 46920 | 152652 | 
| 146 | 75 django/views/defaults.py | 82 | 99| 121 | 47041 | 152652 | 
| 147 | 76 django/conf/locale/it/formats.py | 5 | 44| 772 | 47813 | 153469 | 
| 148 | 77 django/middleware/clickjacking.py | 1 | 49| 368 | 48181 | 153837 | 
| 149 | 78 django/contrib/admin/migrations/0001_initial.py | 1 | 76| 363 | 48544 | 154200 | 
| 150 | 79 django/contrib/redirects/middleware.py | 1 | 51| 354 | 48898 | 154555 | 
| 151 | 79 django/utils/http.py | 1 | 39| 460 | 49358 | 154555 | 
| 152 | 80 django/conf/locale/cy/formats.py | 5 | 34| 531 | 49889 | 155131 | 
| 153 | 81 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 50036 | 155279 | 
| 154 | 82 django/contrib/redirects/apps.py | 1 | 9| 0 | 50036 | 155330 | 
| 155 | 82 django/contrib/admin/options.py | 1524 | 1550| 233 | 50269 | 155330 | 
| 156 | 83 django/conf/locale/az/formats.py | 5 | 31| 363 | 50632 | 155738 | 
| 157 | 84 django/apps/registry.py | 283 | 304| 237 | 50869 | 159158 | 
| 158 | 84 django/contrib/auth/management/commands/createsuperuser.py | 1 | 88| 597 | 51466 | 159158 | 
| 159 | 84 django/core/servers/basehttp.py | 1 | 25| 175 | 51641 | 159158 | 
| 160 | 85 django/contrib/staticfiles/finders.py | 74 | 114| 270 | 51911 | 161293 | 
| 161 | 85 django/core/management/base.py | 115 | 141| 173 | 52084 | 161293 | 
| 162 | 86 django/contrib/admindocs/urls.py | 1 | 51| 307 | 52391 | 161600 | 
| 163 | 87 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 18| 0 | 52391 | 161682 | 
| 164 | 87 django/contrib/auth/__init__.py | 226 | 245| 154 | 52545 | 161682 | 
| 165 | 88 django/contrib/gis/sitemaps/__init__.py | 1 | 5| 0 | 52545 | 161724 | 
| 166 | 88 django/contrib/auth/views.py | 252 | 294| 382 | 52927 | 161724 | 
| 167 | **88 django/core/handlers/asgi.py** | 225 | 244| 177 | 53104 | 161724 | 
| 168 | 88 django/contrib/auth/admin.py | 28 | 40| 130 | 53234 | 161724 | 
| 169 | 88 django/contrib/auth/views.py | 348 | 380| 239 | 53473 | 161724 | 
| 170 | 89 django/conf/locale/es_NI/formats.py | 3 | 27| 271 | 53744 | 162011 | 
| 171 | 90 django/conf/locale/el/formats.py | 5 | 35| 460 | 54204 | 162516 | 
| 172 | 91 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 54839 | 163196 | 
| 173 | 92 scripts/manage_translations.py | 1 | 29| 195 | 55034 | 164894 | 
| 174 | 93 django/contrib/postgres/forms/__init__.py | 1 | 4| 0 | 55034 | 164925 | 
| 175 | 93 django/contrib/admin/options.py | 121 | 154| 235 | 55269 | 164925 | 
| 176 | 93 django/middleware/csrf.py | 270 | 294| 176 | 55445 | 164925 | 
| 177 | 94 django/conf/locale/ka/formats.py | 29 | 49| 476 | 55921 | 165764 | 
| 178 | 95 django/conf/locale/fr/formats.py | 5 | 34| 458 | 56379 | 166267 | 
| 179 | 96 django/conf/locale/fi/formats.py | 5 | 37| 434 | 56813 | 166746 | 
| 180 | 96 django/contrib/auth/admin.py | 43 | 119| 528 | 57341 | 166746 | 
| 181 | 96 django/middleware/csrf.py | 348 | 411| 585 | 57926 | 166746 | 
| 182 | 97 django/conf/locale/lt/formats.py | 5 | 46| 681 | 58607 | 167472 | 
| 183 | 97 django/contrib/admin/options.py | 1763 | 1865| 780 | 59387 | 167472 | 
| 184 | 98 django/contrib/auth/mixins.py | 46 | 73| 232 | 59619 | 168367 | 
| 185 | 99 django/db/models/options.py | 1 | 58| 353 | 59972 | 176063 | 
| 186 | 99 django/apps/registry.py | 1 | 59| 475 | 60447 | 176063 | 
| 187 | 99 django/core/checks/security/csrf.py | 45 | 68| 159 | 60606 | 176063 | 
| 188 | 100 django/views/generic/edit.py | 1 | 73| 491 | 61097 | 177926 | 
| 189 | 101 django/contrib/auth/forms.py | 252 | 277| 173 | 61270 | 181249 | 
| 190 | 102 django/core/checks/templates.py | 1 | 47| 313 | 61583 | 181731 | 
| 191 | 103 django/conf/locale/nb/formats.py | 5 | 42| 610 | 62193 | 182386 | 


### Hint

```
Thanks for the report. It seems that ASGIRequest should take FORCE_SCRIPT_NAME into account (as WSGIRequest), e.g. django/core/handlers/asgi.py diff --git a/django/core/handlers/asgi.py b/django/core/handlers/asgi.py index 569157b277..c5eb87c712 100644 a b class ASGIRequest(HttpRequest): 4040 self._post_parse_error = False 4141 self._read_started = False 4242 self.resolver_match = None 43 self.script_name = self.scope.get("root_path", "") 43 self.script_name = get_script_prefix(scope) 4444 if self.script_name: 4545 # TODO: Better is-prefix checking, slash handling? 4646 self.path_info = scope["path"].removeprefix(self.script_name) … … class ASGIHandler(base.BaseHandler): 169169 except RequestAborted: 170170 return 171171 # Request is complete and can be served. 172 set_script_prefix(self.get_script_prefix(scope)) 172 set_script_prefix(get_script_prefix(scope)) 173173 await signals.request_started.asend(sender=self.__class__, scope=scope) 174174 # Get the request and check for basic issues. 175175 request, error_response = self.create_request(scope, body_file) … … class ASGIHandler(base.BaseHandler): 310310 ) 311311 position += cls.chunk_size 312312 313 def get_script_prefix(self, scope): 314 """ 315 Return the script prefix to use from either the scope or a setting. 316 """ 317 if settings.FORCE_SCRIPT_NAME: 318 return settings.FORCE_SCRIPT_NAME 319 return scope.get("root_path", "") or "" 313 314def get_script_prefix(scope): 315 """ 316 Return the script prefix to use from either the scope or a setting. 317 """ 318 if settings.FORCE_SCRIPT_NAME: 319 return settings.FORCE_SCRIPT_NAME 320 return scope.get("root_path", "") or "" Would you like to prepare a patch via GitHub PR? (a regression test is required.)
FORCE_SCRIPT_NAME doesn't working as expected for both ASGI and WSGI application.
Here, in the above attachment, I've created a simple Django app and set this FORCE_SCRIPT_NAME to /ayush. On running with both WSGI and ASGI applications, it shows the above-mentioned error. Also, the queried URL is different as compared to what Django is looking for. There is the repetition of FORCE_SCRIPT_NAME
Seems like, FORCE_SCRIPT_NAME is also causing some issues with the WSGI application too, after someone's logout the session. @Mariusz Felisiak, your solution for ASGI is working for me. Please correct me, If I'm wrong...
Replying to Ayush Bisht: Please correct me, If I'm wrong... FORCE_SCRIPT_NAME is not intended to work with dev server, see #7930. It works fine when you try with e.g. daphne.
```

## Patch

```diff
diff --git a/django/core/handlers/asgi.py b/django/core/handlers/asgi.py
--- a/django/core/handlers/asgi.py
+++ b/django/core/handlers/asgi.py
@@ -26,6 +26,15 @@
 logger = logging.getLogger("django.request")
 
 
+def get_script_prefix(scope):
+    """
+    Return the script prefix to use from either the scope or a setting.
+    """
+    if settings.FORCE_SCRIPT_NAME:
+        return settings.FORCE_SCRIPT_NAME
+    return scope.get("root_path", "") or ""
+
+
 class ASGIRequest(HttpRequest):
     """
     Custom request subclass that decodes from an ASGI-standard request dict
@@ -41,7 +50,7 @@ def __init__(self, scope, body_file):
         self._post_parse_error = False
         self._read_started = False
         self.resolver_match = None
-        self.script_name = self.scope.get("root_path", "")
+        self.script_name = get_script_prefix(scope)
         if self.script_name:
             # TODO: Better is-prefix checking, slash handling?
             self.path_info = scope["path"].removeprefix(self.script_name)
@@ -170,7 +179,7 @@ async def handle(self, scope, receive, send):
         except RequestAborted:
             return
         # Request is complete and can be served.
-        set_script_prefix(self.get_script_prefix(scope))
+        set_script_prefix(get_script_prefix(scope))
         await signals.request_started.asend(sender=self.__class__, scope=scope)
         # Get the request and check for basic issues.
         request, error_response = self.create_request(scope, body_file)
@@ -344,11 +353,3 @@ def chunk_bytes(cls, data):
                 (position + cls.chunk_size) >= len(data),
             )
             position += cls.chunk_size
-
-    def get_script_prefix(self, scope):
-        """
-        Return the script prefix to use from either the scope or a setting.
-        """
-        if settings.FORCE_SCRIPT_NAME:
-            return settings.FORCE_SCRIPT_NAME
-        return scope.get("root_path", "") or ""

```

## Test Patch

```diff
diff --git a/tests/handlers/tests.py b/tests/handlers/tests.py
--- a/tests/handlers/tests.py
+++ b/tests/handlers/tests.py
@@ -3,6 +3,7 @@
 from django.core.signals import request_finished, request_started
 from django.db import close_old_connections, connection
 from django.test import (
+    AsyncRequestFactory,
     RequestFactory,
     SimpleTestCase,
     TransactionTestCase,
@@ -328,6 +329,12 @@ async def test_unawaited_response(self):
         with self.assertRaisesMessage(ValueError, msg):
             await self.async_client.get("/unawaited/")
 
+    @override_settings(FORCE_SCRIPT_NAME="/FORCED_PREFIX/")
+    def test_force_script_name(self):
+        async_request_factory = AsyncRequestFactory()
+        request = async_request_factory.request(**{"path": "/somepath/"})
+        self.assertEqual(request.path, "/FORCED_PREFIX/somepath/")
+
     async def test_sync_streaming(self):
         response = await self.async_client.get("/streaming/")
         self.assertEqual(response.status_code, 200)

```


## Code snippets

### 1 - django/conf/__init__.py:

Start line: 129, End line: 142

```python
class LazySettings(LazyObject):

    @staticmethod
    def _add_script_prefix(value):
        """
        Add SCRIPT_NAME prefix to relative paths.

        Useful when the app is being served at a subpath and manually prefixing
        subpath to STATIC_URL and MEDIA_URL in settings is inconvenient.
        """
        # Don't apply prefix to absolute paths and URLs.
        if value.startswith(("http://", "https://", "/")):
            return value
        from django.urls import get_script_prefix

        return "%s%s" % (get_script_prefix(), value)
```
### 2 - django/core/handlers/asgi.py:

Start line: 1, End line: 26

```python
import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing

from asgiref.sync import ThreadSensitiveContext, sync_to_async

from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseServerError,
    QueryDict,
    parse_cookie,
)
from django.urls import set_script_prefix
from django.utils.functional import cached_property

logger = logging.getLogger("django.request")
```
### 3 - django/conf/global_settings.py:

Start line: 153, End line: 263

```python
LANGUAGES_BIDI = ["he", "ar", "ar-dz", "ckb", "fa", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = "django_language"
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = "/"
LANGUAGE_COOKIE_SECURE = False
LANGUAGE_COOKIE_HTTPONLY = False
LANGUAGE_COOKIE_SAMESITE = None

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default charset to use for all HttpResponse objects, if a MIME type isn't
# manually specified. It's used to construct the Content-Type header.
DEFAULT_CHARSET = "utf-8"

# Email address that error messages come from.
SERVER_EMAIL = "root@localhost"

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# Host for sending email.
EMAIL_HOST = "localhost"

# Port for sending email.
EMAIL_PORT = 25

# Whether to send SMTP 'Date' header in the local time zone or in UTC.
EMAIL_USE_LOCALTIME = False

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ""
EMAIL_HOST_PASSWORD = ""
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default form rendering class.
FORM_RENDERER = "django.forms.renderers.DjangoTemplates"

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = "webmaster@localhost"

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = "[Django] "

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
#         re.compile(r'^sohu-search'),
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$'),
#        re.compile(r'^/robots.txt$'),
#        re.compile(r'^/phpmyadmin/'),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
```
### 4 - django/core/handlers/wsgi.py:

Start line: 164, End line: 194

```python
def get_script_name(environ):
    """
    Return the equivalent of the HTTP request's SCRIPT_NAME environment
    variable. If Apache mod_rewrite is used, return what would have been
    the script name prior to any rewriting (so it's the script name as seen
    from the client's perspective), unless the FORCE_SCRIPT_NAME setting is
    set (to anything).
    """
    if settings.FORCE_SCRIPT_NAME is not None:
        return settings.FORCE_SCRIPT_NAME

    # If Apache's mod_rewrite had a whack at the URL, Apache set either
    # SCRIPT_URL or REDIRECT_URL to the full resource URL before applying any
    # rewrites. Unfortunately not every web server (lighttpd!) passes this
    # information through all the time, so FORCE_SCRIPT_NAME, above, is still
    # needed.
    script_url = get_bytes_from_wsgi(environ, "SCRIPT_URL", "") or get_bytes_from_wsgi(
        environ, "REDIRECT_URL", ""
    )

    if script_url:
        if b"//" in script_url:
            # mod_wsgi squashes multiple successive slashes in PATH_INFO,
            # do the same with script_url before manipulating paths (#17133).
            script_url = _slashes_re.sub(b"/", script_url)
        path_info = get_bytes_from_wsgi(environ, "PATH_INFO", "")
        script_name = script_url.removesuffix(path_info)
    else:
        script_name = get_bytes_from_wsgi(environ, "SCRIPT_NAME", "")

    return script_name.decode()
```
### 5 - django/core/handlers/asgi.py:

Start line: 163, End line: 204

```python
class ASGIHandler(base.BaseHandler):

    async def handle(self, scope, receive, send):
        """
        Handles the ASGI request. Called via the __call__ method.
        """
        # Receive the HTTP request body as a stream object.
        try:
            body_file = await self.read_body(receive)
        except RequestAborted:
            return
        # Request is complete and can be served.
        set_script_prefix(self.get_script_prefix(scope))
        await signals.request_started.asend(sender=self.__class__, scope=scope)
        # Get the request and check for basic issues.
        request, error_response = self.create_request(scope, body_file)
        if request is None:
            body_file.close()
            await self.send_response(error_response, send)
            return
        # Try to catch a disconnect while getting response.
        tasks = [
            asyncio.create_task(self.run_get_response(request)),
            asyncio.create_task(self.listen_for_disconnect(receive)),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        done, pending = done.pop(), pending.pop()
        # Allow views to handle cancellation.
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            # Task re-raised the CancelledError as expected.
            pass
        try:
            response = done.result()
        except RequestAborted:
            body_file.close()
            return
        except AssertionError:
            body_file.close()
            raise
        # Send the response.
        await self.send_response(response, send)
```
### 6 - django/core/asgi.py:

Start line: 1, End line: 14

```python

```
### 7 - django/conf/global_settings.py:

Start line: 405, End line: 480

```python
DATETIME_INPUT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",  # '2006-10-25 14:30:59'
    "%Y-%m-%d %H:%M:%S.%f",  # '2006-10-25 14:30:59.000200'
    "%Y-%m-%d %H:%M",  # '2006-10-25 14:30'
    "%m/%d/%Y %H:%M:%S",  # '10/25/2006 14:30:59'
    "%m/%d/%Y %H:%M:%S.%f",  # '10/25/2006 14:30:59.000200'
    "%m/%d/%Y %H:%M",  # '10/25/2006 14:30'
    "%m/%d/%y %H:%M:%S",  # '10/25/06 14:30:59'
    "%m/%d/%y %H:%M:%S.%f",  # '10/25/06 14:30:59.000200'
    "%m/%d/%y %H:%M",  # '10/25/06 14:30'
]

# First day of week, to be used on calendars
# 0 means Sunday, 1 means Monday...
FIRST_DAY_OF_WEEK = 0

# Decimal separator symbol
DECIMAL_SEPARATOR = "."

# Boolean that sets whether to add thousand separator when formatting numbers
USE_THOUSAND_SEPARATOR = False

# Number of digits that will be together, when splitting them by
# THOUSAND_SEPARATOR. 0 means no grouping, 3 means splitting by thousands...
NUMBER_GROUPING = 0

# Thousand separator symbol
THOUSAND_SEPARATOR = ","

# The tablespaces to use for each model when not specified otherwise.
DEFAULT_TABLESPACE = ""
DEFAULT_INDEX_TABLESPACE = ""

# Default primary key field type.
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

# Default X-Frame-Options header value
X_FRAME_OPTIONS = "DENY"

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

# List of middleware to use. Order is important; in the request phase, these
# middleware will be applied in the order given, and in the response
# phase the middleware will be applied in reverse order.
MIDDLEWARE = []

############
# SESSIONS #
############

# Cache to store session data if using the cache session backend.
SESSION_CACHE_ALIAS = "default"
# Cookie name. This can be whatever you want.
SESSION_COOKIE_NAME = "sessionid"
# Age of cookie, in seconds (default: 2 weeks).
```
### 8 - django/conf/global_settings.py:

Start line: 481, End line: 596

```python
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7 * 2
# A string like "example.com", or None for standard domain cookie.
SESSION_COOKIE_DOMAIN = None
# Whether the session cookie should be secure (https:// only).
SESSION_COOKIE_SECURE = False
# The path of the session cookie.
SESSION_COOKIE_PATH = "/"
# Whether to use the HttpOnly flag.
SESSION_COOKIE_HTTPONLY = True
# Whether to set the flag restricting cookie leaks on cross-site requests.
# This can be 'Lax', 'Strict', 'None', or False to disable the flag.
SESSION_COOKIE_SAMESITE = "Lax"
# Whether to save the session data on every request.
SESSION_SAVE_EVERY_REQUEST = False
# Whether a user's session cookie expires when the web browser is closed.
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
# The module to store session data
SESSION_ENGINE = "django.contrib.sessions.backends.db"
# Directory to store session files if using the file session module. If None,
# the backend will use a sensible default.
SESSION_FILE_PATH = None
# class to serialize session data
SESSION_SERIALIZER = "django.contrib.sessions.serializers.JSONSerializer"

#########
# CACHE #
#########

# The cache backends to use.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}
CACHE_MIDDLEWARE_KEY_PREFIX = ""
CACHE_MIDDLEWARE_SECONDS = 600
CACHE_MIDDLEWARE_ALIAS = "default"

##################
# AUTHENTICATION #
##################

AUTH_USER_MODEL = "auth.User"

AUTHENTICATION_BACKENDS = ["django.contrib.auth.backends.ModelBackend"]

LOGIN_URL = "/accounts/login/"

LOGIN_REDIRECT_URL = "/accounts/profile/"

LOGOUT_REDIRECT_URL = None

# The number of seconds a password reset link is valid for (default: 3 days).
PASSWORD_RESET_TIMEOUT = 60 * 60 * 24 * 3

# the first hasher in this list is the preferred algorithm.  any
# password using different algorithms will be converted automatically
# upon login
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]

AUTH_PASSWORD_VALIDATORS = []

###########
# SIGNING #
###########

SIGNING_BACKEND = "django.core.signing.TimestampSigner"

########
# CSRF #
########

# Dotted path to callable to be used as view when a request is
# rejected by the CSRF middleware.
CSRF_FAILURE_VIEW = "django.views.csrf.csrf_failure"

# Settings for CSRF cookie.
CSRF_COOKIE_NAME = "csrftoken"
CSRF_COOKIE_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_DOMAIN = None
CSRF_COOKIE_PATH = "/"
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_HEADER_NAME = "HTTP_X_CSRFTOKEN"
CSRF_TRUSTED_ORIGINS = []
CSRF_USE_SESSIONS = False

############
# MESSAGES #
############

# Class to use as messages backend
MESSAGE_STORAGE = "django.contrib.messages.storage.fallback.FallbackStorage"

# Default values of MESSAGE_LEVEL and MESSAGE_TAGS are defined within
# django.contrib.messages to avoid imports in this settings file.

###########
# LOGGING #
###########

# The callable to use to configure logging
LOGGING_CONFIG = "logging.config.dictConfig"

# Custom logging configuration.
LOGGING = {}

# Default exception reporter class used in case none has been
# specifically assigned to the HttpRequest instance.
```
### 9 - django/conf/__init__.py:

Start line: 175, End line: 241

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
            "ALLOWED_HOSTS",
            "INSTALLED_APPS",
            "TEMPLATE_DIRS",
            "LOCALE_PATHS",
            "SECRET_KEY_FALLBACKS",
        )
        self._explicit_settings = set()
        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)

                if setting in tuple_settings and not isinstance(
                    setting_value, (list, tuple)
                ):
                    raise ImproperlyConfigured(
                        "The %s setting must be a list or a tuple." % setting
                    )
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

        if hasattr(time, "tzset") and self.TIME_ZONE:
            # When we can, attempt to validate the timezone. If we can't find
            # this file, no check happens and it's harmless.
            zoneinfo_root = Path("/usr/share/zoneinfo")
            zone_info_file = zoneinfo_root.joinpath(*self.TIME_ZONE.split("/"))
            if zoneinfo_root.exists() and not zone_info_file.exists():
                raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
            # Move the time zone info into os.environ. See ticket #2315 for why
            # we don't do this unconditionally (breaks Windows).
            os.environ["TZ"] = self.TIME_ZONE
            time.tzset()

        if self.is_overridden("DEFAULT_FILE_STORAGE"):
            if self.is_overridden("STORAGES"):
                raise ImproperlyConfigured(
                    "DEFAULT_FILE_STORAGE/STORAGES are mutually exclusive."
                )
            warnings.warn(DEFAULT_FILE_STORAGE_DEPRECATED_MSG, RemovedInDjango51Warning)

        if self.is_overridden("STATICFILES_STORAGE"):
            if self.is_overridden("STORAGES"):
                raise ImproperlyConfigured(
                    "STATICFILES_STORAGE/STORAGES are mutually exclusive."
                )
            warnings.warn(STATICFILES_STORAGE_DEPRECATED_MSG, RemovedInDjango51Warning)

    def is_overridden(self, setting):
        return setting in self._explicit_settings

    def __repr__(self):
        return '<%(cls)s "%(settings_module)s">' % {
            "cls": self.__class__.__name__,
            "settings_module": self.SETTINGS_MODULE,
        }
```
### 10 - django/conf/global_settings.py:

Start line: 597, End line: 668

```python
DEFAULT_EXCEPTION_REPORTER = "django.views.debug.ExceptionReporter"

# Default exception reporter filter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER_FILTER = "django.views.debug.SafeExceptionReporterFilter"

###########
# TESTING #
###########

# The name of the class to use to run the test suite
TEST_RUNNER = "django.test.runner.DiscoverRunner"

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
STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
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
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_HSTS_SECONDS = 0
SECURE_REDIRECT_EXEMPT = []
SECURE_REFERRER_POLICY = "same-origin"
SECURE_SSL_HOST = None
SECURE_SSL_REDIRECT = False
```
### 11 - django/core/handlers/asgi.py:

Start line: 331, End line: 355

```python
class ASGIHandler(base.BaseHandler):

    @classmethod
    def chunk_bytes(cls, data):
        """
        Chunks some data up so it can be sent in reasonable size messages.
        Yields (chunk, last_chunk) tuples.
        """
        position = 0
        if not data:
            yield data, True
            return
        while position < len(data):
            yield (
                data[position : position + cls.chunk_size],
                (position + cls.chunk_size) >= len(data),
            )
            position += cls.chunk_size

    def get_script_prefix(self, scope):
        """
        Return the script prefix to use from either the scope or a setting.
        """
        if settings.FORCE_SCRIPT_NAME:
            return settings.FORCE_SCRIPT_NAME
        return scope.get("root_path", "") or ""
```
### 16 - django/core/handlers/asgi.py:

Start line: 138, End line: 161

```python
class ASGIHandler(base.BaseHandler):
    """Handler for ASGI requests."""

    request_class = ASGIRequest
    # Size to chunk response bodies into for multiple response messages.
    chunk_size = 2**16

    def __init__(self):
        super().__init__()
        self.load_middleware(is_async=True)

    async def __call__(self, scope, receive, send):
        """
        Async entrypoint - parses the request and hands off to get_response.
        """
        # Serve only HTTP connections.
        # FIXME: Allow to override this.
        if scope["type"] != "http":
            raise ValueError(
                "Django can only handle ASGI/HTTP connections, not %s." % scope["type"]
            )

        async with ThreadSensitiveContext():
            await self.handle(scope, receive, send)
```
### 27 - django/core/handlers/asgi.py:

Start line: 29, End line: 135

```python
class ASGIRequest(HttpRequest):
    """
    Custom request subclass that decodes from an ASGI-standard request dict
    and wraps request body handling.
    """

    # Number of seconds until a Request gives up on trying to read a request
    # body and aborts.
    body_receive_timeout = 60

    def __init__(self, scope, body_file):
        self.scope = scope
        self._post_parse_error = False
        self._read_started = False
        self.resolver_match = None
        self.script_name = self.scope.get("root_path", "")
        if self.script_name:
            # TODO: Better is-prefix checking, slash handling?
            self.path_info = scope["path"].removeprefix(self.script_name)
        else:
            self.path_info = scope["path"]
        # The Django path is different from ASGI scope path args, it should
        # combine with script name.
        if self.script_name:
            self.path = "%s/%s" % (
                self.script_name.rstrip("/"),
                self.path_info.replace("/", "", 1),
            )
        else:
            self.path = scope["path"]
        # HTTP basics.
        self.method = self.scope["method"].upper()
        # Ensure query string is encoded correctly.
        query_string = self.scope.get("query_string", "")
        if isinstance(query_string, bytes):
            query_string = query_string.decode()
        self.META = {
            "REQUEST_METHOD": self.method,
            "QUERY_STRING": query_string,
            "SCRIPT_NAME": self.script_name,
            "PATH_INFO": self.path_info,
            # WSGI-expecting code will need these for a while
            "wsgi.multithread": True,
            "wsgi.multiprocess": True,
        }
        if self.scope.get("client"):
            self.META["REMOTE_ADDR"] = self.scope["client"][0]
            self.META["REMOTE_HOST"] = self.META["REMOTE_ADDR"]
            self.META["REMOTE_PORT"] = self.scope["client"][1]
        if self.scope.get("server"):
            self.META["SERVER_NAME"] = self.scope["server"][0]
            self.META["SERVER_PORT"] = str(self.scope["server"][1])
        else:
            self.META["SERVER_NAME"] = "unknown"
            self.META["SERVER_PORT"] = "0"
        # Headers go into META.
        for name, value in self.scope.get("headers", []):
            name = name.decode("latin1")
            if name == "content-length":
                corrected_name = "CONTENT_LENGTH"
            elif name == "content-type":
                corrected_name = "CONTENT_TYPE"
            else:
                corrected_name = "HTTP_%s" % name.upper().replace("-", "_")
            # HTTP/2 say only ASCII chars are allowed in headers, but decode
            # latin1 just in case.
            value = value.decode("latin1")
            if corrected_name in self.META:
                value = self.META[corrected_name] + "," + value
            self.META[corrected_name] = value
        # Pull out request encoding, if provided.
        self._set_content_type_params(self.META)
        # Directly assign the body file to be our stream.
        self._stream = body_file
        # Other bits.
        self.resolver_match = None

    @cached_property
    def GET(self):
        return QueryDict(self.META["QUERY_STRING"])

    def _get_scheme(self):
        return self.scope.get("scheme") or super()._get_scheme()

    def _get_post(self):
        if not hasattr(self, "_post"):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    def _get_files(self):
        if not hasattr(self, "_files"):
            self._load_post_and_files()
        return self._files

    POST = property(_get_post, _set_post)
    FILES = property(_get_files)

    @cached_property
    def COOKIES(self):
        return parse_cookie(self.META.get("HTTP_COOKIE", ""))

    def close(self):
        super().close()
        self._stream.close()
```
### 71 - django/core/handlers/asgi.py:

Start line: 275, End line: 329

```python
class ASGIHandler(base.BaseHandler):

    async def send_response(self, response, send):
        """Encode and send a response out over ASGI."""
        # Collect cookies into headers. Have to preserve header case as there
        # are some non-RFC compliant clients that require e.g. Content-Type.
        response_headers = []
        for header, value in response.items():
            if isinstance(header, str):
                header = header.encode("ascii")
            if isinstance(value, str):
                value = value.encode("latin1")
            response_headers.append((bytes(header), bytes(value)))
        for c in response.cookies.values():
            response_headers.append(
                (b"Set-Cookie", c.output(header="").encode("ascii").strip())
            )
        # Initial response message.
        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": response_headers,
            }
        )
        # Streaming responses need to be pinned to their iterator.
        if response.streaming:
            # - Consume via `__aiter__` and not `streaming_content` directly, to
            #   allow mapping of a sync iterator.
            # - Use aclosing() when consuming aiter.
            #   See https://github.com/python/cpython/commit/6e8dcda
            async with aclosing(aiter(response)) as content:
                async for part in content:
                    for chunk, _ in self.chunk_bytes(part):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                # Ignore "more" as there may be more parts; instead,
                                # use an empty final closing message with False.
                                "more_body": True,
                            }
                        )
            # Final closing message.
            await send({"type": "http.response.body"})
        # Other responses just need chunking.
        else:
            # Yield chunks of response.
            for chunk, last in self.chunk_bytes(response.content):
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": not last,
                    }
                )
        await sync_to_async(response.close, thread_sensitive=True)()
```
### 83 - django/core/handlers/asgi.py:

Start line: 246, End line: 273

```python
class ASGIHandler(base.BaseHandler):

    def create_request(self, scope, body_file):
        """
        Create the Request object and returns either (request, None) or
        (None, response) if there is an error response.
        """
        try:
            return self.request_class(scope, body_file), None
        except UnicodeDecodeError:
            logger.warning(
                "Bad Request (UnicodeDecodeError)",
                exc_info=sys.exc_info(),
                extra={"status_code": 400},
            )
            return None, HttpResponseBadRequest()
        except RequestDataTooBig:
            return None, HttpResponse("413 Payload too large", status=413)

    def handle_uncaught_exception(self, request, resolver, exc_info):
        """Last-chance handler for exceptions."""
        # There's no WSGI server to catch the exception further up
        # if this fails, so translate it into a plain text response.
        try:
            return super().handle_uncaught_exception(request, resolver, exc_info)
        except Exception:
            return HttpResponseServerError(
                traceback.format_exc() if settings.DEBUG else "Internal Server Error",
                content_type="text/plain",
            )
```
### 167 - django/core/handlers/asgi.py:

Start line: 225, End line: 244

```python
class ASGIHandler(base.BaseHandler):

    async def read_body(self, receive):
        """Reads an HTTP body from an ASGI connection."""
        # Use the tempfile that auto rolls-over to a disk file as it fills up.
        body_file = tempfile.SpooledTemporaryFile(
            max_size=settings.FILE_UPLOAD_MAX_MEMORY_SIZE, mode="w+b"
        )
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                body_file.close()
                # Early client disconnect.
                raise RequestAborted()
            # Add a body chunk from the message, if provided.
            if "body" in message:
                body_file.write(message["body"])
            # Quit out if that's the end.
            if not message.get("more_body", False):
                break
        body_file.seek(0)
        return body_file
```
