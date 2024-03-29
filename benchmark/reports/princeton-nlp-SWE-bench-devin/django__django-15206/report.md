# django__django-15206

| **django/django** | `bf7afe9c4e21f5fe5090c47b2b6ffc5a03a85815` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1437 |
| **Any found context length** | 119 |
| **Avg pos** | 10.0 |
| **Min pos** | 1 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/decorators/cache.py b/django/views/decorators/cache.py
--- a/django/views/decorators/cache.py
+++ b/django/views/decorators/cache.py
@@ -1,6 +1,5 @@
 from functools import wraps
 
-from django.http import HttpRequest
 from django.middleware.cache import CacheMiddleware
 from django.utils.cache import add_never_cache_headers, patch_cache_control
 from django.utils.decorators import decorator_from_middleware_with_args
@@ -29,7 +28,8 @@ def cache_control(**kwargs):
     def _cache_controller(viewfunc):
         @wraps(viewfunc)
         def _cache_controlled(request, *args, **kw):
-            if not isinstance(request, HttpRequest):
+            # Ensure argument looks like a request.
+            if not hasattr(request, 'META'):
                 raise TypeError(
                     "cache_control didn't receive an HttpRequest. If you are "
                     "decorating a classmethod, be sure to use "
@@ -48,7 +48,8 @@ def never_cache(view_func):
     """
     @wraps(view_func)
     def _wrapped_view_func(request, *args, **kwargs):
-        if not isinstance(request, HttpRequest):
+        # Ensure argument looks like a request.
+        if not hasattr(request, 'META'):
             raise TypeError(
                 "never_cache didn't receive an HttpRequest. If you are "
                 "decorating a classmethod, be sure to use @method_decorator."

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/decorators/cache.py | 3 | 3 | 7 | 1 | 1437
| django/views/decorators/cache.py | 32 | 32 | 1 | 1 | 119
| django/views/decorators/cache.py | 51 | 51 | 2 | 1 | 242


## Problem Statement

```
never_cache()/cache_control() decorators raise error on duck-typed requests.
Description
	
The cache decorators cache_control, never_cache and sensitive_post_parameters no longer work with Django REST framework because they strictly check for an HttpRequest instance.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/views/decorators/cache.py** | 28 | 42| 119 | 119 | 459 | 
| **-> 2 <-** | **1 django/views/decorators/cache.py** | 45 | 60| 123 | 242 | 459 | 
| 3 | 2 django/utils/cache.py | 1 | 34| 279 | 521 | 4213 | 
| 4 | 3 django/core/checks/caches.py | 22 | 56| 291 | 812 | 4734 | 
| 5 | 3 django/utils/cache.py | 137 | 152| 190 | 1002 | 4734 | 
| 6 | 4 django/core/cache/__init__.py | 1 | 27| 220 | 1222 | 5203 | 
| **-> 7 <-** | **4 django/views/decorators/cache.py** | 1 | 25| 215 | 1437 | 5203 | 
| 8 | 5 django/middleware/cache.py | 1 | 52| 431 | 1868 | 6796 | 
| 9 | 5 django/utils/cache.py | 246 | 277| 257 | 2125 | 6796 | 
| 10 | 5 django/core/cache/__init__.py | 30 | 67| 249 | 2374 | 6796 | 
| 11 | 5 django/core/checks/caches.py | 1 | 19| 119 | 2493 | 6796 | 
| 12 | 6 django/views/decorators/debug.py | 77 | 93| 138 | 2631 | 7391 | 
| 13 | 7 django/core/cache/backends/dummy.py | 1 | 35| 231 | 2862 | 7623 | 
| 14 | 7 django/core/checks/caches.py | 59 | 73| 109 | 2971 | 7623 | 
| 15 | 8 django/core/cache/backends/base.py | 373 | 386| 112 | 3083 | 10699 | 
| 16 | 9 django/core/cache/backends/redis.py | 159 | 231| 643 | 3726 | 12419 | 
| 17 | 10 django/http/request.py | 1 | 39| 273 | 3999 | 17625 | 
| 18 | 11 django/views/csrf.py | 15 | 100| 835 | 4834 | 19169 | 
| 19 | 11 django/middleware/cache.py | 137 | 163| 252 | 5086 | 19169 | 
| 20 | 12 django/core/cache/backends/memcached.py | 66 | 95| 339 | 5425 | 20647 | 
| 21 | 13 django/core/cache/backends/db.py | 192 | 218| 279 | 5704 | 22756 | 
| 22 | 14 django/http/__init__.py | 1 | 22| 197 | 5901 | 22953 | 
| 23 | 15 django/db/models/fields/mixins.py | 1 | 28| 168 | 6069 | 23296 | 
| 24 | 15 django/core/cache/backends/db.py | 106 | 190| 797 | 6866 | 23296 | 
| 25 | 15 django/core/cache/backends/memcached.py | 1 | 37| 220 | 7086 | 23296 | 
| 26 | 16 django/core/cache/utils.py | 1 | 13| 0 | 7086 | 23384 | 
| 27 | 16 django/core/cache/backends/memcached.py | 97 | 112| 152 | 7238 | 23384 | 
| 28 | 16 django/core/cache/backends/db.py | 1 | 37| 229 | 7467 | 23384 | 
| 29 | 16 django/utils/cache.py | 155 | 195| 453 | 7920 | 23384 | 
| 30 | 16 django/core/cache/backends/db.py | 220 | 239| 245 | 8165 | 23384 | 
| 31 | 16 django/core/cache/backends/db.py | 40 | 92| 423 | 8588 | 23384 | 
| 32 | 16 django/core/cache/backends/db.py | 241 | 274| 348 | 8936 | 23384 | 
| 33 | 16 django/core/cache/backends/memcached.py | 114 | 133| 187 | 9123 | 23384 | 
| 34 | 17 django/utils/deprecation.py | 76 | 126| 372 | 9495 | 24409 | 
| 35 | 17 django/middleware/cache.py | 120 | 135| 132 | 9627 | 24409 | 
| 36 | 17 django/views/decorators/debug.py | 47 | 75| 199 | 9826 | 24409 | 
| 37 | 17 django/core/cache/backends/base.py | 1 | 53| 265 | 10091 | 24409 | 
| 38 | 18 django/http/response.py | 560 | 585| 159 | 10250 | 29123 | 
| 39 | 18 django/middleware/cache.py | 55 | 76| 181 | 10431 | 29123 | 
| 40 | 18 django/core/cache/backends/base.py | 56 | 95| 313 | 10744 | 29123 | 
| 41 | 18 django/middleware/cache.py | 166 | 198| 246 | 10990 | 29123 | 
| 42 | 18 django/utils/cache.py | 37 | 102| 559 | 11549 | 29123 | 
| 43 | 19 django/core/checks/security/csrf.py | 1 | 42| 304 | 11853 | 29585 | 
| 44 | 19 django/core/cache/backends/db.py | 94 | 104| 222 | 12075 | 29585 | 
| 45 | 20 django/core/cache/backends/locmem.py | 1 | 63| 495 | 12570 | 30470 | 
| 46 | 21 django/views/decorators/http.py | 77 | 124| 366 | 12936 | 31441 | 
| 47 | 22 django/utils/formats.py | 1 | 59| 380 | 13316 | 33906 | 
| 48 | 23 django/views/defaults.py | 102 | 121| 149 | 13465 | 34934 | 
| 49 | 23 django/views/decorators/http.py | 1 | 52| 350 | 13815 | 34934 | 
| 50 | 23 django/http/request.py | 375 | 398| 185 | 14000 | 34934 | 
| 51 | 24 django/views/debug.py | 164 | 187| 177 | 14177 | 39687 | 
| 52 | 24 django/middleware/cache.py | 78 | 117| 363 | 14540 | 39687 | 
| 53 | 24 django/views/debug.py | 189 | 201| 143 | 14683 | 39687 | 
| 54 | 24 django/http/request.py | 290 | 310| 189 | 14872 | 39687 | 
| 55 | 24 django/utils/cache.py | 305 | 325| 211 | 15083 | 39687 | 
| 56 | 24 django/views/decorators/debug.py | 1 | 44| 274 | 15357 | 39687 | 
| 57 | 24 django/utils/cache.py | 328 | 346| 218 | 15575 | 39687 | 
| 58 | 24 django/http/request.py | 219 | 288| 504 | 16079 | 39687 | 
| 59 | 25 django/utils/functional.py | 1 | 49| 336 | 16415 | 42782 | 
| 60 | 25 django/views/debug.py | 88 | 120| 258 | 16673 | 42782 | 
| 61 | 25 django/views/debug.py | 150 | 162| 148 | 16821 | 42782 | 
| 62 | 26 django/core/checks/security/base.py | 1 | 72| 660 | 17481 | 44823 | 
| 63 | 26 django/core/cache/backends/base.py | 110 | 176| 614 | 18095 | 44823 | 
| 64 | 27 django/middleware/csrf.py | 346 | 409| 585 | 18680 | 48976 | 
| 65 | 28 django/views/decorators/clickjacking.py | 1 | 19| 138 | 18818 | 49352 | 
| 66 | 28 django/core/cache/backends/locmem.py | 79 | 118| 268 | 19086 | 49352 | 
| 67 | 28 django/core/cache/backends/locmem.py | 65 | 77| 134 | 19220 | 49352 | 
| 68 | 28 django/views/defaults.py | 1 | 24| 149 | 19369 | 49352 | 
| 69 | 28 django/utils/cache.py | 105 | 134| 190 | 19559 | 49352 | 
| 70 | 28 django/middleware/csrf.py | 411 | 464| 572 | 20131 | 49352 | 
| 71 | 28 django/utils/cache.py | 371 | 417| 521 | 20652 | 49352 | 
| 72 | 29 django/core/handlers/exception.py | 54 | 122| 557 | 21209 | 50420 | 
| 73 | 29 django/middleware/csrf.py | 294 | 344| 450 | 21659 | 50420 | 
| 74 | 29 django/core/checks/security/base.py | 74 | 168| 746 | 22405 | 50420 | 
| 75 | 30 django/templatetags/cache.py | 1 | 49| 413 | 22818 | 51147 | 
| 76 | 30 django/views/debug.py | 72 | 85| 115 | 22933 | 51147 | 
| 77 | 31 django/core/handlers/base.py | 320 | 351| 212 | 23145 | 53754 | 
| 78 | 32 django/conf/__init__.py | 102 | 116| 129 | 23274 | 56113 | 
| 79 | 32 django/core/cache/backends/memcached.py | 39 | 64| 283 | 23557 | 56113 | 
| 80 | 33 django/conf/global_settings.py | 407 | 501| 782 | 24339 | 61920 | 
| 81 | 33 django/core/cache/backends/redis.py | 31 | 136| 770 | 25109 | 61920 | 
| 82 | 34 django/core/handlers/wsgi.py | 64 | 119| 486 | 25595 | 63679 | 
| 83 | 35 django/contrib/contenttypes/fields.py | 1 | 18| 149 | 25744 | 69234 | 
| 84 | 35 django/utils/cache.py | 349 | 368| 190 | 25934 | 69234 | 
| 85 | 35 django/core/cache/backends/redis.py | 1 | 28| 165 | 26099 | 69234 | 
| 86 | 35 django/http/response.py | 526 | 557| 153 | 26252 | 69234 | 
| 87 | 36 django/db/models/base.py | 1 | 50| 328 | 26580 | 86852 | 
| 88 | 36 django/http/response.py | 65 | 82| 133 | 26713 | 86852 | 
| 89 | 36 django/core/checks/security/csrf.py | 45 | 68| 157 | 26870 | 86852 | 
| 90 | 36 django/core/checks/security/base.py | 171 | 196| 188 | 27058 | 86852 | 
| 91 | 37 django/http/cookie.py | 1 | 24| 156 | 27214 | 87009 | 
| 92 | 37 django/middleware/csrf.py | 1 | 54| 482 | 27696 | 87009 | 
| 93 | 37 django/views/csrf.py | 1 | 13| 132 | 27828 | 87009 | 
| 94 | 37 django/core/cache/backends/base.py | 233 | 283| 457 | 28285 | 87009 | 
| 95 | 37 django/core/checks/security/base.py | 214 | 231| 127 | 28412 | 87009 | 
| 96 | 38 django/views/decorators/csrf.py | 1 | 57| 460 | 28872 | 87469 | 
| 97 | 38 django/middleware/csrf.py | 166 | 216| 387 | 29259 | 87469 | 
| 98 | 38 django/views/defaults.py | 124 | 151| 198 | 29457 | 87469 | 
| 99 | 38 django/http/response.py | 1 | 24| 162 | 29619 | 87469 | 
| 100 | 38 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 29742 | 87469 | 
| 101 | 38 django/middleware/csrf.py | 250 | 266| 186 | 29928 | 87469 | 
| 102 | 38 django/http/request.py | 160 | 181| 164 | 30092 | 87469 | 
| 103 | 38 django/http/request.py | 341 | 372| 255 | 30347 | 87469 | 
| 104 | 39 django/template/context.py | 1 | 24| 128 | 30475 | 89350 | 
| 105 | 40 django/contrib/auth/admin.py | 1 | 22| 188 | 30663 | 91087 | 
| 106 | 41 django/contrib/contenttypes/models.py | 118 | 130| 133 | 30796 | 92502 | 
| 107 | 41 django/conf/global_settings.py | 502 | 632| 799 | 31595 | 92502 | 
| 108 | 42 django/db/models/fields/__init__.py | 1117 | 1149| 237 | 31832 | 110670 | 
| 109 | 42 django/core/cache/backends/memcached.py | 136 | 158| 207 | 32039 | 110670 | 
| 110 | 43 django/views/decorators/gzip.py | 1 | 6| 0 | 32039 | 110721 | 
| 111 | 44 django/contrib/gis/views.py | 1 | 21| 155 | 32194 | 110876 | 
| 112 | 45 django/core/cache/backends/filebased.py | 1 | 44| 289 | 32483 | 112109 | 
| 113 | 45 django/core/cache/backends/filebased.py | 46 | 59| 145 | 32628 | 112109 | 
| 114 | 46 django/utils/decorators.py | 114 | 152| 316 | 32944 | 113511 | 
| 115 | 47 django/urls/resolvers.py | 451 | 479| 288 | 33232 | 119330 | 
| 116 | 47 django/middleware/csrf.py | 466 | 479| 163 | 33395 | 119330 | 
| 117 | 47 django/views/decorators/clickjacking.py | 22 | 54| 238 | 33633 | 119330 | 
| 118 | 48 django/core/handlers/asgi.py | 1 | 19| 111 | 33744 | 121666 | 
| 119 | 48 django/views/defaults.py | 81 | 99| 129 | 33873 | 121666 | 
| 120 | 49 django/contrib/sites/models.py | 1 | 22| 130 | 34003 | 122454 | 
| 121 | 50 django/contrib/auth/views.py | 1 | 37| 284 | 34287 | 125184 | 
| 122 | 51 django/core/checks/security/sessions.py | 1 | 98| 572 | 34859 | 125757 | 
| 123 | 51 django/contrib/sites/models.py | 25 | 46| 192 | 35051 | 125757 | 
| 124 | 52 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 35218 | 126324 | 
| 125 | 52 django/views/decorators/http.py | 55 | 76| 272 | 35490 | 126324 | 
| 126 | 52 django/core/cache/backends/base.py | 301 | 324| 203 | 35693 | 126324 | 
| 127 | 52 django/middleware/csrf.py | 118 | 141| 181 | 35874 | 126324 | 
| 128 | 52 django/core/cache/backends/filebased.py | 98 | 114| 174 | 36048 | 126324 | 
| 129 | 53 django/db/models/fields/related.py | 1 | 34| 246 | 36294 | 140483 | 
| 130 | 53 django/core/cache/backends/filebased.py | 61 | 96| 260 | 36554 | 140483 | 
| 131 | 53 django/contrib/sites/models.py | 78 | 121| 236 | 36790 | 140483 | 
| 132 | 53 django/templatetags/cache.py | 52 | 94| 313 | 37103 | 140483 | 
| 133 | 53 django/conf/__init__.py | 261 | 269| 121 | 37224 | 140483 | 
| 134 | 53 django/utils/deprecation.py | 128 | 146| 122 | 37346 | 140483 | 
| 135 | 54 django/middleware/security.py | 1 | 29| 278 | 37624 | 141001 | 
| 136 | 54 django/core/cache/backends/redis.py | 138 | 156| 147 | 37771 | 141001 | 
| 137 | 54 django/views/csrf.py | 101 | 155| 577 | 38348 | 141001 | 
| 138 | 55 django/core/servers/basehttp.py | 181 | 200| 170 | 38518 | 142905 | 
| 139 | 55 django/core/checks/security/base.py | 234 | 258| 210 | 38728 | 142905 | 
| 140 | 55 django/core/cache/backends/memcached.py | 161 | 173| 120 | 38848 | 142905 | 
| 141 | 56 django/db/backends/utils.py | 1 | 46| 287 | 39135 | 144893 | 
| 142 | 56 django/views/debug.py | 122 | 148| 216 | 39351 | 144893 | 
| 143 | 57 django/contrib/admin/options.py | 1 | 96| 756 | 40107 | 163683 | 
| 144 | 58 django/db/utils.py | 52 | 98| 312 | 40419 | 165701 | 
| 145 | 59 django/contrib/auth/hashers.py | 86 | 108| 167 | 40586 | 171664 | 
| 146 | 60 django/contrib/admin/sites.py | 1 | 35| 225 | 40811 | 176093 | 
| 147 | 61 django/core/checks/model_checks.py | 178 | 211| 332 | 41143 | 177878 | 
| 148 | 62 django/core/checks/templates.py | 1 | 41| 301 | 41444 | 178343 | 
| 149 | 62 django/core/cache/backends/base.py | 285 | 299| 147 | 41591 | 178343 | 
| 150 | 62 django/db/models/base.py | 385 | 421| 293 | 41884 | 178343 | 
| 151 | 62 django/conf/__init__.py | 78 | 100| 221 | 42105 | 178343 | 
| 152 | 62 django/conf/global_settings.py | 157 | 272| 859 | 42964 | 178343 | 
| 153 | 62 django/views/debug.py | 203 | 251| 467 | 43431 | 178343 | 
| 154 | 62 django/db/utils.py | 1 | 49| 177 | 43608 | 178343 | 
| 155 | 62 django/middleware/csrf.py | 268 | 292| 176 | 43784 | 178343 | 
| 156 | 63 django/urls/__init__.py | 1 | 24| 239 | 44023 | 178582 | 
| 157 | 63 django/http/request.py | 151 | 158| 111 | 44134 | 178582 | 
| 158 | 63 django/db/models/fields/related.py | 120 | 137| 155 | 44289 | 178582 | 
| 159 | 63 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 44487 | 178582 | 
| 160 | 63 django/core/cache/backends/base.py | 97 | 108| 119 | 44606 | 178582 | 
| 161 | 64 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 44744 | 178721 | 
| 162 | 64 django/conf/global_settings.py | 633 | 657| 184 | 44928 | 178721 | 
| 163 | 64 django/utils/decorators.py | 89 | 111| 152 | 45080 | 178721 | 
| 164 | 64 django/middleware/security.py | 31 | 58| 246 | 45326 | 178721 | 
| 165 | 64 django/utils/cache.py | 219 | 243| 235 | 45561 | 178721 | 
| 166 | 65 django/contrib/auth/mixins.py | 107 | 129| 146 | 45707 | 179585 | 
| 167 | 66 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 45828 | 180441 | 
| 168 | 66 django/db/models/fields/related.py | 139 | 166| 201 | 46029 | 180441 | 
| 169 | 66 django/core/management/commands/createcachetable.py | 1 | 29| 213 | 46242 | 180441 | 
| 170 | 67 django/middleware/clickjacking.py | 1 | 48| 364 | 46606 | 180805 | 
| 171 | 67 django/utils/functional.py | 131 | 199| 520 | 47126 | 180805 | 
| 172 | 68 django/core/exceptions.py | 107 | 218| 752 | 47878 | 181994 | 
| 173 | 68 django/utils/cache.py | 198 | 216| 184 | 48062 | 181994 | 
| 174 | 69 django/middleware/http.py | 1 | 42| 335 | 48397 | 182329 | 
| 175 | 70 django/db/models/constraints.py | 38 | 90| 416 | 48813 | 184390 | 
| 176 | 71 django/contrib/auth/password_validation.py | 1 | 32| 209 | 49022 | 185914 | 
| 177 | 72 django/views/generic/base.py | 191 | 222| 247 | 49269 | 187564 | 
| 178 | 73 django/utils/http.py | 1 | 37| 464 | 49733 | 190806 | 
| 179 | 74 django/utils/autoreload.py | 59 | 87| 156 | 49889 | 195954 | 
| 180 | 74 django/contrib/auth/mixins.py | 44 | 71| 235 | 50124 | 195954 | 
| 181 | 75 django/urls/exceptions.py | 1 | 10| 0 | 50124 | 195979 | 
| 182 | 75 django/core/handlers/asgi.py | 22 | 123| 872 | 50996 | 195979 | 
| 183 | 75 django/core/checks/model_checks.py | 129 | 153| 268 | 51264 | 195979 | 
| 184 | 75 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 51796 | 195979 | 
| 185 | 76 django/contrib/sessions/middleware.py | 1 | 76| 588 | 52384 | 196568 | 
| 186 | 77 django/contrib/sitemaps/views.py | 1 | 34| 242 | 52626 | 197707 | 
| 187 | 78 django/core/checks/__init__.py | 1 | 28| 307 | 52933 | 198014 | 
| 188 | 78 django/core/exceptions.py | 1 | 104| 436 | 53369 | 198014 | 
| 189 | 79 django/template/response.py | 1 | 43| 389 | 53758 | 199105 | 
| 190 | 79 django/utils/cache.py | 280 | 302| 258 | 54016 | 199105 | 
| 191 | 79 django/http/response.py | 342 | 377| 212 | 54228 | 199105 | 
| 192 | 79 django/db/models/base.py | 1166 | 1193| 286 | 54514 | 199105 | 
| 193 | 79 django/views/generic/base.py | 1 | 26| 142 | 54656 | 199105 | 
| 194 | 80 django/utils/inspect.py | 1 | 24| 148 | 54804 | 199618 | 
| 195 | 80 django/utils/decorators.py | 1 | 19| 138 | 54942 | 199618 | 
| 196 | 80 django/urls/resolvers.py | 645 | 655| 120 | 55062 | 199618 | 
| 197 | 80 django/conf/__init__.py | 152 | 171| 172 | 55234 | 199618 | 


### Hint

```
Hi Terence — Thanks for the report. Can you post a traceback quickly, or better a reproduce? (But a traceback would be good.)
3fd82a62415e748002435e7bad06b5017507777c for #32468 looks likely.
The check has been in sensitive_post_parameters ​for a long time.
I don't see how this can be a regression, these decorators use functions from ​django.utils.cache which are documented as accepting HttpResponse 🤔 Also Response defined in django-rest-framework is a subclass of HttpResponse.
Replying to Mariusz Felisiak: I don't see how this can be a regression, these decorators use functions from ​django.utils.cache which are documented as accepting HttpResponse 🤔 Also Response defined in django-rest-framework is a subclass of HttpResponse. ​Request has never inherited from HttpRequest which has been the source of many issues like this even before 4.0. I am experiencing a lot of issues right now converting Django method views into DRF class-based views right now because all of the decorators keep breaking.
Replying to jason: ​Request has never inherited from HttpRequest which has been the source of many issues like this even before 4.0. I am experiencing a lot of issues right now converting Django method views into DRF class-based views right now because all of the decorators keep breaking. Sorry, I confused HttpRequest with HttpResponse. Nevertheless, IMO, if DRF is duck typing HttpRequest it should be fixed in DRF, e.g. by adding __instancecheck__(). What do you think Tom? I'm happy to provide a patch to DRF.
I think this is going to cause trouble for a lot of folks. The sensitive_post_parameters has been as it is for many years, but the ​caching usage is standard DRF. (Again that's been there a long time.) The check in 3fd82a62415e748002435e7bad06b5017507777c seems overly tight. What methods are being called? Surely any Request-a-like exposing those is OK? I don't think we can just change Request to claim to be a HttpRequest. ​DRF has an `isinstance()` check for this very thing. Introduced in ​https://github.com/encode/django-rest-framework/pull/5618 See also: ​https://github.com/encode/django-rest-framework/issues/5446 ​https://github.com/encode/django-rest-framework/issues/3848 (and others)
This error was added to catch misuses of never_cache() and cache_control() decorators without method_decorator().
We've discussed this with Carlton, and see two possible solutions: allow for HttpRequest duck-typing, or changing errors to be based on missing method_decorator() call, maybe checking that request is not a callable 🤔 or sth similar.
Replying to Mariusz Felisiak: This error was added to catch misuses of never_cache() and cache_control() decorators without method_decorator(). Breaking functionality in order to provide a nice error message seems backwards. Then again, I can't think of a good excuse for DRF not to simply inherit from HttpRequest. I recently fell down a rabbit-hole of github issues that were closed without resolution regarding this conflict, where DRF refuses to inherit because "composition pattern", and django refuses not to check because "nice error message" over and over. Something has to give right?
Something has to give right? This ticket is accepted. Have you seen my last comment? We agreed that this should be fixed in Django itself.
Replying to Mariusz Felisiak: Something has to give right? This ticket is accepted. Have you seen my last comment? We agreed that this should be fixed in Django itself. Just saw it, don't mean to beat a dead horse, thank you for your quick response _
```

## Patch

```diff
diff --git a/django/views/decorators/cache.py b/django/views/decorators/cache.py
--- a/django/views/decorators/cache.py
+++ b/django/views/decorators/cache.py
@@ -1,6 +1,5 @@
 from functools import wraps
 
-from django.http import HttpRequest
 from django.middleware.cache import CacheMiddleware
 from django.utils.cache import add_never_cache_headers, patch_cache_control
 from django.utils.decorators import decorator_from_middleware_with_args
@@ -29,7 +28,8 @@ def cache_control(**kwargs):
     def _cache_controller(viewfunc):
         @wraps(viewfunc)
         def _cache_controlled(request, *args, **kw):
-            if not isinstance(request, HttpRequest):
+            # Ensure argument looks like a request.
+            if not hasattr(request, 'META'):
                 raise TypeError(
                     "cache_control didn't receive an HttpRequest. If you are "
                     "decorating a classmethod, be sure to use "
@@ -48,7 +48,8 @@ def never_cache(view_func):
     """
     @wraps(view_func)
     def _wrapped_view_func(request, *args, **kwargs):
-        if not isinstance(request, HttpRequest):
+        # Ensure argument looks like a request.
+        if not hasattr(request, 'META'):
             raise TypeError(
                 "never_cache didn't receive an HttpRequest. If you are "
                 "decorating a classmethod, be sure to use @method_decorator."

```

## Test Patch

```diff
diff --git a/tests/decorators/tests.py b/tests/decorators/tests.py
--- a/tests/decorators/tests.py
+++ b/tests/decorators/tests.py
@@ -493,6 +493,15 @@ def a_view(request):
         self.assertIsNone(r.get('X-Frame-Options', None))
 
 
+class HttpRequestProxy:
+    def __init__(self, request):
+        self._request = request
+
+    def __getattr__(self, attr):
+        """Proxy to the underlying HttpRequest object."""
+        return getattr(self._request, attr)
+
+
 class NeverCacheDecoratorTest(SimpleTestCase):
 
     @mock.patch('time.time')
@@ -525,12 +534,27 @@ class MyClass:
             @never_cache
             def a_view(self, request):
                 return HttpResponse()
+
+        request = HttpRequest()
         msg = (
             "never_cache didn't receive an HttpRequest. If you are decorating "
             "a classmethod, be sure to use @method_decorator."
         )
         with self.assertRaisesMessage(TypeError, msg):
-            MyClass().a_view(HttpRequest())
+            MyClass().a_view(request)
+        with self.assertRaisesMessage(TypeError, msg):
+            MyClass().a_view(HttpRequestProxy(request))
+
+    def test_never_cache_decorator_http_request_proxy(self):
+        class MyClass:
+            @method_decorator(never_cache)
+            def a_view(self, request):
+                return HttpResponse()
+
+        request = HttpRequest()
+        response = MyClass().a_view(HttpRequestProxy(request))
+        self.assertIn('Cache-Control', response.headers)
+        self.assertIn('Expires', response.headers)
 
 
 class CacheControlDecoratorTest(SimpleTestCase):
@@ -544,5 +568,18 @@ def a_view(self, request):
             "cache_control didn't receive an HttpRequest. If you are "
             "decorating a classmethod, be sure to use @method_decorator."
         )
+        request = HttpRequest()
         with self.assertRaisesMessage(TypeError, msg):
-            MyClass().a_view(HttpRequest())
+            MyClass().a_view(request)
+        with self.assertRaisesMessage(TypeError, msg):
+            MyClass().a_view(HttpRequestProxy(request))
+
+    def test_cache_control_decorator_http_request_proxy(self):
+        class MyClass:
+            @method_decorator(cache_control(a='b'))
+            def a_view(self, request):
+                return HttpResponse()
+
+        request = HttpRequest()
+        response = MyClass().a_view(HttpRequestProxy(request))
+        self.assertEqual(response.headers['Cache-Control'], 'a=b')

```


## Code snippets

### 1 - django/views/decorators/cache.py:

Start line: 28, End line: 42

```python
def cache_control(**kwargs):
    def _cache_controller(viewfunc):
        @wraps(viewfunc)
        def _cache_controlled(request, *args, **kw):
            if not isinstance(request, HttpRequest):
                raise TypeError(
                    "cache_control didn't receive an HttpRequest. If you are "
                    "decorating a classmethod, be sure to use "
                    "@method_decorator."
                )
            response = viewfunc(request, *args, **kw)
            patch_cache_control(response, **kwargs)
            return response
        return _cache_controlled
    return _cache_controller
```
### 2 - django/views/decorators/cache.py:

Start line: 45, End line: 60

```python
def never_cache(view_func):
    """
    Decorator that adds headers to a response so that it will never be cached.
    """
    @wraps(view_func)
    def _wrapped_view_func(request, *args, **kwargs):
        if not isinstance(request, HttpRequest):
            raise TypeError(
                "never_cache didn't receive an HttpRequest. If you are "
                "decorating a classmethod, be sure to use @method_decorator."
            )
        response = view_func(request, *args, **kwargs)
        add_never_cache_headers(response)
        return response
    return _wrapped_view_func
```
### 3 - django/utils/cache.py:

Start line: 1, End line: 34

```python
"""
This module contains helper functions for controlling caching. It does so by
managing the "Vary" header of responses. It includes functions to patch the
header of response objects directly and decorators that change functions to do
that header-patching themselves.

For information on the Vary header, see:

    https://tools.ietf.org/html/rfc7231#section-7.1.4

Essentially, the "Vary" HTTP header defines which headers a cache should take
into account when building its cache key. Requests with the same path but
different header content for headers named in "Vary" need to get different
cache keys to prevent delivery of wrong content.

An example: i18n middleware would need to distinguish caches by the
"Accept-language" header.
"""
import time
from collections import defaultdict

from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.crypto import md5
from django.utils.http import (
    http_date, parse_etags, parse_http_date_safe, quote_etag,
)
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language

cc_delim_re = _lazy_re_compile(r'\s*,\s*')
```
### 4 - django/core/checks/caches.py:

Start line: 22, End line: 56

```python
@register(Tags.caches, deploy=True)
def check_cache_location_not_exposed(app_configs, **kwargs):
    errors = []
    for name in ('MEDIA_ROOT', 'STATIC_ROOT', 'STATICFILES_DIRS'):
        setting = getattr(settings, name, None)
        if not setting:
            continue
        if name == 'STATICFILES_DIRS':
            paths = set()
            for staticfiles_dir in setting:
                if isinstance(staticfiles_dir, (list, tuple)):
                    _, staticfiles_dir = staticfiles_dir
                paths.add(pathlib.Path(staticfiles_dir).resolve())
        else:
            paths = {pathlib.Path(setting).resolve()}
        for alias in settings.CACHES:
            cache = caches[alias]
            if not isinstance(cache, FileBasedCache):
                continue
            cache_path = pathlib.Path(cache._dir).resolve()
            if any(path == cache_path for path in paths):
                relation = 'matches'
            elif any(path in cache_path.parents for path in paths):
                relation = 'is inside'
            elif any(cache_path in path.parents for path in paths):
                relation = 'contains'
            else:
                continue
            errors.append(Warning(
                f"Your '{alias}' cache configuration might expose your cache "
                f"or lead to corruption of your data because its LOCATION "
                f"{relation} {name}.",
                id='caches.W002',
            ))
    return errors
```
### 5 - django/utils/cache.py:

Start line: 137, End line: 152

```python
def _not_modified(request, response=None):
    new_response = HttpResponseNotModified()
    if response:
        # Preserve the headers required by Section 4.1 of RFC 7232, as well as
        # Last-Modified.
        for header in ('Cache-Control', 'Content-Location', 'Date', 'ETag', 'Expires', 'Last-Modified', 'Vary'):
            if header in response:
                new_response.headers[header] = response.headers[header]

        # Preserve cookies as per the cookie specification: "If a proxy server
        # receives a response which contains a Set-cookie header, it should
        # propagate the Set-cookie header to the client, regardless of whether
        # the response was 304 (Not Modified) or 200 (OK).
        # https://curl.haxx.se/rfc/cookie_spec.html
        new_response.cookies = response.cookies
    return new_response
```
### 6 - django/core/cache/__init__.py:

Start line: 1, End line: 27

```python
"""
Caching framework.

This package defines set of cache backends that all conform to a simple API.
In a nutshell, a cache is a set of values -- which can be any object that
may be pickled -- identified by string keys.  For the complete API, see
the abstract BaseCache class in django.core.cache.backends.base.

Client code should use the `cache` variable defined here to access the default
cache backend and look up non-default cache backends in the `caches` dict-like
object.

See docs/topics/cache.txt for information on the public API.
"""
from django.core import signals
from django.core.cache.backends.base import (
    BaseCache, CacheKeyWarning, InvalidCacheBackendError, InvalidCacheKey,
)
from django.utils.connection import BaseConnectionHandler, ConnectionProxy
from django.utils.module_loading import import_string

__all__ = [
    'cache', 'caches', 'DEFAULT_CACHE_ALIAS', 'InvalidCacheBackendError',
    'CacheKeyWarning', 'BaseCache', 'InvalidCacheKey',
]

DEFAULT_CACHE_ALIAS = 'default'
```
### 7 - django/views/decorators/cache.py:

Start line: 1, End line: 25

```python
from functools import wraps

from django.http import HttpRequest
from django.middleware.cache import CacheMiddleware
from django.utils.cache import add_never_cache_headers, patch_cache_control
from django.utils.decorators import decorator_from_middleware_with_args


def cache_page(timeout, *, cache=None, key_prefix=None):
    """
    Decorator for views that tries getting the page from the cache and
    populates the cache if the page isn't in the cache yet.

    The cache is keyed by the URL and some data from the headers.
    Additionally there is the key prefix that is used to distinguish different
    cache areas in a multi-site setup. You could use the
    get_current_site().domain, for example, as that is unique across a Django
    project.

    Additionally, all headers from the response's Vary header will be taken
    into account on caching -- just like the middleware does.
    """
    return decorator_from_middleware_with_args(CacheMiddleware)(
        page_timeout=timeout, cache_alias=cache, key_prefix=key_prefix,
    )
```
### 8 - django/middleware/cache.py:

Start line: 1, End line: 52

```python
"""
Cache middleware. If enabled, each Django-powered page will be cached based on
URL. The canonical way to enable cache middleware is to set
``UpdateCacheMiddleware`` as your first piece of middleware, and
``FetchFromCacheMiddleware`` as the last::

    MIDDLEWARE = [
        'django.middleware.cache.UpdateCacheMiddleware',
        ...
        'django.middleware.cache.FetchFromCacheMiddleware'
    ]

This is counter-intuitive, but correct: ``UpdateCacheMiddleware`` needs to run
last during the response phase, which processes middleware bottom-up;
``FetchFromCacheMiddleware`` needs to run last during the request phase, which
processes middleware top-down.

The single-class ``CacheMiddleware`` can be used for some simple sites.
However, if any other piece of middleware needs to affect the cache key, you'll
need to use the two-part ``UpdateCacheMiddleware`` and
``FetchFromCacheMiddleware``. This'll most often happen when you're using
Django's ``LocaleMiddleware``.

More details about how the caching works:

* Only GET or HEAD-requests with status code 200 are cached.

* The number of seconds each page is stored for is set by the "max-age" section
  of the response's "Cache-Control" header, falling back to the
  CACHE_MIDDLEWARE_SECONDS setting if the section was not found.

* This middleware expects that a HEAD request is answered with the same response
  headers exactly like the corresponding GET request.

* When a hit occurs, a shallow copy of the original response object is returned
  from process_request.

* Pages will be cached based on the contents of the request headers listed in
  the response's "Vary" header.

* This middleware also sets ETag, Last-Modified, Expires and Cache-Control
  headers on the response object.

"""

from django.conf import settings
from django.core.cache import DEFAULT_CACHE_ALIAS, caches
from django.utils.cache import (
    get_cache_key, get_max_age, has_vary_header, learn_cache_key,
    patch_response_headers,
)
from django.utils.deprecation import MiddlewareMixin
```
### 9 - django/utils/cache.py:

Start line: 246, End line: 277

```python
def _if_modified_since_passes(last_modified, if_modified_since):
    """
    Test the If-Modified-Since comparison as defined in section 3.3 of RFC 7232.
    """
    return not last_modified or last_modified > if_modified_since


def patch_response_headers(response, cache_timeout=None):
    """
    Add HTTP caching headers to the given HttpResponse: Expires and
    Cache-Control.

    Each header is only added if it isn't already set.

    cache_timeout is in seconds. The CACHE_MIDDLEWARE_SECONDS setting is used
    by default.
    """
    if cache_timeout is None:
        cache_timeout = settings.CACHE_MIDDLEWARE_SECONDS
    if cache_timeout < 0:
        cache_timeout = 0  # Can't have max-age negative
    if not response.has_header('Expires'):
        response.headers['Expires'] = http_date(time.time() + cache_timeout)
    patch_cache_control(response, max_age=cache_timeout)


def add_never_cache_headers(response):
    """
    Add headers to a response to indicate that a page should never be cached.
    """
    patch_response_headers(response, cache_timeout=-1)
    patch_cache_control(response, no_cache=True, no_store=True, must_revalidate=True, private=True)
```
### 10 - django/core/cache/__init__.py:

Start line: 30, End line: 67

```python
class CacheHandler(BaseConnectionHandler):
    settings_name = 'CACHES'
    exception_class = InvalidCacheBackendError

    def create_connection(self, alias):
        params = self.settings[alias].copy()
        backend = params.pop('BACKEND')
        location = params.pop('LOCATION', '')
        try:
            backend_cls = import_string(backend)
        except ImportError as e:
            raise InvalidCacheBackendError(
                "Could not find backend '%s': %s" % (backend, e)
            ) from e
        return backend_cls(location, params)

    def all(self, initialized_only=False):
        return [
            self[alias] for alias in self
            # If initialized_only is True, return only initialized caches.
            if not initialized_only or hasattr(self._connections, alias)
        ]


caches = CacheHandler()

cache = ConnectionProxy(caches, DEFAULT_CACHE_ALIAS)


def close_caches(**kwargs):
    # Some caches need to do a cleanup at the end of a request cycle. If not
    # implemented in a particular backend cache.close() is a no-op.
    for cache in caches.all(initialized_only=True):
        cache.close()


signals.request_finished.connect(close_caches)
```
