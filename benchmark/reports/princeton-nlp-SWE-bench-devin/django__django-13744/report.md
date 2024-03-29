# django__django-13744

| **django/django** | `d8dfff2ab0edf7a1ca5255eccf45c447b2f9d57e` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 911 |
| **Any found context length** | 318 |
| **Avg pos** | 19.5 |
| **Min pos** | 1 |
| **Max pos** | 35 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/cache/__init__.py b/django/core/cache/__init__.py
--- a/django/core/cache/__init__.py
+++ b/django/core/cache/__init__.py
@@ -114,9 +114,8 @@ def __eq__(self, other):
 
 
 def close_caches(**kwargs):
-    # Some caches -- python-memcached in particular -- need to do a cleanup at the
-    # end of a request cycle. If not implemented in a particular backend
-    # cache.close is a no-op
+    # Some caches need to do a cleanup at the end of a request cycle. If not
+    # implemented in a particular backend cache.close() is a no-op.
     for cache in caches.all():
         cache.close()
 
diff --git a/django/core/cache/backends/memcached.py b/django/core/cache/backends/memcached.py
--- a/django/core/cache/backends/memcached.py
+++ b/django/core/cache/backends/memcached.py
@@ -3,10 +3,12 @@
 import pickle
 import re
 import time
+import warnings
 
 from django.core.cache.backends.base import (
     DEFAULT_TIMEOUT, BaseCache, InvalidCacheKey, memcache_key_warnings,
 )
+from django.utils.deprecation import RemovedInDjango41Warning
 from django.utils.functional import cached_property
 
 
@@ -164,6 +166,11 @@ def validate_key(self, key):
 class MemcachedCache(BaseMemcachedCache):
     "An implementation of a cache binding using python-memcached"
     def __init__(self, server, params):
+        warnings.warn(
+            'MemcachedCache is deprecated in favor of PyMemcacheCache and '
+            'PyLibMCCache.',
+            RemovedInDjango41Warning, stacklevel=2,
+        )
         # python-memcached ≥ 1.45 returns None for a nonexistent key in
         # incr/decr(), python-memcached < 1.45 raises ValueError.
         import memcache

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/cache/__init__.py | 117 | 119 | 35 | 5 | 9549
| django/core/cache/backends/memcached.py | 6 | 6 | 3 | 1 | 911
| django/core/cache/backends/memcached.py | 167 | 167 | 1 | 1 | 318


## Problem Statement

```
Deprecate MemcachedCache.
Description
	
python-memcached is not maintained anymore (see ​python-memcached#95) and it makes difficulties in fixing some issues (e.g. #29867). Moreover we added a cache backend for pymemcache (#29887) so we have a good builtin alternative.
I think it's time to deprecate the django.core.cache.backends.memcached.MemcachedCache backend in Django 3.2 and remove it in Django 4.1.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/cache/backends/memcached.py** | 164 | 190| 318 | 318 | 1987 | 
| 2 | **1 django/core/cache/backends/memcached.py** | 67 | 103| 370 | 688 | 1987 | 
| **-> 3 <-** | **1 django/core/cache/backends/memcached.py** | 1 | 38| 223 | 911 | 1987 | 
| 4 | **1 django/core/cache/backends/memcached.py** | 122 | 137| 149 | 1060 | 1987 | 
| 5 | **1 django/core/cache/backends/memcached.py** | 105 | 120| 147 | 1207 | 1987 | 
| 6 | **1 django/core/cache/backends/memcached.py** | 139 | 161| 203 | 1410 | 1987 | 
| 7 | **1 django/core/cache/backends/memcached.py** | 193 | 216| 211 | 1621 | 1987 | 
| 8 | 2 django/core/cache/backends/base.py | 280 | 293| 112 | 1733 | 4170 | 
| 9 | 3 django/core/cache/backends/locmem.py | 1 | 67| 511 | 2244 | 5083 | 
| 10 | 3 django/core/cache/backends/locmem.py | 84 | 125| 276 | 2520 | 5083 | 
| 11 | 3 django/core/cache/backends/locmem.py | 69 | 82| 138 | 2658 | 5083 | 
| 12 | **3 django/core/cache/backends/memcached.py** | 40 | 65| 283 | 2941 | 5083 | 
| 13 | **3 django/core/cache/backends/memcached.py** | 219 | 231| 120 | 3061 | 5083 | 
| 14 | 4 django/core/cache/backends/db.py | 199 | 228| 285 | 3346 | 7205 | 
| 15 | **5 django/core/cache/__init__.py** | 1 | 29| 223 | 3569 | 8031 | 
| 16 | 5 django/core/cache/backends/base.py | 230 | 277| 348 | 3917 | 8031 | 
| 17 | 5 django/core/cache/backends/db.py | 255 | 283| 324 | 4241 | 8031 | 
| 18 | 5 django/core/cache/backends/base.py | 1 | 51| 254 | 4495 | 8031 | 
| 19 | 6 django/core/cache/backends/dummy.py | 1 | 40| 255 | 4750 | 8287 | 
| 20 | 6 django/core/cache/backends/db.py | 40 | 95| 431 | 5181 | 8287 | 
| 21 | 7 django/middleware/cache.py | 1 | 52| 431 | 5612 | 9979 | 
| 22 | 8 django/utils/deprecation.py | 79 | 120| 343 | 5955 | 11045 | 
| 23 | 8 django/core/cache/backends/db.py | 230 | 253| 259 | 6214 | 11045 | 
| 24 | 8 django/core/cache/backends/db.py | 97 | 110| 234 | 6448 | 11045 | 
| 25 | 9 django/bin/django-admin.py | 1 | 22| 138 | 6586 | 11183 | 
| 26 | 9 django/core/cache/backends/base.py | 54 | 91| 306 | 6892 | 11183 | 
| 27 | **9 django/core/cache/__init__.py** | 32 | 54| 196 | 7088 | 11183 | 
| 28 | 9 django/middleware/cache.py | 164 | 199| 297 | 7385 | 11183 | 
| 29 | 10 django/utils/cache.py | 1 | 34| 274 | 7659 | 14913 | 
| 30 | 11 django/core/cache/backends/filebased.py | 1 | 44| 284 | 7943 | 16139 | 
| 31 | 11 django/core/cache/backends/db.py | 112 | 197| 794 | 8737 | 16139 | 
| 32 | 11 django/core/cache/backends/filebased.py | 46 | 59| 145 | 8882 | 16139 | 
| 33 | 11 django/utils/deprecation.py | 1 | 33| 209 | 9091 | 16139 | 
| 34 | 11 django/core/cache/backends/db.py | 1 | 37| 229 | 9320 | 16139 | 
| **-> 35 <-** | **11 django/core/cache/__init__.py** | 90 | 125| 229 | 9549 | 16139 | 
| 36 | 11 django/utils/deprecation.py | 36 | 76| 336 | 9885 | 16139 | 
| 37 | 12 django/views/decorators/cache.py | 27 | 48| 153 | 10038 | 16502 | 
| 38 | 12 django/core/cache/backends/filebased.py | 61 | 96| 260 | 10298 | 16502 | 
| 39 | 12 django/core/cache/backends/filebased.py | 98 | 114| 174 | 10472 | 16502 | 
| 40 | 12 django/middleware/cache.py | 55 | 75| 205 | 10677 | 16502 | 
| 41 | 13 django/core/checks/caches.py | 22 | 55| 267 | 10944 | 16999 | 
| 42 | 13 django/utils/cache.py | 135 | 150| 190 | 11134 | 16999 | 
| 43 | 14 django/core/cache/utils.py | 1 | 13| 0 | 11134 | 17078 | 
| 44 | 14 django/utils/cache.py | 244 | 275| 257 | 11391 | 17078 | 
| 45 | 14 django/middleware/cache.py | 135 | 161| 252 | 11643 | 17078 | 
| 46 | 14 django/middleware/cache.py | 77 | 116| 363 | 12006 | 17078 | 
| 47 | 15 django/templatetags/cache.py | 1 | 49| 413 | 12419 | 17805 | 
| 48 | 15 django/core/checks/caches.py | 1 | 19| 119 | 12538 | 17805 | 
| 49 | 16 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 12710 | 18226 | 
| 50 | 16 django/core/cache/backends/filebased.py | 116 | 165| 390 | 13100 | 18226 | 
| 51 | 16 django/core/cache/backends/base.py | 214 | 228| 147 | 13247 | 18226 | 
| 52 | 17 django/contrib/messages/storage/cookie.py | 134 | 157| 228 | 13475 | 19754 | 
| 53 | 17 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 13728 | 19754 | 
| 54 | 18 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 13926 | 20321 | 
| 55 | 19 django/db/models/fields/mixins.py | 1 | 28| 168 | 14094 | 20664 | 
| 56 | 19 django/utils/deprecation.py | 122 | 148| 179 | 14273 | 20664 | 
| 57 | 19 django/middleware/cache.py | 119 | 133| 156 | 14429 | 20664 | 
| 58 | 19 django/core/cache/backends/base.py | 180 | 212| 277 | 14706 | 20664 | 
| 59 | 20 django/contrib/postgres/forms/jsonb.py | 1 | 17| 108 | 14814 | 20772 | 
| 60 | 20 django/core/checks/caches.py | 58 | 72| 109 | 14923 | 20772 | 
| 61 | 20 django/views/decorators/cache.py | 1 | 24| 209 | 15132 | 20772 | 
| 62 | 20 django/contrib/sessions/backends/cache.py | 1 | 34| 213 | 15345 | 20772 | 
| 63 | **20 django/core/cache/__init__.py** | 57 | 87| 178 | 15523 | 20772 | 
| 64 | 20 django/core/cache/backends/base.py | 144 | 157| 122 | 15645 | 20772 | 
| 65 | 20 django/core/cache/backends/base.py | 93 | 104| 119 | 15764 | 20772 | 
| 66 | 21 django/contrib/sessions/backends/base.py | 142 | 159| 196 | 15960 | 23602 | 
| 67 | 22 django/contrib/sites/models.py | 78 | 121| 236 | 16196 | 24390 | 
| 68 | 23 django/contrib/gis/views.py | 1 | 21| 155 | 16351 | 24545 | 
| 69 | 24 django/utils/formats.py | 1 | 57| 377 | 16728 | 26651 | 
| 70 | 24 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 16895 | 26651 | 
| 71 | 24 django/utils/cache.py | 37 | 102| 559 | 17454 | 26651 | 
| 72 | 25 django/conf/urls/__init__.py | 1 | 23| 152 | 17606 | 26803 | 
| 73 | 26 django/contrib/postgres/fields/jsonb.py | 1 | 44| 312 | 17918 | 27115 | 
| 74 | 27 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 18039 | 27971 | 
| 75 | 27 django/utils/cache.py | 326 | 344| 206 | 18245 | 27971 | 
| 76 | 28 django/contrib/messages/middleware.py | 1 | 27| 174 | 18419 | 28146 | 
| 77 | 29 django/middleware/gzip.py | 1 | 52| 419 | 18838 | 28566 | 
| 78 | 29 django/core/management/commands/createcachetable.py | 1 | 29| 213 | 19051 | 28566 | 
| 79 | 30 django/middleware/security.py | 1 | 31| 305 | 19356 | 29103 | 
| 80 | 31 django/conf/__init__.py | 96 | 110| 129 | 19485 | 31290 | 
| 81 | 32 django/contrib/contenttypes/models.py | 118 | 130| 133 | 19618 | 32705 | 
| 82 | 33 django/contrib/sessions/middleware.py | 1 | 78| 619 | 20237 | 33325 | 
| 83 | 34 django/utils/functional.py | 1 | 49| 336 | 20573 | 36381 | 
| 84 | 35 django/db/models/options.py | 716 | 731| 144 | 20717 | 43487 | 
| 85 | 35 django/contrib/sessions/backends/base.py | 329 | 396| 459 | 21176 | 43487 | 
| 86 | 36 django/db/__init__.py | 40 | 62| 118 | 21294 | 43880 | 
| 87 | 36 django/utils/cache.py | 105 | 132| 183 | 21477 | 43880 | 
| 88 | 37 django/core/mail/backends/locmem.py | 1 | 31| 183 | 21660 | 44064 | 
| 89 | 38 django/template/loaders/cached.py | 67 | 98| 225 | 21885 | 44782 | 
| 90 | 38 django/contrib/sessions/backends/base.py | 119 | 140| 204 | 22089 | 44782 | 
| 91 | 38 django/templatetags/cache.py | 52 | 94| 313 | 22402 | 44782 | 
| 92 | 38 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 22934 | 44782 | 
| 93 | 39 django/views/decorators/gzip.py | 1 | 6| 0 | 22934 | 44833 | 
| 94 | 40 django/conf/global_settings.py | 401 | 491| 794 | 23728 | 50593 | 
| 95 | 40 django/middleware/security.py | 33 | 58| 256 | 23984 | 50593 | 
| 96 | 41 django/views/debug.py | 142 | 154| 148 | 24132 | 55184 | 
| 97 | 42 django/utils/translation/__init__.py | 70 | 149| 489 | 24621 | 57524 | 
| 98 | 42 django/conf/global_settings.py | 492 | 627| 812 | 25433 | 57524 | 
| 99 | 43 django/contrib/sessions/management/commands/clearsessions.py | 1 | 22| 124 | 25557 | 57648 | 
| 100 | 43 django/core/cache/backends/base.py | 106 | 142| 325 | 25882 | 57648 | 
| 101 | 44 django/core/management/commands/flush.py | 27 | 83| 486 | 26368 | 58335 | 
| 102 | 45 django/db/backends/mysql/base.py | 1 | 49| 404 | 26772 | 61711 | 
| 103 | 45 django/conf/__init__.py | 72 | 94| 221 | 26993 | 61711 | 
| 104 | 46 django/contrib/gis/db/backends/mysql/features.py | 1 | 27| 202 | 27195 | 61914 | 
| 105 | 47 django/template/backends/django.py | 79 | 111| 225 | 27420 | 62770 | 
| 106 | 48 django/utils/decorators.py | 114 | 152| 316 | 27736 | 64169 | 
| 107 | 49 django/http/cookie.py | 1 | 27| 188 | 27924 | 64358 | 
| 108 | 49 django/utils/cache.py | 369 | 415| 521 | 28445 | 64358 | 
| 109 | 50 django/db/models/fields/__init__.py | 367 | 393| 199 | 28644 | 82804 | 
| 110 | 51 django/core/mail/backends/__init__.py | 1 | 2| 0 | 28644 | 82812 | 
| 111 | 52 django/http/__init__.py | 1 | 22| 197 | 28841 | 83009 | 
| 112 | 53 django/db/backends/postgresql/base.py | 1 | 62| 456 | 29297 | 85858 | 
| 113 | 54 django/utils/http.py | 1 | 70| 692 | 29989 | 90194 | 
| 114 | 54 django/db/models/options.py | 256 | 288| 331 | 30320 | 90194 | 
| 115 | 55 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 30467 | 90342 | 
| 116 | 55 django/utils/cache.py | 153 | 193| 453 | 30920 | 90342 | 
| 117 | 56 django/contrib/redirects/middleware.py | 1 | 51| 355 | 31275 | 90698 | 
| 118 | 57 django/db/models/base.py | 1 | 50| 328 | 31603 | 107371 | 
| 119 | 58 django/db/backends/base/base.py | 560 | 607| 300 | 31903 | 112274 | 
| 120 | 58 django/contrib/sessions/backends/base.py | 161 | 244| 563 | 32466 | 112274 | 
| 121 | 58 django/contrib/messages/storage/cookie.py | 159 | 193| 258 | 32724 | 112274 | 
| 122 | 58 django/template/loaders/cached.py | 1 | 65| 497 | 33221 | 112274 | 
| 123 | 58 django/db/backends/base/base.py | 1 | 23| 138 | 33359 | 112274 | 
| 124 | 59 django/core/files/storage.py | 1 | 22| 158 | 33517 | 115153 | 
| 125 | 60 django/contrib/auth/middleware.py | 1 | 23| 171 | 33688 | 116147 | 
| 126 | 61 django/contrib/staticfiles/storage.py | 341 | 362| 230 | 33918 | 119674 | 
| 127 | 61 django/core/files/storage.py | 296 | 368| 483 | 34401 | 119674 | 
| 128 | 62 django/db/backends/utils.py | 1 | 45| 273 | 34674 | 121540 | 
| 129 | 63 django/core/mail/backends/dummy.py | 1 | 11| 0 | 34674 | 121583 | 
| 130 | 64 django/db/backends/dummy/base.py | 1 | 47| 270 | 34944 | 122028 | 
| 131 | 65 django/contrib/auth/hashers.py | 617 | 658| 283 | 35227 | 127166 | 
| 132 | 66 django/utils/archive.py | 185 | 227| 311 | 35538 | 128695 | 
| 133 | 67 django/template/backends/jinja2.py | 1 | 51| 341 | 35879 | 129517 | 
| 134 | 67 django/contrib/staticfiles/storage.py | 323 | 339| 147 | 36026 | 129517 | 
| 135 | 68 django/template/context.py | 1 | 24| 128 | 36154 | 131398 | 
| 136 | 69 django/db/models/query_utils.py | 167 | 231| 492 | 36646 | 134104 | 
| 137 | 70 django/http/response.py | 518 | 549| 153 | 36799 | 138679 | 
| 138 | 70 django/http/response.py | 1 | 24| 166 | 36965 | 138679 | 
| 139 | 70 django/contrib/sessions/backends/base.py | 1 | 36| 208 | 37173 | 138679 | 
| 140 | 70 django/utils/cache.py | 217 | 241| 235 | 37408 | 138679 | 
| 141 | 70 django/utils/cache.py | 347 | 366| 190 | 37598 | 138679 | 
| 142 | 70 django/contrib/staticfiles/storage.py | 251 | 321| 575 | 38173 | 138679 | 
| 143 | 70 django/views/debug.py | 64 | 77| 115 | 38288 | 138679 | 
| 144 | 71 django/db/backends/sqlite3/operations.py | 197 | 216| 209 | 38497 | 141769 | 
| 145 | 72 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 29 | 95| 517 | 39014 | 142489 | 
| 146 | 72 django/utils/http.py | 73 | 98| 195 | 39209 | 142489 | 
| 147 | 73 django/utils/itercompat.py | 1 | 9| 0 | 39209 | 142529 | 
| 148 | 73 django/contrib/auth/middleware.py | 84 | 109| 192 | 39401 | 142529 | 
| 149 | 74 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 39401 | 142599 | 
| 150 | 74 django/contrib/messages/storage/cookie.py | 85 | 103| 144 | 39545 | 142599 | 
| 151 | 74 django/contrib/staticfiles/storage.py | 410 | 442| 275 | 39820 | 142599 | 
| 152 | 75 django/db/models/fields/related.py | 108 | 125| 155 | 39975 | 156475 | 
| 153 | 76 django/db/utils.py | 1 | 49| 154 | 40129 | 158646 | 
| 154 | 77 django/http/request.py | 1 | 50| 397 | 40526 | 164118 | 
| 155 | 77 django/db/models/fields/__init__.py | 1874 | 1902| 191 | 40717 | 164118 | 
| 156 | 78 django/utils/crypto.py | 49 | 74| 212 | 40929 | 164828 | 
| 157 | 79 django/core/files/base.py | 1 | 29| 174 | 41103 | 165880 | 
| 158 | 80 django/db/backends/base/features.py | 217 | 318| 864 | 41967 | 168730 | 
| 159 | 80 django/db/backends/base/base.py | 502 | 525| 189 | 42156 | 168730 | 
| 160 | 81 django/db/models/lookups.py | 1 | 15| 112 | 42268 | 173757 | 
| 161 | 82 django/contrib/auth/__init__.py | 41 | 60| 167 | 42435 | 175377 | 
| 162 | 83 django/template/backends/utils.py | 1 | 15| 0 | 42435 | 175466 | 
| 163 | 83 django/contrib/sites/models.py | 25 | 46| 192 | 42627 | 175466 | 
| 164 | 83 django/db/models/fields/related.py | 156 | 169| 144 | 42771 | 175466 | 
| 165 | 83 django/template/backends/django.py | 48 | 76| 210 | 42981 | 175466 | 
| 166 | 83 django/conf/global_settings.py | 628 | 652| 180 | 43161 | 175466 | 
| 167 | 83 django/utils/functional.py | 129 | 191| 484 | 43645 | 175466 | 
| 168 | 83 django/views/debug.py | 1 | 47| 296 | 43941 | 175466 | 
| 169 | 83 django/db/models/fields/related.py | 127 | 154| 201 | 44142 | 175466 | 
| 170 | 84 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 44352 | 176967 | 
| 171 | 84 django/contrib/messages/storage/cookie.py | 105 | 132| 261 | 44613 | 176967 | 
| 172 | 84 django/utils/cache.py | 278 | 300| 258 | 44871 | 176967 | 
| 173 | 84 django/http/response.py | 552 | 577| 159 | 45030 | 176967 | 
| 174 | 85 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 45170 | 177818 | 
| 175 | 85 django/conf/global_settings.py | 151 | 266| 859 | 46029 | 177818 | 
| 176 | 85 django/core/files/storage.py | 193 | 203| 130 | 46159 | 177818 | 
| 177 | 86 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 46279 | 180140 | 
| 178 | 87 django/urls/base.py | 89 | 155| 383 | 46662 | 181311 | 
| 179 | 87 django/contrib/auth/hashers.py | 85 | 107| 167 | 46829 | 181311 | 
| 180 | 87 django/db/models/options.py | 1 | 34| 285 | 47114 | 181311 | 
| 181 | 87 django/http/response.py | 70 | 87| 133 | 47247 | 181311 | 
| 182 | 88 django/db/backends/sqlite3/base.py | 403 | 423| 183 | 47430 | 187328 | 
| 183 | 88 django/db/models/options.py | 409 | 431| 154 | 47584 | 187328 | 
| 184 | 89 django/db/migrations/autodetector.py | 805 | 854| 567 | 48151 | 198947 | 
| 185 | 89 django/db/models/fields/related.py | 255 | 282| 269 | 48420 | 198947 | 
| 186 | 89 django/db/utils.py | 52 | 98| 312 | 48732 | 198947 | 
| 187 | 90 django/contrib/gis/shortcuts.py | 1 | 41| 248 | 48980 | 199195 | 
| 188 | 90 django/core/management/commands/flush.py | 1 | 25| 206 | 49186 | 199195 | 
| 189 | 90 django/utils/archive.py | 143 | 182| 290 | 49476 | 199195 | 


### Hint

```
​Mailing list discussion
I'm going to accept this. We can give it a few weeks to see if there's any objections/movement on python-memcached.
```

## Patch

```diff
diff --git a/django/core/cache/__init__.py b/django/core/cache/__init__.py
--- a/django/core/cache/__init__.py
+++ b/django/core/cache/__init__.py
@@ -114,9 +114,8 @@ def __eq__(self, other):
 
 
 def close_caches(**kwargs):
-    # Some caches -- python-memcached in particular -- need to do a cleanup at the
-    # end of a request cycle. If not implemented in a particular backend
-    # cache.close is a no-op
+    # Some caches need to do a cleanup at the end of a request cycle. If not
+    # implemented in a particular backend cache.close() is a no-op.
     for cache in caches.all():
         cache.close()
 
diff --git a/django/core/cache/backends/memcached.py b/django/core/cache/backends/memcached.py
--- a/django/core/cache/backends/memcached.py
+++ b/django/core/cache/backends/memcached.py
@@ -3,10 +3,12 @@
 import pickle
 import re
 import time
+import warnings
 
 from django.core.cache.backends.base import (
     DEFAULT_TIMEOUT, BaseCache, InvalidCacheKey, memcache_key_warnings,
 )
+from django.utils.deprecation import RemovedInDjango41Warning
 from django.utils.functional import cached_property
 
 
@@ -164,6 +166,11 @@ def validate_key(self, key):
 class MemcachedCache(BaseMemcachedCache):
     "An implementation of a cache binding using python-memcached"
     def __init__(self, server, params):
+        warnings.warn(
+            'MemcachedCache is deprecated in favor of PyMemcacheCache and '
+            'PyLibMCCache.',
+            RemovedInDjango41Warning, stacklevel=2,
+        )
         # python-memcached ≥ 1.45 returns None for a nonexistent key in
         # incr/decr(), python-memcached < 1.45 raises ValueError.
         import memcache

```

## Test Patch

```diff
diff --git a/tests/cache/tests.py b/tests/cache/tests.py
--- a/tests/cache/tests.py
+++ b/tests/cache/tests.py
@@ -11,6 +11,7 @@
 import threading
 import time
 import unittest
+import warnings
 from pathlib import Path
 from unittest import mock, skipIf
 
@@ -34,13 +35,14 @@
 from django.template.response import TemplateResponse
 from django.test import (
     RequestFactory, SimpleTestCase, TestCase, TransactionTestCase,
-    override_settings,
+    ignore_warnings, override_settings,
 )
 from django.test.signals import setting_changed
 from django.utils import timezone, translation
 from django.utils.cache import (
     get_cache_key, learn_cache_key, patch_cache_control, patch_vary_headers,
 )
+from django.utils.deprecation import RemovedInDjango41Warning
 from django.views.decorators.cache import cache_control, cache_page
 
 from .models import Poll, expensive_calculation
@@ -1275,7 +1277,6 @@ def test_lru_incr(self):
 for _cache_params in settings.CACHES.values():
     configured_caches[_cache_params['BACKEND']] = _cache_params
 
-MemcachedCache_params = configured_caches.get('django.core.cache.backends.memcached.MemcachedCache')
 PyLibMCCache_params = configured_caches.get('django.core.cache.backends.memcached.PyLibMCCache')
 PyMemcacheCache_params = configured_caches.get('django.core.cache.backends.memcached.PyMemcacheCache')
 
@@ -1348,10 +1349,7 @@ def test_memcached_deletes_key_on_failed_set(self):
         # By default memcached allows objects up to 1MB. For the cache_db session
         # backend to always use the current session, memcached needs to delete
         # the old key if it fails to set.
-        # pylibmc doesn't seem to have SERVER_MAX_VALUE_LENGTH as far as I can
-        # tell from a quick check of its source code. This is falling back to
-        # the default value exposed by python-memcached on my system.
-        max_value_length = getattr(cache._lib, 'SERVER_MAX_VALUE_LENGTH', 1048576)
+        max_value_length = 2 ** 20
 
         cache.set('small_value', 'a')
         self.assertEqual(cache.get('small_value'), 'a')
@@ -1360,11 +1358,10 @@ def test_memcached_deletes_key_on_failed_set(self):
         try:
             cache.set('small_value', large_value)
         except Exception:
-            # Some clients (e.g. pylibmc) raise when the value is too large,
-            # while others (e.g. python-memcached) intentionally return True
-            # indicating success. This test is primarily checking that the key
-            # was deleted, so the return/exception behavior for the set()
-            # itself is not important.
+            # Most clients (e.g. pymemcache or pylibmc) raise when the value is
+            # too large. This test is primarily checking that the key was
+            # deleted, so the return/exception behavior for the set() itself is
+            # not important.
             pass
         # small_value should be deleted, or set if configured to accept larger values
         value = cache.get('small_value')
@@ -1389,6 +1386,11 @@ def fail_set_multi(mapping, *args, **kwargs):
             self.assertEqual(failing_keys, ['key'])
 
 
+# RemovedInDjango41Warning.
+MemcachedCache_params = configured_caches.get('django.core.cache.backends.memcached.MemcachedCache')
+
+
+@ignore_warnings(category=RemovedInDjango41Warning)
 @unittest.skipUnless(MemcachedCache_params, "MemcachedCache backend not configured")
 @override_settings(CACHES=caches_setting_for_tests(
     base=MemcachedCache_params,
@@ -1420,6 +1422,32 @@ def test_default_used_when_none_is_set(self):
         self.assertEqual(cache.get('key_default_none', default='default'), 'default')
 
 
+class MemcachedCacheDeprecationTests(SimpleTestCase):
+    def test_warning(self):
+        from django.core.cache.backends.memcached import MemcachedCache
+
+        # Remove warnings filter on MemcachedCache deprecation warning, added
+        # in runtests.py.
+        warnings.filterwarnings(
+            'error',
+            'MemcachedCache is deprecated',
+            category=RemovedInDjango41Warning,
+        )
+        try:
+            msg = (
+                'MemcachedCache is deprecated in favor of PyMemcacheCache and '
+                'PyLibMCCache.'
+            )
+            with self.assertRaisesMessage(RemovedInDjango41Warning, msg):
+                MemcachedCache('127.0.0.1:11211', {})
+        finally:
+            warnings.filterwarnings(
+                'ignore',
+                'MemcachedCache is deprecated',
+                category=RemovedInDjango41Warning,
+            )
+
+
 @unittest.skipUnless(PyLibMCCache_params, "PyLibMCCache backend not configured")
 @override_settings(CACHES=caches_setting_for_tests(
     base=PyLibMCCache_params,
diff --git a/tests/requirements/py3.txt b/tests/requirements/py3.txt
--- a/tests/requirements/py3.txt
+++ b/tests/requirements/py3.txt
@@ -9,6 +9,7 @@ Pillow >= 6.2.0
 # pylibmc/libmemcached can't be built on Windows.
 pylibmc; sys.platform != 'win32'
 pymemcache >= 3.4.0
+# RemovedInDjango41Warning.
 python-memcached >= 1.59
 pytz
 pywatchman; sys.platform != 'win32'
diff --git a/tests/runtests.py b/tests/runtests.py
--- a/tests/runtests.py
+++ b/tests/runtests.py
@@ -47,6 +47,12 @@
 warnings.simplefilter("error", RuntimeWarning)
 # Ignore known warnings in test dependencies.
 warnings.filterwarnings("ignore", "'U' mode is deprecated", DeprecationWarning, module='docutils.io')
+# RemovedInDjango41Warning: Ignore MemcachedCache deprecation warning.
+warnings.filterwarnings(
+    'ignore',
+    'MemcachedCache is deprecated',
+    category=RemovedInDjango41Warning,
+)
 
 RUNTESTS_DIR = os.path.abspath(os.path.dirname(__file__))
 

```


## Code snippets

### 1 - django/core/cache/backends/memcached.py:

Start line: 164, End line: 190

```python
class MemcachedCache(BaseMemcachedCache):
    "An implementation of a cache binding using python-memcached"
    def __init__(self, server, params):
        # python-memcached ≥ 1.45 returns None for a nonexistent key in
        # incr/decr(), python-memcached < 1.45 raises ValueError.
        import memcache
        super().__init__(server, params, library=memcache, value_not_found_exception=ValueError)
        self._options = {'pickleProtocol': pickle.HIGHEST_PROTOCOL, **self._options}

    def get(self, key, default=None, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        val = self._cache.get(key)
        # python-memcached doesn't support default values in get().
        # https://github.com/linsomniac/python-memcached/issues/159
        # Remove this method if that issue is fixed.
        if val is None:
            return default
        return val

    def delete(self, key, version=None):
        # python-memcached's delete() returns True when key doesn't exist.
        # https://github.com/linsomniac/python-memcached/issues/170
        # Call _deletetouch() without the NOT_FOUND in expected results.
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return bool(self._cache._deletetouch([b'DELETED'], 'delete', key))
```
### 2 - django/core/cache/backends/memcached.py:

Start line: 67, End line: 103

```python
class BaseMemcachedCache(BaseCache):

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return self._cache.add(key, value, self.get_backend_timeout(timeout))

    def get(self, key, default=None, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return self._cache.get(key, default)

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        if not self._cache.set(key, value, self.get_backend_timeout(timeout)):
            # make sure the key doesn't keep its old value in case of failure to set (memcached's 1MB limit)
            self._cache.delete(key)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return bool(self._cache.touch(key, self.get_backend_timeout(timeout)))

    def delete(self, key, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return bool(self._cache.delete(key))

    def get_many(self, keys, version=None):
        key_map = {self.make_key(key, version=version): key for key in keys}
        for key in key_map:
            self.validate_key(key)
        ret = self._cache.get_multi(key_map.keys())
        return {key_map[k]: v for k, v in ret.items()}

    def close(self, **kwargs):
        # Many clients don't clean up connections properly.
        self._cache.disconnect_all()
```
### 3 - django/core/cache/backends/memcached.py:

Start line: 1, End line: 38

```python
"Memcached cache backend"

import pickle
import re
import time

from django.core.cache.backends.base import (
    DEFAULT_TIMEOUT, BaseCache, InvalidCacheKey, memcache_key_warnings,
)
from django.utils.functional import cached_property


class BaseMemcachedCache(BaseCache):
    def __init__(self, server, params, library, value_not_found_exception):
        super().__init__(params)
        if isinstance(server, str):
            self._servers = re.split('[;,]', server)
        else:
            self._servers = server

        # Exception type raised by the underlying client library for a
        # nonexistent key.
        self.LibraryValueNotFoundException = value_not_found_exception

        self._lib = library
        self._class = library.Client
        self._options = params.get('OPTIONS') or {}

    @property
    def client_servers(self):
        return self._servers

    @cached_property
    def _cache(self):
        """
        Implement transparent thread-safe access to a memcached client.
        """
        return self._class(self.client_servers, **self._options)
```
### 4 - django/core/cache/backends/memcached.py:

Start line: 122, End line: 137

```python
class BaseMemcachedCache(BaseCache):

    def decr(self, key, delta=1, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        # memcached doesn't support a negative delta
        if delta < 0:
            return self._cache.incr(key, -delta)
        try:
            val = self._cache.decr(key, delta)

        # Normalize an exception raised by the underlying client library to
        # ValueError in the event of a nonexistent key when calling decr().
        except self.LibraryValueNotFoundException:
            val = None
        if val is None:
            raise ValueError("Key '%s' not found" % key)
        return val
```
### 5 - django/core/cache/backends/memcached.py:

Start line: 105, End line: 120

```python
class BaseMemcachedCache(BaseCache):

    def incr(self, key, delta=1, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        # memcached doesn't support a negative delta
        if delta < 0:
            return self._cache.decr(key, -delta)
        try:
            val = self._cache.incr(key, delta)

        # Normalize an exception raised by the underlying client library to
        # ValueError in the event of a nonexistent key when calling incr().
        except self.LibraryValueNotFoundException:
            val = None
        if val is None:
            raise ValueError("Key '%s' not found" % key)
        return val
```
### 6 - django/core/cache/backends/memcached.py:

Start line: 139, End line: 161

```python
class BaseMemcachedCache(BaseCache):

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        safe_data = {}
        original_keys = {}
        for key, value in data.items():
            safe_key = self.make_key(key, version=version)
            self.validate_key(safe_key)
            safe_data[safe_key] = value
            original_keys[safe_key] = key
        failed_keys = self._cache.set_multi(safe_data, self.get_backend_timeout(timeout))
        return [original_keys[k] for k in failed_keys]

    def delete_many(self, keys, version=None):
        keys = [self.make_key(key, version=version) for key in keys]
        for key in keys:
            self.validate_key(key)
        self._cache.delete_multi(keys)

    def clear(self):
        self._cache.flush_all()

    def validate_key(self, key):
        for warning in memcache_key_warnings(key):
            raise InvalidCacheKey(warning)
```
### 7 - django/core/cache/backends/memcached.py:

Start line: 193, End line: 216

```python
class PyLibMCCache(BaseMemcachedCache):
    "An implementation of a cache binding using pylibmc"
    def __init__(self, server, params):
        import pylibmc
        super().__init__(server, params, library=pylibmc, value_not_found_exception=pylibmc.NotFound)

    @property
    def client_servers(self):
        output = []
        for server in self._servers:
            output.append(server[5:] if server.startswith('unix:') else server)
        return output

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        if timeout == 0:
            return self._cache.delete(key)
        return self._cache.touch(key, self.get_backend_timeout(timeout))

    def close(self, **kwargs):
        # libmemcached manages its own connections. Don't call disconnect_all()
        # as it resets the failover state and creates unnecessary reconnects.
        pass
```
### 8 - django/core/cache/backends/base.py:

Start line: 280, End line: 293

```python
def memcache_key_warnings(key):
    if len(key) > MEMCACHE_MAX_KEY_LENGTH:
        yield (
            'Cache key will cause errors if used with memcached: %r '
            '(longer than %s)' % (key, MEMCACHE_MAX_KEY_LENGTH)
        )
    for char in key:
        if ord(char) < 33 or ord(char) == 127:
            yield (
                'Cache key contains characters that will cause errors if '
                'used with memcached: %r' % key
            )
            break
```
### 9 - django/core/cache/backends/locmem.py:

Start line: 1, End line: 67

```python
"Thread-safe in-memory cache backend."
import pickle
import time
from collections import OrderedDict
from threading import Lock

from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache

# Global in-memory store of cache data. Keyed by name, to provide
# multiple named local memory caches.
_caches = {}
_expire_info = {}
_locks = {}


class LocMemCache(BaseCache):
    pickle_protocol = pickle.HIGHEST_PROTOCOL

    def __init__(self, name, params):
        super().__init__(params)
        self._cache = _caches.setdefault(name, OrderedDict())
        self._expire_info = _expire_info.setdefault(name, {})
        self._lock = _locks.setdefault(name, Lock())

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        pickled = pickle.dumps(value, self.pickle_protocol)
        with self._lock:
            if self._has_expired(key):
                self._set(key, pickled, timeout)
                return True
            return False

    def get(self, key, default=None, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        with self._lock:
            if self._has_expired(key):
                self._delete(key)
                return default
            pickled = self._cache[key]
            self._cache.move_to_end(key, last=False)
        return pickle.loads(pickled)

    def _set(self, key, value, timeout=DEFAULT_TIMEOUT):
        if len(self._cache) >= self._max_entries:
            self._cull()
        self._cache[key] = value
        self._cache.move_to_end(key, last=False)
        self._expire_info[key] = self.get_backend_timeout(timeout)

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        pickled = pickle.dumps(value, self.pickle_protocol)
        with self._lock:
            self._set(key, pickled, timeout)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        with self._lock:
            if self._has_expired(key):
                return False
            self._expire_info[key] = self.get_backend_timeout(timeout)
            return True
```
### 10 - django/core/cache/backends/locmem.py:

Start line: 84, End line: 125

```python
class LocMemCache(BaseCache):

    def has_key(self, key, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        with self._lock:
            if self._has_expired(key):
                self._delete(key)
                return False
            return True

    def _has_expired(self, key):
        exp = self._expire_info.get(key, -1)
        return exp is not None and exp <= time.time()

    def _cull(self):
        if self._cull_frequency == 0:
            self._cache.clear()
            self._expire_info.clear()
        else:
            count = len(self._cache) // self._cull_frequency
            for i in range(count):
                key, _ = self._cache.popitem()
                del self._expire_info[key]

    def _delete(self, key):
        try:
            del self._cache[key]
            del self._expire_info[key]
        except KeyError:
            return False
        return True

    def delete(self, key, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        with self._lock:
            return self._delete(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._expire_info.clear()
```
### 12 - django/core/cache/backends/memcached.py:

Start line: 40, End line: 65

```python
class BaseMemcachedCache(BaseCache):

    def get_backend_timeout(self, timeout=DEFAULT_TIMEOUT):
        """
        Memcached deals with long (> 30 days) timeouts in a special
        way. Call this function to obtain a safe value for your timeout.
        """
        if timeout == DEFAULT_TIMEOUT:
            timeout = self.default_timeout

        if timeout is None:
            # Using 0 in memcache sets a non-expiring timeout.
            return 0
        elif int(timeout) == 0:
            # Other cache backends treat 0 as set-and-expire. To achieve this
            # in memcache backends, a negative timeout must be passed.
            timeout = -1

        if timeout > 2592000:  # 60*60*24*30, 30 days
            # See https://github.com/memcached/memcached/wiki/Programming#expiration
            # "Expiration times can be set from 0, meaning "never expire", to
            # 30 days. Any time higher than 30 days is interpreted as a Unix
            # timestamp date. If you want to expire an object on January 1st of
            # next year, this is how you do that."
            #
            # This means that we have to switch to absolute timestamps.
            timeout += int(time.time())
        return int(timeout)
```
### 13 - django/core/cache/backends/memcached.py:

Start line: 219, End line: 231

```python
class PyMemcacheCache(BaseMemcachedCache):
    """An implementation of a cache binding using pymemcache."""
    def __init__(self, server, params):
        import pymemcache.serde
        super().__init__(server, params, library=pymemcache, value_not_found_exception=KeyError)
        self._class = self._lib.HashClient
        self._options = {
            'allow_unicode_keys': True,
            'default_noreply': False,
            'serde': pymemcache.serde.pickle_serde,
            **self._options,
        }
```
### 15 - django/core/cache/__init__.py:

Start line: 1, End line: 29

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
from asgiref.local import Local

from django.conf import settings
from django.core import signals
from django.core.cache.backends.base import (
    BaseCache, CacheKeyWarning, InvalidCacheBackendError, InvalidCacheKey,
)
from django.utils.module_loading import import_string

__all__ = [
    'cache', 'caches', 'DEFAULT_CACHE_ALIAS', 'InvalidCacheBackendError',
    'CacheKeyWarning', 'BaseCache', 'InvalidCacheKey',
]

DEFAULT_CACHE_ALIAS = 'default'
```
### 27 - django/core/cache/__init__.py:

Start line: 32, End line: 54

```python
def _create_cache(backend, **kwargs):
    try:
        # Try to get the CACHES entry for the given backend name first
        try:
            conf = settings.CACHES[backend]
        except KeyError:
            try:
                # Trying to import the given backend, in case it's a dotted path
                import_string(backend)
            except ImportError as e:
                raise InvalidCacheBackendError("Could not find backend '%s': %s" % (
                    backend, e))
            location = kwargs.pop('LOCATION', '')
            params = kwargs
        else:
            params = {**conf, **kwargs}
            backend = params.pop('BACKEND')
            location = params.pop('LOCATION', '')
        backend_cls = import_string(backend)
    except ImportError as e:
        raise InvalidCacheBackendError(
            "Could not find backend '%s': %s" % (backend, e))
    return backend_cls(location, params)
```
### 35 - django/core/cache/__init__.py:

Start line: 90, End line: 125

```python
class DefaultCacheProxy:
    """
    Proxy access to the default Cache object's attributes.

    This allows the legacy `cache` object to be thread-safe using the new
    ``caches`` API.
    """
    def __getattr__(self, name):
        return getattr(caches[DEFAULT_CACHE_ALIAS], name)

    def __setattr__(self, name, value):
        return setattr(caches[DEFAULT_CACHE_ALIAS], name, value)

    def __delattr__(self, name):
        return delattr(caches[DEFAULT_CACHE_ALIAS], name)

    def __contains__(self, key):
        return key in caches[DEFAULT_CACHE_ALIAS]

    def __eq__(self, other):
        return caches[DEFAULT_CACHE_ALIAS] == other


cache = DefaultCacheProxy()


def close_caches(**kwargs):
    # Some caches -- python-memcached in particular -- need to do a cleanup at the
    # end of a request cycle. If not implemented in a particular backend
    # cache.close is a no-op
    for cache in caches.all():
        cache.close()


signals.request_finished.connect(close_caches)
```
### 63 - django/core/cache/__init__.py:

Start line: 57, End line: 87

```python
class CacheHandler:
    """
    A Cache Handler to manage access to Cache instances.

    Ensure only one instance of each alias exists per thread.
    """
    def __init__(self):
        self._caches = Local()

    def __getitem__(self, alias):
        try:
            return self._caches.caches[alias]
        except AttributeError:
            self._caches.caches = {}
        except KeyError:
            pass

        if alias not in settings.CACHES:
            raise InvalidCacheBackendError(
                "Could not find config for '%s' in settings.CACHES" % alias
            )

        cache = _create_cache(alias)
        self._caches.caches[alias] = cache
        return cache

    def all(self):
        return getattr(self._caches, 'caches', {}).values()


caches = CacheHandler()
```
