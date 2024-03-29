# django__django-15044

| **django/django** | `1f9874d4ca3e7376036646aedf6ac3060f22fd69` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2159 |
| **Any found context length** | 556 |
| **Avg pos** | 17.0 |
| **Min pos** | 2 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/middleware/cache.py b/django/middleware/cache.py
--- a/django/middleware/cache.py
+++ b/django/middleware/cache.py
@@ -67,7 +67,10 @@ def __init__(self, get_response):
         self.page_timeout = None
         self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
         self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
-        self.cache = caches[self.cache_alias]
+
+    @property
+    def cache(self):
+        return caches[self.cache_alias]
 
     def _should_update_cache(self, request, response):
         return hasattr(request, '_cache_update_cache') and request._cache_update_cache
@@ -126,7 +129,10 @@ def __init__(self, get_response):
         super().__init__(get_response)
         self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
         self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
-        self.cache = caches[self.cache_alias]
+
+    @property
+    def cache(self):
+        return caches[self.cache_alias]
 
     def process_request(self, request):
         """
@@ -183,7 +189,6 @@ def __init__(self, get_response, cache_timeout=None, page_timeout=None, **kwargs
             if cache_alias is None:
                 cache_alias = DEFAULT_CACHE_ALIAS
             self.cache_alias = cache_alias
-            self.cache = caches[self.cache_alias]
         except KeyError:
             pass
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/middleware/cache.py | 70 | 70 | 7 | 1 | 1904
| django/middleware/cache.py | 129 | 129 | 2 | 1 | 556
| django/middleware/cache.py | 186 | 186 | 8 | 1 | 2159


## Problem Statement

```
CacheMiddleware and FetchFromCacheMiddleware are not thread safe.
Description
	
CacheMiddleware persist self.cache = caches[cache_alias] on startup and it is not thread safe. ​https://github.com/django/django/blob/main/django/middleware/cache.py#L186
I found that after some production errors with pylibmc and uwsgi threaded. Created a small project to reproduce it. Nothing fancy, just pylibmc cache and a @cache_page cached view. It fails even with development server, with concurrent requests.
Traceback (most recent call last):
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/core/handlers/exception.py", line 47, in inner
	response = get_response(request)
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/core/handlers/base.py", line 181, in _get_response
	response = wrapped_callback(request, *callback_args, **callback_kwargs)
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/utils/decorators.py", line 122, in _wrapped_view
	result = middleware.process_request(request)
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/middleware/cache.py", line 145, in process_request
	cache_key = get_cache_key(request, self.key_prefix, 'GET', cache=self.cache)
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/utils/cache.py", line 362, in get_cache_key
	headerlist = cache.get(cache_key)
 File "versions/pylibmcbug/lib/python3.9/site-packages/django/core/cache/backends/memcached.py", line 77, in get
	return self._cache.get(key, default)
pylibmc.ConnectionError: error 3 from memcached_get(:1:views.decorators.cache.cache_): (0x7f290400bd60) FAILURE, poll() returned a value that was not dealt with, host: localhost:11211 -> libmemcached/io.cc:254
Looking for git history, it is this way since 2010. ​https://github.com/django/django/commit/673e6fc7fb243ed44841b9969d26a161c25733b3

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/middleware/cache.py** | 1 | 52| 431 | 431 | 1588 | 
| **-> 2 <-** | **1 django/middleware/cache.py** | 117 | 129| 125 | 556 | 1588 | 
| 3 | **1 django/middleware/cache.py** | 131 | 157| 252 | 808 | 1588 | 
| 4 | 2 django/core/cache/backends/memcached.py | 1 | 37| 220 | 1028 | 3066 | 
| 5 | 2 django/core/cache/backends/memcached.py | 136 | 158| 207 | 1235 | 3066 | 
| 6 | 3 django/core/cache/backends/locmem.py | 1 | 63| 495 | 1730 | 3951 | 
| **-> 7 <-** | **3 django/middleware/cache.py** | 55 | 73| 174 | 1904 | 3951 | 
| **-> 8 <-** | **3 django/middleware/cache.py** | 160 | 193| 255 | 2159 | 3951 | 
| 9 | 3 django/core/cache/backends/memcached.py | 66 | 95| 339 | 2498 | 3951 | 
| 10 | 3 django/core/cache/backends/memcached.py | 114 | 133| 187 | 2685 | 3951 | 
| 11 | 4 django/core/cache/__init__.py | 30 | 67| 249 | 2934 | 4420 | 
| 12 | 4 django/core/cache/backends/locmem.py | 79 | 118| 268 | 3202 | 4420 | 
| 13 | 5 django/core/cache/backends/base.py | 373 | 386| 112 | 3314 | 7496 | 
| 14 | 5 django/core/cache/backends/memcached.py | 97 | 112| 152 | 3466 | 7496 | 
| 15 | 5 django/core/cache/backends/memcached.py | 161 | 173| 120 | 3586 | 7496 | 
| 16 | **5 django/middleware/cache.py** | 75 | 114| 363 | 3949 | 7496 | 
| 17 | 5 django/core/cache/backends/locmem.py | 65 | 77| 134 | 4083 | 7496 | 
| 18 | 6 django/utils/deprecation.py | 76 | 126| 372 | 4455 | 8521 | 
| 19 | 6 django/utils/deprecation.py | 128 | 146| 122 | 4577 | 8521 | 
| 20 | 7 django/core/cache/backends/db.py | 106 | 190| 797 | 5374 | 10598 | 
| 21 | 8 django/core/cache/backends/redis.py | 153 | 225| 643 | 6017 | 12264 | 
| 22 | 9 django/utils/cache.py | 137 | 152| 190 | 6207 | 16018 | 
| 23 | 9 django/utils/cache.py | 1 | 34| 279 | 6486 | 16018 | 
| 24 | 9 django/core/cache/backends/base.py | 110 | 176| 614 | 7100 | 16018 | 
| 25 | 10 django/core/checks/caches.py | 22 | 56| 291 | 7391 | 16539 | 
| 26 | 11 django/core/handlers/asgi.py | 1 | 19| 111 | 7502 | 18875 | 
| 27 | 12 django/core/cache/utils.py | 1 | 13| 0 | 7502 | 18963 | 
| 28 | 13 django/views/decorators/cache.py | 28 | 42| 119 | 7621 | 19422 | 
| 29 | 13 django/core/cache/backends/db.py | 220 | 238| 238 | 7859 | 19422 | 
| 30 | 14 django/core/cache/backends/dummy.py | 1 | 35| 231 | 8090 | 19654 | 
| 31 | 15 django/contrib/auth/middleware.py | 1 | 25| 182 | 8272 | 20659 | 
| 32 | 15 django/core/cache/backends/db.py | 240 | 268| 323 | 8595 | 20659 | 
| 33 | 16 django/middleware/csrf.py | 1 | 54| 482 | 9077 | 24559 | 
| 34 | 17 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 9224 | 24707 | 
| 35 | 17 django/middleware/csrf.py | 388 | 441| 572 | 9796 | 24707 | 
| 36 | 17 django/core/cache/__init__.py | 1 | 27| 220 | 10016 | 24707 | 
| 37 | 17 django/contrib/auth/middleware.py | 48 | 84| 360 | 10376 | 24707 | 
| 38 | 17 django/core/cache/backends/db.py | 40 | 92| 423 | 10799 | 24707 | 
| 39 | 18 django/contrib/sessions/middleware.py | 1 | 76| 588 | 11387 | 25296 | 
| 40 | 18 django/core/cache/backends/db.py | 94 | 104| 222 | 11609 | 25296 | 
| 41 | 19 django/middleware/security.py | 31 | 58| 246 | 11855 | 25814 | 
| 42 | 19 django/core/cache/backends/db.py | 192 | 218| 279 | 12134 | 25814 | 
| 43 | 19 django/core/cache/backends/base.py | 1 | 53| 265 | 12399 | 25814 | 
| 44 | 19 django/utils/cache.py | 246 | 277| 257 | 12656 | 25814 | 
| 45 | 19 django/core/cache/backends/base.py | 56 | 95| 313 | 12969 | 25814 | 
| 46 | 20 django/core/handlers/base.py | 320 | 351| 212 | 13181 | 28421 | 
| 47 | 20 django/utils/cache.py | 155 | 195| 453 | 13634 | 28421 | 
| 48 | 20 django/core/cache/backends/memcached.py | 39 | 64| 283 | 13917 | 28421 | 
| 49 | 20 django/core/cache/backends/redis.py | 1 | 130| 881 | 14798 | 28421 | 
| 50 | 20 django/views/decorators/cache.py | 1 | 25| 215 | 15013 | 28421 | 
| 51 | 21 django/contrib/messages/middleware.py | 1 | 27| 174 | 15187 | 28596 | 
| 52 | 22 django/conf/global_settings.py | 501 | 652| 943 | 16130 | 34352 | 
| 53 | 23 django/utils/decorators.py | 114 | 152| 316 | 16446 | 35754 | 
| 54 | 24 django/core/cache/backends/filebased.py | 1 | 44| 289 | 16735 | 36987 | 
| 55 | 24 django/middleware/csrf.py | 252 | 276| 176 | 16911 | 36987 | 
| 56 | 25 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 17078 | 37554 | 
| 57 | 25 django/core/checks/caches.py | 1 | 19| 119 | 17197 | 37554 | 
| 58 | 25 django/core/checks/caches.py | 59 | 73| 109 | 17306 | 37554 | 
| 59 | 25 django/utils/decorators.py | 155 | 180| 136 | 17442 | 37554 | 
| 60 | 26 django/templatetags/cache.py | 1 | 49| 413 | 17855 | 38281 | 
| 61 | 27 django/core/checks/security/csrf.py | 1 | 42| 304 | 18159 | 38743 | 
| 62 | 27 django/middleware/csrf.py | 330 | 386| 495 | 18654 | 38743 | 
| 63 | 28 django/core/servers/basehttp.py | 81 | 105| 204 | 18858 | 40647 | 
| 64 | 28 django/middleware/security.py | 1 | 29| 278 | 19136 | 40647 | 
| 65 | 28 django/core/cache/backends/filebased.py | 61 | 96| 260 | 19396 | 40647 | 
| 66 | 28 django/core/cache/backends/filebased.py | 46 | 59| 145 | 19541 | 40647 | 
| 67 | 29 django/core/checks/security/base.py | 74 | 168| 746 | 20287 | 42688 | 
| 68 | 29 django/middleware/csrf.py | 443 | 456| 163 | 20450 | 42688 | 
| 69 | 30 django/contrib/sites/middleware.py | 1 | 13| 0 | 20450 | 42747 | 
| 70 | 31 django/contrib/redirects/middleware.py | 1 | 51| 354 | 20804 | 43102 | 
| 71 | 31 django/middleware/csrf.py | 234 | 250| 186 | 20990 | 43102 | 
| 72 | 32 django/db/backends/base/base.py | 1 | 34| 202 | 21192 | 48068 | 
| 73 | 32 django/views/decorators/cache.py | 45 | 60| 123 | 21315 | 48068 | 
| 74 | 32 django/conf/global_settings.py | 406 | 500| 782 | 22097 | 48068 | 
| 75 | 32 django/contrib/auth/middleware.py | 114 | 125| 107 | 22204 | 48068 | 
| 76 | 32 django/middleware/csrf.py | 278 | 328| 450 | 22654 | 48068 | 
| 77 | 32 django/middleware/csrf.py | 209 | 232| 194 | 22848 | 48068 | 
| 78 | 32 django/db/backends/base/base.py | 538 | 569| 227 | 23075 | 48068 | 
| 79 | 33 django/middleware/http.py | 1 | 42| 335 | 23410 | 48403 | 
| 80 | 33 django/utils/cache.py | 305 | 325| 211 | 23621 | 48403 | 
| 81 | 34 django/db/models/fields/mixins.py | 1 | 28| 168 | 23789 | 48746 | 
| 82 | 35 django/middleware/locale.py | 28 | 67| 388 | 24177 | 49367 | 
| 83 | 35 django/utils/cache.py | 328 | 346| 218 | 24395 | 49367 | 
| 84 | 35 django/core/cache/backends/base.py | 178 | 200| 199 | 24594 | 49367 | 
| 85 | 35 django/core/cache/backends/base.py | 301 | 324| 203 | 24797 | 49367 | 
| 86 | 35 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 24995 | 49367 | 
| 87 | 35 django/contrib/auth/middleware.py | 28 | 46| 178 | 25173 | 49367 | 
| 88 | 36 django/template/loaders/cached.py | 1 | 65| 497 | 25670 | 50085 | 
| 89 | 37 django/template/context.py | 1 | 24| 128 | 25798 | 51966 | 
| 90 | 37 django/core/checks/security/base.py | 1 | 72| 660 | 26458 | 51966 | 
| 91 | 37 django/contrib/sessions/backends/cache.py | 1 | 34| 213 | 26671 | 51966 | 
| 92 | 37 django/utils/cache.py | 371 | 417| 521 | 27192 | 51966 | 
| 93 | 38 django/middleware/gzip.py | 1 | 52| 419 | 27611 | 52386 | 
| 94 | 39 django/db/utils.py | 134 | 157| 227 | 27838 | 54393 | 
| 95 | 40 django/db/models/query.py | 1607 | 1663| 487 | 28325 | 72052 | 
| 96 | 40 django/core/cache/backends/redis.py | 132 | 150| 147 | 28472 | 72052 | 
| 97 | 41 django/middleware/common.py | 34 | 61| 257 | 28729 | 73581 | 
| 98 | 41 django/core/cache/backends/db.py | 1 | 37| 229 | 28958 | 73581 | 
| 99 | 42 django/core/checks/async_checks.py | 1 | 17| 0 | 28958 | 73675 | 
| 100 | 43 django/db/models/fields/related.py | 267 | 298| 284 | 29242 | 87671 | 
| 101 | 44 django/middleware/clickjacking.py | 1 | 48| 364 | 29606 | 88035 | 
| 102 | 45 django/urls/base.py | 89 | 155| 383 | 29989 | 89218 | 
| 103 | 45 django/core/cache/backends/filebased.py | 98 | 114| 174 | 30163 | 89218 | 
| 104 | 45 django/contrib/auth/middleware.py | 86 | 111| 192 | 30355 | 89218 | 
| 105 | 45 django/middleware/common.py | 1 | 32| 247 | 30602 | 89218 | 
| 106 | 46 django/utils/autoreload.py | 375 | 414| 266 | 30868 | 94366 | 
| 107 | 47 django/core/handlers/wsgi.py | 64 | 119| 486 | 31354 | 96125 | 
| 108 | 48 django/http/request.py | 1 | 39| 273 | 31627 | 101331 | 
| 109 | 49 django/views/csrf.py | 15 | 100| 835 | 32462 | 102875 | 
| 110 | 49 django/urls/base.py | 1 | 24| 170 | 32632 | 102875 | 
| 111 | 49 django/db/models/fields/related.py | 139 | 166| 201 | 32833 | 102875 | 
| 112 | 50 django/db/backends/utils.py | 1 | 45| 278 | 33111 | 104766 | 
| 113 | 51 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 33364 | 105187 | 
| 114 | 51 django/utils/cache.py | 198 | 216| 184 | 33548 | 105187 | 
| 115 | 52 django/db/migrations/loader.py | 291 | 315| 205 | 33753 | 108295 | 
| 116 | 52 django/middleware/common.py | 63 | 75| 136 | 33889 | 108295 | 
| 117 | 53 django/contrib/auth/views.py | 1 | 37| 284 | 34173 | 111025 | 
| 118 | 53 django/db/models/query.py | 1695 | 1810| 1098 | 35271 | 111025 | 
| 119 | 53 django/utils/cache.py | 219 | 243| 235 | 35506 | 111025 | 
| 120 | 53 django/utils/cache.py | 349 | 368| 190 | 35696 | 111025 | 
| 121 | 54 django/utils/module_loading.py | 1 | 16| 109 | 35805 | 111816 | 
| 122 | 54 django/template/loaders/cached.py | 67 | 98| 225 | 36030 | 111816 | 
| 123 | 54 django/utils/autoreload.py | 320 | 335| 146 | 36176 | 111816 | 
| 124 | 54 django/core/servers/basehttp.py | 222 | 239| 210 | 36386 | 111816 | 
| 125 | 55 django/contrib/sites/models.py | 25 | 46| 192 | 36578 | 112604 | 
| 126 | 55 django/utils/cache.py | 105 | 134| 190 | 36768 | 112604 | 
| 127 | 56 django/db/models/base.py | 1 | 50| 328 | 37096 | 129934 | 
| 128 | 56 django/core/handlers/base.py | 212 | 275| 480 | 37576 | 129934 | 
| 129 | 57 django/utils/itercompat.py | 1 | 9| 0 | 37576 | 129974 | 
| 130 | 57 django/db/models/fields/related.py | 120 | 137| 155 | 37731 | 129974 | 
| 131 | 58 django/db/backends/postgresql/base.py | 255 | 280| 227 | 37958 | 132898 | 
| 132 | 58 django/middleware/common.py | 149 | 175| 254 | 38212 | 132898 | 
| 133 | 59 django/contrib/messages/storage/fallback.py | 1 | 36| 271 | 38483 | 133314 | 
| 134 | 60 django/dispatch/__init__.py | 1 | 10| 0 | 38483 | 133379 | 
| 135 | 60 django/utils/autoreload.py | 555 | 578| 177 | 38660 | 133379 | 
| 136 | 60 django/core/cache/backends/base.py | 233 | 283| 457 | 39117 | 133379 | 
| 137 | 60 django/core/handlers/base.py | 1 | 97| 743 | 39860 | 133379 | 
| 138 | 61 django/db/models/query_utils.py | 94 | 131| 300 | 40160 | 135867 | 
| 139 | 61 django/db/models/fields/related.py | 198 | 266| 687 | 40847 | 135867 | 
| 140 | 61 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 41019 | 135867 | 
| 141 | 61 django/db/models/fields/related.py | 183 | 196| 140 | 41159 | 135867 | 
| 142 | 61 django/core/cache/backends/base.py | 202 | 219| 194 | 41353 | 135867 | 
| 143 | 61 django/core/servers/basehttp.py | 144 | 179| 280 | 41633 | 135867 | 
| 144 | 62 django/http/__init__.py | 1 | 22| 197 | 41830 | 136064 | 
| 145 | 62 django/core/servers/basehttp.py | 181 | 200| 170 | 42000 | 136064 | 
| 146 | 62 django/db/models/query.py | 1932 | 1964| 314 | 42314 | 136064 | 
| 147 | 62 django/core/servers/basehttp.py | 124 | 141| 179 | 42493 | 136064 | 
| 148 | 63 django/core/mail/backends/__init__.py | 1 | 2| 0 | 42493 | 136072 | 
| 149 | 63 django/core/checks/security/base.py | 234 | 258| 210 | 42703 | 136072 | 
| 150 | 63 django/middleware/common.py | 100 | 115| 165 | 42868 | 136072 | 
| 151 | 64 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 43400 | 136928 | 
| 152 | 64 django/core/handlers/base.py | 126 | 158| 253 | 43653 | 136928 | 
| 153 | 64 django/utils/cache.py | 37 | 102| 559 | 44212 | 136928 | 
| 154 | 65 django/core/checks/security/sessions.py | 1 | 98| 572 | 44784 | 137501 | 
| 155 | 65 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 44905 | 137501 | 
| 156 | 65 django/core/cache/backends/filebased.py | 116 | 166| 392 | 45297 | 137501 | 
| 157 | 66 django/contrib/staticfiles/storage.py | 374 | 395| 230 | 45527 | 141413 | 
| 158 | 66 django/utils/autoreload.py | 417 | 447| 349 | 45876 | 141413 | 
| 159 | 66 django/middleware/locale.py | 1 | 26| 239 | 46115 | 141413 | 
| 160 | 67 django/views/debug.py | 150 | 162| 148 | 46263 | 146166 | 
| 161 | 67 django/contrib/staticfiles/storage.py | 284 | 354| 575 | 46838 | 146166 | 
| 162 | 67 django/db/models/fields/related.py | 168 | 181| 144 | 46982 | 146166 | 
| 163 | 67 django/core/cache/backends/base.py | 97 | 108| 119 | 47101 | 146166 | 
| 164 | 67 django/db/utils.py | 255 | 297| 322 | 47423 | 146166 | 
| 165 | 67 django/contrib/messages/storage/fallback.py | 38 | 55| 150 | 47573 | 146166 | 
| 166 | 68 django/http/response.py | 1 | 24| 162 | 47735 | 150880 | 
| 167 | 68 django/db/backends/base/base.py | 571 | 618| 300 | 48035 | 150880 | 
| 168 | 68 django/core/checks/security/base.py | 171 | 196| 188 | 48223 | 150880 | 
| 169 | 68 django/db/backends/postgresql/base.py | 1 | 62| 456 | 48679 | 150880 | 
| 170 | 68 django/contrib/sites/models.py | 78 | 121| 236 | 48915 | 150880 | 
| 171 | 68 django/utils/autoreload.py | 1 | 56| 287 | 49202 | 150880 | 
| 172 | 68 django/core/cache/backends/base.py | 285 | 299| 147 | 49349 | 150880 | 
| 173 | 68 django/core/cache/backends/base.py | 326 | 340| 120 | 49469 | 150880 | 
| 174 | 69 django/utils/asyncio.py | 1 | 34| 214 | 49683 | 151095 | 
| 175 | 69 django/templatetags/cache.py | 52 | 94| 313 | 49996 | 151095 | 
| 176 | 70 django/http/multipartparser.py | 432 | 451| 205 | 50201 | 156317 | 
| 177 | 70 django/conf/global_settings.py | 156 | 271| 859 | 51060 | 156317 | 
| 178 | 71 django/contrib/gis/views.py | 1 | 21| 155 | 51215 | 156472 | 
| 179 | 72 django/contrib/auth/mixins.py | 44 | 71| 235 | 51450 | 157336 | 
| 180 | 73 django/contrib/admin/tests.py | 1 | 36| 265 | 51715 | 158960 | 
| 181 | 73 django/db/migrations/loader.py | 159 | 185| 291 | 52006 | 158960 | 
| 182 | 74 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 52006 | 159030 | 
| 183 | 75 django/core/mail/backends/console.py | 1 | 43| 281 | 52287 | 159312 | 
| 184 | 75 django/utils/decorators.py | 89 | 111| 152 | 52439 | 159312 | 
| 185 | 76 django/core/files/storage.py | 1 | 24| 171 | 52610 | 162264 | 
| 186 | 77 django/utils/connection.py | 1 | 31| 192 | 52802 | 162728 | 
| 187 | 78 django/views/decorators/gzip.py | 1 | 6| 0 | 52802 | 162779 | 
| 188 | 78 django/http/response.py | 526 | 557| 153 | 52955 | 162779 | 
| 189 | 78 django/http/response.py | 65 | 82| 133 | 53088 | 162779 | 
| 190 | 79 django/utils/functional.py | 1 | 49| 336 | 53424 | 165874 | 
| 191 | 80 django/core/wsgi.py | 1 | 14| 0 | 53424 | 165964 | 
| 192 | 80 django/db/models/query.py | 1867 | 1930| 658 | 54082 | 165964 | 
| 193 | 81 django/core/mail/message.py | 177 | 185| 115 | 54197 | 169615 | 
| 194 | 81 django/db/models/query_utils.py | 134 | 198| 492 | 54689 | 169615 | 
| 195 | 82 django/contrib/admin/sites.py | 224 | 243| 221 | 54910 | 174044 | 
| 196 | 82 django/core/checks/security/csrf.py | 45 | 68| 157 | 55067 | 174044 | 
| 197 | 82 django/utils/functional.py | 131 | 199| 520 | 55587 | 174044 | 
| 198 | 83 django/contrib/auth/migrations/0001_initial.py | 1 | 104| 843 | 56430 | 174887 | 
| 199 | 83 django/http/request.py | 290 | 310| 189 | 56619 | 174887 | 
| 200 | 83 django/utils/connection.py | 34 | 77| 271 | 56890 | 174887 | 
| 201 | 84 django/contrib/auth/admin.py | 1 | 22| 188 | 57078 | 176624 | 
| 202 | 85 django/core/mail/backends/locmem.py | 1 | 31| 183 | 57261 | 176808 | 
| 203 | 86 django/core/mail/backends/dummy.py | 1 | 11| 0 | 57261 | 176851 | 
| 204 | 86 django/http/response.py | 560 | 585| 159 | 57420 | 176851 | 
| 205 | 87 django/contrib/gis/geoip2/base.py | 142 | 162| 258 | 57678 | 178867 | 
| 206 | 88 django/contrib/auth/hashers.py | 86 | 108| 167 | 57845 | 184830 | 
| 207 | 88 django/db/models/fields/related.py | 300 | 334| 293 | 58138 | 184830 | 
| 208 | 88 django/core/management/commands/createcachetable.py | 1 | 29| 213 | 58351 | 184830 | 
| 209 | 89 django/core/handlers/exception.py | 54 | 122| 557 | 58908 | 185898 | 
| 210 | 89 django/db/models/query.py | 1666 | 1694| 246 | 59154 | 185898 | 


### Hint

```
PR ​https://github.com/django/django/pull/15044
Possibly related to #33092.
Thanks for the report!
```

## Patch

```diff
diff --git a/django/middleware/cache.py b/django/middleware/cache.py
--- a/django/middleware/cache.py
+++ b/django/middleware/cache.py
@@ -67,7 +67,10 @@ def __init__(self, get_response):
         self.page_timeout = None
         self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
         self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
-        self.cache = caches[self.cache_alias]
+
+    @property
+    def cache(self):
+        return caches[self.cache_alias]
 
     def _should_update_cache(self, request, response):
         return hasattr(request, '_cache_update_cache') and request._cache_update_cache
@@ -126,7 +129,10 @@ def __init__(self, get_response):
         super().__init__(get_response)
         self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
         self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
-        self.cache = caches[self.cache_alias]
+
+    @property
+    def cache(self):
+        return caches[self.cache_alias]
 
     def process_request(self, request):
         """
@@ -183,7 +189,6 @@ def __init__(self, get_response, cache_timeout=None, page_timeout=None, **kwargs
             if cache_alias is None:
                 cache_alias = DEFAULT_CACHE_ALIAS
             self.cache_alias = cache_alias
-            self.cache = caches[self.cache_alias]
         except KeyError:
             pass
 

```

## Test Patch

```diff
diff --git a/tests/cache/tests.py b/tests/cache/tests.py
--- a/tests/cache/tests.py
+++ b/tests/cache/tests.py
@@ -994,9 +994,9 @@ def test_custom_key_func(self):
         self.assertEqual(caches['custom_key'].get('answer2'), 42)
         self.assertEqual(caches['custom_key2'].get('answer2'), 42)
 
+    @override_settings(CACHE_MIDDLEWARE_ALIAS=DEFAULT_CACHE_ALIAS)
     def test_cache_write_unpicklable_object(self):
         fetch_middleware = FetchFromCacheMiddleware(empty_response)
-        fetch_middleware.cache = cache
 
         request = self.factory.get('/cache/test')
         request._cache_update_cache = True
@@ -1011,7 +1011,6 @@ def get_response(req):
             return response
 
         update_middleware = UpdateCacheMiddleware(get_response)
-        update_middleware.cache = cache
         response = update_middleware(request)
 
         get_cache_data = fetch_middleware.process_request(request)
@@ -2489,6 +2488,21 @@ def test_304_response_has_http_caching_headers_but_not_cached(self):
         self.assertIn('Cache-Control', response)
         self.assertIn('Expires', response)
 
+    def test_per_thread(self):
+        """The cache instance is different for each thread."""
+        thread_caches = []
+        middleware = CacheMiddleware(empty_response)
+
+        def runner():
+            thread_caches.append(middleware.cache)
+
+        for _ in range(2):
+            thread = threading.Thread(target=runner)
+            thread.start()
+            thread.join()
+
+        self.assertIsNot(thread_caches[0], thread_caches[1])
+
 
 @override_settings(
     CACHE_MIDDLEWARE_KEY_PREFIX='settingsprefix',

```


## Code snippets

### 1 - django/middleware/cache.py:

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
### 2 - django/middleware/cache.py:

Start line: 117, End line: 129

```python
class FetchFromCacheMiddleware(MiddlewareMixin):
    """
    Request-phase cache middleware that fetches a page from the cache.

    Must be used as part of the two-part update/fetch cache middleware.
    FetchFromCacheMiddleware must be the last piece of middleware in MIDDLEWARE
    so that it'll get called last during the request phase.
    """
    def __init__(self, get_response):
        super().__init__(get_response)
        self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
        self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
        self.cache = caches[self.cache_alias]
```
### 3 - django/middleware/cache.py:

Start line: 131, End line: 157

```python
class FetchFromCacheMiddleware(MiddlewareMixin):

    def process_request(self, request):
        """
        Check whether the page is already cached and return the cached
        version if available.
        """
        if request.method not in ('GET', 'HEAD'):
            request._cache_update_cache = False
            return None  # Don't bother checking the cache.

        # try and get the cached GET response
        cache_key = get_cache_key(request, self.key_prefix, 'GET', cache=self.cache)
        if cache_key is None:
            request._cache_update_cache = True
            return None  # No cache information available, need to rebuild.
        response = self.cache.get(cache_key)
        # if it wasn't found and we are looking for a HEAD, try looking just for that
        if response is None and request.method == 'HEAD':
            cache_key = get_cache_key(request, self.key_prefix, 'HEAD', cache=self.cache)
            response = self.cache.get(cache_key)

        if response is None:
            request._cache_update_cache = True
            return None  # No cache information available, need to rebuild.

        # hit, return cached response
        request._cache_update_cache = False
        return response
```
### 4 - django/core/cache/backends/memcached.py:

Start line: 1, End line: 37

```python
"Memcached cache backend"

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
### 5 - django/core/cache/backends/memcached.py:

Start line: 136, End line: 158

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
        key = self.make_and_validate_key(key, version=version)
        if timeout == 0:
            return self._cache.delete(key)
        return self._cache.touch(key, self.get_backend_timeout(timeout))

    def close(self, **kwargs):
        # libmemcached manages its own connections. Don't call disconnect_all()
        # as it resets the failover state and creates unnecessary reconnects.
        pass
```
### 6 - django/core/cache/backends/locmem.py:

Start line: 1, End line: 63

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
        key = self.make_and_validate_key(key, version=version)
        pickled = pickle.dumps(value, self.pickle_protocol)
        with self._lock:
            if self._has_expired(key):
                self._set(key, pickled, timeout)
                return True
            return False

    def get(self, key, default=None, version=None):
        key = self.make_and_validate_key(key, version=version)
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
        key = self.make_and_validate_key(key, version=version)
        pickled = pickle.dumps(value, self.pickle_protocol)
        with self._lock:
            self._set(key, pickled, timeout)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_and_validate_key(key, version=version)
        with self._lock:
            if self._has_expired(key):
                return False
            self._expire_info[key] = self.get_backend_timeout(timeout)
            return True
```
### 7 - django/middleware/cache.py:

Start line: 55, End line: 73

```python
class UpdateCacheMiddleware(MiddlewareMixin):
    """
    Response-phase cache middleware that updates the cache if the response is
    cacheable.

    Must be used as part of the two-part update/fetch cache middleware.
    UpdateCacheMiddleware must be the first piece of middleware in MIDDLEWARE
    so that it'll get called last during the response phase.
    """
    def __init__(self, get_response):
        super().__init__(get_response)
        self.cache_timeout = settings.CACHE_MIDDLEWARE_SECONDS
        self.page_timeout = None
        self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
        self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS
        self.cache = caches[self.cache_alias]

    def _should_update_cache(self, request, response):
        return hasattr(request, '_cache_update_cache') and request._cache_update_cache
```
### 8 - django/middleware/cache.py:

Start line: 160, End line: 193

```python
class CacheMiddleware(UpdateCacheMiddleware, FetchFromCacheMiddleware):
    """
    Cache middleware that provides basic behavior for many simple sites.

    Also used as the hook point for the cache decorator, which is generated
    using the decorator-from-middleware utility.
    """
    def __init__(self, get_response, cache_timeout=None, page_timeout=None, **kwargs):
        super().__init__(get_response)
        # We need to differentiate between "provided, but using default value",
        # and "not provided". If the value is provided using a default, then
        # we fall back to system defaults. If it is not provided at all,
        # we need to use middleware defaults.

        try:
            key_prefix = kwargs['key_prefix']
            if key_prefix is None:
                key_prefix = ''
            self.key_prefix = key_prefix
        except KeyError:
            pass
        try:
            cache_alias = kwargs['cache_alias']
            if cache_alias is None:
                cache_alias = DEFAULT_CACHE_ALIAS
            self.cache_alias = cache_alias
            self.cache = caches[self.cache_alias]
        except KeyError:
            pass

        if cache_timeout is not None:
            self.cache_timeout = cache_timeout
        self.page_timeout = page_timeout
```
### 9 - django/core/cache/backends/memcached.py:

Start line: 66, End line: 95

```python
class BaseMemcachedCache(BaseCache):

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_and_validate_key(key, version=version)
        return self._cache.add(key, value, self.get_backend_timeout(timeout))

    def get(self, key, default=None, version=None):
        key = self.make_and_validate_key(key, version=version)
        return self._cache.get(key, default)

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_and_validate_key(key, version=version)
        if not self._cache.set(key, value, self.get_backend_timeout(timeout)):
            # make sure the key doesn't keep its old value in case of failure to set (memcached's 1MB limit)
            self._cache.delete(key)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_and_validate_key(key, version=version)
        return bool(self._cache.touch(key, self.get_backend_timeout(timeout)))

    def delete(self, key, version=None):
        key = self.make_and_validate_key(key, version=version)
        return bool(self._cache.delete(key))

    def get_many(self, keys, version=None):
        key_map = {self.make_and_validate_key(key, version=version): key for key in keys}
        ret = self._cache.get_multi(key_map.keys())
        return {key_map[k]: v for k, v in ret.items()}

    def close(self, **kwargs):
        # Many clients don't clean up connections properly.
        self._cache.disconnect_all()
```
### 10 - django/core/cache/backends/memcached.py:

Start line: 114, End line: 133

```python
class BaseMemcachedCache(BaseCache):

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        safe_data = {}
        original_keys = {}
        for key, value in data.items():
            safe_key = self.make_and_validate_key(key, version=version)
            safe_data[safe_key] = value
            original_keys[safe_key] = key
        failed_keys = self._cache.set_multi(safe_data, self.get_backend_timeout(timeout))
        return [original_keys[k] for k in failed_keys]

    def delete_many(self, keys, version=None):
        keys = [self.make_and_validate_key(key, version=version) for key in keys]
        self._cache.delete_multi(keys)

    def clear(self):
        self._cache.flush_all()

    def validate_key(self, key):
        for warning in memcache_key_warnings(key):
            raise InvalidCacheKey(warning)
```
### 16 - django/middleware/cache.py:

Start line: 75, End line: 114

```python
class UpdateCacheMiddleware(MiddlewareMixin):

    def process_response(self, request, response):
        """Set the cache, if needed."""
        if not self._should_update_cache(request, response):
            # We don't need to update the cache, just return.
            return response

        if response.streaming or response.status_code not in (200, 304):
            return response

        # Don't cache responses that set a user-specific (and maybe security
        # sensitive) cookie in response to a cookie-less request.
        if not request.COOKIES and response.cookies and has_vary_header(response, 'Cookie'):
            return response

        # Don't cache a response with 'Cache-Control: private'
        if 'private' in response.get('Cache-Control', ()):
            return response

        # Page timeout takes precedence over the "max-age" and the default
        # cache timeout.
        timeout = self.page_timeout
        if timeout is None:
            # The timeout from the "max-age" section of the "Cache-Control"
            # header takes precedence over the default cache timeout.
            timeout = get_max_age(response)
            if timeout is None:
                timeout = self.cache_timeout
            elif timeout == 0:
                # max-age was set to 0, don't cache.
                return response
        patch_response_headers(response, timeout)
        if timeout and response.status_code == 200:
            cache_key = learn_cache_key(request, response, timeout, self.key_prefix, cache=self.cache)
            if hasattr(response, 'render') and callable(response.render):
                response.add_post_render_callback(
                    lambda r: self.cache.set(cache_key, r, timeout)
                )
            else:
                self.cache.set(cache_key, response, timeout)
        return response
```
