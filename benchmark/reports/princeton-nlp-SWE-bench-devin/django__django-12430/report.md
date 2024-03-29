# django__django-12430

| **django/django** | `20ba3ce4ac8e8438070568ffba76f7d8d4986a53` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1566 |
| **Any found context length** | 178 |
| **Avg pos** | 7.0 |
| **Min pos** | 1 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/cache/__init__.py b/django/core/cache/__init__.py
--- a/django/core/cache/__init__.py
+++ b/django/core/cache/__init__.py
@@ -12,7 +12,7 @@
 
 See docs/topics/cache.txt for information on the public API.
 """
-from threading import local
+from asgiref.local import Local
 
 from django.conf import settings
 from django.core import signals
@@ -61,7 +61,7 @@ class CacheHandler:
     Ensure only one instance of each alias exists per thread.
     """
     def __init__(self):
-        self._caches = local()
+        self._caches = Local()
 
     def __getitem__(self, alias):
         try:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/cache/__init__.py | 15 | 15 | 6 | 1 | 1566
| django/core/cache/__init__.py | 64 | 64 | 1 | 1 | 178


## Problem Statement

```
Possible data loss when using caching from async code.
Description
	
CacheHandler use threading.local instead of asgiref.local.Local, hence it's a chance of data corruption if someone tries to use caching from async code. There is a potential race condition if two coroutines touch the same cache object at exactly the same time.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/cache/__init__.py** | 57 | 87| 178 | 178 | 813 | 
| 2 | 2 django/core/cache/backends/locmem.py | 83 | 124| 276 | 454 | 1720 | 
| 3 | 2 django/core/cache/backends/locmem.py | 1 | 66| 505 | 959 | 1720 | 
| 4 | 2 django/core/cache/backends/locmem.py | 68 | 81| 138 | 1097 | 1720 | 
| 5 | 3 django/core/cache/backends/db.py | 230 | 253| 259 | 1356 | 3826 | 
| **-> 6 <-** | **3 django/core/cache/__init__.py** | 1 | 29| 210 | 1566 | 3826 | 
| 7 | 3 django/core/cache/backends/db.py | 255 | 280| 308 | 1874 | 3826 | 
| 8 | 4 django/core/cache/backends/memcached.py | 128 | 142| 151 | 2025 | 5643 | 
| 9 | 4 django/core/cache/backends/db.py | 112 | 197| 794 | 2819 | 5643 | 
| 10 | 4 django/core/cache/backends/db.py | 97 | 110| 234 | 3053 | 5643 | 
| 11 | 4 django/core/cache/backends/memcached.py | 145 | 178| 362 | 3415 | 5643 | 
| 12 | 4 django/core/cache/backends/db.py | 199 | 228| 285 | 3700 | 5643 | 
| 13 | 4 django/core/cache/backends/memcached.py | 92 | 108| 167 | 3867 | 5643 | 
| 14 | 5 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 4034 | 6210 | 
| 15 | 5 django/core/cache/backends/memcached.py | 110 | 126| 168 | 4202 | 6210 | 
| 16 | 6 django/utils/cache.py | 1 | 35| 284 | 4486 | 9958 | 
| 17 | 6 django/core/cache/backends/memcached.py | 181 | 201| 189 | 4675 | 9958 | 
| 18 | 6 django/core/cache/backends/memcached.py | 1 | 36| 248 | 4923 | 9958 | 
| 19 | 6 django/core/cache/backends/db.py | 40 | 95| 431 | 5354 | 9958 | 
| 20 | 7 django/core/cache/utils.py | 1 | 13| 0 | 5354 | 10037 | 
| 21 | 8 django/middleware/cache.py | 55 | 73| 175 | 5529 | 11655 | 
| 22 | 9 django/core/cache/backends/base.py | 239 | 256| 165 | 5694 | 13812 | 
| 23 | **9 django/core/cache/__init__.py** | 90 | 125| 229 | 5923 | 13812 | 
| 24 | 10 django/core/cache/backends/dummy.py | 1 | 40| 255 | 6178 | 14068 | 
| 25 | 10 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 6376 | 14068 | 
| 26 | **10 django/core/cache/__init__.py** | 32 | 54| 196 | 6572 | 14068 | 
| 27 | 11 django/core/checks/caches.py | 1 | 17| 0 | 6572 | 14168 | 
| 28 | 12 django/utils/asyncio.py | 1 | 35| 222 | 6794 | 14391 | 
| 29 | 12 django/core/cache/backends/memcached.py | 65 | 90| 285 | 7079 | 14391 | 
| 30 | 13 django/db/models/query_utils.py | 127 | 164| 306 | 7385 | 17103 | 
| 31 | 13 django/utils/cache.py | 136 | 151| 188 | 7573 | 17103 | 
| 32 | 14 django/db/models/fields/related.py | 255 | 282| 269 | 7842 | 30787 | 
| 33 | 14 django/middleware/cache.py | 131 | 157| 252 | 8094 | 30787 | 
| 34 | 15 django/contrib/gis/geos/prototypes/threadsafe.py | 1 | 26| 169 | 8263 | 31342 | 
| 35 | 15 django/middleware/cache.py | 1 | 52| 431 | 8694 | 31342 | 
| 36 | 16 django/core/cache/backends/filebased.py | 98 | 114| 174 | 8868 | 32515 | 
| 37 | 16 django/middleware/cache.py | 117 | 129| 125 | 8993 | 32515 | 
| 38 | 16 django/core/cache/backends/base.py | 1 | 47| 245 | 9238 | 32515 | 
| 39 | 16 django/core/cache/backends/filebased.py | 61 | 96| 260 | 9498 | 32515 | 
| 40 | 17 django/views/decorators/cache.py | 27 | 48| 153 | 9651 | 32878 | 
| 41 | 17 django/core/cache/backends/filebased.py | 46 | 59| 145 | 9796 | 32878 | 
| 42 | 17 django/contrib/gis/geos/prototypes/threadsafe.py | 29 | 78| 386 | 10182 | 32878 | 
| 43 | 18 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 10354 | 33299 | 
| 44 | 19 django/templatetags/cache.py | 1 | 49| 413 | 10767 | 34026 | 
| 45 | 20 django/contrib/staticfiles/storage.py | 342 | 363| 230 | 10997 | 37556 | 
| 46 | 21 django/db/models/query.py | 1843 | 1875| 314 | 11311 | 54575 | 
| 47 | 21 django/middleware/cache.py | 160 | 195| 284 | 11595 | 54575 | 
| 48 | 21 django/core/cache/backends/filebased.py | 1 | 44| 284 | 11879 | 54575 | 
| 49 | 21 django/db/models/fields/related.py | 190 | 254| 673 | 12552 | 54575 | 
| 50 | 22 django/urls/base.py | 1 | 25| 177 | 12729 | 55761 | 
| 51 | 23 django/db/backends/postgresql/base.py | 239 | 269| 267 | 12996 | 58564 | 
| 52 | 23 django/utils/cache.py | 242 | 273| 256 | 13252 | 58564 | 
| 53 | 24 django/core/exceptions.py | 99 | 194| 649 | 13901 | 59619 | 
| 54 | 25 django/utils/translation/trans_real.py | 1 | 58| 503 | 14404 | 63465 | 
| 55 | 26 django/utils/autoreload.py | 339 | 378| 266 | 14670 | 68182 | 
| 56 | 26 django/contrib/sessions/backends/cache.py | 1 | 34| 213 | 14883 | 68182 | 
| 57 | 27 django/utils/datastructures.py | 296 | 340| 298 | 15181 | 70439 | 
| 58 | 28 django/db/models/base.py | 385 | 401| 128 | 15309 | 85811 | 
| 59 | 29 django/template/loaders/cached.py | 62 | 93| 225 | 15534 | 86499 | 
| 60 | 29 django/utils/cache.py | 301 | 321| 217 | 15751 | 86499 | 
| 61 | 29 django/utils/cache.py | 154 | 191| 447 | 16198 | 86499 | 
| 62 | 29 django/templatetags/cache.py | 52 | 94| 313 | 16511 | 86499 | 
| 63 | 30 django/contrib/gis/geos/libgeos.py | 73 | 131| 345 | 16856 | 87754 | 
| 64 | 30 django/db/models/fields/related.py | 127 | 154| 202 | 17058 | 87754 | 
| 65 | 30 django/core/cache/backends/base.py | 50 | 87| 306 | 17364 | 87754 | 
| 66 | 30 django/core/cache/backends/memcached.py | 38 | 63| 283 | 17647 | 87754 | 
| 67 | 30 django/utils/cache.py | 367 | 413| 531 | 18178 | 87754 | 
| 68 | 30 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 18431 | 87754 | 
| 69 | 31 django/core/handlers/asgi.py | 127 | 173| 402 | 18833 | 90068 | 
| 70 | 31 django/template/loaders/cached.py | 1 | 60| 467 | 19300 | 90068 | 
| 71 | 31 django/contrib/staticfiles/storage.py | 252 | 322| 575 | 19875 | 90068 | 
| 72 | 32 django/utils/functional.py | 125 | 187| 484 | 20359 | 93094 | 
| 73 | 32 django/middleware/cache.py | 75 | 114| 363 | 20722 | 93094 | 
| 74 | 33 django/db/utils.py | 207 | 237| 194 | 20916 | 95240 | 
| 75 | 33 django/db/models/query.py | 1778 | 1841| 658 | 21574 | 95240 | 
| 76 | 33 django/utils/cache.py | 324 | 342| 214 | 21788 | 95240 | 
| 77 | 33 django/core/cache/backends/base.py | 176 | 208| 277 | 22065 | 95240 | 
| 78 | 33 django/utils/functional.py | 105 | 123| 208 | 22273 | 95240 | 
| 79 | 33 django/utils/functional.py | 1 | 49| 336 | 22609 | 95240 | 
| 80 | 34 django/conf/__init__.py | 66 | 96| 260 | 22869 | 97299 | 
| 81 | 35 django/core/servers/basehttp.py | 97 | 119| 211 | 23080 | 99044 | 
| 82 | 35 django/db/utils.py | 132 | 148| 151 | 23231 | 99044 | 
| 83 | 36 django/db/backends/base/base.py | 527 | 558| 227 | 23458 | 103927 | 
| 84 | 37 django/core/files/storage.py | 296 | 368| 483 | 23941 | 106798 | 
| 85 | 38 django/contrib/sessions/backends/file.py | 111 | 170| 530 | 24471 | 108299 | 
| 86 | 38 django/core/cache/backends/db.py | 1 | 37| 229 | 24700 | 108299 | 
| 87 | 39 django/db/migrations/loader.py | 275 | 299| 205 | 24905 | 111189 | 
| 88 | 39 django/db/models/fields/related.py | 171 | 188| 166 | 25071 | 111189 | 
| 89 | 40 django/db/models/sql/query.py | 696 | 731| 389 | 25460 | 132858 | 
| 90 | 40 django/db/backends/base/base.py | 1 | 23| 138 | 25598 | 132858 | 
| 91 | 41 django/db/backends/sqlite3/base.py | 554 | 567| 135 | 25733 | 138623 | 
| 92 | 42 django/core/serializers/base.py | 232 | 249| 208 | 25941 | 141048 | 
| 93 | 42 django/db/models/query.py | 1534 | 1590| 481 | 26422 | 141048 | 
| 94 | 43 django/contrib/contenttypes/fields.py | 219 | 270| 459 | 26881 | 146481 | 
| 95 | 44 django/contrib/auth/hashers.py | 328 | 341| 120 | 27001 | 151268 | 
| 96 | 45 django/db/models/lookups.py | 119 | 142| 220 | 27221 | 156091 | 
| 97 | 45 django/db/utils.py | 1 | 49| 154 | 27375 | 156091 | 
| 98 | 46 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 27496 | 156948 | 
| 99 | 47 django/db/transaction.py | 196 | 282| 622 | 28118 | 159092 | 
| 100 | 47 django/core/cache/backends/base.py | 258 | 284| 182 | 28300 | 159092 | 
| 101 | 47 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 28423 | 159092 | 
| 102 | 47 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 28955 | 159092 | 
| 103 | 48 django/db/models/fields/mixins.py | 1 | 28| 168 | 29123 | 159435 | 
| 104 | 48 django/db/models/query.py | 1622 | 1728| 1063 | 30186 | 159435 | 
| 105 | 49 django/http/multipartparser.py | 406 | 425| 205 | 30391 | 164462 | 
| 106 | 49 django/core/files/storage.py | 233 | 294| 526 | 30917 | 164462 | 
| 107 | 50 django/db/backends/oracle/operations.py | 1 | 18| 137 | 31054 | 170390 | 
| 108 | 50 django/contrib/staticfiles/storage.py | 324 | 340| 147 | 31201 | 170390 | 
| 109 | 50 django/core/cache/backends/base.py | 89 | 100| 119 | 31320 | 170390 | 
| 110 | 50 django/utils/cache.py | 215 | 239| 235 | 31555 | 170390 | 
| 111 | 50 django/contrib/auth/hashers.py | 80 | 102| 167 | 31722 | 170390 | 
| 112 | 50 django/core/handlers/asgi.py | 175 | 191| 168 | 31890 | 170390 | 
| 113 | 50 django/core/cache/backends/base.py | 155 | 174| 195 | 32085 | 170390 | 
| 114 | 50 django/db/backends/sqlite3/base.py | 570 | 592| 143 | 32228 | 170390 | 
| 115 | 51 django/contrib/gis/db/backends/mysql/operations.py | 51 | 71| 225 | 32453 | 171241 | 
| 116 | 51 django/core/files/storage.py | 1 | 22| 158 | 32611 | 171241 | 
| 117 | 51 django/db/backends/oracle/operations.py | 21 | 73| 574 | 33185 | 171241 | 
| 118 | 52 django/contrib/gis/geos/prototypes/io.py | 288 | 340| 407 | 33592 | 174192 | 
| 119 | 53 django/db/backends/oracle/base.py | 60 | 91| 276 | 33868 | 179310 | 
| 120 | 53 django/db/models/fields/related.py | 108 | 125| 155 | 34023 | 179310 | 
| 121 | 53 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 34233 | 179310 | 
| 122 | 53 django/contrib/sessions/backends/file.py | 75 | 109| 253 | 34486 | 179310 | 
| 123 | 53 django/db/utils.py | 150 | 164| 145 | 34631 | 179310 | 
| 124 | 53 django/db/transaction.py | 285 | 310| 174 | 34805 | 179310 | 
| 125 | 54 django/contrib/gis/db/models/functions.py | 455 | 483| 225 | 35030 | 183214 | 
| 126 | 54 django/contrib/auth/hashers.py | 343 | 358| 146 | 35176 | 183214 | 
| 127 | 55 django/contrib/gis/geos/prototypes/errcheck.py | 54 | 84| 241 | 35417 | 183829 | 
| 128 | 55 django/db/backends/oracle/operations.py | 405 | 451| 480 | 35897 | 183829 | 
| 129 | 56 django/db/backends/utils.py | 1 | 45| 273 | 36170 | 185695 | 
| 130 | 57 django/db/models/fields/reverse_related.py | 156 | 181| 269 | 36439 | 187838 | 
| 131 | 58 django/core/handlers/base.py | 64 | 83| 168 | 36607 | 189016 | 
| 132 | 59 django/template/context.py | 1 | 24| 128 | 36735 | 190897 | 
| 133 | 59 django/db/models/fields/related.py | 284 | 318| 293 | 37028 | 190897 | 
| 134 | 59 django/db/models/fields/reverse_related.py | 1 | 16| 110 | 37138 | 190897 | 
| 135 | 60 django/utils/timezone.py | 1 | 101| 608 | 37746 | 192567 | 
| 136 | 60 django/contrib/auth/hashers.py | 418 | 447| 290 | 38036 | 192567 | 
| 137 | 61 django/db/backends/oracle/creation.py | 130 | 165| 399 | 38435 | 196461 | 
| 138 | 62 django/utils/deprecation.py | 73 | 95| 158 | 38593 | 197139 | 
| 139 | 62 django/db/models/fields/reverse_related.py | 136 | 154| 172 | 38765 | 197139 | 
| 140 | 62 django/db/models/fields/related.py | 156 | 169| 144 | 38909 | 197139 | 
| 141 | 63 django/contrib/gis/geoip2/base.py | 142 | 162| 258 | 39167 | 199155 | 


## Patch

```diff
diff --git a/django/core/cache/__init__.py b/django/core/cache/__init__.py
--- a/django/core/cache/__init__.py
+++ b/django/core/cache/__init__.py
@@ -12,7 +12,7 @@
 
 See docs/topics/cache.txt for information on the public API.
 """
-from threading import local
+from asgiref.local import Local
 
 from django.conf import settings
 from django.core import signals
@@ -61,7 +61,7 @@ class CacheHandler:
     Ensure only one instance of each alias exists per thread.
     """
     def __init__(self):
-        self._caches = local()
+        self._caches = Local()
 
     def __getitem__(self, alias):
         try:

```

## Test Patch

```diff
diff --git a/tests/async/tests.py b/tests/async/tests.py
--- a/tests/async/tests.py
+++ b/tests/async/tests.py
@@ -4,6 +4,7 @@
 
 from asgiref.sync import async_to_sync
 
+from django.core.cache import DEFAULT_CACHE_ALIAS, caches
 from django.core.exceptions import SynchronousOnlyOperation
 from django.test import SimpleTestCase
 from django.utils.asyncio import async_unsafe
@@ -11,6 +12,18 @@
 from .models import SimpleModel
 
 
+@skipIf(sys.platform == 'win32' and (3, 8, 0) < sys.version_info < (3, 8, 1), 'https://bugs.python.org/issue38563')
+class CacheTest(SimpleTestCase):
+    def test_caches_local(self):
+        @async_to_sync
+        async def async_cache():
+            return caches[DEFAULT_CACHE_ALIAS]
+
+        cache_1 = async_cache()
+        cache_2 = async_cache()
+        self.assertIs(cache_1, cache_2)
+
+
 @skipIf(sys.platform == 'win32' and (3, 8, 0) < sys.version_info < (3, 8, 1), 'https://bugs.python.org/issue38563')
 class DatabaseConnectionTest(SimpleTestCase):
     """A database connection cannot be used in an async context."""

```


## Code snippets

### 1 - django/core/cache/__init__.py:

Start line: 57, End line: 87

```python
class CacheHandler:
    """
    A Cache Handler to manage access to Cache instances.

    Ensure only one instance of each alias exists per thread.
    """
    def __init__(self):
        self._caches = local()

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
### 2 - django/core/cache/backends/locmem.py:

Start line: 83, End line: 124

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
### 3 - django/core/cache/backends/locmem.py:

Start line: 1, End line: 66

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
        with self._lock:
            if self._has_expired(key):
                return False
            self._expire_info[key] = self.get_backend_timeout(timeout)
            return True
```
### 4 - django/core/cache/backends/locmem.py:

Start line: 68, End line: 81

```python
class LocMemCache(BaseCache):

    def incr(self, key, delta=1, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        with self._lock:
            if self._has_expired(key):
                self._delete(key)
                raise ValueError("Key '%s' not found" % key)
            pickled = self._cache[key]
            value = pickle.loads(pickled)
            new_value = value + delta
            pickled = pickle.dumps(new_value, self.pickle_protocol)
            self._cache[key] = pickled
            self._cache.move_to_end(key, last=False)
        return new_value
```
### 5 - django/core/cache/backends/db.py:

Start line: 230, End line: 253

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def has_key(self, key, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)

        db = router.db_for_read(self.cache_model_class)
        connection = connections[db]
        quote_name = connection.ops.quote_name

        if settings.USE_TZ:
            now = datetime.utcnow()
        else:
            now = datetime.now()
        now = now.replace(microsecond=0)

        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT %s FROM %s WHERE %s = %%s and expires > %%s' % (
                    quote_name('cache_key'),
                    quote_name(self._table),
                    quote_name('cache_key'),
                ),
                [key, connection.ops.adapt_datetimefield_value(now)]
            )
            return cursor.fetchone() is not None
    # ... other code
```
### 6 - django/core/cache/__init__.py:

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
from threading import local

from django.conf import settings
from django.core import signals
from django.core.cache.backends.base import (
    BaseCache, CacheKeyWarning, InvalidCacheBackendError,
)
from django.utils.module_loading import import_string

__all__ = [
    'cache', 'caches', 'DEFAULT_CACHE_ALIAS', 'InvalidCacheBackendError',
    'CacheKeyWarning', 'BaseCache',
]

DEFAULT_CACHE_ALIAS = 'default'
```
### 7 - django/core/cache/backends/db.py:

Start line: 255, End line: 280

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def _cull(self, db, cursor, now):
        if self._cull_frequency == 0:
            self.clear()
        else:
            connection = connections[db]
            table = connection.ops.quote_name(self._table)
            cursor.execute("DELETE FROM %s WHERE expires < %%s" % table,
                           [connection.ops.adapt_datetimefield_value(now)])
            cursor.execute("SELECT COUNT(*) FROM %s" % table)
            num = cursor.fetchone()[0]
            if num > self._max_entries:
                cull_num = num // self._cull_frequency
                cursor.execute(
                    connection.ops.cache_key_culling_sql() % table,
                    [cull_num])
                cursor.execute("DELETE FROM %s "
                               "WHERE cache_key < %%s" % table,
                               [cursor.fetchone()[0]])

    def clear(self):
        db = router.db_for_write(self.cache_model_class)
        connection = connections[db]
        table = connection.ops.quote_name(self._table)
        with connection.cursor() as cursor:
            cursor.execute('DELETE FROM %s' % table)
```
### 8 - django/core/cache/backends/memcached.py:

Start line: 128, End line: 142

```python
class BaseMemcachedCache(BaseCache):

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        safe_data = {}
        original_keys = {}
        for key, value in data.items():
            safe_key = self.make_key(key, version=version)
            safe_data[safe_key] = value
            original_keys[safe_key] = key
        failed_keys = self._cache.set_multi(safe_data, self.get_backend_timeout(timeout))
        return [original_keys[k] for k in failed_keys]

    def delete_many(self, keys, version=None):
        self._cache.delete_multi(self.make_key(key, version=version) for key in keys)

    def clear(self):
        self._cache.flush_all()
```
### 9 - django/core/cache/backends/db.py:

Start line: 112, End line: 197

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def _base_set(self, mode, key, value, timeout=DEFAULT_TIMEOUT):
        timeout = self.get_backend_timeout(timeout)
        db = router.db_for_write(self.cache_model_class)
        connection = connections[db]
        quote_name = connection.ops.quote_name
        table = quote_name(self._table)

        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM %s" % table)
            num = cursor.fetchone()[0]
            now = timezone.now()
            now = now.replace(microsecond=0)
            if timeout is None:
                exp = datetime.max
            elif settings.USE_TZ:
                exp = datetime.utcfromtimestamp(timeout)
            else:
                exp = datetime.fromtimestamp(timeout)
            exp = exp.replace(microsecond=0)
            if num > self._max_entries:
                self._cull(db, cursor, now)
            pickled = pickle.dumps(value, self.pickle_protocol)
            # The DB column is expecting a string, so make sure the value is a
            # string, not bytes. Refs #19274.
            b64encoded = base64.b64encode(pickled).decode('latin1')
            try:
                # Note: typecasting for datetimes is needed by some 3rd party
                # database backends. All core backends work without typecasting,
                # so be careful about changes here - test suite will NOT pick
                # regressions.
                with transaction.atomic(using=db):
                    cursor.execute(
                        'SELECT %s, %s FROM %s WHERE %s = %%s' % (
                            quote_name('cache_key'),
                            quote_name('expires'),
                            table,
                            quote_name('cache_key'),
                        ),
                        [key]
                    )
                    result = cursor.fetchone()

                    if result:
                        current_expires = result[1]
                        expression = models.Expression(output_field=models.DateTimeField())
                        for converter in (connection.ops.get_db_converters(expression) +
                                          expression.get_db_converters(connection)):
                            current_expires = converter(current_expires, expression, connection)

                    exp = connection.ops.adapt_datetimefield_value(exp)
                    if result and mode == 'touch':
                        cursor.execute(
                            'UPDATE %s SET %s = %%s WHERE %s = %%s' % (
                                table,
                                quote_name('expires'),
                                quote_name('cache_key')
                            ),
                            [exp, key]
                        )
                    elif result and (mode == 'set' or (mode == 'add' and current_expires < now)):
                        cursor.execute(
                            'UPDATE %s SET %s = %%s, %s = %%s WHERE %s = %%s' % (
                                table,
                                quote_name('value'),
                                quote_name('expires'),
                                quote_name('cache_key'),
                            ),
                            [b64encoded, exp, key]
                        )
                    elif mode != 'touch':
                        cursor.execute(
                            'INSERT INTO %s (%s, %s, %s) VALUES (%%s, %%s, %%s)' % (
                                table,
                                quote_name('cache_key'),
                                quote_name('value'),
                                quote_name('expires'),
                            ),
                            [key, b64encoded, exp]
                        )
                    else:
                        return False  # touch failed.
            except DatabaseError:
                # To be threadsafe, updates/inserts are allowed to fail silently
                return False
            else:
                return True
    # ... other code
```
### 10 - django/core/cache/backends/db.py:

Start line: 97, End line: 110

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        self._base_set('set', key, value, timeout)

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return self._base_set('add', key, value, timeout)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        return self._base_set('touch', key, None, timeout)
    # ... other code
```
### 23 - django/core/cache/__init__.py:

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
### 26 - django/core/cache/__init__.py:

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
