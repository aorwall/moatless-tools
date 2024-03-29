# django__django-13022

| **django/django** | `f83b44075dafa429d59e8755aa47e15577cc49f9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 116 |
| **Any found context length** | 116 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/cache/backends/base.py b/django/core/cache/backends/base.py
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -287,6 +287,6 @@ def memcache_key_warnings(key):
         if ord(char) < 33 or ord(char) == 127:
             yield (
                 'Cache key contains characters that will cause errors if '
-                'used with memcached: %r' % key, CacheKeyWarning
+                'used with memcached: %r' % key
             )
             break

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/cache/backends/base.py | 290 | 290 | 1 | 1 | 116


## Problem Statement

```
Memcached key validation raises InvalidCacheKey with clunky message.
Description
	
On Django 2.2.13 the code for memcache_key_warnings in django/core/cache/backends/base.py has a bad format string that results in raising an exception rather than just producing a warning. This can be reproduced with a memcached key with a space in it, e.g. "foo bar".
This code was present before the 2.2.13 release, but becomes more exposed with that release, since it begins validating cache keys.
I think it's as simple as removing the , CacheKeyWarning.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/cache/backends/base.py** | 280 | 293| 116 | 116 | 2187 | 
| 2 | **1 django/core/cache/backends/base.py** | 1 | 51| 254 | 370 | 2187 | 
| 3 | 2 django/core/cache/backends/memcached.py | 119 | 136| 174 | 544 | 4099 | 
| 4 | 2 django/core/cache/backends/memcached.py | 100 | 117| 173 | 717 | 4099 | 
| 5 | 2 django/core/cache/backends/memcached.py | 138 | 157| 185 | 902 | 4099 | 
| 6 | 2 django/core/cache/backends/memcached.py | 67 | 98| 322 | 1224 | 4099 | 
| 7 | 2 django/core/cache/backends/memcached.py | 160 | 193| 362 | 1586 | 4099 | 
| 8 | 3 django/core/cache/backends/locmem.py | 83 | 124| 276 | 1862 | 5006 | 
| 9 | 3 django/core/cache/backends/memcached.py | 1 | 38| 260 | 2122 | 5006 | 
| 10 | 3 django/core/cache/backends/memcached.py | 196 | 216| 189 | 2311 | 5006 | 
| 11 | 3 django/core/cache/backends/locmem.py | 68 | 81| 138 | 2449 | 5006 | 
| 12 | 3 django/core/cache/backends/locmem.py | 1 | 66| 505 | 2954 | 5006 | 
| 13 | 3 django/core/cache/backends/memcached.py | 40 | 65| 283 | 3237 | 5006 | 
| 14 | 4 django/core/cache/backends/dummy.py | 1 | 40| 255 | 3492 | 5262 | 
| 15 | 5 django/templatetags/cache.py | 1 | 49| 413 | 3905 | 5989 | 
| 16 | 6 django/core/cache/backends/db.py | 230 | 253| 259 | 4164 | 8095 | 
| 17 | 7 django/core/cache/__init__.py | 1 | 29| 223 | 4387 | 8921 | 
| 18 | 7 django/core/cache/backends/db.py | 199 | 228| 285 | 4672 | 8921 | 
| 19 | 8 django/core/checks/security/base.py | 1 | 83| 732 | 5404 | 10702 | 
| 20 | 8 django/core/cache/__init__.py | 32 | 54| 196 | 5600 | 10702 | 
| 21 | 8 django/core/cache/backends/db.py | 40 | 95| 431 | 6031 | 10702 | 
| 22 | 8 django/core/cache/backends/db.py | 97 | 110| 234 | 6265 | 10702 | 
| 23 | 8 django/core/cache/backends/db.py | 112 | 197| 794 | 7059 | 10702 | 
| 24 | 9 django/contrib/messages/storage/cookie.py | 126 | 149| 234 | 7293 | 12183 | 
| 25 | **9 django/core/cache/backends/base.py** | 93 | 104| 119 | 7412 | 12183 | 
| 26 | 10 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 7579 | 12750 | 
| 27 | 11 django/core/cache/backends/filebased.py | 61 | 96| 260 | 7839 | 13923 | 
| 28 | 12 django/core/checks/caches.py | 1 | 17| 0 | 7839 | 14023 | 
| 29 | 13 django/middleware/cache.py | 1 | 52| 431 | 8270 | 15775 | 
| 30 | 14 django/views/debug.py | 136 | 148| 148 | 8418 | 20168 | 
| 31 | **14 django/core/cache/backends/base.py** | 180 | 212| 277 | 8695 | 20168 | 
| 32 | 15 django/contrib/sessions/backends/base.py | 1 | 36| 208 | 8903 | 22829 | 
| 33 | 15 django/core/cache/backends/db.py | 1 | 37| 229 | 9132 | 22829 | 
| 34 | 16 django/utils/cache.py | 1 | 35| 284 | 9416 | 26577 | 
| 35 | 17 django/core/cache/utils.py | 1 | 13| 0 | 9416 | 26656 | 
| 36 | 17 django/utils/cache.py | 324 | 342| 214 | 9630 | 26656 | 
| 37 | 17 django/core/cache/backends/filebased.py | 46 | 59| 145 | 9775 | 26656 | 
| 38 | 18 django/db/backends/mysql/validation.py | 1 | 31| 239 | 10014 | 27176 | 
| 39 | 18 django/middleware/cache.py | 166 | 204| 336 | 10350 | 27176 | 
| 40 | **18 django/core/cache/backends/base.py** | 54 | 91| 306 | 10656 | 27176 | 
| 41 | 18 django/core/cache/backends/filebased.py | 1 | 44| 284 | 10940 | 27176 | 
| 42 | **18 django/core/cache/backends/base.py** | 230 | 277| 348 | 11288 | 27176 | 
| 43 | 18 django/core/cache/backends/db.py | 255 | 280| 308 | 11596 | 27176 | 
| 44 | 19 django/core/checks/security/csrf.py | 1 | 41| 299 | 11895 | 27475 | 
| 45 | 20 django/core/checks/security/sessions.py | 1 | 98| 572 | 12467 | 28048 | 
| 46 | 20 django/core/cache/backends/filebased.py | 116 | 159| 337 | 12804 | 28048 | 
| 47 | 21 django/contrib/sites/models.py | 1 | 22| 130 | 12934 | 28836 | 
| 48 | 21 django/middleware/cache.py | 55 | 76| 216 | 13150 | 28836 | 
| 49 | 21 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 13348 | 28836 | 
| 50 | 22 django/core/checks/messages.py | 53 | 76| 161 | 13509 | 29409 | 
| 51 | 23 django/template/loaders/cached.py | 62 | 93| 225 | 13734 | 30097 | 
| 52 | 24 django/contrib/postgres/utils.py | 1 | 30| 218 | 13952 | 30315 | 
| 53 | 25 django/db/models/fields/related.py | 108 | 125| 155 | 14107 | 44146 | 
| 54 | 26 django/views/decorators/cache.py | 27 | 48| 153 | 14260 | 44509 | 
| 55 | 27 django/contrib/auth/hashers.py | 570 | 603| 242 | 14502 | 49365 | 
| 56 | 27 django/utils/cache.py | 136 | 151| 188 | 14690 | 49365 | 
| 57 | 28 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 14827 | 49502 | 
| 58 | 28 django/contrib/sessions/backends/base.py | 143 | 226| 563 | 15390 | 49502 | 
| 59 | 29 django/core/checks/urls.py | 71 | 111| 264 | 15654 | 50203 | 
| 60 | 30 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 15826 | 50624 | 
| 61 | 30 django/middleware/cache.py | 137 | 163| 252 | 16078 | 50624 | 
| 62 | 30 django/utils/cache.py | 38 | 103| 557 | 16635 | 50624 | 
| 63 | 30 django/core/cache/backends/filebased.py | 98 | 114| 174 | 16809 | 50624 | 
| 64 | 31 django/utils/deprecation.py | 1 | 33| 209 | 17018 | 51672 | 
| 65 | 32 django/core/exceptions.py | 99 | 194| 649 | 17667 | 52727 | 
| 66 | 32 django/views/debug.py | 175 | 187| 143 | 17810 | 52727 | 
| 67 | 32 django/core/checks/messages.py | 26 | 50| 259 | 18069 | 52727 | 
| 68 | 32 django/utils/cache.py | 242 | 273| 256 | 18325 | 52727 | 
| 69 | **32 django/core/cache/backends/base.py** | 106 | 142| 325 | 18650 | 52727 | 
| 70 | 32 django/utils/cache.py | 301 | 321| 217 | 18867 | 52727 | 
| 71 | 33 django/contrib/auth/password_validation.py | 91 | 115| 189 | 19056 | 54213 | 
| 72 | 34 django/db/models/base.py | 1476 | 1498| 171 | 19227 | 70309 | 
| 73 | 35 django/utils/crypto.py | 49 | 74| 212 | 19439 | 71021 | 
| 74 | 35 django/views/decorators/cache.py | 1 | 24| 209 | 19648 | 71021 | 
| 75 | 36 django/db/models/fields/mixins.py | 1 | 28| 168 | 19816 | 71364 | 
| 76 | 37 django/contrib/postgres/validators.py | 24 | 77| 370 | 20186 | 71915 | 
| 77 | 37 django/middleware/cache.py | 78 | 117| 363 | 20549 | 71915 | 
| 78 | 37 django/utils/cache.py | 345 | 364| 190 | 20739 | 71915 | 
| 79 | 38 django/contrib/postgres/forms/jsonb.py | 1 | 17| 108 | 20847 | 72023 | 
| 80 | **38 django/core/cache/backends/base.py** | 144 | 157| 122 | 20969 | 72023 | 
| 81 | 38 django/contrib/messages/storage/cookie.py | 151 | 185| 258 | 21227 | 72023 | 
| 82 | 38 django/db/models/base.py | 1255 | 1285| 259 | 21486 | 72023 | 
| 83 | 38 django/core/cache/__init__.py | 57 | 87| 178 | 21664 | 72023 | 
| 84 | 38 django/core/checks/messages.py | 1 | 24| 156 | 21820 | 72023 | 
| 85 | 39 django/db/models/query_utils.py | 25 | 54| 185 | 22005 | 74735 | 
| 86 | 39 django/contrib/auth/password_validation.py | 1 | 32| 206 | 22211 | 74735 | 
| 87 | **39 django/core/cache/backends/base.py** | 214 | 228| 147 | 22358 | 74735 | 
| 88 | 40 django/http/request.py | 1 | 41| 285 | 22643 | 79993 | 
| 89 | 40 django/contrib/sessions/backends/base.py | 124 | 141| 196 | 22839 | 79993 | 
| 90 | 40 django/db/models/fields/related.py | 255 | 282| 269 | 23108 | 79993 | 
| 91 | 40 django/utils/crypto.py | 1 | 46| 369 | 23477 | 79993 | 
| 92 | 40 django/db/models/fields/related.py | 127 | 154| 201 | 23678 | 79993 | 
| 93 | 41 django/core/mail/message.py | 169 | 177| 115 | 23793 | 83531 | 
| 94 | 41 django/db/models/base.py | 1164 | 1192| 213 | 24006 | 83531 | 
| 95 | 41 django/utils/cache.py | 367 | 413| 531 | 24537 | 83531 | 
| 96 | 42 django/core/checks/__init__.py | 1 | 26| 270 | 24807 | 83801 | 
| 97 | 43 django/db/models/fields/__init__.py | 1111 | 1149| 293 | 25100 | 101485 | 
| 98 | 43 django/contrib/auth/hashers.py | 535 | 567| 220 | 25320 | 101485 | 
| 99 | 43 django/db/models/base.py | 385 | 401| 128 | 25448 | 101485 | 
| 100 | 44 django/core/validators.py | 180 | 199| 152 | 25600 | 105731 | 
| 101 | 44 django/db/models/fields/__init__.py | 1251 | 1292| 332 | 25932 | 105731 | 
| 102 | 45 django/contrib/postgres/fields/jsonb.py | 1 | 44| 312 | 26244 | 106043 | 
| 103 | 45 django/db/models/base.py | 1147 | 1162| 138 | 26382 | 106043 | 
| 104 | 45 django/core/validators.py | 224 | 268| 330 | 26712 | 106043 | 
| 105 | 45 django/core/cache/__init__.py | 90 | 125| 229 | 26941 | 106043 | 
| 106 | 45 django/db/models/base.py | 1451 | 1474| 176 | 27117 | 106043 | 
| 107 | 45 django/contrib/sessions/backends/cache.py | 1 | 34| 213 | 27330 | 106043 | 
| 108 | 45 django/db/models/fields/__init__.py | 2098 | 2139| 325 | 27655 | 106043 | 
| 109 | 45 django/core/validators.py | 299 | 329| 227 | 27882 | 106043 | 
| 110 | 45 django/utils/cache.py | 106 | 133| 181 | 28063 | 106043 | 
| 111 | 46 django/views/generic/base.py | 1 | 30| 173 | 28236 | 107824 | 
| 112 | 46 django/core/mail/message.py | 1 | 52| 343 | 28579 | 107824 | 
| 113 | 46 django/core/checks/security/base.py | 183 | 210| 208 | 28787 | 107824 | 
| 114 | 46 django/contrib/auth/hashers.py | 191 | 219| 237 | 29024 | 107824 | 
| 115 | 46 django/db/models/base.py | 1500 | 1532| 231 | 29255 | 107824 | 
| 116 | 47 django/utils/http.py | 1 | 73| 714 | 29969 | 111995 | 
| 117 | 47 django/contrib/auth/hashers.py | 334 | 347| 120 | 30089 | 111995 | 
| 118 | 48 django/http/response.py | 1 | 26| 157 | 30246 | 116314 | 
| 119 | 49 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 30385 | 118374 | 
| 120 | 49 django/core/validators.py | 143 | 178| 387 | 30772 | 118374 | 
| 121 | 50 django/db/backends/base/validation.py | 1 | 26| 192 | 30964 | 118567 | 
| 122 | 50 django/contrib/sessions/backends/base.py | 311 | 378| 459 | 31423 | 118567 | 
| 123 | 50 django/contrib/auth/password_validation.py | 135 | 157| 197 | 31620 | 118567 | 
| 124 | 51 django/conf/urls/__init__.py | 1 | 23| 152 | 31772 | 118719 | 
| 125 | 51 django/core/validators.py | 1 | 14| 111 | 31883 | 118719 | 
| 126 | 52 django/core/signing.py | 1 | 78| 741 | 32624 | 120546 | 
| 127 | 53 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 32745 | 121402 | 
| 128 | 53 django/core/mail/message.py | 236 | 261| 322 | 33067 | 121402 | 
| 129 | 53 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 33599 | 121402 | 
| 130 | 53 django/contrib/auth/hashers.py | 1 | 27| 187 | 33786 | 121402 | 
| 131 | 53 django/db/models/fields/__init__.py | 982 | 1014| 208 | 33994 | 121402 | 
| 132 | 53 django/contrib/auth/hashers.py | 349 | 364| 146 | 34140 | 121402 | 
| 133 | 54 django/views/csrf.py | 15 | 100| 835 | 34975 | 122946 | 
| 134 | 55 django/utils/formats.py | 1 | 57| 377 | 35352 | 125052 | 
| 135 | 55 django/contrib/auth/hashers.py | 424 | 453| 290 | 35642 | 125052 | 
| 136 | 55 django/db/models/fields/related.py | 1198 | 1229| 180 | 35822 | 125052 | 
| 137 | 55 django/core/exceptions.py | 1 | 96| 405 | 36227 | 125052 | 
| 138 | 55 django/contrib/messages/storage/cookie.py | 1 | 25| 190 | 36417 | 125052 | 
| 139 | 55 django/contrib/messages/storage/cookie.py | 50 | 64| 149 | 36566 | 125052 | 
| 140 | 56 django/conf/__init__.py | 138 | 158| 167 | 36733 | 127111 | 
| 141 | 57 django/core/mail/backends/locmem.py | 1 | 31| 183 | 36916 | 127295 | 
| 142 | 57 django/middleware/cache.py | 120 | 135| 166 | 37082 | 127295 | 
| 143 | 58 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 37082 | 127365 | 
| 144 | 58 django/contrib/postgres/validators.py | 1 | 21| 181 | 37263 | 127365 | 
| 145 | 59 django/contrib/auth/validators.py | 1 | 26| 165 | 37428 | 127531 | 
| 146 | 60 django/core/checks/templates.py | 1 | 36| 259 | 37687 | 127791 | 
| 147 | 61 django/contrib/auth/middleware.py | 84 | 109| 192 | 37879 | 128785 | 
| 148 | 61 django/contrib/auth/hashers.py | 504 | 532| 222 | 38101 | 128785 | 
| 149 | 62 django/utils/html.py | 352 | 379| 212 | 38313 | 131887 | 
| 150 | 63 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 38451 | 132025 | 
| 151 | 63 django/contrib/auth/password_validation.py | 160 | 204| 353 | 38804 | 132025 | 
| 152 | 63 django/core/validators.py | 458 | 493| 249 | 39053 | 132025 | 
| 153 | 63 django/views/debug.py | 75 | 106| 267 | 39320 | 132025 | 
| 154 | 64 django/core/paginator.py | 1 | 56| 315 | 39635 | 133312 | 
| 155 | 65 django/db/backends/base/base.py | 1 | 23| 138 | 39773 | 138195 | 
| 156 | 66 django/db/utils.py | 1 | 49| 154 | 39927 | 140341 | 
| 157 | 66 django/db/models/base.py | 1881 | 1968| 619 | 40546 | 140341 | 
| 158 | 66 django/core/validators.py | 496 | 532| 219 | 40765 | 140341 | 
| 159 | 66 django/templatetags/cache.py | 52 | 94| 313 | 41078 | 140341 | 
| 160 | 67 django/forms/fields.py | 1186 | 1223| 199 | 41277 | 149733 | 
| 161 | 68 django/urls/resolvers.py | 112 | 142| 232 | 41509 | 155262 | 
| 162 | 69 django/contrib/gis/geoip2/base.py | 142 | 162| 258 | 41767 | 157278 | 
| 163 | 70 django/core/mail/backends/dummy.py | 1 | 11| 0 | 41767 | 157321 | 
| 164 | 71 django/middleware/security.py | 34 | 59| 252 | 42019 | 157865 | 
| 165 | 71 django/core/validators.py | 99 | 140| 414 | 42433 | 157865 | 
| 166 | 72 django/template/context.py | 1 | 24| 128 | 42561 | 159746 | 
| 167 | 73 django/core/checks/model_checks.py | 129 | 153| 268 | 42829 | 161531 | 
| 168 | 73 django/core/checks/urls.py | 30 | 50| 165 | 42994 | 161531 | 
| 169 | 73 django/contrib/auth/hashers.py | 606 | 643| 276 | 43270 | 161531 | 
| 170 | 74 django/contrib/staticfiles/storage.py | 323 | 339| 147 | 43417 | 165058 | 
| 171 | 74 django/db/backends/mysql/validation.py | 33 | 70| 287 | 43704 | 165058 | 
| 172 | 75 django/utils/encoding.py | 102 | 115| 130 | 43834 | 167420 | 
| 173 | 76 django/contrib/sessions/middleware.py | 1 | 79| 623 | 44457 | 168044 | 
| 174 | 76 django/core/signing.py | 146 | 182| 345 | 44802 | 168044 | 
| 175 | 76 django/core/management/commands/createcachetable.py | 1 | 29| 213 | 45015 | 168044 | 
| 176 | 77 django/core/management/templates.py | 211 | 242| 236 | 45251 | 170718 | 
| 177 | 77 django/core/checks/model_checks.py | 178 | 211| 332 | 45583 | 170718 | 
| 178 | 77 django/db/models/base.py | 1230 | 1253| 172 | 45755 | 170718 | 
| 179 | 78 django/core/checks/translation.py | 1 | 65| 445 | 46200 | 171163 | 
| 180 | 79 django/db/migrations/loader.py | 150 | 176| 291 | 46491 | 174205 | 
| 181 | 79 django/contrib/gis/geoip2/base.py | 1 | 21| 145 | 46636 | 174205 | 
| 182 | 80 django/conf/global_settings.py | 398 | 496| 787 | 47423 | 179853 | 
| 183 | 80 django/db/models/base.py | 1806 | 1879| 572 | 47995 | 179853 | 
| 184 | 81 django/core/files/storage.py | 101 | 173| 610 | 48605 | 182724 | 
| 185 | 81 django/core/files/storage.py | 1 | 22| 158 | 48763 | 182724 | 
| 186 | 81 django/core/mail/message.py | 147 | 166| 218 | 48981 | 182724 | 
| 187 | 82 django/contrib/auth/__init__.py | 1 | 58| 393 | 49374 | 184335 | 
| 188 | 82 django/core/validators.py | 332 | 377| 308 | 49682 | 184335 | 
| 189 | 82 django/utils/cache.py | 194 | 212| 184 | 49866 | 184335 | 
| 190 | 82 django/db/models/fields/related.py | 156 | 169| 144 | 50010 | 184335 | 
| 191 | 82 django/contrib/auth/hashers.py | 235 | 279| 414 | 50424 | 184335 | 
| 192 | 82 django/db/models/fields/mixins.py | 31 | 57| 173 | 50597 | 184335 | 
| 193 | 82 django/views/csrf.py | 1 | 13| 132 | 50729 | 184335 | 
| 194 | 82 django/core/validators.py | 380 | 407| 224 | 50953 | 184335 | 
| 195 | **82 django/core/cache/backends/base.py** | 159 | 178| 195 | 51148 | 184335 | 
| 196 | 82 django/views/debug.py | 150 | 173| 177 | 51325 | 184335 | 
| 197 | 82 django/utils/cache.py | 154 | 191| 447 | 51772 | 184335 | 
| 198 | 83 django/contrib/messages/middleware.py | 1 | 27| 174 | 51946 | 184510 | 
| 199 | 83 django/views/debug.py | 108 | 134| 216 | 52162 | 184510 | 
| 200 | 83 django/db/models/fields/__init__.py | 1061 | 1090| 218 | 52380 | 184510 | 
| 201 | 83 django/db/models/fields/__init__.py | 1045 | 1058| 104 | 52484 | 184510 | 
| 202 | 83 django/contrib/auth/hashers.py | 221 | 232| 150 | 52634 | 184510 | 
| 203 | 83 django/db/models/base.py | 1654 | 1702| 348 | 52982 | 184510 | 
| 204 | 83 django/conf/__init__.py | 66 | 96| 260 | 53242 | 184510 | 
| 205 | 84 django/contrib/messages/storage/base.py | 1 | 41| 264 | 53506 | 185712 | 
| 206 | 84 django/core/checks/security/base.py | 85 | 180| 710 | 54216 | 185712 | 
| 207 | 84 django/db/models/fields/__init__.py | 1587 | 1608| 183 | 54399 | 185712 | 
| 208 | 85 django/core/mail/backends/__init__.py | 1 | 2| 0 | 54399 | 185720 | 


### Hint

```
Thanks for the report, we've already noticed this, see ​PR. It raises InvalidCacheKey for me (even if a message is not perfect), e.g. File "django/core/cache/backends/memcached.py", line 157, in validate_key raise InvalidCacheKey(warning) django.core.cache.backends.base.InvalidCacheKey: ("Cache key contains characters that will cause errors if used with memcached: '38_postgis_pull-requests-bionic_memcache:1:key with spaces and 清'", <class 'django.core.cache.backends.base.CacheKeyWarning'>)
... results in raising an exception rather than just producing a warning. The expected behaviour is to raise the InvalidCacheKey for memcached backends, addressing CVE-2020-13254. Tim, can you post the traceback? (How did you hit this?)
Ah, this was actually against the dummy backend. It was caught in tests: ​https://build.testeng.edx.org/job/edx-platform-python-pipeline-pr/18276/testReport/junit/common.djangoapps.xblock_django.tests.test_api/XBlockSupportTestCase/Run_Tests___lms_unit___test_disabled_blocks/ self = <xblock_django.tests.test_api.XBlockSupportTestCase testMethod=test_disabled_blocks> def setUp(self): super(XBlockSupportTestCase, self).setUp() # Set up XBlockConfigurations for disabled and deprecated states block_config = [ ("poll", True, True), ("survey", False, True), ("done", True, False), ] for name, enabled, deprecated in block_config: > XBlockConfiguration(name=name, enabled=enabled, deprecated=deprecated).save() common/djangoapps/xblock_django/tests/test_api.py:28: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/config_models/models.py:110: in save update_fields ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/db/models/base.py:741: in save force_update=force_update, update_fields=update_fields) ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/db/models/base.py:790: in save_base update_fields=update_fields, raw=raw, using=using, ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/dispatch/dispatcher.py:175: in send for receiver in self._live_receivers(sender) ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/dispatch/dispatcher.py:175: in <listcomp> for receiver in self._live_receivers(sender) openedx/core/lib/cache_utils.py:187: in invalidate TieredCache.delete_all_tiers(key) ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/edx_django_utils/cache/utils.py:226: in delete_all_tiers django_cache.delete(key) ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/core/cache/backends/dummy.py:30: in delete self.validate_key(key) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ self = <django.core.cache.backends.dummy.DummyCache object at 0x7fc3211f7d30> key = ":1:<class 'xblock_django.models.XBlockConfiguration'>.xblock_django.api.deprecated_xblocks" def validate_key(self, key): """ Warn about keys that would not be portable to the memcached backend. This encourages (but does not force) writing backend-portable cache code. """ for warning in memcache_key_warnings(key): > warnings.warn(warning, CacheKeyWarning) E TypeError: 'type' object cannot be interpreted as an integer ../edx-venv-3.5/edx-venv/lib/python3.5/site-packages/django/core/cache/backends/base.py:250: TypeError This error only started happening in tests with the Django 2.2.12 -> 2.2.13 upgrade. I don't understand why we wouldn't have seen the InvalidCacheKey error in production, where we use memcached. (We were in fact creating an invalid cache key, and I have a patch ready for that on our side that unblocks the upgrade, so I'm not sure how much we'll end up digging into that mystery!)
Thanks for the follow up Tim. I still don't get the error using 3.5... Python 3.5.9 (default, Jun 4 2020, 16:47:18) [GCC 4.2.1 Compatible Apple LLVM 10.0.1 (clang-1001.0.46.4)] on darwin Type "help", "copyright", "credits" or "license" for more information. >>> from django.core.cache.backends.base import CacheKeyWarning >>> import warnings >>> a_tuple = ('A string message, plus...', CacheKeyWarning) >>> warnings.warn(a_tuple, CacheKeyWarning) __main__:1: CacheKeyWarning: ('A string message, plus...', <class 'django.core.cache.backends.base.CacheKeyWarning'>) There must be something else going on. Maybe a custom warning formatting? (How else is the TypeError: 'type' object cannot be interpreted as an integer — the warning class being converted to an int? 🤔) ​The EdX change will stop the warning coming up, and is exactly the case that the patch was meant to catch. Treating CacheKeyWarnings as errors would be good — I wonder how many folks would be seeing those and ignoring them... 😬 I think there's enough of a regression here to backport the fix.
I don't understand it either! I'll take a little more time to poke at it today. (For anyone who wants to reproduce this strange test error in place: Python 3.5 virtualenv with pip 20.0.2, make requirements, pytest -s common/djangoapps/xblock_django/tests/test_api.py::XBlockSupportTestCase::test_authorable_blocks -- on commit 9f53525c.) The farthest I've gotten is that it's definitely coming from inside warnings.warn(...): (Pdb) warnings.warn(("hi", str), RuntimeWarning) *** TypeError: 'type' object cannot be interpreted as an integer Why it doesn't happen on a fresh REPL with all dependencies loaded, I'm not sure. My best guess is that it has something to do with warning filtering, since that's configurable and appears to be what's inspecting the warning message. Probably not specifically relevant to Django or the edX code, though. (Edit: Specified pip version.)
Out of curiosity, do you get the same crash if passing category as a keyword argument? warnings.warn(warning, category=CacheKeyWarning)
I can reproduce this error when unpacking tuple >>> warnings.warn(*("hi", str), RuntimeWarning) Traceback (most recent call last): File "<stdin>", line 1, in <module> TypeError: 'type' object cannot be interpreted as an integer because RuntimeWarning is passed as stacklevel, it must be somewhere in EdX. Nevertheless we will fix this message.
```

## Patch

```diff
diff --git a/django/core/cache/backends/base.py b/django/core/cache/backends/base.py
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -287,6 +287,6 @@ def memcache_key_warnings(key):
         if ord(char) < 33 or ord(char) == 127:
             yield (
                 'Cache key contains characters that will cause errors if '
-                'used with memcached: %r' % key, CacheKeyWarning
+                'used with memcached: %r' % key
             )
             break

```

## Test Patch

```diff
diff --git a/tests/cache/tests.py b/tests/cache/tests.py
--- a/tests/cache/tests.py
+++ b/tests/cache/tests.py
@@ -637,8 +637,9 @@ def func(key, *args):
         cache.key_func = func
 
         try:
-            with self.assertWarnsMessage(CacheKeyWarning, expected_warning):
+            with self.assertWarns(CacheKeyWarning) as cm:
                 cache.set(key, 'value')
+            self.assertEqual(str(cm.warning), expected_warning)
         finally:
             cache.key_func = old_func
 
@@ -1276,8 +1277,9 @@ def _perform_invalid_key_test(self, key, expected_warning):
         key.
         """
         msg = expected_warning.replace(key, cache.make_key(key))
-        with self.assertRaisesMessage(InvalidCacheKey, msg):
+        with self.assertRaises(InvalidCacheKey) as cm:
             cache.set(key, 'value')
+        self.assertEqual(str(cm.exception), msg)
 
     def test_default_never_expiring_timeout(self):
         # Regression test for #22845

```


## Code snippets

### 1 - django/core/cache/backends/base.py:

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
                'used with memcached: %r' % key, CacheKeyWarning
            )
            break
```
### 2 - django/core/cache/backends/base.py:

Start line: 1, End line: 51

```python
"Base Cache class."
import time
import warnings

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string


class InvalidCacheBackendError(ImproperlyConfigured):
    pass


class CacheKeyWarning(RuntimeWarning):
    pass


class InvalidCacheKey(ValueError):
    pass


# Stub class to ensure not passing in a `timeout` argument results in
# the default timeout
DEFAULT_TIMEOUT = object()

# Memcached does not accept keys longer than this.
MEMCACHE_MAX_KEY_LENGTH = 250


def default_key_func(key, key_prefix, version):
    """
    Default function to generate keys.

    Construct the key used by all other methods. By default, prepend
    the `key_prefix'. KEY_FUNCTION can be used to specify an alternate
    function with custom key making behavior.
    """
    return '%s:%s:%s' % (key_prefix, version, key)


def get_key_func(key_func):
    """
    Function to decide which key function to use.

    Default to ``default_key_func``.
    """
    if key_func is not None:
        if callable(key_func):
            return key_func
        else:
            return import_string(key_func)
    return default_key_func
```
### 3 - django/core/cache/backends/memcached.py:

Start line: 119, End line: 136

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

        # python-memcache responds to incr on nonexistent keys by
        # raising a ValueError, pylibmc by raising a pylibmc.NotFound
        # and Cmemcache returns None. In all cases,
        # we should raise a ValueError though.
        except self.LibraryValueNotFoundException:
            val = None
        if val is None:
            raise ValueError("Key '%s' not found" % key)
        return val
```
### 4 - django/core/cache/backends/memcached.py:

Start line: 100, End line: 117

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

        # python-memcache responds to incr on nonexistent keys by
        # raising a ValueError, pylibmc by raising a pylibmc.NotFound
        # and Cmemcache returns None. In all cases,
        # we should raise a ValueError though.
        except self.LibraryValueNotFoundException:
            val = None
        if val is None:
            raise ValueError("Key '%s' not found" % key)
        return val
```
### 5 - django/core/cache/backends/memcached.py:

Start line: 138, End line: 157

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
        self._cache.delete_multi(self.make_key(key, version=version) for key in keys)

    def clear(self):
        self._cache.flush_all()

    def validate_key(self, key):
        for warning in memcache_key_warnings(key):
            raise InvalidCacheKey(warning)
```
### 6 - django/core/cache/backends/memcached.py:

Start line: 67, End line: 98

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
### 7 - django/core/cache/backends/memcached.py:

Start line: 160, End line: 193

```python
class MemcachedCache(BaseMemcachedCache):
    "An implementation of a cache binding using python-memcached"
    def __init__(self, server, params):
        import memcache
        super().__init__(server, params, library=memcache, value_not_found_exception=ValueError)

    @property
    def _cache(self):
        if getattr(self, '_client', None) is None:
            client_kwargs = {'pickleProtocol': pickle.HIGHEST_PROTOCOL}
            client_kwargs.update(self._options)
            self._client = self._lib.Client(self._servers, **client_kwargs)
        return self._client

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        return self._cache.touch(key, self.get_backend_timeout(timeout)) != 0

    def get(self, key, default=None, version=None):
        key = self.make_key(key, version=version)
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
        return bool(self._cache._deletetouch([b'DELETED'], 'delete', key))
```
### 8 - django/core/cache/backends/locmem.py:

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
### 9 - django/core/cache/backends/memcached.py:

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

        # The exception type to catch from the underlying library for a key
        # that was not found. This is a ValueError for python-memcache,
        # pylibmc.NotFound for pylibmc, and cmemcache will return None without
        # raising an exception.
        self.LibraryValueNotFoundException = value_not_found_exception

        self._lib = library
        self._options = params.get('OPTIONS') or {}

    @property
    def _cache(self):
        """
        Implement transparent thread-safe access to a memcached client.
        """
        if getattr(self, '_client', None) is None:
            self._client = self._lib.Client(self._servers, **self._options)

        return self._client
```
### 10 - django/core/cache/backends/memcached.py:

Start line: 196, End line: 216

```python
class PyLibMCCache(BaseMemcachedCache):
    "An implementation of a cache binding using pylibmc"
    def __init__(self, server, params):
        import pylibmc
        super().__init__(server, params, library=pylibmc, value_not_found_exception=pylibmc.NotFound)

    @cached_property
    def _cache(self):
        return self._lib.Client(self._servers, **self._options)

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        if timeout == 0:
            return self._cache.delete(key)
        return self._cache.touch(key, self.get_backend_timeout(timeout))

    def close(self, **kwargs):
        # libmemcached manages its own connections. Don't call disconnect_all()
        # as it resets the failover state and creates unnecessary reconnects.
        pass
```
### 25 - django/core/cache/backends/base.py:

Start line: 93, End line: 104

```python
class BaseCache:

    def make_key(self, key, version=None):
        """
        Construct the key used by all other methods. By default, use the
        key_func to generate a key (which, by default, prepends the
        `key_prefix' and 'version'). A different key function can be provided
        at the time of cache construction; alternatively, you can subclass the
        cache backend to provide custom key making behavior.
        """
        if version is None:
            version = self.version

        return self.key_func(key, self.key_prefix, version)
```
### 31 - django/core/cache/backends/base.py:

Start line: 180, End line: 212

```python
class BaseCache:

    def has_key(self, key, version=None):
        """
        Return True if the key is in the cache and has not expired.
        """
        return self.get(key, version=version) is not None

    def incr(self, key, delta=1, version=None):
        """
        Add delta to value in the cache. If the key does not exist, raise a
        ValueError exception.
        """
        value = self.get(key, version=version)
        if value is None:
            raise ValueError("Key '%s' not found" % key)
        new_value = value + delta
        self.set(key, new_value, version=version)
        return new_value

    def decr(self, key, delta=1, version=None):
        """
        Subtract delta from value in the cache. If the key does not exist, raise
        a ValueError exception.
        """
        return self.incr(key, -delta, version=version)

    def __contains__(self, key):
        """
        Return True if the key is in the cache and has not expired.
        """
        # This is a separate method, rather than just a copy of has_key(),
        # so that it always has the same functionality as has_key(), even
        # if a subclass overrides it.
        return self.has_key(key)
```
### 40 - django/core/cache/backends/base.py:

Start line: 54, End line: 91

```python
class BaseCache:
    def __init__(self, params):
        timeout = params.get('timeout', params.get('TIMEOUT', 300))
        if timeout is not None:
            try:
                timeout = int(timeout)
            except (ValueError, TypeError):
                timeout = 300
        self.default_timeout = timeout

        options = params.get('OPTIONS', {})
        max_entries = params.get('max_entries', options.get('MAX_ENTRIES', 300))
        try:
            self._max_entries = int(max_entries)
        except (ValueError, TypeError):
            self._max_entries = 300

        cull_frequency = params.get('cull_frequency', options.get('CULL_FREQUENCY', 3))
        try:
            self._cull_frequency = int(cull_frequency)
        except (ValueError, TypeError):
            self._cull_frequency = 3

        self.key_prefix = params.get('KEY_PREFIX', '')
        self.version = params.get('VERSION', 1)
        self.key_func = get_key_func(params.get('KEY_FUNCTION'))

    def get_backend_timeout(self, timeout=DEFAULT_TIMEOUT):
        """
        Return the timeout value usable by this backend based upon the provided
        timeout.
        """
        if timeout == DEFAULT_TIMEOUT:
            timeout = self.default_timeout
        elif timeout == 0:
            # ticket 21147 - avoid time.time() related precision issues
            timeout = -1
        return None if timeout is None else time.time() + timeout
```
### 42 - django/core/cache/backends/base.py:

Start line: 230, End line: 277

```python
class BaseCache:

    def delete_many(self, keys, version=None):
        """
        Delete a bunch of values in the cache at once. For certain backends
        (memcached), this is much more efficient than calling delete() multiple
        times.
        """
        for key in keys:
            self.delete(key, version=version)

    def clear(self):
        """Remove *all* values from the cache at once."""
        raise NotImplementedError('subclasses of BaseCache must provide a clear() method')

    def validate_key(self, key):
        """
        Warn about keys that would not be portable to the memcached
        backend. This encourages (but does not force) writing backend-portable
        cache code.
        """
        for warning in memcache_key_warnings(key):
            warnings.warn(warning, CacheKeyWarning)

    def incr_version(self, key, delta=1, version=None):
        """
        Add delta to the cache version for the supplied key. Return the new
        version.
        """
        if version is None:
            version = self.version

        value = self.get(key, version=version)
        if value is None:
            raise ValueError("Key '%s' not found" % key)

        self.set(key, value, version=version + delta)
        self.delete(key, version=version)
        return version + delta

    def decr_version(self, key, delta=1, version=None):
        """
        Subtract delta from the cache version for the supplied key. Return the
        new version.
        """
        return self.incr_version(key, -delta, version)

    def close(self, **kwargs):
        """Close the cache connection"""
        pass
```
### 69 - django/core/cache/backends/base.py:

Start line: 106, End line: 142

```python
class BaseCache:

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        """
        Set a value in the cache if the key does not already exist. If
        timeout is given, use that timeout for the key; otherwise use the
        default cache timeout.

        Return True if the value was stored, False otherwise.
        """
        raise NotImplementedError('subclasses of BaseCache must provide an add() method')

    def get(self, key, default=None, version=None):
        """
        Fetch a given key from the cache. If the key does not exist, return
        default, which itself defaults to None.
        """
        raise NotImplementedError('subclasses of BaseCache must provide a get() method')

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        """
        Set a value in the cache. If timeout is given, use that timeout for the
        key; otherwise use the default cache timeout.
        """
        raise NotImplementedError('subclasses of BaseCache must provide a set() method')

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        """
        Update the key's expiry time using timeout. Return True if successful
        or False if the key does not exist.
        """
        raise NotImplementedError('subclasses of BaseCache must provide a touch() method')

    def delete(self, key, version=None):
        """
        Delete a key from the cache and return whether it succeeded, failing
        silently.
        """
        raise NotImplementedError('subclasses of BaseCache must provide a delete() method')
```
### 80 - django/core/cache/backends/base.py:

Start line: 144, End line: 157

```python
class BaseCache:

    def get_many(self, keys, version=None):
        """
        Fetch a bunch of keys from the cache. For certain backends (memcached,
        pgsql) this can be *much* faster when fetching multiple values.

        Return a dict mapping each key in keys to its value. If the given
        key is missing, it will be missing from the response dict.
        """
        d = {}
        for k in keys:
            val = self.get(k, version=version)
            if val is not None:
                d[k] = val
        return d
```
### 87 - django/core/cache/backends/base.py:

Start line: 214, End line: 228

```python
class BaseCache:

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        """
        Set a bunch of values in the cache at once from a dict of key/value
        pairs.  For certain backends (memcached), this is much more efficient
        than calling set() multiple times.

        If timeout is given, use that timeout for the key; otherwise use the
        default cache timeout.

        On backends that support it, return a list of keys that failed
        insertion, or an empty list if all keys were inserted successfully.
        """
        for key, value in data.items():
            self.set(key, value, timeout=timeout, version=version)
        return []
```
### 195 - django/core/cache/backends/base.py:

Start line: 159, End line: 178

```python
class BaseCache:

    def get_or_set(self, key, default, timeout=DEFAULT_TIMEOUT, version=None):
        """
        Fetch a given key from the cache. If the key does not exist,
        add the key and set it to the default value. The default value can
        also be any callable. If timeout is given, use that timeout for the
        key; otherwise use the default cache timeout.

        Return the value of the key stored or retrieved.
        """
        val = self.get(key, version=version)
        if val is None:
            if callable(default):
                default = default()
            if default is not None:
                self.add(key, default, timeout=timeout, version=version)
                # Fetch the value again to avoid a race condition if another
                # caller added a value between the first get() and the add()
                # above.
                return self.get(key, default, version=version)
        return val
```
