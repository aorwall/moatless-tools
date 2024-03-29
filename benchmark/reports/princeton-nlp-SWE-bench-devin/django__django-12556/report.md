# django__django-12556

| **django/django** | `5cc2c63f902412cdd9a8ebbabbd953aa8e2180c0` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 3086 |
| **Any found context length** | 229 |
| **Avg pos** | 4.333333333333333 |
| **Min pos** | 1 |
| **Max pos** | 12 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/contrib/auth/hashers.py b/django/contrib/auth/hashers.py
--- a/django/contrib/auth/hashers.py
+++ b/django/contrib/auth/hashers.py
@@ -185,7 +185,8 @@ def _load_library(self):
 
     def salt(self):
         """Generate a cryptographically secure nonce salt in ASCII."""
-        return get_random_string()
+        # 12 returns a 71-bit value, log_2((26+26+10)^12) =~ 71 bits
+        return get_random_string(12)
 
     def verify(self, password, encoded):
         """Check if the given password is correct."""
diff --git a/django/db/backends/oracle/creation.py b/django/db/backends/oracle/creation.py
--- a/django/db/backends/oracle/creation.py
+++ b/django/db/backends/oracle/creation.py
@@ -341,7 +341,7 @@ def _test_database_passwd(self):
         password = self._test_settings_get('PASSWORD')
         if password is None and self._test_user_create():
             # Oracle passwords are limited to 30 chars and can't contain symbols.
-            password = get_random_string(length=30)
+            password = get_random_string(30)
         return password
 
     def _test_database_tblspace(self):
diff --git a/django/utils/crypto.py b/django/utils/crypto.py
--- a/django/utils/crypto.py
+++ b/django/utils/crypto.py
@@ -4,8 +4,10 @@
 import hashlib
 import hmac
 import secrets
+import warnings
 
 from django.conf import settings
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.encoding import force_bytes
 
 
@@ -44,15 +46,31 @@ def salted_hmac(key_salt, value, secret=None, *, algorithm='sha1'):
     return hmac.new(key, msg=force_bytes(value), digestmod=hasher)
 
 
-def get_random_string(length=12,
-                      allowed_chars='abcdefghijklmnopqrstuvwxyz'
-                                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
+NOT_PROVIDED = object()  # RemovedInDjango40Warning.
+
+
+# RemovedInDjango40Warning: when the deprecation ends, replace with:
+#   def get_random_string(self, length, allowed_chars='...'):
+def get_random_string(length=NOT_PROVIDED, allowed_chars=(
+    'abcdefghijklmnopqrstuvwxyz'
+    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
+)):
     """
     Return a securely generated random string.
 
-    The default length of 12 with the a-z, A-Z, 0-9 character set returns
-    a 71-bit value. log_2((26+26+10)^12) =~ 71 bits
+    The bit length of the returned value can be calculated with the formula:
+        log_2(len(allowed_chars)^length)
+
+    For example, with default `allowed_chars` (26+26+10), this gives:
+      * length: 12, bit length =~ 71 bits
+      * length: 22, bit length =~ 131 bits
     """
+    if length is NOT_PROVIDED:
+        warnings.warn(
+            'Not providing a length argument is deprecated.',
+            RemovedInDjango40Warning,
+        )
+        length = 12
     return ''.join(secrets.choice(allowed_chars) for i in range(length))
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/hashers.py | 188 | 188 | - | 5 | -
| django/db/backends/oracle/creation.py | 344 | 344 | - | - | -
| django/utils/crypto.py | 7 | 7 | 12 | 1 | 3086
| django/utils/crypto.py | 47 | 54 | 1 | 1 | 229


## Problem Statement

```
Deprecate using get_random_string without an explicit length
Description
	
django.utils.crypto.get_random_string currently has a default length value (12). I think we should force callers to specify the length value and not count on a default.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/utils/crypto.py** | 47 | 72| 229 | 229 | 582 | 
| 2 | 2 django/contrib/auth/password_validation.py | 91 | 115| 189 | 418 | 2068 | 
| 3 | 3 django/db/models/fields/__init__.py | 980 | 1012| 208 | 626 | 19655 | 
| 4 | 4 django/core/checks/security/base.py | 1 | 83| 732 | 1358 | 21436 | 
| 5 | **5 django/contrib/auth/hashers.py** | 1 | 27| 187 | 1545 | 26223 | 
| 6 | **5 django/contrib/auth/hashers.py** | 64 | 77| 147 | 1692 | 26223 | 
| 7 | 6 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 1830 | 26361 | 
| 8 | 7 django/utils/text.py | 104 | 122| 165 | 1995 | 29806 | 
| 9 | **7 django/contrib/auth/hashers.py** | 529 | 561| 220 | 2215 | 29806 | 
| 10 | **7 django/contrib/auth/hashers.py** | 564 | 597| 242 | 2457 | 29806 | 
| 11 | **7 django/contrib/auth/hashers.py** | 600 | 637| 276 | 2733 | 29806 | 
| **-> 12 <-** | **7 django/utils/crypto.py** | 1 | 44| 353 | 3086 | 29806 | 
| 13 | **7 django/contrib/auth/hashers.py** | 418 | 447| 290 | 3376 | 29806 | 
| 14 | 8 django/db/models/functions/text.py | 159 | 202| 312 | 3688 | 32261 | 
| 15 | **8 django/contrib/auth/hashers.py** | 406 | 416| 127 | 3815 | 32261 | 
| 16 | 9 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 3815 | 32339 | 
| 17 | 10 django/db/backends/utils.py | 213 | 250| 279 | 4094 | 34205 | 
| 18 | **10 django/contrib/auth/hashers.py** | 343 | 358| 146 | 4240 | 34205 | 
| 19 | 11 django/middleware/csrf.py | 45 | 54| 111 | 4351 | 37091 | 
| 20 | 12 django/contrib/auth/tokens.py | 58 | 71| 180 | 4531 | 37903 | 
| 21 | 13 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 4531 | 37971 | 
| 22 | 14 django/core/management/utils.py | 77 | 109| 222 | 4753 | 39085 | 
| 23 | 15 django/forms/fields.py | 208 | 239| 274 | 5027 | 48098 | 
| 24 | 16 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 5027 | 48175 | 
| 25 | **16 django/contrib/auth/hashers.py** | 467 | 495| 220 | 5247 | 48175 | 
| 26 | 16 django/middleware/csrf.py | 93 | 119| 224 | 5471 | 48175 | 
| 27 | 16 django/db/models/fields/__init__.py | 1439 | 1461| 119 | 5590 | 48175 | 
| 28 | 17 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 5715 | 48424 | 
| 29 | 18 django/db/models/base.py | 1765 | 1836| 565 | 6280 | 63770 | 
| 30 | **18 django/contrib/auth/hashers.py** | 215 | 226| 150 | 6430 | 63770 | 
| 31 | **18 django/contrib/auth/hashers.py** | 498 | 526| 222 | 6652 | 63770 | 
| 32 | **18 django/contrib/auth/hashers.py** | 80 | 102| 167 | 6819 | 63770 | 
| 33 | **18 django/contrib/auth/hashers.py** | 328 | 341| 120 | 6939 | 63770 | 
| 34 | 19 django/middleware/security.py | 34 | 59| 252 | 7191 | 64314 | 
| 35 | 20 django/db/backends/mysql/validation.py | 25 | 57| 246 | 7437 | 64764 | 
| 36 | 21 django/utils/lorem_ipsum.py | 49 | 68| 229 | 7666 | 66264 | 
| 37 | 21 django/forms/fields.py | 1184 | 1214| 182 | 7848 | 66264 | 
| 38 | 21 django/middleware/csrf.py | 57 | 71| 156 | 8004 | 66264 | 
| 39 | 21 django/utils/lorem_ipsum.py | 1 | 15| 144 | 8148 | 66264 | 
| 40 | 21 django/db/models/fields/__init__.py | 1390 | 1413| 171 | 8319 | 66264 | 
| 41 | 22 django/contrib/postgres/validators.py | 1 | 21| 181 | 8500 | 66815 | 
| 42 | 23 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 8500 | 66893 | 
| 43 | **23 django/contrib/auth/hashers.py** | 229 | 273| 414 | 8914 | 66893 | 
| 44 | 23 django/db/models/fields/__init__.py | 1713 | 1736| 146 | 9060 | 66893 | 
| 45 | 23 django/db/models/fields/__init__.py | 1014 | 1040| 229 | 9289 | 66893 | 
| 46 | 23 django/contrib/auth/password_validation.py | 1 | 32| 206 | 9495 | 66893 | 
| 47 | 23 django/db/models/fields/__init__.py | 1415 | 1437| 121 | 9616 | 66893 | 
| 48 | 24 django/conf/__init__.py | 138 | 158| 167 | 9783 | 68952 | 
| 49 | 24 django/middleware/csrf.py | 74 | 90| 195 | 9978 | 68952 | 
| 50 | 25 django/core/checks/security/csrf.py | 1 | 41| 299 | 10277 | 69251 | 
| 51 | 26 django/core/cache/backends/memcached.py | 38 | 63| 283 | 10560 | 71068 | 
| 52 | 26 django/db/models/fields/__init__.py | 2289 | 2339| 339 | 10899 | 71068 | 
| 53 | 26 django/utils/lorem_ipsum.py | 71 | 94| 163 | 11062 | 71068 | 
| 54 | 27 django/core/cache/backends/base.py | 1 | 47| 245 | 11307 | 73225 | 
| 55 | 27 django/db/models/fields/__init__.py | 2004 | 2034| 252 | 11559 | 73225 | 
| 56 | 28 django/conf/global_settings.py | 147 | 262| 859 | 12418 | 78865 | 
| 57 | **28 django/contrib/auth/hashers.py** | 388 | 404| 127 | 12545 | 78865 | 
| 58 | 29 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 12695 | 79015 | 
| 59 | 29 django/middleware/csrf.py | 1 | 42| 330 | 13025 | 79015 | 
| 60 | 29 django/db/models/fields/__init__.py | 1815 | 1843| 191 | 13216 | 79015 | 
| 61 | 30 django/contrib/messages/storage/cookie.py | 126 | 149| 234 | 13450 | 80496 | 
| 62 | 31 django/contrib/sessions/backends/base.py | 228 | 251| 193 | 13643 | 83157 | 
| 63 | 31 django/db/models/fields/__init__.py | 1043 | 1056| 104 | 13747 | 83157 | 
| 64 | 32 django/utils/encoding.py | 102 | 115| 130 | 13877 | 85519 | 
| 65 | 32 django/utils/text.py | 124 | 146| 187 | 14064 | 85519 | 
| 66 | 32 django/contrib/sessions/backends/base.py | 1 | 36| 208 | 14272 | 85519 | 
| 67 | 32 django/utils/encoding.py | 48 | 67| 156 | 14428 | 85519 | 
| 68 | 33 django/core/validators.py | 330 | 375| 308 | 14736 | 89750 | 
| 69 | 33 django/conf/global_settings.py | 397 | 495| 787 | 15523 | 89750 | 
| 70 | 34 django/contrib/gis/db/models/functions.py | 343 | 372| 353 | 15876 | 93672 | 
| 71 | 34 django/db/models/fields/__init__.py | 1585 | 1606| 183 | 16059 | 93672 | 
| 72 | 34 django/core/validators.py | 378 | 405| 224 | 16283 | 93672 | 
| 73 | **34 django/contrib/auth/hashers.py** | 105 | 125| 139 | 16422 | 93672 | 
| 74 | **34 django/contrib/auth/hashers.py** | 128 | 146| 189 | 16611 | 93672 | 
| 75 | 35 django/contrib/sites/models.py | 1 | 22| 130 | 16741 | 94460 | 
| 76 | 36 django/core/management/commands/startproject.py | 1 | 21| 137 | 16878 | 94597 | 
| 77 | **36 django/contrib/auth/hashers.py** | 450 | 464| 126 | 17004 | 94597 | 
| 78 | 36 django/core/cache/backends/base.py | 239 | 256| 165 | 17169 | 94597 | 
| 79 | 36 django/db/models/fields/__init__.py | 1463 | 1522| 398 | 17567 | 94597 | 
| 80 | 37 django/contrib/auth/validators.py | 1 | 26| 165 | 17732 | 94763 | 
| 81 | 37 django/utils/text.py | 81 | 102| 202 | 17934 | 94763 | 
| 82 | 37 django/db/models/fields/__init__.py | 1846 | 1923| 567 | 18501 | 94763 | 
| 83 | 38 django/core/cache/backends/filebased.py | 98 | 114| 174 | 18675 | 95936 | 
| 84 | 38 django/db/models/fields/__init__.py | 2202 | 2222| 163 | 18838 | 95936 | 
| 85 | 38 django/middleware/security.py | 1 | 32| 316 | 19154 | 95936 | 
| 86 | 38 django/conf/global_settings.py | 346 | 396| 785 | 19939 | 95936 | 
| 87 | 39 django/core/checks/security/sessions.py | 1 | 98| 572 | 20511 | 96509 | 
| 88 | 39 django/contrib/auth/password_validation.py | 160 | 204| 353 | 20864 | 96509 | 
| 89 | 39 django/db/models/fields/__init__.py | 1249 | 1290| 332 | 21196 | 96509 | 
| 90 | 39 django/utils/encoding.py | 1 | 45| 290 | 21486 | 96509 | 
| 91 | 40 django/utils/deprecation.py | 73 | 106| 256 | 21742 | 97284 | 
| 92 | 40 django/core/cache/backends/memcached.py | 110 | 126| 168 | 21910 | 97284 | 
| 93 | 41 django/contrib/auth/__init__.py | 1 | 58| 393 | 22303 | 98856 | 
| 94 | 42 django/views/defaults.py | 100 | 119| 149 | 22452 | 99898 | 
| 95 | 42 django/utils/text.py | 148 | 218| 552 | 23004 | 99898 | 
| 96 | 43 django/contrib/auth/base_user.py | 1 | 44| 298 | 23302 | 100782 | 
| 97 | 44 django/template/defaultfilters.py | 805 | 849| 378 | 23680 | 106856 | 
| 98 | **44 django/contrib/auth/hashers.py** | 149 | 184| 245 | 23925 | 106856 | 
| 99 | 44 django/db/models/base.py | 1 | 50| 327 | 24252 | 106856 | 
| 100 | 45 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 24389 | 106993 | 
| 101 | 46 django/views/csrf.py | 15 | 100| 835 | 25224 | 108537 | 
| 102 | 46 django/forms/fields.py | 480 | 505| 174 | 25398 | 108537 | 
| 103 | 47 django/http/request.py | 396 | 419| 185 | 25583 | 113788 | 
| 104 | 47 django/core/validators.py | 1 | 14| 111 | 25694 | 113788 | 
| 105 | 48 django/core/files/storage.py | 54 | 69| 138 | 25832 | 116659 | 
| 106 | **48 django/contrib/auth/hashers.py** | 276 | 326| 391 | 26223 | 116659 | 
| 107 | 49 django/core/signing.py | 1 | 78| 732 | 26955 | 118366 | 
| 108 | 49 django/core/validators.py | 141 | 176| 387 | 27342 | 118366 | 
| 109 | 49 django/forms/fields.py | 242 | 259| 164 | 27506 | 118366 | 
| 110 | 49 django/utils/lorem_ipsum.py | 97 | 115| 148 | 27654 | 118366 | 
| 111 | 49 django/db/models/fields/__init__.py | 1738 | 1765| 215 | 27869 | 118366 | 
| 112 | 49 django/utils/deprecation.py | 1 | 27| 181 | 28050 | 118366 | 
| 113 | 49 django/utils/text.py | 277 | 318| 289 | 28339 | 118366 | 
| 114 | 49 django/core/checks/security/base.py | 213 | 226| 131 | 28470 | 118366 | 
| 115 | 49 django/core/checks/security/base.py | 183 | 210| 208 | 28678 | 118366 | 
| 116 | 49 django/forms/fields.py | 1169 | 1181| 113 | 28791 | 118366 | 
| 117 | 49 django/core/cache/backends/memcached.py | 92 | 108| 167 | 28958 | 118366 | 
| 118 | 50 django/contrib/postgres/forms/array.py | 1 | 37| 258 | 29216 | 119960 | 
| 119 | 50 django/http/request.py | 1 | 41| 285 | 29501 | 119960 | 
| 120 | 50 django/contrib/sessions/backends/base.py | 143 | 226| 563 | 30064 | 119960 | 
| 121 | 51 django/utils/http.py | 1 | 73| 714 | 30778 | 124138 | 
| 122 | 52 django/db/backends/postgresql/operations.py | 209 | 293| 674 | 31452 | 126807 | 
| 123 | 52 django/core/validators.py | 494 | 530| 219 | 31671 | 126807 | 
| 124 | 52 django/contrib/auth/tokens.py | 1 | 56| 365 | 32036 | 126807 | 
| 125 | 52 django/db/models/fields/__init__.py | 1767 | 1812| 279 | 32315 | 126807 | 
| 126 | 53 django/template/defaulttags.py | 1084 | 1128| 370 | 32685 | 137848 | 
| 127 | 53 django/db/models/fields/__init__.py | 1109 | 1147| 293 | 32978 | 137848 | 
| 128 | 53 django/contrib/sessions/backends/base.py | 274 | 298| 199 | 33177 | 137848 | 
| 129 | 53 django/http/request.py | 162 | 194| 235 | 33412 | 137848 | 
| 130 | 53 django/core/validators.py | 99 | 138| 398 | 33810 | 137848 | 
| 131 | 53 django/conf/global_settings.py | 263 | 345| 800 | 34610 | 137848 | 
| 132 | 54 django/db/models/lookups.py | 580 | 592| 124 | 34734 | 142671 | 
| 133 | 54 django/core/validators.py | 407 | 453| 415 | 35149 | 142671 | 
| 134 | 54 django/core/validators.py | 222 | 266| 330 | 35479 | 142671 | 
| 135 | 55 django/views/debug.py | 146 | 169| 177 | 35656 | 147016 | 
| 136 | 55 django/utils/http.py | 418 | 480| 318 | 35974 | 147016 | 
| 137 | 55 django/db/models/fields/__init__.py | 2088 | 2129| 325 | 36299 | 147016 | 
| 138 | 55 django/contrib/auth/password_validation.py | 135 | 157| 197 | 36496 | 147016 | 
| 139 | 55 django/contrib/auth/tokens.py | 73 | 102| 278 | 36774 | 147016 | 
| 140 | 55 django/contrib/auth/password_validation.py | 118 | 133| 154 | 36928 | 147016 | 
| 141 | 56 django/core/handlers/asgi.py | 269 | 293| 169 | 37097 | 149330 | 
| 142 | 56 django/db/models/fields/__init__.py | 367 | 393| 199 | 37296 | 149330 | 
| 143 | 56 django/http/request.py | 614 | 628| 118 | 37414 | 149330 | 
| 144 | 56 django/core/signing.py | 145 | 169| 243 | 37657 | 149330 | 
| 145 | 57 django/core/handlers/wsgi.py | 1 | 42| 308 | 37965 | 151089 | 
| 146 | 57 django/forms/fields.py | 323 | 349| 227 | 38192 | 151089 | 
| 147 | 58 django/utils/safestring.py | 40 | 64| 159 | 38351 | 151475 | 
| 148 | 58 django/template/defaulttags.py | 517 | 540| 188 | 38539 | 151475 | 
| 149 | 58 django/db/models/fields/__init__.py | 1234 | 1247| 157 | 38696 | 151475 | 
| 150 | 59 django/db/backends/sqlite3/operations.py | 1 | 39| 267 | 38963 | 154371 | 
| 151 | 60 django/db/models/options.py | 1 | 34| 285 | 39248 | 161467 | 
| 152 | 60 django/conf/__init__.py | 1 | 40| 240 | 39488 | 161467 | 
| 153 | 61 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 39637 | 162144 | 
| 154 | 61 django/core/cache/backends/base.py | 50 | 87| 306 | 39943 | 162144 | 
| 155 | 61 django/template/defaultfilters.py | 56 | 91| 203 | 40146 | 162144 | 
| 156 | 61 django/db/models/fields/__init__.py | 2037 | 2067| 199 | 40345 | 162144 | 
| 157 | 62 django/utils/cache.py | 106 | 133| 181 | 40526 | 165892 | 
| 158 | 63 django/core/checks/urls.py | 71 | 111| 264 | 40790 | 166593 | 
| 159 | 63 django/template/defaultfilters.py | 325 | 409| 499 | 41289 | 166593 | 
| 160 | 64 django/contrib/auth/forms.py | 53 | 78| 162 | 41451 | 169802 | 
| 161 | 64 django/template/defaultfilters.py | 31 | 53| 191 | 41642 | 169802 | 
| 162 | 65 django/contrib/gis/geos/prototypes/errcheck.py | 54 | 84| 241 | 41883 | 170417 | 
| 163 | 65 django/db/backends/utils.py | 178 | 210| 225 | 42108 | 170417 | 
| 164 | 65 django/core/handlers/wsgi.py | 190 | 211| 176 | 42284 | 170417 | 
| 165 | 65 django/contrib/sessions/backends/base.py | 124 | 141| 196 | 42480 | 170417 | 
| 166 | 65 django/utils/encoding.py | 150 | 165| 182 | 42662 | 170417 | 
| 167 | 65 django/forms/fields.py | 351 | 368| 160 | 42822 | 170417 | 
| 168 | 65 django/core/validators.py | 178 | 197| 152 | 42974 | 170417 | 
| 169 | 66 django/db/backends/sqlite3/base.py | 570 | 592| 143 | 43117 | 176182 | 
| 170 | 66 django/views/csrf.py | 101 | 155| 577 | 43694 | 176182 | 
| 171 | 67 django/core/mail/message.py | 147 | 166| 218 | 43912 | 179720 | 
| 172 | 67 django/core/validators.py | 284 | 294| 109 | 44021 | 179720 | 
| 173 | 68 django/contrib/auth/middleware.py | 84 | 109| 192 | 44213 | 180714 | 
| 174 | 69 django/http/__init__.py | 1 | 22| 197 | 44410 | 180911 | 
| 175 | 69 django/middleware/csrf.py | 122 | 156| 267 | 44677 | 180911 | 
| 176 | 70 django/contrib/sessions/models.py | 1 | 36| 250 | 44927 | 181161 | 
| 177 | 71 django/utils/formats.py | 1 | 57| 377 | 45304 | 183253 | 
| 178 | 72 django/db/models/fields/mixins.py | 31 | 57| 173 | 45477 | 183596 | 
| 179 | 72 django/contrib/auth/middleware.py | 1 | 23| 171 | 45648 | 183596 | 
| 180 | 73 django/db/models/functions/mixins.py | 1 | 20| 161 | 45809 | 184010 | 
| 181 | 74 django/utils/html.py | 352 | 379| 212 | 46021 | 187112 | 
| 182 | 75 django/views/decorators/debug.py | 1 | 44| 274 | 46295 | 187701 | 
| 183 | 75 django/middleware/csrf.py | 158 | 179| 173 | 46468 | 187701 | 
| 184 | 75 django/db/models/fields/__init__.py | 1059 | 1088| 218 | 46686 | 187701 | 
| 185 | 75 django/core/checks/security/base.py | 85 | 180| 710 | 47396 | 187701 | 
| 186 | 76 django/db/backends/mysql/operations.py | 1 | 35| 282 | 47678 | 191032 | 
| 187 | 76 django/db/models/functions/text.py | 24 | 55| 217 | 47895 | 191032 | 
| 188 | 76 django/template/defaultfilters.py | 1 | 28| 207 | 48102 | 191032 | 
| 189 | **76 django/contrib/auth/hashers.py** | 360 | 385| 252 | 48354 | 191032 | 
| 190 | 76 django/views/decorators/debug.py | 77 | 92| 132 | 48486 | 191032 | 
| 191 | 76 django/http/request.py | 153 | 160| 111 | 48597 | 191032 | 
| 192 | 76 django/db/models/functions/text.py | 205 | 224| 190 | 48787 | 191032 | 
| 193 | 77 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 49124 | 191571 | 
| 194 | 77 django/utils/formats.py | 165 | 184| 202 | 49326 | 191571 | 
| 195 | 78 django/utils/translation/__init__.py | 1 | 37| 297 | 49623 | 193907 | 
| 196 | 78 django/views/defaults.py | 1 | 24| 149 | 49772 | 193907 | 
| 197 | 78 django/db/models/options.py | 409 | 431| 154 | 49926 | 193907 | 
| 198 | 79 django/bin/django-admin.py | 1 | 22| 138 | 50064 | 194045 | 
| 199 | 79 django/views/decorators/debug.py | 47 | 75| 199 | 50263 | 194045 | 
| 200 | 80 django/contrib/auth/backends.py | 183 | 234| 369 | 50632 | 195807 | 
| 201 | 80 django/core/files/storage.py | 1 | 22| 158 | 50790 | 195807 | 
| 202 | 80 django/conf/__init__.py | 161 | 220| 541 | 51331 | 195807 | 
| 203 | 81 django/template/backends/dummy.py | 1 | 53| 325 | 51656 | 196132 | 
| 204 | 81 django/views/debug.py | 132 | 144| 148 | 51804 | 196132 | 
| 205 | 81 django/contrib/auth/forms.py | 32 | 50| 161 | 51965 | 196132 | 
| 206 | 81 django/db/models/options.py | 147 | 206| 587 | 52552 | 196132 | 


## Missing Patch Files

 * 1: django/contrib/auth/hashers.py
 * 2: django/db/backends/oracle/creation.py
 * 3: django/utils/crypto.py

## Patch

```diff
diff --git a/django/contrib/auth/hashers.py b/django/contrib/auth/hashers.py
--- a/django/contrib/auth/hashers.py
+++ b/django/contrib/auth/hashers.py
@@ -185,7 +185,8 @@ def _load_library(self):
 
     def salt(self):
         """Generate a cryptographically secure nonce salt in ASCII."""
-        return get_random_string()
+        # 12 returns a 71-bit value, log_2((26+26+10)^12) =~ 71 bits
+        return get_random_string(12)
 
     def verify(self, password, encoded):
         """Check if the given password is correct."""
diff --git a/django/db/backends/oracle/creation.py b/django/db/backends/oracle/creation.py
--- a/django/db/backends/oracle/creation.py
+++ b/django/db/backends/oracle/creation.py
@@ -341,7 +341,7 @@ def _test_database_passwd(self):
         password = self._test_settings_get('PASSWORD')
         if password is None and self._test_user_create():
             # Oracle passwords are limited to 30 chars and can't contain symbols.
-            password = get_random_string(length=30)
+            password = get_random_string(30)
         return password
 
     def _test_database_tblspace(self):
diff --git a/django/utils/crypto.py b/django/utils/crypto.py
--- a/django/utils/crypto.py
+++ b/django/utils/crypto.py
@@ -4,8 +4,10 @@
 import hashlib
 import hmac
 import secrets
+import warnings
 
 from django.conf import settings
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.encoding import force_bytes
 
 
@@ -44,15 +46,31 @@ def salted_hmac(key_salt, value, secret=None, *, algorithm='sha1'):
     return hmac.new(key, msg=force_bytes(value), digestmod=hasher)
 
 
-def get_random_string(length=12,
-                      allowed_chars='abcdefghijklmnopqrstuvwxyz'
-                                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
+NOT_PROVIDED = object()  # RemovedInDjango40Warning.
+
+
+# RemovedInDjango40Warning: when the deprecation ends, replace with:
+#   def get_random_string(self, length, allowed_chars='...'):
+def get_random_string(length=NOT_PROVIDED, allowed_chars=(
+    'abcdefghijklmnopqrstuvwxyz'
+    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
+)):
     """
     Return a securely generated random string.
 
-    The default length of 12 with the a-z, A-Z, 0-9 character set returns
-    a 71-bit value. log_2((26+26+10)^12) =~ 71 bits
+    The bit length of the returned value can be calculated with the formula:
+        log_2(len(allowed_chars)^length)
+
+    For example, with default `allowed_chars` (26+26+10), this gives:
+      * length: 12, bit length =~ 71 bits
+      * length: 22, bit length =~ 131 bits
     """
+    if length is NOT_PROVIDED:
+        warnings.warn(
+            'Not providing a length argument is deprecated.',
+            RemovedInDjango40Warning,
+        )
+        length = 12
     return ''.join(secrets.choice(allowed_chars) for i in range(length))
 
 

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_crypto.py b/tests/utils_tests/test_crypto.py
--- a/tests/utils_tests/test_crypto.py
+++ b/tests/utils_tests/test_crypto.py
@@ -1,10 +1,12 @@
 import hashlib
 import unittest
 
-from django.test import SimpleTestCase
+from django.test import SimpleTestCase, ignore_warnings
 from django.utils.crypto import (
-    InvalidAlgorithm, constant_time_compare, pbkdf2, salted_hmac,
+    InvalidAlgorithm, constant_time_compare, get_random_string, pbkdf2,
+    salted_hmac,
 )
+from django.utils.deprecation import RemovedInDjango40Warning
 
 
 class TestUtilsCryptoMisc(SimpleTestCase):
@@ -183,3 +185,14 @@ def test_regression_vectors(self):
     def test_default_hmac_alg(self):
         kwargs = {'password': b'password', 'salt': b'salt', 'iterations': 1, 'dklen': 20}
         self.assertEqual(pbkdf2(**kwargs), hashlib.pbkdf2_hmac(hash_name=hashlib.sha256().name, **kwargs))
+
+
+class DeprecationTests(SimpleTestCase):
+    @ignore_warnings(category=RemovedInDjango40Warning)
+    def test_get_random_string(self):
+        self.assertEqual(len(get_random_string()), 12)
+
+    def test_get_random_string_warning(self):
+        msg = 'Not providing a length argument is deprecated.'
+        with self.assertRaisesMessage(RemovedInDjango40Warning, msg):
+            get_random_string()

```


## Code snippets

### 1 - django/utils/crypto.py:

Start line: 47, End line: 72

```python
def get_random_string(length=12,
                      allowed_chars='abcdefghijklmnopqrstuvwxyz'
                                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """
    Return a securely generated random string.

    The default length of 12 with the a-z, A-Z, 0-9 character set returns
    a 71-bit value. log_2((26+26+10)^12) =~ 71 bits
    """
    return ''.join(secrets.choice(allowed_chars) for i in range(length))


def constant_time_compare(val1, val2):
    """Return True if the two strings are equal, False otherwise."""
    return secrets.compare_digest(force_bytes(val1), force_bytes(val2))


def pbkdf2(password, salt, iterations, dklen=0, digest=None):
    """Return the hash of password using pbkdf2."""
    if digest is None:
        digest = hashlib.sha256
    dklen = dklen or None
    password = force_bytes(password)
    salt = force_bytes(salt)
    return hashlib.pbkdf2_hmac(digest().name, password, salt, iterations, dklen)
```
### 2 - django/contrib/auth/password_validation.py:

Start line: 91, End line: 115

```python
class MinimumLengthValidator:
    """
    Validate whether the password is of a minimum length.
    """
    def __init__(self, min_length=8):
        self.min_length = min_length

    def validate(self, password, user=None):
        if len(password) < self.min_length:
            raise ValidationError(
                ngettext(
                    "This password is too short. It must contain at least %(min_length)d character.",
                    "This password is too short. It must contain at least %(min_length)d characters.",
                    self.min_length
                ),
                code='password_too_short',
                params={'min_length': self.min_length},
            )

    def get_help_text(self):
        return ngettext(
            "Your password must contain at least %(min_length)d character.",
            "Your password must contain at least %(min_length)d characters.",
            self.min_length
        ) % {'min_length': self.min_length}
```
### 3 - django/db/models/fields/__init__.py:

Start line: 980, End line: 1012

```python
class CharField(Field):
    description = _("String (up to %(max_length)s)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validators.append(validators.MaxLengthValidator(self.max_length))

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_max_length_attribute(**kwargs),
        ]

    def _check_max_length_attribute(self, **kwargs):
        if self.max_length is None:
            return [
                checks.Error(
                    "CharFields must define a 'max_length' attribute.",
                    obj=self,
                    id='fields.E120',
                )
            ]
        elif (not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or
                self.max_length <= 0):
            return [
                checks.Error(
                    "'max_length' must be a positive integer.",
                    obj=self,
                    id='fields.E121',
                )
            ]
        else:
            return []
```
### 4 - django/core/checks/security/base.py:

Start line: 1, End line: 83

```python
from django.conf import settings

from .. import Error, Tags, Warning, register

REFERRER_POLICY_VALUES = {
    'no-referrer', 'no-referrer-when-downgrade', 'origin',
    'origin-when-cross-origin', 'same-origin', 'strict-origin',
    'strict-origin-when-cross-origin', 'unsafe-url',
}

SECRET_KEY_MIN_LENGTH = 50
SECRET_KEY_MIN_UNIQUE_CHARACTERS = 5

W001 = Warning(
    "You do not have 'django.middleware.security.SecurityMiddleware' "
    "in your MIDDLEWARE so the SECURE_HSTS_SECONDS, "
    "SECURE_CONTENT_TYPE_NOSNIFF, SECURE_BROWSER_XSS_FILTER, "
    "SECURE_REFERRER_POLICY, and SECURE_SSL_REDIRECT settings will have no "
    "effect.",
    id='security.W001',
)

W002 = Warning(
    "You do not have "
    "'django.middleware.clickjacking.XFrameOptionsMiddleware' in your "
    "MIDDLEWARE, so your pages will not be served with an "
    "'x-frame-options' header. Unless there is a good reason for your "
    "site to be served in a frame, you should consider enabling this "
    "header to help prevent clickjacking attacks.",
    id='security.W002',
)

W004 = Warning(
    "You have not set a value for the SECURE_HSTS_SECONDS setting. "
    "If your entire site is served only over SSL, you may want to consider "
    "setting a value and enabling HTTP Strict Transport Security. "
    "Be sure to read the documentation first; enabling HSTS carelessly "
    "can cause serious, irreversible problems.",
    id='security.W004',
)

W005 = Warning(
    "You have not set the SECURE_HSTS_INCLUDE_SUBDOMAINS setting to True. "
    "Without this, your site is potentially vulnerable to attack "
    "via an insecure connection to a subdomain. Only set this to True if "
    "you are certain that all subdomains of your domain should be served "
    "exclusively via SSL.",
    id='security.W005',
)

W006 = Warning(
    "Your SECURE_CONTENT_TYPE_NOSNIFF setting is not set to True, "
    "so your pages will not be served with an "
    "'X-Content-Type-Options: nosniff' header. "
    "You should consider enabling this header to prevent the "
    "browser from identifying content types incorrectly.",
    id='security.W006',
)

W008 = Warning(
    "Your SECURE_SSL_REDIRECT setting is not set to True. "
    "Unless your site should be available over both SSL and non-SSL "
    "connections, you may want to either set this setting True "
    "or configure a load balancer or reverse-proxy server "
    "to redirect all connections to HTTPS.",
    id='security.W008',
)

W009 = Warning(
    "Your SECRET_KEY has less than %(min_length)s characters or less than "
    "%(min_unique_chars)s unique characters. Please generate a long and random "
    "SECRET_KEY, otherwise many of Django's security-critical features will be "
    "vulnerable to attack." % {
        'min_length': SECRET_KEY_MIN_LENGTH,
        'min_unique_chars': SECRET_KEY_MIN_UNIQUE_CHARACTERS,
    },
    id='security.W009',
)

W018 = Warning(
    "You should not have DEBUG set to True in deployment.",
    id='security.W018',
)
```
### 5 - django/contrib/auth/hashers.py:

Start line: 1, End line: 27

```python
import base64
import binascii
import functools
import hashlib
import importlib
import warnings

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import (
    constant_time_compare, get_random_string, pbkdf2,
)
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _

UNUSABLE_PASSWORD_PREFIX = '!'  # This will never be a valid encoded hash
UNUSABLE_PASSWORD_SUFFIX_LENGTH = 40  # number of random chars to add after UNUSABLE_PASSWORD_PREFIX


def is_password_usable(encoded):
    """
    Return True if this password wasn't generated by
    User.set_unusable_password(), i.e. make_password(None).
    """
    return encoded is None or not encoded.startswith(UNUSABLE_PASSWORD_PREFIX)
```
### 6 - django/contrib/auth/hashers.py:

Start line: 64, End line: 77

```python
def make_password(password, salt=None, hasher='default'):
    """
    Turn a plain-text password into a hash for database storage

    Same as encode() but generate a new random salt. If password is None then
    return a concatenation of UNUSABLE_PASSWORD_PREFIX and a random string,
    which disallows logins. Additional random string reduces chances of gaining
    access to staff or superuser accounts. See ticket #20079 for more info.
    """
    if password is None:
        return UNUSABLE_PASSWORD_PREFIX + get_random_string(UNUSABLE_PASSWORD_SUFFIX_LENGTH)
    hasher = get_hasher(hasher)
    salt = salt or hasher.salt()
    return hasher.encode(password, salt)
```
### 7 - django/contrib/auth/migrations/0008_alter_user_username_max_length.py:

Start line: 1, End line: 25

```python
from django.contrib.auth import validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0007_alter_validators_add_error_messages'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='username',
            field=models.CharField(
                error_messages={'unique': 'A user with that username already exists.'},
                help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.',
                max_length=150,
                unique=True,
                validators=[validators.UnicodeUsernameValidator()],
                verbose_name='username',
            ),
        ),
    ]
```
### 8 - django/utils/text.py:

Start line: 104, End line: 122

```python
class Truncator(SimpleLazyObject):

    def _text_chars(self, length, truncate, text, truncate_len):
        """Truncate a string after a certain number of chars."""
        s_len = 0
        end_index = None
        for i, char in enumerate(text):
            if unicodedata.combining(char):
                # Don't consider combining characters
                # as adding to the string length
                continue
            s_len += 1
            if end_index is None and s_len > truncate_len:
                end_index = i
            if s_len > length:
                # Return the truncated string
                return self.add_truncation_text(text[:end_index or 0],
                                                truncate)

        # Return the original string since no truncation was necessary
        return text
```
### 9 - django/contrib/auth/hashers.py:

Start line: 529, End line: 561

```python
class UnsaltedSHA1PasswordHasher(BasePasswordHasher):
    """
    Very insecure algorithm that you should *never* use; store SHA1 hashes
    with an empty salt.

    This class is implemented because Django used to accept such password
    hashes. Some older Django installs still have these values lingering
    around so we need to handle and upgrade them properly.
    """
    algorithm = "unsalted_sha1"

    def salt(self):
        return ''

    def encode(self, password, salt):
        assert salt == ''
        hash = hashlib.sha1(password.encode()).hexdigest()
        return 'sha1$$%s' % hash

    def verify(self, password, encoded):
        encoded_2 = self.encode(password, '')
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        assert encoded.startswith('sha1$$')
        hash = encoded[6:]
        return {
            _('algorithm'): self.algorithm,
            _('hash'): mask_hash(hash),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 10 - django/contrib/auth/hashers.py:

Start line: 564, End line: 597

```python
class UnsaltedMD5PasswordHasher(BasePasswordHasher):
    """
    Incredibly insecure algorithm that you should *never* use; stores unsalted
    MD5 hashes without the algorithm prefix, also accepts MD5 hashes with an
    empty salt.

    This class is implemented because Django used to store passwords this way
    and to accept such password hashes. Some older Django installs still have
    these values lingering around so we need to handle and upgrade them
    properly.
    """
    algorithm = "unsalted_md5"

    def salt(self):
        return ''

    def encode(self, password, salt):
        assert salt == ''
        return hashlib.md5(password.encode()).hexdigest()

    def verify(self, password, encoded):
        if len(encoded) == 37 and encoded.startswith('md5$$'):
            encoded = encoded[5:]
        encoded_2 = self.encode(password, '')
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        return {
            _('algorithm'): self.algorithm,
            _('hash'): mask_hash(encoded, show=3),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 11 - django/contrib/auth/hashers.py:

Start line: 600, End line: 637

```python
class CryptPasswordHasher(BasePasswordHasher):
    """
    Password hashing using UNIX crypt (not recommended)

    The crypt module is not supported on all platforms.
    """
    algorithm = "crypt"
    library = "crypt"

    def salt(self):
        return get_random_string(2)

    def encode(self, password, salt):
        crypt = self._load_library()
        assert len(salt) == 2
        data = crypt.crypt(password, salt)
        assert data is not None  # A platform like OpenBSD with a dummy crypt module.
        # we don't need to store the salt, but Django used to do this
        return "%s$%s$%s" % (self.algorithm, '', data)

    def verify(self, password, encoded):
        crypt = self._load_library()
        algorithm, salt, data = encoded.split('$', 2)
        assert algorithm == self.algorithm
        return constant_time_compare(data, crypt.crypt(password, data))

    def safe_summary(self, encoded):
        algorithm, salt, data = encoded.split('$', 2)
        assert algorithm == self.algorithm
        return {
            _('algorithm'): algorithm,
            _('salt'): salt,
            _('hash'): mask_hash(data, show=3),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 12 - django/utils/crypto.py:

Start line: 1, End line: 44

```python
"""
Django's standard crypto functions and utilities.
"""
import hashlib
import hmac
import secrets

from django.conf import settings
from django.utils.encoding import force_bytes


class InvalidAlgorithm(ValueError):
    """Algorithm is not supported by hashlib."""
    pass


def salted_hmac(key_salt, value, secret=None, *, algorithm='sha1'):
    """
    Return the HMAC of 'value', using a key generated from key_salt and a
    secret (which defaults to settings.SECRET_KEY). Default algorithm is SHA1,
    but any algorithm name supported by hashlib.new() can be passed.

    A different key_salt should be passed in for every application of HMAC.
    """
    if secret is None:
        secret = settings.SECRET_KEY

    key_salt = force_bytes(key_salt)
    secret = force_bytes(secret)
    try:
        hasher = getattr(hashlib, algorithm)
    except AttributeError as e:
        raise InvalidAlgorithm(
            '%r is not an algorithm accepted by the hashlib module.'
            % algorithm
        ) from e
    # We need to generate a derived key from our base key.  We can do this by
    # passing the key_salt and our base key through a pseudo-random function.
    key = hasher(key_salt + secret).digest()
    # If len(key_salt + secret) > block size of the hash algorithm, the above
    # line is redundant and could be replaced by key = key_salt + secret, since
    # the hmac module does the same thing for keys longer than the block size.
    # However, we need to ensure that we *always* do this.
    return hmac.new(key, msg=force_bytes(value), digestmod=hasher)
```
### 13 - django/contrib/auth/hashers.py:

Start line: 418, End line: 447

```python
class BCryptSHA256PasswordHasher(BasePasswordHasher):

    def verify(self, password, encoded):
        algorithm, data = encoded.split('$', 1)
        assert algorithm == self.algorithm
        encoded_2 = self.encode(password, data.encode('ascii'))
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        algorithm, empty, algostr, work_factor, data = encoded.split('$', 4)
        assert algorithm == self.algorithm
        salt, checksum = data[:22], data[22:]
        return {
            _('algorithm'): algorithm,
            _('work factor'): work_factor,
            _('salt'): mask_hash(salt),
            _('checksum'): mask_hash(checksum),
        }

    def must_update(self, encoded):
        algorithm, empty, algostr, rounds, data = encoded.split('$', 4)
        return int(rounds) != self.rounds

    def harden_runtime(self, password, encoded):
        _, data = encoded.split('$', 1)
        salt = data[:29]  # Length of the salt in bcrypt.
        rounds = data.split('$')[2]
        # work factor is logarithmic, adding one doubles the load.
        diff = 2**(self.rounds - int(rounds)) - 1
        while diff > 0:
            self.encode(password, salt.encode('ascii'))
            diff -= 1
```
### 15 - django/contrib/auth/hashers.py:

Start line: 406, End line: 416

```python
class BCryptSHA256PasswordHasher(BasePasswordHasher):

    def encode(self, password, salt):
        bcrypt = self._load_library()
        password = password.encode()
        # Hash the password prior to using bcrypt to prevent password
        # truncation as described in #20138.
        if self.digest is not None:
            # Use binascii.hexlify() because a hex encoded bytestring is str.
            password = binascii.hexlify(self.digest(password).digest())

        data = bcrypt.hashpw(password, salt)
        return "%s$%s" % (self.algorithm, data.decode('ascii'))
```
### 18 - django/contrib/auth/hashers.py:

Start line: 343, End line: 358

```python
class Argon2PasswordHasher(BasePasswordHasher):

    def must_update(self, encoded):
        (algorithm, variety, version, time_cost, memory_cost, parallelism,
            salt, data) = self._decode(encoded)
        assert algorithm == self.algorithm
        argon2 = self._load_library()
        return (
            argon2.low_level.ARGON2_VERSION != version or
            self.time_cost != time_cost or
            self.memory_cost != memory_cost or
            self.parallelism != parallelism
        )

    def harden_runtime(self, password, encoded):
        # The runtime for Argon2 is too complicated to implement a sensible
        # hardening algorithm.
        pass
```
### 25 - django/contrib/auth/hashers.py:

Start line: 467, End line: 495

```python
class SHA1PasswordHasher(BasePasswordHasher):
    """
    The SHA1 password hashing algorithm (not recommended)
    """
    algorithm = "sha1"

    def encode(self, password, salt):
        assert password is not None
        assert salt and '$' not in salt
        hash = hashlib.sha1((salt + password).encode()).hexdigest()
        return "%s$%s$%s" % (self.algorithm, salt, hash)

    def verify(self, password, encoded):
        algorithm, salt, hash = encoded.split('$', 2)
        assert algorithm == self.algorithm
        encoded_2 = self.encode(password, salt)
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        algorithm, salt, hash = encoded.split('$', 2)
        assert algorithm == self.algorithm
        return {
            _('algorithm'): algorithm,
            _('salt'): mask_hash(salt, show=2),
            _('hash'): mask_hash(hash),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 30 - django/contrib/auth/hashers.py:

Start line: 215, End line: 226

```python
class BasePasswordHasher:

    def harden_runtime(self, password, encoded):
        """
        Bridge the runtime gap between the work factor supplied in `encoded`
        and the work factor suggested by this hasher.

        Taking PBKDF2 as an example, if `encoded` contains 20000 iterations and
        `self.iterations` is 30000, this method should run password through
        another 10000 iterations of PBKDF2. Similar approaches should exist
        for any hasher that has a work factor. If not, this method should be
        defined as a no-op to silence the warning.
        """
        warnings.warn('subclasses of BasePasswordHasher should provide a harden_runtime() method')
```
### 31 - django/contrib/auth/hashers.py:

Start line: 498, End line: 526

```python
class MD5PasswordHasher(BasePasswordHasher):
    """
    The Salted MD5 password hashing algorithm (not recommended)
    """
    algorithm = "md5"

    def encode(self, password, salt):
        assert password is not None
        assert salt and '$' not in salt
        hash = hashlib.md5((salt + password).encode()).hexdigest()
        return "%s$%s$%s" % (self.algorithm, salt, hash)

    def verify(self, password, encoded):
        algorithm, salt, hash = encoded.split('$', 2)
        assert algorithm == self.algorithm
        encoded_2 = self.encode(password, salt)
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        algorithm, salt, hash = encoded.split('$', 2)
        assert algorithm == self.algorithm
        return {
            _('algorithm'): algorithm,
            _('salt'): mask_hash(salt, show=2),
            _('hash'): mask_hash(hash),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 32 - django/contrib/auth/hashers.py:

Start line: 80, End line: 102

```python
@functools.lru_cache()
def get_hashers():
    hashers = []
    for hasher_path in settings.PASSWORD_HASHERS:
        hasher_cls = import_string(hasher_path)
        hasher = hasher_cls()
        if not getattr(hasher, 'algorithm'):
            raise ImproperlyConfigured("hasher doesn't specify an "
                                       "algorithm name: %s" % hasher_path)
        hashers.append(hasher)
    return hashers


@functools.lru_cache()
def get_hashers_by_algorithm():
    return {hasher.algorithm: hasher for hasher in get_hashers()}


@receiver(setting_changed)
def reset_hashers(**kwargs):
    if kwargs['setting'] == 'PASSWORD_HASHERS':
        get_hashers.cache_clear()
        get_hashers_by_algorithm.cache_clear()
```
### 33 - django/contrib/auth/hashers.py:

Start line: 328, End line: 341

```python
class Argon2PasswordHasher(BasePasswordHasher):

    def safe_summary(self, encoded):
        (algorithm, variety, version, time_cost, memory_cost, parallelism,
            salt, data) = self._decode(encoded)
        assert algorithm == self.algorithm
        return {
            _('algorithm'): algorithm,
            _('variety'): variety,
            _('version'): version,
            _('memory cost'): memory_cost,
            _('time cost'): time_cost,
            _('parallelism'): parallelism,
            _('salt'): mask_hash(salt),
            _('hash'): mask_hash(data),
        }
```
### 43 - django/contrib/auth/hashers.py:

Start line: 229, End line: 273

```python
class PBKDF2PasswordHasher(BasePasswordHasher):
    """
    Secure password hashing using the PBKDF2 algorithm (recommended)

    Configured to use PBKDF2 + HMAC + SHA256.
    The result is a 64 byte binary string.  Iterations may be changed
    safely but you must rename the algorithm if you change SHA256.
    """
    algorithm = "pbkdf2_sha256"
    iterations = 216000
    digest = hashlib.sha256

    def encode(self, password, salt, iterations=None):
        assert password is not None
        assert salt and '$' not in salt
        iterations = iterations or self.iterations
        hash = pbkdf2(password, salt, iterations, digest=self.digest)
        hash = base64.b64encode(hash).decode('ascii').strip()
        return "%s$%d$%s$%s" % (self.algorithm, iterations, salt, hash)

    def verify(self, password, encoded):
        algorithm, iterations, salt, hash = encoded.split('$', 3)
        assert algorithm == self.algorithm
        encoded_2 = self.encode(password, salt, int(iterations))
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        algorithm, iterations, salt, hash = encoded.split('$', 3)
        assert algorithm == self.algorithm
        return {
            _('algorithm'): algorithm,
            _('iterations'): iterations,
            _('salt'): mask_hash(salt),
            _('hash'): mask_hash(hash),
        }

    def must_update(self, encoded):
        algorithm, iterations, salt, hash = encoded.split('$', 3)
        return int(iterations) != self.iterations

    def harden_runtime(self, password, encoded):
        algorithm, iterations, salt, hash = encoded.split('$', 3)
        extra_iterations = self.iterations - int(iterations)
        if extra_iterations > 0:
            self.encode(password, salt, extra_iterations)
```
### 57 - django/contrib/auth/hashers.py:

Start line: 388, End line: 404

```python
class BCryptSHA256PasswordHasher(BasePasswordHasher):
    """
    Secure password hashing using the bcrypt algorithm (recommended)

    This is considered by many to be the most secure algorithm but you
    must first install the bcrypt library.  Please be warned that
    this library depends on native C code and might cause portability
    issues.
    """
    algorithm = "bcrypt_sha256"
    digest = hashlib.sha256
    library = ("bcrypt", "bcrypt")
    rounds = 12

    def salt(self):
        bcrypt = self._load_library()
        return bcrypt.gensalt(self.rounds)
```
### 73 - django/contrib/auth/hashers.py:

Start line: 105, End line: 125

```python
def get_hasher(algorithm='default'):
    """
    Return an instance of a loaded password hasher.

    If algorithm is 'default', return the default hasher. Lazily import hashers
    specified in the project's settings file if needed.
    """
    if hasattr(algorithm, 'algorithm'):
        return algorithm

    elif algorithm == 'default':
        return get_hashers()[0]

    else:
        hashers = get_hashers_by_algorithm()
        try:
            return hashers[algorithm]
        except KeyError:
            raise ValueError("Unknown password hashing algorithm '%s'. "
                             "Did you specify it in the PASSWORD_HASHERS "
                             "setting?" % algorithm)
```
### 74 - django/contrib/auth/hashers.py:

Start line: 128, End line: 146

```python
def identify_hasher(encoded):
    """
    Return an instance of a loaded password hasher.

    Identify hasher algorithm by examining encoded hash, and call
    get_hasher() to return hasher. Raise ValueError if
    algorithm cannot be identified, or if hasher is not loaded.
    """
    # Ancient versions of Django created plain MD5 passwords and accepted
    # MD5 passwords with an empty salt.
    if ((len(encoded) == 32 and '$' not in encoded) or
            (len(encoded) == 37 and encoded.startswith('md5$$'))):
        algorithm = 'unsalted_md5'
    # Ancient versions of Django accepted SHA1 passwords with an empty salt.
    elif len(encoded) == 46 and encoded.startswith('sha1$$'):
        algorithm = 'unsalted_sha1'
    else:
        algorithm = encoded.split('$', 1)[0]
    return get_hasher(algorithm)
```
### 77 - django/contrib/auth/hashers.py:

Start line: 450, End line: 464

```python
class BCryptPasswordHasher(BCryptSHA256PasswordHasher):
    """
    Secure password hashing using the bcrypt algorithm

    This is considered by many to be the most secure algorithm but you
    must first install the bcrypt library.  Please be warned that
    this library depends on native C code and might cause portability
    issues.

    This hasher does not first hash the password which means it is subject to
    bcrypt's 72 bytes password truncation. Most use cases should prefer the
    BCryptSHA256PasswordHasher.
    """
    algorithm = "bcrypt"
    digest = None
```
### 98 - django/contrib/auth/hashers.py:

Start line: 149, End line: 184

```python
def mask_hash(hash, show=6, char="*"):
    """
    Return the given hash, with only the first ``show`` number shown. The
    rest are masked with ``char`` for security reasons.
    """
    masked = hash[:show]
    masked += char * len(hash[show:])
    return masked


class BasePasswordHasher:
    """
    Abstract base class for password hashers

    When creating your own hasher, you need to override algorithm,
    verify(), encode() and safe_summary().

    PasswordHasher objects are immutable.
    """
    algorithm = None
    library = None

    def _load_library(self):
        if self.library is not None:
            if isinstance(self.library, (tuple, list)):
                name, mod_path = self.library
            else:
                mod_path = self.library
            try:
                module = importlib.import_module(mod_path)
            except ImportError as e:
                raise ValueError("Couldn't load %r algorithm library: %s" %
                                 (self.__class__.__name__, e))
            return module
        raise ValueError("Hasher %r doesn't specify a library attribute" %
                         self.__class__.__name__)
```
### 106 - django/contrib/auth/hashers.py:

Start line: 276, End line: 326

```python
class PBKDF2SHA1PasswordHasher(PBKDF2PasswordHasher):
    """
    Alternate PBKDF2 hasher which uses SHA1, the default PRF
    recommended by PKCS #5. This is compatible with other
    implementations of PBKDF2, such as openssl's
    PKCS5_PBKDF2_HMAC_SHA1().
    """
    algorithm = "pbkdf2_sha1"
    digest = hashlib.sha1


class Argon2PasswordHasher(BasePasswordHasher):
    """
    Secure password hashing using the argon2 algorithm.

    This is the winner of the Password Hashing Competition 2013-2015
    (https://password-hashing.net). It requires the argon2-cffi library which
    depends on native C code and might cause portability issues.
    """
    algorithm = 'argon2'
    library = 'argon2'

    time_cost = 2
    memory_cost = 512
    parallelism = 2

    def encode(self, password, salt):
        argon2 = self._load_library()
        data = argon2.low_level.hash_secret(
            password.encode(),
            salt.encode(),
            time_cost=self.time_cost,
            memory_cost=self.memory_cost,
            parallelism=self.parallelism,
            hash_len=argon2.DEFAULT_HASH_LENGTH,
            type=argon2.low_level.Type.I,
        )
        return self.algorithm + data.decode('ascii')

    def verify(self, password, encoded):
        argon2 = self._load_library()
        algorithm, rest = encoded.split('$', 1)
        assert algorithm == self.algorithm
        try:
            return argon2.low_level.verify_secret(
                ('$' + rest).encode('ascii'),
                password.encode(),
                type=argon2.low_level.Type.I,
            )
        except argon2.exceptions.VerificationError:
            return False
```
### 189 - django/contrib/auth/hashers.py:

Start line: 360, End line: 385

```python
class Argon2PasswordHasher(BasePasswordHasher):

    def _decode(self, encoded):
        """
        Split an encoded hash and return: (
            algorithm, variety, version, time_cost, memory_cost,
            parallelism, salt, data,
        ).
        """
        bits = encoded.split('$')
        if len(bits) == 5:
            # Argon2 < 1.3
            algorithm, variety, raw_params, salt, data = bits
            version = 0x10
        else:
            assert len(bits) == 6
            algorithm, variety, raw_version, raw_params, salt, data = bits
            assert raw_version.startswith('v=')
            version = int(raw_version[len('v='):])
        params = dict(bit.split('=', 1) for bit in raw_params.split(','))
        assert len(params) == 3 and all(x in params for x in ('t', 'm', 'p'))
        time_cost = int(params['t'])
        memory_cost = int(params['m'])
        parallelism = int(params['p'])
        return (
            algorithm, variety, version, time_cost, memory_cost, parallelism,
            salt, data,
        )
```
