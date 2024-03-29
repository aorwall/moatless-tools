# django__django-16631

| **django/django** | `9b224579875e30203d079cc2fee83b116d98eb78` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 25433 |
| **Any found context length** | 10028 |
| **Avg pos** | 115.5 |
| **Min pos** | 32 |
| **Max pos** | 119 |
| **Top file pos** | 9 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/auth/__init__.py b/django/contrib/auth/__init__.py
--- a/django/contrib/auth/__init__.py
+++ b/django/contrib/auth/__init__.py
@@ -199,12 +199,26 @@ def get_user(request):
             # Verify the session
             if hasattr(user, "get_session_auth_hash"):
                 session_hash = request.session.get(HASH_SESSION_KEY)
-                session_hash_verified = session_hash and constant_time_compare(
-                    session_hash, user.get_session_auth_hash()
-                )
+                if not session_hash:
+                    session_hash_verified = False
+                else:
+                    session_auth_hash = user.get_session_auth_hash()
+                    session_hash_verified = constant_time_compare(
+                        session_hash, session_auth_hash
+                    )
                 if not session_hash_verified:
-                    request.session.flush()
-                    user = None
+                    # If the current secret does not verify the session, try
+                    # with the fallback secrets and stop when a matching one is
+                    # found.
+                    if session_hash and any(
+                        constant_time_compare(session_hash, fallback_auth_hash)
+                        for fallback_auth_hash in user.get_session_auth_fallback_hash()
+                    ):
+                        request.session.cycle_key()
+                        request.session[HASH_SESSION_KEY] = session_auth_hash
+                    else:
+                        request.session.flush()
+                        user = None
 
     return user or AnonymousUser()
 
diff --git a/django/contrib/auth/base_user.py b/django/contrib/auth/base_user.py
--- a/django/contrib/auth/base_user.py
+++ b/django/contrib/auth/base_user.py
@@ -5,6 +5,7 @@
 import unicodedata
 import warnings
 
+from django.conf import settings
 from django.contrib.auth import password_validation
 from django.contrib.auth.hashers import (
     check_password,
@@ -135,10 +136,18 @@ def get_session_auth_hash(self):
         """
         Return an HMAC of the password field.
         """
+        return self._get_session_auth_hash()
+
+    def get_session_auth_fallback_hash(self):
+        for fallback_secret in settings.SECRET_KEY_FALLBACKS:
+            yield self._get_session_auth_hash(secret=fallback_secret)
+
+    def _get_session_auth_hash(self, secret=None):
         key_salt = "django.contrib.auth.models.AbstractBaseUser.get_session_auth_hash"
         return salted_hmac(
             key_salt,
             self.password,
+            secret=secret,
             algorithm="sha256",
         ).hexdigest()
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/__init__.py | 202 | 207 | 119 | 9 | 34227
| django/contrib/auth/base_user.py | 8 | 8 | 80 | 14 | 25433
| django/contrib/auth/base_user.py | 138 | 138 | 32 | 14 | 10028


## Problem Statement

```
SECRET_KEY_FALLBACKS is not used for sessions
Description
	
I recently rotated my secret key, made the old one available in SECRET_KEY_FALLBACKS and I'm pretty sure everyone on our site is logged out now.
I think the docs for ​SECRET_KEY_FALLBACKS may be incorrect when stating the following:
In order to rotate your secret keys, set a new SECRET_KEY and move the previous value to the beginning of SECRET_KEY_FALLBACKS. Then remove the old values from the end of the SECRET_KEY_FALLBACKS when you are ready to expire the sessions, password reset tokens, and so on, that make use of them.
When looking at the Django source code, I see that the ​salted_hmac function uses the SECRET_KEY by default and the ​AbstractBaseUser.get_session_auth_hash method does not call salted_hmac with a value for the secret keyword argument.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/checks/security/base.py | 226 | 239| 127 | 127 | 2189 | 
| 2 | 2 django/contrib/auth/tokens.py | 84 | 96| 139 | 266 | 3129 | 
| 3 | 3 django/contrib/sessions/backends/base.py | 168 | 198| 218 | 484 | 5508 | 
| 4 | 4 django/contrib/sessions/backends/signed_cookies.py | 26 | 82| 367 | 851 | 6030 | 
| 5 | 5 django/contrib/auth/hashers.py | 772 | 825| 358 | 1209 | 11912 | 
| 6 | 5 django/contrib/sessions/backends/base.py | 33 | 166| 858 | 2067 | 11912 | 
| 7 | 5 django/contrib/auth/hashers.py | 719 | 769| 335 | 2402 | 11912 | 
| 8 | 6 django/contrib/sessions/base_session.py | 26 | 48| 139 | 2541 | 12201 | 
| 9 | 6 django/contrib/sessions/backends/base.py | 1 | 30| 148 | 2689 | 12201 | 
| 10 | 7 django/utils/crypto.py | 1 | 44| 350 | 3039 | 12810 | 
| 11 | 7 django/contrib/auth/hashers.py | 288 | 341| 450 | 3489 | 12810 | 
| 12 | 7 django/contrib/auth/tokens.py | 98 | 133| 313 | 3802 | 12810 | 
| 13 | 8 django/contrib/sessions/backends/cached_db.py | 46 | 73| 180 | 3982 | 13243 | 
| 14 | 8 django/contrib/auth/hashers.py | 489 | 527| 332 | 4314 | 13243 | 
| 15 | **9 django/contrib/auth/__init__.py** | 1 | 38| 240 | 4554 | 14847 | 
| 16 | 9 django/contrib/sessions/backends/signed_cookies.py | 1 | 24| 160 | 4714 | 14847 | 
| 17 | 10 django/core/checks/security/sessions.py | 1 | 100| 580 | 5294 | 15428 | 
| 18 | 10 django/contrib/auth/hashers.py | 477 | 487| 127 | 5421 | 15428 | 
| 19 | 10 django/contrib/auth/hashers.py | 1 | 34| 213 | 5634 | 15428 | 
| 20 | 11 django/contrib/sessions/backends/file.py | 180 | 211| 210 | 5844 | 16928 | 
| 21 | 12 django/contrib/sessions/backends/cache.py | 54 | 86| 206 | 6050 | 17502 | 
| 22 | 13 django/core/signing.py | 179 | 238| 489 | 6539 | 19866 | 
| 23 | 13 django/contrib/auth/hashers.py | 576 | 623| 334 | 6873 | 19866 | 
| 24 | 13 django/contrib/auth/tokens.py | 1 | 49| 308 | 7181 | 19866 | 
| 25 | 13 django/contrib/auth/hashers.py | 677 | 716| 274 | 7455 | 19866 | 
| 26 | 13 django/contrib/sessions/backends/cache.py | 35 | 52| 169 | 7624 | 19866 | 
| 27 | 13 django/contrib/sessions/backends/base.py | 278 | 366| 586 | 8210 | 19866 | 
| 28 | 13 django/contrib/sessions/backends/file.py | 117 | 178| 534 | 8744 | 19866 | 
| 29 | 13 django/contrib/auth/hashers.py | 626 | 674| 339 | 9083 | 19866 | 
| 30 | 13 django/contrib/auth/hashers.py | 458 | 475| 127 | 9210 | 19866 | 
| 31 | 13 django/contrib/auth/hashers.py | 548 | 574| 215 | 9425 | 19866 | 
| **-> 32 <-** | **14 django/contrib/auth/base_user.py** | 56 | 159| 603 | 10028 | 20817 | 
| 33 | 14 django/contrib/sessions/backends/file.py | 79 | 115| 257 | 10285 | 20817 | 
| 34 | 15 django/conf/global_settings.py | 481 | 596| 797 | 11082 | 26655 | 
| 35 | **15 django/contrib/auth/__init__.py** | 41 | 60| 167 | 11249 | 26655 | 
| 36 | 15 django/contrib/auth/hashers.py | 213 | 270| 414 | 11663 | 26655 | 
| 37 | 15 django/core/signing.py | 95 | 130| 252 | 11915 | 26655 | 
| 38 | 15 django/contrib/auth/hashers.py | 530 | 545| 126 | 12041 | 26655 | 
| 39 | 15 django/core/signing.py | 1 | 92| 777 | 12818 | 26655 | 
| 40 | 16 django/contrib/sessions/middleware.py | 1 | 78| 586 | 13404 | 27242 | 
| 41 | 17 django/core/cache/backends/base.py | 1 | 53| 265 | 13669 | 30350 | 
| 42 | 17 django/contrib/auth/hashers.py | 93 | 116| 169 | 13838 | 30350 | 
| 43 | 17 django/core/checks/security/base.py | 1 | 79| 691 | 14529 | 30350 | 
| 44 | 17 django/conf/global_settings.py | 264 | 355| 832 | 15361 | 30350 | 
| 45 | 17 django/contrib/auth/hashers.py | 407 | 455| 419 | 15780 | 30350 | 
| 46 | 17 django/contrib/auth/hashers.py | 386 | 405| 185 | 15965 | 30350 | 
| 47 | 18 django/middleware/csrf.py | 58 | 67| 111 | 16076 | 34461 | 
| 48 | 19 django/contrib/sessions/models.py | 1 | 36| 247 | 16323 | 34708 | 
| 49 | 19 django/middleware/csrf.py | 220 | 250| 259 | 16582 | 34708 | 
| 50 | 19 django/contrib/sessions/backends/base.py | 200 | 225| 214 | 16796 | 34708 | 
| 51 | 19 django/contrib/auth/hashers.py | 71 | 90| 188 | 16984 | 34708 | 
| 52 | 19 django/core/checks/security/base.py | 183 | 223| 306 | 17290 | 34708 | 
| 53 | 19 django/conf/global_settings.py | 153 | 263| 832 | 18122 | 34708 | 
| 54 | 19 django/contrib/auth/tokens.py | 51 | 82| 197 | 18319 | 34708 | 
| 55 | 19 django/utils/crypto.py | 64 | 77| 131 | 18450 | 34708 | 
| 56 | 19 django/contrib/sessions/backends/file.py | 1 | 45| 264 | 18714 | 34708 | 
| 57 | 19 django/middleware/csrf.py | 116 | 139| 181 | 18895 | 34708 | 
| 58 | 20 django/contrib/auth/admin.py | 1 | 25| 195 | 19090 | 36479 | 
| 59 | 21 django/contrib/auth/views.py | 1 | 32| 255 | 19345 | 39193 | 
| 60 | 21 django/contrib/sessions/backends/cached_db.py | 1 | 44| 257 | 19602 | 39193 | 
| 61 | 21 django/contrib/auth/hashers.py | 37 | 68| 246 | 19848 | 39193 | 
| 62 | 21 django/contrib/sessions/backends/cache.py | 1 | 33| 210 | 20058 | 39193 | 
| 63 | 22 django/contrib/sessions/backends/db.py | 1 | 71| 457 | 20515 | 39917 | 
| 64 | 22 django/core/signing.py | 157 | 176| 105 | 20620 | 39917 | 
| 65 | 23 django/core/management/utils.py | 79 | 111| 222 | 20842 | 41150 | 
| 66 | 23 django/contrib/auth/hashers.py | 344 | 384| 300 | 21142 | 41150 | 
| 67 | 23 django/utils/crypto.py | 47 | 61| 128 | 21270 | 41150 | 
| 68 | 23 django/core/cache/backends/base.py | 391 | 404| 112 | 21382 | 41150 | 
| 69 | 23 django/middleware/csrf.py | 142 | 156| 167 | 21549 | 41150 | 
| 70 | 23 django/core/checks/security/base.py | 81 | 180| 732 | 22281 | 41150 | 
| 71 | 24 django/contrib/auth/middleware.py | 128 | 140| 107 | 22388 | 42229 | 
| 72 | 24 django/contrib/sessions/backends/db.py | 73 | 111| 273 | 22661 | 42229 | 
| 73 | 24 django/contrib/auth/middleware.py | 59 | 96| 362 | 23023 | 42229 | 
| 74 | 24 django/contrib/sessions/backends/base.py | 250 | 276| 213 | 23236 | 42229 | 
| 75 | 24 django/middleware/csrf.py | 70 | 92| 209 | 23445 | 42229 | 
| 76 | **24 django/contrib/auth/__init__.py** | 94 | 144| 402 | 23847 | 42229 | 
| 77 | 25 django/contrib/sessions/migrations/0001_initial.py | 1 | 38| 173 | 24020 | 42402 | 
| 78 | 25 django/middleware/csrf.py | 1 | 55| 480 | 24500 | 42402 | 
| 79 | 25 django/middleware/csrf.py | 348 | 411| 585 | 25085 | 42402 | 
| **-> 80 <-** | **25 django/contrib/auth/base_user.py** | 1 | 53| 348 | 25433 | 42402 | 
| 81 | 25 django/contrib/auth/middleware.py | 25 | 36| 107 | 25540 | 42402 | 
| 82 | 25 django/contrib/sessions/backends/file.py | 47 | 60| 129 | 25669 | 42402 | 
| 83 | 25 django/contrib/auth/views.py | 229 | 249| 163 | 25832 | 42402 | 
| 84 | 25 django/contrib/auth/hashers.py | 166 | 178| 116 | 25948 | 42402 | 
| 85 | 25 django/contrib/auth/middleware.py | 39 | 57| 178 | 26126 | 42402 | 
| 86 | 25 django/contrib/auth/admin.py | 149 | 214| 477 | 26603 | 42402 | 
| 87 | 25 django/contrib/auth/hashers.py | 272 | 285| 154 | 26757 | 42402 | 
| 88 | 25 django/conf/global_settings.py | 405 | 480| 790 | 27547 | 42402 | 
| 89 | 26 django/contrib/auth/forms.py | 63 | 81| 124 | 27671 | 45721 | 
| 90 | 26 django/contrib/auth/middleware.py | 98 | 125| 195 | 27866 | 45721 | 
| 91 | 26 django/contrib/auth/views.py | 252 | 294| 382 | 28248 | 45721 | 
| 92 | 26 django/contrib/auth/hashers.py | 144 | 163| 189 | 28437 | 45721 | 
| 93 | 27 django/contrib/auth/backends.py | 183 | 234| 377 | 28814 | 47477 | 
| 94 | 28 django/core/cache/backends/locmem.py | 79 | 118| 268 | 29082 | 48362 | 
| 95 | 28 django/contrib/auth/backends.py | 31 | 49| 142 | 29224 | 48362 | 
| 96 | 28 django/core/signing.py | 133 | 154| 210 | 29434 | 48362 | 
| 97 | 29 django/contrib/sessions/exceptions.py | 1 | 20| 0 | 29434 | 48433 | 
| 98 | 29 django/contrib/auth/hashers.py | 181 | 211| 191 | 29625 | 48433 | 
| 99 | 30 django/contrib/auth/management/commands/changepassword.py | 38 | 82| 339 | 29964 | 48980 | 
| 100 | 30 django/contrib/auth/middleware.py | 1 | 22| 143 | 30107 | 48980 | 
| 101 | 31 django/core/cache/backends/dummy.py | 1 | 35| 231 | 30338 | 49212 | 
| 102 | 32 django/core/checks/security/csrf.py | 1 | 42| 305 | 30643 | 49677 | 
| 103 | 33 django/contrib/sessions/management/commands/clearsessions.py | 1 | 22| 123 | 30766 | 49800 | 
| 104 | 33 django/contrib/auth/views.py | 296 | 345| 325 | 31091 | 49800 | 
| 105 | 33 django/contrib/sessions/backends/base.py | 227 | 248| 157 | 31248 | 49800 | 
| 106 | 33 django/contrib/auth/views.py | 213 | 227| 133 | 31381 | 49800 | 
| 107 | 33 django/core/signing.py | 265 | 276| 178 | 31559 | 49800 | 
| 108 | 34 django/contrib/sessions/apps.py | 1 | 8| 0 | 31559 | 49837 | 
| 109 | 35 django/middleware/security.py | 1 | 31| 281 | 31840 | 50363 | 
| 110 | 35 django/middleware/csrf.py | 296 | 346| 450 | 32290 | 50363 | 
| 111 | 35 django/contrib/sessions/base_session.py | 1 | 23| 149 | 32439 | 50363 | 
| 112 | 35 django/contrib/auth/views.py | 348 | 380| 239 | 32678 | 50363 | 
| 113 | 36 django/contrib/auth/password_validation.py | 1 | 38| 218 | 32896 | 52257 | 
| 114 | 37 django/contrib/redirects/middleware.py | 1 | 51| 354 | 33250 | 52612 | 
| 115 | 38 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 18| 0 | 33250 | 52691 | 
| 116 | 38 django/contrib/auth/forms.py | 33 | 60| 186 | 33436 | 52691 | 
| 117 | 38 django/contrib/auth/backends.py | 163 | 181| 146 | 33582 | 52691 | 
| 118 | 38 django/conf/global_settings.py | 597 | 668| 453 | 34035 | 52691 | 
| **-> 119 <-** | **38 django/contrib/auth/__init__.py** | 182 | 209| 192 | 34227 | 52691 | 
| 120 | 38 django/contrib/auth/backends.py | 1 | 28| 171 | 34398 | 52691 | 
| 121 | 39 django/contrib/sessions/serializers.py | 1 | 4| 0 | 34398 | 52711 | 
| 122 | 39 django/core/cache/backends/base.py | 56 | 95| 313 | 34711 | 52711 | 
| 123 | 40 django/db/migrations/loader.py | 169 | 197| 295 | 35006 | 55864 | 
| 124 | 41 django/core/cache/backends/memcached.py | 120 | 141| 190 | 35196 | 57354 | 
| 125 | 42 django/core/cache/backends/filebased.py | 61 | 97| 260 | 35456 | 58592 | 
| 126 | 43 django/utils/cache.py | 349 | 373| 226 | 35682 | 62394 | 
| 127 | 44 django/contrib/auth/models.py | 158 | 172| 164 | 35846 | 65777 | 
| 128 | 44 django/contrib/auth/hashers.py | 119 | 141| 143 | 35989 | 65777 | 
| 129 | 44 django/core/cache/backends/base.py | 110 | 190| 638 | 36627 | 65777 | 
| 130 | 45 django/contrib/messages/storage/session.py | 1 | 53| 339 | 36966 | 66117 | 
| 131 | 46 django/contrib/auth/mixins.py | 46 | 73| 232 | 37198 | 67012 | 
| 132 | 46 django/contrib/auth/models.py | 1 | 35| 231 | 37429 | 67012 | 
| 133 | 46 django/contrib/auth/forms.py | 246 | 271| 173 | 37602 | 67012 | 
| 134 | 46 django/contrib/auth/forms.py | 1 | 30| 232 | 37834 | 67012 | 
| 135 | 47 django/core/cache/backends/db.py | 112 | 203| 809 | 38643 | 69156 | 
| 136 | 47 django/core/cache/backends/memcached.py | 69 | 101| 345 | 38988 | 69156 | 
| 137 | 48 django/http/request.py | 194 | 216| 166 | 39154 | 74734 | 
| 138 | 48 django/core/cache/backends/base.py | 97 | 108| 119 | 39273 | 74734 | 
| 139 | 48 django/contrib/sessions/backends/file.py | 62 | 77| 133 | 39406 | 74734 | 
| 140 | 48 django/contrib/auth/password_validation.py | 217 | 267| 386 | 39792 | 74734 | 
| 141 | **48 django/contrib/auth/__init__.py** | 212 | 231| 154 | 39946 | 74734 | 
| 142 | 49 django/core/management/commands/startproject.py | 1 | 22| 159 | 40105 | 74893 | 
| 143 | 49 django/contrib/auth/password_validation.py | 179 | 214| 246 | 40351 | 74893 | 
| 144 | 50 django/conf/__init__.py | 264 | 286| 209 | 40560 | 77258 | 
| 145 | 51 django/contrib/auth/signals.py | 1 | 6| 0 | 40560 | 77282 | 
| 146 | **51 django/contrib/auth/__init__.py** | 147 | 162| 129 | 40689 | 77282 | 
| 147 | 51 django/contrib/auth/views.py | 194 | 210| 127 | 40816 | 77282 | 
| 148 | 51 django/contrib/auth/backends.py | 51 | 64| 131 | 40947 | 77282 | 
| 149 | 51 django/core/signing.py | 279 | 303| 196 | 41143 | 77282 | 
| 150 | 51 django/core/checks/security/base.py | 259 | 284| 211 | 41354 | 77282 | 
| 151 | 51 django/core/cache/backends/memcached.py | 103 | 118| 152 | 41506 | 77282 | 
| 152 | 51 django/middleware/csrf.py | 413 | 468| 577 | 42083 | 77282 | 
| 153 | 51 django/middleware/security.py | 33 | 67| 251 | 42334 | 77282 | 
| 154 | 52 django/views/debug.py | 182 | 194| 148 | 42482 | 82338 | 
| 155 | 52 django/contrib/auth/forms.py | 451 | 505| 360 | 42842 | 82338 | 
| 156 | 52 django/utils/cache.py | 376 | 395| 190 | 43032 | 82338 | 
| 157 | 52 django/contrib/auth/mixins.py | 112 | 136| 150 | 43182 | 82338 | 
| 158 | 52 django/core/cache/backends/db.py | 234 | 254| 246 | 43428 | 82338 | 
| 159 | 53 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 43675 | 82585 | 
| 160 | 54 django/contrib/auth/checks.py | 1 | 104| 728 | 44403 | 84101 | 
| 161 | 55 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 18| 0 | 44403 | 84183 | 
| 162 | 55 django/contrib/auth/password_validation.py | 159 | 177| 179 | 44582 | 84183 | 
| 163 | 55 django/contrib/auth/models.py | 405 | 500| 537 | 45119 | 84183 | 
| 164 | 55 django/contrib/auth/views.py | 90 | 121| 216 | 45335 | 84183 | 
| 165 | 55 django/core/cache/backends/base.py | 247 | 299| 461 | 45796 | 84183 | 
| 166 | 55 django/contrib/auth/forms.py | 326 | 369| 300 | 46096 | 84183 | 
| 167 | 55 django/core/cache/backends/locmem.py | 65 | 77| 134 | 46230 | 84183 | 
| 168 | 55 django/views/debug.py | 237 | 289| 471 | 46701 | 84183 | 
| 169 | 55 django/core/cache/backends/db.py | 205 | 232| 280 | 46981 | 84183 | 
| 170 | 55 django/core/cache/backends/db.py | 100 | 110| 222 | 47203 | 84183 | 
| 171 | 55 django/middleware/csrf.py | 252 | 268| 186 | 47389 | 84183 | 
| 172 | 56 django/contrib/auth/urls.py | 1 | 37| 253 | 47642 | 84436 | 
| 173 | 56 django/contrib/auth/forms.py | 372 | 413| 286 | 47928 | 84436 | 
| 174 | 56 django/views/debug.py | 146 | 180| 283 | 48211 | 84436 | 
| 175 | 56 django/contrib/auth/password_validation.py | 60 | 96| 281 | 48492 | 84436 | 
| 176 | 56 django/core/cache/backends/db.py | 42 | 98| 430 | 48922 | 84436 | 
| 177 | 56 django/contrib/auth/password_validation.py | 129 | 156| 279 | 49201 | 84436 | 
| 178 | 56 django/views/debug.py | 196 | 221| 181 | 49382 | 84436 | 
| 179 | 57 django/http/__init__.py | 1 | 53| 241 | 49623 | 84677 | 
| 180 | 58 django/db/migrations/state.py | 512 | 536| 184 | 49807 | 92845 | 
| 181 | 59 django/db/__init__.py | 1 | 62| 299 | 50106 | 93144 | 
| 182 | 59 django/contrib/auth/forms.py | 305 | 324| 151 | 50257 | 93144 | 
| 183 | 60 django/views/csrf.py | 30 | 88| 587 | 50844 | 93944 | 
| 184 | 60 django/core/checks/security/csrf.py | 45 | 68| 159 | 51003 | 93944 | 
| 185 | 61 django/core/mail/backends/dummy.py | 1 | 11| 0 | 51003 | 93987 | 
| 186 | 61 django/contrib/auth/management/commands/changepassword.py | 1 | 36| 212 | 51215 | 93987 | 
| 187 | 62 django/core/cache/backends/redis.py | 159 | 234| 652 | 51867 | 95702 | 
| 188 | 62 django/middleware/csrf.py | 159 | 182| 159 | 52026 | 95702 | 
| 189 | 63 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 28| 158 | 52184 | 95860 | 
| 190 | 63 django/middleware/csrf.py | 95 | 113| 215 | 52399 | 95860 | 
| 191 | 64 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 18| 0 | 52399 | 95942 | 
| 192 | 65 django/utils/http.py | 1 | 39| 460 | 52859 | 99144 | 
| 193 | 65 django/core/cache/backends/locmem.py | 1 | 63| 495 | 53354 | 99144 | 
| 194 | 66 django/core/handlers/asgi.py | 1 | 25| 123 | 53477 | 101565 | 
| 195 | 66 django/utils/cache.py | 137 | 160| 201 | 53678 | 101565 | 
| 196 | 67 django/contrib/messages/storage/cookie.py | 62 | 77| 149 | 53827 | 102865 | 
| 197 | 67 django/contrib/auth/password_validation.py | 41 | 57| 114 | 53941 | 102865 | 
| 198 | 67 django/contrib/auth/forms.py | 149 | 190| 294 | 54235 | 102865 | 
| 199 | 67 django/contrib/auth/admin.py | 43 | 119| 528 | 54763 | 102865 | 
| 200 | 68 django/contrib/messages/storage/fallback.py | 40 | 57| 150 | 54913 | 103283 | 
| 201 | 68 django/contrib/auth/models.py | 174 | 199| 194 | 55107 | 103283 | 
| 202 | 68 django/core/cache/backends/filebased.py | 46 | 59| 145 | 55252 | 103283 | 
| 203 | 68 django/contrib/auth/forms.py | 416 | 448| 198 | 55450 | 103283 | 
| 204 | 68 django/core/signing.py | 240 | 263| 255 | 55705 | 103283 | 
| 205 | 69 django/db/models/fields/json.py | 547 | 637| 515 | 56220 | 108054 | 


### Hint

```
Hi! I'm a colleague of Eric's, and we were discussing some of the ramifications of fixing this issue and I thought I'd write them here for posterity. In particular for user sessions, using fallback keys in the AuthenticationMiddleware/auth.get_user(request) will keep existing _auth_user_hash values from before the rotation being seen as valid, which is nice during the rotation period, but without any upgrading of the _auth_user_hash values, when the rotation is finished and the fallback keys are removed, all of those sessions will essentially be invalidated again. So, I think possibly an additional need here is a way to upgrade the cookies when a fallback key is used? Or at least documentation calling out this drawback. Edit: It's possible I'm conflating a cookie value and a session value, but either way I think the principle of what I wrote stands?
Thanks for the report. Agreed, we should check fallback session hashes. Bug in 0dcd549bbe36c060f536ec270d34d9e7d4b8e6c7. In particular for user sessions, using fallback keys in the AuthenticationMiddleware/auth.get_user(request) will keep existing _auth_user_hash values from before the rotation being seen as valid, which is nice during the rotation period, but without any upgrading of the _auth_user_hash values, when the rotation is finished and the fallback keys are removed, all of those sessions will essentially be invalidated again. So, I think possibly an additional need here is a way to upgrade the cookies when a fallback key is used? Or at least documentation calling out this drawback. Edit: It's possible I'm conflating a cookie value and a session value, but either way I think the principle of what I wrote stands? As far as I'm aware, this is a new feature request not a bug in #30360, so we should discuss it separately. Maybe we could call update_session_auth_hash() when a fallback hash is valid 🤔
```

## Patch

```diff
diff --git a/django/contrib/auth/__init__.py b/django/contrib/auth/__init__.py
--- a/django/contrib/auth/__init__.py
+++ b/django/contrib/auth/__init__.py
@@ -199,12 +199,26 @@ def get_user(request):
             # Verify the session
             if hasattr(user, "get_session_auth_hash"):
                 session_hash = request.session.get(HASH_SESSION_KEY)
-                session_hash_verified = session_hash and constant_time_compare(
-                    session_hash, user.get_session_auth_hash()
-                )
+                if not session_hash:
+                    session_hash_verified = False
+                else:
+                    session_auth_hash = user.get_session_auth_hash()
+                    session_hash_verified = constant_time_compare(
+                        session_hash, session_auth_hash
+                    )
                 if not session_hash_verified:
-                    request.session.flush()
-                    user = None
+                    # If the current secret does not verify the session, try
+                    # with the fallback secrets and stop when a matching one is
+                    # found.
+                    if session_hash and any(
+                        constant_time_compare(session_hash, fallback_auth_hash)
+                        for fallback_auth_hash in user.get_session_auth_fallback_hash()
+                    ):
+                        request.session.cycle_key()
+                        request.session[HASH_SESSION_KEY] = session_auth_hash
+                    else:
+                        request.session.flush()
+                        user = None
 
     return user or AnonymousUser()
 
diff --git a/django/contrib/auth/base_user.py b/django/contrib/auth/base_user.py
--- a/django/contrib/auth/base_user.py
+++ b/django/contrib/auth/base_user.py
@@ -5,6 +5,7 @@
 import unicodedata
 import warnings
 
+from django.conf import settings
 from django.contrib.auth import password_validation
 from django.contrib.auth.hashers import (
     check_password,
@@ -135,10 +136,18 @@ def get_session_auth_hash(self):
         """
         Return an HMAC of the password field.
         """
+        return self._get_session_auth_hash()
+
+    def get_session_auth_fallback_hash(self):
+        for fallback_secret in settings.SECRET_KEY_FALLBACKS:
+            yield self._get_session_auth_hash(secret=fallback_secret)
+
+    def _get_session_auth_hash(self, secret=None):
         key_salt = "django.contrib.auth.models.AbstractBaseUser.get_session_auth_hash"
         return salted_hmac(
             key_salt,
             self.password,
+            secret=secret,
             algorithm="sha256",
         ).hexdigest()
 

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_basic.py b/tests/auth_tests/test_basic.py
--- a/tests/auth_tests/test_basic.py
+++ b/tests/auth_tests/test_basic.py
@@ -1,3 +1,4 @@
+from django.conf import settings
 from django.contrib.auth import get_user, get_user_model
 from django.contrib.auth.models import AnonymousUser, User
 from django.core.exceptions import ImproperlyConfigured
@@ -138,3 +139,26 @@ def test_get_user(self):
         user = get_user(request)
         self.assertIsInstance(user, User)
         self.assertEqual(user.username, created_user.username)
+
+    def test_get_user_fallback_secret(self):
+        created_user = User.objects.create_user(
+            "testuser", "test@example.com", "testpw"
+        )
+        self.client.login(username="testuser", password="testpw")
+        request = HttpRequest()
+        request.session = self.client.session
+        prev_session_key = request.session.session_key
+        with override_settings(
+            SECRET_KEY="newsecret",
+            SECRET_KEY_FALLBACKS=[settings.SECRET_KEY],
+        ):
+            user = get_user(request)
+            self.assertIsInstance(user, User)
+            self.assertEqual(user.username, created_user.username)
+            self.assertNotEqual(request.session.session_key, prev_session_key)
+        # Remove the fallback secret.
+        # The session hash should be updated using the current secret.
+        with override_settings(SECRET_KEY="newsecret"):
+            user = get_user(request)
+            self.assertIsInstance(user, User)
+            self.assertEqual(user.username, created_user.username)

```


## Code snippets

### 1 - django/core/checks/security/base.py:

Start line: 226, End line: 239

```python
@register(Tags.security, deploy=True)
def check_secret_key_fallbacks(app_configs, **kwargs):
    warnings = []
    try:
        fallbacks = settings.SECRET_KEY_FALLBACKS
    except (ImproperlyConfigured, AttributeError):
        warnings.append(Warning(W025.msg % "SECRET_KEY_FALLBACKS", id=W025.id))
    else:
        for index, key in enumerate(fallbacks):
            if not _check_secret_key(key):
                warnings.append(
                    Warning(W025.msg % f"SECRET_KEY_FALLBACKS[{index}]", id=W025.id)
                )
    return warnings
```
### 2 - django/contrib/auth/tokens.py:

Start line: 84, End line: 96

```python
class PasswordResetTokenGenerator:

    def _make_token_with_timestamp(self, user, timestamp, secret):
        # timestamp is number of seconds since 2001-1-1. Converted to base 36,
        # this gives us a 6 digit string until about 2069.
        ts_b36 = int_to_base36(timestamp)
        hash_string = salted_hmac(
            self.key_salt,
            self._make_hash_value(user, timestamp),
            secret=secret,
            algorithm=self.algorithm,
        ).hexdigest()[
            ::2
        ]  # Limit to shorten the URL.
        return "%s-%s" % (ts_b36, hash_string)
```
### 3 - django/contrib/sessions/backends/base.py:

Start line: 168, End line: 198

```python
class SessionBase:

    def _set_session_key(self, value):
        """
        Validate session key on assignment. Invalid values will set to None.
        """
        if self._validate_session_key(value):
            self.__session_key = value
        else:
            self.__session_key = None

    session_key = property(_get_session_key)
    _session_key = property(_get_session_key, _set_session_key)

    def _get_session(self, no_load=False):
        """
        Lazily load session from storage (unless "no_load" is True, when only
        an empty dict is stored) and store it in the current instance.
        """
        self.accessed = True
        try:
            return self._session_cache
        except AttributeError:
            if self.session_key is None or no_load:
                self._session_cache = {}
            else:
                self._session_cache = self.load()
        return self._session_cache

    _session = property(_get_session)

    def get_session_cookie_age(self):
        return settings.SESSION_COOKIE_AGE
```
### 4 - django/contrib/sessions/backends/signed_cookies.py:

Start line: 26, End line: 82

```python
class SessionStore(SessionBase):

    def create(self):
        """
        To create a new key, set the modified flag so that the cookie is set
        on the client for the current request.
        """
        self.modified = True

    def save(self, must_create=False):
        """
        To save, get the session key as a securely signed string and then set
        the modified flag so that the cookie is set on the client for the
        current request.
        """
        self._session_key = self._get_session_key()
        self.modified = True

    def exists(self, session_key=None):
        """
        This method makes sense when you're talking to a shared resource, but
        it doesn't matter when you're storing the information in the client's
        cookie.
        """
        return False

    def delete(self, session_key=None):
        """
        To delete, clear the session key and the underlying data structure
        and set the modified flag so that the cookie is set on the client for
        the current request.
        """
        self._session_key = ""
        self._session_cache = {}
        self.modified = True

    def cycle_key(self):
        """
        Keep the same data but with a new key. Call save() and it will
        automatically save a cookie with a new key at the end of the request.
        """
        self.save()

    def _get_session_key(self):
        """
        Instead of generating a random string, generate a secure url-safe
        base64-encoded string of data as our session key.
        """
        return signing.dumps(
            self._session,
            compress=True,
            salt="django.contrib.sessions.backends.signed_cookies",
            serializer=self.serializer,
        )

    @classmethod
    def clear_expired(cls):
        pass
```
### 5 - django/contrib/auth/hashers.py:

Start line: 772, End line: 825

```python
# RemovedInDjango51Warning.
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

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "django.contrib.auth.hashers.UnsaltedMD5PasswordHasher is deprecated.",
            RemovedInDjango51Warning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def salt(self):
        return ""

    def encode(self, password, salt):
        if salt != "":
            raise ValueError("salt must be empty.")
        return hashlib.md5(password.encode()).hexdigest()

    def decode(self, encoded):
        return {
            "algorithm": self.algorithm,
            "hash": encoded,
            "salt": None,
        }

    def verify(self, password, encoded):
        if len(encoded) == 37:
            encoded = encoded.removeprefix("md5$$")
        encoded_2 = self.encode(password, "")
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {
            _("algorithm"): decoded["algorithm"],
            _("hash"): mask_hash(decoded["hash"], show=3),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 6 - django/contrib/sessions/backends/base.py:

Start line: 33, End line: 166

```python
class SessionBase:
    """
    Base class for all Session classes.
    """

    TEST_COOKIE_NAME = "testcookie"
    TEST_COOKIE_VALUE = "worked"

    __not_given = object()

    def __init__(self, session_key=None):
        self._session_key = session_key
        self.accessed = False
        self.modified = False
        self.serializer = import_string(settings.SESSION_SERIALIZER)

    def __contains__(self, key):
        return key in self._session

    def __getitem__(self, key):
        return self._session[key]

    def __setitem__(self, key, value):
        self._session[key] = value
        self.modified = True

    def __delitem__(self, key):
        del self._session[key]
        self.modified = True

    @property
    def key_salt(self):
        return "django.contrib.sessions." + self.__class__.__qualname__

    def get(self, key, default=None):
        return self._session.get(key, default)

    def pop(self, key, default=__not_given):
        self.modified = self.modified or key in self._session
        args = () if default is self.__not_given else (default,)
        return self._session.pop(key, *args)

    def setdefault(self, key, value):
        if key in self._session:
            return self._session[key]
        else:
            self.modified = True
            self._session[key] = value
            return value

    def set_test_cookie(self):
        self[self.TEST_COOKIE_NAME] = self.TEST_COOKIE_VALUE

    def test_cookie_worked(self):
        return self.get(self.TEST_COOKIE_NAME) == self.TEST_COOKIE_VALUE

    def delete_test_cookie(self):
        del self[self.TEST_COOKIE_NAME]

    def encode(self, session_dict):
        "Return the given session dictionary serialized and encoded as a string."
        return signing.dumps(
            session_dict,
            salt=self.key_salt,
            serializer=self.serializer,
            compress=True,
        )

    def decode(self, session_data):
        try:
            return signing.loads(
                session_data, salt=self.key_salt, serializer=self.serializer
            )
        except signing.BadSignature:
            logger = logging.getLogger("django.security.SuspiciousSession")
            logger.warning("Session data corrupted")
        except Exception:
            # ValueError, unpickling exceptions. If any of these happen, just
            # return an empty dictionary (an empty session).
            pass
        return {}

    def update(self, dict_):
        self._session.update(dict_)
        self.modified = True

    def has_key(self, key):
        return key in self._session

    def keys(self):
        return self._session.keys()

    def values(self):
        return self._session.values()

    def items(self):
        return self._session.items()

    def clear(self):
        # To avoid unnecessary persistent storage accesses, we set up the
        # internals directly (loading data wastes time, since we are going to
        # set it to an empty dict anyway).
        self._session_cache = {}
        self.accessed = True
        self.modified = True

    def is_empty(self):
        "Return True when there is no session_key and the session is empty."
        try:
            return not self._session_key and not self._session_cache
        except AttributeError:
            return True

    def _get_new_session_key(self):
        "Return session key that isn't being used."
        while True:
            session_key = get_random_string(32, VALID_KEY_CHARS)
            if not self.exists(session_key):
                return session_key

    def _get_or_create_session_key(self):
        if self._session_key is None:
            self._session_key = self._get_new_session_key()
        return self._session_key

    def _validate_session_key(self, key):
        """
        Key must be truthy and at least 8 characters long. 8 characters is an
        arbitrary lower bound for some minimal key security.
        """
        return key and len(key) >= 8

    def _get_session_key(self):
        return self.__session_key
```
### 7 - django/contrib/auth/hashers.py:

Start line: 719, End line: 769

```python
# RemovedInDjango51Warning.
class UnsaltedSHA1PasswordHasher(BasePasswordHasher):
    """
    Very insecure algorithm that you should *never* use; store SHA1 hashes
    with an empty salt.

    This class is implemented because Django used to accept such password
    hashes. Some older Django installs still have these values lingering
    around so we need to handle and upgrade them properly.
    """

    algorithm = "unsalted_sha1"

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher is deprecated.",
            RemovedInDjango51Warning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def salt(self):
        return ""

    def encode(self, password, salt):
        if salt != "":
            raise ValueError("salt must be empty.")
        hash = hashlib.sha1(password.encode()).hexdigest()
        return "sha1$$%s" % hash

    def decode(self, encoded):
        assert encoded.startswith("sha1$$")
        return {
            "algorithm": self.algorithm,
            "hash": encoded[6:],
            "salt": None,
        }

    def verify(self, password, encoded):
        encoded_2 = self.encode(password, "")
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {
            _("algorithm"): decoded["algorithm"],
            _("hash"): mask_hash(decoded["hash"]),
        }

    def harden_runtime(self, password, encoded):
        pass
```
### 8 - django/contrib/sessions/base_session.py:

Start line: 26, End line: 48

```python
class AbstractBaseSession(models.Model):
    session_key = models.CharField(_("session key"), max_length=40, primary_key=True)
    session_data = models.TextField(_("session data"))
    expire_date = models.DateTimeField(_("expire date"), db_index=True)

    objects = BaseSessionManager()

    class Meta:
        abstract = True
        verbose_name = _("session")
        verbose_name_plural = _("sessions")

    def __str__(self):
        return self.session_key

    @classmethod
    def get_session_store_class(cls):
        raise NotImplementedError

    def get_decoded(self):
        session_store_class = self.get_session_store_class()
        return session_store_class().decode(self.session_data)
```
### 9 - django/contrib/sessions/backends/base.py:

Start line: 1, End line: 30

```python
import logging
import string
from datetime import datetime, timedelta

from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string

# session_key should not be case sensitive because some backends can store it
# on case insensitive file systems.
VALID_KEY_CHARS = string.ascii_lowercase + string.digits


class CreateError(Exception):
    """
    Used internally as a consistent exception type to catch from save (see the
    docstring for SessionBase.save() for details).
    """

    pass


class UpdateError(Exception):
    """
    Occurs if Django tries to update a session that was deleted.
    """

    pass
```
### 10 - django/utils/crypto.py:

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


def salted_hmac(key_salt, value, secret=None, *, algorithm="sha1"):
    """
    Return the HMAC of 'value', using a key generated from key_salt and a
    secret (which defaults to settings.SECRET_KEY). Default algorithm is SHA1,
    but any algorithm name supported by hashlib can be passed.

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
            "%r is not an algorithm accepted by the hashlib module." % algorithm
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
### 15 - django/contrib/auth/__init__.py:

Start line: 1, End line: 38

```python
import inspect
import re

from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.middleware.csrf import rotate_token
from django.utils.crypto import constant_time_compare
from django.utils.module_loading import import_string
from django.views.decorators.debug import sensitive_variables

from .signals import user_logged_in, user_logged_out, user_login_failed

SESSION_KEY = "_auth_user_id"
BACKEND_SESSION_KEY = "_auth_user_backend"
HASH_SESSION_KEY = "_auth_user_hash"
REDIRECT_FIELD_NAME = "next"


def load_backend(path):
    return import_string(path)()


def _get_backends(return_tuples=False):
    backends = []
    for backend_path in settings.AUTHENTICATION_BACKENDS:
        backend = load_backend(backend_path)
        backends.append((backend, backend_path) if return_tuples else backend)
    if not backends:
        raise ImproperlyConfigured(
            "No authentication backends have been defined. Does "
            "AUTHENTICATION_BACKENDS contain anything?"
        )
    return backends


def get_backends():
    return _get_backends(return_tuples=False)
```
### 32 - django/contrib/auth/base_user.py:

Start line: 56, End line: 159

```python
class AbstractBaseUser(models.Model):
    password = models.CharField(_("password"), max_length=128)
    last_login = models.DateTimeField(_("last login"), blank=True, null=True)

    is_active = True

    REQUIRED_FIELDS = []

    # Stores the raw password if set_password() is called so that it can
    # be passed to password_changed() after the model is saved.
    _password = None

    class Meta:
        abstract = True

    def __str__(self):
        return self.get_username()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self._password is not None:
            password_validation.password_changed(self._password, self)
            self._password = None

    def get_username(self):
        """Return the username for this User."""
        return getattr(self, self.USERNAME_FIELD)

    def clean(self):
        setattr(self, self.USERNAME_FIELD, self.normalize_username(self.get_username()))

    def natural_key(self):
        return (self.get_username(),)

    @property
    def is_anonymous(self):
        """
        Always return False. This is a way of comparing User objects to
        anonymous users.
        """
        return False

    @property
    def is_authenticated(self):
        """
        Always return True. This is a way to tell if the user has been
        authenticated in templates.
        """
        return True

    def set_password(self, raw_password):
        self.password = make_password(raw_password)
        self._password = raw_password

    def check_password(self, raw_password):
        """
        Return a boolean of whether the raw_password was correct. Handles
        hashing formats behind the scenes.
        """

        def setter(raw_password):
            self.set_password(raw_password)
            # Password hash upgrades shouldn't be considered password changes.
            self._password = None
            self.save(update_fields=["password"])

        return check_password(raw_password, self.password, setter)

    def set_unusable_password(self):
        # Set a value that will never be a valid hash
        self.password = make_password(None)

    def has_usable_password(self):
        """
        Return False if set_unusable_password() has been called for this user.
        """
        return is_password_usable(self.password)

    def get_session_auth_hash(self):
        """
        Return an HMAC of the password field.
        """
        key_salt = "django.contrib.auth.models.AbstractBaseUser.get_session_auth_hash"
        return salted_hmac(
            key_salt,
            self.password,
            algorithm="sha256",
        ).hexdigest()

    @classmethod
    def get_email_field_name(cls):
        try:
            return cls.EMAIL_FIELD
        except AttributeError:
            return "email"

    @classmethod
    def normalize_username(cls, username):
        return (
            unicodedata.normalize("NFKC", username)
            if isinstance(username, str)
            else username
        )
```
### 35 - django/contrib/auth/__init__.py:

Start line: 41, End line: 60

```python
@sensitive_variables("credentials")
def _clean_credentials(credentials):
    """
    Clean a dictionary of credentials of potentially sensitive info before
    sending to less secure functions.

    Not comprehensive - intended for user_login_failed signal
    """
    SENSITIVE_CREDENTIALS = re.compile("api|token|key|secret|password|signature", re.I)
    CLEANSED_SUBSTITUTE = "********************"
    for key in credentials:
        if SENSITIVE_CREDENTIALS.search(key):
            credentials[key] = CLEANSED_SUBSTITUTE
    return credentials


def _get_user_session_key(request):
    # This value in the session is always serialized to a string, so we need
    # to convert it back to Python whenever we access it.
    return get_user_model()._meta.pk.to_python(request.session[SESSION_KEY])
```
### 76 - django/contrib/auth/__init__.py:

Start line: 94, End line: 144

```python
def login(request, user, backend=None):
    """
    Persist a user id and a backend in the request. This way a user doesn't
    have to reauthenticate on every request. Note that data set during
    the anonymous session is retained when the user logs in.
    """
    session_auth_hash = ""
    if user is None:
        user = request.user
    if hasattr(user, "get_session_auth_hash"):
        session_auth_hash = user.get_session_auth_hash()

    if SESSION_KEY in request.session:
        if _get_user_session_key(request) != user.pk or (
            session_auth_hash
            and not constant_time_compare(
                request.session.get(HASH_SESSION_KEY, ""), session_auth_hash
            )
        ):
            # To avoid reusing another user's session, create a new, empty
            # session if the existing session corresponds to a different
            # authenticated user.
            request.session.flush()
    else:
        request.session.cycle_key()

    try:
        backend = backend or user.backend
    except AttributeError:
        backends = _get_backends(return_tuples=True)
        if len(backends) == 1:
            _, backend = backends[0]
        else:
            raise ValueError(
                "You have multiple authentication backends configured and "
                "therefore must provide the `backend` argument or set the "
                "`backend` attribute on the user."
            )
    else:
        if not isinstance(backend, str):
            raise TypeError(
                "backend must be a dotted import path string (got %r)." % backend
            )

    request.session[SESSION_KEY] = user._meta.pk.value_to_string(user)
    request.session[BACKEND_SESSION_KEY] = backend
    request.session[HASH_SESSION_KEY] = session_auth_hash
    if hasattr(request, "user"):
        request.user = user
    rotate_token(request)
    user_logged_in.send(sender=user.__class__, request=request, user=user)
```
### 80 - django/contrib/auth/base_user.py:

Start line: 1, End line: 53

```python
"""
This module allows importing AbstractBaseUser even when django.contrib.auth is
not in INSTALLED_APPS.
"""
import unicodedata
import warnings

from django.contrib.auth import password_validation
from django.contrib.auth.hashers import (
    check_password,
    is_password_usable,
    make_password,
)
from django.db import models
from django.utils.crypto import get_random_string, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _


class BaseUserManager(models.Manager):
    @classmethod
    def normalize_email(cls, email):
        """
        Normalize the email address by lowercasing the domain part of it.
        """
        email = email or ""
        try:
            email_name, domain_part = email.strip().rsplit("@", 1)
        except ValueError:
            pass
        else:
            email = email_name + "@" + domain_part.lower()
        return email

    def make_random_password(
        self,
        length=10,
        allowed_chars="abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789",
    ):
        """
        Generate a random password with the given length and given
        allowed_chars. The default value of allowed_chars does not have "I" or
        "O" or letters and digits that look similar -- just to avoid confusion.
        """
        warnings.warn(
            "BaseUserManager.make_random_password() is deprecated.",
            category=RemovedInDjango51Warning,
            stacklevel=2,
        )
        return get_random_string(length, allowed_chars)

    def get_by_natural_key(self, username):
        return self.get(**{self.model.USERNAME_FIELD: username})
```
### 119 - django/contrib/auth/__init__.py:

Start line: 182, End line: 209

```python
def get_user(request):
    """
    Return the user model instance associated with the given request session.
    If no user is retrieved, return an instance of `AnonymousUser`.
    """
    from .models import AnonymousUser

    user = None
    try:
        user_id = _get_user_session_key(request)
        backend_path = request.session[BACKEND_SESSION_KEY]
    except KeyError:
        pass
    else:
        if backend_path in settings.AUTHENTICATION_BACKENDS:
            backend = load_backend(backend_path)
            user = backend.get_user(user_id)
            # Verify the session
            if hasattr(user, "get_session_auth_hash"):
                session_hash = request.session.get(HASH_SESSION_KEY)
                session_hash_verified = session_hash and constant_time_compare(
                    session_hash, user.get_session_auth_hash()
                )
                if not session_hash_verified:
                    request.session.flush()
                    user = None

    return user or AnonymousUser()
```
### 141 - django/contrib/auth/__init__.py:

Start line: 212, End line: 231

```python
def get_permission_codename(action, opts):
    """
    Return the codename of the permission for the specified action.
    """
    return "%s_%s" % (action, opts.model_name)


def update_session_auth_hash(request, user):
    """
    Updating a user's password logs out all sessions for the user.

    Take the current request and the updated user object from which the new
    session hash will be derived and update the session hash appropriately to
    prevent a password change from logging out the session from which the
    password was changed.
    """
    request.session.cycle_key()
    if hasattr(user, "get_session_auth_hash") and request.user == user:
        request.session[HASH_SESSION_KEY] = user.get_session_auth_hash()
```
### 146 - django/contrib/auth/__init__.py:

Start line: 147, End line: 162

```python
def logout(request):
    """
    Remove the authenticated user's ID from the request and flush their session
    data.
    """
    # Dispatch the signal before the user is logged out so the receivers have a
    # chance to find out *who* logged out.
    user = getattr(request, "user", None)
    if not getattr(user, "is_authenticated", True):
        user = None
    user_logged_out.send(sender=user.__class__, request=request, user=user)
    request.session.flush()
    if hasattr(request, "user"):
        from django.contrib.auth.models import AnonymousUser

        request.user = AnonymousUser()
```
