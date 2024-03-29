# django__django-12198

| **django/django** | `d6505273cd889886caca57884fa79941b18c2ea6` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 21894 |
| **Any found context length** | 21894 |
| **Avg pos** | 48.333333333333336 |
| **Min pos** | 72 |
| **Max pos** | 73 |
| **Top file pos** | 3 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/auth/__init__.py b/django/contrib/auth/__init__.py
--- a/django/contrib/auth/__init__.py
+++ b/django/contrib/auth/__init__.py
@@ -63,8 +63,9 @@ def authenticate(request=None, **credentials):
     If the given credentials are valid, return a User object.
     """
     for backend, backend_path in _get_backends(return_tuples=True):
+        backend_signature = inspect.signature(backend.authenticate)
         try:
-            inspect.getcallargs(backend.authenticate, request, **credentials)
+            backend_signature.bind(request, **credentials)
         except TypeError:
             # This backend doesn't accept these credentials as arguments. Try the next one.
             continue
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1916,9 +1916,8 @@ def set_group_by(self):
         group_by = list(self.select)
         if self.annotation_select:
             for alias, annotation in self.annotation_select.items():
-                try:
-                    inspect.getcallargs(annotation.get_group_by_cols, alias=alias)
-                except TypeError:
+                signature = inspect.signature(annotation.get_group_by_cols)
+                if 'alias' not in signature.parameters:
                     annotation_class = annotation.__class__
                     msg = (
                         '`alias=None` must be added to the signature of '
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -50,10 +50,10 @@
 '<html></html>'
 """
 
+import inspect
 import logging
 import re
 from enum import Enum
-from inspect import getcallargs, getfullargspec, unwrap
 
 from django.template.context import BaseContext
 from django.utils.formats import localize
@@ -707,9 +707,9 @@ def args_check(name, func, provided):
         # First argument, filter input, is implied.
         plen = len(provided) + 1
         # Check to see if a decorator is providing the real function.
-        func = unwrap(func)
+        func = inspect.unwrap(func)
 
-        args, _, _, defaults, _, _, _ = getfullargspec(func)
+        args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
         alen = len(args)
         dlen = len(defaults or [])
         # Not enough OR Too many
@@ -857,8 +857,9 @@ def _resolve_lookup(self, context):
                         try:  # method call (assuming no args required)
                             current = current()
                         except TypeError:
+                            signature = inspect.signature(current)
                             try:
-                                getcallargs(current)
+                                signature.bind()
                             except TypeError:  # arguments *were* required
                                 current = context.template.engine.string_if_invalid  # invalid method call
                             else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/__init__.py | 66 | 66 | 72 | 3 | 21894
| django/db/models/sql/query.py | 1919 | 1921 | - | 71 | -
| django/template/base.py | 53 | 53 | - | 47 | -
| django/template/base.py | 710 | 712 | - | 47 | -
| django/template/base.py | 860 | 860 | 73 | 47 | 22429


## Problem Statement

```
Allow sensitive_variables() to preserve the signature of its decorated function
Description
	
When the method authenticate of a custom AuthenticationBackend is decorated with sensitive_variables, inspect.getcallargs will always match.
Calling the authenticate function will attempt to call this backend with any set of credentials and will raise an uncaught TypeError for an unmatching backend.
Authentication with such decorated backends used to work in version 1.6.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/views/decorators/debug.py | 1 | 38| 218 | 218 | 473 | 
| 2 | 1 django/views/decorators/debug.py | 41 | 63| 139 | 357 | 473 | 
| 3 | 2 django/views/debug.py | 194 | 242| 462 | 819 | 4751 | 
| 4 | **3 django/contrib/auth/__init__.py** | 1 | 58| 393 | 1212 | 6318 | 
| 5 | 3 django/views/decorators/debug.py | 64 | 79| 126 | 1338 | 6318 | 
| 6 | 3 django/views/debug.py | 126 | 153| 248 | 1586 | 6318 | 
| 7 | 4 django/contrib/auth/backends.py | 31 | 49| 142 | 1728 | 8080 | 
| 8 | **4 django/contrib/auth/__init__.py** | 86 | 131| 392 | 2120 | 8080 | 
| 9 | 4 django/views/debug.py | 155 | 178| 176 | 2296 | 8080 | 
| 10 | 5 django/db/backends/base/features.py | 117 | 215| 844 | 3140 | 10659 | 
| 11 | 6 django/core/signing.py | 1 | 78| 732 | 3872 | 12378 | 
| 12 | 7 django/contrib/auth/middleware.py | 1 | 23| 171 | 4043 | 13372 | 
| 13 | 7 django/contrib/auth/backends.py | 163 | 181| 146 | 4189 | 13372 | 
| 14 | 8 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 4436 | 13619 | 
| 15 | 8 django/contrib/auth/backends.py | 183 | 234| 369 | 4805 | 13619 | 
| 16 | 9 django/contrib/auth/mixins.py | 47 | 85| 277 | 5082 | 14349 | 
| 17 | 9 django/contrib/auth/middleware.py | 84 | 109| 192 | 5274 | 14349 | 
| 18 | 10 django/db/backends/oracle/base.py | 367 | 388| 143 | 5417 | 19464 | 
| 19 | 11 django/utils/asyncio.py | 1 | 35| 222 | 5639 | 19687 | 
| 20 | 12 django/core/checks/security/sessions.py | 1 | 98| 572 | 6211 | 20260 | 
| 21 | 12 django/contrib/auth/backends.py | 51 | 65| 143 | 6354 | 20260 | 
| 22 | 13 django/db/backends/base/validation.py | 1 | 26| 192 | 6546 | 20453 | 
| 23 | 13 django/db/backends/base/features.py | 217 | 303| 681 | 7227 | 20453 | 
| 24 | 14 django/core/checks/security/base.py | 1 | 83| 732 | 7959 | 22234 | 
| 25 | 15 django/contrib/sessions/backends/base.py | 1 | 35| 202 | 8161 | 24765 | 
| 26 | 16 django/conf/global_settings.py | 497 | 641| 900 | 9061 | 30404 | 
| 27 | 17 django/db/backends/oracle/features.py | 1 | 62| 504 | 9565 | 30909 | 
| 28 | 17 django/contrib/auth/mixins.py | 1 | 44| 307 | 9872 | 30909 | 
| 29 | 18 django/contrib/auth/models.py | 159 | 183| 192 | 10064 | 34098 | 
| 30 | 19 django/contrib/auth/password_validation.py | 1 | 32| 206 | 10270 | 35584 | 
| 31 | 20 django/http/request.py | 147 | 179| 235 | 10505 | 40424 | 
| 32 | 21 django/contrib/sessions/backends/signed_cookies.py | 1 | 25| 160 | 10665 | 40945 | 
| 33 | 21 django/views/debug.py | 73 | 96| 196 | 10861 | 40945 | 
| 34 | 21 django/core/checks/security/base.py | 183 | 210| 208 | 11069 | 40945 | 
| 35 | 22 django/views/decorators/clickjacking.py | 22 | 54| 238 | 11307 | 41321 | 
| 36 | 22 django/contrib/auth/backends.py | 1 | 28| 171 | 11478 | 41321 | 
| 37 | 22 django/contrib/sessions/backends/base.py | 109 | 125| 173 | 11651 | 41321 | 
| 38 | 22 django/db/backends/base/features.py | 1 | 115| 904 | 12555 | 41321 | 
| 39 | 23 django/contrib/auth/decorators.py | 1 | 35| 313 | 12868 | 41908 | 
| 40 | 24 django/db/backends/mysql/validation.py | 1 | 27| 248 | 13116 | 42396 | 
| 41 | 25 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 13253 | 42533 | 
| 42 | 26 django/contrib/auth/admin.py | 1 | 22| 188 | 13441 | 44259 | 
| 43 | 27 django/views/decorators/csrf.py | 1 | 57| 460 | 13901 | 44719 | 
| 44 | 28 django/contrib/auth/checks.py | 1 | 94| 646 | 14547 | 45892 | 
| 45 | 28 django/core/signing.py | 145 | 170| 255 | 14802 | 45892 | 
| 46 | 29 django/core/checks/security/csrf.py | 1 | 41| 299 | 15101 | 46191 | 
| 47 | 29 django/contrib/auth/decorators.py | 38 | 74| 273 | 15374 | 46191 | 
| 48 | 30 django/contrib/admin/views/decorators.py | 1 | 19| 135 | 15509 | 46327 | 
| 49 | 31 django/contrib/auth/hashers.py | 1 | 27| 187 | 15696 | 51114 | 
| 50 | 32 django/db/backends/utils.py | 49 | 65| 176 | 15872 | 53011 | 
| 51 | 33 django/db/models/fields/__init__.py | 338 | 362| 184 | 16056 | 70523 | 
| 52 | 34 django/db/backends/oracle/utils.py | 1 | 38| 250 | 16306 | 71052 | 
| 53 | 35 django/contrib/auth/apps.py | 1 | 29| 213 | 16519 | 71265 | 
| 54 | 36 django/contrib/auth/views.py | 1 | 37| 278 | 16797 | 73929 | 
| 55 | 36 django/contrib/auth/hashers.py | 418 | 447| 290 | 17087 | 73929 | 
| 56 | 37 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 17237 | 74079 | 
| 57 | 38 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 17490 | 74500 | 
| 58 | 39 django/contrib/sessions/middleware.py | 1 | 75| 576 | 18066 | 75077 | 
| 59 | 39 django/contrib/auth/middleware.py | 46 | 82| 360 | 18426 | 75077 | 
| 60 | 40 django/middleware/csrf.py | 1 | 42| 330 | 18756 | 77931 | 
| 61 | 41 django/contrib/admin/forms.py | 1 | 31| 184 | 18940 | 78115 | 
| 62 | 41 django/contrib/auth/hashers.py | 564 | 597| 242 | 19182 | 78115 | 
| 63 | 42 django/db/utils.py | 52 | 98| 312 | 19494 | 80261 | 
| 64 | 43 django/db/backends/sqlite3/base.py | 1 | 77| 534 | 20028 | 86025 | 
| 65 | 44 django/contrib/auth/forms.py | 213 | 238| 176 | 20204 | 89096 | 
| 66 | 44 django/contrib/sessions/backends/base.py | 127 | 210| 563 | 20767 | 89096 | 
| 67 | 45 django/contrib/auth/context_processors.py | 24 | 64| 247 | 21014 | 89510 | 
| 68 | 45 django/contrib/auth/hashers.py | 80 | 102| 167 | 21181 | 89510 | 
| 69 | 45 django/contrib/auth/hashers.py | 328 | 341| 120 | 21301 | 89510 | 
| 70 | 46 django/contrib/admin/tests.py | 1 | 36| 264 | 21565 | 90987 | 
| 71 | 46 django/views/decorators/clickjacking.py | 1 | 19| 138 | 21703 | 90987 | 
| **-> 72 <-** | **46 django/contrib/auth/__init__.py** | 61 | 83| 191 | 21894 | 90987 | 
| **-> 73 <-** | **47 django/template/base.py** | 816 | 880| 535 | 22429 | 98872 | 
| 74 | 47 django/core/checks/security/base.py | 85 | 180| 710 | 23139 | 98872 | 
| 75 | 48 django/core/checks/model_checks.py | 155 | 176| 263 | 23402 | 100659 | 
| 76 | 48 django/contrib/auth/admin.py | 128 | 189| 465 | 23867 | 100659 | 
| 77 | 49 django/views/decorators/vary.py | 1 | 42| 232 | 24099 | 100892 | 
| 78 | 49 django/middleware/csrf.py | 93 | 119| 224 | 24323 | 100892 | 
| 79 | 49 django/views/debug.py | 49 | 70| 160 | 24483 | 100892 | 
| 80 | 49 django/contrib/auth/hashers.py | 529 | 561| 220 | 24703 | 100892 | 
| 81 | 49 django/core/signing.py | 126 | 142| 170 | 24873 | 100892 | 
| 82 | 50 django/db/backends/base/base.py | 423 | 500| 525 | 25398 | 105746 | 
| 83 | 50 django/contrib/auth/models.py | 186 | 223| 249 | 25647 | 105746 | 
| 84 | 51 django/utils/inspect.py | 36 | 64| 179 | 25826 | 106142 | 
| 85 | 51 django/db/utils.py | 207 | 237| 194 | 26020 | 106142 | 
| 86 | 51 django/middleware/csrf.py | 45 | 54| 112 | 26132 | 106142 | 
| 87 | 51 django/contrib/auth/hashers.py | 30 | 61| 246 | 26378 | 106142 | 
| 88 | 51 django/conf/global_settings.py | 146 | 261| 854 | 27232 | 106142 | 
| 89 | 52 django/db/backends/base/operations.py | 465 | 482| 163 | 27395 | 111633 | 
| 90 | 52 django/contrib/auth/hashers.py | 343 | 358| 146 | 27541 | 111633 | 
| 91 | 53 django/db/backends/oracle/introspection.py | 108 | 141| 286 | 27827 | 113992 | 
| 92 | 53 django/db/backends/base/operations.py | 1 | 100| 829 | 28656 | 113992 | 
| 93 | 54 django/contrib/gis/db/models/functions.py | 55 | 85| 275 | 28931 | 117900 | 
| 94 | 55 django/template/context_processors.py | 35 | 49| 126 | 29057 | 118389 | 
| 95 | 55 django/contrib/auth/password_validation.py | 135 | 157| 197 | 29254 | 118389 | 
| 96 | 56 django/views/decorators/http.py | 1 | 52| 350 | 29604 | 119341 | 
| 97 | 57 django/db/backends/postgresql/features.py | 1 | 75| 615 | 30219 | 119956 | 
| 98 | 58 django/db/backends/mysql/base.py | 194 | 229| 330 | 30549 | 123079 | 
| 99 | 58 django/contrib/auth/hashers.py | 229 | 273| 414 | 30963 | 123079 | 
| 100 | 59 django/db/models/functions/text.py | 24 | 55| 217 | 31180 | 125535 | 
| 101 | 60 django/core/cache/backends/base.py | 1 | 47| 245 | 31425 | 127692 | 
| 102 | 61 django/contrib/gis/gdal/prototypes/errcheck.py | 66 | 139| 536 | 31961 | 128675 | 
| 103 | 61 django/contrib/admin/tests.py | 157 | 171| 160 | 32121 | 128675 | 
| 104 | 61 django/contrib/auth/middleware.py | 26 | 44| 178 | 32299 | 128675 | 
| 105 | 61 django/middleware/csrf.py | 205 | 327| 1189 | 33488 | 128675 | 
| 106 | 61 django/db/backends/utils.py | 1 | 47| 287 | 33775 | 128675 | 
| 107 | 61 django/db/backends/utils.py | 216 | 253| 279 | 34054 | 128675 | 
| 108 | 62 django/db/backends/base/introspection.py | 1 | 33| 224 | 34278 | 130045 | 
| 109 | 62 django/db/utils.py | 150 | 164| 145 | 34423 | 130045 | 
| 110 | 63 django/utils/crypto.py | 1 | 34| 279 | 34702 | 130553 | 
| 111 | 63 django/middleware/csrf.py | 57 | 71| 156 | 34858 | 130553 | 
| 112 | 63 django/contrib/auth/hashers.py | 388 | 404| 127 | 34985 | 130553 | 
| 113 | 64 django/contrib/admin/utils.py | 287 | 305| 175 | 35160 | 134670 | 
| 114 | 64 django/db/backends/sqlite3/base.py | 202 | 244| 783 | 35943 | 134670 | 
| 115 | 64 django/contrib/auth/middleware.py | 112 | 123| 107 | 36050 | 134670 | 
| 116 | 65 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 36188 | 134808 | 
| 117 | 66 django/db/backends/mysql/introspection.py | 16 | 38| 218 | 36406 | 137038 | 
| 118 | 66 django/contrib/gis/db/models/functions.py | 18 | 53| 312 | 36718 | 137038 | 
| 119 | 66 django/contrib/sessions/backends/signed_cookies.py | 27 | 82| 366 | 37084 | 137038 | 
| 120 | 67 django/db/backends/mysql/features.py | 103 | 156| 497 | 37581 | 138363 | 
| 121 | 68 django/utils/cache.py | 1 | 35| 284 | 37865 | 142111 | 
| 122 | 69 django/utils/safestring.py | 40 | 64| 159 | 38024 | 142497 | 
| 123 | 69 django/contrib/auth/forms.py | 1 | 20| 137 | 38161 | 142497 | 
| 124 | 70 django/db/backends/signals.py | 1 | 4| 0 | 38161 | 142514 | 
| 125 | 70 django/core/cache/backends/base.py | 239 | 256| 165 | 38326 | 142514 | 
| 126 | **70 django/template/base.py** | 792 | 814| 190 | 38516 | 142514 | 
| 127 | 70 django/db/backends/utils.py | 94 | 131| 297 | 38813 | 142514 | 
| 128 | 70 django/conf/global_settings.py | 396 | 495| 792 | 39605 | 142514 | 
| 129 | 70 django/core/checks/security/base.py | 213 | 226| 131 | 39736 | 142514 | 
| 130 | 70 django/contrib/auth/hashers.py | 128 | 146| 189 | 39925 | 142514 | 
| 131 | **71 django/db/models/sql/query.py** | 2049 | 2074| 214 | 40139 | 163974 | 
| 132 | 71 django/contrib/admin/tests.py | 126 | 137| 153 | 40292 | 163974 | 
| 133 | 71 django/contrib/auth/hashers.py | 450 | 464| 126 | 40418 | 163974 | 
| 134 | 71 django/db/backends/mysql/base.py | 1 | 49| 457 | 40875 | 163974 | 
| 135 | 72 django/middleware/security.py | 31 | 56| 234 | 41109 | 164477 | 
| 136 | 73 django/contrib/sessions/models.py | 1 | 36| 250 | 41359 | 164727 | 
| 137 | 74 django/contrib/auth/signals.py | 1 | 6| 0 | 41359 | 164778 | 
| 138 | 74 django/db/backends/base/base.py | 302 | 320| 148 | 41507 | 164778 | 
| 139 | 74 django/contrib/gis/gdal/prototypes/errcheck.py | 1 | 35| 221 | 41728 | 164778 | 
| 140 | 75 django/dispatch/dispatcher.py | 272 | 293| 139 | 41867 | 166880 | 
| 141 | 76 django/utils/deprecation.py | 73 | 95| 158 | 42025 | 167558 | 
| 142 | 77 django/db/backends/mysql/operations.py | 193 | 236| 329 | 42354 | 170879 | 
| 143 | 77 django/contrib/auth/hashers.py | 406 | 416| 127 | 42481 | 170879 | 
| 144 | 78 django/utils/decorators.py | 22 | 50| 297 | 42778 | 172138 | 
| 145 | 79 django/contrib/auth/validators.py | 1 | 26| 165 | 42943 | 172304 | 
| 146 | 79 django/contrib/admin/tests.py | 173 | 194| 205 | 43148 | 172304 | 
| 147 | 79 django/views/decorators/http.py | 55 | 76| 272 | 43420 | 172304 | 
| 148 | 79 django/db/backends/base/base.py | 560 | 607| 300 | 43720 | 172304 | 
| 149 | 79 django/contrib/auth/hashers.py | 467 | 495| 220 | 43940 | 172304 | 
| 150 | 80 django/contrib/postgres/signals.py | 37 | 65| 257 | 44197 | 172805 | 
| 151 | 80 django/contrib/auth/password_validation.py | 118 | 133| 154 | 44351 | 172805 | 
| 152 | 80 django/contrib/gis/db/models/functions.py | 88 | 119| 231 | 44582 | 172805 | 
| 153 | 81 django/db/backends/dummy/base.py | 1 | 47| 270 | 44852 | 173250 | 
| 154 | 81 django/contrib/auth/password_validation.py | 91 | 115| 189 | 45041 | 173250 | 
| 155 | 81 django/db/backends/utils.py | 67 | 91| 247 | 45288 | 173250 | 
| 156 | 82 django/db/backends/sqlite3/operations.py | 118 | 143| 279 | 45567 | 176138 | 
| 157 | 82 django/db/backends/mysql/features.py | 1 | 101| 834 | 46401 | 176138 | 
| 158 | 82 django/contrib/auth/hashers.py | 498 | 526| 222 | 46623 | 176138 | 
| 159 | 83 django/db/migrations/autodetector.py | 1146 | 1180| 296 | 46919 | 187873 | 
| 160 | 84 django/utils/http.py | 1 | 73| 714 | 47633 | 192051 | 
| 161 | 84 django/middleware/csrf.py | 122 | 156| 267 | 47900 | 192051 | 
| 162 | 84 django/contrib/auth/forms.py | 44 | 69| 162 | 48062 | 192051 | 
| 163 | 84 django/utils/decorators.py | 114 | 153| 313 | 48375 | 192051 | 
| 164 | 84 django/contrib/gis/db/models/functions.py | 455 | 483| 225 | 48600 | 192051 | 
| 165 | 84 django/contrib/auth/password_validation.py | 35 | 51| 114 | 48714 | 192051 | 


### Hint

```
Could you please try ​bisecting to find the commit where the behavior changed?
It is commit b89c2a5d9eb70ca36629ef657c98e3371e9a5c4f.
Thanks! I'm not sure what can be done to fix this. Any ideas?
Nothing apart from going back to the previous masking of TypeError... I think that these two behaviours go against each other...
It might be possible to allow sensitive_variables to preserve the signature of whatever it decorates. Here's code that works until @sensitive_variables is added: import inspect from django.views.decorators.debug import sensitive_variables class Backend(object): @sensitive_variables def authenticate(self, username=None, password=None): print(username) inspect.getcallargs(Backend().authenticate, username='foo', password='bar')
What about something like this: def sensitive_variables(*variables): def decorator(func): @functools.wraps(func) def sensitive_variables_wrapper(*func_args, **func_kwargs): ... # Keep the original function for inspection in `authenticate` sensitive_variables_wrapper.sensitive_variables_func = func return sensitive_variables_wrapper return decorator Function authenticate would then check the sensitive_variables_func first.
```

## Patch

```diff
diff --git a/django/contrib/auth/__init__.py b/django/contrib/auth/__init__.py
--- a/django/contrib/auth/__init__.py
+++ b/django/contrib/auth/__init__.py
@@ -63,8 +63,9 @@ def authenticate(request=None, **credentials):
     If the given credentials are valid, return a User object.
     """
     for backend, backend_path in _get_backends(return_tuples=True):
+        backend_signature = inspect.signature(backend.authenticate)
         try:
-            inspect.getcallargs(backend.authenticate, request, **credentials)
+            backend_signature.bind(request, **credentials)
         except TypeError:
             # This backend doesn't accept these credentials as arguments. Try the next one.
             continue
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1916,9 +1916,8 @@ def set_group_by(self):
         group_by = list(self.select)
         if self.annotation_select:
             for alias, annotation in self.annotation_select.items():
-                try:
-                    inspect.getcallargs(annotation.get_group_by_cols, alias=alias)
-                except TypeError:
+                signature = inspect.signature(annotation.get_group_by_cols)
+                if 'alias' not in signature.parameters:
                     annotation_class = annotation.__class__
                     msg = (
                         '`alias=None` must be added to the signature of '
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -50,10 +50,10 @@
 '<html></html>'
 """
 
+import inspect
 import logging
 import re
 from enum import Enum
-from inspect import getcallargs, getfullargspec, unwrap
 
 from django.template.context import BaseContext
 from django.utils.formats import localize
@@ -707,9 +707,9 @@ def args_check(name, func, provided):
         # First argument, filter input, is implied.
         plen = len(provided) + 1
         # Check to see if a decorator is providing the real function.
-        func = unwrap(func)
+        func = inspect.unwrap(func)
 
-        args, _, _, defaults, _, _, _ = getfullargspec(func)
+        args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
         alen = len(args)
         dlen = len(defaults or [])
         # Not enough OR Too many
@@ -857,8 +857,9 @@ def _resolve_lookup(self, context):
                         try:  # method call (assuming no args required)
                             current = current()
                         except TypeError:
+                            signature = inspect.signature(current)
                             try:
-                                getcallargs(current)
+                                signature.bind()
                             except TypeError:  # arguments *were* required
                                 current = context.template.engine.string_if_invalid  # invalid method call
                             else:

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_auth_backends.py b/tests/auth_tests/test_auth_backends.py
--- a/tests/auth_tests/test_auth_backends.py
+++ b/tests/auth_tests/test_auth_backends.py
@@ -13,6 +13,7 @@
 from django.test import (
     SimpleTestCase, TestCase, modify_settings, override_settings,
 )
+from django.views.decorators.debug import sensitive_variables
 
 from .models import (
     CustomPermissionsUser, CustomUser, CustomUserWithoutIsActiveField,
@@ -642,6 +643,12 @@ def authenticate(self):
         pass
 
 
+class SkippedBackendWithDecoratedMethod:
+    @sensitive_variables()
+    def authenticate(self):
+        pass
+
+
 class AuthenticateTests(TestCase):
     @classmethod
     def setUpTestData(cls):
@@ -664,6 +671,13 @@ def test_skips_backends_without_arguments(self):
         """
         self.assertEqual(authenticate(username='test', password='test'), self.user1)
 
+    @override_settings(AUTHENTICATION_BACKENDS=(
+        'auth_tests.test_auth_backends.SkippedBackendWithDecoratedMethod',
+        'django.contrib.auth.backends.ModelBackend',
+    ))
+    def test_skips_backends_with_decorated_method(self):
+        self.assertEqual(authenticate(username='test', password='test'), self.user1)
+
 
 class ImproperlyConfiguredUserModelTest(TestCase):
     """

```


## Code snippets

### 1 - django/views/decorators/debug.py:

Start line: 1, End line: 38

```python
import functools

from django.http import HttpRequest


def sensitive_variables(*variables):
    """
    Indicate which variables used in the decorated function are sensitive so
    that those variables can later be treated in a special way, for example
    by hiding them when logging unhandled exceptions.

    Accept two forms:

    * with specified variable names:

        @sensitive_variables('user', 'password', 'credit_card')
        def my_function(user):
            password = user.pass_word
            credit_card = user.credit_card_number
            ...

    * without any specified variable names, in which case consider all
      variables are sensitive:

        @sensitive_variables()
        def my_function()
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def sensitive_variables_wrapper(*func_args, **func_kwargs):
            if variables:
                sensitive_variables_wrapper.sensitive_variables = variables
            else:
                sensitive_variables_wrapper.sensitive_variables = '__ALL__'
            return func(*func_args, **func_kwargs)
        return sensitive_variables_wrapper
    return decorator
```
### 2 - django/views/decorators/debug.py:

Start line: 41, End line: 63

```python
def sensitive_post_parameters(*parameters):
    """
    Indicate which POST parameters used in the decorated view are sensitive,
    so that those parameters can later be treated in a special way, for example
    by hiding them when logging unhandled exceptions.

    Accept two forms:

    * with specified parameters:

        @sensitive_post_parameters('password', 'credit_card')
        def my_view(request):
            pw = request.POST['password']
            cc = request.POST['credit_card']
            ...

    * without any specified parameters, in which case consider all
      variables are sensitive:

        @sensitive_post_parameters()
        def my_view(request)
            ...
    """
    # ... other code
```
### 3 - django/views/debug.py:

Start line: 194, End line: 242

```python
class SafeExceptionReporterFilter(ExceptionReporterFilter):

    def get_traceback_frame_variables(self, request, tb_frame):
        """
        Replace the values of variables marked as sensitive with
        stars (*********).
        """
        # Loop through the frame's callers to see if the sensitive_variables
        # decorator was used.
        current_frame = tb_frame.f_back
        sensitive_variables = None
        while current_frame is not None:
            if (current_frame.f_code.co_name == 'sensitive_variables_wrapper' and
                    'sensitive_variables_wrapper' in current_frame.f_locals):
                # The sensitive_variables decorator was used, so we take note
                # of the sensitive variables' names.
                wrapper = current_frame.f_locals['sensitive_variables_wrapper']
                sensitive_variables = getattr(wrapper, 'sensitive_variables', None)
                break
            current_frame = current_frame.f_back

        cleansed = {}
        if self.is_active(request) and sensitive_variables:
            if sensitive_variables == '__ALL__':
                # Cleanse all variables
                for name in tb_frame.f_locals:
                    cleansed[name] = CLEANSED_SUBSTITUTE
            else:
                # Cleanse specified variables
                for name, value in tb_frame.f_locals.items():
                    if name in sensitive_variables:
                        value = CLEANSED_SUBSTITUTE
                    else:
                        value = self.cleanse_special_types(request, value)
                    cleansed[name] = value
        else:
            # Potentially cleanse the request and any MultiValueDicts if they
            # are one of the frame variables.
            for name, value in tb_frame.f_locals.items():
                cleansed[name] = self.cleanse_special_types(request, value)

        if (tb_frame.f_code.co_name == 'sensitive_variables_wrapper' and
                'sensitive_variables_wrapper' in tb_frame.f_locals):
            # For good measure, obfuscate the decorated function's arguments in
            # the sensitive_variables decorator's frame, in case the variables
            # associated with those arguments were meant to be obfuscated from
            # the decorated function's frame.
            cleansed['func_args'] = CLEANSED_SUBSTITUTE
            cleansed['func_kwargs'] = CLEANSED_SUBSTITUTE

        return cleansed.items()
```
### 4 - django/contrib/auth/__init__.py:

Start line: 1, End line: 58

```python
import inspect
import re

from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.middleware.csrf import rotate_token
from django.utils.crypto import constant_time_compare
from django.utils.module_loading import import_string

from .signals import user_logged_in, user_logged_out, user_login_failed

SESSION_KEY = '_auth_user_id'
BACKEND_SESSION_KEY = '_auth_user_backend'
HASH_SESSION_KEY = '_auth_user_hash'
REDIRECT_FIELD_NAME = 'next'


def load_backend(path):
    return import_string(path)()


def _get_backends(return_tuples=False):
    backends = []
    for backend_path in settings.AUTHENTICATION_BACKENDS:
        backend = load_backend(backend_path)
        backends.append((backend, backend_path) if return_tuples else backend)
    if not backends:
        raise ImproperlyConfigured(
            'No authentication backends have been defined. Does '
            'AUTHENTICATION_BACKENDS contain anything?'
        )
    return backends


def get_backends():
    return _get_backends(return_tuples=False)


def _clean_credentials(credentials):
    """
    Clean a dictionary of credentials of potentially sensitive info before
    sending to less secure functions.

    Not comprehensive - intended for user_login_failed signal
    """
    SENSITIVE_CREDENTIALS = re.compile('api|token|key|secret|password|signature', re.I)
    CLEANSED_SUBSTITUTE = '********************'
    for key in credentials:
        if SENSITIVE_CREDENTIALS.search(key):
            credentials[key] = CLEANSED_SUBSTITUTE
    return credentials


def _get_user_session_key(request):
    # This value in the session is always serialized to a string, so we need
    # to convert it back to Python whenever we access it.
    return get_user_model()._meta.pk.to_python(request.session[SESSION_KEY])
```
### 5 - django/views/decorators/debug.py:

Start line: 64, End line: 79

```python
def sensitive_post_parameters(*parameters):
    def decorator(view):
        @functools.wraps(view)
        def sensitive_post_parameters_wrapper(request, *args, **kwargs):
            assert isinstance(request, HttpRequest), (
                "sensitive_post_parameters didn't receive an HttpRequest. "
                "If you are decorating a classmethod, be sure to use "
                "@method_decorator."
            )
            if parameters:
                request.sensitive_post_parameters = parameters
            else:
                request.sensitive_post_parameters = '__ALL__'
            return view(request, *args, **kwargs)
        return sensitive_post_parameters_wrapper
    return decorator
```
### 6 - django/views/debug.py:

Start line: 126, End line: 153

```python
class SafeExceptionReporterFilter(ExceptionReporterFilter):
    """
    Use annotations made by the sensitive_post_parameters and
    sensitive_variables decorators to filter out sensitive information.
    """

    def is_active(self, request):
        """
        This filter is to add safety in production environments (i.e. DEBUG
        is False). If DEBUG is True then your site is not safe anyway.
        This hook is provided as a convenience to easily activate or
        deactivate the filter on a per request basis.
        """
        return settings.DEBUG is False

    def get_cleansed_multivaluedict(self, request, multivaluedict):
        """
        Replace the keys in a MultiValueDict marked as sensitive with stars.
        This mitigates leaking sensitive POST parameters if something like
        request.POST['nonexistent_key'] throws an exception (#21098).
        """
        sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
        if self.is_active(request) and sensitive_post_parameters:
            multivaluedict = multivaluedict.copy()
            for param in sensitive_post_parameters:
                if param in multivaluedict:
                    multivaluedict[param] = CLEANSED_SUBSTITUTE
        return multivaluedict
```
### 7 - django/contrib/auth/backends.py:

Start line: 31, End line: 49

```python
class ModelBackend(BaseBackend):
    """
    Authenticates against settings.AUTH_USER_MODEL.
    """

    def authenticate(self, request, username=None, password=None, **kwargs):
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)
        if username is None or password is None:
            return
        try:
            user = UserModel._default_manager.get_by_natural_key(username)
        except UserModel.DoesNotExist:
            # Run the default password hasher once to reduce the timing
            # difference between an existing and a nonexistent user (#20760).
            UserModel().set_password(password)
        else:
            if user.check_password(password) and self.user_can_authenticate(user):
                return user
```
### 8 - django/contrib/auth/__init__.py:

Start line: 86, End line: 131

```python
def login(request, user, backend=None):
    """
    Persist a user id and a backend in the request. This way a user doesn't
    have to reauthenticate on every request. Note that data set during
    the anonymous session is retained when the user logs in.
    """
    session_auth_hash = ''
    if user is None:
        user = request.user
    if hasattr(user, 'get_session_auth_hash'):
        session_auth_hash = user.get_session_auth_hash()

    if SESSION_KEY in request.session:
        if _get_user_session_key(request) != user.pk or (
                session_auth_hash and
                not constant_time_compare(request.session.get(HASH_SESSION_KEY, ''), session_auth_hash)):
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
                'You have multiple authentication backends configured and '
                'therefore must provide the `backend` argument or set the '
                '`backend` attribute on the user.'
            )
    else:
        if not isinstance(backend, str):
            raise TypeError('backend must be a dotted import path string (got %r).' % backend)

    request.session[SESSION_KEY] = user._meta.pk.value_to_string(user)
    request.session[BACKEND_SESSION_KEY] = backend
    request.session[HASH_SESSION_KEY] = session_auth_hash
    if hasattr(request, 'user'):
        request.user = user
    rotate_token(request)
    user_logged_in.send(sender=user.__class__, request=request, user=user)
```
### 9 - django/views/debug.py:

Start line: 155, End line: 178

```python
class SafeExceptionReporterFilter(ExceptionReporterFilter):

    def get_post_parameters(self, request):
        """
        Replace the values of POST parameters marked as sensitive with
        stars (*********).
        """
        if request is None:
            return {}
        else:
            sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
            if self.is_active(request) and sensitive_post_parameters:
                cleansed = request.POST.copy()
                if sensitive_post_parameters == '__ALL__':
                    # Cleanse all parameters.
                    for k in cleansed:
                        cleansed[k] = CLEANSED_SUBSTITUTE
                    return cleansed
                else:
                    # Cleanse only the specified parameters.
                    for param in sensitive_post_parameters:
                        if param in cleansed:
                            cleansed[param] = CLEANSED_SUBSTITUTE
                    return cleansed
            else:
                return request.POST
```
### 10 - django/db/backends/base/features.py:

Start line: 117, End line: 215

```python
class BaseDatabaseFeatures:

    # Confirm support for introspected foreign keys
    # Every database can do this reliably, except MySQL,
    # which can't do it for MyISAM tables
    can_introspect_foreign_keys = True

    # Can the backend introspect an AutoField, instead of an IntegerField?
    can_introspect_autofield = False

    # Can the backend introspect a BigIntegerField, instead of an IntegerField?
    can_introspect_big_integer_field = True

    # Can the backend introspect an BinaryField, instead of an TextField?
    can_introspect_binary_field = True

    # Can the backend introspect an DecimalField, instead of an FloatField?
    can_introspect_decimal_field = True

    # Can the backend introspect a DurationField, instead of a BigIntegerField?
    can_introspect_duration_field = True

    # Can the backend introspect an IPAddressField, instead of an CharField?
    can_introspect_ip_address_field = False

    # Can the backend introspect a PositiveIntegerField, instead of an IntegerField?
    can_introspect_positive_integer_field = False

    # Can the backend introspect a SmallIntegerField, instead of an IntegerField?
    can_introspect_small_integer_field = False

    # Can the backend introspect a TimeField, instead of a DateTimeField?
    can_introspect_time_field = True

    # Some backends may not be able to differentiate BigAutoField or
    # SmallAutoField from other fields such as AutoField.
    introspected_big_auto_field_type = 'BigAutoField'
    introspected_small_auto_field_type = 'SmallAutoField'

    # Some backends may not be able to differentiate BooleanField from other
    # fields such as IntegerField.
    introspected_boolean_field_type = 'BooleanField'

    # Can the backend introspect the column order (ASC/DESC) for indexes?
    supports_index_column_ordering = True

    # Does the backend support introspection of materialized views?
    can_introspect_materialized_views = False

    # Support for the DISTINCT ON clause
    can_distinct_on_fields = False

    # Does the backend prevent running SQL queries in broken transactions?
    atomic_transactions = True

    # Can we roll back DDL in a transaction?
    can_rollback_ddl = False

    # Does it support operations requiring references rename in a transaction?
    supports_atomic_references_rename = True

    # Can we issue more than one ALTER COLUMN clause in an ALTER TABLE?
    supports_combined_alters = False

    # Does it support foreign keys?
    supports_foreign_keys = True

    # Can it create foreign key constraints inline when adding columns?
    can_create_inline_fk = True

    # Does it support CHECK constraints?
    supports_column_check_constraints = True
    supports_table_check_constraints = True
    # Does the backend support introspection of CHECK constraints?
    can_introspect_check_constraints = True

    # Does the backend support 'pyformat' style ("... %(name)s ...", {'name': value})
    # parameter passing? Note this can be provided by the backend even if not
    # supported by the Python driver
    supports_paramstyle_pyformat = True

    # Does the backend require literal defaults, rather than parameterized ones?
    requires_literal_defaults = False

    # Does the backend require a connection reset after each material schema change?
    connection_persists_old_columns = False

    # What kind of error does the backend throw when accessing closed cursor?
    closed_cursor_error_class = ProgrammingError

    # Does 'a' LIKE 'A' match?
    has_case_insensitive_like = True

    # Suffix for backends that don't support "SELECT xxx;" queries.
    bare_select_suffix = ''

    # If NULL is implied on columns without needing to be explicitly specified
    implied_column_null = False

    # Does the backend support "select for update" queries with limit (and offset)?
    supports_select_for_update_with_limit = True
```
### 72 - django/contrib/auth/__init__.py:

Start line: 61, End line: 83

```python
def authenticate(request=None, **credentials):
    """
    If the given credentials are valid, return a User object.
    """
    for backend, backend_path in _get_backends(return_tuples=True):
        try:
            inspect.getcallargs(backend.authenticate, request, **credentials)
        except TypeError:
            # This backend doesn't accept these credentials as arguments. Try the next one.
            continue
        try:
            user = backend.authenticate(request, **credentials)
        except PermissionDenied:
            # This backend says to stop in our tracks - this user should not be allowed in at all.
            break
        if user is None:
            continue
        # Annotate the user object with the path of the backend.
        user.backend = backend_path
        return user

    # The credentials supplied are invalid to all backends, fire signal
    user_login_failed.send(sender=__name__, credentials=_clean_credentials(credentials), request=request)
```
### 73 - django/template/base.py:

Start line: 816, End line: 880

```python
class Variable:

    def _resolve_lookup(self, context):
        """
        Perform resolution of a real variable (i.e. not a literal) against the
        given context.

        As indicated by the method's name, this method is an implementation
        detail and shouldn't be called by external code. Use Variable.resolve()
        instead.
        """
        current = context
        try:  # catch-all for silent variable failures
            for bit in self.lookups:
                try:  # dictionary lookup
                    current = current[bit]
                    # ValueError/IndexError are for numpy.array lookup on
                    # numpy < 1.9 and 1.9+ respectively
                except (TypeError, AttributeError, KeyError, ValueError, IndexError):
                    try:  # attribute lookup
                        # Don't return class attributes if the class is the context:
                        if isinstance(current, BaseContext) and getattr(type(current), bit):
                            raise AttributeError
                        current = getattr(current, bit)
                    except (TypeError, AttributeError):
                        # Reraise if the exception was raised by a @property
                        if not isinstance(current, BaseContext) and bit in dir(current):
                            raise
                        try:  # list-index lookup
                            current = current[int(bit)]
                        except (IndexError,  # list index out of range
                                ValueError,  # invalid literal for int()
                                KeyError,    # current is a dict without `int(bit)` key
                                TypeError):  # unsubscriptable object
                            raise VariableDoesNotExist("Failed lookup for key "
                                                       "[%s] in %r",
                                                       (bit, current))  # missing attribute
                if callable(current):
                    if getattr(current, 'do_not_call_in_templates', False):
                        pass
                    elif getattr(current, 'alters_data', False):
                        current = context.template.engine.string_if_invalid
                    else:
                        try:  # method call (assuming no args required)
                            current = current()
                        except TypeError:
                            try:
                                getcallargs(current)
                            except TypeError:  # arguments *were* required
                                current = context.template.engine.string_if_invalid  # invalid method call
                            else:
                                raise
        except Exception as e:
            template_name = getattr(context, 'template_name', None) or 'unknown'
            logger.debug(
                "Exception while resolving variable '%s' in template '%s'.",
                bit,
                template_name,
                exc_info=True,
            )

            if getattr(e, 'silent_variable_failure', False):
                current = context.template.engine.string_if_invalid
            else:
                raise

        return current
```
### 126 - django/template/base.py:

Start line: 792, End line: 814

```python
class Variable:

    def resolve(self, context):
        """Resolve this variable against a given context."""
        if self.lookups is not None:
            # We're dealing with a variable that needs to be resolved
            value = self._resolve_lookup(context)
        else:
            # We're dealing with a literal, so it's already been "resolved"
            value = self.literal
        if self.translate:
            is_safe = isinstance(value, SafeData)
            msgid = value.replace('%', '%%')
            msgid = mark_safe(msgid) if is_safe else msgid
            if self.message_context:
                return pgettext_lazy(self.message_context, msgid)
            else:
                return gettext_lazy(msgid)
        return value

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self.var)

    def __str__(self):
        return self.var
```
### 131 - django/db/models/sql/query.py:

Start line: 2049, End line: 2074

```python
class Query(BaseExpression):

    def get_loaded_field_names_cb(self, target, model, fields):
        """Callback used by get_deferred_field_names()."""
        target[model] = {f.attname for f in fields}

    def set_annotation_mask(self, names):
        """Set the mask of annotations that will be returned by the SELECT."""
        if names is None:
            self.annotation_select_mask = None
        else:
            self.annotation_select_mask = set(names)
        self._annotation_select_cache = None

    def append_annotation_mask(self, names):
        if self.annotation_select_mask is not None:
            self.set_annotation_mask(self.annotation_select_mask.union(names))

    def set_extra_mask(self, names):
        """
        Set the mask of extra select items that will be returned by SELECT.
        Don't remove them from the Query since they might be used later.
        """
        if names is None:
            self.extra_select_mask = None
        else:
            self.extra_select_mask = set(names)
        self._extra_select_cache = None
```
