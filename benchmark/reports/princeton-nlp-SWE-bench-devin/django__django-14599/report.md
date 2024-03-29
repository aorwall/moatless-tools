# django__django-14599

| **django/django** | `019424e44efe495bc5981eb9848c0bb398a6f068` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 675 |
| **Any found context length** | 675 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/middleware/csrf.py b/django/middleware/csrf.py
--- a/django/middleware/csrf.py
+++ b/django/middleware/csrf.py
@@ -437,15 +437,25 @@ def process_view(self, request, callback, callback_args, callback_kwargs):
         return self._accept(request)
 
     def process_response(self, request, response):
-        if not getattr(request, 'csrf_cookie_needs_reset', False):
-            if getattr(response, 'csrf_cookie_set', False):
-                return response
-
-        if not request.META.get("CSRF_COOKIE_USED", False):
-            return response
+        # Send the CSRF cookie whenever the cookie is being used (even if the
+        # client already has it) in order to renew the expiry timer, but only
+        # if it hasn't already been sent during this request-response cycle.
+        # Also, send the cookie no matter what if a reset was requested.
+        if (
+            getattr(request, 'csrf_cookie_needs_reset', False) or (
+                request.META.get('CSRF_COOKIE_USED') and
+                not getattr(response, 'csrf_cookie_set', False)
+            )
+        ):
+            self._set_token(request, response)
+            # Update state to prevent _set_token() from being unnecessarily
+            # called again in process_response() by other instances of
+            # CsrfViewMiddleware. This can happen e.g. when both a decorator
+            # and middleware are used. However, the csrf_cookie_needs_reset
+            # attribute is still respected in subsequent calls e.g. in case
+            # rotate_token() is called in process_response() later by custom
+            # middleware but before those subsequent calls.
+            response.csrf_cookie_set = True
+            request.csrf_cookie_needs_reset = False
 
-        # Set the CSRF cookie even if it's already set, so we renew
-        # the expiry timer.
-        self._set_token(request, response)
-        response.csrf_cookie_set = True
         return response

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/middleware/csrf.py | 440 | 450 | 1 | 1 | 675


## Problem Statement

```
CsrfViewMiddleware.process_response()'s csrf_cookie_needs_reset and csrf_cookie_set logic isn't right
Description
	
I noticed that the csrf_cookie_needs_reset and csrf_cookie_set logic inside CsrfViewMiddleware.process_response() isn't right: ​https://github.com/django/django/blob/fa35c8bdbc6aca65d94d6280fa463d5bc7baa5c0/django/middleware/csrf.py#L439-L451
Consequently--
self._set_token(request, response) can get called twice in some circumstances, even if response.csrf_cookie_set is true at the beginning, and
the cookie can fail to be reset in some circumstances, even if csrf_cookie_needs_reset is true at the beginning.
(I previously let security@djangoproject.com know about this issue, and they said it was okay to resolve this publicly.)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/middleware/csrf.py** | 384 | 452| 675 | 675 | 3790 | 
| 2 | **1 django/middleware/csrf.py** | 325 | 382| 501 | 1176 | 3790 | 
| 3 | **1 django/middleware/csrf.py** | 204 | 227| 189 | 1365 | 3790 | 
| 4 | **1 django/middleware/csrf.py** | 229 | 245| 184 | 1549 | 3790 | 
| 5 | **1 django/middleware/csrf.py** | 273 | 323| 450 | 1999 | 3790 | 
| 6 | **1 django/middleware/csrf.py** | 247 | 271| 176 | 2175 | 3790 | 
| 7 | **1 django/middleware/csrf.py** | 152 | 202| 387 | 2562 | 3790 | 
| 8 | 2 django/views/csrf.py | 15 | 100| 835 | 3397 | 5334 | 
| 9 | 3 django/views/decorators/csrf.py | 1 | 57| 460 | 3857 | 5794 | 
| 10 | **3 django/middleware/csrf.py** | 104 | 149| 342 | 4199 | 5794 | 
| 11 | 4 django/core/checks/security/csrf.py | 1 | 42| 304 | 4503 | 6256 | 
| 12 | **4 django/middleware/csrf.py** | 1 | 53| 472 | 4975 | 6256 | 
| 13 | 5 django/middleware/security.py | 31 | 58| 246 | 5221 | 6774 | 
| 14 | 5 django/views/csrf.py | 101 | 155| 577 | 5798 | 6774 | 
| 15 | 5 django/core/checks/security/csrf.py | 45 | 68| 157 | 5955 | 6774 | 
| 16 | 6 django/contrib/auth/views.py | 251 | 291| 378 | 6333 | 9504 | 
| 17 | **6 django/middleware/csrf.py** | 85 | 101| 195 | 6528 | 9504 | 
| 18 | 6 django/contrib/auth/views.py | 228 | 248| 163 | 6691 | 9504 | 
| 19 | **6 django/middleware/csrf.py** | 68 | 82| 156 | 6847 | 9504 | 
| 20 | 6 django/contrib/auth/views.py | 212 | 226| 133 | 6980 | 9504 | 
| 21 | 6 django/views/csrf.py | 1 | 13| 132 | 7112 | 9504 | 
| 22 | 7 django/contrib/auth/middleware.py | 48 | 84| 360 | 7472 | 10509 | 
| 23 | 8 django/contrib/sessions/middleware.py | 1 | 76| 588 | 8060 | 11098 | 
| 24 | 8 django/contrib/auth/views.py | 293 | 334| 314 | 8374 | 11098 | 
| 25 | 8 django/middleware/security.py | 1 | 29| 278 | 8652 | 11098 | 
| 26 | 9 django/views/defaults.py | 102 | 121| 149 | 8801 | 12126 | 
| 27 | 10 django/utils/cache.py | 135 | 150| 190 | 8991 | 15856 | 
| 28 | 11 django/views/decorators/clickjacking.py | 22 | 54| 238 | 9229 | 16232 | 
| 29 | 11 django/contrib/auth/views.py | 1 | 37| 284 | 9513 | 16232 | 
| 30 | 12 django/core/handlers/base.py | 320 | 351| 212 | 9725 | 18839 | 
| 31 | 13 django/core/checks/security/base.py | 1 | 72| 660 | 10385 | 20880 | 
| 32 | 13 django/contrib/auth/views.py | 337 | 369| 239 | 10624 | 20880 | 
| 33 | **13 django/middleware/csrf.py** | 56 | 65| 111 | 10735 | 20880 | 
| 34 | 14 django/http/__init__.py | 1 | 22| 197 | 10932 | 21077 | 
| 35 | 14 django/views/defaults.py | 1 | 24| 149 | 11081 | 21077 | 
| 36 | 15 django/middleware/clickjacking.py | 1 | 48| 364 | 11445 | 21441 | 
| 37 | 15 django/contrib/auth/views.py | 193 | 209| 122 | 11567 | 21441 | 
| 38 | 15 django/core/checks/security/base.py | 234 | 258| 210 | 11777 | 21441 | 
| 39 | 16 django/http/response.py | 1 | 26| 176 | 11953 | 26084 | 
| 40 | 17 django/contrib/auth/tokens.py | 35 | 60| 170 | 12123 | 26885 | 
| 41 | 18 django/middleware/common.py | 34 | 61| 257 | 12380 | 28414 | 
| 42 | 18 django/core/checks/security/base.py | 74 | 168| 746 | 13126 | 28414 | 
| 43 | 18 django/core/handlers/base.py | 212 | 275| 480 | 13606 | 28414 | 
| 44 | 19 django/conf/global_settings.py | 496 | 647| 943 | 14549 | 34114 | 
| 45 | 20 django/utils/decorators.py | 114 | 152| 316 | 14865 | 35513 | 
| 46 | 20 django/core/handlers/base.py | 160 | 210| 379 | 15244 | 35513 | 
| 47 | 21 django/contrib/auth/admin.py | 1 | 22| 188 | 15432 | 37250 | 
| 48 | 22 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 15570 | 37389 | 
| 49 | 23 django/views/debug.py | 150 | 162| 148 | 15718 | 42142 | 
| 50 | 24 django/middleware/cache.py | 75 | 114| 363 | 16081 | 43730 | 
| 51 | 24 django/middleware/cache.py | 55 | 73| 174 | 16255 | 43730 | 
| 52 | 24 django/views/defaults.py | 27 | 78| 403 | 16658 | 43730 | 
| 53 | 25 django/template/context_processors.py | 1 | 32| 218 | 16876 | 44219 | 
| 54 | 25 django/views/defaults.py | 81 | 99| 129 | 17005 | 44219 | 
| 55 | 25 django/views/decorators/clickjacking.py | 1 | 19| 138 | 17143 | 44219 | 
| 56 | 25 django/contrib/auth/middleware.py | 1 | 25| 182 | 17325 | 44219 | 
| 57 | 26 django/contrib/sites/middleware.py | 1 | 13| 0 | 17325 | 44278 | 
| 58 | 26 django/middleware/common.py | 100 | 115| 165 | 17490 | 44278 | 
| 59 | 26 django/views/defaults.py | 124 | 151| 198 | 17688 | 44278 | 
| 60 | 26 django/views/debug.py | 164 | 187| 177 | 17865 | 44278 | 
| 61 | 27 django/views/decorators/cache.py | 28 | 42| 119 | 17984 | 44737 | 
| 62 | 28 django/contrib/redirects/middleware.py | 1 | 51| 354 | 18338 | 45092 | 
| 63 | 29 django/core/checks/security/sessions.py | 1 | 98| 572 | 18910 | 45665 | 
| 64 | 29 django/middleware/common.py | 63 | 75| 136 | 19046 | 45665 | 
| 65 | 30 django/contrib/auth/urls.py | 1 | 21| 224 | 19270 | 45889 | 
| 66 | 31 django/http/request.py | 375 | 398| 185 | 19455 | 51095 | 
| 67 | 32 django/middleware/locale.py | 28 | 62| 332 | 19787 | 51660 | 
| 68 | 32 django/http/response.py | 69 | 86| 133 | 19920 | 51660 | 
| 69 | 32 django/contrib/auth/middleware.py | 86 | 111| 192 | 20112 | 51660 | 
| 70 | 33 django/views/generic/base.py | 191 | 222| 247 | 20359 | 53310 | 
| 71 | 33 django/middleware/common.py | 149 | 175| 254 | 20613 | 53310 | 
| 72 | 34 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 20760 | 53458 | 
| 73 | 35 django/template/context.py | 1 | 24| 128 | 20888 | 55339 | 
| 74 | 36 django/contrib/admindocs/middleware.py | 1 | 31| 254 | 21142 | 55594 | 
| 75 | 37 django/contrib/admin/sites.py | 224 | 243| 221 | 21363 | 60023 | 
| 76 | 37 django/utils/cache.py | 153 | 193| 453 | 21816 | 60023 | 
| 77 | 38 django/core/handlers/asgi.py | 1 | 19| 111 | 21927 | 62359 | 
| 78 | 38 django/contrib/auth/tokens.py | 62 | 72| 133 | 22060 | 62359 | 
| 79 | 39 django/core/handlers/exception.py | 54 | 122| 557 | 22617 | 63427 | 
| 80 | 40 django/contrib/sitemaps/views.py | 1 | 19| 132 | 22749 | 64220 | 
| 81 | 40 django/middleware/common.py | 1 | 32| 247 | 22996 | 64220 | 
| 82 | 40 django/middleware/cache.py | 131 | 157| 252 | 23248 | 64220 | 
| 83 | 40 django/contrib/auth/middleware.py | 28 | 46| 178 | 23426 | 64220 | 
| 84 | 40 django/http/response.py | 524 | 555| 153 | 23579 | 64220 | 
| 85 | 41 django/contrib/auth/__init__.py | 1 | 38| 241 | 23820 | 65801 | 
| 86 | 42 django/middleware/http.py | 1 | 42| 335 | 24155 | 66136 | 
| 87 | 42 django/http/response.py | 192 | 236| 460 | 24615 | 66136 | 
| 88 | 43 django/views/decorators/debug.py | 77 | 93| 138 | 24753 | 66731 | 
| 89 | 44 django/urls/conf.py | 57 | 86| 222 | 24975 | 67400 | 
| 90 | 45 django/utils/deprecation.py | 79 | 129| 372 | 25347 | 68439 | 
| 91 | 46 django/core/servers/basehttp.py | 181 | 200| 170 | 25517 | 70343 | 
| 92 | 47 django/contrib/messages/storage/cookie.py | 93 | 111| 144 | 25661 | 71722 | 
| 93 | 47 django/http/response.py | 238 | 258| 237 | 25898 | 71722 | 
| 94 | 48 django/template/defaulttags.py | 51 | 68| 152 | 26050 | 82346 | 
| 95 | 48 django/http/response.py | 350 | 385| 212 | 26262 | 82346 | 
| 96 | 49 django/core/handlers/wsgi.py | 64 | 119| 486 | 26748 | 84105 | 
| 97 | 49 django/core/checks/security/base.py | 171 | 196| 188 | 26936 | 84105 | 
| 98 | 49 django/contrib/admin/sites.py | 1 | 35| 225 | 27161 | 84105 | 
| 99 | 50 django/contrib/auth/mixins.py | 44 | 71| 235 | 27396 | 84969 | 
| 100 | 50 django/http/response.py | 503 | 521| 186 | 27582 | 84969 | 
| 101 | 50 django/middleware/common.py | 118 | 147| 277 | 27859 | 84969 | 
| 102 | 50 django/contrib/auth/tokens.py | 1 | 33| 208 | 28067 | 84969 | 
| 103 | 51 django/contrib/auth/hashers.py | 86 | 108| 167 | 28234 | 90892 | 
| 104 | 51 django/views/debug.py | 203 | 251| 467 | 28701 | 90892 | 
| 105 | 51 django/views/debug.py | 189 | 201| 143 | 28844 | 90892 | 
| 106 | 51 django/views/decorators/cache.py | 45 | 60| 123 | 28967 | 90892 | 
| 107 | 51 django/http/request.py | 160 | 181| 164 | 29131 | 90892 | 
| 108 | 52 django/views/__init__.py | 1 | 4| 0 | 29131 | 90907 | 
| 109 | 53 django/contrib/messages/context_processors.py | 1 | 14| 0 | 29131 | 90978 | 
| 110 | 53 django/contrib/auth/middleware.py | 114 | 125| 107 | 29238 | 90978 | 
| 111 | 54 django/urls/base.py | 27 | 86| 438 | 29676 | 92161 | 
| 112 | 54 django/conf/global_settings.py | 401 | 495| 782 | 30458 | 92161 | 
| 113 | 54 django/core/handlers/base.py | 277 | 292| 125 | 30583 | 92161 | 
| 114 | 55 django/http/cookie.py | 1 | 24| 156 | 30739 | 92318 | 
| 115 | 56 django/contrib/messages/views.py | 1 | 19| 0 | 30739 | 92414 | 
| 116 | 56 django/utils/deprecation.py | 131 | 149| 122 | 30861 | 92414 | 
| 117 | 56 django/template/context.py | 233 | 260| 199 | 31060 | 92414 | 
| 118 | 57 django/urls/resolvers.py | 637 | 647| 120 | 31180 | 98169 | 
| 119 | 57 django/core/servers/basehttp.py | 144 | 179| 280 | 31460 | 98169 | 
| 120 | 57 django/contrib/auth/hashers.py | 462 | 500| 331 | 31791 | 98169 | 
| 121 | 57 django/utils/cache.py | 244 | 275| 257 | 32048 | 98169 | 
| 122 | 57 django/urls/resolvers.py | 443 | 471| 288 | 32336 | 98169 | 
| 123 | 57 django/contrib/auth/__init__.py | 41 | 60| 167 | 32503 | 98169 | 
| 124 | 57 django/contrib/auth/views.py | 133 | 167| 269 | 32772 | 98169 | 
| 125 | 57 django/urls/resolvers.py | 649 | 722| 681 | 33453 | 98169 | 
| 126 | 57 django/utils/cache.py | 303 | 323| 211 | 33664 | 98169 | 
| 127 | 57 django/http/response.py | 558 | 583| 159 | 33823 | 98169 | 
| 128 | 58 django/contrib/admin/views/decorators.py | 1 | 19| 135 | 33958 | 98305 | 
| 129 | 59 django/contrib/redirects/admin.py | 1 | 11| 0 | 33958 | 98373 | 
| 130 | 60 django/views/decorators/http.py | 77 | 124| 366 | 34324 | 99344 | 
| 131 | 61 django/contrib/admin/options.py | 1 | 97| 761 | 35085 | 118002 | 
| 132 | 61 django/utils/cache.py | 1 | 34| 274 | 35359 | 118002 | 
| 133 | 61 django/http/request.py | 151 | 158| 111 | 35470 | 118002 | 
| 134 | 62 django/contrib/admindocs/views.py | 1 | 31| 234 | 35704 | 121341 | 
| 135 | 62 django/http/request.py | 1 | 39| 273 | 35977 | 121341 | 
| 136 | 63 django/contrib/admin/tests.py | 1 | 36| 265 | 36242 | 122819 | 
| 137 | 63 django/middleware/common.py | 77 | 98| 227 | 36469 | 122819 | 
| 138 | 64 django/utils/autoreload.py | 550 | 573| 177 | 36646 | 127917 | 
| 139 | 64 django/utils/cache.py | 37 | 102| 559 | 37205 | 127917 | 
| 140 | 64 django/http/request.py | 290 | 310| 189 | 37394 | 127917 | 
| 141 | 64 django/core/checks/security/base.py | 214 | 231| 127 | 37521 | 127917 | 
| 142 | 65 django/contrib/flatpages/views.py | 1 | 45| 399 | 37920 | 128507 | 
| 143 | 66 django/conf/__init__.py | 133 | 189| 511 | 38431 | 130344 | 
| 144 | 66 django/views/decorators/debug.py | 47 | 75| 199 | 38630 | 130344 | 
| 145 | 66 django/conf/global_settings.py | 151 | 266| 859 | 39489 | 130344 | 
| 146 | 67 django/views/i18n.py | 79 | 182| 702 | 40191 | 132805 | 
| 147 | 67 django/core/handlers/exception.py | 125 | 150| 167 | 40358 | 132805 | 
| 148 | 67 django/middleware/cache.py | 1 | 52| 431 | 40789 | 132805 | 
| 149 | 68 django/contrib/sessions/exceptions.py | 1 | 17| 0 | 40789 | 132876 | 
| 150 | 68 django/conf/global_settings.py | 51 | 150| 1160 | 41949 | 132876 | 
| 151 | 68 django/views/decorators/http.py | 1 | 52| 350 | 42299 | 132876 | 
| 152 | 68 django/core/checks/security/base.py | 199 | 211| 110 | 42409 | 132876 | 
| 153 | 68 django/core/handlers/base.py | 1 | 97| 743 | 43152 | 132876 | 
| 154 | 68 django/contrib/auth/hashers.py | 547 | 592| 330 | 43482 | 132876 | 
| 155 | 69 django/contrib/auth/decorators.py | 1 | 35| 313 | 43795 | 133463 | 
| 156 | 69 django/urls/resolvers.py | 282 | 298| 160 | 43955 | 133463 | 
| 157 | 69 django/utils/cache.py | 278 | 300| 258 | 44213 | 133463 | 
| 158 | 70 django/conf/urls/__init__.py | 1 | 10| 0 | 44213 | 133528 | 
| 159 | 71 django/views/decorators/vary.py | 1 | 42| 232 | 44445 | 133761 | 
| 160 | 72 django/utils/http.py | 317 | 355| 402 | 44847 | 137003 | 
| 161 | 72 django/core/servers/basehttp.py | 124 | 141| 179 | 45026 | 137003 | 
| 162 | 73 django/core/cache/utils.py | 1 | 13| 0 | 45026 | 137082 | 
| 163 | 73 django/template/defaulttags.py | 1167 | 1190| 176 | 45202 | 137082 | 
| 164 | 73 django/contrib/auth/admin.py | 128 | 190| 476 | 45678 | 137082 | 
| 165 | 73 django/views/generic/base.py | 47 | 83| 339 | 46017 | 137082 | 
| 166 | 73 django/views/debug.py | 88 | 120| 258 | 46275 | 137082 | 
| 167 | 74 django/views/generic/edit.py | 1 | 67| 481 | 46756 | 138935 | 
| 168 | 75 django/contrib/auth/forms.py | 265 | 281| 146 | 46902 | 142061 | 
| 169 | 75 django/contrib/admin/sites.py | 419 | 433| 129 | 47031 | 142061 | 
| 170 | 75 django/urls/resolvers.py | 576 | 612| 347 | 47378 | 142061 | 
| 171 | 75 django/views/debug.py | 122 | 148| 216 | 47594 | 142061 | 
| 172 | 76 django/views/static.py | 1 | 16| 110 | 47704 | 143113 | 
| 173 | 77 django/contrib/redirects/apps.py | 1 | 9| 0 | 47704 | 143164 | 
| 174 | 77 django/http/request.py | 341 | 372| 255 | 47959 | 143164 | 
| 175 | 77 django/contrib/admin/options.py | 1261 | 1334| 659 | 48618 | 143164 | 
| 176 | 77 django/views/debug.py | 72 | 85| 115 | 48733 | 143164 | 
| 177 | 77 django/contrib/auth/views.py | 66 | 108| 319 | 49052 | 143164 | 
| 178 | 77 django/contrib/auth/tokens.py | 74 | 105| 307 | 49359 | 143164 | 
| 179 | 78 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 49359 | 143240 | 
| 180 | 78 django/views/generic/edit.py | 129 | 149| 182 | 49541 | 143240 | 
| 181 | 79 django/contrib/messages/middleware.py | 1 | 27| 174 | 49715 | 143415 | 
| 182 | 80 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 49983 | 143683 | 
| 183 | 80 django/contrib/auth/forms.py | 319 | 360| 289 | 50272 | 143683 | 
| 184 | 80 django/contrib/admin/sites.py | 325 | 340| 158 | 50430 | 143683 | 
| 185 | 80 django/http/response.py | 278 | 316| 282 | 50712 | 143683 | 
| 186 | 81 django/views/generic/__init__.py | 1 | 23| 189 | 50901 | 143873 | 
| 187 | 82 django/contrib/sessions/serializers.py | 1 | 21| 0 | 50901 | 143960 | 
| 188 | 82 django/views/generic/base.py | 1 | 26| 142 | 51043 | 143960 | 
| 189 | 82 django/contrib/auth/forms.py | 57 | 75| 124 | 51167 | 143960 | 
| 190 | 83 django/template/autoreload.py | 34 | 54| 136 | 51303 | 144300 | 
| 191 | 83 django/contrib/admin/options.py | 1336 | 1361| 232 | 51535 | 144300 | 
| 192 | 84 django/core/validators.py | 1 | 16| 127 | 51662 | 148999 | 
| 193 | 84 django/views/decorators/debug.py | 1 | 44| 274 | 51936 | 148999 | 
| 194 | 84 django/utils/cache.py | 105 | 132| 183 | 52119 | 148999 | 
| 195 | 85 django/contrib/sites/checks.py | 1 | 14| 0 | 52119 | 149078 | 
| 196 | 86 django/template/response.py | 1 | 43| 389 | 52508 | 150169 | 


## Patch

```diff
diff --git a/django/middleware/csrf.py b/django/middleware/csrf.py
--- a/django/middleware/csrf.py
+++ b/django/middleware/csrf.py
@@ -437,15 +437,25 @@ def process_view(self, request, callback, callback_args, callback_kwargs):
         return self._accept(request)
 
     def process_response(self, request, response):
-        if not getattr(request, 'csrf_cookie_needs_reset', False):
-            if getattr(response, 'csrf_cookie_set', False):
-                return response
-
-        if not request.META.get("CSRF_COOKIE_USED", False):
-            return response
+        # Send the CSRF cookie whenever the cookie is being used (even if the
+        # client already has it) in order to renew the expiry timer, but only
+        # if it hasn't already been sent during this request-response cycle.
+        # Also, send the cookie no matter what if a reset was requested.
+        if (
+            getattr(request, 'csrf_cookie_needs_reset', False) or (
+                request.META.get('CSRF_COOKIE_USED') and
+                not getattr(response, 'csrf_cookie_set', False)
+            )
+        ):
+            self._set_token(request, response)
+            # Update state to prevent _set_token() from being unnecessarily
+            # called again in process_response() by other instances of
+            # CsrfViewMiddleware. This can happen e.g. when both a decorator
+            # and middleware are used. However, the csrf_cookie_needs_reset
+            # attribute is still respected in subsequent calls e.g. in case
+            # rotate_token() is called in process_response() later by custom
+            # middleware but before those subsequent calls.
+            response.csrf_cookie_set = True
+            request.csrf_cookie_needs_reset = False
 
-        # Set the CSRF cookie even if it's already set, so we renew
-        # the expiry timer.
-        self._set_token(request, response)
-        response.csrf_cookie_set = True
         return response

```

## Test Patch

```diff
diff --git a/tests/csrf_tests/tests.py b/tests/csrf_tests/tests.py
--- a/tests/csrf_tests/tests.py
+++ b/tests/csrf_tests/tests.py
@@ -14,8 +14,9 @@
 from django.views.decorators.csrf import csrf_exempt, requires_csrf_token
 
 from .views import (
-    ensure_csrf_cookie_view, non_token_view_using_request_processor,
-    post_form_view, token_view,
+    ensure_csrf_cookie_view, ensured_and_protected_view,
+    non_token_view_using_request_processor, post_form_view, protected_view,
+    sandwiched_rotate_token_view, token_view,
 )
 
 # This is a test (unmasked) CSRF cookie / secret.
@@ -69,14 +70,30 @@ def test_mask_cipher_secret(self):
                 self.assertMaskedSecretCorrect(masked, secret)
 
 
+class TestingSessionStore(SessionStore):
+    """
+    A version of SessionStore that stores what cookie values are passed to
+    set_cookie() when CSRF_USE_SESSIONS=True.
+    """
+    def __init__(self, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+        # This is a list of the cookie values passed to set_cookie() over
+        # the course of the request-response.
+        self._cookies_set = []
+
+    def __setitem__(self, key, value):
+        super().__setitem__(key, value)
+        self._cookies_set.append(value)
+
+
 class TestingHttpRequest(HttpRequest):
     """
-    A version of HttpRequest that allows us to change some things
-    more easily
+    A version of HttpRequest that lets one track and change some things more
+    easily.
     """
     def __init__(self):
         super().__init__()
-        self.session = SessionStore()
+        self.session = TestingSessionStore()
 
     def is_secure(self):
         return getattr(self, '_is_secure_override', False)
@@ -99,6 +116,21 @@ def _read_csrf_cookie(self, req, resp):
         """
         raise NotImplementedError('This method must be implemented by a subclass.')
 
+    def _get_cookies_set(self, req, resp):
+        """
+        Return a list of the cookie values passed to set_cookie() over the
+        course of the request-response.
+        """
+        raise NotImplementedError('This method must be implemented by a subclass.')
+
+    def assertCookiesSet(self, req, resp, expected_secrets):
+        """
+        Assert that set_cookie() was called with the given sequence of secrets.
+        """
+        cookies_set = self._get_cookies_set(req, resp)
+        secrets_set = [_unmask_cipher_token(cookie) for cookie in cookies_set]
+        self.assertEqual(secrets_set, expected_secrets)
+
     def _get_request(self, method=None, cookie=None):
         if method is None:
             method = 'GET'
@@ -332,6 +364,21 @@ def test_put_and_delete_allowed(self):
         resp = mw.process_view(req, post_form_view, (), {})
         self.assertIsNone(resp)
 
+    def test_rotate_token_triggers_second_reset(self):
+        """
+        If rotate_token() is called after the token is reset in
+        CsrfViewMiddleware's process_response() and before another call to
+        the same process_response(), the cookie is reset a second time.
+        """
+        req = self._get_POST_request_with_token()
+        resp = sandwiched_rotate_token_view(req)
+        self.assertContains(resp, 'OK')
+        csrf_cookie = self._read_csrf_cookie(req, resp)
+        actual_secret = _unmask_cipher_token(csrf_cookie)
+        # set_cookie() was called a second time with a different secret.
+        self.assertCookiesSet(req, resp, [TEST_SECRET, actual_secret])
+        self.assertNotEqual(actual_secret, TEST_SECRET)
+
     # Tests for the template tag method
     def test_token_node_no_csrf_cookie(self):
         """
@@ -875,6 +922,9 @@ def _read_csrf_cookie(self, req, resp):
         csrf_cookie = resp.cookies[settings.CSRF_COOKIE_NAME]
         return csrf_cookie.value
 
+    def _get_cookies_set(self, req, resp):
+        return resp._cookies_set
+
     def test_ensures_csrf_cookie_no_middleware(self):
         """
         The ensure_csrf_cookie() decorator works without middleware.
@@ -1016,6 +1066,32 @@ def test_masked_unmasked_combinations(self):
                 resp = mw.process_view(req, token_view, (), {})
                 self.assertIsNone(resp)
 
+    def test_cookie_reset_only_once(self):
+        """
+        A CSRF cookie that needs to be reset is reset only once when the view
+        is decorated with both ensure_csrf_cookie and csrf_protect.
+        """
+        # Pass an unmasked cookie to trigger a cookie reset.
+        req = self._get_POST_request_with_token(cookie=TEST_SECRET)
+        resp = ensured_and_protected_view(req)
+        self.assertContains(resp, 'OK')
+        csrf_cookie = self._read_csrf_cookie(req, resp)
+        actual_secret = _unmask_cipher_token(csrf_cookie)
+        self.assertEqual(actual_secret, TEST_SECRET)
+        # set_cookie() was called only once and with the expected secret.
+        self.assertCookiesSet(req, resp, [TEST_SECRET])
+
+    def test_invalid_cookie_replaced_on_GET(self):
+        """
+        A CSRF cookie with the wrong format is replaced during a GET request.
+        """
+        req = self._get_request(cookie='badvalue')
+        resp = protected_view(req)
+        self.assertContains(resp, 'OK')
+        csrf_cookie = self._read_csrf_cookie(req, resp)
+        self.assertTrue(csrf_cookie, msg='No CSRF cookie was sent.')
+        self.assertEqual(len(csrf_cookie), CSRF_TOKEN_LENGTH)
+
     def test_bare_secret_accepted_and_replaced(self):
         """
         The csrf token is reset from a bare secret.
@@ -1089,6 +1165,9 @@ def _read_csrf_cookie(self, req, resp=None):
             return False
         return req.session[CSRF_SESSION_KEY]
 
+    def _get_cookies_set(self, req, resp):
+        return req.session._cookies_set
+
     def test_no_session_on_request(self):
         msg = (
             'CSRF_USE_SESSIONS is enabled, but request.session is not set. '
diff --git a/tests/csrf_tests/views.py b/tests/csrf_tests/views.py
--- a/tests/csrf_tests/views.py
+++ b/tests/csrf_tests/views.py
@@ -1,8 +1,63 @@
 from django.http import HttpResponse
-from django.middleware.csrf import get_token
+from django.middleware.csrf import get_token, rotate_token
 from django.template import Context, RequestContext, Template
 from django.template.context_processors import csrf
-from django.views.decorators.csrf import ensure_csrf_cookie
+from django.utils.decorators import decorator_from_middleware
+from django.utils.deprecation import MiddlewareMixin
+from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
+
+
+class TestingHttpResponse(HttpResponse):
+    """
+    A version of HttpResponse that stores what cookie values are passed to
+    set_cookie() when CSRF_USE_SESSIONS=False.
+    """
+    def __init__(self, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+        # This is a list of the cookie values passed to set_cookie() over
+        # the course of the request-response.
+        self._cookies_set = []
+
+    def set_cookie(self, key, value, **kwargs):
+        super().set_cookie(key, value, **kwargs)
+        self._cookies_set.append(value)
+
+
+class _CsrfCookieRotator(MiddlewareMixin):
+
+    def process_response(self, request, response):
+        rotate_token(request)
+        return response
+
+
+csrf_rotating_token = decorator_from_middleware(_CsrfCookieRotator)
+
+
+@csrf_protect
+def protected_view(request):
+    return HttpResponse('OK')
+
+
+@ensure_csrf_cookie
+def ensure_csrf_cookie_view(request):
+    return HttpResponse('OK')
+
+
+@csrf_protect
+@ensure_csrf_cookie
+def ensured_and_protected_view(request):
+    return TestingHttpResponse('OK')
+
+
+@csrf_protect
+@csrf_rotating_token
+@ensure_csrf_cookie
+def sandwiched_rotate_token_view(request):
+    """
+    This is a view that calls rotate_token() in process_response() between two
+    calls to CsrfViewMiddleware.process_response().
+    """
+    return TestingHttpResponse('OK')
 
 
 def post_form_view(request):
@@ -12,12 +67,6 @@ def post_form_view(request):
 """)
 
 
-@ensure_csrf_cookie
-def ensure_csrf_cookie_view(request):
-    # Doesn't insert a token or anything.
-    return HttpResponse()
-
-
 def token_view(request):
     context = RequestContext(request, processors=[csrf])
     template = Template('{% csrf_token %}')

```


## Code snippets

### 1 - django/middleware/csrf.py:

Start line: 384, End line: 452

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def process_view(self, request, callback, callback_args, callback_kwargs):
        if getattr(request, 'csrf_processing_done', False):
            return None

        # Wait until request.META["CSRF_COOKIE"] has been manipulated before
        # bailing out, so that get_token still works
        if getattr(callback, 'csrf_exempt', False):
            return None

        # Assume that anything not defined as 'safe' by RFC7231 needs protection
        if request.method in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            return self._accept(request)

        if getattr(request, '_dont_enforce_csrf_checks', False):
            # Mechanism to turn off CSRF checks for test suite. It comes after
            # the creation of CSRF cookies, so that everything else continues
            # to work exactly the same (e.g. cookies are sent, etc.), but
            # before any branches that call the _reject method.
            return self._accept(request)

        # Reject the request if the Origin header doesn't match an allowed
        # value.
        if 'HTTP_ORIGIN' in request.META:
            if not self._origin_verified(request):
                return self._reject(request, REASON_BAD_ORIGIN % request.META['HTTP_ORIGIN'])
        elif request.is_secure():
            # If the Origin header wasn't provided, reject HTTPS requests if
            # the Referer header doesn't match an allowed value.
            #
            # Suppose user visits http://example.com/
            # An active network attacker (man-in-the-middle, MITM) sends a
            # POST form that targets https://example.com/detonate-bomb/ and
            # submits it via JavaScript.
            #
            # The attacker will need to provide a CSRF cookie and token, but
            # that's no problem for a MITM and the session-independent secret
            # we're using. So the MITM can circumvent the CSRF protection. This
            # is true for any HTTP connection, but anyone using HTTPS expects
            # better! For this reason, for https://example.com/ we need
            # additional protection that treats http://example.com/ as
            # completely untrusted. Under HTTPS, Barth et al. found that the
            # Referer header is missing for same-domain requests in only about
            # 0.2% of cases or less, so we can use strict Referer checking.
            try:
                self._check_referer(request)
            except RejectRequest as exc:
                return self._reject(request, exc.reason)

        try:
            self._check_token(request)
        except RejectRequest as exc:
            return self._reject(request, exc.reason)

        return self._accept(request)

    def process_response(self, request, response):
        if not getattr(request, 'csrf_cookie_needs_reset', False):
            if getattr(response, 'csrf_cookie_set', False):
                return response

        if not request.META.get("CSRF_COOKIE_USED", False):
            return response

        # Set the CSRF cookie even if it's already set, so we renew
        # the expiry timer.
        self._set_token(request, response)
        response.csrf_cookie_set = True
        return response
```
### 2 - django/middleware/csrf.py:

Start line: 325, End line: 382

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _check_token(self, request):
        # Access csrf_token via self._get_token() as rotate_token() may have
        # been called by an authentication middleware during the
        # process_request() phase.
        try:
            csrf_token = self._get_token(request)
        except InvalidTokenFormat as exc:
            raise RejectRequest(f'CSRF cookie {exc.reason}.')

        if csrf_token is None:
            # No CSRF cookie. For POST requests, we insist on a CSRF cookie,
            # and in this way we can avoid all CSRF attacks, including login
            # CSRF.
            raise RejectRequest(REASON_NO_CSRF_COOKIE)

        # Check non-cookie token for match.
        request_csrf_token = ''
        if request.method == 'POST':
            try:
                request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
            except OSError:
                # Handle a broken connection before we've completed reading the
                # POST data. process_view shouldn't raise any exceptions, so
                # we'll ignore and serve the user a 403 (assuming they're still
                # listening, which they probably aren't because of the error).
                pass

        if request_csrf_token == '':
            # Fall back to X-CSRFToken, to make things easier for AJAX, and
            # possible for PUT/DELETE.
            try:
                request_csrf_token = request.META[settings.CSRF_HEADER_NAME]
            except KeyError:
                raise RejectRequest(REASON_CSRF_TOKEN_MISSING)
            token_source = settings.CSRF_HEADER_NAME
        else:
            token_source = 'POST'

        try:
            request_csrf_token = _sanitize_token(request_csrf_token)
        except InvalidTokenFormat as exc:
            reason = self._bad_token_message(exc.reason, token_source)
            raise RejectRequest(reason)

        if not _compare_masked_tokens(request_csrf_token, csrf_token):
            reason = self._bad_token_message('incorrect', token_source)
            raise RejectRequest(reason)

    def process_request(self, request):
        try:
            csrf_token = self._get_token(request)
        except InvalidTokenFormat:
            csrf_token = _get_new_csrf_token()
            request.csrf_cookie_needs_reset = True

        if csrf_token is not None:
            # Use same token next time.
            request.META['CSRF_COOKIE'] = csrf_token
```
### 3 - django/middleware/csrf.py:

Start line: 204, End line: 227

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _get_token(self, request):
        if settings.CSRF_USE_SESSIONS:
            try:
                return request.session.get(CSRF_SESSION_KEY)
            except AttributeError:
                raise ImproperlyConfigured(
                    'CSRF_USE_SESSIONS is enabled, but request.session is not '
                    'set. SessionMiddleware must appear before CsrfViewMiddleware '
                    'in MIDDLEWARE.'
                )
        else:
            try:
                cookie_token = request.COOKIES[settings.CSRF_COOKIE_NAME]
            except KeyError:
                return None

            # This can raise InvalidTokenFormat.
            csrf_token = _sanitize_token(cookie_token)

            if csrf_token != cookie_token:
                # Then the cookie token had length CSRF_SECRET_LENGTH, so flag
                # to replace it with the masked version.
                request.csrf_cookie_needs_reset = True
            return csrf_token
```
### 4 - django/middleware/csrf.py:

Start line: 229, End line: 245

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _set_token(self, request, response):
        if settings.CSRF_USE_SESSIONS:
            if request.session.get(CSRF_SESSION_KEY) != request.META['CSRF_COOKIE']:
                request.session[CSRF_SESSION_KEY] = request.META['CSRF_COOKIE']
        else:
            response.set_cookie(
                settings.CSRF_COOKIE_NAME,
                request.META['CSRF_COOKIE'],
                max_age=settings.CSRF_COOKIE_AGE,
                domain=settings.CSRF_COOKIE_DOMAIN,
                path=settings.CSRF_COOKIE_PATH,
                secure=settings.CSRF_COOKIE_SECURE,
                httponly=settings.CSRF_COOKIE_HTTPONLY,
                samesite=settings.CSRF_COOKIE_SAMESITE,
            )
            # Set the Vary header since content varies with the CSRF cookie.
            patch_vary_headers(response, ('Cookie',))
```
### 5 - django/middleware/csrf.py:

Start line: 273, End line: 323

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
        if referer is None:
            raise RejectRequest(REASON_NO_REFERER)

        try:
            referer = urlparse(referer)
        except ValueError:
            raise RejectRequest(REASON_MALFORMED_REFERER)

        # Make sure we have a valid URL for Referer.
        if '' in (referer.scheme, referer.netloc):
            raise RejectRequest(REASON_MALFORMED_REFERER)

        # Ensure that our Referer is also secure.
        if referer.scheme != 'https':
            raise RejectRequest(REASON_INSECURE_REFERER)

        if any(
            is_same_domain(referer.netloc, host)
            for host in self.csrf_trusted_origins_hosts
        ):
            return
        # Allow matching the configured cookie domain.
        good_referer = (
            settings.SESSION_COOKIE_DOMAIN
            if settings.CSRF_USE_SESSIONS
            else settings.CSRF_COOKIE_DOMAIN
        )
        if good_referer is None:
            # If no cookie domain is configured, allow matching the current
            # host:port exactly if it's permitted by ALLOWED_HOSTS.
            try:
                # request.get_host() includes the port.
                good_referer = request.get_host()
            except DisallowedHost:
                raise RejectRequest(REASON_BAD_REFERER % referer.geturl())
        else:
            server_port = request.get_port()
            if server_port not in ('443', '80'):
                good_referer = '%s:%s' % (good_referer, server_port)

        if not is_same_domain(referer.netloc, good_referer):
            raise RejectRequest(REASON_BAD_REFERER % referer.geturl())

    def _bad_token_message(self, reason, token_source):
        if token_source != 'POST':
            # Assume it is a settings.CSRF_HEADER_NAME value.
            header_name = HttpHeaders.parse_header_name(token_source)
            token_source = f'the {header_name!r} HTTP header'
        return f'CSRF token from {token_source} {reason}.'
```
### 6 - django/middleware/csrf.py:

Start line: 247, End line: 271

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _origin_verified(self, request):
        request_origin = request.META['HTTP_ORIGIN']
        try:
            good_host = request.get_host()
        except DisallowedHost:
            pass
        else:
            good_origin = '%s://%s' % (
                'https' if request.is_secure() else 'http',
                good_host,
            )
            if request_origin == good_origin:
                return True
        if request_origin in self.allowed_origins_exact:
            return True
        try:
            parsed_origin = urlparse(request_origin)
        except ValueError:
            return False
        request_scheme = parsed_origin.scheme
        request_netloc = parsed_origin.netloc
        return any(
            is_same_domain(request_netloc, host)
            for host in self.allowed_origin_subdomains.get(request_scheme, ())
        )
```
### 7 - django/middleware/csrf.py:

Start line: 152, End line: 202

```python
class CsrfViewMiddleware(MiddlewareMixin):
    """
    Require a present and correct csrfmiddlewaretoken for POST requests that
    have a CSRF cookie, and set an outgoing CSRF cookie.

    This middleware should be used in conjunction with the {% csrf_token %}
    template tag.
    """
    @cached_property
    def csrf_trusted_origins_hosts(self):
        return [
            urlparse(origin).netloc.lstrip('*')
            for origin in settings.CSRF_TRUSTED_ORIGINS
        ]

    @cached_property
    def allowed_origins_exact(self):
        return {
            origin for origin in settings.CSRF_TRUSTED_ORIGINS
            if '*' not in origin
        }

    @cached_property
    def allowed_origin_subdomains(self):
        """
        A mapping of allowed schemes to list of allowed netlocs, where all
        subdomains of the netloc are allowed.
        """
        allowed_origin_subdomains = defaultdict(list)
        for parsed in (urlparse(origin) for origin in settings.CSRF_TRUSTED_ORIGINS if '*' in origin):
            allowed_origin_subdomains[parsed.scheme].append(parsed.netloc.lstrip('*'))
        return allowed_origin_subdomains

    # The _accept and _reject methods currently only exist for the sake of the
    # requires_csrf_token decorator.
    def _accept(self, request):
        # Avoid checking the request twice by adding a custom attribute to
        # request.  This will be relevant when both decorator and middleware
        # are used.
        request.csrf_processing_done = True
        return None

    def _reject(self, request, reason):
        response = _get_failure_view()(request, reason=reason)
        log_response(
            'Forbidden (%s): %s', reason, request.path,
            response=response,
            request=request,
            logger=logger,
        )
        return response
```
### 8 - django/views/csrf.py:

Start line: 15, End line: 100

```python
CSRF_FAILURE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta http-equiv="content-type" content="text/html; charset=utf-8">
  <meta name="robots" content="NONE,NOARCHIVE">
  <title>403 Forbidden</title>
  <style type="text/css">
    html * { padding:0; margin:0; }
    body * { padding:10px 20px; }
    body * * { padding:0; }
    body { font:small sans-serif; background:#eee; color:#000; }
    body>div { border-bottom:1px solid #ddd; }
    h1 { font-weight:normal; margin-bottom:.4em; }
    h1 span { font-size:60%; color:#666; font-weight:normal; }
    #info { background:#f6f6f6; }
    #info ul { margin: 0.5em 4em; }
    #info p, #summary p { padding-top:10px; }
    #summary { background: #ffc; }
    #explanation { background:#eee; border-bottom: 0px none; }
  </style>
</head>
<body>
<div id="summary">
  <h1>{{ title }} <span>(403)</span></h1>
  <p>{{ main }}</p>
{% if no_referer %}
  <p>{{ no_referer1 }}</p>
  <p>{{ no_referer2 }}</p>
  <p>{{ no_referer3 }}</p>
{% endif %}
{% if no_cookie %}
  <p>{{ no_cookie1 }}</p>
  <p>{{ no_cookie2 }}</p>
{% endif %}
</div>
{% if DEBUG %}
<div id="info">
  <h2>Help</h2>
    {% if reason %}
    <p>Reason given for failure:</p>
    <pre>
    {{ reason }}
    </pre>
    {% endif %}

  <p>In general, this can occur when there is a genuine Cross Site Request Forgery, or when
  <a
  href="https://docs.djangoproject.com/en/{{ docs_version }}/ref/csrf/">Django’s
  CSRF mechanism</a> has not been used correctly.  For POST forms, you need to
  ensure:</p>

  <ul>
    <li>Your browser is accepting cookies.</li>

    <li>The view function passes a <code>request</code> to the template’s <a
    href="https://docs.djangoproject.com/en/dev/topics/templates/#django.template.backends.base.Template.render"><code>render</code></a>
    method.</li>

    <li>In the template, there is a <code>{% templatetag openblock %} csrf_token
    {% templatetag closeblock %}</code> template tag inside each POST form that
    targets an internal URL.</li>

    <li>If you are not using <code>CsrfViewMiddleware</code>, then you must use
    <code>csrf_protect</code> on any views that use the <code>csrf_token</code>
    template tag, as well as those that accept the POST data.</li>

    <li>The form has a valid CSRF token. After logging in in another browser
    tab or hitting the back button after a login, you may need to reload the
    page with the form, because the token is rotated after a login.</li>
  </ul>

  <p>You’re seeing the help section of this page because you have <code>DEBUG =
  True</code> in your Django settings file. Change that to <code>False</code>,
  and only the initial error message will be displayed.  </p>

  <p>You can customize this page using the CSRF_FAILURE_VIEW setting.</p>
</div>
{% else %}
<div id="explanation">
  <p><small>{{ more }}</small></p>
</div>
{% endif %}
</body>
</html>
"""
```
### 9 - django/views/decorators/csrf.py:

Start line: 1, End line: 57

```python
from functools import wraps

from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.utils.decorators import decorator_from_middleware

csrf_protect = decorator_from_middleware(CsrfViewMiddleware)
csrf_protect.__name__ = "csrf_protect"
csrf_protect.__doc__ = """
This decorator adds CSRF protection in exactly the same way as
CsrfViewMiddleware, but it can be used on a per view basis.  Using both, or
using the decorator multiple times, is harmless and efficient.
"""


class _EnsureCsrfToken(CsrfViewMiddleware):
    # Behave like CsrfViewMiddleware but don't reject requests or log warnings.
    def _reject(self, request, reason):
        return None


requires_csrf_token = decorator_from_middleware(_EnsureCsrfToken)
requires_csrf_token.__name__ = 'requires_csrf_token'
requires_csrf_token.__doc__ = """
Use this decorator on views that need a correct csrf_token available to
RequestContext, but without the CSRF protection that csrf_protect
enforces.
"""


class _EnsureCsrfCookie(CsrfViewMiddleware):
    def _reject(self, request, reason):
        return None

    def process_view(self, request, callback, callback_args, callback_kwargs):
        retval = super().process_view(request, callback, callback_args, callback_kwargs)
        # Force process_response to send the cookie
        get_token(request)
        return retval


ensure_csrf_cookie = decorator_from_middleware(_EnsureCsrfCookie)
ensure_csrf_cookie.__name__ = 'ensure_csrf_cookie'
ensure_csrf_cookie.__doc__ = """
Use this decorator to ensure that a view sets a CSRF cookie, whether or not it
uses the csrf_token template tag, or the CsrfViewMiddleware is used.
"""


def csrf_exempt(view_func):
    """Mark a view function as being exempt from the CSRF view protection."""
    # view_func.csrf_exempt = True would also work, but decorators are nicer
    # if they don't have side effects, so return a new function.
    def wrapped_view(*args, **kwargs):
        return view_func(*args, **kwargs)
    wrapped_view.csrf_exempt = True
    return wraps(view_func)(wrapped_view)
```
### 10 - django/middleware/csrf.py:

Start line: 104, End line: 149

```python
def rotate_token(request):
    """
    Change the CSRF token in use for a request - should be done on login
    for security purposes.
    """
    request.META.update({
        "CSRF_COOKIE_USED": True,
        "CSRF_COOKIE": _get_new_csrf_token(),
    })
    request.csrf_cookie_needs_reset = True


class InvalidTokenFormat(Exception):
    def __init__(self, reason):
        self.reason = reason


def _sanitize_token(token):
    if len(token) not in (CSRF_TOKEN_LENGTH, CSRF_SECRET_LENGTH):
        raise InvalidTokenFormat(REASON_INCORRECT_LENGTH)
    # Make sure all characters are in CSRF_ALLOWED_CHARS.
    if invalid_token_chars_re.search(token):
        raise InvalidTokenFormat(REASON_INVALID_CHARACTERS)
    if len(token) == CSRF_SECRET_LENGTH:
        # Older Django versions set cookies to values of CSRF_SECRET_LENGTH
        # alphanumeric characters. For backwards compatibility, accept
        # such values as unmasked secrets.
        # It's easier to mask here and be consistent later, rather than add
        # different code paths in the checks, although that might be a tad more
        # efficient.
        return _mask_cipher_secret(token)
    return token


def _compare_masked_tokens(request_csrf_token, csrf_token):
    # Assume both arguments are sanitized -- that is, strings of
    # length CSRF_TOKEN_LENGTH, all CSRF_ALLOWED_CHARS.
    return constant_time_compare(
        _unmask_cipher_token(request_csrf_token),
        _unmask_cipher_token(csrf_token),
    )


class RejectRequest(Exception):
    def __init__(self, reason):
        self.reason = reason
```
### 12 - django/middleware/csrf.py:

Start line: 1, End line: 53

```python
"""
Cross Site Request Forgery Middleware.

This module provides a middleware that implements protection
against request forgeries from other sites.
"""
import logging
import string
from collections import defaultdict
from urllib.parse import urlparse

from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.http.request import HttpHeaders
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile

logger = logging.getLogger('django.security.csrf')
# This matches if any character is not in CSRF_ALLOWED_CHARS.
invalid_token_chars_re = _lazy_re_compile('[^a-zA-Z0-9]')

REASON_BAD_ORIGIN = "Origin checking failed - %s does not match any trusted origins."
REASON_NO_REFERER = "Referer checking failed - no Referer."
REASON_BAD_REFERER = "Referer checking failed - %s does not match any trusted origins."
REASON_NO_CSRF_COOKIE = "CSRF cookie not set."
REASON_CSRF_TOKEN_MISSING = 'CSRF token missing.'
REASON_MALFORMED_REFERER = "Referer checking failed - Referer is malformed."
REASON_INSECURE_REFERER = "Referer checking failed - Referer is insecure while host is secure."
# The reason strings below are for passing to InvalidTokenFormat. They are
# phrases without a subject because they can be in reference to either the CSRF
# cookie or non-cookie token.
REASON_INCORRECT_LENGTH = 'has incorrect length'
REASON_INVALID_CHARACTERS = 'has invalid characters'

CSRF_SECRET_LENGTH = 32
CSRF_TOKEN_LENGTH = 2 * CSRF_SECRET_LENGTH
CSRF_ALLOWED_CHARS = string.ascii_letters + string.digits
CSRF_SESSION_KEY = '_csrftoken'


def _get_failure_view():
    """Return the view to be used for CSRF rejections."""
    return get_callable(settings.CSRF_FAILURE_VIEW)


def _get_new_csrf_string():
    return get_random_string(CSRF_SECRET_LENGTH, allowed_chars=CSRF_ALLOWED_CHARS)
```
### 17 - django/middleware/csrf.py:

Start line: 85, End line: 101

```python
def get_token(request):
    """
    Return the CSRF token required for a POST form. The token is an
    alphanumeric value. A new token is created if one is not already set.

    A side effect of calling this function is to make the csrf_protect
    decorator and the CsrfViewMiddleware add a CSRF cookie and a 'Vary: Cookie'
    header to the outgoing response.  For this reason, you may need to use this
    function lazily, as is done by the csrf context processor.
    """
    if "CSRF_COOKIE" not in request.META:
        csrf_secret = _get_new_csrf_string()
        request.META["CSRF_COOKIE"] = _mask_cipher_secret(csrf_secret)
    else:
        csrf_secret = _unmask_cipher_token(request.META["CSRF_COOKIE"])
    request.META["CSRF_COOKIE_USED"] = True
    return _mask_cipher_secret(csrf_secret)
```
### 19 - django/middleware/csrf.py:

Start line: 68, End line: 82

```python
def _unmask_cipher_token(token):
    """
    Given a token (assumed to be a string of CSRF_ALLOWED_CHARS, of length
    CSRF_TOKEN_LENGTH, and that its first half is a mask), use it to decrypt
    the second half to produce the original secret.
    """
    mask = token[:CSRF_SECRET_LENGTH]
    token = token[CSRF_SECRET_LENGTH:]
    chars = CSRF_ALLOWED_CHARS
    pairs = zip((chars.index(x) for x in token), (chars.index(x) for x in mask))
    return ''.join(chars[x - y] for x, y in pairs)  # Note negative values are ok


def _get_new_csrf_token():
    return _mask_cipher_secret(_get_new_csrf_string())
```
### 33 - django/middleware/csrf.py:

Start line: 56, End line: 65

```python
def _mask_cipher_secret(secret):
    """
    Given a secret (assumed to be a string of CSRF_ALLOWED_CHARS), generate a
    token by adding a mask and applying it to the secret.
    """
    mask = _get_new_csrf_string()
    chars = CSRF_ALLOWED_CHARS
    pairs = zip((chars.index(x) for x in secret), (chars.index(x) for x in mask))
    cipher = ''.join(chars[(x + y) % len(chars)] for x, y in pairs)
    return mask + cipher
```
