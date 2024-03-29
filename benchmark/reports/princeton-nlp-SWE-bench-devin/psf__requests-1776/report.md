# psf__requests-1776

| **psf/requests** | `4bceb312f1b99d36a25f2985b5606e98b6f0d8cd` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 8321 |
| **Any found context length** | 2945 |
| **Avg pos** | 108.66666666666667 |
| **Min pos** | 13 |
| **Max pos** | 136 |
| **Top file pos** | 2 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/auth.py b/requests/auth.py
--- a/requests/auth.py
+++ b/requests/auth.py
@@ -16,6 +16,7 @@
 from base64 import b64encode
 
 from .compat import urlparse, str
+from .cookies import extract_cookies_to_jar
 from .utils import parse_dict_header
 
 log = logging.getLogger(__name__)
@@ -169,7 +170,8 @@ def handle_401(self, r, **kwargs):
             r.content
             r.raw.release_conn()
             prep = r.request.copy()
-            prep.prepare_cookies(r.cookies)
+            extract_cookies_to_jar(prep._cookies, r.request, r.raw)
+            prep.prepare_cookies(prep._cookies)
 
             prep.headers['Authorization'] = self.build_digest_header(
                 prep.method, prep.url)
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -270,6 +270,9 @@ def __init__(self):
         self.url = None
         #: dictionary of HTTP headers.
         self.headers = None
+        # The `CookieJar` used to create the Cookie header will be stored here
+        # after prepare_cookies is called
+        self._cookies = None
         #: request body to send to the server.
         self.body = None
         #: dictionary of callback hooks, for internal usage.
@@ -299,6 +302,7 @@ def copy(self):
         p.method = self.method
         p.url = self.url
         p.headers = self.headers.copy()
+        p._cookies = self._cookies.copy()
         p.body = self.body
         p.hooks = self.hooks
         return p
@@ -474,14 +478,13 @@ def prepare_cookies(self, cookies):
         """Prepares the given HTTP cookie data."""
 
         if isinstance(cookies, cookielib.CookieJar):
-            cookies = cookies
+            self._cookies = cookies
         else:
-            cookies = cookiejar_from_dict(cookies)
+            self._cookies = cookiejar_from_dict(cookies)
 
-        if 'cookie' not in self.headers:
-            cookie_header = get_cookie_header(cookies, self)
-            if cookie_header is not None:
-                self.headers['Cookie'] = cookie_header
+        cookie_header = get_cookie_header(self._cookies, self)
+        if cookie_header is not None:
+            self.headers['Cookie'] = cookie_header
 
     def prepare_hooks(self, hooks):
         """Prepares the given hooks."""
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -153,7 +153,9 @@ def resolve_redirects(self, resp, req, stream=False, timeout=None,
             except KeyError:
                 pass
 
-            prepared_request.prepare_cookies(self.cookies)
+            extract_cookies_to_jar(prepared_request._cookies,
+                                   prepared_request, resp.raw)
+            prepared_request.prepare_cookies(prepared_request._cookies)
 
             resp = self.send(
                 prepared_request,
@@ -345,9 +347,6 @@ def request(self, method, url,
         )
         prep = self.prepare_request(req)
 
-        # Add param cookies to session cookies
-        self.cookies = merge_cookies(self.cookies, cookies)
-
         proxies = proxies or {}
 
         # Gather clues from the surrounding environment.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/auth.py | 19 | 19 | 45 | 5 | 13293
| requests/auth.py | 172 | 172 | 21 | 5 | 5809
| requests/models.py | 273 | 273 | 136 | 4 | 41315
| requests/models.py | 302 | 302 | 63 | 4 | 18521
| requests/models.py | 477 | 484 | 13 | 4 | 2945
| requests/sessions.py | 156 | 156 | 18 | 2 | 4376
| requests/sessions.py | 348 | 350 | 30 | 2 | 8321


## Problem Statement

```
Request cookies should not be persisted to session
After the fix for #1630, cookies sent with a request are now incorrectly persisted to the session.

Specifically, problem lies here: https://github.com/kennethreitz/requests/blob/1511dfa637643bae5b6111a20ecb80ec9ae26032/requests/sessions.py#L330

Removing that breaks the test case for #1630 though, still investigating a solution.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 requests/cookies.py | 61 | 90| 208 | 208 | 3543 | 
| 2 | **2 requests/sessions.py** | 174 | 247| 523 | 731 | 7651 | 
| 3 | **2 requests/sessions.py** | 1 | 35| 226 | 957 | 7651 | 
| 4 | 2 requests/cookies.py | 47 | 59| 124 | 1081 | 7651 | 
| 5 | 2 requests/cookies.py | 131 | 156| 218 | 1299 | 7651 | 
| 6 | 2 requests/cookies.py | 293 | 304| 155 | 1454 | 7651 | 
| 7 | 2 requests/cookies.py | 114 | 128| 150 | 1604 | 7651 | 
| 8 | 2 requests/cookies.py | 1 | 45| 296 | 1900 | 7651 | 
| 9 | 2 requests/cookies.py | 306 | 322| 227 | 2127 | 7651 | 
| 10 | 2 requests/cookies.py | 344 | 376| 233 | 2360 | 7651 | 
| 11 | 2 requests/cookies.py | 324 | 341| 160 | 2520 | 7651 | 
| 12 | 3 requests/__init__.py | 1 | 78| 293 | 2813 | 8147 | 
| **-> 13 <-** | **4 requests/models.py** | 473 | 489| 132 | 2945 | 13334 | 
| 14 | 4 requests/cookies.py | 379 | 403| 230 | 3175 | 13334 | 
| 15 | 4 requests/cookies.py | 185 | 199| 151 | 3326 | 13334 | 
| 16 | 4 requests/cookies.py | 159 | 183| 249 | 3575 | 13334 | 
| 17 | 4 requests/cookies.py | 426 | 446| 156 | 3731 | 13334 | 
| **-> 18 <-** | **4 requests/sessions.py** | 85 | 171| 645 | 4376 | 13334 | 
| 19 | 4 requests/cookies.py | 201 | 291| 766 | 5142 | 13334 | 
| 20 | **4 requests/sessions.py** | 38 | 82| 299 | 5441 | 13334 | 
| **-> 21 <-** | **5 requests/auth.py** | 151 | 195| 368 | 5809 | 14817 | 
| 22 | 6 requests/packages/urllib3/_collections.py | 61 | 74| 152 | 5961 | 15489 | 
| 23 | **6 requests/sessions.py** | 456 | 516| 488 | 6449 | 15489 | 
| 24 | 6 requests/cookies.py | 406 | 423| 163 | 6612 | 15489 | 
| 25 | 6 requests/packages/urllib3/_collections.py | 76 | 104| 174 | 6786 | 15489 | 
| 26 | 7 requests/utils.py | 248 | 271| 137 | 6923 | 19598 | 
| 27 | 7 requests/cookies.py | 93 | 111| 134 | 7057 | 19598 | 
| 28 | 8 requests/packages/urllib3/connectionpool.py | 1 | 57| 245 | 7302 | 24840 | 
| 29 | **8 requests/models.py** | 1 | 37| 261 | 7563 | 24840 | 
| **-> 30 <-** | **8 requests/sessions.py** | 288 | 385| 758 | 8321 | 24840 | 
| 31 | **8 requests/sessions.py** | 249 | 286| 294 | 8615 | 24840 | 
| 32 | 9 requests/compat.py | 1 | 116| 805 | 9420 | 25645 | 
| 33 | 10 requests/exceptions.py | 1 | 64| 286 | 9706 | 25931 | 
| 34 | 11 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 9864 | 28773 | 
| 35 | **11 requests/models.py** | 439 | 471| 273 | 10137 | 28773 | 
| 36 | **11 requests/sessions.py** | 417 | 425| 112 | 10249 | 28773 | 
| 37 | 12 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 14| 0 | 10249 | 28870 | 
| 38 | 13 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 10963 | 29890 | 
| 39 | **13 requests/auth.py** | 58 | 149| 810 | 11773 | 29890 | 
| 40 | **13 requests/sessions.py** | 387 | 415| 252 | 12025 | 29890 | 
| 41 | **13 requests/sessions.py** | 427 | 435| 112 | 12137 | 29890 | 
| 42 | 13 requests/utils.py | 1 | 63| 320 | 12457 | 29890 | 
| 43 | 14 requests/packages/urllib3/util.py | 1 | 48| 269 | 12726 | 34693 | 
| 44 | 15 requests/packages/urllib3/response.py | 275 | 313| 257 | 12983 | 36890 | 
| **-> 45 <-** | **15 requests/auth.py** | 1 | 55| 310 | 13293 | 36890 | 
| 46 | 15 requests/packages/urllib3/connectionpool.py | 467 | 557| 746 | 14039 | 36890 | 
| 47 | 16 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 14288 | 38878 | 
| 48 | 17 requests/packages/urllib3/filepost.py | 1 | 44| 183 | 14471 | 39459 | 
| 49 | **17 requests/sessions.py** | 437 | 454| 185 | 14656 | 39459 | 
| 50 | 18 requests/packages/urllib3/__init__.py | 1 | 38| 180 | 14836 | 39847 | 
| 51 | 19 requests/packages/__init__.py | 1 | 4| 0 | 14836 | 39861 | 
| 52 | 20 requests/packages/urllib3/exceptions.py | 1 | 122| 701 | 15537 | 40615 | 
| 53 | 20 requests/packages/urllib3/util.py | 164 | 178| 137 | 15674 | 40615 | 
| 54 | 20 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 15829 | 40615 | 
| 55 | **20 requests/models.py** | 378 | 437| 398 | 16227 | 40615 | 
| 56 | 21 requests/adapters.py | 1 | 45| 299 | 16526 | 43366 | 
| 57 | 22 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 16526 | 43381 | 
| 58 | 22 requests/packages/urllib3/util.py | 420 | 485| 236 | 16762 | 43381 | 
| 59 | 23 requests/packages/urllib3/connection.py | 1 | 46| 205 | 16967 | 44063 | 
| 60 | **23 requests/sessions.py** | 518 | 554| 257 | 17224 | 44063 | 
| 61 | **23 requests/models.py** | 89 | 146| 447 | 17671 | 44063 | 
| 62 | 24 requests/packages/urllib3/contrib/pyopenssl.py | 103 | 169| 585 | 18256 | 46641 | 
| **-> 63 <-** | **24 requests/models.py** | 278 | 310| 265 | 18521 | 46641 | 
| 64 | 24 requests/packages/urllib3/filepost.py | 47 | 63| 116 | 18637 | 46641 | 
| 65 | 24 requests/packages/urllib3/connection.py | 68 | 108| 297 | 18934 | 46641 | 
| 66 | 24 requests/utils.py | 65 | 100| 263 | 19197 | 46641 | 
| 67 | 24 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 69| 541 | 19738 | 46641 | 
| 68 | 24 requests/packages/urllib3/contrib/pyopenssl.py | 171 | 266| 740 | 20478 | 46641 | 
| 69 | 25 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 20771 | 48798 | 
| 70 | 26 requests/packages/urllib3/fields.py | 161 | 178| 172 | 20943 | 50110 | 
| 71 | 26 requests/packages/urllib3/connectionpool.py | 347 | 378| 203 | 21146 | 50110 | 
| 72 | **26 requests/models.py** | 149 | 170| 161 | 21307 | 50110 | 
| 73 | 26 requests/packages/urllib3/response.py | 1 | 50| 237 | 21544 | 50110 | 
| 74 | 26 requests/packages/urllib3/util.py | 520 | 556| 256 | 21800 | 50110 | 
| 75 | 26 requests/packages/urllib3/util.py | 130 | 162| 269 | 22069 | 50110 | 
| 76 | **26 requests/models.py** | 312 | 376| 478 | 22547 | 50110 | 
| 77 | 27 requests/status_codes.py | 1 | 89| 899 | 23446 | 51009 | 
| 78 | 27 requests/packages/urllib3/contrib/pyopenssl.py | 317 | 347| 252 | 23698 | 51009 | 
| 79 | 27 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 24034 | 51009 | 
| 80 | 27 requests/adapters.py | 216 | 246| 231 | 24265 | 51009 | 
| 81 | 27 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 24493 | 51009 | 
| 82 | 27 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 24627 | 51009 | 
| 83 | 27 requests/packages/urllib3/_collections.py | 1 | 59| 313 | 24940 | 51009 | 
| 84 | 27 requests/adapters.py | 248 | 260| 141 | 25081 | 51009 | 
| 85 | 28 requests/hooks.py | 1 | 46| 188 | 25269 | 51198 | 
| 86 | 28 requests/packages/urllib3/fields.py | 142 | 159| 142 | 25411 | 51198 | 
| 87 | 28 requests/packages/urllib3/contrib/pyopenssl.py | 291 | 314| 141 | 25552 | 51198 | 
| 88 | 28 requests/packages/urllib3/poolmanager.py | 243 | 259| 162 | 25714 | 51198 | 
| 89 | 28 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 26453 | 51198 | 
| 90 | 28 requests/packages/urllib3/connectionpool.py | 260 | 345| 807 | 27260 | 51198 | 
| 91 | 28 requests/packages/urllib3/response.py | 53 | 139| 636 | 27896 | 51198 | 
| 92 | 28 requests/packages/urllib3/util.py | 559 | 593| 242 | 28138 | 51198 | 
| 93 | 28 requests/adapters.py | 48 | 95| 389 | 28527 | 51198 | 
| 94 | 29 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 28571 | 72066 | 
| 95 | 30 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 1 | 64| 539 | 29110 | 72989 | 
| 96 | 30 requests/packages/urllib3/contrib/pyopenssl.py | 72 | 100| 207 | 29317 | 72989 | 
| 97 | 30 requests/utils.py | 471 | 516| 271 | 29588 | 72989 | 
| 98 | 30 requests/utils.py | 408 | 437| 284 | 29872 | 72989 | 
| 99 | 30 requests/packages/urllib3/connectionpool.py | 219 | 258| 298 | 30170 | 72989 | 
| 100 | **30 requests/models.py** | 633 | 657| 167 | 30337 | 72989 | 
| 101 | 30 requests/packages/urllib3/response.py | 242 | 273| 241 | 30578 | 72989 | 
| 102 | 30 requests/utils.py | 519 | 545| 336 | 30914 | 72989 | 
| 103 | 30 requests/packages/urllib3/packages/ordered_dict.py | 92 | 113| 178 | 31092 | 72989 | 
| 104 | 30 requests/packages/urllib3/contrib/pyopenssl.py | 269 | 289| 121 | 31213 | 72989 | 
| 105 | 30 requests/packages/urllib3/packages/ordered_dict.py | 1 | 43| 369 | 31582 | 72989 | 
| 106 | 30 requests/packages/urllib3/util.py | 488 | 517| 218 | 31800 | 72989 | 
| 107 | 30 requests/packages/urllib3/connectionpool.py | 627 | 654| 212 | 32012 | 72989 | 
| 108 | 30 requests/packages/urllib3/util.py | 195 | 217| 196 | 32208 | 72989 | 
| 109 | 31 requests/api.py | 47 | 99| 438 | 32646 | 74062 | 
| 110 | 32 docs/conf.py | 1 | 136| 1059 | 33705 | 75962 | 
| 111 | 32 requests/utils.py | 548 | 585| 222 | 33927 | 75962 | 
| 112 | 32 requests/packages/urllib3/fields.py | 55 | 74| 129 | 34056 | 75962 | 
| 113 | 32 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 34256 | 75962 | 
| 114 | 32 requests/adapters.py | 288 | 378| 587 | 34843 | 75962 | 
| 115 | 33 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 35027 | 76408 | 
| 116 | 34 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 35080 | 103182 | 
| 117 | 34 requests/packages/urllib3/util.py | 51 | 127| 748 | 35828 | 103182 | 
| 118 | 34 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 36403 | 103182 | 
| 119 | 34 requests/packages/urllib3/util.py | 180 | 193| 130 | 36533 | 103182 | 
| 120 | 34 requests/packages/urllib3/connection.py | 48 | 66| 138 | 36671 | 103182 | 
| 121 | 34 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 67 | 106| 350 | 37021 | 103182 | 
| 122 | 35 requests/structures.py | 1 | 34| 167 | 37188 | 104022 | 
| 123 | 36 requests/packages/urllib3/request.py | 1 | 57| 377 | 37565 | 105228 | 
| 124 | 36 requests/packages/urllib3/util.py | 595 | 644| 393 | 37958 | 105228 | 
| 125 | 36 requests/api.py | 102 | 121| 171 | 38129 | 105228 | 
| 126 | 37 requests/packages/charade/big5freq.py | 43 | 926| 52 | 38181 | 151824 | 
| 127 | **37 requests/models.py** | 732 | 771| 248 | 38429 | 151824 | 
| 128 | 37 requests/packages/urllib3/util.py | 219 | 235| 124 | 38553 | 151824 | 
| 129 | **37 requests/models.py** | 173 | 244| 513 | 39066 | 151824 | 
| 130 | 37 requests/packages/urllib3/response.py | 141 | 217| 626 | 39692 | 151824 | 
| 131 | 37 requests/packages/urllib3/connectionpool.py | 599 | 625| 216 | 39908 | 151824 | 
| 132 | 37 requests/api.py | 1 | 44| 423 | 40331 | 151824 | 
| 133 | 37 requests/packages/urllib3/connectionpool.py | 170 | 217| 365 | 40696 | 151824 | 
| 134 | 37 requests/packages/urllib3/util.py | 269 | 299| 233 | 40929 | 151824 | 
| 135 | 37 requests/packages/urllib3/response.py | 219 | 239| 178 | 41107 | 151824 | 
| **-> 136 <-** | **37 requests/models.py** | 247 | 276| 208 | 41315 | 151824 | 
| 137 | 37 requests/packages/urllib3/packages/ordered_dict.py | 174 | 261| 665 | 41980 | 151824 | 
| 138 | 37 requests/adapters.py | 151 | 185| 273 | 42253 | 151824 | 
| 139 | 37 requests/adapters.py | 262 | 286| 209 | 42462 | 151824 | 
| 140 | 37 requests/packages/urllib3/request.py | 59 | 88| 283 | 42745 | 151824 | 
| 141 | 37 requests/adapters.py | 97 | 112| 175 | 42920 | 151824 | 
| 142 | 37 docs/conf.py | 137 | 244| 615 | 43535 | 151824 | 
| 143 | **37 requests/models.py** | 492 | 594| 660 | 44195 | 151824 | 
| 144 | 37 requests/packages/urllib3/connectionpool.py | 560 | 597| 369 | 44564 | 151824 | 
| 145 | 38 requests/packages/charade/compat.py | 21 | 35| 69 | 44633 | 152090 | 
| 146 | 38 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 44778 | 152090 | 
| 147 | 39 requests/certs.py | 1 | 25| 120 | 44898 | 152210 | 
| 148 | **39 requests/models.py** | 40 | 87| 294 | 45192 | 152210 | 
| 149 | **39 requests/models.py** | 659 | 681| 152 | 45344 | 152210 | 
| 150 | 39 requests/packages/urllib3/util.py | 237 | 266| 283 | 45627 | 152210 | 
| 151 | **39 requests/models.py** | 596 | 631| 244 | 45871 | 152210 | 
| 152 | 39 requests/packages/urllib3/request.py | 90 | 143| 508 | 46379 | 152210 | 
| 153 | 39 requests/packages/urllib3/connectionpool.py | 85 | 168| 702 | 47081 | 152210 | 
| 154 | 39 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 47593 | 152210 | 
| 155 | 39 requests/packages/urllib3/fields.py | 76 | 107| 274 | 47867 | 152210 | 
| 156 | 39 requests/packages/urllib3/connectionpool.py | 60 | 83| 183 | 48050 | 152210 | 
| 157 | 39 requests/packages/urllib3/fields.py | 109 | 140| 232 | 48282 | 152210 | 
| 158 | 39 requests/utils.py | 335 | 371| 191 | 48473 | 152210 | 
| 159 | 39 requests/adapters.py | 114 | 149| 269 | 48742 | 152210 | 
| 160 | 39 requests/structures.py | 37 | 109| 558 | 49300 | 152210 | 
| 161 | 39 requests/packages/urllib3/poolmanager.py | 71 | 95| 190 | 49490 | 152210 | 
| 162 | 39 requests/packages/urllib3/packages/ordered_dict.py | 143 | 172| 311 | 49801 | 152210 | 
| 163 | 39 requests/packages/urllib3/connectionpool.py | 380 | 466| 770 | 50571 | 152210 | 
| 164 | 39 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 51101 | 152210 | 
| 165 | 39 requests/packages/urllib3/filepost.py | 66 | 102| 233 | 51334 | 152210 | 
| 166 | 39 requests/packages/urllib3/util.py | 335 | 417| 561 | 51895 | 152210 | 
| 167 | 39 requests/utils.py | 289 | 332| 237 | 52132 | 152210 | 
| 168 | 39 requests/utils.py | 189 | 220| 266 | 52398 | 152210 | 
| 169 | 39 requests/packages/urllib3/util.py | 302 | 332| 226 | 52624 | 152210 | 
| 170 | 39 requests/packages/urllib3/fields.py | 1 | 24| 114 | 52738 | 152210 | 
| 171 | 39 requests/packages/urllib3/fields.py | 27 | 52| 217 | 52955 | 152210 | 
| 172 | 40 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 54231 | 153486 | 
| 173 | 40 requests/utils.py | 440 | 468| 259 | 54490 | 153486 | 
| 174 | 41 requests/packages/charade/euctwfreq.py | 44 | 429| 56 | 54546 | 173976 | 
| 175 | 41 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 55215 | 173976 | 
| 176 | 41 requests/utils.py | 158 | 186| 250 | 55465 | 173976 | 
| 177 | 41 requests/packages/urllib3/poolmanager.py | 97 | 133| 298 | 55763 | 173976 | 


## Patch

```diff
diff --git a/requests/auth.py b/requests/auth.py
--- a/requests/auth.py
+++ b/requests/auth.py
@@ -16,6 +16,7 @@
 from base64 import b64encode
 
 from .compat import urlparse, str
+from .cookies import extract_cookies_to_jar
 from .utils import parse_dict_header
 
 log = logging.getLogger(__name__)
@@ -169,7 +170,8 @@ def handle_401(self, r, **kwargs):
             r.content
             r.raw.release_conn()
             prep = r.request.copy()
-            prep.prepare_cookies(r.cookies)
+            extract_cookies_to_jar(prep._cookies, r.request, r.raw)
+            prep.prepare_cookies(prep._cookies)
 
             prep.headers['Authorization'] = self.build_digest_header(
                 prep.method, prep.url)
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -270,6 +270,9 @@ def __init__(self):
         self.url = None
         #: dictionary of HTTP headers.
         self.headers = None
+        # The `CookieJar` used to create the Cookie header will be stored here
+        # after prepare_cookies is called
+        self._cookies = None
         #: request body to send to the server.
         self.body = None
         #: dictionary of callback hooks, for internal usage.
@@ -299,6 +302,7 @@ def copy(self):
         p.method = self.method
         p.url = self.url
         p.headers = self.headers.copy()
+        p._cookies = self._cookies.copy()
         p.body = self.body
         p.hooks = self.hooks
         return p
@@ -474,14 +478,13 @@ def prepare_cookies(self, cookies):
         """Prepares the given HTTP cookie data."""
 
         if isinstance(cookies, cookielib.CookieJar):
-            cookies = cookies
+            self._cookies = cookies
         else:
-            cookies = cookiejar_from_dict(cookies)
+            self._cookies = cookiejar_from_dict(cookies)
 
-        if 'cookie' not in self.headers:
-            cookie_header = get_cookie_header(cookies, self)
-            if cookie_header is not None:
-                self.headers['Cookie'] = cookie_header
+        cookie_header = get_cookie_header(self._cookies, self)
+        if cookie_header is not None:
+            self.headers['Cookie'] = cookie_header
 
     def prepare_hooks(self, hooks):
         """Prepares the given hooks."""
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -153,7 +153,9 @@ def resolve_redirects(self, resp, req, stream=False, timeout=None,
             except KeyError:
                 pass
 
-            prepared_request.prepare_cookies(self.cookies)
+            extract_cookies_to_jar(prepared_request._cookies,
+                                   prepared_request, resp.raw)
+            prepared_request.prepare_cookies(prepared_request._cookies)
 
             resp = self.send(
                 prepared_request,
@@ -345,9 +347,6 @@ def request(self, method, url,
         )
         prep = self.prepare_request(req)
 
-        # Add param cookies to session cookies
-        self.cookies = merge_cookies(self.cookies, cookies)
-
         proxies = proxies or {}
 
         # Gather clues from the surrounding environment.

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -165,7 +165,7 @@ def test_cookie_quote_wrapped(self):
 
     def test_cookie_persists_via_api(self):
         s = requests.session()
-        r = s.get(httpbin('redirect/1'), cookies={'foo':'bar'})
+        r = s.get(httpbin('redirect/1'), cookies={'foo': 'bar'})
         assert 'foo' in r.request.headers['Cookie']
         assert 'foo' in r.history[0].request.headers['Cookie']
 
@@ -177,6 +177,12 @@ def test_request_cookie_overrides_session_cookie(self):
         # Session cookie should not be modified
         assert s.cookies['foo'] == 'bar'
 
+    def test_request_cookies_not_persisted(self):
+        s = requests.session()
+        s.get(httpbin('cookies'), cookies={'foo': 'baz'})
+        # Sending a request with cookies should not add cookies to the session
+        assert not s.cookies
+
     def test_generic_cookiejar_works(self):
         cj = cookielib.CookieJar()
         cookiejar_from_dict({'foo': 'bar'}, cj)

```


## Code snippets

### 1 - requests/cookies.py:

Start line: 61, End line: 90

```python
class MockRequest(object):

    def is_unverifiable(self):
        return True

    def has_header(self, name):
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name, default=None):
        return self._r.headers.get(name, self._new_headers.get(name, default))

    def add_header(self, key, val):
        """cookielib has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError("Cookie headers should be added with add_unredirected_header()")

    def add_unredirected_header(self, name, value):
        self._new_headers[name] = value

    def get_new_headers(self):
        return self._new_headers

    @property
    def unverifiable(self):
        return self.is_unverifiable()

    @property
    def origin_req_host(self):
        return self.get_origin_req_host()

    @property
    def host(self):
        return self.get_host()
```
### 2 - requests/sessions.py:

Start line: 174, End line: 247

```python
class Session(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('http://httpbin.org/get')
      200
    """

    __attrs__ = [
        'headers', 'cookies', 'auth', 'timeout', 'proxies', 'hooks',
        'params', 'verify', 'cert', 'prefetch', 'adapters', 'stream',
        'trust_env', 'max_redirects']

    def __init__(self):

        #: A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers = default_headers()

        #: Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth = None

        #: Dictionary mapping protocol to the URL of the proxy (e.g.
        #: {'http': 'foo.bar:3128'}) to be used on each
        #: :class:`Request <Request>`.
        self.proxies = {}

        #: Event-handling hooks.
        self.hooks = default_hooks()

        #: Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params = {}

        #: Stream response content default.
        self.stream = False

        #: SSL Verification default.
        self.verify = True

        #: SSL certificate default.
        self.cert = None

        #: Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        self.max_redirects = DEFAULT_REDIRECT_LIMIT

        #: Should we trust the environment?
        self.trust_env = True

        #: A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies = cookiejar_from_dict({})

        # Default connection adapters.
        self.adapters = OrderedDict()
        self.mount('https://', HTTPAdapter())
        self.mount('http://', HTTPAdapter())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```
### 3 - requests/sessions.py:

Start line: 1, End line: 35

```python
# -*- coding: utf-8 -*-

"""
requests.session
~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).

"""
import os
from collections import Mapping
from datetime import datetime

from .compat import cookielib, OrderedDict, urljoin, urlparse, builtin_str
from .cookies import (
    cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar, merge_cookies)
from .models import Request, PreparedRequest
from .hooks import default_hooks, dispatch_hook
from .utils import to_key_val_list, default_headers
from .exceptions import TooManyRedirects, InvalidSchema
from .structures import CaseInsensitiveDict

from .adapters import HTTPAdapter

from .utils import requote_uri, get_environ_proxies, get_netrc_auth

from .status_codes import codes
REDIRECT_STATI = (
    codes.moved, # 301
    codes.found, # 302
    codes.other, # 303
    codes.temporary_moved, # 307
)
DEFAULT_REDIRECT_LIMIT = 30
```
### 4 - requests/cookies.py:

Start line: 47, End line: 59

```python
class MockRequest(object):

    def get_full_url(self):
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get('Host'):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = self._r.headers['Host']
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse([
            parsed.scheme, host, parsed.path, parsed.params, parsed.query,
            parsed.fragment
        ])
```
### 5 - requests/cookies.py:

Start line: 131, End line: 156

```python
def get_cookie_header(jar, request):
    """Produce an appropriate Cookie header string to be sent with `request`, or None."""
    r = MockRequest(request)
    jar.add_cookie_header(r)
    return r.get_new_headers().get('Cookie')


def remove_cookie_by_name(cookiejar, name, domain=None, path=None):
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables = []
    for cookie in cookiejar:
        if cookie.name == name:
            if domain is None or domain == cookie.domain:
                if path is None or path == cookie.path:
                    clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific."""
```
### 6 - requests/cookies.py:

Start line: 293, End line: 304

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):

    def _find(self, name, domain=None, path=None):
        """Requests uses this method internally to get cookie values. Takes as args name
        and optional domain and path. Returns a cookie.value. If there are conflicting cookies,
        _find arbitrarily chooses one. See _find_no_duplicates if you want an exception thrown
        if there are conflicting cookies."""
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value

        raise KeyError('name=%r, domain=%r, path=%r' % (name, domain, path))
```
### 7 - requests/cookies.py:

Start line: 114, End line: 128

```python
def extract_cookies_to_jar(jar, request, response):
    """Extract the cookies from the response into a CookieJar.

    :param jar: cookielib.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, '_original_response') and
            response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)
```
### 8 - requests/cookies.py:

Start line: 1, End line: 45

```python
# -*- coding: utf-8 -*-

"""
Compatibility code to be able to use `cookielib.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import time
import collections
from .compat import cookielib, urlparse, urlunparse, Morsel

try:
    import threading
    # grr, pyflakes: this fixes "redefinition of unused 'threading'"
    threading
except ImportError:
    import dummy_threading as threading


class MockRequest(object):
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `cookielib.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    def __init__(self, request):
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self):
        return self.type

    def get_host(self):
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self):
        return self.get_host()
```
### 9 - requests/cookies.py:

Start line: 306, End line: 322

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):

    def _find_no_duplicates(self, name, domain=None, path=None):
        """__get_item__ and get call _find_no_duplicates -- never used in Requests internally.
        Takes as args name and optional domain and path. Returns a cookie.value.
        Throws KeyError if cookie is not found and CookieConflictError if there are
        multiple cookies that match name and optionally domain and path."""
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:  # if there are multiple cookies that meet passed in criteria
                            raise CookieConflictError('There are multiple cookies with name, %r' % (name))
                        toReturn = cookie.value  # we will eventually return this as long as no cookie conflict

        if toReturn:
            return toReturn
        raise KeyError('name=%r, domain=%r, path=%r' % (name, domain, path))
```
### 10 - requests/cookies.py:

Start line: 344, End line: 376

```python
def create_cookie(name, value, **kwargs):
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    result = dict(
        version=0,
        name=name,
        value=value,
        port=None,
        domain='',
        path='/',
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={'HttpOnly': None},
        rfc2109=False,)

    badargs = set(kwargs) - set(result)
    if badargs:
        err = 'create_cookie() got unexpected keyword arguments: %s'
        raise TypeError(err % list(badargs))

    result.update(kwargs)
    result['port_specified'] = bool(result['port'])
    result['domain_specified'] = bool(result['domain'])
    result['domain_initial_dot'] = result['domain'].startswith('.')
    result['path_specified'] = bool(result['path'])

    return cookielib.Cookie(**result)
```
### 13 - requests/models.py:

Start line: 473, End line: 489

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data."""

        if isinstance(cookies, cookielib.CookieJar):
            cookies = cookies
        else:
            cookies = cookiejar_from_dict(cookies)

        if 'cookie' not in self.headers:
            cookie_header = get_cookie_header(cookies, self)
            if cookie_header is not None:
                self.headers['Cookie'] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        for event in hooks:
            self.register_hook(event, hooks[event])
```
### 18 - requests/sessions.py:

Start line: 85, End line: 171

```python
class SessionRedirectMixin(object):
    def resolve_redirects(self, resp, req, stream=False, timeout=None,
                          verify=True, cert=None, proxies=None):
        """Receives a Response. Returns a generator of Responses."""

        i = 0

        # ((resp.status_code is codes.see_other))
        while ('location' in resp.headers and resp.status_code in REDIRECT_STATI):
            prepared_request = req.copy()

            resp.content  # Consume socket so it can be released

            if i >= self.max_redirects:
                raise TooManyRedirects('Exceeded %s redirects.' % self.max_redirects)

            # Release the connection back into the pool.
            resp.close()

            url = resp.headers['location']
            method = req.method

            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if url.startswith('//'):
                parsed_rurl = urlparse(resp.url)
                url = '%s:%s' % (parsed_rurl.scheme, url)

            # The scheme should be lower case...
            parsed = urlparse(url)
            url = parsed.geturl()

            # Facilitate non-RFC2616-compliant 'location' headers
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not urlparse(url).netloc:
                url = urljoin(resp.url, requote_uri(url))
            else:
                url = requote_uri(url)

            prepared_request.url = url

            # http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3.4
            if (resp.status_code == codes.see_other and
                    method != 'HEAD'):
                method = 'GET'

            # Do what the browsers do, despite standards...
            # First, turn 302s into GETs.
            if resp.status_code == codes.found and method != 'HEAD':
                method = 'GET'

            # Second, if a POST is responded to with a 301, turn it into a GET.
            # This bizarre behaviour is explained in Issue 1704.
            if resp.status_code == codes.moved and method == 'POST':
                method = 'GET'

            prepared_request.method = method

            # https://github.com/kennethreitz/requests/issues/1084
            if resp.status_code not in (codes.temporary, codes.resume):
                if 'Content-Length' in prepared_request.headers:
                    del prepared_request.headers['Content-Length']

                prepared_request.body = None

            headers = prepared_request.headers
            try:
                del headers['Cookie']
            except KeyError:
                pass

            prepared_request.prepare_cookies(self.cookies)

            resp = self.send(
                prepared_request,
                stream=stream,
                timeout=timeout,
                verify=verify,
                cert=cert,
                proxies=proxies,
                allow_redirects=False,
            )

            extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)

            i += 1
            yield resp
```
### 20 - requests/sessions.py:

Start line: 38, End line: 82

```python
def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
    """
    Determines appropriate setting for a given request, taking into account the
    explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`
    """

    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
            isinstance(session_setting, Mapping) and
            isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))

    # Remove keys that are set to None.
    for (k, v) in request_setting.items():
        if v is None:
            del merged_setting[k]

    return merged_setting


def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """
    Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get('response') == []:
        return request_hooks

    if request_hooks is None or request_hooks.get('response') == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)
```
### 21 - requests/auth.py:

Start line: 151, End line: 195

```python
class HTTPDigestAuth(AuthBase):

    def handle_401(self, r, **kwargs):
        """Takes the given response and tries digest-auth, if needed."""

        if self.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            r.request.body.seek(self.pos)
        num_401_calls = getattr(self, 'num_401_calls', 1)
        s_auth = r.headers.get('www-authenticate', '')

        if 'digest' in s_auth.lower() and num_401_calls < 2:

            setattr(self, 'num_401_calls', num_401_calls + 1)
            pat = re.compile(r'digest ', flags=re.IGNORECASE)
            self.chal = parse_dict_header(pat.sub('', s_auth, count=1))

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.raw.release_conn()
            prep = r.request.copy()
            prep.prepare_cookies(r.cookies)

            prep.headers['Authorization'] = self.build_digest_header(
                prep.method, prep.url)
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep

            return _r

        setattr(self, 'num_401_calls', 1)
        return r

    def __call__(self, r):
        # If we have a saved nonce, skip the 401
        if self.last_nonce:
            r.headers['Authorization'] = self.build_digest_header(r.method, r.url)
        try:
            self.pos = r.body.tell()
        except AttributeError:
            pass
        r.register_hook('response', self.handle_401)
        return r
```
### 23 - requests/sessions.py:

Start line: 456, End line: 516

```python
class Session(SessionRedirectMixin):

    def send(self, request, **kwargs):
        """Send a given PreparedRequest."""
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        kwargs.setdefault('stream', self.stream)
        kwargs.setdefault('verify', self.verify)
        kwargs.setdefault('cert', self.cert)
        kwargs.setdefault('proxies', self.proxies)

        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if not isinstance(request, PreparedRequest):
            raise ValueError('You can only send PreparedRequests.')

        # Set up variables needed for resolve_redirects and dispatching of
        # hooks
        allow_redirects = kwargs.pop('allow_redirects', True)
        stream = kwargs.get('stream')
        timeout = kwargs.get('timeout')
        verify = kwargs.get('verify')
        cert = kwargs.get('cert')
        proxies = kwargs.get('proxies')
        hooks = request.hooks

        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)

        # Start time (approximately) of the request
        start = datetime.utcnow()
        # Send the request
        r = adapter.send(request, **kwargs)
        # Total elapsed time of the request (approximately)
        r.elapsed = datetime.utcnow() - start

        # Response manipulation hooks
        r = dispatch_hook('response', hooks, r, **kwargs)

        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)
        extract_cookies_to_jar(self.cookies, request, r.raw)

        # Redirect resolving generator.
        gen = self.resolve_redirects(r, request, stream=stream,
                                     timeout=timeout, verify=verify, cert=cert,
                                     proxies=proxies)

        # Resolve redirects if allowed.
        history = [resp for resp in gen] if allow_redirects else []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = tuple(history)

        return r
```
### 29 - requests/models.py:

Start line: 1, End line: 37

```python
# -*- coding: utf-8 -*-

"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import collections
import logging
import datetime

from io import BytesIO, UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict

from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header
from .packages.urllib3.fields import RequestField
from .packages.urllib3.filepost import encode_multipart_formdata
from .packages.urllib3.util import parse_url
from .exceptions import (
    HTTPError, RequestException, MissingSchema, InvalidURL,
    ChunkedEncodingError)
from .utils import (
    guess_filename, get_auth_from_url, requote_uri,
    stream_decode_response_unicode, to_key_val_list, parse_header_links,
    iter_slices, guess_json_utf, super_len, to_native_string)
from .compat import (
    cookielib, urlunparse, urlsplit, urlencode, str, bytes, StringIO,
    is_py2, chardet, json, builtin_str, basestring, IncompleteRead)

CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512

log = logging.getLogger(__name__)
```
### 30 - requests/sessions.py:

Start line: 288, End line: 385

```python
class Session(SessionRedirectMixin):

    def request(self, method, url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary or bytes to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of 'filename': file-like-objects
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) Float describing the timeout of the
            request.
        :param allow_redirects: (optional) Boolean. Set to True by default.
        :param proxies: (optional) Dictionary mapping protocol to the URL of
            the proxy.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) if ``True``, the SSL cert will be verified.
            A CA_BUNDLE path can also be provided.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        """

        method = builtin_str(method)

        # Create the Request.
        req = Request(
            method = method.upper(),
            url = url,
            headers = headers,
            files = files,
            data = data or {},
            params = params or {},
            auth = auth,
            cookies = cookies,
            hooks = hooks,
        )
        prep = self.prepare_request(req)

        # Add param cookies to session cookies
        self.cookies = merge_cookies(self.cookies, cookies)

        proxies = proxies or {}

        # Gather clues from the surrounding environment.
        if self.trust_env:
            # Set environment's proxies.
            env_proxies = get_environ_proxies(url) or {}
            for (k, v) in env_proxies.items():
                proxies.setdefault(k, v)

            # Look for configuration.
            if not verify and verify is not False:
                verify = os.environ.get('REQUESTS_CA_BUNDLE')

            # Curl compatibility.
            if not verify and verify is not False:
                verify = os.environ.get('CURL_CA_BUNDLE')

        # Merge all the kwargs.
        proxies = merge_setting(proxies, self.proxies)
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)

        # Send the request.
        send_kwargs = {
            'stream': stream,
            'timeout': timeout,
            'verify': verify,
            'cert': cert,
            'proxies': proxies,
            'allow_redirects': allow_redirects,
        }
        resp = self.send(prep, **send_kwargs)

        return resp
```
### 31 - requests/sessions.py:

Start line: 249, End line: 286

```python
class Session(SessionRedirectMixin):

    def prepare_request(self, request):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
        session's settings.
        """
        cookies = request.cookies or {}

        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)

        # Merge with session cookies
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self.cookies), cookies)


        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(request.url)

        p = PreparedRequest()
        p.prepare(
            method=request.method.upper(),
            url=request.url,
            files=request.files,
            data=request.data,
            headers=merge_setting(request.headers, self.headers, dict_class=CaseInsensitiveDict),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p
```
### 35 - requests/models.py:

Start line: 439, End line: 471

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_content_length(self, body):
        if hasattr(body, 'seek') and hasattr(body, 'tell'):
            body.seek(0, 2)
            self.headers['Content-Length'] = builtin_str(body.tell())
            body.seek(0, 0)
        elif body is not None:
            l = super_len(body)
            if l:
                self.headers['Content-Length'] = builtin_str(l)
        elif self.method not in ('GET', 'HEAD'):
            self.headers['Content-Length'] = '0'

    def prepare_auth(self, auth, url=''):
        """Prepares the given HTTP auth data."""

        # If no Auth is explicitly provided, extract it from the URL first.
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None

        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                # special-case basic HTTP auth
                auth = HTTPBasicAuth(*auth)

            # Allow auth to make its changes.
            r = auth(self)

            # Update self to reflect the auth changes.
            self.__dict__.update(r.__dict__)

            # Recompute Content-Length
            self.prepare_content_length(self.body)
```
### 36 - requests/sessions.py:

Start line: 417, End line: 425

```python
class Session(SessionRedirectMixin):

    def post(self, url, data=None, **kwargs):
        """Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        return self.request('POST', url, data=data, **kwargs)
```
### 39 - requests/auth.py:

Start line: 58, End line: 149

```python
class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.last_nonce = ''
        self.nonce_count = 0
        self.chal = {}
        self.pos = None

    def build_digest_header(self, method, url):

        realm = self.chal['realm']
        nonce = self.chal['nonce']
        qop = self.chal.get('qop')
        algorithm = self.chal.get('algorithm')
        opaque = self.chal.get('opaque')

        if algorithm is None:
            _algorithm = 'MD5'
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == 'MD5' or _algorithm == 'MD5-SESS':
            def md5_utf8(x):
                if isinstance(x, str):
                    x = x.encode('utf-8')
                return hashlib.md5(x).hexdigest()
            hash_utf8 = md5_utf8
        elif _algorithm == 'SHA':
            def sha_utf8(x):
                if isinstance(x, str):
                    x = x.encode('utf-8')
                return hashlib.sha1(x).hexdigest()
            hash_utf8 = sha_utf8

        KD = lambda s, d: hash_utf8("%s:%s" % (s, d))

        if hash_utf8 is None:
            return None

        # XXX not implemented yet
        entdig = None
        p_parsed = urlparse(url)
        path = p_parsed.path
        if p_parsed.query:
            path += '?' + p_parsed.query

        A1 = '%s:%s:%s' % (self.username, realm, self.password)
        A2 = '%s:%s' % (method, path)

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == self.last_nonce:
            self.nonce_count += 1
        else:
            self.nonce_count = 1
        ncvalue = '%08x' % self.nonce_count
        s = str(self.nonce_count).encode('utf-8')
        s += nonce.encode('utf-8')
        s += time.ctime().encode('utf-8')
        s += os.urandom(8)

        cnonce = (hashlib.sha1(s).hexdigest()[:16])
        noncebit = "%s:%s:%s:%s:%s" % (nonce, ncvalue, cnonce, qop, HA2)
        if _algorithm == 'MD5-SESS':
            HA1 = hash_utf8('%s:%s:%s' % (HA1, nonce, cnonce))

        if qop is None:
            respdig = KD(HA1, "%s:%s" % (nonce, HA2))
        elif qop == 'auth' or 'auth' in qop.split(','):
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

        self.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = 'username="%s", realm="%s", nonce="%s", uri="%s", ' \
               'response="%s"' % (self.username, realm, nonce, path, respdig)
        if opaque:
            base += ', opaque="%s"' % opaque
        if algorithm:
            base += ', algorithm="%s"' % algorithm
        if entdig:
            base += ', digest="%s"' % entdig
        if qop:
            base += ', qop="auth", nc=%s, cnonce="%s"' % (ncvalue, cnonce)

        return 'Digest %s' % (base)
```
### 40 - requests/sessions.py:

Start line: 387, End line: 415

```python
class Session(SessionRedirectMixin):

    def get(self, url, **kwargs):
        """Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        kwargs.setdefault('allow_redirects', True)
        return self.request('GET', url, **kwargs)

    def options(self, url, **kwargs):
        """Sends a OPTIONS request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        kwargs.setdefault('allow_redirects', True)
        return self.request('OPTIONS', url, **kwargs)

    def head(self, url, **kwargs):
        """Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        kwargs.setdefault('allow_redirects', False)
        return self.request('HEAD', url, **kwargs)
```
### 41 - requests/sessions.py:

Start line: 427, End line: 435

```python
class Session(SessionRedirectMixin):

    def put(self, url, data=None, **kwargs):
        """Sends a PUT request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        return self.request('PUT', url, data=data, **kwargs)
```
### 45 - requests/auth.py:

Start line: 1, End line: 55

```python
# -*- coding: utf-8 -*-

"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

import os
import re
import time
import hashlib
import logging

from base64 import b64encode

from .compat import urlparse, str
from .utils import parse_dict_header

log = logging.getLogger(__name__)

CONTENT_TYPE_FORM_URLENCODED = 'application/x-www-form-urlencoded'
CONTENT_TYPE_MULTI_PART = 'multipart/form-data'


def _basic_auth_str(username, password):
    """Returns a Basic Auth string."""

    return 'Basic ' + b64encode(('%s:%s' % (username, password)).encode('latin1')).strip().decode('latin1')


class AuthBase(object):
    """Base class that all auth implementations derive from"""

    def __call__(self, r):
        raise NotImplementedError('Auth hooks must be callable.')


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __call__(self, r):
        r.headers['Authorization'] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""
    def __call__(self, r):
        r.headers['Proxy-Authorization'] = _basic_auth_str(self.username, self.password)
        return r
```
### 49 - requests/sessions.py:

Start line: 437, End line: 454

```python
class Session(SessionRedirectMixin):

    def patch(self, url, data=None, **kwargs):
        """Sends a PATCH request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        return self.request('PATCH', url,  data=data, **kwargs)

    def delete(self, url, **kwargs):
        """Sends a DELETE request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        return self.request('DELETE', url, **kwargs)
```
### 55 - requests/models.py:

Start line: 378, End line: 437

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        if headers:
            self.headers = CaseInsensitiveDict((to_native_string(name), value) for name, value in headers.items())
        else:
            self.headers = CaseInsensitiveDict()

    def prepare_body(self, data, files):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None
        length = None

        is_stream = all([
            hasattr(data, '__iter__'),
            not isinstance(data, basestring),
            not isinstance(data, list),
            not isinstance(data, dict)
        ])

        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None

        if is_stream:
            body = data

            if files:
                raise NotImplementedError('Streamed bodies and files are mutually exclusive.')

            if length is not None:
                self.headers['Content-Length'] = builtin_str(length)
            else:
                self.headers['Transfer-Encoding'] = 'chunked'
        else:
            # Multi-part file uploads.
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, str) or isinstance(data, builtin_str) or hasattr(data, 'read'):
                        content_type = None
                    else:
                        content_type = 'application/x-www-form-urlencoded'

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if (content_type) and (not 'content-type' in self.headers):
                self.headers['Content-Type'] = content_type

        self.body = body
```
### 60 - requests/sessions.py:

Start line: 518, End line: 554

```python
class Session(SessionRedirectMixin):

    def get_adapter(self, url):
        """Returns the appropriate connnection adapter for the given URL."""
        for (prefix, adapter) in self.adapters.items():

            if url.lower().startswith(prefix):
                return adapter

        # Nothing matches :-/
        raise InvalidSchema("No connection adapters were found for '%s'" % url)

    def close(self):
        """Closes all adapters and as such the session"""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix, adapter):
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by key length."""
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]
        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self):
        return dict((attr, getattr(self, attr, None)) for attr in self.__attrs__)

    def __setstate__(self, state):
        for attr, value in state.items():
            setattr(self, attr, value)


def session():
    """Returns a :class:`Session` for context-management."""

    return Session()
```
### 61 - requests/models.py:

Start line: 89, End line: 146

```python
class RequestEncodingMixin(object):

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.

        """
        if (not files):
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, '__iter__'):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)

                    new_fields.append(
                        (field.decode('utf-8') if isinstance(field, bytes) else field,
                         v.encode('utf-8') if isinstance(v, str) else v))

        for (k, v) in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v
            if isinstance(fp, str):
                fp = StringIO(fp)
            if isinstance(fp, bytes):
                fp = BytesIO(fp)

            rf = RequestField(name=k, data=fp.read(),
                              filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type
```
### 63 - requests/models.py:

Start line: 278, End line: 310

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare(self, method=None, url=None, headers=None, files=None,
                data=None, params=None, auth=None, cookies=None, hooks=None):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files)
        self.prepare_auth(auth, url)
        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self):
        return '<PreparedRequest [%s]>' % (self.method)

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy()
        p.body = self.body
        p.hooks = self.hooks
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = self.method.upper()
```
### 72 - requests/models.py:

Start line: 149, End line: 170

```python
class RequestHooksMixin(object):
    def register_hook(self, event, hook):
        """Properly register a hook."""

        if event not in self.hooks:
            raise ValueError('Unsupported event specified, with event name "%s"' % (event))

        if isinstance(hook, collections.Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, '__iter__'):
            self.hooks[event].extend(h for h in hook if isinstance(h, collections.Callable))

    def deregister_hook(self, event, hook):
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """

        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False
```
### 76 - requests/models.py:

Start line: 312, End line: 376

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        try:
            url = unicode(url)
        except NameError:
            # We're on Python 3.
            url = str(url)
        except UnicodeDecodeError:
            pass

        # Don't do any URL preparation for oddball schemes
        if ':' in url and not url.lower().startswith('http'):
            self.url = url
            return

        # Support for unicode domain names and paths.
        scheme, auth, host, port, path, query, fragment = parse_url(url)

        if not scheme:
            raise MissingSchema("Invalid URL {0!r}: No schema supplied. "
                                "Perhaps you meant http://{0}?".format(url))

        if not host:
            raise InvalidURL("Invalid URL %r: No host supplied" % url)

        # Only want to apply IDNA to the hostname
        try:
            host = host.encode('idna').decode('utf-8')
        except UnicodeError:
            raise InvalidURL('URL has an invalid label.')

        # Carefully reconstruct the network location
        netloc = auth or ''
        if netloc:
            netloc += '@'
        netloc += host
        if port:
            netloc += ':' + str(port)

        # Bare domains aren't valid URLs.
        if not path:
            path = '/'

        if is_py2:
            if isinstance(scheme, str):
                scheme = scheme.encode('utf-8')
            if isinstance(netloc, str):
                netloc = netloc.encode('utf-8')
            if isinstance(path, str):
                path = path.encode('utf-8')
            if isinstance(query, str):
                query = query.encode('utf-8')
            if isinstance(fragment, str):
                fragment = fragment.encode('utf-8')

        enc_params = self._encode_params(params)
        if enc_params:
            if query:
                query = '%s&%s' % (query, enc_params)
            else:
                query = enc_params

        url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
        self.url = url
```
### 100 - requests/models.py:

Start line: 633, End line: 657

```python
class Response(object):

    def iter_lines(self, chunk_size=ITER_CHUNK_SIZE, decode_unicode=None):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.
        """

        pending = None

        for chunk in self.iter_content(chunk_size=chunk_size,
                                       decode_unicode=decode_unicode):

            if pending is not None:
                chunk = pending + chunk
            lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                yield line

        if pending is not None:
            yield pending
```
### 127 - requests/models.py:

Start line: 732, End line: 771

```python
class Response(object):

    @property
    def links(self):
        """Returns the parsed header links of the response, if any."""

        header = self.headers.get('link')

        # l = MultiDict()
        l = {}

        if header:
            links = parse_header_links(header)

            for link in links:
                key = link.get('rel') or link.get('url')
                l[key] = link

        return l

    def raise_for_status(self):
        """Raises stored :class:`HTTPError`, if one occurred."""

        http_error_msg = ''

        if 400 <= self.status_code < 500:
            http_error_msg = '%s Client Error: %s' % (self.status_code, self.reason)

        elif 500 <= self.status_code < 600:
            http_error_msg = '%s Server Error: %s' % (self.status_code, self.reason)

        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)

    def close(self):
        """Closes the underlying file descriptor and releases the connection
        back to the pool.

        *Note: Should not normally need to be called explicitly.*
        """
        return self.raw.release_conn()
```
### 129 - requests/models.py:

Start line: 173, End line: 244

```python
class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach the request. If a dictionary is provided, form-encoding will take place.
    :param params: dictionary of URL parameters to append to the URL.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'http://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>

    """
    def __init__(self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None):

        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks

        self.hooks = default_hooks()
        for (k, v) in list(hooks.items()):
            self.register_hook(event=k, hook=v)

        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self):
        return '<Request [%s]>' % (self.method)

    def prepare(self):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p
```
### 136 - requests/models.py:

Start line: 247, End line: 276

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Generated from either a :class:`Request <Request>` object or manually.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'http://httpbin.org/get')
      >>> r = req.prepare()
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>

    """

    def __init__(self):
        #: HTTP verb to send to the server.
        self.method = None
        #: HTTP URL to send the request to.
        self.url = None
        #: dictionary of HTTP headers.
        self.headers = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
```
### 143 - requests/models.py:

Start line: 492, End line: 594

```python
class Response(object):
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

    __attrs__ = [
        '_content',
        'status_code',
        'headers',
        'url',
        'history',
        'encoding',
        'reason',
        'cookies',
        'elapsed',
        'request',
    ]

    def __init__(self):
        super(Response, self).__init__()

        self._content = False
        self._content_consumed = False

        #: Integer Code of responded HTTP Status.
        self.status_code = None

        #: Case-insensitive Dictionary of Response Headers.
        #: For example, ``headers['content-encoding']`` will return the
        #: value of a ``'Content-Encoding'`` response header.
        self.headers = CaseInsensitiveDict()

        #: File-like object representation of response (for advanced usage).
        #: Requires that ``stream=True` on the request.
        # This requirement does not apply for use internally to Requests.
        self.raw = None

        #: Final URL location of Response.
        self.url = None

        #: Encoding to decode with when accessing r.text.
        self.encoding = None

        #: A list of :class:`Response <Response>` objects from
        #: the history of the Request. Any redirect responses will end
        #: up here. The list is sorted from the oldest to the most recent request.
        self.history = []

        self.reason = None

        #: A CookieJar of Cookies the server sent back.
        self.cookies = cookiejar_from_dict({})

        #: The amount of time elapsed between sending the request
        #: and the arrival of the response (as a timedelta)
        self.elapsed = datetime.timedelta(0)

    def __getstate__(self):
        # Consume everything; accessing the content attribute makes
        # sure the content has been fully read.
        if not self._content_consumed:
            self.content

        return dict(
            (attr, getattr(self, attr, None))
            for attr in self.__attrs__
        )

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # pickled objects do not have .raw
        setattr(self, '_content_consumed', True)

    def __repr__(self):
        return '<Response [%s]>' % (self.status_code)

    def __bool__(self):
        """Returns true if :attr:`status_code` is 'OK'."""
        return self.ok

    def __nonzero__(self):
        """Returns true if :attr:`status_code` is 'OK'."""
        return self.ok

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def ok(self):
        try:
            self.raise_for_status()
        except RequestException:
            return False
        return True

    @property
    def apparent_encoding(self):
        """The apparent encoding, provided by the lovely Charade library
        (Thanks, Ian!)."""
        return chardet.detect(self.content)['encoding']
```
### 148 - requests/models.py:

Start line: 40, End line: 87

```python
class RequestEncodingMixin(object):
    @property
    def path_url(self):
        """Build the path URL to use."""

        url = []

        p = urlsplit(self.url)

        path = p.path
        if not path:
            path = '/'

        url.append(path)

        query = p.query
        if query:
            url.append('?')
            url.append(query)

        return ''.join(url)

    @staticmethod
    def _encode_params(data):
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, 'read'):
            return data
        elif hasattr(data, '__iter__'):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (k.encode('utf-8') if isinstance(k, str) else k,
                             v.encode('utf-8') if isinstance(v, str) else v))
            return urlencode(result, doseq=True)
        else:
            return data
```
### 149 - requests/models.py:

Start line: 659, End line: 681

```python
class Response(object):

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            try:
                if self._content_consumed:
                    raise RuntimeError(
                        'The content for this response was already consumed')

                if self.status_code == 0:
                    self._content = None
                else:
                    self._content = bytes().join(self.iter_content(CONTENT_CHUNK_SIZE)) or bytes()

            except AttributeError:
                self._content = None

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content
```
### 151 - requests/models.py:

Start line: 596, End line: 631

```python
class Response(object):

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.
        """
        if self._content_consumed:
            # simulate reading small chunks of the content
            return iter_slices(self._content, chunk_size)

        def generate():
            try:
                # Special case for urllib3.
                try:
                    for chunk in self.raw.stream(chunk_size,
                                                 decode_content=True):
                        yield chunk
                except IncompleteRead as e:
                    raise ChunkedEncodingError(e)
            except AttributeError:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        gen = generate()

        if decode_unicode:
            gen = stream_decode_response_unicode(gen, self)

        return gen
```
