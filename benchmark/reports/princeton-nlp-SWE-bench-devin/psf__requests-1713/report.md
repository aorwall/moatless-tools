# psf__requests-1713

| **psf/requests** | `340b2459031feb421d678c3c75865c3b11c07938` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 2716 |
| **Any found context length** | 2716 |
| **Avg pos** | 103.5 |
| **Min pos** | 11 |
| **Max pos** | 96 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/cookies.py b/requests/cookies.py
--- a/requests/cookies.py
+++ b/requests/cookies.py
@@ -421,3 +421,25 @@ def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
                 cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
 
     return cookiejar
+
+
+def merge_cookies(cookiejar, cookies):
+    """Add cookies to cookiejar and returns a merged CookieJar.
+
+    :param cookiejar: CookieJar object to add the cookies to.
+    :param cookies: Dictionary or CookieJar object to be added.
+    """
+    if not isinstance(cookiejar, cookielib.CookieJar):
+        raise ValueError('You can only merge into CookieJar')
+    
+    if isinstance(cookies, dict):
+        cookiejar = cookiejar_from_dict(
+            cookies, cookiejar=cookiejar, overwrite=False)
+    elif isinstance(cookies, cookielib.CookieJar):
+        try:
+            cookiejar.update(cookies)
+        except AttributeError:
+            for cookie_in_jar in cookies:
+                cookiejar.set_cookie(cookie_in_jar)
+
+    return cookiejar
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -13,7 +13,8 @@
 from datetime import datetime
 
 from .compat import cookielib, OrderedDict, urljoin, urlparse, builtin_str
-from .cookies import cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar
+from .cookies import (
+    cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar, merge_cookies)
 from .models import Request, PreparedRequest
 from .hooks import default_hooks, dispatch_hook
 from .utils import to_key_val_list, default_headers
@@ -245,9 +246,8 @@ def prepare_request(self, request):
             cookies = cookiejar_from_dict(cookies)
 
         # Merge with session cookies
-        merged_cookies = RequestsCookieJar()
-        merged_cookies.update(self.cookies)
-        merged_cookies.update(cookies)
+        merged_cookies = merge_cookies(
+            merge_cookies(RequestsCookieJar(), self.cookies), cookies)
 
 
         # Set environment's basic authentication if not explicitly set.
@@ -330,7 +330,7 @@ def request(self, method, url,
         prep = self.prepare_request(req)
 
         # Add param cookies to session cookies
-        self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)
+        self.cookies = merge_cookies(self.cookies, cookies)
 
         proxies = proxies or {}
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/cookies.py | 424 | 424 | 11 | 1 | 2716
| requests/sessions.py | 16 | 16 | 21 | 6 | 5377
| requests/sessions.py | 248 | 250 | 96 | 6 | 31619
| requests/sessions.py | 333 | 333 | 79 | 6 | 24492


## Problem Statement

```
Regression 2.0.1: Using MozillaCookieJar does not work
Could not find an issue raised for this, not sure if this was an expected change either. This is reproducible on master.

Existing code fails on update to `requests-2.0.1`. The cause seems to be triggered by the change at https://github.com/kennethreitz/requests/commit/012f0334ce43fe23044fc58e4246a804db88650d#diff-28e67177469c0d36b068d68d9f6043bfR326

The parameter `cookies` expects either `Dict` or `CookieJar`. Treating `MozillaCookieJar` as a dict triggers the error in this instance.

The following code highlights the issue:

\`\`\` py
import sys
import requests
from os.path import expanduser

if sys.version_info.major >= 3:
    from http.cookiejar import MozillaCookieJar
else:
    from cookielib import MozillaCookieJar

URL = 'https://bugzilla.redhat.com'
COOKIE_FILE = expanduser('~/.bugzillacookies')

cookiejar = MozillaCookieJar(COOKIE_FILE)
cookiejar.load()

requests.get(URL, cookies=cookiejar)
\`\`\`

The following `AttributeError` is thrown:

\`\`\`
Traceback (most recent call last):
  File "rtest.py", line 16, in <module>
    requests.get(URL, cookies=cookiejar)
  File "/tmp/rtestenv/lib/python2.7/site-packages/requests/api.py", line 55, in get
    return request('get', url, **kwargs)
  File "/tmp/rtestenv/lib/python2.7/site-packages/requests/api.py", line 44, in request
    return session.request(method=method, url=url, **kwargs)
  File "/tmp/rtestenv/lib/python2.7/site-packages/requests/sessions.py", line 327, in request
    self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)
  File "/tmp/rtestenv/lib/python2.7/site-packages/requests/cookies.py", line 410, in cookiejar_from_dict
    cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
AttributeError: MozillaCookieJar instance has no attribute '__getitem__'
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 requests/cookies.py** | 1 | 45| 296 | 296 | 3385 | 
| 2 | **1 requests/cookies.py** | 61 | 90| 208 | 504 | 3385 | 
| 3 | **1 requests/cookies.py** | 131 | 156| 218 | 722 | 3385 | 
| 4 | **1 requests/cookies.py** | 114 | 128| 150 | 872 | 3385 | 
| 5 | **1 requests/cookies.py** | 324 | 341| 160 | 1032 | 3385 | 
| 6 | **1 requests/cookies.py** | 201 | 291| 766 | 1798 | 3385 | 
| 7 | **1 requests/cookies.py** | 293 | 304| 155 | 1953 | 3385 | 
| 8 | **1 requests/cookies.py** | 159 | 183| 249 | 2202 | 3385 | 
| 9 | **1 requests/cookies.py** | 47 | 59| 124 | 2326 | 3385 | 
| 10 | **1 requests/cookies.py** | 306 | 322| 227 | 2553 | 3385 | 
| **-> 11 <-** | **1 requests/cookies.py** | 406 | 424| 163 | 2716 | 3385 | 
| 12 | **1 requests/cookies.py** | 185 | 199| 151 | 2867 | 3385 | 
| 13 | 2 requests/compat.py | 1 | 116| 805 | 3672 | 4190 | 
| 14 | 3 requests/utils.py | 248 | 271| 137 | 3809 | 8297 | 
| 15 | 4 requests/__init__.py | 1 | 78| 293 | 4102 | 8793 | 
| 16 | **4 requests/cookies.py** | 344 | 376| 233 | 4335 | 8793 | 
| 17 | **4 requests/cookies.py** | 379 | 403| 230 | 4565 | 8793 | 
| 18 | **4 requests/cookies.py** | 93 | 111| 134 | 4699 | 8793 | 
| 19 | 5 requests/models.py | 473 | 489| 132 | 4831 | 13980 | 
| 20 | 5 requests/utils.py | 1 | 63| 325 | 5156 | 13980 | 
| **-> 21 <-** | **6 requests/sessions.py** | 1 | 34| 221 | 5377 | 17988 | 
| 22 | **6 requests/sessions.py** | 157 | 230| 523 | 5900 | 17988 | 
| 23 | 7 requests/packages/urllib3/connectionpool.py | 1 | 57| 245 | 6145 | 23230 | 
| 24 | 8 requests/packages/urllib3/util.py | 1 | 48| 269 | 6414 | 28033 | 
| 25 | 8 requests/models.py | 1 | 37| 261 | 6675 | 28033 | 
| 26 | 9 requests/packages/urllib3/_collections.py | 67 | 95| 174 | 6849 | 28658 | 
| 27 | 10 requests/packages/urllib3/exceptions.py | 1 | 122| 701 | 7550 | 29412 | 
| 28 | 11 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 7708 | 32254 | 
| 29 | 11 requests/packages/urllib3/_collections.py | 52 | 65| 152 | 7860 | 32254 | 
| 30 | 12 requests/exceptions.py | 1 | 64| 286 | 8146 | 32540 | 
| 31 | 13 requests/packages/urllib3/connection.py | 1 | 46| 205 | 8351 | 33222 | 
| 32 | 14 setup.py | 1 | 64| 394 | 8745 | 33616 | 
| 33 | 14 requests/packages/urllib3/util.py | 420 | 485| 236 | 8981 | 33616 | 
| 34 | 15 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 9136 | 35604 | 
| 35 | 16 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 69| 541 | 9677 | 38182 | 
| 36 | 17 requests/packages/urllib3/response.py | 1 | 50| 237 | 9914 | 40379 | 
| 37 | 18 requests/packages/urllib3/__init__.py | 1 | 38| 180 | 10094 | 40767 | 
| 38 | 19 requests/auth.py | 1 | 55| 310 | 10404 | 42251 | 
| 39 | 19 requests/auth.py | 151 | 195| 368 | 10772 | 42251 | 
| 40 | 20 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 14| 0 | 10772 | 42348 | 
| 41 | 20 requests/packages/urllib3/response.py | 275 | 313| 257 | 11029 | 42348 | 
| 42 | 20 requests/packages/urllib3/contrib/pyopenssl.py | 103 | 169| 585 | 11614 | 42348 | 
| 43 | 21 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 12328 | 43368 | 
| 44 | 22 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 12328 | 43383 | 
| 45 | 22 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 12556 | 43383 | 
| 46 | 22 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 13295 | 43383 | 
| 47 | 22 requests/packages/urllib3/connectionpool.py | 467 | 557| 746 | 14041 | 43383 | 
| 48 | 22 requests/packages/urllib3/contrib/pyopenssl.py | 171 | 266| 740 | 14781 | 43383 | 
| 49 | 22 requests/packages/urllib3/contrib/pyopenssl.py | 72 | 100| 207 | 14988 | 43383 | 
| 50 | 22 requests/packages/urllib3/connection.py | 68 | 108| 297 | 15285 | 43383 | 
| 51 | 22 requests/packages/urllib3/util.py | 520 | 556| 256 | 15541 | 43383 | 
| 52 | 22 requests/utils.py | 65 | 100| 263 | 15804 | 43383 | 
| 53 | 23 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 1 | 64| 539 | 16343 | 44306 | 
| 54 | 23 requests/packages/urllib3/util.py | 595 | 644| 393 | 16736 | 44306 | 
| 55 | 24 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 17029 | 46463 | 
| 56 | 25 requests/packages/__init__.py | 1 | 4| 0 | 17029 | 46477 | 
| 57 | 25 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 17163 | 46477 | 
| 58 | 25 requests/packages/urllib3/packages/ordered_dict.py | 1 | 43| 369 | 17532 | 46477 | 
| 59 | 25 requests/packages/urllib3/poolmanager.py | 243 | 259| 162 | 17694 | 46477 | 
| 60 | 26 requests/adapters.py | 1 | 45| 299 | 17993 | 49228 | 
| 61 | 26 requests/packages/urllib3/_collections.py | 1 | 50| 266 | 18259 | 49228 | 
| 62 | 27 requests/status_codes.py | 1 | 89| 899 | 19158 | 50127 | 
| 63 | 27 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 19670 | 50127 | 
| 64 | 27 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 20006 | 50127 | 
| 65 | 28 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 20050 | 70995 | 
| 66 | 28 requests/packages/urllib3/util.py | 164 | 178| 137 | 20187 | 70995 | 
| 67 | 28 requests/packages/urllib3/response.py | 242 | 273| 241 | 20428 | 70995 | 
| 68 | 28 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 20628 | 70995 | 
| 69 | 28 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 21158 | 70995 | 
| 70 | 28 requests/packages/urllib3/contrib/pyopenssl.py | 317 | 347| 252 | 21410 | 70995 | 
| 71 | 28 requests/packages/urllib3/util.py | 559 | 593| 242 | 21652 | 70995 | 
| 72 | 28 requests/utils.py | 548 | 584| 215 | 21867 | 70995 | 
| 73 | **28 requests/sessions.py** | 68 | 154| 645 | 22512 | 70995 | 
| 74 | 29 requests/packages/urllib3/filepost.py | 1 | 44| 183 | 22695 | 71576 | 
| 75 | 30 requests/packages/charade/compat.py | 21 | 35| 69 | 22764 | 71842 | 
| 76 | 30 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 23339 | 71842 | 
| 77 | 30 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 23588 | 71842 | 
| 78 | 30 requests/packages/urllib3/connection.py | 48 | 66| 138 | 23726 | 71842 | 
| **-> 79 <-** | **30 requests/sessions.py** | 272 | 369| 766 | 24492 | 71842 | 
| 80 | 30 requests/packages/urllib3/contrib/pyopenssl.py | 291 | 314| 141 | 24633 | 71842 | 
| 81 | 31 docs/conf.py | 1 | 136| 1059 | 25692 | 73742 | 
| 82 | 31 requests/packages/urllib3/contrib/pyopenssl.py | 269 | 289| 121 | 25813 | 73742 | 
| 83 | 31 requests/auth.py | 58 | 149| 810 | 26623 | 73742 | 
| 84 | 31 requests/packages/urllib3/util.py | 130 | 162| 269 | 26892 | 73742 | 
| 85 | 31 requests/packages/urllib3/util.py | 488 | 517| 218 | 27110 | 73742 | 
| 86 | 31 requests/adapters.py | 151 | 185| 273 | 27383 | 73742 | 
| 87 | 31 requests/models.py | 149 | 170| 161 | 27544 | 73742 | 
| 88 | 31 requests/packages/urllib3/response.py | 53 | 139| 636 | 28180 | 73742 | 
| 89 | 31 requests/packages/urllib3/packages/ordered_dict.py | 174 | 261| 665 | 28845 | 73742 | 
| 90 | 31 requests/packages/urllib3/connectionpool.py | 560 | 597| 369 | 29214 | 73742 | 
| 91 | 31 requests/utils.py | 440 | 468| 259 | 29473 | 73742 | 
| 92 | 31 requests/packages/urllib3/response.py | 141 | 217| 626 | 30099 | 73742 | 
| 93 | 31 requests/utils.py | 519 | 545| 336 | 30435 | 73742 | 
| 94 | 31 requests/models.py | 439 | 471| 273 | 30708 | 73742 | 
| 95 | 31 docs/conf.py | 137 | 244| 615 | 31323 | 73742 | 
| **-> 96 <-** | **31 requests/sessions.py** | 232 | 270| 296 | 31619 | 73742 | 
| 97 | 32 requests/hooks.py | 1 | 46| 188 | 31807 | 73931 | 
| 98 | 32 requests/utils.py | 408 | 437| 284 | 32091 | 73931 | 
| 99 | 32 requests/packages/urllib3/util.py | 51 | 127| 748 | 32839 | 73931 | 
| 100 | 32 requests/adapters.py | 48 | 95| 389 | 33228 | 73931 | 
| 101 | 32 requests/packages/urllib3/connectionpool.py | 260 | 345| 807 | 34035 | 73931 | 
| 102 | **32 requests/sessions.py** | 37 | 65| 193 | 34228 | 73931 | 
| 103 | 32 requests/utils.py | 471 | 516| 271 | 34499 | 73931 | 
| 104 | 33 requests/certs.py | 1 | 25| 120 | 34619 | 74051 | 
| 105 | 34 requests/structures.py | 1 | 34| 167 | 34786 | 74891 | 
| 106 | 35 requests/packages/charade/big5freq.py | 43 | 926| 52 | 34838 | 121487 | 
| 107 | 35 requests/packages/urllib3/packages/ordered_dict.py | 143 | 172| 311 | 35149 | 121487 | 
| 108 | 35 requests/packages/urllib3/connectionpool.py | 627 | 654| 212 | 35361 | 121487 | 
| 109 | 35 requests/models.py | 312 | 376| 478 | 35839 | 121487 | 
| 110 | **35 requests/sessions.py** | 371 | 399| 252 | 36091 | 121487 | 
| 111 | 36 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 37367 | 122763 | 
| 112 | 36 requests/packages/urllib3/util.py | 195 | 217| 196 | 37563 | 122763 | 
| 113 | 36 requests/packages/urllib3/connectionpool.py | 599 | 625| 216 | 37779 | 122763 | 
| 114 | 36 requests/packages/urllib3/response.py | 219 | 239| 178 | 37957 | 122763 | 
| 115 | 36 requests/packages/urllib3/filepost.py | 47 | 63| 116 | 38073 | 122763 | 
| 116 | 36 requests/models.py | 378 | 437| 398 | 38471 | 122763 | 
| 117 | 36 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 67 | 106| 350 | 38821 | 122763 | 
| 118 | 36 requests/packages/urllib3/util.py | 219 | 235| 124 | 38945 | 122763 | 
| 119 | 36 requests/packages/urllib3/packages/ordered_dict.py | 92 | 113| 178 | 39123 | 122763 | 
| 120 | 37 requests/packages/urllib3/fields.py | 76 | 107| 274 | 39397 | 124075 | 
| 121 | 37 requests/packages/urllib3/fields.py | 161 | 178| 172 | 39569 | 124075 | 
| 122 | 37 requests/packages/urllib3/util.py | 180 | 193| 130 | 39699 | 124075 | 
| 123 | 37 requests/packages/urllib3/connectionpool.py | 60 | 83| 183 | 39882 | 124075 | 
| 124 | 37 requests/models.py | 492 | 594| 660 | 40542 | 124075 | 
| 125 | 37 requests/packages/urllib3/connectionpool.py | 347 | 378| 203 | 40745 | 124075 | 
| 126 | 38 requests/packages/charade/jisfreq.py | 44 | 570| 53 | 40798 | 152048 | 
| 127 | 38 requests/models.py | 173 | 244| 513 | 41311 | 152048 | 
| 128 | 38 requests/models.py | 278 | 310| 265 | 41576 | 152048 | 
| 129 | 38 requests/packages/urllib3/fields.py | 1 | 24| 114 | 41690 | 152048 | 
| 130 | **38 requests/sessions.py** | 502 | 538| 257 | 41947 | 152048 | 
| 131 | 38 requests/models.py | 89 | 146| 447 | 42394 | 152048 | 
| 132 | **38 requests/sessions.py** | 440 | 500| 488 | 42882 | 152048 | 
| 133 | 39 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 43066 | 152494 | 
| 134 | 39 requests/packages/urllib3/connectionpool.py | 85 | 168| 702 | 43768 | 152494 | 
| 135 | 40 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 43821 | 179268 | 
| 136 | 40 requests/packages/urllib3/poolmanager.py | 97 | 133| 298 | 44119 | 179268 | 
| 137 | 40 requests/packages/urllib3/fields.py | 142 | 159| 142 | 44261 | 179268 | 
| 138 | 41 requests/api.py | 47 | 99| 438 | 44699 | 180341 | 
| 139 | 41 requests/models.py | 732 | 771| 248 | 44947 | 180341 | 
| 140 | 42 requests/packages/urllib3/request.py | 1 | 57| 377 | 45324 | 181547 | 
| 141 | 42 requests/packages/urllib3/poolmanager.py | 71 | 95| 190 | 45514 | 181547 | 
| 142 | 42 requests/adapters.py | 114 | 149| 269 | 45783 | 181547 | 
| 143 | 42 requests/packages/urllib3/util.py | 335 | 417| 561 | 46344 | 181547 | 
| 144 | 42 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 47013 | 181547 | 
| 145 | 42 requests/adapters.py | 248 | 260| 141 | 47154 | 181547 | 
| 146 | 42 requests/utils.py | 189 | 220| 266 | 47420 | 181547 | 
| 147 | 42 requests/structures.py | 37 | 109| 558 | 47978 | 181547 | 
| 148 | 42 requests/packages/urllib3/util.py | 269 | 299| 233 | 48211 | 181547 | 
| 149 | 42 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 48356 | 181547 | 
| 150 | 42 requests/packages/urllib3/connectionpool.py | 170 | 217| 365 | 48721 | 181547 | 
| 151 | 42 requests/adapters.py | 97 | 112| 175 | 48896 | 181547 | 
| 152 | 42 requests/adapters.py | 262 | 286| 209 | 49105 | 181547 | 
| 153 | 42 requests/utils.py | 103 | 130| 211 | 49316 | 181547 | 
| 154 | 42 requests/packages/urllib3/util.py | 237 | 266| 283 | 49599 | 181547 | 
| 155 | 42 requests/utils.py | 289 | 332| 237 | 49836 | 181547 | 
| 156 | 42 requests/packages/urllib3/request.py | 59 | 88| 283 | 50119 | 181547 | 
| 157 | **42 requests/sessions.py** | 401 | 409| 112 | 50231 | 181547 | 
| 158 | 42 requests/models.py | 40 | 87| 294 | 50525 | 181547 | 
| 159 | 42 requests/models.py | 633 | 657| 167 | 50692 | 181547 | 
| 160 | 42 requests/utils.py | 335 | 371| 191 | 50883 | 181547 | 
| 161 | 42 requests/packages/urllib3/connectionpool.py | 380 | 466| 770 | 51653 | 181547 | 
| 162 | **42 requests/sessions.py** | 411 | 419| 112 | 51765 | 181547 | 
| 163 | **42 requests/sessions.py** | 421 | 438| 185 | 51950 | 181547 | 
| 164 | 42 requests/packages/urllib3/connectionpool.py | 219 | 258| 298 | 52248 | 181547 | 
| 165 | 42 requests/adapters.py | 288 | 378| 587 | 52835 | 181547 | 
| 166 | 42 requests/utils.py | 158 | 186| 250 | 53085 | 181547 | 
| 167 | 42 requests/api.py | 1 | 44| 423 | 53508 | 181547 | 
| 168 | 42 requests/adapters.py | 216 | 246| 231 | 53739 | 181547 | 
| 169 | 42 requests/structures.py | 112 | 129| 114 | 53853 | 181547 | 
| 170 | 42 requests/packages/urllib3/request.py | 90 | 143| 508 | 54361 | 181547 | 
| 171 | 42 requests/packages/urllib3/fields.py | 55 | 74| 129 | 54490 | 181547 | 
| 172 | 42 requests/models.py | 659 | 681| 152 | 54642 | 181547 | 
| 173 | 42 requests/adapters.py | 187 | 214| 218 | 54860 | 181547 | 
| 174 | 42 requests/utils.py | 133 | 155| 166 | 55026 | 181547 | 
| 175 | 42 requests/packages/urllib3/filepost.py | 66 | 102| 233 | 55259 | 181547 | 
| 176 | 42 requests/models.py | 596 | 631| 244 | 55503 | 181547 | 
| 177 | 42 requests/packages/urllib3/connectionpool.py | 657 | 682| 206 | 55709 | 181547 | 
| 178 | 42 requests/api.py | 102 | 121| 171 | 55880 | 181547 | 
| 179 | 42 requests/packages/urllib3/fields.py | 109 | 140| 232 | 56112 | 181547 | 
| 180 | 42 requests/models.py | 247 | 276| 208 | 56320 | 181547 | 


### Hint

```
Mm, good spot. I think we should try to do something smarter here. Thanks for raising this issue! :cake:

```

## Patch

```diff
diff --git a/requests/cookies.py b/requests/cookies.py
--- a/requests/cookies.py
+++ b/requests/cookies.py
@@ -421,3 +421,25 @@ def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
                 cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
 
     return cookiejar
+
+
+def merge_cookies(cookiejar, cookies):
+    """Add cookies to cookiejar and returns a merged CookieJar.
+
+    :param cookiejar: CookieJar object to add the cookies to.
+    :param cookies: Dictionary or CookieJar object to be added.
+    """
+    if not isinstance(cookiejar, cookielib.CookieJar):
+        raise ValueError('You can only merge into CookieJar')
+    
+    if isinstance(cookies, dict):
+        cookiejar = cookiejar_from_dict(
+            cookies, cookiejar=cookiejar, overwrite=False)
+    elif isinstance(cookies, cookielib.CookieJar):
+        try:
+            cookiejar.update(cookies)
+        except AttributeError:
+            for cookie_in_jar in cookies:
+                cookiejar.set_cookie(cookie_in_jar)
+
+    return cookiejar
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -13,7 +13,8 @@
 from datetime import datetime
 
 from .compat import cookielib, OrderedDict, urljoin, urlparse, builtin_str
-from .cookies import cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar
+from .cookies import (
+    cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar, merge_cookies)
 from .models import Request, PreparedRequest
 from .hooks import default_hooks, dispatch_hook
 from .utils import to_key_val_list, default_headers
@@ -245,9 +246,8 @@ def prepare_request(self, request):
             cookies = cookiejar_from_dict(cookies)
 
         # Merge with session cookies
-        merged_cookies = RequestsCookieJar()
-        merged_cookies.update(self.cookies)
-        merged_cookies.update(cookies)
+        merged_cookies = merge_cookies(
+            merge_cookies(RequestsCookieJar(), self.cookies), cookies)
 
 
         # Set environment's basic authentication if not explicitly set.
@@ -330,7 +330,7 @@ def request(self, method, url,
         prep = self.prepare_request(req)
 
         # Add param cookies to session cookies
-        self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)
+        self.cookies = merge_cookies(self.cookies, cookies)
 
         proxies = proxies or {}
 

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -187,6 +187,14 @@ def test_generic_cookiejar_works(self):
         assert r.json()['cookies']['foo'] == 'bar'
         # Make sure the session cj is still the custom one
         assert s.cookies is cj
+    
+    def test_param_cookiejar_works(self):
+        cj = cookielib.CookieJar()
+        cookiejar_from_dict({'foo' : 'bar'}, cj)
+        s = requests.session()
+        r = s.get(httpbin('cookies'), cookies=cj)
+        # Make sure the cookie was sent
+        assert r.json()['cookies']['foo'] == 'bar'
 
     def test_requests_in_history_are_not_overridden(self):
         resp = requests.get(httpbin('redirect/3'))

```


## Code snippets

### 1 - requests/cookies.py:

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
### 2 - requests/cookies.py:

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
### 3 - requests/cookies.py:

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
### 4 - requests/cookies.py:

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
### 5 - requests/cookies.py:

Start line: 324, End line: 341

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):

    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        # remove the unpickleable RLock object
        state.pop('_cookies_lock')
        return state

    def __setstate__(self, state):
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if '_cookies_lock' not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self):
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.update(self)
        return new_cj
```
### 6 - requests/cookies.py:

Start line: 201, End line: 291

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):

    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the jar.
        See values() and items()."""
        keys = []
        for cookie in iter(self):
            keys.append(cookie.name)
        return keys

    def values(self):
        """Dict-like values() that returns a list of values of cookies from the jar.
        See keys() and items()."""
        values = []
        for cookie in iter(self):
            values.append(cookie.value)
        return values

    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the jar.
        See keys() and values(). Allows client-code to call "dict(RequestsCookieJar)
        and get a vanilla python dict of key value pairs."""
        items = []
        for cookie in iter(self):
            items.append((cookie.name, cookie.value))
        return items

    def list_domains(self):
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self):
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise."""
        domains = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(self, domain=None, path=None):
        """Takes as an argument an optional domain and path and returns a plain old
        Python dict of name-value pairs of cookies that meet the requirements."""
        dictionary = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (path is None
                                                or cookie.path == path):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws exception
        if there are more than one cookie with name. In that case, use the more
        explicit get() method instead. Caution: operation is O(n), not O(1)."""

        return self._find_no_duplicates(name)

    def __setitem__(self, name, value):
        """Dict-like __setitem__ for compatibility with client code. Throws exception
        if there is already a cookie of that name in the jar. In that case, use the more
        explicit set() method instead."""

        self.set(name, value)

    def __delitem__(self, name):
        """Deletes a cookie given a name. Wraps cookielib.CookieJar's remove_cookie_by_name()."""
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie, *args, **kwargs):
        if hasattr(cookie.value, 'startswith') and cookie.value.startswith('"') and cookie.value.endswith('"'):
            cookie.value = cookie.value.replace('\\"', '')
        return super(RequestsCookieJar, self).set_cookie(cookie, *args, **kwargs)

    def update(self, other):
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(cookie)
        else:
            super(RequestsCookieJar, self).update(other)
```
### 7 - requests/cookies.py:

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
### 8 - requests/cookies.py:

Start line: 159, End line: 183

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):
    """Compatibility class; is a cookielib.CookieJar, but exposes a dict interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Don't use the dict interface internally; it's just for compatibility with
    with external client code. All `requests` code should work out of the box
    with externally provided instances of CookieJar, e.g., LWPCookieJar and
    FileCookieJar.

    Caution: dictionary operations that are normally O(1) may be O(n).

    Unlike a regular CookieJar, this class is pickleable.
    """

    def get(self, name, default=None, domain=None, path=None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains. Caution: operation is O(n), not O(1)."""
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default
```
### 9 - requests/cookies.py:

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
### 10 - requests/cookies.py:

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
### 11 - requests/cookies.py:

Start line: 406, End line: 424

```python
def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or (name not in names_from_jar):
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))

    return cookiejar
```
### 12 - requests/cookies.py:

Start line: 185, End line: 199

```python
class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):

    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains."""
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(self, name, domain=kwargs.get('domain'), path=kwargs.get('path'))
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c
```
### 16 - requests/cookies.py:

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
### 17 - requests/cookies.py:

Start line: 379, End line: 403

```python
def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""
    expires = None
    if morsel["max-age"]:
        expires = time.time() + morsel["max-age"]
    elif morsel['expires']:
        expires = morsel['expires']
        if type(expires) == type(""):
            time_template = "%a, %d-%b-%Y %H:%M:%S GMT"
            expires = time.mktime(time.strptime(expires, time_template))
    c = create_cookie(
        name=morsel.key,
        value=morsel.value,
        version=morsel['version'] or 0,
        port=None,
        domain=morsel['domain'],
        path=morsel['path'],
        secure=bool(morsel['secure']),
        expires=expires,
        discard=False,
        comment=morsel['comment'],
        comment_url=bool(morsel['comment']),
        rest={'HttpOnly': morsel['httponly']},
        rfc2109=False,)
    return c
```
### 18 - requests/cookies.py:

Start line: 93, End line: 111

```python
class MockResponse(object):
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `cookielib` expects to see them.
    """

    def __init__(self, headers):
        """Make a MockResponse for `cookielib` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self):
        return self._headers

    def getheaders(self, name):
        self._headers.getheaders(name)
```
### 21 - requests/sessions.py:

Start line: 1, End line: 34

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
from .cookies import cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar
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
### 22 - requests/sessions.py:

Start line: 157, End line: 230

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
### 73 - requests/sessions.py:

Start line: 68, End line: 154

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
### 79 - requests/sessions.py:

Start line: 272, End line: 369

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
        self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)

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
### 96 - requests/sessions.py:

Start line: 232, End line: 270

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
        merged_cookies = RequestsCookieJar()
        merged_cookies.update(self.cookies)
        merged_cookies.update(cookies)


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
            hooks=merge_setting(request.hooks, self.hooks),
        )
        return p
```
### 102 - requests/sessions.py:

Start line: 37, End line: 65

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
```
### 110 - requests/sessions.py:

Start line: 371, End line: 399

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
### 130 - requests/sessions.py:

Start line: 502, End line: 538

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
### 132 - requests/sessions.py:

Start line: 440, End line: 500

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
### 157 - requests/sessions.py:

Start line: 401, End line: 409

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
### 162 - requests/sessions.py:

Start line: 411, End line: 419

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
### 163 - requests/sessions.py:

Start line: 421, End line: 438

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
