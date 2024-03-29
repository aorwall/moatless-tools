# psf__requests-1635

| **psf/requests** | `9968a10fcfad7268b552808c4f8946eecafc956a` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 4750 |
| **Any found context length** | 4750 |
| **Avg pos** | 40.333333333333336 |
| **Min pos** | 16 |
| **Max pos** | 71 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/cookies.py b/requests/cookies.py
--- a/requests/cookies.py
+++ b/requests/cookies.py
@@ -392,15 +392,21 @@ def morsel_to_cookie(morsel):
     return c
 
 
-def cookiejar_from_dict(cookie_dict, cookiejar=None):
+def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
     """Returns a CookieJar from a key/value dictionary.
 
     :param cookie_dict: Dict of key/values to insert into CookieJar.
+    :param cookiejar: (optional) A cookiejar to add the cookies to.
+    :param overwrite: (optional) If False, will not replace cookies
+        already in the jar with new ones.
     """
     if cookiejar is None:
         cookiejar = RequestsCookieJar()
 
     if cookie_dict is not None:
+        names_from_jar = [cookie.name for cookie in cookiejar]
         for name in cookie_dict:
-            cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
+            if overwrite or (name not in names_from_jar):
+                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
+
     return cookiejar
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -295,7 +295,7 @@ def copy(self):
         p = PreparedRequest()
         p.method = self.method
         p.url = self.url
-        p.headers = self.headers
+        p.headers = self.headers.copy()
         p.body = self.body
         p.hooks = self.hooks
         return p
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -322,6 +322,9 @@ def request(self, method, url,
         )
         prep = self.prepare_request(req)
 
+        # Add param cookies to session cookies
+        self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)
+
         proxies = proxies or {}
 
         # Gather clues from the surrounding environment.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/cookies.py | 395 | 405 | 16 | 2 | 4750
| requests/models.py | 298 | 298 | 71 | 3 | 23904
| requests/sessions.py | 325 | 325 | 34 | 1 | 10589


## Problem Statement

```
Cookies not persisted when set via functional API.
Cookies set as part of a call to `Session.request()` (or any of the top level methods that call it) are _not_ persisted, including on redirects.

Expected behaviour:

\`\`\` python
>>> s = requests.Session()
>>> r = s.get('http://httpbin.org/redirect/1', cookies={'Hi': 'There'})
>>> print r.request.headers['Cookie']
'hi=there'
\`\`\`

Actual behaviour:

\`\`\` python
>>> s = requests.Session()
>>> r = s.get('http://httpbin.org/redirect/1', cookies={'Hi': 'There'})
>>> print r.request.headers['Cookie']
KeyError: 'cookie'
\`\`\`

And, a super extra bonus bug:

\`\`\` python
>>> r.history[0].request.headers['Cookie']
KeyError: 'cookie'
\`\`\`

even though we definitely sent the cookie on the first request.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 requests/sessions.py** | 153 | 226| 523 | 523 | 3938 | 
| 2 | **1 requests/sessions.py** | 1 | 34| 218 | 741 | 3938 | 
| 3 | **2 requests/cookies.py** | 1 | 79| 509 | 1250 | 7132 | 
| 4 | **2 requests/cookies.py** | 313 | 330| 160 | 1410 | 7132 | 
| 5 | **2 requests/cookies.py** | 148 | 172| 249 | 1659 | 7132 | 
| 6 | **2 requests/cookies.py** | 190 | 280| 758 | 2417 | 7132 | 
| 7 | **2 requests/cookies.py** | 295 | 311| 227 | 2644 | 7132 | 
| 8 | **2 requests/cookies.py** | 282 | 293| 155 | 2799 | 7132 | 
| 9 | **2 requests/cookies.py** | 333 | 365| 233 | 3032 | 7132 | 
| 10 | **2 requests/cookies.py** | 174 | 188| 151 | 3183 | 7132 | 
| 11 | **2 requests/cookies.py** | 120 | 145| 218 | 3401 | 7132 | 
| 12 | **2 requests/cookies.py** | 103 | 117| 150 | 3551 | 7132 | 
| 13 | **2 requests/sessions.py** | 68 | 150| 613 | 4164 | 7132 | 
| 14 | **3 requests/models.py** | 464 | 480| 132 | 4296 | 12073 | 
| 15 | **3 requests/cookies.py** | 82 | 100| 134 | 4430 | 12073 | 
| **-> 16 <-** | **3 requests/cookies.py** | 368 | 407| 320 | 4750 | 12073 | 
| 17 | 4 requests/compat.py | 1 | 116| 805 | 5555 | 12878 | 
| 18 | 5 requests/utils.py | 236 | 259| 137 | 5692 | 16930 | 
| 19 | 6 requests/__init__.py | 1 | 78| 293 | 5985 | 17426 | 
| 20 | **6 requests/sessions.py** | 361 | 389| 252 | 6237 | 17426 | 
| 21 | 7 requests/auth.py | 1 | 55| 310 | 6547 | 18797 | 
| 22 | 7 requests/auth.py | 145 | 181| 311 | 6858 | 18797 | 
| 23 | 7 requests/utils.py | 1 | 52| 282 | 7140 | 18797 | 
| 24 | **7 requests/models.py** | 1 | 36| 250 | 7390 | 18797 | 
| 25 | **7 requests/sessions.py** | 391 | 399| 112 | 7502 | 18797 | 
| 26 | **7 requests/sessions.py** | 430 | 490| 488 | 7990 | 18797 | 
| 27 | **7 requests/sessions.py** | 401 | 409| 112 | 8102 | 18797 | 
| 28 | **7 requests/sessions.py** | 492 | 528| 257 | 8359 | 18797 | 
| 29 | **7 requests/sessions.py** | 228 | 266| 296 | 8655 | 18797 | 
| 30 | 8 requests/packages/urllib3/response.py | 264 | 302| 257 | 8912 | 20910 | 
| 31 | 9 requests/packages/urllib3/_collections.py | 67 | 95| 174 | 9086 | 21535 | 
| 32 | 10 requests/adapters.py | 257 | 281| 209 | 9295 | 24222 | 
| 33 | 10 requests/packages/urllib3/response.py | 53 | 130| 563 | 9858 | 24222 | 
| **-> 34 <-** | **10 requests/sessions.py** | 268 | 359| 731 | 10589 | 24222 | 
| 35 | 10 requests/auth.py | 58 | 143| 755 | 11344 | 24222 | 
| 36 | 11 requests/exceptions.py | 1 | 60| 272 | 11616 | 24494 | 
| 37 | 11 requests/packages/urllib3/_collections.py | 52 | 65| 152 | 11768 | 24494 | 
| 38 | 11 requests/adapters.py | 1 | 44| 281 | 12049 | 24494 | 
| 39 | 12 requests/packages/urllib3/util.py | 411 | 468| 195 | 12244 | 29131 | 
| 40 | 12 requests/utils.py | 459 | 504| 271 | 12515 | 29131 | 
| 41 | 12 requests/adapters.py | 243 | 255| 141 | 12656 | 29131 | 
| 42 | **12 requests/sessions.py** | 411 | 428| 185 | 12841 | 29131 | 
| 43 | 13 requests/packages/urllib3/exceptions.py | 1 | 122| 701 | 13542 | 29885 | 
| 44 | 14 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 13700 | 32727 | 
| 45 | **14 requests/models.py** | 483 | 554| 494 | 14194 | 32727 | 
| 46 | 15 requests/api.py | 47 | 99| 438 | 14632 | 33800 | 
| 47 | 15 requests/adapters.py | 47 | 94| 389 | 15021 | 33800 | 
| 48 | 15 requests/utils.py | 396 | 425| 284 | 15305 | 33800 | 
| 49 | 16 requests/structures.py | 37 | 109| 558 | 15863 | 34640 | 
| 50 | 17 requests/hooks.py | 1 | 46| 188 | 16051 | 34829 | 
| 51 | **17 requests/sessions.py** | 37 | 65| 193 | 16244 | 34829 | 
| 52 | 17 requests/packages/urllib3/response.py | 132 | 206| 616 | 16860 | 34829 | 
| 53 | 17 requests/utils.py | 55 | 88| 250 | 17110 | 34829 | 
| 54 | 18 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 17244 | 36986 | 
| 55 | 19 requests/status_codes.py | 1 | 89| 899 | 18143 | 37885 | 
| 56 | 20 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 18392 | 39873 | 
| 57 | 20 requests/packages/urllib3/response.py | 231 | 262| 241 | 18633 | 39873 | 
| 58 | 20 requests/packages/urllib3/util.py | 1 | 48| 269 | 18902 | 39873 | 
| 59 | 20 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 19641 | 39873 | 
| 60 | 20 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 19977 | 39873 | 
| 61 | 20 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 20552 | 39873 | 
| 62 | 20 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 20845 | 39873 | 
| 63 | 21 requests/packages/urllib3/connectionpool.py | 1 | 80| 373 | 21218 | 45559 | 
| 64 | 21 requests/packages/urllib3/poolmanager.py | 243 | 260| 162 | 21380 | 45559 | 
| 65 | 21 requests/packages/urllib3/connectionpool.py | 534 | 620| 771 | 22151 | 45559 | 
| 66 | 21 requests/utils.py | 177 | 208| 266 | 22417 | 45559 | 
| 67 | **21 requests/models.py** | 692 | 731| 248 | 22665 | 45559 | 
| 68 | 21 requests/packages/urllib3/packages/ordered_dict.py | 174 | 261| 665 | 23330 | 45559 | 
| 69 | 22 requests/packages/urllib3/fields.py | 142 | 159| 142 | 23472 | 46871 | 
| 70 | 22 requests/structures.py | 1 | 34| 167 | 23639 | 46871 | 
| **-> 71 <-** | **22 requests/models.py** | 275 | 307| 265 | 23904 | 46871 | 
| 72 | 22 requests/packages/urllib3/packages/ordered_dict.py | 1 | 43| 369 | 24273 | 46871 | 
| 73 | **22 requests/models.py** | 430 | 462| 271 | 24544 | 46871 | 
| 74 | **22 requests/models.py** | 369 | 428| 397 | 24941 | 46871 | 
| 75 | 23 requests/packages/urllib3/contrib/pyopenssl.py | 102 | 168| 585 | 25526 | 49444 | 
| 76 | 24 requests/packages/urllib3/filepost.py | 47 | 63| 116 | 25642 | 50025 | 
| 77 | **24 requests/models.py** | 244 | 273| 208 | 25850 | 50025 | 
| 78 | 24 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 26050 | 50025 | 
| 79 | 24 requests/packages/urllib3/contrib/pyopenssl.py | 268 | 288| 121 | 26171 | 50025 | 
| 80 | 24 requests/packages/urllib3/util.py | 160 | 174| 137 | 26308 | 50025 | 
| 81 | 24 requests/packages/urllib3/fields.py | 109 | 140| 232 | 26540 | 50025 | 
| 82 | **24 requests/models.py** | 146 | 167| 161 | 26701 | 50025 | 
| 83 | 24 requests/packages/urllib3/util.py | 51 | 123| 707 | 27408 | 50025 | 
| 84 | 24 requests/packages/urllib3/contrib/pyopenssl.py | 170 | 265| 740 | 28148 | 50025 | 
| 85 | 24 requests/adapters.py | 212 | 241| 220 | 28368 | 50025 | 
| 86 | 25 requests/packages/urllib3/__init__.py | 1 | 38| 180 | 28548 | 50413 | 
| 87 | 26 requests/packages/urllib3/request.py | 1 | 57| 377 | 28925 | 51619 | 
| 88 | 26 requests/packages/urllib3/_collections.py | 1 | 50| 266 | 29191 | 51619 | 
| 89 | 26 requests/utils.py | 536 | 572| 215 | 29406 | 51619 | 
| 90 | 26 requests/utils.py | 277 | 320| 237 | 29643 | 51619 | 
| 91 | 27 requests/packages/__init__.py | 1 | 4| 0 | 29643 | 51633 | 
| 92 | 27 requests/packages/urllib3/filepost.py | 1 | 44| 183 | 29826 | 51633 | 
| 93 | 28 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 31102 | 52909 | 
| 94 | 28 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 31614 | 52909 | 
| 95 | 28 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 68| 544 | 32158 | 52909 | 
| 96 | 28 requests/packages/urllib3/fields.py | 55 | 74| 129 | 32287 | 52909 | 
| 97 | 28 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 32442 | 52909 | 
| 98 | 28 requests/packages/urllib3/response.py | 1 | 50| 237 | 32679 | 52909 | 
| 99 | 29 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 33393 | 53929 | 
| 100 | **29 requests/models.py** | 619 | 641| 152 | 33545 | 53929 | 
| 101 | 30 docs/conf.py | 1 | 136| 1059 | 34604 | 55829 | 
| 102 | 30 requests/packages/urllib3/util.py | 265 | 295| 233 | 34837 | 55829 | 
| 103 | 30 requests/packages/urllib3/util.py | 471 | 500| 218 | 35055 | 55829 | 
| 104 | **30 requests/models.py** | 593 | 617| 167 | 35222 | 55829 | 
| 105 | 30 requests/packages/urllib3/connectionpool.py | 623 | 659| 362 | 35584 | 55829 | 
| 106 | **30 requests/models.py** | 170 | 241| 513 | 36097 | 55829 | 
| 107 | **30 requests/models.py** | 309 | 367| 429 | 36526 | 55829 | 
| 108 | 30 requests/packages/urllib3/util.py | 578 | 627| 393 | 36919 | 55829 | 
| 109 | 30 requests/packages/urllib3/packages/ordered_dict.py | 92 | 113| 178 | 37097 | 55829 | 
| 110 | 31 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 37097 | 55844 | 
| 111 | 31 docs/conf.py | 137 | 244| 615 | 37712 | 55844 | 
| 112 | 31 requests/api.py | 102 | 121| 171 | 37883 | 55844 | 
| 113 | 32 requests/packages/charade/compat.py | 21 | 35| 69 | 37952 | 56110 | 
| 114 | 32 requests/api.py | 1 | 44| 423 | 38375 | 56110 | 
| 115 | 32 requests/packages/urllib3/response.py | 208 | 228| 178 | 38553 | 56110 | 
| 116 | 32 requests/utils.py | 323 | 359| 191 | 38744 | 56110 | 
| 117 | 32 requests/packages/urllib3/connectionpool.py | 105 | 139| 285 | 39029 | 56110 | 
| 118 | 32 requests/packages/urllib3/connectionpool.py | 341 | 415| 710 | 39739 | 56110 | 
| 119 | 32 requests/adapters.py | 283 | 370| 573 | 40312 | 56110 | 
| 120 | 32 requests/packages/urllib3/util.py | 176 | 189| 130 | 40442 | 56110 | 
| 121 | 32 requests/packages/urllib3/contrib/pyopenssl.py | 290 | 313| 141 | 40583 | 56110 | 
| 122 | 32 requests/packages/urllib3/util.py | 126 | 158| 269 | 40852 | 56110 | 
| 123 | 32 requests/packages/urllib3/fields.py | 27 | 52| 217 | 41069 | 56110 | 
| 124 | 32 requests/packages/urllib3/fields.py | 161 | 178| 172 | 41241 | 56110 | 
| 125 | 32 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 41910 | 56110 | 
| 126 | 32 requests/packages/urllib3/fields.py | 76 | 107| 274 | 42184 | 56110 | 
| 127 | 32 requests/packages/urllib3/connectionpool.py | 83 | 103| 147 | 42331 | 56110 | 
| 128 | **32 requests/models.py** | 556 | 591| 245 | 42576 | 56110 | 
| 129 | 32 requests/packages/urllib3/request.py | 59 | 88| 283 | 42859 | 56110 | 
| 130 | 32 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 43389 | 56110 | 
| 131 | 32 requests/structures.py | 112 | 129| 114 | 43503 | 56110 | 
| 132 | 32 requests/packages/urllib3/contrib/pyopenssl.py | 316 | 345| 244 | 43747 | 56110 | 
| 133 | 32 requests/packages/urllib3/packages/ordered_dict.py | 143 | 172| 311 | 44058 | 56110 | 
| 134 | 32 requests/utils.py | 146 | 174| 250 | 44308 | 56110 | 
| 135 | 32 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 44536 | 56110 | 
| 136 | 32 requests/packages/urllib3/connectionpool.py | 689 | 718| 233 | 44769 | 56110 | 
| 137 | **32 requests/models.py** | 39 | 86| 294 | 45063 | 56110 | 
| 138 | 33 requests/certs.py | 1 | 25| 120 | 45183 | 56230 | 
| 139 | 33 requests/packages/urllib3/connectionpool.py | 417 | 448| 203 | 45386 | 56230 | 
| 140 | 33 requests/adapters.py | 150 | 184| 273 | 45659 | 56230 | 
| 141 | 33 requests/packages/urllib3/util.py | 215 | 231| 124 | 45783 | 56230 | 
| 142 | 33 requests/packages/urllib3/connectionpool.py | 251 | 298| 364 | 46147 | 56230 | 
| 143 | 33 requests/packages/urllib3/request.py | 90 | 143| 508 | 46655 | 56230 | 
| 144 | 33 requests/utils.py | 428 | 456| 259 | 46914 | 56230 | 
| 145 | 34 requests/packages/charade/big5freq.py | 43 | 926| 52 | 46966 | 102826 | 
| 146 | 34 requests/utils.py | 211 | 233| 280 | 47246 | 102826 | 
| 147 | 34 requests/packages/urllib3/util.py | 503 | 539| 256 | 47502 | 102826 | 
| 148 | 35 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 57| 509 | 48011 | 103685 | 
| 149 | 35 requests/packages/urllib3/contrib/pyopenssl.py | 71 | 99| 207 | 48218 | 103685 | 
| 150 | 35 requests/packages/urllib3/util.py | 191 | 213| 196 | 48414 | 103685 | 
| 151 | 35 requests/utils.py | 507 | 533| 336 | 48750 | 103685 | 
| 152 | 35 requests/utils.py | 121 | 143| 166 | 48916 | 103685 | 
| 153 | 35 requests/packages/urllib3/connectionpool.py | 142 | 165| 183 | 49099 | 103685 | 
| 154 | 35 requests/packages/urllib3/util.py | 298 | 328| 226 | 49325 | 103685 | 
| 155 | 35 requests/packages/urllib3/util.py | 331 | 408| 504 | 49829 | 103685 | 
| 156 | 35 requests/packages/urllib3/util.py | 542 | 576| 242 | 50071 | 103685 | 
| 157 | 35 requests/packages/urllib3/connectionpool.py | 167 | 249| 695 | 50766 | 103685 | 
| 158 | 35 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 50911 | 103685 | 
| 159 | 35 requests/packages/urllib3/connectionpool.py | 661 | 687| 216 | 51127 | 103685 | 
| 160 | 35 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 60 | 99| 350 | 51477 | 103685 | 
| 161 | 36 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 51661 | 104131 | 
| 162 | 36 requests/packages/urllib3/util.py | 233 | 262| 283 | 51944 | 104131 | 
| 163 | 37 setup.py | 1 | 58| 364 | 52308 | 104495 | 
| 164 | **37 requests/models.py** | 676 | 690| 168 | 52476 | 104495 | 
| 165 | 37 requests/packages/urllib3/filepost.py | 66 | 102| 233 | 52709 | 104495 | 
| 166 | 37 requests/adapters.py | 186 | 210| 197 | 52906 | 104495 | 
| 167 | 38 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 52950 | 125363 | 
| 168 | 38 requests/packages/urllib3/connectionpool.py | 450 | 532| 727 | 53677 | 125363 | 
| 169 | 38 requests/adapters.py | 113 | 148| 269 | 53946 | 125363 | 
| 170 | 38 requests/packages/urllib3/connectionpool.py | 300 | 339| 298 | 54244 | 125363 | 
| 171 | 39 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 54297 | 152137 | 
| 172 | **39 requests/models.py** | 643 | 674| 198 | 54495 | 152137 | 
| 173 | **39 requests/models.py** | 88 | 143| 424 | 54919 | 152137 | 
| 174 | 39 requests/adapters.py | 96 | 111| 175 | 55094 | 152137 | 
| 175 | 39 requests/packages/urllib3/fields.py | 1 | 24| 114 | 55208 | 152137 | 
| 176 | 39 requests/packages/urllib3/poolmanager.py | 97 | 133| 298 | 55506 | 152137 | 
| 177 | 39 requests/utils.py | 91 | 118| 211 | 55717 | 152137 | 
| 178 | 40 requests/packages/charade/jisfreq.py | 44 | 570| 53 | 55770 | 180110 | 
| 179 | 40 requests/packages/urllib3/connectionpool.py | 721 | 746| 206 | 55976 | 180110 | 
| 180 | 40 requests/packages/urllib3/poolmanager.py | 71 | 95| 190 | 56166 | 180110 | 
| 181 | 40 requests/utils.py | 262 | 274| 137 | 56303 | 180110 | 
| 182 | 40 requests/utils.py | 362 | 393| 269 | 56572 | 180110 | 


## Patch

```diff
diff --git a/requests/cookies.py b/requests/cookies.py
--- a/requests/cookies.py
+++ b/requests/cookies.py
@@ -392,15 +392,21 @@ def morsel_to_cookie(morsel):
     return c
 
 
-def cookiejar_from_dict(cookie_dict, cookiejar=None):
+def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
     """Returns a CookieJar from a key/value dictionary.
 
     :param cookie_dict: Dict of key/values to insert into CookieJar.
+    :param cookiejar: (optional) A cookiejar to add the cookies to.
+    :param overwrite: (optional) If False, will not replace cookies
+        already in the jar with new ones.
     """
     if cookiejar is None:
         cookiejar = RequestsCookieJar()
 
     if cookie_dict is not None:
+        names_from_jar = [cookie.name for cookie in cookiejar]
         for name in cookie_dict:
-            cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
+            if overwrite or (name not in names_from_jar):
+                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
+
     return cookiejar
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -295,7 +295,7 @@ def copy(self):
         p = PreparedRequest()
         p.method = self.method
         p.url = self.url
-        p.headers = self.headers
+        p.headers = self.headers.copy()
         p.body = self.body
         p.hooks = self.hooks
         return p
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -322,6 +322,9 @@ def request(self, method, url,
         )
         prep = self.prepare_request(req)
 
+        # Add param cookies to session cookies
+        self.cookies = cookiejar_from_dict(cookies, cookiejar=self.cookies, overwrite=False)
+
         proxies = proxies or {}
 
         # Gather clues from the surrounding environment.

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -164,6 +164,12 @@ def test_cookie_quote_wrapped(self):
         s.get(httpbin('cookies/set?foo="bar:baz"'))
         self.assertTrue(s.cookies['foo'] == '"bar:baz"')
 
+    def test_cookie_persists_via_api(self):
+        s = requests.session()
+        r = s.get(httpbin('redirect/1'), cookies={'foo':'bar'})
+        self.assertTrue('foo' in r.request.headers['Cookie'])
+        self.assertTrue('foo' in r.history[0].request.headers['Cookie'])
+
     def test_request_cookie_overrides_session_cookie(self):
         s = requests.session()
         s.cookies['foo'] = 'bar'

```


## Code snippets

### 1 - requests/sessions.py:

Start line: 153, End line: 226

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
### 2 - requests/sessions.py:

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

from .compat import cookielib, OrderedDict, urljoin, urlparse
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
### 3 - requests/cookies.py:

Start line: 1, End line: 79

```python
# -*- coding: utf-8 -*-

"""
Compatibility code to be able to use `cookielib.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import time
import collections
from .compat import cookielib, urlparse, Morsel

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

    def get_full_url(self):
        return self._r.url

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
### 4 - requests/cookies.py:

Start line: 313, End line: 330

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
### 5 - requests/cookies.py:

Start line: 148, End line: 172

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
### 6 - requests/cookies.py:

Start line: 190, End line: 280

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
        if cookie.value.startswith('"') and cookie.value.endswith('"'):
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

Start line: 295, End line: 311

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
### 8 - requests/cookies.py:

Start line: 282, End line: 293

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
### 9 - requests/cookies.py:

Start line: 333, End line: 365

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
### 10 - requests/cookies.py:

Start line: 174, End line: 188

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
### 11 - requests/cookies.py:

Start line: 120, End line: 145

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
### 12 - requests/cookies.py:

Start line: 103, End line: 117

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
### 13 - requests/sessions.py:

Start line: 68, End line: 150

```python
class SessionRedirectMixin(object):
    def resolve_redirects(self, resp, req, stream=False, timeout=None,
                          verify=True, cert=None, proxies=None):
        """Receives a Response. Returns a generator of Responses."""

        i = 0

        # ((resp.status_code is codes.see_other))
        while (('location' in resp.headers and resp.status_code in REDIRECT_STATI)):
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
            if '://' in url:
                scheme, uri = url.split('://', 1)
                url = '%s://%s' % (scheme.lower(), uri)

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
            if (resp.status_code in (codes.moved, codes.found) and
                    method not in ('GET', 'HEAD')):
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
### 14 - requests/models.py:

Start line: 464, End line: 480

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
### 15 - requests/cookies.py:

Start line: 82, End line: 100

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
### 16 - requests/cookies.py:

Start line: 368, End line: 407

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


def cookiejar_from_dict(cookie_dict, cookiejar=None):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        for name in cookie_dict:
            cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
    return cookiejar
```
### 20 - requests/sessions.py:

Start line: 361, End line: 389

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
### 24 - requests/models.py:

Start line: 1, End line: 36

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
### 25 - requests/sessions.py:

Start line: 391, End line: 399

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
### 26 - requests/sessions.py:

Start line: 430, End line: 490

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
### 27 - requests/sessions.py:

Start line: 401, End line: 409

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
### 28 - requests/sessions.py:

Start line: 492, End line: 528

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
### 29 - requests/sessions.py:

Start line: 228, End line: 266

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
### 34 - requests/sessions.py:

Start line: 268, End line: 359

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
### 42 - requests/sessions.py:

Start line: 411, End line: 428

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
### 45 - requests/models.py:

Start line: 483, End line: 554

```python
class Response(object):
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

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
### 51 - requests/sessions.py:

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
### 67 - requests/models.py:

Start line: 692, End line: 731

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
### 71 - requests/models.py:

Start line: 275, End line: 307

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare(self, method=None, url=None, headers=None, files=None,
                data=None, params=None, auth=None, cookies=None, hooks=None):
        """Prepares the the entire request with the given parameters."""

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
        p.headers = self.headers
        p.body = self.body
        p.hooks = self.hooks
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = self.method.upper()
```
### 73 - requests/models.py:

Start line: 430, End line: 462

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_content_length(self, body):
        if hasattr(body, 'seek') and hasattr(body, 'tell'):
            body.seek(0, 2)
            self.headers['Content-Length'] = str(body.tell())
            body.seek(0, 0)
        elif body is not None:
            l = super_len(body)
            if l:
                self.headers['Content-Length'] = str(l)
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
### 74 - requests/models.py:

Start line: 369, End line: 428

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
                self.headers['Content-Length'] = str(length)
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
### 77 - requests/models.py:

Start line: 244, End line: 273

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
### 82 - requests/models.py:

Start line: 146, End line: 167

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
### 100 - requests/models.py:

Start line: 619, End line: 641

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
### 104 - requests/models.py:

Start line: 593, End line: 617

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
### 106 - requests/models.py:

Start line: 170, End line: 241

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
### 107 - requests/models.py:

Start line: 309, End line: 367

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

        # Support for unicode domain names and paths.
        scheme, auth, host, port, path, query, fragment = parse_url(url)

        if not scheme:
            raise MissingSchema("Invalid URL %r: No schema supplied" % url)

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
### 128 - requests/models.py:

Start line: 556, End line: 591

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
                while 1:
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
### 137 - requests/models.py:

Start line: 39, End line: 86

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
### 164 - requests/models.py:

Start line: 676, End line: 690

```python
class Response(object):

    def json(self, **kwargs):
        """Returns the json-encoded content of a response, if any.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        """

        if not self.encoding and len(self.content) > 3:
            # No encoding set. JSON RFC 4627 section 3 states we should expect
            # UTF-8, -16 or -32. Detect which one to use; If the detection or
            # decoding fails, fall back to `self.text` (using chardet to make
            # a best guess).
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                return json.loads(self.content.decode(encoding), **kwargs)
        return json.loads(self.text or self.content, **kwargs)
```
### 172 - requests/models.py:

Start line: 643, End line: 674

```python
class Response(object):

    @property
    def text(self):
        """Content of the response, in unicode.

        if Response.encoding is None and chardet module is available, encoding
        will be guessed.
        """

        # Try charset from content-type
        content = None
        encoding = self.encoding

        if not self.content:
            return str('')

        # Fallback to auto-detected encoding.
        if self.encoding is None:
            encoding = self.apparent_encoding

        # Decode unicode from given encoding.
        try:
            content = str(self.content, encoding, errors='replace')
        except (LookupError, TypeError):
            # A LookupError is raised if the encoding was not found which could
            # indicate a misspelling or similar mistake.
            #
            # A TypeError can be raised if encoding is None
            #
            # So we try blindly encoding.
            content = str(self.content, errors='replace')

        return content
```
### 173 - requests/models.py:

Start line: 88, End line: 143

```python
class RequestEncodingMixin(object):

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but abritrary
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
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                else:
                    fn, fp, ft = v
            else:
                fn = guess_filename(v) or k
                fp = v
            if isinstance(fp, str):
                fp = StringIO(fp)
            if isinstance(fp, bytes):
                fp = BytesIO(fp)

            if ft:
                new_v = (fn, fp.read(), ft)
            else:
                new_v = (fn, fp.read())
            new_fields.append((k, new_v))

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type
```
