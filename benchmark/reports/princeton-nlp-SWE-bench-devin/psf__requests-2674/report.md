# psf__requests-2674

| **psf/requests** | `0be38a0c37c59c4b66ce908731da15b401655113` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 56283 |
| **Any found context length** | 18373 |
| **Avg pos** | 239.0 |
| **Min pos** | 63 |
| **Max pos** | 176 |
| **Top file pos** | 20 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/adapters.py b/requests/adapters.py
--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -19,6 +19,7 @@
 from .utils import (DEFAULT_CA_BUNDLE_PATH, get_encoding_from_headers,
                     prepend_scheme_if_needed, get_auth_from_url, urldefragauth)
 from .structures import CaseInsensitiveDict
+from .packages.urllib3.exceptions import ClosedPoolError
 from .packages.urllib3.exceptions import ConnectTimeoutError
 from .packages.urllib3.exceptions import HTTPError as _HTTPError
 from .packages.urllib3.exceptions import MaxRetryError
@@ -421,6 +422,9 @@ def send(self, request, stream=False, timeout=None, verify=True, cert=None, prox
 
             raise ConnectionError(e, request=request)
 
+        except ClosedPoolError as e:
+            raise ConnectionError(e, request=request)
+
         except _ProxyError as e:
             raise ProxyError(e)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/adapters.py | 22 | 22 | 63 | 20 | 18373
| requests/adapters.py | 424 | 424 | 176 | 20 | 56283


## Problem Statement

```
urllib3 exceptions passing through requests API
I don't know if it's a design goal of requests to hide urllib3's exceptions and wrap them around requests.exceptions types.

(If it's not IMHO it should be, but that's another discussion)

If it is, I have at least two of them passing through that I have to catch in addition to requests' exceptions. They are requests.packages.urllib3.exceptions.DecodeError and requests.packages.urllib3.exceptions.TimeoutError (this one I get when a proxy timeouts)

Thanks!


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 requests/packages/urllib3/exceptions.py | 2 | 57| 296 | 296 | 976 | 
| 2 | 2 requests/packages/urllib3/util/__init__.py | 1 | 25| 114 | 410 | 1090 | 
| 3 | 3 requests/packages/urllib3/response.py | 1 | 65| 360 | 770 | 4546 | 
| 4 | 4 requests/__init__.py | 1 | 78| 293 | 1063 | 5043 | 
| 5 | 5 requests/exceptions.py | 1 | 27| 163 | 1226 | 5541 | 
| 6 | 5 requests/packages/urllib3/exceptions.py | 81 | 170| 545 | 1771 | 5541 | 
| 7 | 6 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 1771 | 5556 | 
| 8 | 7 requests/packages/urllib3/connectionpool.py | 1 | 50| 251 | 2022 | 11794 | 
| 9 | 8 requests/packages/urllib3/contrib/pyopenssl.py | 169 | 190| 176 | 2198 | 13924 | 
| 10 | 8 requests/packages/urllib3/connectionpool.py | 300 | 315| 247 | 2445 | 13924 | 
| 11 | 8 requests/packages/urllib3/contrib/pyopenssl.py | 192 | 215| 153 | 2598 | 13924 | 
| 12 | 9 requests/packages/urllib3/_collections.py | 1 | 23| 124 | 2722 | 16246 | 
| 13 | 10 requests/packages/urllib3/connection.py | 1 | 60| 301 | 3023 | 18234 | 
| 14 | 10 requests/packages/urllib3/connectionpool.py | 610 | 650| 405 | 3428 | 18234 | 
| 15 | 11 requests/packages/urllib3/poolmanager.py | 1 | 28| 167 | 3595 | 20304 | 
| 16 | 12 requests/packages/urllib3/util/request.py | 1 | 72| 231 | 3826 | 20778 | 
| 17 | 12 requests/packages/urllib3/response.py | 340 | 401| 439 | 4265 | 20778 | 
| 18 | 12 requests/packages/urllib3/exceptions.py | 60 | 78| 132 | 4397 | 20778 | 
| 19 | 12 requests/packages/urllib3/connectionpool.py | 504 | 608| 850 | 5247 | 20778 | 
| 20 | 12 requests/packages/urllib3/response.py | 403 | 423| 232 | 5479 | 20778 | 
| 21 | 13 requests/packages/urllib3/__init__.py | 1 | 35| 208 | 5687 | 21224 | 
| 22 | 13 requests/exceptions.py | 30 | 100| 335 | 6022 | 21224 | 
| 23 | 13 requests/packages/urllib3/poolmanager.py | 265 | 281| 168 | 6190 | 21224 | 
| 24 | 14 requests/packages/urllib3/util/ssl_.py | 47 | 102| 477 | 6667 | 23576 | 
| 25 | 14 requests/packages/urllib3/poolmanager.py | 192 | 263| 572 | 7239 | 23576 | 
| 26 | 14 requests/packages/urllib3/contrib/pyopenssl.py | 217 | 249| 191 | 7430 | 23576 | 
| 27 | 14 requests/packages/urllib3/poolmanager.py | 31 | 73| 293 | 7723 | 23576 | 
| 28 | 14 requests/packages/urllib3/connection.py | 143 | 156| 158 | 7881 | 23576 | 
| 29 | 14 requests/packages/urllib3/response.py | 176 | 203| 247 | 8128 | 23576 | 
| 30 | 14 requests/packages/urllib3/connectionpool.py | 278 | 298| 146 | 8274 | 23576 | 
| 31 | 14 requests/packages/urllib3/connectionpool.py | 711 | 727| 151 | 8425 | 23576 | 
| 32 | 15 requests/packages/urllib3/util/retry.py | 106 | 142| 313 | 8738 | 25775 | 
| 33 | 15 requests/packages/urllib3/contrib/pyopenssl.py | 149 | 167| 151 | 8889 | 25775 | 
| 34 | 15 requests/packages/urllib3/util/retry.py | 144 | 156| 116 | 9005 | 25775 | 
| 35 | 16 requests/packages/urllib3/util/timeout.py | 138 | 152| 136 | 9141 | 27784 | 
| 36 | 17 requests/packages/urllib3/filepost.py | 1 | 37| 172 | 9313 | 28310 | 
| 37 | 17 requests/packages/urllib3/connection.py | 202 | 265| 534 | 9847 | 28310 | 
| 38 | 18 requests/packages/urllib3/request.py | 1 | 50| 353 | 10200 | 29496 | 
| 39 | 18 requests/packages/urllib3/response.py | 286 | 310| 208 | 10408 | 29496 | 
| 40 | 18 requests/packages/urllib3/contrib/pyopenssl.py | 100 | 115| 152 | 10560 | 29496 | 
| 41 | 19 requests/utils.py | 488 | 531| 374 | 10934 | 34620 | 
| 42 | **20 requests/adapters.py** | 233 | 262| 229 | 11163 | 37948 | 
| 43 | **20 requests/adapters.py** | 137 | 158| 187 | 11350 | 37948 | 
| 44 | 20 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 97| 747 | 12097 | 37948 | 
| 45 | 20 requests/packages/urllib3/contrib/pyopenssl.py | 252 | 294| 363 | 12460 | 37948 | 
| 46 | 20 requests/packages/urllib3/_collections.py | 306 | 324| 157 | 12617 | 37948 | 
| 47 | **20 requests/adapters.py** | 54 | 115| 550 | 13167 | 37948 | 
| 48 | 20 requests/packages/urllib3/util/ssl_.py | 1 | 45| 420 | 13587 | 37948 | 
| 49 | 20 requests/packages/urllib3/poolmanager.py | 141 | 189| 391 | 13978 | 37948 | 
| 50 | 21 requests/packages/urllib3/contrib/ntlmpool.py | 41 | 115| 714 | 14692 | 38909 | 
| 51 | 21 requests/packages/urllib3/util/timeout.py | 154 | 167| 130 | 14822 | 38909 | 
| 52 | 21 requests/packages/urllib3/connection.py | 159 | 178| 170 | 14992 | 38909 | 
| 53 | 21 requests/packages/urllib3/connectionpool.py | 694 | 709| 121 | 15113 | 38909 | 
| 54 | 22 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 14| 0 | 15113 | 39006 | 
| 55 | 22 requests/packages/urllib3/_collections.py | 71 | 104| 201 | 15314 | 39006 | 
| 56 | 22 requests/packages/urllib3/connection.py | 63 | 118| 556 | 15870 | 39006 | 
| 57 | 23 requests/models.py | 1 | 49| 381 | 16251 | 45206 | 
| 58 | 23 requests/packages/urllib3/__init__.py | 37 | 70| 238 | 16489 | 45206 | 
| 59 | 23 requests/packages/urllib3/response.py | 312 | 338| 210 | 16699 | 45206 | 
| 60 | 23 requests/packages/urllib3/util/timeout.py | 1 | 98| 842 | 17541 | 45206 | 
| 61 | 24 requests/auth.py | 1 | 58| 330 | 17871 | 46849 | 
| 62 | 24 requests/packages/urllib3/connectionpool.py | 753 | 768| 144 | 18015 | 46849 | 
| **-> 63 <-** | **24 requests/adapters.py** | 1 | 51| 358 | 18373 | 46849 | 
| 64 | 25 requests/sessions.py | 229 | 264| 189 | 18562 | 51925 | 
| 65 | 25 requests/packages/urllib3/response.py | 68 | 174| 820 | 19382 | 51925 | 
| 66 | 25 requests/packages/urllib3/connectionpool.py | 317 | 384| 655 | 20037 | 51925 | 
| 67 | 25 requests/packages/urllib3/filepost.py | 40 | 55| 120 | 20157 | 51925 | 
| 68 | 26 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 20315 | 54767 | 
| 69 | 26 requests/packages/urllib3/util/timeout.py | 193 | 209| 123 | 20438 | 54767 | 
| 70 | 26 requests/packages/urllib3/response.py | 425 | 467| 333 | 20771 | 54767 | 
| 71 | 26 requests/packages/urllib3/connectionpool.py | 157 | 208| 371 | 21142 | 54767 | 
| 72 | 26 requests/utils.py | 1 | 67| 334 | 21476 | 54767 | 
| 73 | 26 requests/packages/urllib3/response.py | 205 | 284| 651 | 22127 | 54767 | 
| 74 | 26 requests/packages/urllib3/util/timeout.py | 211 | 241| 282 | 22409 | 54767 | 
| 75 | 26 requests/packages/urllib3/util/timeout.py | 169 | 191| 196 | 22605 | 54767 | 
| 76 | 26 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 39| 256 | 22861 | 54767 | 
| 77 | 27 requests/cookies.py | 1 | 46| 299 | 23160 | 58568 | 
| 78 | **27 requests/adapters.py** | 197 | 231| 273 | 23433 | 58568 | 
| 79 | 27 requests/packages/urllib3/connectionpool.py | 53 | 91| 261 | 23694 | 58568 | 
| 80 | 27 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 24206 | 58568 | 
| 81 | 27 requests/cookies.py | 94 | 112| 134 | 24340 | 58568 | 
| 82 | 27 requests/packages/urllib3/util/ssl_.py | 181 | 241| 575 | 24915 | 58568 | 
| 83 | 27 requests/packages/urllib3/util/timeout.py | 100 | 136| 320 | 25235 | 58568 | 
| 84 | 27 requests/utils.py | 70 | 114| 327 | 25562 | 58568 | 
| 85 | 27 requests/packages/urllib3/connection.py | 181 | 200| 145 | 25707 | 58568 | 
| 86 | 28 requests/packages/urllib3/packages/ordered_dict.py | 54 | 89| 293 | 26000 | 60725 | 
| 87 | 28 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 26739 | 60725 | 
| 88 | 28 requests/packages/urllib3/packages/ordered_dict.py | 114 | 140| 200 | 26939 | 60725 | 
| 89 | 28 requests/packages/urllib3/_collections.py | 56 | 69| 152 | 27091 | 60725 | 
| 90 | 29 requests/packages/urllib3/fields.py | 137 | 154| 142 | 27233 | 61988 | 
| 91 | 29 requests/packages/urllib3/_collections.py | 184 | 205| 155 | 27388 | 61988 | 
| 92 | **29 requests/adapters.py** | 302 | 322| 174 | 27562 | 61988 | 
| 93 | 29 requests/packages/urllib3/connectionpool.py | 386 | 419| 224 | 27786 | 61988 | 
| 94 | 30 requests/packages/urllib3/util/url.py | 1 | 43| 269 | 28055 | 63442 | 
| 95 | 31 requests/status_codes.py | 1 | 90| 918 | 28973 | 64360 | 
| 96 | 31 requests/packages/urllib3/connection.py | 120 | 141| 147 | 29120 | 64360 | 
| 97 | 31 requests/packages/urllib3/connectionpool.py | 729 | 751| 179 | 29299 | 64360 | 
| 98 | 32 requests/packages/urllib3/util/response.py | 1 | 23| 118 | 29417 | 64478 | 
| 99 | 32 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 29947 | 64478 | 
| 100 | 32 requests/packages/urllib3/contrib/pyopenssl.py | 118 | 146| 207 | 30154 | 64478 | 
| 101 | 32 requests/packages/urllib3/packages/ordered_dict.py | 44 | 52| 134 | 30288 | 64478 | 
| 102 | 33 requests/packages/__init__.py | 1 | 4| 0 | 30288 | 64492 | 
| 103 | 33 requests/cookies.py | 48 | 60| 124 | 30412 | 64492 | 
| 104 | 33 requests/packages/urllib3/connectionpool.py | 653 | 692| 385 | 30797 | 64492 | 
| 105 | 33 requests/packages/urllib3/util/retry.py | 158 | 207| 383 | 31180 | 64492 | 
| 106 | 33 requests/packages/urllib3/_collections.py | 256 | 304| 342 | 31522 | 64492 | 
| 107 | 33 requests/utils.py | 533 | 569| 305 | 31827 | 64492 | 
| 108 | 33 requests/packages/urllib3/util/ssl_.py | 142 | 178| 256 | 32083 | 64492 | 
| 109 | 33 requests/packages/urllib3/request.py | 52 | 81| 283 | 32366 | 64492 | 
| 110 | 33 requests/packages/urllib3/util/ssl_.py | 244 | 281| 373 | 32739 | 64492 | 
| 111 | 34 requests/compat.py | 1 | 63| 394 | 33133 | 64886 | 
| 112 | 35 requests/packages/urllib3/util/connection.py | 1 | 43| 309 | 33442 | 65625 | 
| 113 | 35 requests/packages/urllib3/util/url.py | 121 | 215| 606 | 34048 | 65625 | 
| 114 | 35 requests/sessions.py | 609 | 631| 220 | 34268 | 65625 | 
| 115 | 36 docs/conf.py | 1 | 139| 1071 | 35339 | 67561 | 
| 116 | 36 requests/packages/urllib3/fields.py | 70 | 102| 274 | 35613 | 67561 | 
| 117 | 36 requests/auth.py | 61 | 156| 842 | 36455 | 67561 | 
| 118 | 36 requests/packages/urllib3/packages/ordered_dict.py | 1 | 42| 369 | 36824 | 67561 | 
| 119 | 36 requests/cookies.py | 115 | 129| 150 | 36974 | 67561 | 
| 120 | 36 requests/utils.py | 390 | 415| 208 | 37182 | 67561 | 
| 121 | **36 requests/adapters.py** | 264 | 286| 191 | 37373 | 67561 | 
| 122 | 36 requests/models.py | 632 | 687| 422 | 37795 | 67561 | 
| 123 | 36 requests/cookies.py | 62 | 91| 208 | 38003 | 67561 | 
| 124 | 36 requests/packages/urllib3/util/retry.py | 1 | 104| 803 | 38806 | 67561 | 
| 125 | 36 requests/sessions.py | 1 | 39| 260 | 39066 | 67561 | 
| 126 | 36 requests/sessions.py | 91 | 202| 862 | 39928 | 67561 | 
| 127 | 36 requests/models.py | 324 | 396| 591 | 40519 | 67561 | 
| 128 | 36 requests/packages/urllib3/util/retry.py | 209 | 286| 598 | 41117 | 67561 | 
| 129 | 36 requests/utils.py | 650 | 661| 152 | 41269 | 67561 | 
| 130 | 36 requests/auth.py | 198 | 213| 150 | 41419 | 67561 | 
| 131 | 36 requests/utils.py | 572 | 618| 281 | 41700 | 67561 | 
| 132 | 36 requests/packages/urllib3/_collections.py | 230 | 254| 209 | 41909 | 67561 | 
| 133 | 36 requests/utils.py | 418 | 435| 197 | 42106 | 67561 | 
| 134 | 36 requests/packages/urllib3/request.py | 83 | 142| 560 | 42666 | 67561 | 
| 135 | 36 requests/sessions.py | 204 | 227| 189 | 42855 | 67561 | 
| 136 | 37 requests/packages/chardet/__init__.py | 18 | 33| 114 | 42969 | 67857 | 
| 137 | 37 requests/packages/urllib3/fields.py | 104 | 135| 232 | 43201 | 67857 | 
| 138 | 38 requests/hooks.py | 1 | 46| 188 | 43389 | 68046 | 
| 139 | 38 requests/models.py | 473 | 493| 165 | 43554 | 68046 | 
| 140 | 38 requests/packages/urllib3/_collections.py | 207 | 228| 222 | 43776 | 68046 | 
| 141 | 38 requests/packages/urllib3/_collections.py | 107 | 182| 596 | 44372 | 68046 | 
| 142 | 38 requests/packages/urllib3/poolmanager.py | 101 | 139| 313 | 44685 | 68046 | 
| 143 | 38 requests/sessions.py | 267 | 344| 550 | 45235 | 68046 | 
| 144 | 38 requests/packages/urllib3/fields.py | 49 | 68| 129 | 45364 | 68046 | 
| 145 | 38 requests/packages/urllib3/connectionpool.py | 771 | 796| 204 | 45568 | 68046 | 
| 146 | 38 requests/packages/urllib3/connectionpool.py | 421 | 503| 793 | 46361 | 68046 | 
| 147 | 38 requests/models.py | 807 | 849| 291 | 46652 | 68046 | 
| 148 | 38 requests/packages/urllib3/filepost.py | 58 | 94| 233 | 46885 | 68046 | 
| 149 | 38 requests/packages/urllib3/fields.py | 156 | 178| 178 | 47063 | 68046 | 
| 150 | 39 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 1 | 64| 539 | 47602 | 68969 | 
| 151 | 40 requests/packages/chardet/universaldetector.py | 134 | 171| 316 | 47918 | 70651 | 
| 152 | 40 requests/packages/urllib3/util/url.py | 45 | 86| 334 | 48252 | 70651 | 
| 153 | 40 requests/models.py | 288 | 303| 155 | 48407 | 70651 | 
| 154 | **40 requests/adapters.py** | 117 | 135| 203 | 48610 | 70651 | 
| 155 | 40 requests/models.py | 689 | 718| 196 | 48806 | 70651 | 
| 156 | 40 requests/packages/chardet/universaldetector.py | 64 | 132| 809 | 49615 | 70651 | 
| 157 | 40 requests/packages/urllib3/packages/ordered_dict.py | 91 | 112| 178 | 49793 | 70651 | 
| 158 | 40 requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py | 67 | 106| 350 | 50143 | 70651 | 
| 159 | 40 requests/auth.py | 158 | 196| 332 | 50475 | 70651 | 
| 160 | 40 requests/packages/urllib3/packages/ordered_dict.py | 173 | 260| 665 | 51140 | 70651 | 
| 161 | 40 requests/packages/urllib3/connectionpool.py | 94 | 155| 550 | 51690 | 70651 | 
| 162 | 40 requests/packages/chardet/universaldetector.py | 29 | 41| 127 | 51817 | 70651 | 
| 163 | 40 requests/cookies.py | 343 | 375| 257 | 52074 | 70651 | 
| 164 | 41 requests/packages/chardet/compat.py | 21 | 35| 69 | 52143 | 70917 | 
| 165 | 41 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 52812 | 70917 | 
| 166 | 41 requests/utils.py | 664 | 709| 269 | 53081 | 70917 | 
| 167 | 41 requests/models.py | 52 | 99| 294 | 53375 | 70917 | 
| 168 | 42 requests/packages/chardet/constants.py | 29 | 40| 54 | 53429 | 71235 | 
| 169 | 42 docs/conf.py | 140 | 249| 639 | 54068 | 71235 | 
| 170 | 42 requests/models.py | 305 | 322| 149 | 54217 | 71235 | 
| 171 | 42 requests/cookies.py | 160 | 186| 255 | 54472 | 71235 | 
| 172 | 42 requests/models.py | 398 | 459| 412 | 54884 | 71235 | 
| 173 | 42 requests/packages/urllib3/util/ssl_.py | 105 | 139| 249 | 55133 | 71235 | 
| 174 | 42 requests/cookies.py | 297 | 308| 143 | 55276 | 71235 | 
| 175 | 42 requests/packages/urllib3/_collections.py | 26 | 54| 206 | 55482 | 71235 | 
| **-> 176 <-** | **42 requests/adapters.py** | 324 | 436| 801 | 56283 | 71235 | 
| 177 | 42 requests/packages/urllib3/connectionpool.py | 210 | 246| 317 | 56600 | 71235 | 
| 178 | 42 requests/utils.py | 621 | 647| 336 | 56936 | 71235 | 
| 179 | 43 requests/api.py | 72 | 95| 180 | 57116 | 72586 | 
| 180 | 43 requests/packages/chardet/universaldetector.py | 44 | 62| 175 | 57291 | 72586 | 
| 181 | 43 requests/utils.py | 355 | 387| 191 | 57482 | 72586 | 
| 182 | 43 requests/packages/urllib3/util/connection.py | 46 | 99| 429 | 57911 | 72586 | 
| 183 | 43 requests/cookies.py | 132 | 157| 218 | 58129 | 72586 | 
| 184 | 43 requests/models.py | 461 | 471| 132 | 58261 | 72586 | 
| 185 | 43 requests/sessions.py | 386 | 467| 659 | 58920 | 72586 | 
| 186 | 43 requests/models.py | 161 | 182| 161 | 59081 | 72586 | 
| 187 | 43 requests/cookies.py | 310 | 322| 158 | 59239 | 72586 | 
| 188 | 44 requests/structures.py | 1 | 86| 572 | 59811 | 73272 | 
| 189 | 44 requests/cookies.py | 204 | 295| 761 | 60572 | 73272 | 
| 190 | 45 requests/packages/chardet/latin1prober.py | 84 | 94| 318 | 60890 | 75354 | 
| 191 | 45 requests/models.py | 525 | 630| 800 | 61690 | 75354 | 
| 192 | 45 requests/models.py | 185 | 251| 541 | 62231 | 75354 | 
| 193 | 46 requests/packages/chardet/charsetgroupprober.py | 58 | 76| 152 | 62383 | 76205 | 
| 194 | 47 requests/packages/chardet/utf8prober.py | 28 | 77| 361 | 62744 | 76815 | 
| 195 | 47 requests/packages/urllib3/fields.py | 21 | 46| 217 | 62961 | 76815 | 
| 196 | 47 requests/utils.py | 309 | 352| 237 | 63198 | 76815 | 
| 197 | 47 requests/packages/urllib3/util/url.py | 88 | 118| 224 | 63422 | 76815 | 
| 198 | 48 requests/packages/chardet/langcyrillicmodel.py | 125 | 142| 649 | 64071 | 89681 | 
| 199 | 48 requests/packages/urllib3/poolmanager.py | 75 | 99| 190 | 64261 | 89681 | 
| 200 | 48 requests/packages/urllib3/fields.py | 1 | 18| 107 | 64368 | 89681 | 
| 201 | 49 requests/packages/chardet/mbcssm.py | 482 | 500| 311 | 64679 | 101267 | 
| 202 | 49 requests/packages/chardet/latin1prober.py | 29 | 83| 1165 | 65844 | 101267 | 
| 203 | 49 requests/packages/urllib3/connectionpool.py | 248 | 276| 216 | 66060 | 101267 | 
| 204 | 50 requests/certs.py | 1 | 26| 133 | 66193 | 101400 | 
| 205 | 50 requests/packages/chardet/langcyrillicmodel.py | 106 | 123| 649 | 66842 | 101400 | 
| 206 | **50 requests/adapters.py** | 288 | 300| 141 | 66983 | 101400 | 
| 207 | 50 requests/sessions.py | 42 | 88| 326 | 67309 | 101400 | 
| 208 | 50 requests/cookies.py | 188 | 202| 151 | 67460 | 101400 | 
| 209 | 51 requests/packages/chardet/langhebrewmodel.py | 37 | 54| 670 | 68130 | 110737 | 
| 210 | 52 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 69406 | 112013 | 


### Hint

```
I definitely agree with you and would agree that these should be wrapped.

Could you give us stack-traces so we can find where they're bleeding through?

Sorry I don't have stack traces readily available :/

No worries. I have ideas as to where the DecodeError might be coming from but I'm not certain where the TimeoutError could be coming from.

If you run into them again, please save us the stack traces. =) Thanks for reporting them. (We'll never know what we're missing until someone tells us.)

`TimeoutError` is almost certainly being raised from either [`HTTPConnectionPool.urlopen()`](https://github.com/kennethreitz/requests/blob/master/requests/adapters.py#L282-L293) or from [`HTTPConnection.putrequest()`](https://github.com/kennethreitz/requests/blob/master/requests/adapters.py#L301). Adding a new clause to [here](https://github.com/kennethreitz/requests/blob/master/requests/adapters.py#L323-L335) should cover us.

Actually, that can't be right, we should be catching and rethrowing as a Requests `Timeout` exception in that block. Hmm, I'll do another spin through the code to see if I can see the problem.

Yeah, a quick search of the `urllib3` code reveals that the only place that `TimeoutError`s are thrown is from `HTTPConnectionPool.urlopen()`. These should not be leaking. We really need a stack trace to track this down.

I've added a few logs to get the traces if they happen again. What may have confused me for the TimeoutError is that requests' Timeout actually wraps the urllib3's TimeoutError and we were logging the content of the error as well. 

So DecodeError was definitely being thrown but probably not TimeoutError, sorry for the confusion. I'll report here it I ever see it happening now that we're watching for it.

Thanks for the help!

I also got urllib3 exceptions passing through when use Session in several threads, trace:

\`\`\`
......
  File "C:\Python27\lib\site-packages\requests\sessions.py", line 347, in get
    return self.request('GET', url, **kwargs)
  File "C:\Python27\lib\site-packages\requests\sessions.py", line 335, in request
    resp = self.send(prep, **send_kwargs)
  File "C:\Python27\lib\site-packages\requests\sessions.py", line 438, in send
    r = adapter.send(request, **kwargs)
  File "C:\Python27\lib\site-packages\requests\adapters.py", line 292, in send
    timeout=timeout
  File "C:\Python27\lib\site-packages\requests\packages\urllib3\connectionpool.py", line 423, in url
open
    conn = self._get_conn(timeout=pool_timeout)
  File "C:\Python27\lib\site-packages\requests\packages\urllib3\connectionpool.py", line 224, in _ge
t_conn
    raise ClosedPoolError(self, "Pool is closed.")
ClosedPoolError: HTTPConnectionPool(host='......', port=80): Pool is closed.
\`\`\`

Ah, we should rewrap that `ClosedPoolError` too.

But it's still the summer... How can any pool be closed? :smirk_cat: 

But yes :+1:

I've added a fix for the `ClosedPoolError` to #1475. Which apparently broke in the last month for no adequately understandable reason.

If it's still needed, here is the traceback of DecodeError I got using proxy on requests 2.0.0:

\`\`\`
Traceback (most recent call last):
  File "/home/krat/Projects/Grubhub/source/Pit/pit/web.py", line 52, in request
    response = session.request(method, url, **kw)
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/sessions.py", line 357, in request
    resp = self.send(prep, **send_kwargs)
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/sessions.py", line 460, in send
    r = adapter.send(request, **kwargs)
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/adapters.py", line 367, in send
    r.content
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/models.py", line 633, in content
    self._content = bytes().join(self.iter_content(CONTENT_CHUNK_SIZE)) or bytes()
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/models.py", line 572, in generate
    decode_content=True):
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/packages/urllib3/response.py", line 225, in stream
    data = self.read(amt=amt, decode_content=decode_content)
  File "/home/krat/.virtualenvs/grubhub/local/lib/python2.7/site-packages/requests/packages/urllib3/response.py", line 193, in read
    e)
DecodeError: ('Received response with content-encoding: gzip, but failed to decode it.', error('Error -3 while decompressing: incorrect header check',))
\`\`\`

Slightly different to the above, but urllib3's LocationParseError leaks through which could probably do with being wrapped in InvalidURL.

\`\`\`
Traceback (most recent call last):
  File "/home/oliver/wc/trunk/mtmCore/python/asagent/samplers/net/web.py", line 255, in process_url
    resp = self.request(self.params.httpverb, url, data=data)
  File "/home/oliver/wc/trunk/mtmCore/python/asagent/samplers/net/web.py", line 320, in request
    verb, url, data=data))
  File "abilisoft/requests/opt/abilisoft.com/thirdparty/requests/lib/python2.7/site-packages/requests/sessions.py", line 286, in prepare_request
  File "abilisoft/requests/opt/abilisoft.com/thirdparty/requests/lib/python2.7/site-packages/requests/models.py", line 286, in prepare
  File "abilisoft/requests/opt/abilisoft.com/thirdparty/requests/lib/python2.7/site-packages/requests/models.py", line 333, in prepare_url
  File "abilisoft/requests/opt/abilisoft.com/thirdparty/requests/lib/python2.7/site-packages/requests/packages/urllib3/util.py", line 397, in parse_url
LocationParseError: Failed to parse: Failed to parse: fe80::5054:ff:fe5a:fc0
\`\`\`

```

## Patch

```diff
diff --git a/requests/adapters.py b/requests/adapters.py
--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -19,6 +19,7 @@
 from .utils import (DEFAULT_CA_BUNDLE_PATH, get_encoding_from_headers,
                     prepend_scheme_if_needed, get_auth_from_url, urldefragauth)
 from .structures import CaseInsensitiveDict
+from .packages.urllib3.exceptions import ClosedPoolError
 from .packages.urllib3.exceptions import ConnectTimeoutError
 from .packages.urllib3.exceptions import HTTPError as _HTTPError
 from .packages.urllib3.exceptions import MaxRetryError
@@ -421,6 +422,9 @@ def send(self, request, stream=False, timeout=None, verify=True, cert=None, prox
 
             raise ConnectionError(e, request=request)
 
+        except ClosedPoolError as e:
+            raise ConnectionError(e, request=request)
+
         except _ProxyError as e:
             raise ProxyError(e)
 

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -1655,6 +1655,16 @@ def test_urllib3_retries():
     with pytest.raises(RetryError):
         s.get(httpbin('status/500'))
 
+
+def test_urllib3_pool_connection_closed():
+    s = requests.Session()
+    s.mount('http://', HTTPAdapter(pool_connections=0, pool_maxsize=0))
+
+    try:
+        s.get(httpbin('status/200'))
+    except ConnectionError as e:
+        assert u"HTTPConnectionPool(host='httpbin.org', port=80): Pool is closed." in str(e.message)
+
 def test_vendor_aliases():
     from requests.packages import urllib3
     from requests.packages import chardet

```


## Code snippets

### 1 - requests/packages/urllib3/exceptions.py:

Start line: 2, End line: 57

```python
## Base Exceptions

class HTTPError(Exception):
    "Base exception used by this module."
    pass

class HTTPWarning(Warning):
    "Base warning used by this module."
    pass



class PoolError(HTTPError):
    "Base exception for errors caused within a pool."
    def __init__(self, pool, message):
        self.pool = pool
        HTTPError.__init__(self, "%s: %s" % (pool, message))

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, None)


class RequestError(PoolError):
    "Base exception for PoolErrors that have associated URLs."
    def __init__(self, pool, url, message):
        self.url = url
        PoolError.__init__(self, pool, message)

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, self.url, None)


class SSLError(HTTPError):
    "Raised when SSL certificate fails in an HTTPS connection."
    pass


class ProxyError(HTTPError):
    "Raised when the connection to a proxy fails."
    pass


class DecodeError(HTTPError):
    "Raised when automatic decoding based on Content-Type fails."
    pass


class ProtocolError(HTTPError):
    "Raised when something unexpected happens mid-request/response."
    pass


#: Renamed to ProtocolError but aliased for backwards compatibility.
ConnectionError = ProtocolError
```
### 2 - requests/packages/urllib3/util/__init__.py:

Start line: 1, End line: 25

```python
# For backwards compatibility, provide imports that used to be here.
from .connection import is_connection_dropped
from .request import make_headers
from .response import is_fp_closed
from .ssl_ import (
    SSLContext,
    HAS_SNI,
    assert_fingerprint,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .timeout import (
    current_time,
    Timeout,
)

from .retry import Retry
from .url import (
    get_host,
    parse_url,
    split_first,
    Url,
)
```
### 3 - requests/packages/urllib3/response.py:

Start line: 1, End line: 65

```python
try:
    import http.client as httplib
except ImportError:
    import httplib
import zlib
import io
from socket import timeout as SocketTimeout

from ._collections import HTTPHeaderDict
from .exceptions import (
    ProtocolError, DecodeError, ReadTimeoutError, ResponseNotChunked
)
from .packages.six import string_types as basestring, binary_type, PY3
from .connection import HTTPException, BaseSSLError
from .util.response import is_fp_closed


class DeflateDecoder(object):

    def __init__(self):
        self._first_try = True
        self._data = binary_type()
        self._obj = zlib.decompressobj()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        if not data:
            return data

        if not self._first_try:
            return self._obj.decompress(data)

        self._data += data
        try:
            return self._obj.decompress(data)
        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                return self.decompress(self._data)
            finally:
                self._data = None


class GzipDecoder(object):

    def __init__(self):
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        if not data:
            return data
        return self._obj.decompress(data)


def _get_decoder(mode):
    if mode == 'gzip':
        return GzipDecoder()

    return DeflateDecoder()
```
### 4 - requests/__init__.py:

Start line: 1, End line: 78

```python
# -*- coding: utf-8 -*-

#   __
#  /__)  _  _     _   _ _/   _
# / (   (- (/ (/ (- _)  /  _)
#          /

__title__ = 'requests'
__version__ = '2.7.0'
__build__ = 0x020700
__author__ = 'Kenneth Reitz'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright 2015 Kenneth Reitz'

# Attempt to enable urllib3's SNI support, if possible
try:
    from .packages.urllib3.contrib import pyopenssl
    pyopenssl.inject_into_urllib3()
except ImportError:
    pass

from . import utils
from .models import Request, Response, PreparedRequest
from .api import request, get, head, post, patch, put, delete, options
from .sessions import session, Session
from .status_codes import codes
from .exceptions import (
    RequestException, Timeout, URLRequired,
    TooManyRedirects, HTTPError, ConnectionError
)

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
```
### 5 - requests/exceptions.py:

Start line: 1, End line: 27

```python
# -*- coding: utf-8 -*-

"""
requests.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of Requests' exceptions.

"""
from .packages.urllib3.exceptions import HTTPError as BaseHTTPError


class RequestException(IOError):
    """There was an ambiguous exception that occurred while handling your
    request."""

    def __init__(self, *args, **kwargs):
        """
        Initialize RequestException with `request` and `response` objects.
        """
        response = kwargs.pop('response', None)
        self.response = response
        self.request = kwargs.pop('request', None)
        if (response is not None and not self.request and
                hasattr(response, 'request')):
            self.request = self.response.request
        super(RequestException, self).__init__(*args, **kwargs)
```
### 6 - requests/packages/urllib3/exceptions.py:

Start line: 81, End line: 170

```python
class HostChangedError(RequestError):
    "Raised when an existing pool gets a request for a foreign host."

    def __init__(self, pool, url, retries=3):
        message = "Tried to open a foreign host with url: %s" % url
        RequestError.__init__(self, pool, url, message)
        self.retries = retries


class TimeoutStateError(HTTPError):
    """ Raised when passing an invalid state to a timeout """
    pass


class TimeoutError(HTTPError):
    """ Raised when a socket timeout error occurs.

    Catching this error will catch both :exc:`ReadTimeoutErrors
    <ReadTimeoutError>` and :exc:`ConnectTimeoutErrors <ConnectTimeoutError>`.
    """
    pass


class ReadTimeoutError(TimeoutError, RequestError):
    "Raised when a socket timeout occurs while receiving data from a server"
    pass


# This timeout error does not have a URL attached and needs to inherit from the
# base HTTPError
class ConnectTimeoutError(TimeoutError):
    "Raised when a socket timeout occurs while connecting to a server"
    pass


class EmptyPoolError(PoolError):
    "Raised when a pool runs out of connections and no more are allowed."
    pass


class ClosedPoolError(PoolError):
    "Raised when a request enters a pool after the pool has been closed."
    pass


class LocationValueError(ValueError, HTTPError):
    "Raised when there is something wrong with a given URL input."
    pass


class LocationParseError(LocationValueError):
    "Raised when get_host or similar fails to parse the URL input."

    def __init__(self, location):
        message = "Failed to parse: %s" % location
        HTTPError.__init__(self, message)

        self.location = location


class ResponseError(HTTPError):
    "Used as a container for an error reason supplied in a MaxRetryError."
    GENERIC_ERROR = 'too many error responses'
    SPECIFIC_ERROR = 'too many {status_code} error responses'


class SecurityWarning(HTTPWarning):
    "Warned when perfoming security reducing actions"
    pass


class InsecureRequestWarning(SecurityWarning):
    "Warned when making an unverified HTTPS request."
    pass


class SystemTimeWarning(SecurityWarning):
    "Warned when system time is suspected to be wrong"
    pass


class InsecurePlatformWarning(SecurityWarning):
    "Warned when certain SSL configuration is not available on a platform."
    pass


class ResponseNotChunked(ProtocolError, ValueError):
    "Response needs to be chunked in order to read it as chunks."
    pass
```
### 7 - requests/packages/urllib3/packages/__init__.py:

Start line: 1, End line: 5

```python

```
### 8 - requests/packages/urllib3/connectionpool.py:

Start line: 1, End line: 50

```python
import errno
import logging
import sys
import warnings

from socket import error as SocketError, timeout as SocketTimeout
import socket

try:  # Python 3
    from queue import LifoQueue, Empty, Full
except ImportError:
    from Queue import LifoQueue, Empty, Full
    import Queue as _  # Platform-specific: Windows


from .exceptions import (
    ClosedPoolError,
    ProtocolError,
    EmptyPoolError,
    HostChangedError,
    LocationValueError,
    MaxRetryError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
    InsecureRequestWarning,
)
from .packages.ssl_match_hostname import CertificateError
from .packages import six
from .connection import (
    port_by_scheme,
    DummyConnection,
    HTTPConnection, HTTPSConnection, VerifiedHTTPSConnection,
    HTTPException, BaseSSLError, ConnectionError
)
from .request import RequestMethods
from .response import HTTPResponse

from .util.connection import is_connection_dropped
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import get_host


xrange = six.moves.xrange

log = logging.getLogger(__name__)

_Default = object()
```
### 9 - requests/packages/urllib3/contrib/pyopenssl.py:

Start line: 169, End line: 190

```python
class WrappedSocket(object):

    def recv(self, *args, **kwargs):
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return b''
            else:
                raise
        except OpenSSL.SSL.ZeroReturnError as e:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b''
            else:
                raise
        except OpenSSL.SSL.WantReadError:
            rd, wd, ed = select.select(
                [self.socket], [], [], self.socket.gettimeout())
            if not rd:
                raise timeout('The read operation timed out')
            else:
                return self.recv(*args, **kwargs)
        else:
            return data
```
### 10 - requests/packages/urllib3/connectionpool.py:

Start line: 300, End line: 315

```python
class HTTPConnectionPool(ConnectionPool, RequestMethods):

    def _raise_timeout(self, err, url, timeout_value):
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""

        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)

        # See the above comment about EAGAIN in Python 3. In Python 2 we have
        # to specifically catch it and throw the timeout error
        if hasattr(err, 'errno') and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)

        # Catch possible read timeouts thrown as SSL errors. If not the
        # case, rethrow the original. We need to do this because of:
        # http://bugs.python.org/issue10272
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):  # Python 2.6
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)
```
### 42 - requests/adapters.py:

Start line: 233, End line: 262

```python
class HTTPAdapter(BaseAdapter):

    def get_connection(self, url, proxies=None):
        """Returns a urllib3 connection for the given URL. This should not be
        called from user code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param url: The URL to connect to.
        :param proxies: (optional) A Requests-style dictionary of proxies used on this request.
        """
        proxies = proxies or {}
        proxy = proxies.get(urlparse(url.lower()).scheme)

        if proxy:
            proxy = prepend_scheme_if_needed(proxy, 'http')
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_url(url)
        else:
            # Only scheme should be lower case
            parsed = urlparse(url)
            url = parsed.geturl()
            conn = self.poolmanager.connection_from_url(url)

        return conn

    def close(self):
        """Disposes of any internal state.

        Currently, this just closes the PoolManager, which closes pooled
        connections.
        """
        self.poolmanager.clear()
```
### 43 - requests/adapters.py:

Start line: 137, End line: 158

```python
class HTTPAdapter(BaseAdapter):

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Return urllib3 ProxyManager for the given proxy.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The proxy to return a urllib3 ProxyManager for.
        :param proxy_kwargs: Extra keyword arguments used to configure the Proxy Manager.
        :returns: ProxyManager
        """
        if not proxy in self.proxy_manager:
            proxy_headers = self.proxy_headers(proxy)
            self.proxy_manager[proxy] = proxy_from_url(
                proxy,
                proxy_headers=proxy_headers,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs)

        return self.proxy_manager[proxy]
```
### 47 - requests/adapters.py:

Start line: 54, End line: 115

```python
class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.

    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface. This class will
    usually be created by the :class:`Session <Session>` class under the
    covers.

    :param pool_connections: The number of urllib3 connection pools to cache.
    :param pool_maxsize: The maximum number of connections to save in the pool.
    :param int max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups, socket
        connections and connection timeouts, never to requests where data has
        made it to the server. By default, Requests does not retry failed
        connections. If you need granular control over the conditions under
        which we retry a request, import urllib3's ``Retry`` class and pass
        that instead.
    :param pool_block: Whether the connection pool should block for connections.

    Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> a = requests.adapters.HTTPAdapter(max_retries=3)
      >>> s.mount('http://', a)
    """
    __attrs__ = ['max_retries', 'config', '_pool_connections', '_pool_maxsize',
                 '_pool_block']

    def __init__(self, pool_connections=DEFAULT_POOLSIZE,
                 pool_maxsize=DEFAULT_POOLSIZE, max_retries=DEFAULT_RETRIES,
                 pool_block=DEFAULT_POOLBLOCK):
        if max_retries == DEFAULT_RETRIES:
            self.max_retries = Retry(0, read=False)
        else:
            self.max_retries = Retry.from_int(max_retries)
        self.config = {}
        self.proxy_manager = {}

        super(HTTPAdapter, self).__init__()

        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._pool_block = pool_block

        self.init_poolmanager(pool_connections, pool_maxsize, block=pool_block)

    def __getstate__(self):
        return dict((attr, getattr(self, attr, None)) for attr in
                    self.__attrs__)

    def __setstate__(self, state):
        # Can't handle by adding 'proxy_manager' to self.__attrs__ because
        # because self.poolmanager uses a lambda function, which isn't pickleable.
        self.proxy_manager = {}
        self.config = {}

        for attr, value in state.items():
            setattr(self, attr, value)

        self.init_poolmanager(self._pool_connections, self._pool_maxsize,
                              block=self._pool_block)
```
### 63 - requests/adapters.py:

Start line: 1, End line: 51

```python
# -*- coding: utf-8 -*-

"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""

import socket

from .models import Response
from .packages.urllib3.poolmanager import PoolManager, proxy_from_url
from .packages.urllib3.response import HTTPResponse
from .packages.urllib3.util import Timeout as TimeoutSauce
from .packages.urllib3.util.retry import Retry
from .compat import urlparse, basestring
from .utils import (DEFAULT_CA_BUNDLE_PATH, get_encoding_from_headers,
                    prepend_scheme_if_needed, get_auth_from_url, urldefragauth)
from .structures import CaseInsensitiveDict
from .packages.urllib3.exceptions import ConnectTimeoutError
from .packages.urllib3.exceptions import HTTPError as _HTTPError
from .packages.urllib3.exceptions import MaxRetryError
from .packages.urllib3.exceptions import ProxyError as _ProxyError
from .packages.urllib3.exceptions import ProtocolError
from .packages.urllib3.exceptions import ReadTimeoutError
from .packages.urllib3.exceptions import SSLError as _SSLError
from .packages.urllib3.exceptions import ResponseError
from .cookies import extract_cookies_to_jar
from .exceptions import (ConnectionError, ConnectTimeout, ReadTimeout, SSLError,
                         ProxyError, RetryError)
from .auth import _basic_auth_str

DEFAULT_POOLBLOCK = False
DEFAULT_POOLSIZE = 10
DEFAULT_RETRIES = 0
DEFAULT_POOL_TIMEOUT = None


class BaseAdapter(object):
    """The Base Transport Adapter"""

    def __init__(self):
        super(BaseAdapter, self).__init__()

    def send(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
```
### 78 - requests/adapters.py:

Start line: 197, End line: 231

```python
class HTTPAdapter(BaseAdapter):

    def build_response(self, req, resp):
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        """
        response = Response()

        # Fallback to None if there's no status_code, for whatever reason.
        response.status_code = getattr(resp, 'status', None)

        # Make headers case-insensitive.
        response.headers = CaseInsensitiveDict(getattr(resp, 'headers', {}))

        # Set encoding.
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = response.raw.reason

        if isinstance(req.url, bytes):
            response.url = req.url.decode('utf-8')
        else:
            response.url = req.url

        # Add new cookies from the server.
        extract_cookies_to_jar(response.cookies, req, resp)

        # Give the Response some context.
        response.request = req
        response.connection = self

        return response
```
### 92 - requests/adapters.py:

Start line: 302, End line: 322

```python
class HTTPAdapter(BaseAdapter):

    def proxy_headers(self, proxy):
        """Returns a dictionary of the headers to add to any request sent
        through a proxy. This works with urllib3 magic to ensure that they are
        correctly sent to the proxy, rather than in a tunnelled request if
        CONNECT is being used.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxies: The url of the proxy being used for this request.
        :param kwargs: Optional additional keyword arguments.
        """
        headers = {}
        username, password = get_auth_from_url(proxy)

        if username and password:
            headers['Proxy-Authorization'] = _basic_auth_str(username,
                                                             password)

        return headers
```
### 121 - requests/adapters.py:

Start line: 264, End line: 286

```python
class HTTPAdapter(BaseAdapter):

    def request_url(self, request, proxies):
        """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes to proxy URLs.
        """
        proxies = proxies or {}
        scheme = urlparse(request.url).scheme
        proxy = proxies.get(scheme)

        if proxy and scheme != 'https':
            url = urldefragauth(request.url)
        else:
            url = request.path_url

        return url
```
### 154 - requests/adapters.py:

Start line: 117, End line: 135

```python
class HTTPAdapter(BaseAdapter):

    def init_poolmanager(self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs):
        """Initializes a urllib3 PoolManager.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param connections: The number of urllib3 connection pools to cache.
        :param maxsize: The maximum number of connections to save in the pool.
        :param block: Block when no free connections are available.
        :param pool_kwargs: Extra keyword arguments used to initialize the Pool Manager.
        """
        # save these values for pickling
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block

        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize,
                                       block=block, strict=True, **pool_kwargs)
```
### 176 - requests/adapters.py:

Start line: 324, End line: 436

```python
class HTTPAdapter(BaseAdapter):

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a (`connect timeout, read
            timeout <user/advanced.html#timeouts>`_) tuple.
        :type timeout: float or tuple
        :param verify: (optional) Whether to verify SSL certificates.
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """

        conn = self.get_connection(request.url, proxies)

        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(request)

        chunked = not (request.body is None or 'Content-Length' in request.headers)

        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError as e:
                # this may raise a string formatting error.
                err = ("Invalid timeout {0}. Pass a (connect, read) "
                       "timeout tuple, or a single float to set "
                       "both timeouts to the same value".format(timeout))
                raise ValueError(err)
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)

        try:
            if not chunked:
                resp = conn.urlopen(
                    method=request.method,
                    url=url,
                    body=request.body,
                    headers=request.headers,
                    redirect=False,
                    assert_same_host=False,
                    preload_content=False,
                    decode_content=False,
                    retries=self.max_retries,
                    timeout=timeout
                )

            # Send the request.
            else:
                if hasattr(conn, 'proxy_pool'):
                    conn = conn.proxy_pool

                low_conn = conn._get_conn(timeout=DEFAULT_POOL_TIMEOUT)

                try:
                    low_conn.putrequest(request.method,
                                        url,
                                        skip_accept_encoding=True)

                    for header, value in request.headers.items():
                        low_conn.putheader(header, value)

                    low_conn.endheaders()

                    for i in request.body:
                        low_conn.send(hex(len(i))[2:].encode('utf-8'))
                        low_conn.send(b'\r\n')
                        low_conn.send(i)
                        low_conn.send(b'\r\n')
                    low_conn.send(b'0\r\n\r\n')

                    r = low_conn.getresponse()
                    resp = HTTPResponse.from_httplib(
                        r,
                        pool=conn,
                        connection=low_conn,
                        preload_content=False,
                        decode_content=False
                    )
                except:
                    # If we hit any problems here, clean up the connection.
                    # Then, reraise so that we can handle the actual exception.
                    low_conn.close()
                    raise

        except (ProtocolError, socket.error) as err:
            raise ConnectionError(err, request=request)

        except MaxRetryError as e:
            if isinstance(e.reason, ConnectTimeoutError):
                raise ConnectTimeout(e, request=request)

            if isinstance(e.reason, ResponseError):
                raise RetryError(e, request=request)

            raise ConnectionError(e, request=request)

        except _ProxyError as e:
            raise ProxyError(e)

        except (_SSLError, _HTTPError) as e:
            if isinstance(e, _SSLError):
                raise SSLError(e, request=request)
            elif isinstance(e, ReadTimeoutError):
                raise ReadTimeout(e, request=request)
            else:
                raise

        return self.build_response(request, resp)
```
### 206 - requests/adapters.py:

Start line: 288, End line: 300

```python
class HTTPAdapter(BaseAdapter):

    def add_headers(self, request, **kwargs):
        """Add any headers needed by the connection. As of v2.0 this does
        nothing by default, but is left for overriding by users that subclass
        the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` to add headers to.
        :param kwargs: The keyword arguments from the call to send().
        """
        pass
```
