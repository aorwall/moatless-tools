# psf__requests-1689

| **psf/requests** | `e91ee0e2461cc9b6822e7c3cc422038604ace08d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3554 |
| **Any found context length** | 1042 |
| **Avg pos** | 14.0 |
| **Min pos** | 4 |
| **Max pos** | 10 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -407,7 +407,7 @@ def prepare_body(self, data, files):
                 raise NotImplementedError('Streamed bodies and files are mutually exclusive.')
 
             if length is not None:
-                self.headers['Content-Length'] = str(length)
+                self.headers['Content-Length'] = builtin_str(length)
             else:
                 self.headers['Transfer-Encoding'] = 'chunked'
         else:
@@ -433,12 +433,12 @@ def prepare_body(self, data, files):
     def prepare_content_length(self, body):
         if hasattr(body, 'seek') and hasattr(body, 'tell'):
             body.seek(0, 2)
-            self.headers['Content-Length'] = str(body.tell())
+            self.headers['Content-Length'] = builtin_str(body.tell())
             body.seek(0, 0)
         elif body is not None:
             l = super_len(body)
             if l:
-                self.headers['Content-Length'] = str(l)
+                self.headers['Content-Length'] = builtin_str(l)
         elif self.method not in ('GET', 'HEAD'):
             self.headers['Content-Length'] = '0'
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/models.py | 410 | 410 | 10 | 2 | 3554
| requests/models.py | 436 | 441 | 4 | 2 | 1042


## Problem Statement

```
Problem POST'ing png file because of UnicodeError
Here is the code I'm using:

\`\`\` python
files = {'file': (upload_handle.upload_token.key, open("test.png", "rb"))}
resp = requests.post(url, files=files)
\`\`\`

This raises the error:

\`\`\`
UnicodeDecodeError: 'utf8' codec can't decode byte 0x89 in position 140: invalid start byte
\`\`\`

This problem is caused by the fact that the content-length header is actually a unicode object. When the actual body of the request is being constructed, python attempts to coerce the entire request into unicode resulting in the decode error.

After tracing it, the cause is the following lines:

requests/models.py: 

\`\`\`
self.prepare_content_length(body)
# -------
l = super_len(body)
self.headers['Content-Length'] = str(l)
\`\`\`

where `str = unicode` is declared in compat.py


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 requests/packages/urllib3/response.py | 1 | 50| 237 | 237 | 2113 | 
| 2 | **2 requests/models.py** | 646 | 677| 198 | 435 | 7085 | 
| 3 | 3 requests/utils.py | 518 | 544| 336 | 771 | 11191 | 
| **-> 4 <-** | **3 requests/models.py** | 433 | 465| 271 | 1042 | 11191 | 
| 5 | **3 requests/models.py** | 622 | 644| 152 | 1194 | 11191 | 
| 6 | **3 requests/models.py** | 89 | 146| 447 | 1641 | 11191 | 
| 7 | 4 requests/packages/urllib3/contrib/pyopenssl.py | 170 | 265| 740 | 2381 | 13764 | 
| 8 | 4 requests/packages/urllib3/contrib/pyopenssl.py | 102 | 168| 585 | 2966 | 13764 | 
| 9 | 4 requests/utils.py | 334 | 370| 191 | 3157 | 13764 | 
| **-> 10 <-** | **4 requests/models.py** | 372 | 431| 397 | 3554 | 13764 | 
| 11 | 4 requests/packages/urllib3/response.py | 264 | 302| 257 | 3811 | 13764 | 
| 12 | 5 requests/__init__.py | 1 | 78| 293 | 4104 | 14260 | 
| 13 | 6 requests/packages/urllib3/request.py | 90 | 143| 508 | 4612 | 15466 | 
| 14 | 7 requests/packages/urllib3/filepost.py | 1 | 44| 183 | 4795 | 16047 | 
| 15 | 8 requests/packages/urllib3/__init__.py | 1 | 38| 180 | 4975 | 16435 | 
| 16 | 9 requests/status_codes.py | 1 | 89| 899 | 5874 | 17334 | 
| 17 | 10 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 6588 | 18354 | 
| 18 | **10 requests/models.py** | 40 | 87| 294 | 6882 | 18354 | 
| 19 | **10 requests/models.py** | 312 | 370| 429 | 7311 | 18354 | 
| 20 | 11 requests/compat.py | 1 | 116| 805 | 8116 | 19159 | 
| 21 | 12 requests/packages/urllib3/connectionpool.py | 1 | 80| 373 | 8489 | 24845 | 
| 22 | **12 requests/models.py** | 1 | 37| 261 | 8750 | 24845 | 
| 23 | 13 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 8908 | 27687 | 
| 24 | 14 requests/packages/charade/compat.py | 21 | 35| 69 | 8977 | 27953 | 
| 25 | 15 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 8977 | 27968 | 
| 26 | 15 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 68| 544 | 9521 | 27968 | 
| 27 | 16 requests/packages/urllib3/util.py | 578 | 627| 393 | 9914 | 32605 | 
| 28 | 17 requests/packages/urllib3/exceptions.py | 1 | 122| 701 | 10615 | 33359 | 
| 29 | **17 requests/models.py** | 596 | 620| 167 | 10782 | 33359 | 
| 30 | 17 requests/packages/urllib3/request.py | 1 | 57| 377 | 11159 | 33359 | 
| 31 | 17 requests/packages/urllib3/util.py | 1 | 48| 269 | 11428 | 33359 | 
| 32 | 17 requests/packages/urllib3/util.py | 411 | 468| 195 | 11623 | 33359 | 
| 33 | 17 requests/packages/urllib3/filepost.py | 66 | 102| 233 | 11856 | 33359 | 
| 34 | 18 requests/auth.py | 1 | 55| 310 | 12166 | 34730 | 
| 35 | **18 requests/models.py** | 559 | 594| 244 | 12410 | 34730 | 
| 36 | 18 requests/packages/urllib3/contrib/pyopenssl.py | 290 | 313| 141 | 12551 | 34730 | 
| 37 | 19 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 12595 | 55598 | 
| 38 | 19 requests/packages/urllib3/response.py | 132 | 206| 616 | 13211 | 55598 | 
| 39 | **19 requests/models.py** | 679 | 693| 165 | 13376 | 55598 | 
| 40 | **19 requests/models.py** | 278 | 310| 266 | 13642 | 55598 | 
| 41 | 19 requests/utils.py | 273 | 285| 137 | 13779 | 55598 | 
| 42 | 19 requests/packages/urllib3/response.py | 53 | 130| 563 | 14342 | 55598 | 
| 43 | 19 requests/packages/urllib3/contrib/pyopenssl.py | 316 | 345| 244 | 14586 | 55598 | 
| 44 | **19 requests/models.py** | 695 | 734| 248 | 14834 | 55598 | 
| 45 | 19 requests/packages/urllib3/contrib/pyopenssl.py | 71 | 99| 207 | 15041 | 55598 | 
| 46 | 19 requests/packages/urllib3/connectionpool.py | 534 | 620| 771 | 15812 | 55598 | 
| 47 | 20 requests/exceptions.py | 1 | 64| 286 | 16098 | 55884 | 
| 48 | 21 requests/packages/urllib3/fields.py | 27 | 52| 217 | 16315 | 57196 | 
| 49 | 21 requests/packages/urllib3/response.py | 208 | 228| 178 | 16493 | 57196 | 
| 50 | **21 requests/models.py** | 486 | 557| 494 | 16987 | 57196 | 
| 51 | 21 requests/packages/urllib3/filepost.py | 47 | 63| 116 | 17103 | 57196 | 
| 52 | 21 requests/packages/urllib3/fields.py | 142 | 159| 142 | 17245 | 57196 | 
| 53 | 22 setup.py | 1 | 58| 364 | 17609 | 57560 | 
| 54 | 22 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 17837 | 57560 | 
| 55 | 23 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 18021 | 58006 | 
| 56 | 23 requests/packages/urllib3/contrib/pyopenssl.py | 268 | 288| 121 | 18142 | 58006 | 
| 57 | 23 requests/packages/urllib3/fields.py | 161 | 178| 172 | 18314 | 58006 | 
| 58 | 23 requests/utils.py | 66 | 99| 250 | 18564 | 58006 | 
| 59 | 24 requests/sessions.py | 395 | 403| 112 | 18676 | 61980 | 
| 60 | 24 requests/auth.py | 58 | 143| 755 | 19431 | 61980 | 
| 61 | 24 requests/packages/urllib3/fields.py | 1 | 24| 114 | 19545 | 61980 | 
| 62 | 25 requests/packages/charade/big5freq.py | 43 | 926| 52 | 19597 | 108576 | 
| 63 | 25 requests/utils.py | 288 | 331| 237 | 19834 | 108576 | 
| 64 | 25 requests/auth.py | 145 | 181| 311 | 20145 | 108576 | 
| 65 | 25 requests/packages/urllib3/fields.py | 55 | 74| 129 | 20274 | 108576 | 
| 66 | 26 requests/packages/__init__.py | 1 | 4| 0 | 20274 | 108590 | 
| 67 | **26 requests/models.py** | 247 | 276| 208 | 20482 | 108590 | 
| 68 | **26 requests/models.py** | 467 | 483| 132 | 20614 | 108590 | 
| 69 | 27 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 20769 | 110578 | 
| 70 | 27 requests/packages/urllib3/request.py | 59 | 88| 283 | 21052 | 110578 | 
| 71 | 27 requests/utils.py | 157 | 185| 250 | 21302 | 110578 | 
| 72 | 27 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 21814 | 110578 | 
| 73 | 28 docs/conf.py | 137 | 244| 615 | 22429 | 112478 | 
| 74 | 29 requests/api.py | 47 | 99| 438 | 22867 | 113551 | 
| 75 | 29 requests/packages/urllib3/fields.py | 109 | 140| 232 | 23099 | 113551 | 
| 76 | 29 requests/utils.py | 188 | 219| 266 | 23365 | 113551 | 
| 77 | 29 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 24104 | 113551 | 
| 78 | 29 requests/packages/urllib3/connectionpool.py | 105 | 139| 285 | 24389 | 113551 | 
| 79 | **29 requests/models.py** | 173 | 244| 513 | 24902 | 113551 | 
| 80 | 29 requests/utils.py | 222 | 244| 280 | 25182 | 113551 | 
| 81 | 29 requests/utils.py | 373 | 404| 269 | 25451 | 113551 | 
| 82 | 30 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 26727 | 114827 | 
| 83 | 30 requests/packages/urllib3/fields.py | 76 | 107| 274 | 27001 | 114827 | 
| 84 | 30 requests/utils.py | 547 | 583| 215 | 27216 | 114827 | 
| 85 | 31 requests/adapters.py | 1 | 45| 299 | 27515 | 117558 | 
| 86 | 31 requests/packages/urllib3/poolmanager.py | 243 | 260| 162 | 27677 | 117558 | 
| 87 | 31 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 28013 | 117558 | 
| 88 | 31 requests/utils.py | 1 | 64| 337 | 28350 | 117558 | 
| 89 | 31 requests/packages/urllib3/util.py | 503 | 539| 256 | 28606 | 117558 | 
| 90 | 31 requests/packages/urllib3/util.py | 542 | 576| 242 | 28848 | 117558 | 
| 91 | 31 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 29378 | 117558 | 
| 92 | 31 requests/packages/urllib3/util.py | 471 | 500| 218 | 29596 | 117558 | 
| 93 | 32 requests/packages/urllib3/_collections.py | 67 | 95| 174 | 29770 | 118183 | 
| 94 | 32 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 30019 | 118183 | 
| 95 | 32 requests/packages/urllib3/response.py | 231 | 262| 241 | 30260 | 118183 | 
| 96 | 33 requests/cookies.py | 1 | 79| 509 | 30769 | 121451 | 
| 97 | 33 requests/packages/urllib3/util.py | 160 | 174| 137 | 30906 | 121451 | 
| 98 | 33 requests/packages/urllib3/connectionpool.py | 142 | 165| 183 | 31089 | 121451 | 
| 99 | 33 requests/packages/urllib3/connectionpool.py | 341 | 415| 710 | 31799 | 121451 | 
| 100 | 33 requests/utils.py | 439 | 467| 259 | 32058 | 121451 | 
| 101 | 33 requests/api.py | 1 | 44| 423 | 32481 | 121451 | 
| 102 | 34 requests/certs.py | 1 | 25| 120 | 32601 | 121571 | 
| 103 | 34 requests/packages/urllib3/util.py | 191 | 213| 196 | 32797 | 121571 | 
| 104 | 34 requests/adapters.py | 151 | 185| 273 | 33070 | 121571 | 
| 105 | 34 requests/utils.py | 102 | 129| 211 | 33281 | 121571 | 
| 106 | 35 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 60 | 99| 350 | 33631 | 122430 | 
| 107 | 36 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 33831 | 124587 | 
| 108 | 36 requests/packages/urllib3/util.py | 126 | 158| 269 | 34100 | 124587 | 
| 109 | 36 requests/api.py | 102 | 121| 171 | 34271 | 124587 | 
| 110 | 36 requests/sessions.py | 269 | 363| 759 | 35030 | 124587 | 
| 111 | 36 requests/utils.py | 470 | 515| 271 | 35301 | 124587 | 
| 112 | 36 requests/cookies.py | 82 | 100| 134 | 35435 | 124587 | 
| 113 | 36 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 35728 | 124587 | 
| 114 | 36 requests/packages/urllib3/util.py | 215 | 231| 124 | 35852 | 124587 | 
| 115 | 36 requests/packages/urllib3/util.py | 331 | 408| 504 | 36356 | 124587 | 
| 116 | 36 requests/packages/urllib3/util.py | 265 | 295| 233 | 36589 | 124587 | 
| 117 | 36 requests/packages/urllib3/connectionpool.py | 83 | 103| 147 | 36736 | 124587 | 
| 118 | 36 requests/adapters.py | 285 | 375| 587 | 37323 | 124587 | 
| 119 | 36 docs/conf.py | 1 | 136| 1059 | 38382 | 124587 | 
| 120 | 36 requests/adapters.py | 48 | 95| 389 | 38771 | 124587 | 
| 121 | 36 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 57| 509 | 39280 | 124587 | 
| 122 | 36 requests/sessions.py | 405 | 413| 112 | 39392 | 124587 | 
| 123 | 36 requests/packages/urllib3/util.py | 51 | 123| 707 | 40099 | 124587 | 
| 124 | 36 requests/packages/urllib3/connectionpool.py | 623 | 659| 362 | 40461 | 124587 | 
| 125 | 36 requests/sessions.py | 434 | 494| 488 | 40949 | 124587 | 
| 126 | 37 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 41002 | 151361 | 
| 127 | 37 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 41147 | 151361 | 
| 128 | 38 requests/packages/charade/jisfreq.py | 44 | 570| 53 | 41200 | 179334 | 
| 129 | 38 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 41334 | 179334 | 
| 130 | 38 requests/packages/urllib3/_collections.py | 52 | 65| 152 | 41486 | 179334 | 
| 131 | 39 requests/hooks.py | 1 | 46| 188 | 41674 | 179523 | 
| 132 | 39 requests/packages/urllib3/connectionpool.py | 450 | 532| 727 | 42401 | 179523 | 
| 133 | 39 requests/packages/urllib3/connectionpool.py | 661 | 687| 216 | 42617 | 179523 | 
| 134 | 39 requests/packages/urllib3/util.py | 233 | 262| 283 | 42900 | 179523 | 
| 135 | 39 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 43475 | 179523 | 
| 136 | 39 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 44144 | 179523 | 
| 137 | 40 requests/structures.py | 1 | 34| 167 | 44311 | 180363 | 
| 138 | 40 requests/packages/urllib3/connectionpool.py | 167 | 249| 695 | 45006 | 180363 | 
| 139 | 40 requests/sessions.py | 415 | 432| 185 | 45191 | 180363 | 
| 140 | 40 requests/packages/urllib3/connectionpool.py | 300 | 339| 298 | 45489 | 180363 | 
| 141 | 40 requests/adapters.py | 97 | 112| 175 | 45664 | 180363 | 
| 142 | 40 requests/sessions.py | 229 | 267| 296 | 45960 | 180363 | 
| 143 | **40 requests/models.py** | 149 | 170| 161 | 46121 | 180363 | 
| 144 | 40 requests/packages/urllib3/connectionpool.py | 689 | 718| 233 | 46354 | 180363 | 
| 145 | 40 requests/packages/urllib3/connectionpool.py | 417 | 448| 203 | 46557 | 180363 | 
| 146 | 40 requests/packages/urllib3/util.py | 176 | 189| 130 | 46687 | 180363 | 


### Hint

```
Yep, that's a crass little bug. Thanks so much for pointing it out! I'll go through the headers we set and make sure we don't do this anywhere else. =)

```

## Patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -407,7 +407,7 @@ def prepare_body(self, data, files):
                 raise NotImplementedError('Streamed bodies and files are mutually exclusive.')
 
             if length is not None:
-                self.headers['Content-Length'] = str(length)
+                self.headers['Content-Length'] = builtin_str(length)
             else:
                 self.headers['Transfer-Encoding'] = 'chunked'
         else:
@@ -433,12 +433,12 @@ def prepare_body(self, data, files):
     def prepare_content_length(self, body):
         if hasattr(body, 'seek') and hasattr(body, 'tell'):
             body.seek(0, 2)
-            self.headers['Content-Length'] = str(body.tell())
+            self.headers['Content-Length'] = builtin_str(body.tell())
             body.seek(0, 0)
         elif body is not None:
             l = super_len(body)
             if l:
-                self.headers['Content-Length'] = str(l)
+                self.headers['Content-Length'] = builtin_str(l)
         elif self.method not in ('GET', 'HEAD'):
             self.headers['Content-Length'] = '0'
 

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -684,6 +684,14 @@ def test_can_send_nonstring_objects_with_files(self):
 
         self.assertTrue('multipart/form-data' in p.headers['Content-Type'])
 
+    def test_autoset_header_values_are_native(self):
+        data = 'this is a string'
+        length = '16'
+        req = requests.Request('POST', httpbin('post'), data=data)
+        p = req.prepare()
+
+        self.assertEqual(p.headers['Content-Length'], length)
+
 
 class TestContentEncodingDetection(unittest.TestCase):
 

```


## Code snippets

### 1 - requests/packages/urllib3/response.py:

Start line: 1, End line: 50

```python
# urllib3/response.py


import logging
import zlib
import io

from .exceptions import DecodeError
from .packages.six import string_types as basestring, binary_type
from .util import is_fp_closed


log = logging.getLogger(__name__)


class DeflateDecoder(object):

    def __init__(self):
        self._first_try = True
        self._data = binary_type()
        self._obj = zlib.decompressobj()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
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


def _get_decoder(mode):
    if mode == 'gzip':
        return zlib.decompressobj(16 + zlib.MAX_WBITS)

    return DeflateDecoder()
```
### 2 - requests/models.py:

Start line: 646, End line: 677

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
### 3 - requests/utils.py:

Start line: 518, End line: 544

```python
def guess_json_utf(data):
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM32_BE):
        return 'utf-32'     # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return 'utf-8-sig'  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return 'utf-16'     # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return 'utf-8'
    if nullcount == 2:
        if sample[::2] == _null2:   # 1st and 3rd are null
            return 'utf-16-be'
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return 'utf-16-le'
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return 'utf-32-be'
        if sample[1:] == _null3:
            return 'utf-32-le'
        # Did not detect a valid UTF-32 ascii-range character
    return None
```
### 4 - requests/models.py:

Start line: 433, End line: 465

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
### 5 - requests/models.py:

Start line: 622, End line: 644

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
### 6 - requests/models.py:

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
### 7 - requests/packages/urllib3/contrib/pyopenssl.py:

Start line: 170, End line: 265

```python
class fileobject(_fileobject):

    def readline(self, size=-1):
        buf = self._rbuf
        buf.seek(0, 2)  # seek end
        if buf.tell() > 0:
            # check if we already have it in our buffer
            buf.seek(0)
            bline = buf.readline(size)
            if bline.endswith('\n') or len(bline) == size:
                self._rbuf = StringIO()
                self._rbuf.write(buf.read())
                return bline
            del bline
        if size < 0:
            # Read until \n or EOF, whichever comes first
            if self._rbufsize <= 1:
                # Speed up unbuffered case
                buf.seek(0)
                buffers = [buf.read()]
                self._rbuf = StringIO()  # reset _rbuf.  we consume it via buf.
                data = None
                recv = self._sock.recv
                while True:
                    try:
                        while data != "\n":
                            data = recv(1)
                            if not data:
                                break
                            buffers.append(data)
                    except OpenSSL.SSL.WantReadError:
                        continue
                    break
                return "".join(buffers)

            buf.seek(0, 2)  # seek end
            self._rbuf = StringIO()  # reset _rbuf.  we consume it via buf.
            while True:
                try:
                    data = self._sock.recv(self._rbufsize)
                except OpenSSL.SSL.WantReadError:
                    continue
                if not data:
                    break
                nl = data.find('\n')
                if nl >= 0:
                    nl += 1
                    buf.write(data[:nl])
                    self._rbuf.write(data[nl:])
                    del data
                    break
                buf.write(data)
            return buf.getvalue()
        else:
            # Read until size bytes or \n or EOF seen, whichever comes first
            buf.seek(0, 2)  # seek end
            buf_len = buf.tell()
            if buf_len >= size:
                buf.seek(0)
                rv = buf.read(size)
                self._rbuf = StringIO()
                self._rbuf.write(buf.read())
                return rv
            self._rbuf = StringIO()  # reset _rbuf.  we consume it via buf.
            while True:
                try:
                    data = self._sock.recv(self._rbufsize)
                except OpenSSL.SSL.WantReadError:
                        continue
                if not data:
                    break
                left = size - buf_len
                # did we just receive a newline?
                nl = data.find('\n', 0, left)
                if nl >= 0:
                    nl += 1
                    # save the excess data to _rbuf
                    self._rbuf.write(data[nl:])
                    if buf_len:
                        buf.write(data[:nl])
                        break
                    else:
                        # Shortcut.  Avoid data copy through buf when returning
                        # a substring of our first recv().
                        return data[:nl]
                n = len(data)
                if n == size and not buf_len:
                    # Shortcut.  Avoid data copy through buf when
                    # returning exactly all of our first recv().
                    return data
                if n >= left:
                    buf.write(data[:left])
                    self._rbuf.write(data[left:])
                    break
                buf.write(data)
                buf_len += n
                #assert buf_len == buf.tell()
            return buf.getvalue()
```
### 8 - requests/packages/urllib3/contrib/pyopenssl.py:

Start line: 102, End line: 168

```python
class fileobject(_fileobject):

    def read(self, size=-1):
        # Use max, disallow tiny reads in a loop as they are very inefficient.
        # We never leave read() with any leftover data from a new recv() call
        # in our internal buffer.
        rbufsize = max(self._rbufsize, self.default_bufsize)
        # Our use of StringIO rather than lists of string objects returned by
        # recv() minimizes memory usage and fragmentation that occurs when
        # rbufsize is large compared to the typical return value of recv().
        buf = self._rbuf
        buf.seek(0, 2)  # seek end
        if size < 0:
            # Read until EOF
            self._rbuf = StringIO()  # reset _rbuf.  we consume it via buf.
            while True:
                try:
                    data = self._sock.recv(rbufsize)
                except OpenSSL.SSL.WantReadError:
                    continue
                if not data:
                    break
                buf.write(data)
            return buf.getvalue()
        else:
            # Read until size bytes or EOF seen, whichever comes first
            buf_len = buf.tell()
            if buf_len >= size:
                # Already have size bytes in our buffer?  Extract and return.
                buf.seek(0)
                rv = buf.read(size)
                self._rbuf = StringIO()
                self._rbuf.write(buf.read())
                return rv

            self._rbuf = StringIO()  # reset _rbuf.  we consume it via buf.
            while True:
                left = size - buf_len
                # recv() will malloc the amount of memory given as its
                # parameter even though it often returns much less data
                # than that.  The returned data string is short lived
                # as we copy it into a StringIO and free it.  This avoids
                # fragmentation issues on many platforms.
                try:
                    data = self._sock.recv(left)
                except OpenSSL.SSL.WantReadError:
                    continue
                if not data:
                    break
                n = len(data)
                if n == size and not buf_len:
                    # Shortcut.  Avoid buffer data copies when:
                    # - We have no data in our buffer.
                    # AND
                    # - Our call to recv returned exactly the
                    #   number of bytes we were asked to read.
                    return data
                if n == left:
                    buf.write(data)
                    del data  # explicit free
                    break
                assert n <= left, "recv(%d) returned %d bytes" % (left, n)
                buf.write(data)
                buf_len += n
                del data  # explicit free
                #assert buf_len == buf.tell()
            return buf.getvalue()
```
### 9 - requests/utils.py:

Start line: 334, End line: 370

```python
def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type

    2. every encodings from ``<meta ... charset=XXX>``

    3. fall back and replace all unicode characters

    """

    tried_encodings = []

    # Try charset from content-type
    encoding = get_encoding_from_headers(r.headers)

    if encoding:
        try:
            return str(r.content, encoding)
        except UnicodeError:
            tried_encodings.append(encoding)

    # Fall back:
    try:
        return str(r.content, encoding, errors='replace')
    except TypeError:
        return r.content


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    + "0123456789-._~")
```
### 10 - requests/models.py:

Start line: 372, End line: 431

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
### 18 - requests/models.py:

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
### 19 - requests/models.py:

Start line: 312, End line: 370

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
### 22 - requests/models.py:

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
### 29 - requests/models.py:

Start line: 596, End line: 620

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
### 35 - requests/models.py:

Start line: 559, End line: 594

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
### 39 - requests/models.py:

Start line: 679, End line: 693

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
        return json.loads(self.text, **kwargs)
```
### 40 - requests/models.py:

Start line: 278, End line: 310

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
### 44 - requests/models.py:

Start line: 695, End line: 734

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
### 50 - requests/models.py:

Start line: 486, End line: 557

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
### 67 - requests/models.py:

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
### 68 - requests/models.py:

Start line: 467, End line: 483

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
### 79 - requests/models.py:

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
### 143 - requests/models.py:

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
