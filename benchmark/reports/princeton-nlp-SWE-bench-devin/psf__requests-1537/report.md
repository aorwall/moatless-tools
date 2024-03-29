# psf__requests-1537

| **psf/requests** | `d8268fb7b44da7b8aa225eb1ca6fbdb4f9dc2457` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 371 |
| **Any found context length** | 371 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -106,6 +106,10 @@ def _encode_files(files, data):
                 val = [val]
             for v in val:
                 if v is not None:
+                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
+                    if not isinstance(v, bytes):
+                        v = str(v)
+
                     new_fields.append(
                         (field.decode('utf-8') if isinstance(field, bytes) else field,
                          v.encode('utf-8') if isinstance(v, str) else v))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/models.py | 109 | 109 | 1 | 1 | 371


## Problem Statement

```
multipart/form-data and datetime data
I raise an bug that you already fix in the past on this issue : https://github.com/kennethreitz/requests/issues/661 or https://github.com/kennethreitz/requests/issues/737

I tried the same methodology with that code :

\`\`\`
import requets

requests.post("http://httpbin.org/post", data={'a': 0})
requests.post("http://httpbin.org/post", data={'a': 0.0})
requests.post("http://httpbin.org/post", data={'a': 0}, files={'b': 'foo'})
requests.post("http://httpbin.org/post", data={'a': 0.0}, files={'b': 'foo'})
\`\`\`

With the 1.2.0 version, no error is raised.

With 1.2.3 version, I have that traceback :

\`\`\`
Traceback (most recent call last):
  File "test.py", line 8, in <module>
    requests.post("http://httpbin.org/post", data={'a': 0.0}, files={'b': 'foo'})
  File ".../dev/lib/python2.7/site-packages/requests/api.py", line 88, in post
    return request('post', url, data=data, **kwargs)
  File ".../dev/lib/python2.7/site-packages/requests/api.py", line 44, in request
    return session.request(method=method, url=url, **kwargs)
  File ".../dev/lib/python2.7/site-packages/requests/sessions.py", line 324, in request
    prep = req.prepare()
  File ".../dev/lib/python2.7/site-packages/requests/models.py", line 225, in prepare
    p.prepare_body(self.data, self.files)
  File ".../dev/lib/python2.7/site-packages/requests/models.py", line 385, in prepare_body
    (body, content_type) = self._encode_files(files, data)
  File ".../dev/lib/python2.7/site-packages/requests/models.py", line 133, in _encode_files
    body, content_type = encode_multipart_formdata(new_fields)
  File ".../dev/lib/python2.7/site-packages/requests/packages/urllib3/filepost.py", line 90, in encode_multipart_formdata
    body.write(data)
TypeError: 'float' does not have the buffer interface
\`\`\`

My original problem was with a python datetime in the data dict
Thanks,


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 requests/models.py** | 88 | 137| 371 | 371 | 4883 | 
| 2 | 2 requests/packages/urllib3/filepost.py | 42 | 99| 467 | 838 | 5566 | 
| 3 | 3 requests/__init__.py | 1 | 78| 294 | 1132 | 6063 | 
| 4 | 3 requests/packages/urllib3/filepost.py | 1 | 39| 167 | 1299 | 6063 | 
| 5 | 4 requests/packages/urllib3/util.py | 1 | 35| 198 | 1497 | 8806 | 
| 6 | **4 requests/models.py** | 424 | 456| 271 | 1768 | 8806 | 
| 7 | 5 requests/packages/urllib3/__init__.py | 1 | 38| 178 | 1946 | 9192 | 
| 8 | 6 requests/packages/urllib3/request.py | 90 | 143| 508 | 2454 | 10398 | 
| 9 | 6 requests/packages/urllib3/request.py | 1 | 57| 377 | 2831 | 10398 | 
| 10 | 7 requests/packages/urllib3/response.py | 1 | 50| 237 | 3068 | 12496 | 
| 11 | 8 requests/packages/urllib3/connectionpool.py | 1 | 72| 344 | 3412 | 17227 | 
| 12 | 9 requests/sessions.py | 391 | 399| 112 | 3524 | 21167 | 
| 13 | 10 requests/packages/urllib3/exceptions.py | 1 | 96| 546 | 4070 | 21766 | 
| 14 | **10 requests/models.py** | 360 | 422| 419 | 4489 | 21766 | 
| 15 | 11 setup.py | 1 | 58| 364 | 4853 | 22130 | 
| 16 | **11 requests/models.py** | 1 | 36| 250 | 5103 | 22130 | 
| 17 | 12 requests/utils.py | 503 | 529| 336 | 5439 | 26108 | 
| 18 | 12 requests/packages/urllib3/connectionpool.py | 438 | 520| 701 | 6140 | 26108 | 
| 19 | 13 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 6140 | 26123 | 
| 20 | 13 requests/utils.py | 1 | 52| 282 | 6422 | 26123 | 
| 21 | 13 requests/packages/urllib3/response.py | 263 | 301| 257 | 6679 | 26123 | 
| 22 | **13 requests/models.py** | 39 | 86| 294 | 6973 | 26123 | 
| 23 | 13 requests/sessions.py | 401 | 409| 112 | 7085 | 26123 | 
| 24 | **13 requests/models.py** | 266 | 298| 265 | 7350 | 26123 | 
| 25 | 14 requests/packages/urllib3/_collections.py | 67 | 95| 174 | 7524 | 26748 | 
| 26 | 15 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 7682 | 29590 | 
| 27 | 15 requests/sessions.py | 268 | 359| 731 | 8413 | 29590 | 
| 28 | 16 requests/exceptions.py | 1 | 60| 272 | 8685 | 29862 | 
| 29 | 17 requests/auth.py | 1 | 55| 310 | 8995 | 31223 | 
| 30 | 18 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 9039 | 52091 | 
| 31 | 19 tasks.py | 1 | 33| 229 | 9268 | 52320 | 
| 32 | 19 requests/packages/urllib3/response.py | 131 | 205| 616 | 9884 | 52320 | 
| 33 | **19 requests/models.py** | 458 | 474| 132 | 10016 | 52320 | 
| 34 | 20 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 10171 | 54308 | 
| 35 | 20 requests/sessions.py | 411 | 428| 185 | 10356 | 54308 | 
| 36 | 21 requests/api.py | 1 | 44| 423 | 10779 | 55381 | 
| 37 | 21 requests/auth.py | 145 | 181| 311 | 11090 | 55381 | 
| 38 | 22 requests/compat.py | 1 | 116| 805 | 11895 | 56186 | 
| 39 | 22 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 12407 | 56186 | 
| 40 | 22 requests/api.py | 102 | 121| 171 | 12578 | 56186 | 
| 41 | 22 requests/api.py | 47 | 99| 438 | 13016 | 56186 | 
| 42 | 23 requests/packages/__init__.py | 1 | 4| 0 | 13016 | 56200 | 
| 43 | 24 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 13730 | 57220 | 
| 44 | 24 requests/packages/urllib3/_collections.py | 52 | 65| 152 | 13882 | 57220 | 
| 45 | 25 requests/packages/urllib3/packages/ordered_dict.py | 1 | 43| 369 | 14251 | 59377 | 
| 46 | 25 requests/packages/urllib3/request.py | 59 | 88| 283 | 14534 | 59377 | 
| 47 | **25 requests/models.py** | 300 | 358| 429 | 14963 | 59377 | 
| 48 | **25 requests/models.py** | 235 | 264| 208 | 15171 | 59377 | 
| 49 | 25 requests/auth.py | 58 | 143| 745 | 15916 | 59377 | 
| 50 | **25 requests/models.py** | 587 | 611| 167 | 16083 | 59377 | 
| 51 | 25 requests/packages/urllib3/response.py | 53 | 129| 548 | 16631 | 59377 | 
| 52 | **25 requests/models.py** | 161 | 232| 513 | 17144 | 59377 | 
| 53 | **25 requests/models.py** | 613 | 635| 152 | 17296 | 59377 | 
| 54 | 25 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 18035 | 59377 | 
| 55 | 25 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 18169 | 59377 | 
| 56 | 25 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 18369 | 59377 | 
| 57 | 26 requests/packages/urllib3/contrib/pyopenssl.py | 124 | 146| 144 | 18513 | 60617 | 
| 58 | 26 requests/packages/urllib3/connectionpool.py | 288 | 317| 259 | 18772 | 60617 | 
| 59 | 26 requests/packages/urllib3/util.py | 184 | 241| 195 | 18967 | 60617 | 
| 60 | 27 requests/packages/charade/compat.py | 21 | 35| 69 | 19036 | 60883 | 
| 61 | 27 requests/packages/urllib3/poolmanager.py | 243 | 260| 162 | 19198 | 60883 | 
| 62 | 27 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 19728 | 60883 | 
| 63 | 27 requests/packages/urllib3/response.py | 207 | 227| 178 | 19906 | 60883 | 
| 64 | 27 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 68| 545 | 20451 | 60883 | 
| 65 | 27 requests/sessions.py | 228 | 266| 296 | 20747 | 60883 | 
| 66 | 27 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 21040 | 60883 | 
| 67 | 27 requests/sessions.py | 430 | 490| 488 | 21528 | 60883 | 
| 68 | 27 requests/packages/urllib3/_collections.py | 1 | 50| 266 | 21794 | 60883 | 
| 69 | 27 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 22463 | 60883 | 
| 70 | 28 requests/cookies.py | 1 | 79| 509 | 22972 | 64054 | 
| 71 | 29 requests/packages/charade/big5freq.py | 43 | 926| 52 | 23024 | 110650 | 
| 72 | 29 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 23360 | 110650 | 
| 73 | 29 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 23609 | 110650 | 
| 74 | 29 requests/sessions.py | 153 | 226| 524 | 24133 | 110650 | 
| 75 | 29 requests/packages/urllib3/util.py | 38 | 68| 233 | 24366 | 110650 | 
| 76 | 30 requests/status_codes.py | 1 | 89| 899 | 25265 | 111549 | 
| 77 | 30 requests/packages/urllib3/util.py | 276 | 312| 256 | 25521 | 111549 | 
| 78 | 31 requests/hooks.py | 1 | 46| 188 | 25709 | 111738 | 
| 79 | 32 requests/adapters.py | 1 | 43| 267 | 25976 | 114383 | 
| 80 | 32 requests/utils.py | 177 | 208| 266 | 26242 | 114383 | 
| 81 | 32 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 26470 | 114383 | 
| 82 | 32 requests/utils.py | 55 | 88| 250 | 26720 | 114383 | 
| 83 | 32 requests/packages/urllib3/contrib/pyopenssl.py | 102 | 122| 122 | 26842 | 114383 | 
| 84 | 32 requests/packages/urllib3/contrib/pyopenssl.py | 149 | 174| 224 | 27066 | 114383 | 
| 85 | 32 requests/cookies.py | 310 | 327| 160 | 27226 | 114383 | 
| 86 | 32 requests/cookies.py | 330 | 362| 233 | 27459 | 114383 | 
| 87 | 33 docs/conf.py | 137 | 244| 615 | 28074 | 116283 | 
| 88 | 33 requests/utils.py | 532 | 568| 215 | 28289 | 116283 | 
| 89 | 33 requests/packages/urllib3/util.py | 351 | 400| 393 | 28682 | 116283 | 
| 90 | 33 requests/packages/urllib3/response.py | 230 | 261| 241 | 28923 | 116283 | 
| 91 | 33 requests/adapters.py | 46 | 93| 389 | 29312 | 116283 | 
| 92 | 33 requests/packages/urllib3/util.py | 104 | 181| 504 | 29816 | 116283 | 
| 93 | 33 docs/conf.py | 1 | 136| 1059 | 30875 | 116283 | 
| 94 | 34 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 30928 | 143057 | 
| 95 | **34 requests/models.py** | 550 | 585| 245 | 31173 | 143057 | 
| 96 | 34 requests/packages/urllib3/connectionpool.py | 352 | 436| 721 | 31894 | 143057 | 
| 97 | 34 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 32039 | 143057 | 
| 98 | **34 requests/models.py** | 140 | 158| 134 | 32173 | 143057 | 
| 99 | 34 requests/sessions.py | 361 | 389| 252 | 32425 | 143057 | 
| 100 | 34 requests/sessions.py | 1 | 34| 218 | 32643 | 143057 | 
| 101 | 34 requests/cookies.py | 82 | 100| 134 | 32777 | 143057 | 
| 102 | 35 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 34053 | 144333 | 
| 103 | 35 requests/adapters.py | 282 | 364| 542 | 34595 | 144333 | 
| 104 | 35 requests/utils.py | 319 | 355| 191 | 34786 | 144333 | 
| 105 | 35 requests/packages/urllib3/connectionpool.py | 75 | 95| 147 | 34933 | 144333 | 
| 106 | 35 requests/packages/urllib3/connectionpool.py | 562 | 588| 216 | 35149 | 144333 | 
| 107 | 36 requests/packages/charade/jisfreq.py | 44 | 570| 53 | 35202 | 172306 | 
| 108 | 36 requests/packages/urllib3/connectionpool.py | 590 | 615| 197 | 35399 | 172306 | 
| 109 | 36 requests/packages/urllib3/contrib/pyopenssl.py | 71 | 99| 207 | 35606 | 172306 | 
| 110 | 36 requests/packages/urllib3/util.py | 315 | 349| 242 | 35848 | 172306 | 
| 111 | **36 requests/models.py** | 686 | 725| 248 | 36096 | 172306 | 
| 112 | **36 requests/models.py** | 637 | 668| 198 | 36294 | 172306 | 
| 113 | 36 requests/adapters.py | 242 | 254| 141 | 36435 | 172306 | 
| 114 | 37 requests/packages/charade/euctwfreq.py | 44 | 429| 56 | 36491 | 192796 | 
| 115 | 37 requests/packages/urllib3/packages/ordered_dict.py | 92 | 113| 178 | 36669 | 192796 | 
| 116 | 37 requests/cookies.py | 187 | 277| 758 | 37427 | 192796 | 
| 117 | 37 requests/adapters.py | 149 | 183| 273 | 37700 | 192796 | 
| 118 | 37 requests/utils.py | 146 | 174| 250 | 37950 | 192796 | 
| 119 | 38 requests/structures.py | 37 | 109| 558 | 38508 | 193636 | 
| 120 | 38 requests/packages/urllib3/packages/ordered_dict.py | 174 | 261| 665 | 39173 | 193636 | 
| 121 | 38 requests/cookies.py | 365 | 404| 320 | 39493 | 193636 | 
| 122 | 38 requests/packages/urllib3/connectionpool.py | 523 | 560| 363 | 39856 | 193636 | 
| 123 | 38 requests/packages/urllib3/packages/ordered_dict.py | 143 | 172| 311 | 40167 | 193636 | 
| 124 | **38 requests/models.py** | 477 | 548| 494 | 40661 | 193636 | 
| 125 | 38 requests/packages/urllib3/connectionpool.py | 146 | 225| 644 | 41305 | 193636 | 
| 126 | 38 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 41880 | 193636 | 
| 127 | 38 requests/adapters.py | 95 | 110| 175 | 42055 | 193636 | 
| 128 | 38 requests/cookies.py | 103 | 114| 127 | 42182 | 193636 | 
| 129 | 38 requests/packages/urllib3/connectionpool.py | 319 | 350| 203 | 42385 | 193636 | 
| 130 | 39 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 21| 160 | 42545 | 194146 | 
| 131 | 39 requests/cookies.py | 117 | 142| 218 | 42763 | 194146 | 
| 132 | 39 requests/cookies.py | 292 | 308| 227 | 42990 | 194146 | 
| 133 | 39 requests/structures.py | 1 | 34| 167 | 43157 | 194146 | 
| 134 | 40 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 43341 | 194592 | 
| 135 | 40 requests/cookies.py | 279 | 290| 155 | 43496 | 194592 | 
| 136 | 40 requests/sessions.py | 37 | 65| 193 | 43689 | 194592 | 
| 137 | 40 requests/packages/urllib3/connectionpool.py | 260 | 286| 210 | 43899 | 194592 | 
| 138 | 40 requests/utils.py | 236 | 316| 442 | 44341 | 194592 | 
| 139 | 40 requests/packages/urllib3/connectionpool.py | 97 | 143| 352 | 44693 | 194592 | 
| 140 | 40 requests/utils.py | 455 | 500| 271 | 44964 | 194592 | 
| 141 | 40 requests/sessions.py | 68 | 150| 613 | 45577 | 194592 | 
| 142 | 40 requests/utils.py | 211 | 233| 280 | 45857 | 194592 | 
| 143 | 40 requests/packages/urllib3/util.py | 244 | 273| 218 | 46075 | 194592 | 
| 144 | 40 requests/packages/urllib3/poolmanager.py | 71 | 95| 190 | 46265 | 194592 | 
| 145 | 40 requests/cookies.py | 171 | 185| 151 | 46416 | 194592 | 
| 146 | 40 requests/adapters.py | 211 | 240| 220 | 46636 | 194592 | 
| 147 | 40 requests/cookies.py | 145 | 169| 249 | 46885 | 194592 | 
| 148 | 40 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 23 | 62| 350 | 47235 | 194592 | 
| 149 | 40 requests/packages/urllib3/util.py | 71 | 101| 226 | 47461 | 194592 | 
| 150 | 41 requests/certs.py | 1 | 25| 120 | 47581 | 194712 | 
| 151 | **41 requests/models.py** | 670 | 684| 168 | 47749 | 194712 | 
| 152 | 41 requests/utils.py | 424 | 452| 259 | 48008 | 194712 | 
| 153 | 41 requests/utils.py | 358 | 389| 269 | 48277 | 194712 | 
| 154 | 41 requests/sessions.py | 492 | 528| 258 | 48535 | 194712 | 
| 155 | 41 requests/packages/urllib3/poolmanager.py | 97 | 133| 298 | 48833 | 194712 | 
| 156 | 41 requests/utils.py | 121 | 143| 166 | 48999 | 194712 | 
| 157 | 41 requests/adapters.py | 256 | 280| 209 | 49208 | 194712 | 
| 158 | 41 requests/utils.py | 91 | 118| 211 | 49419 | 194712 | 
| 159 | 41 requests/packages/urllib3/connectionpool.py | 227 | 258| 256 | 49675 | 194712 | 
| 160 | 41 requests/packages/urllib3/connectionpool.py | 618 | 643| 206 | 49881 | 194712 | 
| 161 | 41 requests/utils.py | 392 | 421| 280 | 50161 | 194712 | 
| 162 | 41 requests/adapters.py | 112 | 147| 269 | 50430 | 194712 | 
| 163 | 41 requests/adapters.py | 185 | 209| 200 | 50630 | 194712 | 
| 164 | 41 requests/structures.py | 112 | 129| 114 | 50744 | 194712 | 


### Hint

```
Hi @ppavril, thanks for raising this issue!

So the problem here is that we don't ask for a string representation of keys or values. I think the correct fix is changing the following code (at [line 102 of models.py](https://github.com/kennethreitz/requests/blob/master/requests/models.py#L102)) from:

\`\`\` python
for field, val in fields:
    if isinstance(val, basestring) or not hasattr(val, '__iter__'):
        val = [val]
    for v in val:
        if v is not None:
            new_fields.append(
                (field.decode('utf-8') if isinstance(field, bytes) else field,
                 v.encode('utf-8') if isinstance(v, str) else v))
\`\`\`

to:

\`\`\` python
for field, val in fields:
    if isinstance(val, basestring) or not hasattr(val, '__iter__'):
        val = [val]
    for v in val:
        if v is not None:
            if not isinstance(v, basestring):
                v = str(v)

            new_fields.append(
                (field.decode('utf-8') if isinstance(field, bytes) else field,
                 v.encode('utf-8') if isinstance(v, str) else v))
\`\`\`

However, this is a breaking API change (we now coerce non-string types in the data dict), so should become part of #1459. We should also take advantage of that to clean this section of code up, because it's not totally easy to follow.

In the meantime @ppavril, you can work around this by calling `str()` on all your data values before passing them to Requests.

Thank you for your quick answer.
I think remaining on 1.2.0 version now and look at your improvements and chagement when I'll upgrade.
Thanks,

Funny thing. I misread @Lukasa's snippet and typed out this whole response as to why it was not optimal then looked at it again and deleted it. :-) 

I rewrote that snippet twice. =P

```

## Patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -106,6 +106,10 @@ def _encode_files(files, data):
                 val = [val]
             for v in val:
                 if v is not None:
+                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
+                    if not isinstance(v, bytes):
+                        v = str(v)
+
                     new_fields.append(
                         (field.decode('utf-8') if isinstance(field, bytes) else field,
                          v.encode('utf-8') if isinstance(v, str) else v))

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -663,6 +663,14 @@ def test_header_keys_are_native(self):
         self.assertTrue('unicode' in p.headers.keys())
         self.assertTrue('byte' in p.headers.keys())
 
+    def test_can_send_nonstring_objects_with_files(self):
+        data = {'a': 0.0}
+        files = {'b': 'foo'}
+        r = requests.Request('POST', httpbin('post'), data=data, files=files)
+        p = r.prepare()
+
+        self.assertTrue('multipart/form-data' in p.headers['Content-Type'])
+
 
 class TestCaseInsensitiveDict(unittest.TestCase):
 

```


## Code snippets

### 1 - requests/models.py:

Start line: 88, End line: 137

```python
class RequestEncodingMixin(object):

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but abritrary
        if parameters are supplied as a dict.

        """
        if (not files) or isinstance(data, str):
            return None

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, '__iter__'):
                val = [val]
            for v in val:
                if v is not None:
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
### 2 - requests/packages/urllib3/filepost.py:

Start line: 42, End line: 99

```python
def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, value) or (key, value, MIME type)
        field tuples.  The key is treated as the field name, and the value as
        the body of the form-data bytes. If the value is a tuple of two
        elements, then the first element is treated as the filename of the
        form-data section and a suitable MIME type is guessed based on the
        filename. If the value is a tuple of three elements, then the third
        element is treated as an explicit MIME type of the form-data section.

        Field names and filenames must be unicode.

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`mimetools.choose_boundary`.
    """
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()

    for fieldname, value in iter_fields(fields):
        body.write(b('--%s\r\n' % (boundary)))

        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            else:
                filename, data = value
                content_type = get_content_type(filename)
            writer(body).write('Content-Disposition: form-data; name="%s"; '
                               'filename="%s"\r\n' % (fieldname, filename))
            body.write(b('Content-Type: %s\r\n\r\n' %
                       (content_type,)))
        else:
            data = value
            writer(body).write('Content-Disposition: form-data; name="%s"\r\n'
                               % (fieldname))
            body.write(b'\r\n')

        if isinstance(data, int):
            data = str(data)  # Backwards compatibility

        if isinstance(data, six.text_type):
            writer(body).write(data)
        else:
            body.write(data)

        body.write(b'\r\n')

    body.write(b('--%s--\r\n' % (boundary)))

    content_type = str('multipart/form-data; boundary=%s' % boundary)

    return body.getvalue(), content_type
```
### 3 - requests/__init__.py:

Start line: 1, End line: 78

```python
# -*- coding: utf-8 -*-

#   __
#  /__)  _  _     _   _ _/   _
# / (   (- (/ (/ (- _)  /  _)
#          /

__title__ = 'requests'
__version__ = '1.2.3'
__build__ = 0x010203
__author__ = 'Kenneth Reitz'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright 2013 Kenneth Reitz'

# Attempt to enable urllib3's SNI support, if possible
try:
    from requests.packages.urllib3.contrib import pyopenssl
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
### 4 - requests/packages/urllib3/filepost.py:

Start line: 1, End line: 39

```python
# urllib3/filepost.py

import codecs
import mimetypes

from uuid import uuid4
from io import BytesIO

from .packages import six
from .packages.six import b

writer = codecs.lookup('utf-8')[3]


def choose_boundary():
    """
    Our embarassingly-simple replacement for mimetools.choose_boundary.
    """
    return uuid4().hex


def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'


def iter_fields(fields):
    """
    Iterate over fields.

    Supports list of (k, v) tuples and dicts.
    """
    if isinstance(fields, dict):
        return ((k, v) for k, v in six.iteritems(fields))

    return ((k, v) for k, v in fields)
```
### 5 - requests/packages/urllib3/util.py:

Start line: 1, End line: 35

```python
# urllib3/util.py


from base64 import b64encode
from collections import namedtuple
from socket import error as SocketError
from hashlib import md5, sha1
from binascii import hexlify, unhexlify

try:
    from select import poll, POLLIN
except ImportError:  # `poll` doesn't exist on OSX and other platforms
    poll = False
    try:
        from select import select
    except ImportError:  # `select` doesn't exist on AppEngine.
        select = False

try:  # Test for SSL features
    SSLContext = None
    HAS_SNI = False

    import ssl
    from ssl import wrap_socket, CERT_NONE, PROTOCOL_SSLv23
    from ssl import SSLContext  # Modern SSL?
    from ssl import HAS_SNI  # Has SNI?
except ImportError:
    pass

from .packages import six
from .exceptions import LocationParseError, SSLError
```
### 6 - requests/models.py:

Start line: 424, End line: 456

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
### 7 - requests/packages/urllib3/__init__.py:

Start line: 1, End line: 38

```python
# urllib3/__init__.py

__author__ = 'Andrey Petrov (andrey.petrov@shazow.net)'
__license__ = 'MIT'
__version__ = 'dev'


from .connectionpool import (
    HTTPConnectionPool,
    HTTPSConnectionPool,
    connection_from_url
)

from . import exceptions
from .filepost import encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import HTTPResponse
from .util import make_headers, get_host


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
### 8 - requests/packages/urllib3/request.py:

Start line: 90, End line: 143

```python
class RequestMethods(object):

    def request_encode_body(self, method, url, fields=None, headers=None,
                            encode_multipart=True, multipart_boundary=None,
                            **urlopen_kw):
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.

        When ``encode_multipart=True`` (default), then
        :meth:`urllib3.filepost.encode_multipart_formdata` is used to encode the
        payload with the appropriate content type. Otherwise
        :meth:`urllib.urlencode` is used with the
        'application/x-www-form-urlencoded' content type.

        Multipart encoding must be used when posting files, and it's reasonably
        safe to use it in other times too. However, it may break request signing,
        such as with OAuth.

        Supports an optional ``fields`` parameter of key/value strings AND
        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where
        the MIME type is optional. For example: ::

            fields = {
                'foo': 'bar',
                'fakefile': ('foofile.txt', 'contents of foofile'),
                'realfile': ('barfile.txt', open('realfile').read()),
                'typedfile': ('bazfile.bin', open('bazfile').read(),
                              'image/jpeg'),
                'nonamefile': 'contents of nonamefile field',
            }

        When uploading a file, providing a filename (the first parameter of the
        tuple) is optional but recommended to best mimick behavior of browsers.

        Note that if ``headers`` are supplied, the 'Content-Type' header will be
        overwritten because it depends on the dynamic random boundary string
        which is used to compose the body of the request. The random boundary
        string can be explicitly set with the ``multipart_boundary`` parameter.
        """
        if encode_multipart:
            body, content_type = encode_multipart_formdata(fields or {},
                                    boundary=multipart_boundary)
        else:
            body, content_type = (urlencode(fields or {}),
                                    'application/x-www-form-urlencoded')

        if headers is None:
            headers = self.headers

        headers_ = {'Content-Type': content_type}
        headers_.update(headers)

        return self.urlopen(method, url, body=body, headers=headers_,
                            **urlopen_kw)
```
### 9 - requests/packages/urllib3/request.py:

Start line: 1, End line: 57

```python
# urllib3/request.py

try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode

from .filepost import encode_multipart_formdata


__all__ = ['RequestMethods']


class RequestMethods(object):
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`~urllib3.connectionpool.HTTPConnectionPool` and
    :class:`~urllib3.poolmanager.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.

    Specifically,

    :meth:`.request_encode_url` is for sending requests whose fields are encoded
    in the URL (such as GET, HEAD, DELETE).

    :meth:`.request_encode_body` is for sending requests whose fields are
    encoded in the *body* of the request using multipart or www-form-urlencoded
    (such as for POST, PUT, PATCH).

    :meth:`.request` is for making any kind of request, it will look up the
    appropriate encoding format and use one of the above two methods to make
    the request.

    Initializer parameters:

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    """

    _encode_url_methods = set(['DELETE', 'GET', 'HEAD', 'OPTIONS'])
    _encode_body_methods = set(['PATCH', 'POST', 'PUT', 'TRACE'])

    def __init__(self, headers=None):
        self.headers = headers or {}

    def urlopen(self, method, url, body=None, headers=None,
                encode_multipart=True, multipart_boundary=None,
                **kw): # Abstract
        raise NotImplemented("Classes extending RequestMethods must implement "
                             "their own ``urlopen`` method.")
```
### 10 - requests/packages/urllib3/response.py:

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
### 14 - requests/models.py:

Start line: 360, End line: 422

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
        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

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
### 16 - requests/models.py:

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
### 22 - requests/models.py:

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
### 24 - requests/models.py:

Start line: 266, End line: 298

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
### 33 - requests/models.py:

Start line: 458, End line: 474

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
### 47 - requests/models.py:

Start line: 300, End line: 358

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
### 48 - requests/models.py:

Start line: 235, End line: 264

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
### 50 - requests/models.py:

Start line: 587, End line: 611

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
### 52 - requests/models.py:

Start line: 161, End line: 232

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
        data=dict(),
        params=dict(),
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
### 53 - requests/models.py:

Start line: 613, End line: 635

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
### 95 - requests/models.py:

Start line: 550, End line: 585

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
### 98 - requests/models.py:

Start line: 140, End line: 158

```python
class RequestHooksMixin(object):
    def register_hook(self, event, hook):
        """Properly register a hook."""

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
### 111 - requests/models.py:

Start line: 686, End line: 725

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
### 112 - requests/models.py:

Start line: 637, End line: 668

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
### 124 - requests/models.py:

Start line: 477, End line: 548

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
### 151 - requests/models.py:

Start line: 670, End line: 684

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
