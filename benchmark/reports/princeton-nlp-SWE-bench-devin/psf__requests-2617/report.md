# psf__requests-2617

| **psf/requests** | `636b946af5eac8ba4cffa63a727523cd8c2c01ab` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 8834 |
| **Any found context length** | 8834 |
| **Avg pos** | 75.0 |
| **Min pos** | 25 |
| **Max pos** | 125 |
| **Top file pos** | 6 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -328,8 +328,9 @@ def copy(self):
     def prepare_method(self, method):
         """Prepares the given HTTP method."""
         self.method = method
-        if self.method is not None:
-            self.method = self.method.upper()
+        if self.method is None:
+            raise ValueError('Request method cannot be "None"')
+        self.method = to_native_string(self.method).upper()
 
     def prepare_url(self, url, params):
         """Prepares the given HTTP URL."""
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -432,9 +432,6 @@ def request(self, method, url,
         :param cert: (optional) if String, path to ssl client cert file (.pem).
             If Tuple, ('cert', 'key') pair.
         """
-
-        method = to_native_string(method)
-
         # Create the Request.
         req = Request(
             method = method.upper(),

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/models.py | 331 | 332 | 25 | 6 | 8834
| requests/sessions.py | 435 | 437 | 125 | 21 | 44415


## Problem Statement

```
Prepared requests containing binary files will not send when unicode_literals is imported
\`\`\` python
#!/usr/bin/env python                                                                                                                                                              
from __future__ import unicode_literals
import requests
import sys


def main():
    request = requests.Request(method='PUT', url='https://httpbin.org/put')
    with open(sys.argv[1], 'rb') as fp:
        request.files = {'hello': fp}
        prepared = request.prepare()
        requests.Session().send(prepared)

if __name__ == '__main__':
    sys.exit(main())
\`\`\`

The above program works perfectly in python3, and in python2 when `unicode_literals` is not imported. If the request isn't prepared it works without a problem unfortunately, I require both prepared requests and `unicode_literals` in my project.

The exception raised is:

\`\`\`\`\`\`
Traceback (most recent call last):
  File "./test.py", line 15, in <module>
    sys.exit(main())
  File "./test.py", line 12, in main
    requests.Session().send(prepared)
  File "/Users/bboe/.venv/p27/lib/python2.7/site-packages/requests-2.7.0-py2.7.egg/requests/sessions.py", line 573, in send
    r = adapter.send(request, **kwargs)
  File "/Users/bboe/.venv/p27/lib/python2.7/site-packages/requests-2.7.0-py2.7.egg/requests/adapters.py", line 370, in send
    timeout=timeout
  File "/Users/bboe/.venv/p27/lib/python2.7/site-packages/requests-2.7.0-py2.7.egg/requests/packages/urllib3/connectionpool.py", line 544, in urlopen
    body=body, headers=headers)
  File "/Users/bboe/.venv/p27/lib/python2.7/site-packages/requests-2.7.0-py2.7.egg/requests/packages/urllib3/connectionpool.py", line 349, in _make_request
    conn.request(method, url, **httplib_request_kw)
  File "/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7/httplib.py", line 1001, in request
    self._send_request(method, url, body, headers)
  File "/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7/httplib.py", line 1035, in _send_request
    self.endheaders(body)
  File "/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7/httplib.py", line 997, in endheaders
    self._send_output(message_body)
  File "/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7/httplib.py", line 848, in _send_output
    msg += message_body
UnicodeDecodeError: 'ascii' codec can't decode byte 0xff in position 109: ordinal not in range(128)\`\`\`
\`\`\`\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 requests/__init__.py | 1 | 78| 293 | 293 | 497 | 
| 2 | 2 requests/packages/urllib3/response.py | 1 | 65| 360 | 653 | 3953 | 
| 3 | 3 requests/packages/urllib3/util/ssl_.py | 47 | 102| 477 | 1130 | 6305 | 
| 4 | 4 requests/packages/urllib3/util/request.py | 1 | 72| 231 | 1361 | 6779 | 
| 5 | 5 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 97| 747 | 2108 | 8909 | 
| 6 | **6 requests/models.py** | 470 | 480| 132 | 2240 | 15026 | 
| 7 | 6 requests/packages/urllib3/util/ssl_.py | 1 | 45| 420 | 2660 | 15026 | 
| 8 | **6 requests/models.py** | 407 | 468| 412 | 3072 | 15026 | 
| 9 | 7 requests/compat.py | 1 | 63| 394 | 3466 | 15420 | 
| 10 | 8 requests/packages/urllib3/util/__init__.py | 1 | 25| 114 | 3580 | 15534 | 
| 11 | 9 requests/packages/urllib3/connection.py | 1 | 60| 301 | 3881 | 17522 | 
| 12 | 9 requests/packages/urllib3/contrib/pyopenssl.py | 169 | 190| 176 | 4057 | 17522 | 
| 13 | **9 requests/models.py** | 334 | 405| 588 | 4645 | 17522 | 
| 14 | 10 requests/packages/urllib3/connectionpool.py | 1 | 50| 251 | 4896 | 23760 | 
| 15 | 11 setup.py | 1 | 75| 522 | 5418 | 24282 | 
| 16 | 12 requests/packages/urllib3/exceptions.py | 2 | 57| 296 | 5714 | 25258 | 
| 17 | **12 requests/models.py** | 482 | 502| 165 | 5879 | 25258 | 
| 18 | 12 requests/packages/urllib3/response.py | 340 | 401| 439 | 6318 | 25258 | 
| 19 | 12 requests/packages/urllib3/contrib/pyopenssl.py | 192 | 215| 153 | 6471 | 25258 | 
| 20 | 13 requests/utils.py | 354 | 386| 191 | 6662 | 30367 | 
| 21 | 13 requests/packages/urllib3/exceptions.py | 81 | 170| 545 | 7207 | 30367 | 
| 22 | 14 requests/auth.py | 61 | 156| 842 | 8049 | 32012 | 
| 23 | 15 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 8561 | 34854 | 
| 24 | 16 requests/packages/urllib3/_collections.py | 1 | 23| 124 | 8685 | 37176 | 
| **-> 25 <-** | **16 requests/models.py** | 315 | 332| 149 | 8834 | 37176 | 
| 26 | 16 requests/packages/urllib3/contrib/pyopenssl.py | 217 | 249| 191 | 9025 | 37176 | 
| 27 | 16 requests/packages/urllib3/response.py | 403 | 423| 232 | 9257 | 37176 | 
| 28 | 17 requests/adapters.py | 323 | 438| 823 | 10080 | 40520 | 
| 29 | **17 requests/models.py** | 101 | 159| 447 | 10527 | 40520 | 
| 30 | 18 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 10527 | 40535 | 
| 31 | **18 requests/models.py** | 298 | 313| 156 | 10683 | 40535 | 
| 32 | 19 requests/packages/urllib3/filepost.py | 1 | 37| 172 | 10855 | 41061 | 
| 33 | 20 requests/packages/chardet/compat.py | 21 | 35| 69 | 10924 | 41327 | 
| 34 | 20 requests/adapters.py | 1 | 50| 352 | 11276 | 41327 | 
| 35 | **21 requests/sessions.py** | 539 | 607| 491 | 11767 | 46403 | 
| 36 | 22 requests/packages/chardet/universaldetector.py | 29 | 41| 127 | 11894 | 48085 | 
| 37 | 22 requests/packages/urllib3/connection.py | 143 | 156| 158 | 12052 | 48085 | 
| 38 | 22 requests/auth.py | 1 | 58| 330 | 12382 | 48085 | 
| 39 | 22 requests/packages/urllib3/contrib/pyopenssl.py | 252 | 294| 363 | 12745 | 48085 | 
| 40 | 23 requests/packages/urllib3/request.py | 83 | 142| 560 | 13305 | 49271 | 
| 41 | 24 requests/packages/chardet/mbcssm.py | 482 | 500| 311 | 13616 | 60857 | 
| 42 | 24 requests/packages/urllib3/connection.py | 159 | 178| 170 | 13786 | 60857 | 
| 43 | 25 requests/exceptions.py | 30 | 100| 335 | 14121 | 61355 | 
| 44 | 25 requests/packages/chardet/mbcssm.py | 537 | 573| 785 | 14906 | 61355 | 
| 45 | 25 requests/packages/urllib3/request.py | 1 | 50| 353 | 15259 | 61355 | 
| 46 | 25 requests/exceptions.py | 1 | 27| 163 | 15422 | 61355 | 
| 47 | 25 requests/packages/urllib3/connectionpool.py | 610 | 650| 405 | 15827 | 61355 | 
| 48 | **25 requests/models.py** | 1 | 49| 381 | 16208 | 61355 | 
| 49 | 26 requests/packages/urllib3/contrib/ntlmpool.py | 41 | 115| 714 | 16922 | 62316 | 
| 50 | 26 requests/adapters.py | 53 | 114| 550 | 17472 | 62316 | 
| 51 | 26 requests/packages/urllib3/response.py | 176 | 203| 247 | 17719 | 62316 | 
| 52 | **26 requests/models.py** | 264 | 296| 241 | 17960 | 62316 | 
| 53 | 26 requests/packages/urllib3/packages/six.py | 302 | 386| 530 | 18490 | 62316 | 
| 54 | 26 requests/packages/chardet/mbcssm.py | 427 | 445| 320 | 18810 | 62316 | 
| 55 | 26 requests/packages/urllib3/response.py | 425 | 467| 333 | 19143 | 62316 | 
| 56 | **26 requests/models.py** | 504 | 531| 272 | 19415 | 62316 | 
| 57 | 27 requests/status_codes.py | 1 | 90| 918 | 20333 | 63234 | 
| 58 | 27 requests/packages/urllib3/contrib/pyopenssl.py | 100 | 115| 152 | 20485 | 63234 | 
| 59 | 27 requests/packages/urllib3/connectionpool.py | 504 | 608| 850 | 21335 | 63234 | 
| 60 | 27 requests/utils.py | 620 | 646| 336 | 21671 | 63234 | 
| 61 | **27 requests/models.py** | 752 | 788| 259 | 21930 | 63234 | 
| 62 | 27 requests/packages/urllib3/connection.py | 202 | 265| 534 | 22464 | 63234 | 
| 63 | 27 requests/packages/urllib3/response.py | 286 | 310| 208 | 22672 | 63234 | 
| 64 | 27 requests/utils.py | 70 | 113| 312 | 22984 | 63234 | 
| 65 | 27 requests/packages/urllib3/request.py | 52 | 81| 283 | 23267 | 63234 | 
| 66 | 27 requests/utils.py | 1 | 67| 334 | 23601 | 63234 | 
| 67 | 27 requests/packages/urllib3/filepost.py | 40 | 55| 120 | 23721 | 63234 | 
| 68 | 28 requests/packages/chardet/escsm.py | 226 | 243| 227 | 23948 | 67726 | 
| 69 | 28 requests/packages/urllib3/contrib/pyopenssl.py | 149 | 167| 151 | 24099 | 67726 | 
| 70 | 28 requests/packages/urllib3/_collections.py | 306 | 324| 157 | 24256 | 67726 | 
| 71 | 28 requests/packages/urllib3/connectionpool.py | 694 | 709| 121 | 24377 | 67726 | 
| 72 | 28 requests/packages/urllib3/filepost.py | 58 | 94| 233 | 24610 | 67726 | 
| 73 | 28 requests/packages/chardet/escsm.py | 171 | 189| 335 | 24945 | 67726 | 
| 74 | 28 requests/packages/chardet/escsm.py | 117 | 134| 304 | 25249 | 67726 | 
| 75 | 29 requests/packages/chardet/mbcharsetprober.py | 30 | 87| 449 | 25698 | 68443 | 
| 76 | 30 requests/packages/chardet/__init__.py | 18 | 33| 114 | 25812 | 68739 | 
| 77 | 30 requests/packages/urllib3/connectionpool.py | 317 | 384| 655 | 26467 | 68739 | 
| 78 | 30 requests/packages/urllib3/response.py | 205 | 284| 651 | 27118 | 68739 | 
| 79 | 30 requests/utils.py | 389 | 414| 208 | 27326 | 68739 | 
| 80 | 30 requests/packages/chardet/mbcssm.py | 315 | 337| 312 | 27638 | 68739 | 
| 81 | 30 requests/packages/chardet/mbcssm.py | 261 | 278| 245 | 27883 | 68739 | 
| 82 | 30 requests/adapters.py | 196 | 230| 273 | 28156 | 68739 | 
| 83 | **30 requests/models.py** | 52 | 99| 294 | 28450 | 68739 | 
| 84 | 31 requests/packages/urllib3/__init__.py | 1 | 35| 208 | 28658 | 69185 | 
| 85 | 32 requests/packages/urllib3/poolmanager.py | 1 | 28| 167 | 28825 | 71255 | 
| 86 | 33 requests/packages/chardet/gb2312freq.py | 42 | 473| 44 | 28869 | 92123 | 
| 87 | 34 requests/cookies.py | 1 | 46| 299 | 29168 | 95893 | 
| 88 | 35 requests/packages/urllib3/util/retry.py | 106 | 142| 313 | 29481 | 98092 | 
| 89 | 35 requests/packages/chardet/mbcssm.py | 102 | 121| 336 | 29817 | 98092 | 
| 90 | 36 requests/packages/chardet/latin1prober.py | 97 | 122| 199 | 30016 | 100174 | 
| 91 | 36 requests/packages/urllib3/connectionpool.py | 711 | 727| 151 | 30167 | 100174 | 
| 92 | 36 requests/packages/chardet/mbcssm.py | 67 | 81| 154 | 30321 | 100174 | 
| 93 | 36 requests/packages/chardet/mbcssm.py | 376 | 390| 161 | 30482 | 100174 | 
| 94 | 37 requests/packages/chardet/utf8prober.py | 28 | 77| 361 | 30843 | 100784 | 
| 95 | 38 requests/packages/urllib3/packages/ssl_match_hostname/__init__.py | 1 | 14| 0 | 30843 | 100881 | 
| 96 | 38 requests/packages/chardet/mbcssm.py | 502 | 535| 852 | 31695 | 100881 | 
| 97 | 38 requests/packages/urllib3/response.py | 68 | 174| 820 | 32515 | 100881 | 
| 98 | 38 requests/packages/urllib3/connection.py | 63 | 118| 556 | 33071 | 100881 | 
| 99 | 38 requests/auth.py | 198 | 213| 150 | 33221 | 100881 | 
| 100 | 38 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 33960 | 100881 | 
| 101 | **38 requests/models.py** | 186 | 261| 550 | 34510 | 100881 | 
| 102 | 39 requests/packages/urllib3/fields.py | 70 | 102| 274 | 34784 | 102144 | 
| 103 | 39 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 34942 | 102144 | 
| 104 | 39 requests/packages/urllib3/util/retry.py | 144 | 156| 116 | 35058 | 102144 | 
| 105 | 39 requests/packages/chardet/escsm.py | 82 | 115| 780 | 35838 | 102144 | 
| 106 | 39 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 36507 | 102144 | 
| 107 | 39 requests/packages/urllib3/fields.py | 21 | 46| 217 | 36724 | 102144 | 
| 108 | 40 requests/packages/__init__.py | 1 | 4| 0 | 36724 | 102158 | 
| 109 | **40 requests/sessions.py** | 346 | 384| 299 | 37023 | 102158 | 
| 110 | 41 requests/packages/chardet/chardetect.py | 1 | 23| 124 | 37147 | 102700 | 
| 111 | 41 requests/packages/chardet/escsm.py | 136 | 169| 780 | 37927 | 102700 | 
| 112 | 41 requests/packages/urllib3/connectionpool.py | 753 | 768| 144 | 38071 | 102700 | 
| 113 | 42 docs/conf.py | 140 | 249| 639 | 38710 | 104636 | 
| 114 | 43 requests/packages/chardet/langcyrillicmodel.py | 49 | 66| 650 | 39360 | 117502 | 
| 115 | 44 requests/api.py | 112 | 122| 114 | 39474 | 118853 | 
| 116 | 45 requests/packages/chardet/sbcsgroupprober.py | 29 | 39| 178 | 39652 | 119638 | 
| 117 | 45 docs/conf.py | 1 | 139| 1071 | 40723 | 119638 | 
| 118 | 45 requests/packages/chardet/langcyrillicmodel.py | 144 | 330| 473 | 41196 | 119638 | 
| 119 | 46 requests/packages/urllib3/packages/ordered_dict.py | 1 | 42| 369 | 41565 | 121795 | 
| 120 | 46 requests/packages/chardet/escsm.py | 191 | 224| 780 | 42345 | 121795 | 
| 121 | 47 requests/packages/urllib3/util/connection.py | 1 | 43| 309 | 42654 | 122534 | 
| 122 | 48 requests/packages/chardet/gb2312prober.py | 28 | 42| 127 | 42781 | 122911 | 
| 123 | 48 requests/packages/chardet/mbcssm.py | 447 | 480| 843 | 43624 | 122911 | 
| 124 | 48 requests/packages/urllib3/exceptions.py | 60 | 78| 132 | 43756 | 122911 | 
| **-> 125 <-** | **48 requests/sessions.py** | 386 | 467| 659 | 44415 | 122911 | 
| 126 | 49 requests/packages/urllib3/util/timeout.py | 1 | 98| 842 | 45257 | 124920 | 
| 127 | 49 requests/api.py | 72 | 95| 180 | 45437 | 124920 | 
| 128 | 49 requests/cookies.py | 48 | 60| 124 | 45561 | 124920 | 
| 129 | 49 requests/packages/urllib3/util/ssl_.py | 142 | 178| 256 | 45817 | 124920 | 
| 130 | 49 requests/packages/chardet/mbcssm.py | 280 | 313| 842 | 46659 | 124920 | 
| 131 | 49 requests/packages/urllib3/util/timeout.py | 138 | 152| 136 | 46795 | 124920 | 
| 132 | 49 requests/packages/urllib3/poolmanager.py | 265 | 281| 168 | 46963 | 124920 | 
| 133 | 50 requests/packages/chardet/sbcharsetprober.py | 29 | 68| 326 | 47289 | 126026 | 
| 134 | 50 requests/packages/chardet/universaldetector.py | 64 | 132| 809 | 48098 | 126026 | 
| 135 | 50 requests/packages/chardet/langcyrillicmodel.py | 68 | 85| 649 | 48747 | 126026 | 
| 136 | 50 requests/packages/chardet/mbcssm.py | 392 | 425| 843 | 49590 | 126026 | 
| 137 | 51 requests/packages/urllib3/util/response.py | 1 | 23| 118 | 49708 | 126144 | 
| 138 | **51 requests/models.py** | 728 | 750| 152 | 49860 | 126144 | 
| 139 | 52 requests/packages/chardet/charsetprober.py | 29 | 63| 184 | 50044 | 126590 | 
| 140 | 52 requests/packages/urllib3/poolmanager.py | 31 | 73| 293 | 50337 | 126590 | 
| 141 | 52 requests/packages/urllib3/util/ssl_.py | 244 | 281| 373 | 50710 | 126590 | 
| 142 | 52 requests/packages/urllib3/connection.py | 181 | 200| 145 | 50855 | 126590 | 
| 143 | 52 requests/packages/chardet/escsm.py | 65 | 80| 246 | 51101 | 126590 | 
| 144 | 52 requests/packages/urllib3/response.py | 312 | 338| 210 | 51311 | 126590 | 
| 145 | 53 requests/packages/chardet/langhebrewmodel.py | 37 | 54| 670 | 51981 | 135927 | 
| 146 | 53 requests/packages/chardet/mbcssm.py | 158 | 174| 220 | 52201 | 135927 | 
| 147 | 54 requests/packages/chardet/langbulgarianmodel.py | 75 | 230| 217 | 52418 | 146088 | 
| 148 | 55 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 53694 | 147364 | 
| 149 | 55 requests/cookies.py | 94 | 112| 134 | 53828 | 147364 | 
| 150 | 55 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 39| 256 | 54084 | 147364 | 
| 151 | 55 requests/packages/urllib3/_collections.py | 230 | 254| 209 | 54293 | 147364 | 
| 152 | 55 requests/packages/urllib3/util/timeout.py | 169 | 191| 196 | 54489 | 147364 | 
| 153 | 55 requests/packages/chardet/chardetect.py | 48 | 81| 272 | 54761 | 147364 | 
| 154 | 55 requests/packages/urllib3/packages/ordered_dict.py | 114 | 140| 200 | 54961 | 147364 | 
| 155 | 55 requests/packages/urllib3/connectionpool.py | 300 | 315| 247 | 55208 | 147364 | 
| 156 | 55 requests/api.py | 98 | 109| 142 | 55350 | 147364 | 
| 157 | 55 requests/adapters.py | 232 | 261| 229 | 55579 | 147364 | 
| 158 | 55 requests/packages/urllib3/poolmanager.py | 141 | 189| 391 | 55970 | 147364 | 
| 159 | 56 requests/packages/chardet/big5prober.py | 28 | 43| 121 | 56091 | 147735 | 
| 160 | 56 requests/packages/chardet/sbcharsetprober.py | 70 | 121| 530 | 56621 | 147735 | 
| 161 | 56 requests/packages/chardet/mbcssm.py | 123 | 156| 842 | 57463 | 147735 | 
| 162 | 56 requests/packages/chardet/mbcssm.py | 83 | 100| 699 | 58162 | 147735 | 
| 163 | 56 requests/packages/chardet/latin1prober.py | 29 | 83| 1165 | 59327 | 147735 | 
| 164 | **56 requests/sessions.py** | 510 | 518| 112 | 59439 | 147735 | 
| 165 | 57 requests/certs.py | 1 | 26| 133 | 59572 | 147868 | 
| 166 | 57 requests/packages/chardet/langcyrillicmodel.py | 125 | 142| 649 | 60221 | 147868 | 
| 167 | 57 requests/packages/chardet/mbcssm.py | 28 | 65| 869 | 61090 | 147868 | 
| 168 | 58 requests/packages/chardet/langhungarianmodel.py | 72 | 226| 213 | 61303 | 157854 | 
| 169 | 58 requests/utils.py | 308 | 351| 237 | 61540 | 157854 | 
| 170 | 59 requests/packages/chardet/euctwprober.py | 28 | 42| 127 | 61667 | 158231 | 
| 171 | 59 requests/packages/chardet/langcyrillicmodel.py | 106 | 123| 649 | 62316 | 158231 | 
| 172 | 59 requests/packages/chardet/mbcssm.py | 339 | 373| 871 | 63187 | 158231 | 
| 173 | 60 requests/packages/chardet/langgreekmodel.py | 72 | 226| 205 | 63392 | 168345 | 
| 174 | 60 requests/cookies.py | 62 | 91| 208 | 63600 | 168345 | 
| 175 | 60 requests/packages/chardet/mbcssm.py | 211 | 224| 138 | 63738 | 168345 | 
| 176 | **60 requests/sessions.py** | 229 | 264| 189 | 63927 | 168345 | 
| 177 | 60 requests/utils.py | 288 | 305| 194 | 64121 | 168345 | 
| 178 | 60 requests/packages/urllib3/_collections.py | 71 | 104| 201 | 64322 | 168345 | 
| 179 | 60 requests/packages/urllib3/contrib/pyopenssl.py | 118 | 146| 207 | 64529 | 168345 | 
| 180 | 60 requests/packages/urllib3/connectionpool.py | 278 | 298| 146 | 64675 | 168345 | 
| 181 | 60 requests/packages/chardet/langhungarianmodel.py | 34 | 51| 638 | 65313 | 168345 | 
| 182 | 60 requests/packages/urllib3/fields.py | 137 | 154| 142 | 65455 | 168345 | 
| 183 | 60 requests/packages/chardet/langhungarianmodel.py | 53 | 70| 640 | 66095 | 168345 | 
| 184 | 60 requests/packages/urllib3/packages/ordered_dict.py | 44 | 52| 134 | 66229 | 168345 | 
| 185 | **60 requests/sessions.py** | 499 | 508| 139 | 66368 | 168345 | 
| 186 | 60 requests/packages/urllib3/fields.py | 156 | 178| 178 | 66546 | 168345 | 
| 187 | 60 requests/packages/chardet/latin1prober.py | 84 | 94| 318 | 66864 | 168345 | 
| 188 | 61 requests/packages/chardet/escprober.py | 66 | 87| 202 | 67066 | 169050 | 
| 189 | 61 requests/packages/urllib3/packages/ordered_dict.py | 54 | 89| 293 | 67359 | 169050 | 
| 190 | 62 requests/hooks.py | 1 | 46| 188 | 67547 | 169239 | 
| 191 | 62 requests/packages/chardet/universaldetector.py | 134 | 171| 316 | 67863 | 169239 | 
| 192 | 62 requests/packages/urllib3/fields.py | 104 | 135| 232 | 68095 | 169239 | 
| 193 | 62 requests/utils.py | 649 | 660| 152 | 68247 | 169239 | 
| 194 | 62 requests/packages/urllib3/poolmanager.py | 192 | 263| 572 | 68819 | 169239 | 
| 195 | 63 requests/structures.py | 1 | 86| 572 | 69391 | 169925 | 
| 196 | 63 requests/packages/urllib3/connectionpool.py | 421 | 503| 793 | 70184 | 169925 | 
| 197 | 63 requests/packages/urllib3/__init__.py | 37 | 70| 238 | 70422 | 169925 | 
| 198 | 64 requests/packages/chardet/charsetgroupprober.py | 28 | 56| 214 | 70636 | 170776 | 
| 199 | 64 requests/packages/chardet/langhebrewmodel.py | 56 | 202| 141 | 70777 | 170776 | 
| 200 | 64 requests/packages/urllib3/util/ssl_.py | 181 | 241| 575 | 71352 | 170776 | 


### Hint

```
Unfortunately this is a bit of a limitation imposed on us by httplib. As you can see, the place where unicode and bytes are concatenated together is actually deep inside httplib. I'm afraid you'll have to pass bytestrings to requests.

Can you explain why it works fine when the request isn't prepared? That seems inconsistent.

Because the higher level code coerces your strings to the platform-native string type (bytes on Python 2, unicode on Python 3). One of the problems when you step this 'more control' abstraction is that we stop doing some of the helpful things we do at the higher abstraction levels.

We have a `to_native_string` function that you could use (it's what we use).

I'll check that out. Thanks.

@bboe it's in your best interest to copy and paste `to_native_string` out of requests though. It's an undocumented function that's effectively meant to be internal to requests. If we move it around or change something in it, it could cause compatibility problems for you and there's no guarantee of backwards compatibility for that function as it isn't a defined member of the API.

That said, @Lukasa and I agree that it's highly unlikely to break, change, or disappear. So, while I'd prefer you to copy and paste it out, there's nothing I can do to enforce that. ;)

As it turns out, only the request `method` needs to be in the right format. In my above example changing:

\`\`\`
    request = requests.Request(method='PUT', url='https://httpbin.org/put')
\`\`\`

to

\`\`\`
    request = requests.Request(method=to_native_string('PUT'), url='https://httpbin.org/put')
\`\`\`

is all that is needed. Maybe this simple of a fix could be included in requests?

I'm frankly starting to wonder why we don't do all of this `to_native_string` work when we prepare the request in the first place. It seems like the more correct place for this conversion than [here](https://github.com/kennethreitz/requests/blob/a0d9e0bc57c971823811de38e5733b4b85e575ae/requests/sessions.py#L436).

Suits me. =)

```

## Patch

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -328,8 +328,9 @@ def copy(self):
     def prepare_method(self, method):
         """Prepares the given HTTP method."""
         self.method = method
-        if self.method is not None:
-            self.method = self.method.upper()
+        if self.method is None:
+            raise ValueError('Request method cannot be "None"')
+        self.method = to_native_string(self.method).upper()
 
     def prepare_url(self, url, params):
         """Prepares the given HTTP URL."""
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -432,9 +432,6 @@ def request(self, method, url,
         :param cert: (optional) if String, path to ssl client cert file (.pem).
             If Tuple, ('cert', 'key') pair.
         """
-
-        method = to_native_string(method)
-
         # Create the Request.
         req = Request(
             method = method.upper(),

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -89,7 +89,7 @@ def test_invalid_url(self):
             requests.get('http://')
 
     def test_basic_building(self):
-        req = requests.Request()
+        req = requests.Request(method='GET')
         req.url = 'http://kennethreitz.org/'
         req.data = {'life': '42'}
 
@@ -813,7 +813,7 @@ def test_get_auth_from_url_encoded_hashes(self):
         assert ('user', 'pass#pass') == requests.utils.get_auth_from_url(url)
 
     def test_cannot_send_unprepared_requests(self):
-        r = requests.Request(url=HTTPBIN)
+        r = requests.Request(method='GET', url=HTTPBIN)
         with pytest.raises(ValueError):
             requests.Session().send(r)
 
@@ -1617,6 +1617,16 @@ def test_prepare_unicode_url():
     assert_copy(p, p.copy())
 
 
+def test_prepare_requires_a_request_method():
+    req = requests.Request()
+    with pytest.raises(ValueError):
+        req.prepare()
+
+    prepped = PreparedRequest()
+    with pytest.raises(ValueError):
+        prepped.prepare()
+
+
 def test_urllib3_retries():
     from requests.packages.urllib3.util import Retry
     s = requests.Session()

```


## Code snippets

### 1 - requests/__init__.py:

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
### 2 - requests/packages/urllib3/response.py:

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
### 3 - requests/packages/urllib3/util/ssl_.py:

Start line: 47, End line: 102

```python
try:
    from ssl import SSLContext  # Modern SSL?
except ImportError:
    import sys

    class SSLContext(object):  # Platform-specific: Python 2 & 3.1
        supports_set_ciphers = ((2, 7) <= sys.version_info < (3,) or
                                (3, 2) <= sys.version_info)

        def __init__(self, protocol_version):
            self.protocol = protocol_version
            # Use default values from a real SSLContext
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None

        def load_cert_chain(self, certfile, keyfile):
            self.certfile = certfile
            self.keyfile = keyfile

        def load_verify_locations(self, location):
            self.ca_certs = location

        def set_ciphers(self, cipher_suite):
            if not self.supports_set_ciphers:
                raise TypeError(
                    'Your version of Python does not support setting '
                    'a custom cipher suite. Please upgrade to Python '
                    '2.7, 3.2, or later if you need this functionality.'
                )
            self.ciphers = cipher_suite

        def wrap_socket(self, socket, server_hostname=None):
            warnings.warn(
                'A true SSLContext object is not available. This prevents '
                'urllib3 from configuring SSL appropriately and may cause '
                'certain SSL connections to fail. For more information, see '
                'https://urllib3.readthedocs.org/en/latest/security.html'
                '#insecureplatformwarning.',
                InsecurePlatformWarning
            )
            kwargs = {
                'keyfile': self.keyfile,
                'certfile': self.certfile,
                'ca_certs': self.ca_certs,
                'cert_reqs': self.verify_mode,
                'ssl_version': self.protocol,
            }
            if self.supports_set_ciphers:  # Platform-specific: Python 2.7+
                return wrap_socket(socket, ciphers=self.ciphers, **kwargs)
            else:  # Platform-specific: Python 2.6
                return wrap_socket(socket, **kwargs)
```
### 4 - requests/packages/urllib3/util/request.py:

Start line: 1, End line: 72

```python
from base64 import b64encode

from ..packages.six import b

ACCEPT_ENCODING = 'gzip,deflate'


def make_headers(keep_alive=None, accept_encoding=None, user_agent=None,
                 basic_auth=None, proxy_basic_auth=None, disable_cache=None):
    headers = {}
    if accept_encoding:
        if isinstance(accept_encoding, str):
            pass
        elif isinstance(accept_encoding, list):
            accept_encoding = ','.join(accept_encoding)
        else:
            accept_encoding = ACCEPT_ENCODING
        headers['accept-encoding'] = accept_encoding

    if user_agent:
        headers['user-agent'] = user_agent

    if keep_alive:
        headers['connection'] = 'keep-alive'

    if basic_auth:
        headers['authorization'] = 'Basic ' + \
            b64encode(b(basic_auth)).decode('utf-8')

    if proxy_basic_auth:
        headers['proxy-authorization'] = 'Basic ' + \
            b64encode(b(proxy_basic_auth)).decode('utf-8')

    if disable_cache:
        headers['cache-control'] = 'no-cache'

    return headers
```
### 5 - requests/packages/urllib3/contrib/pyopenssl.py:

Start line: 1, End line: 97

```python
'''SSL with SNI_-support for Python 2. Follow these instructions if you would
like to verify SSL certificates in Python 2. Note, the default libraries do
*not* do certificate checking; you need to do additional work to validate
certificates yourself.

This needs the following packages installed:

* pyOpenSSL (tested with 0.13)
* ndg-httpsclient (tested with 0.3.2)
* pyasn1 (tested with 0.1.6)

You can install them with the following command:

    pip install pyopenssl ndg-httpsclient pyasn1

To activate certificate checking, call
:func:`~urllib3.contrib.pyopenssl.inject_into_urllib3` from your Python code
before you begin making HTTP requests. This can be done in a ``sitecustomize``
module, or at any other time before your application begins using ``urllib3``,
like this::

    try:
        import urllib3.contrib.pyopenssl
        urllib3.contrib.pyopenssl.inject_into_urllib3()
    except ImportError:
        pass

Now you can use :mod:`urllib3` as you normally would, and it will support SNI
when the required modules are installed.

Activating this module also has the positive side effect of disabling SSL/TLS
compression in Python 2 (see `CRIME attack`_).

If you want to configure the default list of supported cipher suites, you can
set the ``urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST`` variable.

Module Variables
----------------

:var DEFAULT_SSL_CIPHER_LIST: The list of supported SSL/TLS cipher suites.

.. _sni: https://en.wikipedia.org/wiki/Server_Name_Indication
.. _crime attack: https://en.wikipedia.org/wiki/CRIME_(security_exploit)

'''

try:
    from ndg.httpsclient.ssl_peer_verification import SUBJ_ALT_NAME_SUPPORT
    from ndg.httpsclient.subj_alt_name import SubjectAltName as BaseSubjectAltName
except SyntaxError as e:
    raise ImportError(e)

import OpenSSL.SSL
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.type import univ, constraint
from socket import _fileobject, timeout
import ssl
import select

from .. import connection
from .. import util

__all__ = ['inject_into_urllib3', 'extract_from_urllib3']

# SNI only *really* works if we can read the subjectAltName of certificates.
HAS_SNI = SUBJ_ALT_NAME_SUPPORT

# Map from urllib3 to PyOpenSSL compatible parameter-values.
_openssl_versions = {
    ssl.PROTOCOL_SSLv23: OpenSSL.SSL.SSLv23_METHOD,
    ssl.PROTOCOL_TLSv1: OpenSSL.SSL.TLSv1_METHOD,
}

try:
    _openssl_versions.update({ssl.PROTOCOL_SSLv3: OpenSSL.SSL.SSLv3_METHOD})
except AttributeError:
    pass

_openssl_verify = {
    ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE,
    ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER,
    ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER
                       + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT,
}

DEFAULT_SSL_CIPHER_LIST = util.ssl_.DEFAULT_CIPHERS


orig_util_HAS_SNI = util.HAS_SNI
orig_connection_ssl_wrap_socket = connection.ssl_wrap_socket


def inject_into_urllib3():
    'Monkey-patch urllib3 with PyOpenSSL-backed SSL-support.'

    connection.ssl_wrap_socket = ssl_wrap_socket
    util.HAS_SNI = HAS_SNI
```
### 6 - requests/models.py:

Start line: 470, End line: 480

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
        elif (self.method not in ('GET', 'HEAD')) and (self.headers.get('Content-Length') is None):
            self.headers['Content-Length'] = '0'
```
### 7 - requests/packages/urllib3/util/ssl_.py:

Start line: 1, End line: 45

```python
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256

from ..exceptions import SSLError, InsecurePlatformWarning


SSLContext = None
HAS_SNI = False
create_default_context = None

import errno
import warnings

try:  # Test for SSL features
    import ssl
    from ssl import wrap_socket, CERT_NONE, PROTOCOL_SSLv23
    from ssl import HAS_SNI  # Has SNI?
except ImportError:
    pass


try:
    from ssl import OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION
except ImportError:
    OP_NO_SSLv2, OP_NO_SSLv3 = 0x1000000, 0x2000000
    OP_NO_COMPRESSION = 0x20000

# A secure default.
# Sources for more information on TLS ciphers:
#
# - https://wiki.mozilla.org/Security/Server_Side_TLS
# - https://www.ssllabs.com/projects/best-practices/index.html
# - https://hynek.me/articles/hardening-your-web-servers-ssl-ciphers/
#
# The general intent is:
# - Prefer cipher suites that offer perfect forward secrecy (DHE/ECDHE),
# - prefer ECDHE over DHE for better performance,
# - prefer any AES-GCM over any AES-CBC for better performance and security,
# - use 3DES as fallback which is secure but slow,
# - disable NULL authentication, MD5 MACs and DSS for security reasons.
DEFAULT_CIPHERS = (
    'ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:ECDH+HIGH:'
    'DH+HIGH:ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+HIGH:RSA+3DES:!aNULL:'
    '!eNULL:!MD5'
)
```
### 8 - requests/models.py:

Start line: 407, End line: 468

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        if headers:
            self.headers = CaseInsensitiveDict((to_native_string(name), value) for name, value in headers.items())
        else:
            self.headers = CaseInsensitiveDict()

    def prepare_body(self, data, files, json=None):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None
        length = None

        if json is not None:
            content_type = 'application/json'
            body = json_dumps(json)

        is_stream = all([
            hasattr(data, '__iter__'),
            not isinstance(data, (basestring, list, tuple, dict))
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
                if data and json is None:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, 'read'):
                        content_type = None
                    else:
                        content_type = 'application/x-www-form-urlencoded'

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if content_type and ('content-type' not in self.headers):
                self.headers['Content-Type'] = content_type

        self.body = body
```
### 9 - requests/compat.py:

Start line: 1, End line: 63

```python
# -*- coding: utf-8 -*-

"""
pythoncompat
"""

from .packages import chardet

import sys

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = (_ver[0] == 2)

#: Python 3.x?
is_py3 = (_ver[0] == 3)

try:
    import simplejson as json
except (ImportError, SyntaxError):
    # simplejson does not support Python 3.2, it throws a SyntaxError
    # because of u'...' Unicode literals.
    import json

# ---------
# Specifics
# ---------

if is_py2:
    from urllib import quote, unquote, quote_plus, unquote_plus, urlencode, getproxies, proxy_bypass
    from urlparse import urlparse, urlunparse, urljoin, urlsplit, urldefrag
    from urllib2 import parse_http_list
    import cookielib
    from Cookie import Morsel
    from StringIO import StringIO
    from .packages.urllib3.packages.ordered_dict import OrderedDict

    builtin_str = str
    bytes = str
    str = unicode
    basestring = basestring
    numeric_types = (int, long, float)

elif is_py3:
    from urllib.parse import urlparse, urlunparse, urljoin, urlsplit, urlencode, quote, unquote, quote_plus, unquote_plus, urldefrag
    from urllib.request import parse_http_list, getproxies, proxy_bypass
    from http import cookiejar as cookielib
    from http.cookies import Morsel
    from io import StringIO
    from collections import OrderedDict

    builtin_str = str
    str = str
    bytes = bytes
    basestring = (str, bytes)
    numeric_types = (int, float)
```
### 10 - requests/packages/urllib3/util/__init__.py:

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
### 13 - requests/models.py:

Start line: 334, End line: 405

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        #: We're unable to blindy call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/kennethreitz/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode('utf8')
        else:
            url = unicode(url) if is_py2 else str(url)

        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ':' in url and not url.lower().startswith('http'):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            scheme, auth, host, port, path, query, fragment = parse_url(url)
        except LocationParseError as e:
            raise InvalidURL(*e.args)

        if not scheme:
            raise MissingSchema("Invalid URL {0!r}: No schema supplied. "
                                "Perhaps you meant http://{0}?".format(
                                    to_native_string(url, 'utf8')))

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
### 17 - requests/models.py:

Start line: 482, End line: 502

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

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
### 25 - requests/models.py:

Start line: 315, End line: 332

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def __repr__(self):
        return '<PreparedRequest [%s]>' % (self.method)

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = self.method.upper()
```
### 29 - requests/models.py:

Start line: 101, End line: 159

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

            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            else:
                fdata = fp.read()

            rf = RequestField(name=k, data=fdata,
                              filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type
```
### 31 - requests/models.py:

Start line: 298, End line: 313

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare(self, method=None, url=None, headers=None, files=None,
                data=None, params=None, auth=None, cookies=None, hooks=None,
                json=None):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)
        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)
```
### 35 - requests/sessions.py:

Start line: 539, End line: 607

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

        checked_urls = set()
        while request.url in self.redirect_cache:
            checked_urls.add(request.url)
            new_url = self.redirect_cache.get(request.url)
            if new_url in checked_urls:
                break
            request.url = new_url

        # Set up variables needed for resolve_redirects and dispatching of hooks
        allow_redirects = kwargs.pop('allow_redirects', True)
        stream = kwargs.get('stream')
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
        gen = self.resolve_redirects(r, request, **kwargs)

        # Resolve redirects if allowed.
        history = [resp for resp in gen] if allow_redirects else []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = history

        if not stream:
            r.content

        return r
```
### 48 - requests/models.py:

Start line: 1, End line: 49

```python
# -*- coding: utf-8 -*-

"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import collections
import datetime

from io import BytesIO, UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict

from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .packages.urllib3.fields import RequestField
from .packages.urllib3.filepost import encode_multipart_formdata
from .packages.urllib3.util import parse_url
from .packages.urllib3.exceptions import (
    DecodeError, ReadTimeoutError, ProtocolError, LocationParseError)
from .exceptions import (
    HTTPError, MissingSchema, InvalidURL, ChunkedEncodingError,
    ContentDecodingError, ConnectionError, StreamConsumedError)
from .utils import (
    guess_filename, get_auth_from_url, requote_uri,
    stream_decode_response_unicode, to_key_val_list, parse_header_links,
    iter_slices, guess_json_utf, super_len, to_native_string)
from .compat import (
    cookielib, urlunparse, urlsplit, urlencode, str, bytes, StringIO,
    is_py2, chardet, json, builtin_str, basestring)
from .status_codes import codes

#: The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes.moved,              # 301
    codes.found,              # 302
    codes.other,              # 303
    codes.temporary_redirect, # 307
    codes.permanent_redirect, # 308
)
DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512

json_dumps = json.dumps
```
### 52 - requests/models.py:

Start line: 264, End line: 296

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
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
```
### 56 - requests/models.py:

Start line: 504, End line: 531

```python
class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand."""

        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)

        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers['Cookie'] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        # hooks can be passed as None to the prepare method and to this
        # method. To prevent iterating over None, simply use an empty list
        # if hooks is False-y
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])
```
### 61 - requests/models.py:

Start line: 752, End line: 788

```python
class Response(object):

    @property
    def text(self):
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
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
### 83 - requests/models.py:

Start line: 52, End line: 99

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
### 101 - requests/models.py:

Start line: 186, End line: 261

```python
class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary is provided, form-encoding will take place.
    :param json: json for the body to attach to the request (if data is not specified).
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
        hooks=None,
        json=None):

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
        self.json = json
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
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p
```
### 109 - requests/sessions.py:

Start line: 346, End line: 384

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
            json=request.json,
            headers=merge_setting(request.headers, self.headers, dict_class=CaseInsensitiveDict),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p
```
### 125 - requests/sessions.py:

Start line: 386, End line: 467

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
        cert=None,
        json=None):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary or bytes to send in the body of the
            :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a (`connect timeout, read
            timeout <user/advanced.html#timeouts>`_) tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol to the URL of
            the proxy.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) if ``True``, the SSL cert will be verified.
            A CA_BUNDLE path can also be provided.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        """

        method = to_native_string(method)

        # Create the Request.
        req = Request(
            method = method.upper(),
            url = url,
            headers = headers,
            files = files,
            data = data or {},
            json = json,
            params = params or {},
            auth = auth,
            cookies = cookies,
            hooks = hooks,
        )
        prep = self.prepare_request(req)

        proxies = proxies or {}

        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )

        # Send the request.
        send_kwargs = {
            'timeout': timeout,
            'allow_redirects': allow_redirects,
        }
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)

        return resp
```
### 138 - requests/models.py:

Start line: 728, End line: 750

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
### 164 - requests/sessions.py:

Start line: 510, End line: 518

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
### 176 - requests/sessions.py:

Start line: 229, End line: 264

```python
class SessionRedirectMixin(object):

    def rebuild_proxies(self, prepared_request, proxies):
        headers = prepared_request.headers
        url = prepared_request.url
        scheme = urlparse(url).scheme
        new_proxies = proxies.copy() if proxies is not None else {}

        if self.trust_env and not should_bypass_proxies(url):
            environ_proxies = get_environ_proxies(url)

            proxy = environ_proxies.get(scheme)

            if proxy:
                new_proxies.setdefault(scheme, environ_proxies[scheme])

        if 'Proxy-Authorization' in headers:
            del headers['Proxy-Authorization']

        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        if username and password:
            headers['Proxy-Authorization'] = _basic_auth_str(username, password)

        return new_proxies
```
### 185 - requests/sessions.py:

Start line: 499, End line: 508

```python
class Session(SessionRedirectMixin):

    def post(self, url, data=None, json=None, **kwargs):
        """Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        """

        return self.request('POST', url, data=data, json=json, **kwargs)
```
