# psf__requests-1657

| **psf/requests** | `43477edc91a8f49de1e9d96117f9cc6d087e71d9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7523 |
| **Any found context length** | 4691 |
| **Avg pos** | 40.0 |
| **Min pos** | 15 |
| **Max pos** | 25 |
| **Top file pos** | 3 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -65,6 +65,22 @@ def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
     return merged_setting
 
 
+def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
+    """
+    Properly merges both requests and session hooks.
+
+    This is necessary because when request_hooks == {'response': []}, the
+    merge breaks Session hooks entirely.
+    """
+    if session_hooks is None or session_hooks.get('response') == []:
+        return request_hooks
+
+    if request_hooks is None or request_hooks.get('response') == []:
+        return session_hooks
+
+    return merge_setting(request_hooks, session_hooks, dict_class)
+
+
 class SessionRedirectMixin(object):
     def resolve_redirects(self, resp, req, stream=False, timeout=None,
                           verify=True, cert=None, proxies=None):
@@ -261,7 +277,7 @@ def prepare_request(self, request):
             params=merge_setting(request.params, self.params),
             auth=merge_setting(auth, self.auth),
             cookies=merged_cookies,
-            hooks=merge_setting(request.hooks, self.hooks),
+            hooks=merge_hooks(request.hooks, self.hooks),
         )
         return p
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/sessions.py | 68 | 68 | 15 | 3 | 4691
| requests/sessions.py | 264 | 264 | 25 | 3 | 7523


## Problem Statement

```
Session hooks broken
Request hooks are being [merged](https://github.com/kennethreitz/requests/blob/master/requests/sessions.py#L264) with session hooks; since both hook dicts have a list as the value, one simply overwrites the other.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 requests/hooks.py | 1 | 46| 188 | 188 | 189 | 
| 2 | 2 requests/models.py | 149 | 170| 161 | 349 | 5167 | 
| 3 | **3 requests/sessions.py** | 1 | 34| 222 | 571 | 9141 | 
| 4 | **3 requests/sessions.py** | 154 | 227| 523 | 1094 | 9141 | 
| 5 | **3 requests/sessions.py** | 37 | 65| 193 | 1287 | 9141 | 
| 6 | **3 requests/sessions.py** | 434 | 494| 488 | 1775 | 9141 | 
| 7 | 4 requests/__init__.py | 1 | 78| 293 | 2068 | 9637 | 
| 8 | 4 requests/models.py | 467 | 483| 132 | 2200 | 9637 | 
| 9 | 5 requests/auth.py | 1 | 55| 310 | 2510 | 11008 | 
| 10 | 5 requests/models.py | 433 | 465| 271 | 2781 | 11008 | 
| 11 | 5 requests/models.py | 1 | 37| 261 | 3042 | 11008 | 
| 12 | 6 requests/cookies.py | 1 | 79| 509 | 3551 | 14276 | 
| 13 | 6 requests/models.py | 278 | 310| 266 | 3817 | 14276 | 
| 14 | **6 requests/sessions.py** | 496 | 532| 257 | 4074 | 14276 | 
| **-> 15 <-** | **6 requests/sessions.py** | 68 | 151| 617 | 4691 | 14276 | 
| 16 | **6 requests/sessions.py** | 405 | 413| 112 | 4803 | 14276 | 
| 17 | **6 requests/sessions.py** | 415 | 432| 185 | 4988 | 14276 | 
| 18 | **6 requests/sessions.py** | 269 | 363| 759 | 5747 | 14276 | 
| 19 | 6 requests/cookies.py | 313 | 330| 160 | 5907 | 14276 | 
| 20 | **6 requests/sessions.py** | 395 | 403| 112 | 6019 | 14276 | 
| 21 | 6 requests/models.py | 372 | 431| 397 | 6416 | 14276 | 
| 22 | 7 requests/exceptions.py | 1 | 64| 286 | 6702 | 14562 | 
| 23 | 8 requests/packages/urllib3/connectionpool.py | 1 | 80| 373 | 7075 | 20248 | 
| 24 | 9 requests/packages/urllib3/_collections.py | 52 | 65| 152 | 7227 | 20873 | 
| **-> 25 <-** | **9 requests/sessions.py** | 229 | 267| 296 | 7523 | 20873 | 
| 26 | 10 requests/adapters.py | 1 | 45| 299 | 7822 | 23604 | 
| 27 | 11 requests/packages/urllib3/request.py | 1 | 57| 377 | 8199 | 24810 | 
| 28 | 11 requests/packages/urllib3/_collections.py | 67 | 95| 174 | 8373 | 24810 | 
| 29 | 11 requests/cookies.py | 174 | 188| 151 | 8524 | 24810 | 
| 30 | 11 requests/auth.py | 58 | 143| 755 | 9279 | 24810 | 
| 31 | 12 requests/utils.py | 1 | 52| 282 | 9561 | 28862 | 
| 32 | 13 requests/packages/urllib3/exceptions.py | 1 | 122| 701 | 10262 | 29616 | 
| 33 | 13 requests/packages/urllib3/connectionpool.py | 534 | 620| 771 | 11033 | 29616 | 
| 34 | **13 requests/sessions.py** | 365 | 393| 252 | 11285 | 29616 | 
| 35 | 14 requests/packages/__init__.py | 1 | 4| 0 | 11285 | 29630 | 
| 36 | 14 requests/cookies.py | 120 | 145| 218 | 11503 | 29630 | 
| 37 | 14 requests/cookies.py | 295 | 311| 227 | 11730 | 29630 | 
| 38 | 14 requests/adapters.py | 245 | 257| 141 | 11871 | 29630 | 
| 39 | 14 requests/auth.py | 145 | 181| 311 | 12182 | 29630 | 
| 40 | 15 requests/packages/urllib3/packages/ordered_dict.py | 45 | 53| 134 | 12316 | 31787 | 
| 41 | 15 requests/packages/urllib3/connectionpool.py | 417 | 448| 203 | 12519 | 31787 | 
| 42 | 15 requests/models.py | 173 | 244| 513 | 13032 | 31787 | 
| 43 | 16 requests/packages/urllib3/poolmanager.py | 1 | 32| 155 | 13187 | 33775 | 
| 44 | 16 requests/cookies.py | 148 | 172| 249 | 13436 | 33775 | 
| 45 | 16 requests/packages/urllib3/poolmanager.py | 35 | 69| 249 | 13685 | 33775 | 
| 46 | 16 requests/cookies.py | 103 | 117| 150 | 13835 | 33775 | 
| 47 | 16 requests/cookies.py | 190 | 280| 758 | 14593 | 33775 | 
| 48 | 17 requests/packages/urllib3/packages/six.py | 106 | 128| 158 | 14751 | 36617 | 
| 49 | 18 requests/status_codes.py | 1 | 89| 899 | 15650 | 37516 | 
| 50 | 18 requests/packages/urllib3/packages/ordered_dict.py | 55 | 90| 293 | 15943 | 37516 | 
| 51 | 19 requests/packages/urllib3/__init__.py | 1 | 38| 180 | 16123 | 37904 | 
| 52 | 20 requests/compat.py | 1 | 116| 805 | 16928 | 38709 | 
| 53 | 20 requests/packages/urllib3/connectionpool.py | 341 | 415| 710 | 17638 | 38709 | 
| 54 | 20 requests/cookies.py | 282 | 293| 155 | 17793 | 38709 | 
| 55 | 21 requests/packages/urllib3/filepost.py | 1 | 44| 183 | 17976 | 39290 | 
| 56 | 21 requests/adapters.py | 48 | 95| 389 | 18365 | 39290 | 
| 57 | 22 requests/packages/urllib3/fields.py | 161 | 178| 172 | 18537 | 40602 | 
| 58 | 23 setup.py | 1 | 58| 364 | 18901 | 40966 | 
| 59 | 23 requests/packages/urllib3/packages/ordered_dict.py | 174 | 261| 665 | 19566 | 40966 | 
| 60 | 24 requests/structures.py | 1 | 34| 167 | 19733 | 41806 | 
| 61 | 24 requests/packages/urllib3/packages/ordered_dict.py | 115 | 141| 200 | 19933 | 41806 | 
| 62 | 24 requests/adapters.py | 213 | 243| 232 | 20165 | 41806 | 
| 63 | 24 requests/packages/urllib3/_collections.py | 1 | 50| 266 | 20431 | 41806 | 
| 64 | 24 requests/models.py | 312 | 370| 429 | 20860 | 41806 | 
| 65 | 24 requests/packages/urllib3/connectionpool.py | 300 | 339| 298 | 21158 | 41806 | 
| 66 | 24 requests/models.py | 596 | 620| 167 | 21325 | 41806 | 
| 67 | 24 requests/packages/urllib3/connectionpool.py | 167 | 249| 695 | 22020 | 41806 | 
| 68 | 24 requests/models.py | 247 | 276| 208 | 22228 | 41806 | 
| 69 | 25 requests/api.py | 102 | 121| 171 | 22399 | 42879 | 
| 70 | 26 requests/packages/urllib3/response.py | 264 | 302| 257 | 22656 | 44992 | 
| 71 | 27 requests/packages/urllib3/util.py | 160 | 174| 137 | 22793 | 49629 | 
| 72 | 27 requests/packages/urllib3/packages/ordered_dict.py | 92 | 113| 178 | 22971 | 49629 | 
| 73 | 28 requests/packages/urllib3/contrib/pyopenssl.py | 1 | 68| 544 | 23515 | 52202 | 
| 74 | 28 requests/packages/urllib3/filepost.py | 47 | 63| 116 | 23631 | 52202 | 
| 75 | 28 requests/utils.py | 177 | 208| 266 | 23897 | 52202 | 
| 76 | 29 requests/packages/urllib3/contrib/ntlmpool.py | 47 | 121| 714 | 24611 | 53222 | 
| 77 | 29 requests/packages/urllib3/connectionpool.py | 142 | 165| 183 | 24794 | 53222 | 
| 78 | 29 requests/utils.py | 236 | 259| 137 | 24931 | 53222 | 
| 79 | 29 requests/packages/urllib3/packages/ordered_dict.py | 1 | 43| 369 | 25300 | 53222 | 
| 80 | 29 requests/packages/urllib3/packages/six.py | 132 | 186| 739 | 26039 | 53222 | 
| 81 | 29 requests/cookies.py | 82 | 100| 134 | 26173 | 53222 | 
| 82 | 30 requests/packages/urllib3/packages/__init__.py | 1 | 5| 0 | 26173 | 53237 | 
| 83 | 30 requests/packages/urllib3/connectionpool.py | 105 | 139| 285 | 26458 | 53237 | 
| 84 | 30 requests/cookies.py | 368 | 392| 230 | 26688 | 53237 | 
| 85 | 30 requests/packages/urllib3/fields.py | 142 | 159| 142 | 26830 | 53237 | 
| 86 | 30 requests/packages/urllib3/response.py | 1 | 50| 237 | 27067 | 53237 | 
| 87 | 30 requests/packages/urllib3/poolmanager.py | 135 | 171| 336 | 27403 | 53237 | 
| 88 | 30 requests/packages/urllib3/contrib/pyopenssl.py | 268 | 288| 121 | 27524 | 53237 | 
| 89 | 30 requests/packages/urllib3/fields.py | 109 | 140| 232 | 27756 | 53237 | 
| 90 | 30 requests/packages/urllib3/connectionpool.py | 83 | 103| 147 | 27903 | 53237 | 
| 91 | 30 requests/cookies.py | 395 | 413| 163 | 28066 | 53237 | 
| 92 | 30 requests/packages/urllib3/poolmanager.py | 243 | 260| 162 | 28228 | 53237 | 
| 93 | 31 requests/packages/charade/gb2312freq.py | 42 | 473| 44 | 28272 | 74105 | 
| 94 | 31 requests/utils.py | 459 | 504| 271 | 28543 | 74105 | 
| 95 | 31 requests/models.py | 89 | 146| 450 | 28993 | 74105 | 
| 96 | 31 requests/packages/urllib3/response.py | 132 | 206| 616 | 29609 | 74105 | 
| 97 | 31 requests/packages/urllib3/poolmanager.py | 174 | 241| 575 | 30184 | 74105 | 
| 98 | 31 requests/packages/urllib3/packages/ordered_dict.py | 143 | 172| 311 | 30495 | 74105 | 
| 99 | 31 requests/adapters.py | 285 | 375| 587 | 31082 | 74105 | 
| 100 | 31 requests/structures.py | 37 | 109| 558 | 31640 | 74105 | 
| 101 | 31 requests/api.py | 47 | 99| 438 | 32078 | 74105 | 
| 102 | 31 requests/models.py | 695 | 734| 248 | 32326 | 74105 | 
| 103 | 31 requests/packages/urllib3/__init__.py | 40 | 59| 145 | 32471 | 74105 | 
| 104 | 31 requests/packages/urllib3/util.py | 1 | 48| 269 | 32740 | 74105 | 
| 105 | 32 requests/packages/charade/big5freq.py | 43 | 926| 52 | 32792 | 120701 | 
| 106 | 33 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 34068 | 121977 | 
| 107 | 33 requests/packages/urllib3/response.py | 53 | 130| 563 | 34631 | 121977 | 
| 108 | 33 requests/packages/urllib3/connectionpool.py | 661 | 687| 216 | 34847 | 121977 | 
| 109 | 33 requests/packages/urllib3/util.py | 471 | 500| 218 | 35065 | 121977 | 
| 110 | 33 requests/packages/urllib3/poolmanager.py | 71 | 95| 190 | 35255 | 121977 | 
| 111 | 34 requests/packages/charade/euckrfreq.py | 41 | 597| 53 | 35308 | 148751 | 
| 112 | 35 requests/packages/charade/compat.py | 21 | 35| 69 | 35377 | 149017 | 
| 113 | 35 requests/models.py | 40 | 87| 294 | 35671 | 149017 | 
| 114 | 35 requests/utils.py | 55 | 88| 250 | 35921 | 149017 | 
| 115 | 35 requests/packages/urllib3/response.py | 231 | 262| 241 | 36162 | 149017 | 
| 116 | 35 requests/packages/urllib3/contrib/pyopenssl.py | 170 | 265| 740 | 36902 | 149017 | 
| 117 | 35 requests/adapters.py | 259 | 283| 209 | 37111 | 149017 | 
| 118 | 35 requests/packages/urllib3/util.py | 176 | 189| 130 | 37241 | 149017 | 
| 119 | 36 docs/conf.py | 1 | 136| 1059 | 38300 | 150917 | 
| 120 | 36 requests/packages/urllib3/fields.py | 55 | 74| 129 | 38429 | 150917 | 
| 121 | 36 requests/packages/urllib3/util.py | 51 | 123| 707 | 39136 | 150917 | 
| 122 | 36 requests/packages/urllib3/packages/six.py | 1 | 103| 512 | 39648 | 150917 | 
| 123 | 36 requests/packages/urllib3/request.py | 59 | 88| 283 | 39931 | 150917 | 
| 124 | 36 requests/api.py | 1 | 44| 423 | 40354 | 150917 | 
| 125 | 36 requests/packages/urllib3/contrib/pyopenssl.py | 316 | 345| 244 | 40598 | 150917 | 
| 126 | 37 requests/packages/charade/charsetprober.py | 29 | 63| 184 | 40782 | 151363 | 
| 127 | 37 requests/packages/urllib3/contrib/pyopenssl.py | 102 | 168| 585 | 41367 | 151363 | 
| 128 | 37 requests/packages/urllib3/connectionpool.py | 251 | 298| 364 | 41731 | 151363 | 
| 129 | 37 requests/adapters.py | 151 | 185| 273 | 42004 | 151363 | 
| 130 | 37 requests/utils.py | 146 | 174| 250 | 42254 | 151363 | 
| 131 | 37 requests/models.py | 559 | 594| 244 | 42498 | 151363 | 
| 132 | 37 requests/packages/urllib3/contrib/ntlmpool.py | 1 | 45| 228 | 42726 | 151363 | 
| 133 | 37 requests/cookies.py | 333 | 365| 233 | 42959 | 151363 | 
| 134 | 37 requests/packages/urllib3/connectionpool.py | 689 | 718| 233 | 43192 | 151363 | 
| 135 | 37 requests/packages/urllib3/fields.py | 76 | 107| 274 | 43466 | 151363 | 
| 136 | 37 requests/packages/urllib3/connectionpool.py | 623 | 659| 362 | 43828 | 151363 | 
| 137 | 37 requests/packages/urllib3/response.py | 208 | 228| 178 | 44006 | 151363 | 
| 138 | 37 requests/packages/urllib3/contrib/pyopenssl.py | 290 | 313| 141 | 44147 | 151363 | 
| 139 | 37 requests/utils.py | 277 | 320| 237 | 44384 | 151363 | 
| 140 | 37 requests/packages/urllib3/request.py | 90 | 143| 508 | 44892 | 151363 | 
| 141 | 38 requests/packages/charade/jisfreq.py | 44 | 570| 53 | 44945 | 179336 | 
| 142 | 38 docs/conf.py | 137 | 244| 615 | 45560 | 179336 | 
| 143 | 38 requests/adapters.py | 97 | 112| 175 | 45735 | 179336 | 
| 144 | 38 requests/packages/urllib3/packages/six.py | 189 | 299| 669 | 46404 | 179336 | 
| 145 | 38 requests/packages/urllib3/util.py | 191 | 213| 196 | 46600 | 179336 | 
| 146 | 38 requests/packages/urllib3/connectionpool.py | 450 | 532| 727 | 47327 | 179336 | 
| 147 | 38 requests/packages/urllib3/util.py | 411 | 468| 195 | 47522 | 179336 | 
| 148 | 38 requests/structures.py | 112 | 129| 114 | 47636 | 179336 | 
| 149 | 38 requests/packages/urllib3/filepost.py | 66 | 102| 233 | 47869 | 179336 | 
| 150 | 38 requests/packages/urllib3/util.py | 265 | 295| 233 | 48102 | 179336 | 
| 151 | 39 requests/packages/charade/euctwfreq.py | 44 | 429| 56 | 48158 | 199826 | 


### Hint

```
Great spot, thanks! We should improve our merging logic here.

I might take a crack at this in an hour or so

Hm. This has always been the behaviour of how per-request hooks work with session hooks but it isn't exactly intuitive. My concern is whether people are relying on this behaviour since the logic in `merge_setting` hasn't really changed in over a year (at least).

I have to wonder if this did the same thing on older versions of requests or if I'm just remembering incorrectly. Either way, the simplest solution would be to not try to special case this inside of `merge_setting` but to use a different function, e.g., `merge_hooks` (or more generally `merge_lists`) to create a merged list of both.

The simplest way to avoid duplication is: `set(session_setting).merge(request_setting)`.

Still I'm not 100% certain this is the expected behaviour and I don't have the time to test it right now. I'll take a peak later tonight.

Let me clarify: session hooks are completely ignored (in version 2.0.0) regardless of whether any per-request hooks were set. They used to work in Requests 1.2.3.

Ah that's a big help.

```

## Patch

```diff
diff --git a/requests/sessions.py b/requests/sessions.py
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -65,6 +65,22 @@ def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
     return merged_setting
 
 
+def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
+    """
+    Properly merges both requests and session hooks.
+
+    This is necessary because when request_hooks == {'response': []}, the
+    merge breaks Session hooks entirely.
+    """
+    if session_hooks is None or session_hooks.get('response') == []:
+        return request_hooks
+
+    if request_hooks is None or request_hooks.get('response') == []:
+        return session_hooks
+
+    return merge_setting(request_hooks, session_hooks, dict_class)
+
+
 class SessionRedirectMixin(object):
     def resolve_redirects(self, resp, req, stream=False, timeout=None,
                           verify=True, cert=None, proxies=None):
@@ -261,7 +277,7 @@ def prepare_request(self, request):
             params=merge_setting(request.params, self.params),
             auth=merge_setting(auth, self.auth),
             cookies=merged_cookies,
-            hooks=merge_setting(request.hooks, self.hooks),
+            hooks=merge_hooks(request.hooks, self.hooks),
         )
         return p
 

```

## Test Patch

```diff
diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -449,6 +449,25 @@ def hook(resp, **kwargs):
 
         requests.Request('GET', HTTPBIN, hooks={'response': hook})
 
+    def test_session_hooks_are_used_with_no_request_hooks(self):
+        hook = lambda x, *args, **kwargs: x
+        s = requests.Session()
+        s.hooks['response'].append(hook)
+        r = requests.Request('GET', HTTPBIN)
+        prep = s.prepare_request(r)
+        assert prep.hooks['response'] != []
+        assert prep.hooks['response'] == [hook]
+
+    def test_session_hooks_are_overriden_by_request_hooks(self):
+        hook1 = lambda x, *args, **kwargs: x
+        hook2 = lambda x, *args, **kwargs: x
+        assert hook1 is not hook2
+        s = requests.Session()
+        s.hooks['response'].append(hook2)
+        r = requests.Request('GET', HTTPBIN, hooks={'response': [hook1]})
+        prep = s.prepare_request(r)
+        assert prep.hooks['response'] == [hook1]
+
     def test_prepared_request_hook(self):
         def hook(resp, **kwargs):
             resp.hook_working = True

```


## Code snippets

### 1 - requests/hooks.py:

Start line: 1, End line: 46

```python
# -*- coding: utf-8 -*-

"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.

"""


HOOKS = ['response']


def default_hooks():
    hooks = {}
    for event in HOOKS:
        hooks[event] = []
    return hooks

# TODO: response is the only one


def dispatch_hook(key, hooks, hook_data, **kwargs):
    """Dispatches a hook dictionary on a given piece of data."""

    hooks = hooks or dict()

    if key in hooks:
        hooks = hooks.get(key)

        if hasattr(hooks, '__call__'):
            hooks = [hooks]

        for hook in hooks:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data

    return hook_data
```
### 2 - requests/models.py:

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
### 3 - requests/sessions.py:

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

from .compat import cookielib, OrderedDict, urljoin, urlparse, urlunparse
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
### 4 - requests/sessions.py:

Start line: 154, End line: 227

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
### 5 - requests/sessions.py:

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
### 6 - requests/sessions.py:

Start line: 434, End line: 494

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
### 7 - requests/__init__.py:

Start line: 1, End line: 78

```python
# -*- coding: utf-8 -*-

#   __
#  /__)  _  _     _   _ _/   _
# / (   (- (/ (/ (- _)  /  _)
#          /

__title__ = 'requests'
__version__ = '2.0.0'
__build__ = 0x020000
__author__ = 'Kenneth Reitz'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright 2013 Kenneth Reitz'

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
### 8 - requests/models.py:

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
### 9 - requests/auth.py:

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
### 10 - requests/models.py:

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
### 14 - requests/sessions.py:

Start line: 496, End line: 532

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
### 15 - requests/sessions.py:

Start line: 68, End line: 151

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
            parsed = urlparse(url)
            parsed = (parsed.scheme.lower(), parsed.netloc, parsed.path,
                      parsed.params, parsed.query, parsed.fragment)
            url = urlunparse(parsed)

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
### 16 - requests/sessions.py:

Start line: 405, End line: 413

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
### 17 - requests/sessions.py:

Start line: 415, End line: 432

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
### 18 - requests/sessions.py:

Start line: 269, End line: 363

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
### 20 - requests/sessions.py:

Start line: 395, End line: 403

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
### 25 - requests/sessions.py:

Start line: 229, End line: 267

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

Start line: 365, End line: 393

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
