# django__django-14151

| **django/django** | `474cc420bf6bc1067e2aaa4b40cf6a08d62096f7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1507 |
| **Any found context length** | 1507 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/middleware/csrf.py b/django/middleware/csrf.py
--- a/django/middleware/csrf.py
+++ b/django/middleware/csrf.py
@@ -298,7 +298,10 @@ def process_view(self, request, callback, callback_args, callback_kwargs):
                 if referer is None:
                     return self._reject(request, REASON_NO_REFERER)
 
-                referer = urlparse(referer)
+                try:
+                    referer = urlparse(referer)
+                except ValueError:
+                    return self._reject(request, REASON_MALFORMED_REFERER)
 
                 # Make sure we have a valid URL for Referer.
                 if '' in (referer.scheme, referer.netloc):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/middleware/csrf.py | 301 | 301 | 2 | 1 | 1507


## Problem Statement

```
CsrfViewMiddleware assumes referer header can be parsed
Description
	
Django's CsrfViewMiddleware assumes that the HTTP referer header is valid when checking it. Specifically, it doesn't handle the case of urlparse() raising a ValueError in this line (e.g. for urls like 'https://['):
​https://github.com/django/django/blob/45814af6197cfd8f4dc72ee43b90ecde305a1d5a/django/middleware/csrf.py#L244

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/middleware/csrf.py** | 227 | 252| 200 | 200 | 3337 | 
| **-> 2 <-** | **1 django/middleware/csrf.py** | 254 | 387| 1307 | 1507 | 3337 | 
| 3 | **1 django/middleware/csrf.py** | 186 | 207| 173 | 1680 | 3337 | 
| 4 | **1 django/middleware/csrf.py** | 209 | 225| 184 | 1864 | 3337 | 
| 5 | **1 django/middleware/csrf.py** | 1 | 45| 362 | 2226 | 3337 | 
| 6 | **1 django/middleware/csrf.py** | 134 | 184| 387 | 2613 | 3337 | 
| 7 | 2 django/views/csrf.py | 15 | 100| 835 | 3448 | 4881 | 
| 8 | 3 django/core/checks/security/csrf.py | 1 | 42| 304 | 3752 | 5343 | 
| 9 | 4 django/core/checks/security/base.py | 222 | 235| 131 | 3883 | 7214 | 
| 10 | 4 django/core/checks/security/csrf.py | 45 | 68| 157 | 4040 | 7214 | 
| 11 | 4 django/views/csrf.py | 101 | 155| 577 | 4617 | 7214 | 
| 12 | 5 django/views/decorators/csrf.py | 1 | 57| 460 | 5077 | 7674 | 
| 13 | 6 django/middleware/common.py | 1 | 32| 247 | 5324 | 9203 | 
| 14 | 7 django/http/request.py | 386 | 409| 185 | 5509 | 14479 | 
| 15 | 7 django/views/csrf.py | 1 | 13| 132 | 5641 | 14479 | 
| 16 | 8 django/middleware/security.py | 1 | 29| 274 | 5915 | 14985 | 
| 17 | 8 django/middleware/common.py | 149 | 175| 254 | 6169 | 14985 | 
| 18 | 8 django/middleware/common.py | 34 | 61| 257 | 6426 | 14985 | 
| 19 | 8 django/middleware/security.py | 31 | 56| 238 | 6664 | 14985 | 
| 20 | 8 django/middleware/common.py | 63 | 75| 136 | 6800 | 14985 | 
| 21 | 9 django/utils/http.py | 317 | 355| 402 | 7202 | 18219 | 
| 22 | 9 django/core/checks/security/base.py | 1 | 69| 631 | 7833 | 18219 | 
| 23 | **9 django/middleware/csrf.py** | 96 | 131| 293 | 8126 | 18219 | 
| 24 | 10 django/views/defaults.py | 1 | 24| 149 | 8275 | 19247 | 
| 25 | **10 django/middleware/csrf.py** | 77 | 93| 195 | 8470 | 19247 | 
| 26 | 10 django/views/defaults.py | 102 | 121| 149 | 8619 | 19247 | 
| 27 | 11 django/contrib/auth/views.py | 1 | 37| 278 | 8897 | 21941 | 
| 28 | **11 django/middleware/csrf.py** | 48 | 57| 111 | 9008 | 21941 | 
| 29 | 11 django/http/request.py | 151 | 158| 111 | 9119 | 21941 | 
| 30 | 12 django/contrib/admindocs/middleware.py | 1 | 31| 254 | 9373 | 22196 | 
| 31 | 13 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 9511 | 22335 | 
| 32 | 14 django/http/response.py | 1 | 26| 176 | 9687 | 26911 | 
| 33 | **14 django/middleware/csrf.py** | 60 | 74| 156 | 9843 | 26911 | 
| 34 | 14 django/middleware/common.py | 100 | 115| 165 | 10008 | 26911 | 
| 35 | 14 django/views/defaults.py | 27 | 78| 403 | 10411 | 26911 | 
| 36 | 14 django/core/checks/security/base.py | 71 | 166| 754 | 11165 | 26911 | 
| 37 | 14 django/http/request.py | 160 | 192| 235 | 11400 | 26911 | 
| 38 | 14 django/http/response.py | 496 | 514| 186 | 11586 | 26911 | 
| 39 | 15 django/http/__init__.py | 1 | 22| 197 | 11783 | 27108 | 
| 40 | 16 django/views/generic/base.py | 188 | 219| 247 | 12030 | 28709 | 
| 41 | 17 django/contrib/auth/management/__init__.py | 89 | 149| 441 | 12471 | 29819 | 
| 42 | 18 django/urls/resolvers.py | 622 | 695| 681 | 13152 | 35413 | 
| 43 | 19 django/views/decorators/clickjacking.py | 22 | 54| 238 | 13390 | 35789 | 
| 44 | 19 django/http/response.py | 69 | 86| 133 | 13523 | 35789 | 
| 45 | 20 django/contrib/auth/middleware.py | 28 | 46| 178 | 13701 | 36794 | 
| 46 | 21 django/utils/decorators.py | 114 | 152| 316 | 14017 | 38193 | 
| 47 | 22 django/urls/conf.py | 57 | 78| 162 | 14179 | 38802 | 
| 48 | 23 django/views/decorators/http.py | 1 | 52| 350 | 14529 | 39756 | 
| 49 | 24 django/contrib/redirects/middleware.py | 1 | 51| 354 | 14883 | 40111 | 
| 50 | 24 django/http/request.py | 1 | 39| 273 | 15156 | 40111 | 
| 51 | 24 django/views/defaults.py | 124 | 151| 198 | 15354 | 40111 | 
| 52 | 24 django/contrib/auth/middleware.py | 48 | 84| 360 | 15714 | 40111 | 
| 53 | 25 django/core/validators.py | 1 | 16| 127 | 15841 | 44656 | 
| 54 | 26 django/contrib/sites/middleware.py | 1 | 13| 0 | 15841 | 44715 | 
| 55 | 27 django/middleware/clickjacking.py | 1 | 48| 364 | 16205 | 45079 | 
| 56 | 28 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 16352 | 45227 | 
| 57 | 28 django/urls/resolvers.py | 417 | 445| 288 | 16640 | 45227 | 
| 58 | 28 django/middleware/common.py | 77 | 98| 227 | 16867 | 45227 | 
| 59 | 29 django/urls/base.py | 27 | 86| 438 | 17305 | 46410 | 
| 60 | 29 django/http/request.py | 98 | 115| 183 | 17488 | 46410 | 
| 61 | 29 django/urls/resolvers.py | 113 | 143| 232 | 17720 | 46410 | 
| 62 | 29 django/views/decorators/clickjacking.py | 1 | 19| 138 | 17858 | 46410 | 
| 63 | 29 django/urls/resolvers.py | 610 | 620| 120 | 17978 | 46410 | 
| 64 | 29 django/urls/resolvers.py | 275 | 291| 160 | 18138 | 46410 | 
| 65 | 30 django/http/cookie.py | 1 | 24| 156 | 18294 | 46567 | 
| 66 | 30 django/utils/http.py | 1 | 38| 467 | 18761 | 46567 | 
| 67 | 31 django/core/servers/basehttp.py | 158 | 177| 170 | 18931 | 48296 | 
| 68 | 31 django/contrib/auth/middleware.py | 114 | 125| 107 | 19038 | 48296 | 
| 69 | 31 django/http/response.py | 551 | 576| 159 | 19197 | 48296 | 
| 70 | 31 django/http/request.py | 653 | 673| 184 | 19381 | 48296 | 
| 71 | 32 django/contrib/gis/views.py | 1 | 21| 155 | 19536 | 48451 | 
| 72 | 32 django/urls/base.py | 1 | 24| 170 | 19706 | 48451 | 
| 73 | 32 django/http/response.py | 517 | 548| 153 | 19859 | 48451 | 
| 74 | 33 django/utils/cache.py | 303 | 323| 211 | 20070 | 52181 | 
| 75 | 33 django/core/validators.py | 64 | 99| 551 | 20621 | 52181 | 
| 76 | 34 django/contrib/auth/urls.py | 1 | 21| 224 | 20845 | 52405 | 
| 77 | 34 django/utils/http.py | 239 | 265| 268 | 21113 | 52405 | 
| 78 | 34 django/urls/resolvers.py | 399 | 415| 164 | 21277 | 52405 | 
| 79 | 35 django/middleware/cache.py | 131 | 157| 252 | 21529 | 53993 | 
| 80 | 35 django/http/request.py | 352 | 383| 255 | 21784 | 53993 | 
| 81 | 36 django/conf/global_settings.py | 496 | 646| 926 | 22710 | 59676 | 
| 82 | 36 django/core/checks/security/base.py | 169 | 184| 118 | 22828 | 59676 | 
| 83 | 36 django/utils/http.py | 41 | 79| 301 | 23129 | 59676 | 
| 84 | 37 django/middleware/http.py | 1 | 42| 335 | 23464 | 60011 | 
| 85 | 38 django/views/debug.py | 156 | 179| 177 | 23641 | 64602 | 
| 86 | 39 django/core/handlers/wsgi.py | 64 | 119| 486 | 24127 | 66361 | 
| 87 | 40 django/contrib/flatpages/views.py | 1 | 45| 399 | 24526 | 66951 | 
| 88 | 40 django/conf/global_settings.py | 151 | 266| 859 | 25385 | 66951 | 
| 89 | 40 django/middleware/common.py | 118 | 147| 277 | 25662 | 66951 | 
| 90 | 41 django/urls/exceptions.py | 1 | 10| 0 | 25662 | 66976 | 
| 91 | 41 django/views/debug.py | 181 | 193| 143 | 25805 | 66976 | 
| 92 | 41 django/core/validators.py | 101 | 153| 487 | 26292 | 66976 | 
| 93 | 41 django/urls/resolvers.py | 550 | 586| 347 | 26639 | 66976 | 
| 94 | 42 django/conf/urls/__init__.py | 1 | 10| 0 | 26639 | 67041 | 
| 95 | 43 django/contrib/sitemaps/views.py | 1 | 19| 132 | 26771 | 67817 | 
| 96 | 44 django/utils/html.py | 200 | 232| 307 | 27078 | 70919 | 
| 97 | 44 django/http/request.py | 230 | 299| 504 | 27582 | 70919 | 
| 98 | 44 django/http/request.py | 580 | 611| 241 | 27823 | 70919 | 
| 99 | 44 django/http/request.py | 301 | 321| 189 | 28012 | 70919 | 
| 100 | 45 django/contrib/redirects/admin.py | 1 | 11| 0 | 28012 | 70987 | 
| 101 | 45 django/urls/resolvers.py | 588 | 608| 177 | 28189 | 70987 | 
| 102 | 46 django/template/defaulttags.py | 1259 | 1323| 504 | 28693 | 81611 | 
| 103 | 47 django/views/generic/__init__.py | 1 | 23| 189 | 28882 | 81801 | 
| 104 | 48 django/contrib/auth/__init__.py | 1 | 38| 241 | 29123 | 83382 | 
| 105 | 49 django/middleware/locale.py | 28 | 62| 332 | 29455 | 83947 | 
| 106 | 49 django/core/validators.py | 253 | 297| 348 | 29803 | 83947 | 
| 107 | 50 django/contrib/admin/views/autocomplete.py | 48 | 103| 421 | 30224 | 84715 | 
| 108 | 51 django/urls/__init__.py | 1 | 24| 239 | 30463 | 84954 | 
| 109 | 51 django/core/servers/basehttp.py | 121 | 156| 280 | 30743 | 84954 | 
| 110 | 52 django/forms/fields.py | 660 | 699| 293 | 31036 | 94298 | 
| 111 | 52 django/views/debug.py | 142 | 154| 148 | 31184 | 94298 | 
| 112 | 52 django/urls/resolvers.py | 68 | 87| 168 | 31352 | 94298 | 
| 113 | 52 django/views/defaults.py | 81 | 99| 129 | 31481 | 94298 | 
| 114 | 53 django/contrib/auth/mixins.py | 44 | 71| 235 | 31716 | 95162 | 
| 115 | 54 django/contrib/redirects/apps.py | 1 | 9| 0 | 31716 | 95213 | 
| 116 | 55 django/contrib/auth/admin.py | 1 | 22| 188 | 31904 | 96939 | 
| 117 | 56 django/contrib/flatpages/urls.py | 1 | 7| 0 | 31904 | 96977 | 
| 118 | 57 django/core/handlers/base.py | 320 | 351| 212 | 32116 | 99584 | 
| 119 | 58 django/contrib/admin/sites.py | 221 | 240| 221 | 32337 | 103955 | 
| 120 | 59 django/core/checks/security/sessions.py | 1 | 98| 572 | 32909 | 104528 | 
| 121 | 60 django/views/decorators/debug.py | 77 | 92| 132 | 33041 | 105117 | 
| 122 | 60 django/contrib/admin/sites.py | 416 | 431| 129 | 33170 | 105117 | 
| 123 | 61 django/contrib/admindocs/urls.py | 1 | 51| 307 | 33477 | 105424 | 
| 124 | 61 django/utils/cache.py | 135 | 150| 190 | 33667 | 105424 | 
| 125 | 62 django/template/defaultfilters.py | 323 | 337| 111 | 33778 | 111654 | 
| 126 | 62 django/http/request.py | 117 | 135| 187 | 33965 | 111654 | 
| 127 | 62 django/urls/resolvers.py | 1 | 29| 209 | 34174 | 111654 | 
| 128 | 62 django/contrib/auth/middleware.py | 1 | 25| 182 | 34356 | 111654 | 
| 129 | 62 django/contrib/auth/middleware.py | 86 | 111| 192 | 34548 | 111654 | 
| 130 | 63 django/core/checks/urls.py | 71 | 111| 264 | 34812 | 112355 | 
| 131 | 63 django/contrib/auth/views.py | 66 | 108| 319 | 35131 | 112355 | 
| 132 | 64 django/core/handlers/exception.py | 54 | 122| 557 | 35688 | 113423 | 
| 133 | 65 django/contrib/auth/decorators.py | 1 | 35| 313 | 36001 | 114010 | 
| 134 | 66 django/contrib/messages/views.py | 1 | 19| 0 | 36001 | 114106 | 
| 135 | 66 django/utils/http.py | 195 | 206| 137 | 36138 | 114106 | 
| 136 | 67 django/contrib/sessions/middleware.py | 1 | 76| 588 | 36726 | 114695 | 
| 137 | 67 django/middleware/cache.py | 1 | 52| 431 | 37157 | 114695 | 
| 138 | 68 django/shortcuts.py | 102 | 141| 280 | 37437 | 115792 | 
| 139 | 68 django/utils/http.py | 136 | 158| 166 | 37603 | 115792 | 
| 140 | 69 django/db/models/fields/__init__.py | 2310 | 2330| 163 | 37766 | 134210 | 
| 141 | 69 django/urls/resolvers.py | 251 | 273| 173 | 37939 | 134210 | 
| 142 | 70 django/contrib/gis/db/backends/postgis/const.py | 1 | 53| 620 | 38559 | 134831 | 
| 143 | 71 django/http/multipartparser.py | 654 | 699| 399 | 38958 | 139916 | 
| 144 | 72 django/contrib/admin/options.py | 1 | 97| 762 | 39720 | 158495 | 
| 145 | 73 django/contrib/syndication/views.py | 1 | 26| 220 | 39940 | 160235 | 
| 146 | 73 django/core/handlers/base.py | 277 | 292| 125 | 40065 | 160235 | 
| 147 | 73 django/core/checks/security/base.py | 202 | 219| 127 | 40192 | 160235 | 
| 148 | 73 django/urls/resolvers.py | 328 | 376| 367 | 40559 | 160235 | 
| 149 | 74 django/contrib/admindocs/utils.py | 1 | 25| 151 | 40710 | 162140 | 
| 150 | 74 django/utils/html.py | 306 | 349| 438 | 41148 | 162140 | 
| 151 | 74 django/utils/cache.py | 1 | 34| 274 | 41422 | 162140 | 
| 152 | 74 django/utils/cache.py | 244 | 275| 257 | 41679 | 162140 | 
| 153 | 74 django/http/request.py | 42 | 96| 394 | 42073 | 162140 | 
| 154 | 75 django/views/decorators/cache.py | 45 | 60| 123 | 42196 | 162599 | 
| 155 | 76 django/contrib/admindocs/views.py | 1 | 30| 225 | 42421 | 165927 | 
| 156 | 76 django/http/response.py | 238 | 258| 237 | 42658 | 165927 | 
| 157 | 76 django/core/checks/urls.py | 1 | 27| 142 | 42800 | 165927 | 
| 158 | 76 django/urls/resolvers.py | 447 | 506| 548 | 43348 | 165927 | 
| 159 | 76 django/contrib/auth/views.py | 228 | 248| 163 | 43511 | 165927 | 
| 160 | 76 django/contrib/admin/sites.py | 1 | 35| 225 | 43736 | 165927 | 
| 161 | 76 django/views/debug.py | 1 | 47| 296 | 44032 | 165927 | 
| 162 | 76 django/utils/cache.py | 347 | 366| 190 | 44222 | 165927 | 
| 163 | 76 django/urls/base.py | 89 | 155| 383 | 44605 | 165927 | 
| 164 | 77 django/views/__init__.py | 1 | 4| 0 | 44605 | 165942 | 
| 165 | 77 django/contrib/admindocs/utils.py | 86 | 112| 175 | 44780 | 165942 | 
| 166 | 78 django/contrib/admin/views/decorators.py | 1 | 19| 135 | 44915 | 166078 | 
| 167 | 79 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 44915 | 166175 | 
| 168 | 79 django/urls/resolvers.py | 32 | 65| 350 | 45265 | 166175 | 
| 169 | 79 django/conf/global_settings.py | 401 | 495| 782 | 46047 | 166175 | 
| 170 | 80 django/contrib/messages/context_processors.py | 1 | 14| 0 | 46047 | 166246 | 
| 171 | 80 django/contrib/auth/views.py | 290 | 331| 314 | 46361 | 166246 | 
| 172 | 81 django/utils/deprecation.py | 79 | 129| 372 | 46733 | 167285 | 
| 173 | 81 django/contrib/auth/views.py | 251 | 288| 348 | 47081 | 167285 | 
| 174 | 81 django/contrib/sitemaps/views.py | 48 | 93| 392 | 47473 | 167285 | 
| 175 | 81 django/core/handlers/base.py | 294 | 318| 218 | 47691 | 167285 | 
| 176 | 81 django/utils/decorators.py | 89 | 111| 152 | 47843 | 167285 | 
| 177 | 81 django/utils/http.py | 268 | 283| 219 | 48062 | 167285 | 
| 178 | 81 django/core/handlers/base.py | 160 | 210| 379 | 48441 | 167285 | 
| 179 | 81 django/contrib/auth/views.py | 212 | 226| 133 | 48574 | 167285 | 
| 180 | 82 django/core/asgi.py | 1 | 14| 0 | 48574 | 167370 | 
| 181 | 82 django/http/response.py | 350 | 385| 212 | 48786 | 167370 | 
| 182 | 82 django/template/defaulttags.py | 51 | 68| 152 | 48938 | 167370 | 
| 183 | 82 django/urls/resolvers.py | 508 | 548| 282 | 49220 | 167370 | 
| 184 | 83 django/core/mail/message.py | 1 | 52| 346 | 49566 | 171021 | 
| 185 | 83 django/views/decorators/debug.py | 47 | 75| 199 | 49765 | 171021 | 
| 186 | 83 django/contrib/admindocs/views.py | 406 | 419| 127 | 49892 | 171021 | 
| 187 | 84 django/middleware/gzip.py | 1 | 52| 419 | 50311 | 171441 | 
| 188 | 84 django/middleware/locale.py | 1 | 26| 239 | 50550 | 171441 | 
| 189 | 84 django/core/mail/message.py | 177 | 185| 115 | 50665 | 171441 | 
| 190 | 84 django/contrib/auth/decorators.py | 38 | 74| 273 | 50938 | 171441 | 
| 191 | 85 django/contrib/sites/checks.py | 1 | 14| 0 | 50938 | 171520 | 
| 192 | 85 django/core/handlers/wsgi.py | 122 | 156| 326 | 51264 | 171520 | 
| 193 | 85 django/utils/cache.py | 369 | 415| 521 | 51785 | 171520 | 
| 194 | 86 django/contrib/auth/backends.py | 163 | 181| 146 | 51931 | 173282 | 
| 195 | 87 django/contrib/staticfiles/handlers.py | 64 | 77| 127 | 52058 | 174013 | 
| 196 | 87 django/views/decorators/cache.py | 1 | 25| 215 | 52273 | 174013 | 
| 197 | 87 django/template/defaultfilters.py | 340 | 424| 499 | 52772 | 174013 | 
| 198 | 88 django/utils/encoding.py | 186 | 200| 207 | 52979 | 176229 | 
| 199 | 89 django/db/utils.py | 255 | 297| 322 | 53301 | 178236 | 
| 200 | 89 django/utils/cache.py | 196 | 214| 184 | 53485 | 178236 | 
| 201 | 89 django/contrib/admindocs/views.py | 378 | 403| 186 | 53671 | 178236 | 
| 202 | 90 django/core/mail/backends/dummy.py | 1 | 11| 0 | 53671 | 178279 | 
| 203 | 90 django/utils/http.py | 96 | 133| 374 | 54045 | 178279 | 
| 204 | 90 django/middleware/cache.py | 117 | 129| 125 | 54170 | 178279 | 
| 205 | 90 django/http/response.py | 260 | 276| 181 | 54351 | 178279 | 
| 206 | 90 django/http/request.py | 137 | 149| 133 | 54484 | 178279 | 
| 207 | 90 django/core/handlers/base.py | 212 | 275| 480 | 54964 | 178279 | 
| 208 | 91 django/views/generic/edit.py | 1 | 67| 479 | 55443 | 179995 | 
| 209 | 92 django/views/decorators/gzip.py | 1 | 6| 0 | 55443 | 180046 | 
| 210 | 92 django/views/decorators/http.py | 55 | 76| 272 | 55715 | 180046 | 
| 211 | 92 django/utils/cache.py | 153 | 193| 453 | 56168 | 180046 | 
| 212 | 93 django/core/checks/model_checks.py | 178 | 211| 332 | 56500 | 181831 | 
| 213 | 93 django/utils/html.py | 291 | 304| 154 | 56654 | 181831 | 
| 214 | 94 django/contrib/admin/tests.py | 1 | 36| 265 | 56919 | 183309 | 


### Hint

```
Should the response in this scenario be something like this line? Or would a different response reason make more sense ​https://github.com/django/django/blob/45814af6197cfd8f4dc72ee43b90ecde305a1d5a/django/middleware/csrf.py#L248
Replying to AdamDonna: Should the response in this scenario be something like this line? Or would a different response reason make more sense ​https://github.com/django/django/blob/45814af6197cfd8f4dc72ee43b90ecde305a1d5a/django/middleware/csrf.py#L248 Yes, we should reject immediately.
```

## Patch

```diff
diff --git a/django/middleware/csrf.py b/django/middleware/csrf.py
--- a/django/middleware/csrf.py
+++ b/django/middleware/csrf.py
@@ -298,7 +298,10 @@ def process_view(self, request, callback, callback_args, callback_kwargs):
                 if referer is None:
                     return self._reject(request, REASON_NO_REFERER)
 
-                referer = urlparse(referer)
+                try:
+                    referer = urlparse(referer)
+                except ValueError:
+                    return self._reject(request, REASON_MALFORMED_REFERER)
 
                 # Make sure we have a valid URL for Referer.
                 if '' in (referer.scheme, referer.netloc):

```

## Test Patch

```diff
diff --git a/tests/csrf_tests/tests.py b/tests/csrf_tests/tests.py
--- a/tests/csrf_tests/tests.py
+++ b/tests/csrf_tests/tests.py
@@ -353,6 +353,12 @@ def test_https_malformed_referer(self):
         req.META['HTTP_REFERER'] = 'https://'
         response = mw.process_view(req, post_form_view, (), {})
         self.assertContains(response, malformed_referer_msg, status_code=403)
+        # Invalid URL
+        # >>> urlparse('https://[')
+        # ValueError: Invalid IPv6 URL
+        req.META['HTTP_REFERER'] = 'https://['
+        response = mw.process_view(req, post_form_view, (), {})
+        self.assertContains(response, malformed_referer_msg, status_code=403)
 
     @override_settings(ALLOWED_HOSTS=['www.example.com'])
     def test_https_good_referer(self):

```


## Code snippets

### 1 - django/middleware/csrf.py:

Start line: 227, End line: 252

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _origin_verified(self, request):
        request_origin = request.META['HTTP_ORIGIN']
        good_origin = '%s://%s' % (
            'https' if request.is_secure() else 'http',
            request.get_host(),
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

    def process_request(self, request):
        csrf_token = self._get_token(request)
        if csrf_token is not None:
            # Use same token next time.
            request.META['CSRF_COOKIE'] = csrf_token
```
### 2 - django/middleware/csrf.py:

Start line: 254, End line: 387

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
        if request.method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            if getattr(request, '_dont_enforce_csrf_checks', False):
                # Mechanism to turn off CSRF checks for test suite.
                # It comes after the creation of CSRF cookies, so that
                # everything else continues to work exactly the same
                # (e.g. cookies are sent, etc.), but before any
                # branches that call reject().
                return self._accept(request)

            # Reject the request if the Origin header doesn't match an allowed
            # value.
            if 'HTTP_ORIGIN' in request.META:
                if not self._origin_verified(request):
                    return self._reject(request, REASON_BAD_ORIGIN % request.META['HTTP_ORIGIN'])
            elif request.is_secure():
                # If the Origin header wasn't provided, reject HTTPS requests
                # if the Referer header doesn't match an allowed value.
                #
                # Suppose user visits http://example.com/
                # An active network attacker (man-in-the-middle, MITM) sends a
                # POST form that targets https://example.com/detonate-bomb/ and
                # submits it via JavaScript.
                #
                # The attacker will need to provide a CSRF cookie and token, but
                # that's no problem for a MITM and the session-independent
                # secret we're using. So the MITM can circumvent the CSRF
                # protection. This is true for any HTTP connection, but anyone
                # using HTTPS expects better! For this reason, for
                # https://example.com/ we need additional protection that treats
                # http://example.com/ as completely untrusted. Under HTTPS,
                # Barth et al. found that the Referer header is missing for
                # same-domain requests in only about 0.2% of cases or less, so
                # we can use strict Referer checking.
                referer = request.META.get('HTTP_REFERER')
                if referer is None:
                    return self._reject(request, REASON_NO_REFERER)

                referer = urlparse(referer)

                # Make sure we have a valid URL for Referer.
                if '' in (referer.scheme, referer.netloc):
                    return self._reject(request, REASON_MALFORMED_REFERER)

                # Ensure that our Referer is also secure.
                if referer.scheme != 'https':
                    return self._reject(request, REASON_INSECURE_REFERER)

                # If there isn't a CSRF_COOKIE_DOMAIN, require an exact match
                # match on host:port. If not, obey the cookie rules (or those
                # for the session cookie, if CSRF_USE_SESSIONS).
                good_referer = (
                    settings.SESSION_COOKIE_DOMAIN
                    if settings.CSRF_USE_SESSIONS
                    else settings.CSRF_COOKIE_DOMAIN
                )
                if good_referer is not None:
                    server_port = request.get_port()
                    if server_port not in ('443', '80'):
                        good_referer = '%s:%s' % (good_referer, server_port)
                else:
                    try:
                        # request.get_host() includes the port.
                        good_referer = request.get_host()
                    except DisallowedHost:
                        pass

                # Create a list of all acceptable HTTP referers, including the
                # current host if it's permitted by ALLOWED_HOSTS.
                good_hosts = list(self.csrf_trusted_origins_hosts)
                if good_referer is not None:
                    good_hosts.append(good_referer)

                if not any(is_same_domain(referer.netloc, host) for host in good_hosts):
                    reason = REASON_BAD_REFERER % referer.geturl()
                    return self._reject(request, reason)

            # Access csrf_token via self._get_token() as rotate_token() may
            # have been called by an authentication middleware during the
            # process_request() phase.
            csrf_token = self._get_token(request)
            if csrf_token is None:
                # No CSRF cookie. For POST requests, we insist on a CSRF cookie,
                # and in this way we can avoid all CSRF attacks, including login
                # CSRF.
                return self._reject(request, REASON_NO_CSRF_COOKIE)

            # Check non-cookie token for match.
            request_csrf_token = ""
            if request.method == "POST":
                try:
                    request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
                except OSError:
                    # Handle a broken connection before we've completed reading
                    # the POST data. process_view shouldn't raise any
                    # exceptions, so we'll ignore and serve the user a 403
                    # (assuming they're still listening, which they probably
                    # aren't because of the error).
                    pass

            if request_csrf_token == "":
                # Fall back to X-CSRFToken, to make things easier for AJAX,
                # and possible for PUT/DELETE.
                request_csrf_token = request.META.get(settings.CSRF_HEADER_NAME, '')

            request_csrf_token = _sanitize_token(request_csrf_token)
            if not _compare_masked_tokens(request_csrf_token, csrf_token):
                return self._reject(request, REASON_BAD_TOKEN)

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
### 3 - django/middleware/csrf.py:

Start line: 186, End line: 207

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

            csrf_token = _sanitize_token(cookie_token)
            if csrf_token != cookie_token:
                # Cookie token needed to be replaced;
                # the cookie needs to be reset.
                request.csrf_cookie_needs_reset = True
            return csrf_token
```
### 4 - django/middleware/csrf.py:

Start line: 209, End line: 225

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

Start line: 1, End line: 45

```python
"""
Cross Site Request Forgery Middleware.

This module provides a middleware that implements protection
against request forgeries from other sites.
"""
import logging
import re
import string
from collections import defaultdict
from urllib.parse import urlparse

from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response

logger = logging.getLogger('django.security.csrf')

REASON_BAD_ORIGIN = "Origin checking failed - %s does not match any trusted origins."
REASON_NO_REFERER = "Referer checking failed - no Referer."
REASON_BAD_REFERER = "Referer checking failed - %s does not match any trusted origins."
REASON_NO_CSRF_COOKIE = "CSRF cookie not set."
REASON_BAD_TOKEN = "CSRF token missing or incorrect."
REASON_MALFORMED_REFERER = "Referer checking failed - Referer is malformed."
REASON_INSECURE_REFERER = "Referer checking failed - Referer is insecure while host is secure."

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
### 6 - django/middleware/csrf.py:

Start line: 134, End line: 184

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
### 7 - django/views/csrf.py:

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
### 8 - django/core/checks/security/csrf.py:

Start line: 1, End line: 42

```python
import inspect

from django.conf import settings

from .. import Error, Tags, Warning, register

W003 = Warning(
    "You don't appear to be using Django's built-in "
    "cross-site request forgery protection via the middleware "
    "('django.middleware.csrf.CsrfViewMiddleware' is not in your "
    "MIDDLEWARE). Enabling the middleware is the safest approach "
    "to ensure you don't leave any holes.",
    id='security.W003',
)

W016 = Warning(
    "You have 'django.middleware.csrf.CsrfViewMiddleware' in your "
    "MIDDLEWARE, but you have not set CSRF_COOKIE_SECURE to True. "
    "Using a secure-only CSRF cookie makes it more difficult for network "
    "traffic sniffers to steal the CSRF token.",
    id='security.W016',
)


def _csrf_middleware():
    return 'django.middleware.csrf.CsrfViewMiddleware' in settings.MIDDLEWARE


@register(Tags.security, deploy=True)
def check_csrf_middleware(app_configs, **kwargs):
    passed_check = _csrf_middleware()
    return [] if passed_check else [W003]


@register(Tags.security, deploy=True)
def check_csrf_cookie_secure(app_configs, **kwargs):
    passed_check = (
        settings.CSRF_USE_SESSIONS or
        not _csrf_middleware() or
        settings.CSRF_COOKIE_SECURE
    )
    return [] if passed_check else [W016]
```
### 9 - django/core/checks/security/base.py:

Start line: 222, End line: 235

```python
@register(Tags.security, deploy=True)
def check_referrer_policy(app_configs, **kwargs):
    if _security_middleware():
        if settings.SECURE_REFERRER_POLICY is None:
            return [W022]
        # Support a comma-separated string or iterable of values to allow fallback.
        if isinstance(settings.SECURE_REFERRER_POLICY, str):
            values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(',')}
        else:
            values = set(settings.SECURE_REFERRER_POLICY)
        if not values <= REFERRER_POLICY_VALUES:
            return [E023]
    return []
```
### 10 - django/core/checks/security/csrf.py:

Start line: 45, End line: 68

```python
@register(Tags.security)
def check_csrf_failure_view(app_configs, **kwargs):
    from django.middleware.csrf import _get_failure_view

    errors = []
    try:
        view = _get_failure_view()
    except ImportError:
        msg = (
            "The CSRF failure view '%s' could not be imported." %
            settings.CSRF_FAILURE_VIEW
        )
        errors.append(Error(msg, id='security.E102'))
    else:
        try:
            inspect.signature(view).bind(None, reason=None)
        except TypeError:
            msg = (
                "The CSRF failure view '%s' does not take the correct number of arguments." %
                settings.CSRF_FAILURE_VIEW
            )
            errors.append(Error(msg, id='security.E101'))
    return errors
```
### 23 - django/middleware/csrf.py:

Start line: 96, End line: 131

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


def _sanitize_token(token):
    # Allow only ASCII alphanumerics
    if re.search('[^a-zA-Z0-9]', token):
        return _get_new_csrf_token()
    elif len(token) == CSRF_TOKEN_LENGTH:
        return token
    elif len(token) == CSRF_SECRET_LENGTH:
        # Older Django versions set cookies to values of CSRF_SECRET_LENGTH
        # alphanumeric characters. For backwards compatibility, accept
        # such values as unmasked secrets.
        # It's easier to mask here and be consistent later, rather than add
        # different code paths in the checks, although that might be a tad more
        # efficient.
        return _mask_cipher_secret(token)
    return _get_new_csrf_token()


def _compare_masked_tokens(request_csrf_token, csrf_token):
    # Assume both arguments are sanitized -- that is, strings of
    # length CSRF_TOKEN_LENGTH, all CSRF_ALLOWED_CHARS.
    return constant_time_compare(
        _unmask_cipher_token(request_csrf_token),
        _unmask_cipher_token(csrf_token),
    )
```
### 25 - django/middleware/csrf.py:

Start line: 77, End line: 93

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
### 28 - django/middleware/csrf.py:

Start line: 48, End line: 57

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
### 33 - django/middleware/csrf.py:

Start line: 60, End line: 74

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
