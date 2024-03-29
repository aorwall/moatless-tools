# django__django-11165

| **django/django** | `af5ec222ccd24e81f9fec6c34836a4e503e7ccf7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 151 |
| **Any found context length** | 151 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/http/request.py b/django/http/request.py
--- a/django/http/request.py
+++ b/django/http/request.py
@@ -369,6 +369,10 @@ def __init__(self, environ):
                 headers[name] = value
         super().__init__(headers)
 
+    def __getitem__(self, key):
+        """Allow header lookup using underscores in place of hyphens."""
+        return super().__getitem__(key.replace('_', '-'))
+
     @classmethod
     def parse_header_name(cls, header):
         if header.startswith(cls.HTTP_PREFIX):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/http/request.py | 372 | 372 | 1 | 1 | 151


## Problem Statement

```
New HTTPRequest.headers not usable in templates because of hyphens
Description
	
With the release of 2.2, I took the opportunity from the new ​HTTPRequest.headers object to clean up old code using e.g. request.META['HTTP_X_REAL_IP'] to request.headers['X-Real-IP'].
However, this new approach does not work with templates, as ​variable lookups cannot use hyphens.
Could the object contain a parallel set of keys in underscored variables? e.g. request.headers['foo-bar'] is also available in request.headers['foo_bar'] and can be looked up with {{ request.headers.foo_bar }} in a template?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/http/request.py** | 359 | 378| 151 | 151 | 4686 | 
| 2 | 2 django/http/response.py | 134 | 155| 176 | 327 | 8944 | 
| 3 | 3 django/utils/cache.py | 342 | 388| 531 | 858 | 12493 | 
| 4 | 3 django/utils/cache.py | 256 | 273| 206 | 1064 | 12493 | 
| 5 | 3 django/http/response.py | 107 | 132| 270 | 1334 | 12493 | 
| 6 | 3 django/http/response.py | 1 | 25| 144 | 1478 | 12493 | 
| 7 | 3 django/http/response.py | 203 | 219| 193 | 1671 | 12493 | 
| 8 | 3 django/http/response.py | 28 | 105| 614 | 2285 | 12493 | 
| 9 | **3 django/http/request.py** | 72 | 89| 183 | 2468 | 12493 | 
| 10 | **3 django/http/request.py** | 125 | 132| 111 | 2579 | 12493 | 
| 11 | 4 django/views/decorators/http.py | 1 | 52| 350 | 2929 | 13445 | 
| 12 | 5 django/views/decorators/vary.py | 1 | 42| 232 | 3161 | 13678 | 
| 13 | 6 django/utils/http.py | 1 | 72| 693 | 3854 | 17631 | 
| 14 | 6 django/utils/cache.py | 299 | 317| 214 | 4068 | 17631 | 
| 15 | 6 django/utils/cache.py | 116 | 131| 188 | 4256 | 17631 | 
| 16 | 6 django/utils/cache.py | 276 | 296| 217 | 4473 | 17631 | 
| 17 | 7 django/core/servers/basehttp.py | 156 | 175| 170 | 4643 | 19343 | 
| 18 | **7 django/http/request.py** | 168 | 199| 391 | 5034 | 19343 | 
| 19 | 8 django/core/mail/message.py | 55 | 71| 171 | 5205 | 23068 | 
| 20 | **8 django/http/request.py** | 1 | 35| 245 | 5450 | 23068 | 
| 21 | 9 django/utils/html.py | 1 | 71| 606 | 6056 | 26171 | 
| 22 | 10 django/contrib/admindocs/middleware.py | 1 | 31| 253 | 6309 | 26425 | 
| 23 | **10 django/http/request.py** | 91 | 109| 187 | 6496 | 26425 | 
| 24 | 10 django/utils/cache.py | 1 | 34| 269 | 6765 | 26425 | 
| 25 | 11 django/views/defaults.py | 83 | 99| 138 | 6903 | 27369 | 
| 26 | 12 django/template/response.py | 60 | 145| 587 | 7490 | 28448 | 
| 27 | 12 django/template/response.py | 1 | 43| 383 | 7873 | 28448 | 
| 28 | 13 django/template/base.py | 726 | 789| 531 | 8404 | 36309 | 
| 29 | **13 django/http/request.py** | 274 | 294| 189 | 8593 | 36309 | 
| 30 | 14 django/views/csrf.py | 15 | 100| 835 | 9428 | 37852 | 
| 31 | 15 django/template/defaultfilters.py | 56 | 91| 203 | 9631 | 43917 | 
| 32 | 16 django/views/decorators/clickjacking.py | 22 | 54| 238 | 9869 | 44293 | 
| 33 | 17 django/middleware/http.py | 1 | 42| 335 | 10204 | 44628 | 
| 34 | 18 django/views/decorators/cache.py | 27 | 48| 153 | 10357 | 44991 | 
| 35 | **18 django/http/request.py** | 539 | 553| 118 | 10475 | 44991 | 
| 36 | 18 django/http/response.py | 412 | 447| 339 | 10814 | 44991 | 
| 37 | 19 django/template/backends/dummy.py | 1 | 54| 330 | 11144 | 45321 | 
| 38 | 19 django/http/response.py | 450 | 468| 186 | 11330 | 45321 | 
| 39 | 19 django/utils/cache.py | 222 | 253| 253 | 11583 | 45321 | 
| 40 | 19 django/template/defaultfilters.py | 324 | 408| 499 | 12082 | 45321 | 
| 41 | **19 django/http/request.py** | 201 | 272| 510 | 12592 | 45321 | 
| 42 | 19 django/utils/http.py | 398 | 460| 318 | 12910 | 45321 | 
| 43 | 19 django/views/csrf.py | 1 | 13| 132 | 13042 | 45321 | 
| 44 | 19 django/utils/cache.py | 37 | 83| 428 | 13470 | 45321 | 
| 45 | 19 django/core/mail/message.py | 1 | 52| 350 | 13820 | 45321 | 
| 46 | 19 django/utils/html.py | 302 | 345| 443 | 14263 | 45321 | 
| 47 | 20 django/views/debug.py | 154 | 177| 176 | 14439 | 49539 | 
| 48 | 20 django/utils/http.py | 75 | 100| 195 | 14634 | 49539 | 
| 49 | 20 django/http/response.py | 471 | 502| 153 | 14787 | 49539 | 
| 50 | 20 django/http/response.py | 221 | 237| 181 | 14968 | 49539 | 
| 51 | **20 django/http/request.py** | 38 | 70| 243 | 15211 | 49539 | 
| 52 | 20 django/views/debug.py | 193 | 241| 462 | 15673 | 49539 | 
| 53 | 20 django/views/decorators/http.py | 77 | 122| 347 | 16020 | 49539 | 
| 54 | 20 django/utils/cache.py | 320 | 339| 190 | 16210 | 49539 | 
| 55 | 21 django/http/multipartparser.py | 645 | 690| 399 | 16609 | 54566 | 
| 56 | 21 django/http/response.py | 309 | 341| 203 | 16812 | 54566 | 
| 57 | 22 django/template/utils.py | 1 | 62| 401 | 17213 | 55275 | 
| 58 | 22 django/utils/http.py | 103 | 140| 281 | 17494 | 55275 | 
| 59 | 23 django/template/context_processors.py | 1 | 32| 218 | 17712 | 55764 | 
| 60 | **23 django/http/request.py** | 325 | 356| 254 | 17966 | 55764 | 
| 61 | 24 django/core/handlers/wsgi.py | 163 | 191| 326 | 18292 | 57495 | 
| 62 | 25 django/http/__init__.py | 1 | 22| 197 | 18489 | 57692 | 
| 63 | 25 django/http/response.py | 505 | 530| 159 | 18648 | 57692 | 
| 64 | **25 django/http/request.py** | 509 | 536| 208 | 18856 | 57692 | 
| 65 | **25 django/http/request.py** | 134 | 166| 235 | 19091 | 57692 | 
| 66 | 26 django/http/cookie.py | 1 | 27| 188 | 19279 | 57881 | 
| 67 | 27 django/contrib/gis/views.py | 1 | 21| 155 | 19434 | 58036 | 
| 68 | 27 django/core/servers/basehttp.py | 97 | 116| 178 | 19612 | 58036 | 
| 69 | 27 django/template/base.py | 815 | 879| 535 | 20147 | 58036 | 
| 70 | 28 docs/_ext/djangodocs.py | 108 | 169| 524 | 20671 | 61109 | 
| 71 | 28 django/template/utils.py | 64 | 90| 195 | 20866 | 61109 | 
| 72 | 29 django/contrib/staticfiles/storage.py | 83 | 115| 346 | 21212 | 65004 | 
| 73 | 30 django/template/backends/jinja2.py | 1 | 49| 336 | 21548 | 65711 | 
| 74 | 30 django/views/csrf.py | 101 | 155| 576 | 22124 | 65711 | 
| 75 | 31 django/utils/deprecation.py | 76 | 98| 158 | 22282 | 66403 | 
| 76 | 32 django/views/generic/detail.py | 111 | 171| 504 | 22786 | 67718 | 
| 77 | 32 django/template/defaultfilters.py | 31 | 53| 191 | 22977 | 67718 | 
| 78 | 33 django/views/decorators/debug.py | 1 | 38| 218 | 23195 | 68191 | 
| 79 | 33 django/template/defaultfilters.py | 166 | 187| 206 | 23401 | 68191 | 
| 80 | 33 django/template/defaultfilters.py | 239 | 304| 427 | 23828 | 68191 | 
| 81 | 33 django/utils/html.py | 348 | 375| 212 | 24040 | 68191 | 
| 82 | 33 django/core/servers/basehttp.py | 119 | 154| 280 | 24320 | 68191 | 
| 83 | 33 django/views/defaults.py | 102 | 126| 203 | 24523 | 68191 | 
| 84 | 34 django/db/models/fields/__init__.py | 1857 | 1885| 191 | 24714 | 85185 | 
| 85 | 34 django/views/decorators/clickjacking.py | 1 | 19| 138 | 24852 | 85185 | 
| 86 | 35 django/shortcuts.py | 1 | 20| 155 | 25007 | 86283 | 
| 87 | 35 django/utils/html.py | 255 | 285| 322 | 25329 | 86283 | 
| 88 | 36 django/contrib/admin/templatetags/admin_list.py | 105 | 194| 788 | 26117 | 90112 | 
| 89 | 37 django/utils/text.py | 339 | 368| 209 | 26326 | 93542 | 
| 90 | 37 django/template/defaultfilters.py | 307 | 321| 111 | 26437 | 93542 | 
| 91 | 37 django/views/decorators/debug.py | 64 | 79| 126 | 26563 | 93542 | 
| 92 | 37 django/http/response.py | 239 | 275| 267 | 26830 | 93542 | 
| 93 | 37 django/template/base.py | 92 | 114| 153 | 26983 | 93542 | 
| 94 | 38 django/middleware/csrf.py | 205 | 327| 1189 | 28172 | 96412 | 
| 95 | 38 django/utils/cache.py | 134 | 171| 447 | 28619 | 96412 | 
| 96 | 39 django/utils/translation/template.py | 1 | 32| 343 | 28962 | 98336 | 
| 97 | 39 django/http/response.py | 278 | 307| 212 | 29174 | 98336 | 
| 98 | 39 django/template/context_processors.py | 35 | 49| 126 | 29300 | 98336 | 
| 99 | 39 django/template/context_processors.py | 52 | 82| 143 | 29443 | 98336 | 
| 100 | 40 django/template/context.py | 235 | 262| 199 | 29642 | 100225 | 
| 101 | 41 django/conf/global_settings.py | 145 | 263| 876 | 30518 | 105829 | 
| 102 | 41 django/contrib/staticfiles/storage.py | 117 | 152| 307 | 30825 | 105829 | 
| 103 | 41 django/core/mail/message.py | 98 | 123| 238 | 31063 | 105829 | 
| 104 | 42 django/contrib/auth/middleware.py | 47 | 83| 360 | 31423 | 106845 | 
| 105 | 43 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 31828 | 107250 | 
| 106 | 44 django/core/handlers/base.py | 64 | 83| 168 | 31996 | 108428 | 
| 107 | 45 django/template/defaulttags.py | 50 | 67| 152 | 32148 | 119468 | 
| 108 | 45 django/template/base.py | 667 | 702| 272 | 32420 | 119468 | 
| 109 | 46 django/db/models/functions/text.py | 24 | 55| 217 | 32637 | 121924 | 
| 110 | 46 django/middleware/csrf.py | 181 | 203| 230 | 32867 | 121924 | 
| 111 | 47 django/contrib/admin/options.py | 1 | 96| 769 | 33636 | 140277 | 
| 112 | 47 django/utils/http.py | 252 | 263| 138 | 33774 | 140277 | 
| 113 | 48 django/template/__init__.py | 1 | 69| 360 | 34134 | 140637 | 
| 114 | 48 django/contrib/auth/middleware.py | 27 | 45| 178 | 34312 | 140637 | 
| 115 | 48 django/utils/http.py | 369 | 395| 326 | 34638 | 140637 | 
| 116 | 49 django/utils/translation/trans_real.py | 487 | 510| 189 | 34827 | 144462 | 
| 117 | 50 django/middleware/locale.py | 28 | 62| 331 | 35158 | 145029 | 
| 118 | 51 django/views/i18n.py | 77 | 180| 711 | 35869 | 147536 | 
| 119 | 51 django/middleware/csrf.py | 158 | 179| 189 | 36058 | 147536 | 
| 120 | 52 django/contrib/admindocs/utils.py | 1 | 23| 133 | 36191 | 149436 | 
| 121 | 53 django/middleware/security.py | 1 | 28| 260 | 36451 | 149854 | 
| 122 | 53 django/utils/html.py | 196 | 228| 312 | 36763 | 149854 | 
| 123 | 54 django/contrib/admindocs/views.py | 317 | 347| 208 | 36971 | 153164 | 
| 124 | 55 django/utils/encoding.py | 118 | 147| 399 | 37370 | 155460 | 
| 125 | 55 django/core/handlers/wsgi.py | 66 | 127| 528 | 37898 | 155460 | 
| 126 | 56 django/middleware/gzip.py | 1 | 53| 405 | 38303 | 155866 | 
| 127 | 57 django/views/generic/base.py | 117 | 150| 241 | 38544 | 157469 | 
| 128 | 58 django/contrib/sitemaps/views.py | 1 | 19| 131 | 38675 | 158243 | 
| 129 | 58 django/template/defaultfilters.py | 423 | 458| 233 | 38908 | 158243 | 
| 130 | 58 django/views/decorators/debug.py | 41 | 63| 139 | 39047 | 158243 | 
| 131 | 58 django/utils/html.py | 88 | 111| 200 | 39247 | 158243 | 
| 132 | 59 django/core/checks/security/base.py | 1 | 86| 752 | 39999 | 159869 | 
| 133 | 59 django/template/base.py | 571 | 606| 359 | 40358 | 159869 | 
| 134 | 60 django/templatetags/static.py | 57 | 90| 159 | 40517 | 160833 | 
| 135 | 60 django/template/response.py | 45 | 58| 120 | 40637 | 160833 | 
| 136 | 60 django/utils/http.py | 193 | 215| 166 | 40803 | 160833 | 
| 137 | 60 django/middleware/csrf.py | 93 | 119| 224 | 41027 | 160833 | 
| 138 | 60 django/utils/html.py | 175 | 193| 159 | 41186 | 160833 | 
| 139 | 60 django/template/base.py | 791 | 813| 190 | 41376 | 160833 | 
| 140 | **60 django/http/request.py** | 296 | 323| 279 | 41655 | 160833 | 
| 141 | 60 django/views/defaults.py | 65 | 80| 118 | 41773 | 160833 | 
| 142 | 61 docs/conf.py | 97 | 199| 893 | 42666 | 163810 | 
| 143 | 61 django/middleware/security.py | 30 | 47| 164 | 42830 | 163810 | 
| 144 | 61 django/views/defaults.py | 1 | 62| 485 | 43315 | 163810 | 
| 145 | 61 django/utils/text.py | 1 | 24| 238 | 43553 | 163810 | 
| 146 | 62 django/middleware/clickjacking.py | 1 | 46| 361 | 43914 | 164171 | 
| 147 | 63 django/contrib/gis/geoip2/resources.py | 1 | 23| 167 | 44081 | 164338 | 
| 148 | 64 django/core/handlers/exception.py | 41 | 102| 499 | 44580 | 165279 | 
| 149 | **64 django/http/request.py** | 487 | 507| 146 | 44726 | 165279 | 
| 150 | 64 django/template/base.py | 1 | 91| 725 | 45451 | 165279 | 
| 151 | 64 django/contrib/staticfiles/storage.py | 327 | 343| 147 | 45598 | 165279 | 
| 152 | 64 django/db/models/fields/__init__.py | 1888 | 1965| 567 | 46165 | 165279 | 
| 153 | 64 django/template/defaulttags.py | 420 | 454| 270 | 46435 | 165279 | 
| 154 | 64 django/utils/encoding.py | 150 | 165| 182 | 46617 | 165279 | 
| 155 | 64 django/utils/html.py | 146 | 172| 149 | 46766 | 165279 | 
| 156 | 65 django/conf/locale/hr/formats.py | 22 | 48| 620 | 47386 | 166162 | 
| 157 | 65 django/utils/encoding.py | 168 | 201| 317 | 47703 | 166162 | 
| 158 | 65 django/views/decorators/http.py | 55 | 76| 272 | 47975 | 166162 | 
| 159 | 66 django/contrib/auth/views.py | 1 | 35| 272 | 48247 | 168814 | 
| 160 | 66 django/template/defaulttags.py | 1314 | 1378| 504 | 48751 | 168814 | 
| 161 | 66 django/contrib/staticfiles/storage.py | 49 | 81| 267 | 49018 | 168814 | 
| 162 | 67 django/middleware/common.py | 76 | 97| 227 | 49245 | 170325 | 
| 163 | 67 django/template/defaultfilters.py | 411 | 420| 111 | 49356 | 170325 | 
| 164 | 67 django/template/defaultfilters.py | 804 | 848| 378 | 49734 | 170325 | 
| 165 | 67 django/middleware/common.py | 1 | 32| 247 | 49981 | 170325 | 
| 166 | 67 django/utils/text.py | 276 | 317| 287 | 50268 | 170325 | 
| 167 | 68 django/contrib/gis/geoip2/base.py | 144 | 164| 258 | 50526 | 172361 | 
| 168 | 69 django/forms/fields.py | 1158 | 1170| 113 | 50639 | 181305 | 
| 169 | 69 django/utils/translation/template.py | 35 | 59| 165 | 50804 | 181305 | 
| 170 | 70 django/contrib/humanize/templatetags/humanize.py | 218 | 261| 731 | 51535 | 184446 | 
| 171 | 70 django/contrib/admin/options.py | 1235 | 1308| 651 | 52186 | 184446 | 
| 172 | 71 django/contrib/sites/requests.py | 1 | 20| 131 | 52317 | 184577 | 
| 173 | 72 django/contrib/auth/hashers.py | 80 | 102| 167 | 52484 | 189364 | 
| 174 | 72 django/utils/text.py | 392 | 405| 153 | 52637 | 189364 | 
| 175 | 72 django/views/debug.py | 72 | 95| 196 | 52833 | 189364 | 
| 176 | 73 django/template/loaders/cached.py | 62 | 93| 225 | 53058 | 190052 | 
| 177 | 73 django/contrib/auth/hashers.py | 600 | 637| 276 | 53334 | 190052 | 
| 178 | 73 django/utils/encoding.py | 221 | 233| 134 | 53468 | 190052 | 
| 179 | 73 django/middleware/csrf.py | 1 | 42| 330 | 53798 | 190052 | 
| 180 | 73 django/template/context.py | 215 | 233| 185 | 53983 | 190052 | 
| 181 | **73 django/http/request.py** | 111 | 123| 133 | 54116 | 190052 | 
| 182 | 73 django/utils/encoding.py | 102 | 115| 130 | 54246 | 190052 | 
| 183 | 73 django/middleware/csrf.py | 122 | 156| 267 | 54513 | 190052 | 
| 184 | 73 django/template/defaulttags.py | 1 | 47| 310 | 54823 | 190052 | 
| 185 | 74 django/template/engine.py | 1 | 53| 388 | 55211 | 191362 | 
| 186 | 75 django/core/checks/templates.py | 1 | 36| 259 | 55470 | 191622 | 
| 187 | 76 django/conf/locale/cy/formats.py | 5 | 36| 582 | 56052 | 192249 | 
| 188 | 76 django/utils/cache.py | 86 | 113| 178 | 56230 | 192249 | 
| 189 | 77 django/utils/decorators.py | 114 | 165| 371 | 56601 | 193566 | 
| 190 | 77 django/middleware/csrf.py | 57 | 71| 156 | 56757 | 193566 | 
| 191 | 78 django/views/static.py | 108 | 136| 206 | 56963 | 194623 | 
| 192 | 78 django/contrib/staticfiles/storage.py | 256 | 325| 569 | 57532 | 194623 | 
| 193 | 78 django/contrib/auth/hashers.py | 406 | 416| 127 | 57659 | 194623 | 
| 194 | 78 django/template/context.py | 265 | 281| 132 | 57791 | 194623 | 
| 195 | 79 django/template/loaders/locmem.py | 1 | 28| 127 | 57918 | 194750 | 


### Hint

```
Patch added.
Hi Mark. The default answer for the template later is to implement a filter that will let you do the lookup with a string. (​StackOverflow has lots of examples.) Maybe we could allow this by adding a key.replace('_', '-') implementing HttpHeaders.__getitem__()? (Probably worth seeing what a patch there looks like anyway.) (This instead of storing the items twice in the underlying store.)
New patch using getitem()
Replying to Carlton Gibson: Hi Mark. The default answer for the template later is to implement a filter that will let you do the lookup with a string. (​StackOverflow has lots of examples.) Maybe we could allow this by adding a key.replace('_', '-') implementing HttpHeaders.__getitem__()? (Probably worth seeing what a patch there looks like anyway.) (This instead of storing the items twice in the underlying store.) Yes, I found the filter solution. Whilst that's fine for fixing up user-generated problems, this new problem comes out-of-the-box in 2.2. It feels like there should be a solution along with it. Your suggestion is much better than mine. New patch added that allows request.headers['foo-bar'] to be looked up in templates as {{ request.headers.foo_bar }} without inelegant double storage.
Thanks Mark! Are you able to send PR via GitHub?
```

## Patch

```diff
diff --git a/django/http/request.py b/django/http/request.py
--- a/django/http/request.py
+++ b/django/http/request.py
@@ -369,6 +369,10 @@ def __init__(self, environ):
                 headers[name] = value
         super().__init__(headers)
 
+    def __getitem__(self, key):
+        """Allow header lookup using underscores in place of hyphens."""
+        return super().__getitem__(key.replace('_', '-'))
+
     @classmethod
     def parse_header_name(cls, header):
         if header.startswith(cls.HTTP_PREFIX):

```

## Test Patch

```diff
diff --git a/tests/requests/tests.py b/tests/requests/tests.py
--- a/tests/requests/tests.py
+++ b/tests/requests/tests.py
@@ -896,6 +896,7 @@ def test_wsgi_request_headers_getitem(self):
         request = WSGIRequest(self.ENVIRON)
         self.assertEqual(request.headers['User-Agent'], 'python-requests/1.2.0')
         self.assertEqual(request.headers['user-agent'], 'python-requests/1.2.0')
+        self.assertEqual(request.headers['user_agent'], 'python-requests/1.2.0')
         self.assertEqual(request.headers['Content-Type'], 'text/html')
         self.assertEqual(request.headers['Content-Length'], '100')
 

```


## Code snippets

### 1 - django/http/request.py:

Start line: 359, End line: 378

```python
class HttpHeaders(CaseInsensitiveMapping):
    HTTP_PREFIX = 'HTTP_'
    # PEP 333 gives two headers which aren't prepended with HTTP_.
    UNPREFIXED_HEADERS = {'CONTENT_TYPE', 'CONTENT_LENGTH'}

    def __init__(self, environ):
        headers = {}
        for header, value in environ.items():
            name = self.parse_header_name(header)
            if name:
                headers[name] = value
        super().__init__(headers)

    @classmethod
    def parse_header_name(cls, header):
        if header.startswith(cls.HTTP_PREFIX):
            header = header[len(cls.HTTP_PREFIX):]
        elif header not in cls.UNPREFIXED_HEADERS:
            return None
        return header.replace('_', '-').title()
```
### 2 - django/http/response.py:

Start line: 134, End line: 155

```python
class HttpResponseBase:

    def __setitem__(self, header, value):
        header = self._convert_to_charset(header, 'ascii')
        value = self._convert_to_charset(value, 'latin-1', mime_encode=True)
        self._headers[header.lower()] = (header, value)

    def __delitem__(self, header):
        self._headers.pop(header.lower(), False)

    def __getitem__(self, header):
        return self._headers[header.lower()][1]

    def has_header(self, header):
        """Case-insensitive check for a header."""
        return header.lower() in self._headers

    __contains__ = has_header

    def items(self):
        return self._headers.values()

    def get(self, header, alternate=None):
        return self._headers.get(header.lower(), (None, alternate))[1]
```
### 3 - django/utils/cache.py:

Start line: 342, End line: 388

```python
def learn_cache_key(request, response, cache_timeout=None, key_prefix=None, cache=None):
    """
    Learn what headers to take into account for some request URL from the
    response object. Store those headers in a global URL registry so that
    later access to that URL will know what headers to take into account
    without building the response object itself. The headers are named in the
    Vary header of the response, but we want to prevent response generation.

    The list of headers to use for cache key generation is stored in the same
    cache as the pages themselves. If the cache ages some data out of the
    cache, this just means that we have to build the response once to get at
    the Vary header and so at the list of headers to use for the cache key.
    """
    if key_prefix is None:
        key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
    if cache_timeout is None:
        cache_timeout = settings.CACHE_MIDDLEWARE_SECONDS
    cache_key = _generate_cache_header_key(key_prefix, request)
    if cache is None:
        cache = caches[settings.CACHE_MIDDLEWARE_ALIAS]
    if response.has_header('Vary'):
        is_accept_language_redundant = settings.USE_I18N or settings.USE_L10N
        # If i18n or l10n are used, the generated cache key will be suffixed
        # with the current locale. Adding the raw value of Accept-Language is
        # redundant in that case and would result in storing the same content
        # under multiple keys in the cache. See #18191 for details.
        headerlist = []
        for header in cc_delim_re.split(response['Vary']):
            header = header.upper().replace('-', '_')
            if header != 'ACCEPT_LANGUAGE' or not is_accept_language_redundant:
                headerlist.append('HTTP_' + header)
        headerlist.sort()
        cache.set(cache_key, headerlist, cache_timeout)
        return _generate_cache_key(request, request.method, headerlist, key_prefix)
    else:
        # if there is no Vary header, we still need a cache key
        # for the request.build_absolute_uri()
        cache.set(cache_key, [], cache_timeout)
        return _generate_cache_key(request, request.method, [], key_prefix)


def _to_tuple(s):
    t = s.split('=', 1)
    if len(t) == 2:
        return t[0].lower(), t[1]
    return t[0].lower(), True
```
### 4 - django/utils/cache.py:

Start line: 256, End line: 273

```python
def patch_vary_headers(response, newheaders):
    """
    Add (or update) the "Vary" header in the given HttpResponse object.
    newheaders is a list of header names that should be in "Vary". Existing
    headers in "Vary" aren't removed.
    """
    # Note that we need to keep the original order intact, because cache
    # implementations may rely on the order of the Vary contents in, say,
    # computing an MD5 hash.
    if response.has_header('Vary'):
        vary_headers = cc_delim_re.split(response['Vary'])
    else:
        vary_headers = []
    # Use .lower() here so we treat headers as case-insensitive.
    existing_headers = {header.lower() for header in vary_headers}
    additional_headers = [newheader for newheader in newheaders
                          if newheader.lower() not in existing_headers]
    response['Vary'] = ', '.join(vary_headers + additional_headers)
```
### 5 - django/http/response.py:

Start line: 107, End line: 132

```python
class HttpResponseBase:

    def _convert_to_charset(self, value, charset, mime_encode=False):
        """
        Convert headers key/value to ascii/latin-1 native strings.

        `charset` must be 'ascii' or 'latin-1'. If `mime_encode` is True and
        `value` can't be represented in the given charset, apply MIME-encoding.
        """
        if not isinstance(value, (bytes, str)):
            value = str(value)
        if ((isinstance(value, bytes) and (b'\n' in value or b'\r' in value)) or
                isinstance(value, str) and ('\n' in value or '\r' in value)):
            raise BadHeaderError("Header values can't contain newlines (got %r)" % value)
        try:
            if isinstance(value, str):
                # Ensure string is valid in given charset
                value.encode(charset)
            else:
                # Convert bytestring using given charset
                value = value.decode(charset)
        except UnicodeError as e:
            if mime_encode:
                value = Header(value, 'utf-8', maxlinelen=sys.maxsize).encode()
            else:
                e.reason += ', HTTP response headers must be in %s format' % charset
                raise
        return value
```
### 6 - django/http/response.py:

Start line: 1, End line: 25

```python
import datetime
import json
import mimetypes
import os
import re
import sys
import time
from email.header import Header
from http.client import responses
from urllib.parse import quote, urlparse

from django.conf import settings
from django.core import signals, signing
from django.core.exceptions import DisallowedRedirect
from django.core.serializers.json import DjangoJSONEncoder
from django.http.cookie import SimpleCookie
from django.utils import timezone
from django.utils.encoding import iri_to_uri
from django.utils.http import http_date

_charset_from_content_type_re = re.compile(r';\s*charset=(?P<charset>[^\s;]+)', re.I)


class BadHeaderError(ValueError):
    pass
```
### 7 - django/http/response.py:

Start line: 203, End line: 219

```python
class HttpResponseBase:

    def setdefault(self, key, value):
        """Set a header unless it has already been set."""
        if key not in self:
            self[key] = value

    def set_signed_cookie(self, key, value, salt='', **kwargs):
        value = signing.get_cookie_signer(salt=key + salt).sign(value)
        return self.set_cookie(key, value, **kwargs)

    def delete_cookie(self, key, path='/', domain=None):
        # Most browsers ignore the Set-Cookie header if the cookie name starts
        # with __Host- or __Secure- and the cookie doesn't use the secure flag.
        secure = key.startswith(('__Secure-', '__Host-'))
        self.set_cookie(
            key, max_age=0, path=path, domain=domain, secure=secure,
            expires='Thu, 01 Jan 1970 00:00:00 GMT',
        )
```
### 8 - django/http/response.py:

Start line: 28, End line: 105

```python
class HttpResponseBase:
    """
    An HTTP response base class with dictionary-accessed headers.

    This class doesn't handle content. It should not be used directly.
    Use the HttpResponse and StreamingHttpResponse subclasses instead.
    """

    status_code = 200

    def __init__(self, content_type=None, status=None, reason=None, charset=None):
        # _headers is a mapping of the lowercase name to the original case of
        # the header (required for working with legacy systems) and the header
        # value. Both the name of the header and its value are ASCII strings.
        self._headers = {}
        self._closable_objects = []
        # This parameter is set by the handler. It's necessary to preserve the
        # historical behavior of request_finished.
        self._handler_class = None
        self.cookies = SimpleCookie()
        self.closed = False
        if status is not None:
            try:
                self.status_code = int(status)
            except (ValueError, TypeError):
                raise TypeError('HTTP status code must be an integer.')

            if not 100 <= self.status_code <= 599:
                raise ValueError('HTTP status code must be an integer from 100 to 599.')
        self._reason_phrase = reason
        self._charset = charset
        if content_type is None:
            content_type = 'text/html; charset=%s' % self.charset
        self['Content-Type'] = content_type

    @property
    def reason_phrase(self):
        if self._reason_phrase is not None:
            return self._reason_phrase
        # Leave self._reason_phrase unset in order to use the default
        # reason phrase for status code.
        return responses.get(self.status_code, 'Unknown Status Code')

    @reason_phrase.setter
    def reason_phrase(self, value):
        self._reason_phrase = value

    @property
    def charset(self):
        if self._charset is not None:
            return self._charset
        content_type = self.get('Content-Type', '')
        matched = _charset_from_content_type_re.search(content_type)
        if matched:
            # Extract the charset and strip its double quotes
            return matched.group('charset').replace('"', '')
        return settings.DEFAULT_CHARSET

    @charset.setter
    def charset(self, value):
        self._charset = value

    def serialize_headers(self):
        """HTTP headers as a bytestring."""
        def to_bytes(val, encoding):
            return val if isinstance(val, bytes) else val.encode(encoding)

        headers = [
            (to_bytes(key, 'ascii') + b': ' + to_bytes(value, 'latin-1'))
            for key, value in self._headers.values()
        ]
        return b'\r\n'.join(headers)

    __bytes__ = serialize_headers

    @property
    def _content_type_for_repr(self):
        return ', "%s"' % self['Content-Type'] if 'Content-Type' in self else ''
```
### 9 - django/http/request.py:

Start line: 72, End line: 89

```python
class HttpRequest:

    def _get_raw_host(self):
        """
        Return the HTTP host using the environment or request headers. Skip
        allowed hosts protection, so may return an insecure host.
        """
        # We try three options, in order of decreasing preference.
        if settings.USE_X_FORWARDED_HOST and (
                'HTTP_X_FORWARDED_HOST' in self.META):
            host = self.META['HTTP_X_FORWARDED_HOST']
        elif 'HTTP_HOST' in self.META:
            host = self.META['HTTP_HOST']
        else:
            # Reconstruct the host using the algorithm from PEP 333.
            host = self.META['SERVER_NAME']
            server_port = self.get_port()
            if server_port != ('443' if self.is_secure() else '80'):
                host = '%s:%s' % (host, server_port)
        return host
```
### 10 - django/http/request.py:

Start line: 125, End line: 132

```python
class HttpRequest:

    def _get_full_path(self, path, force_append_slash):
        # RFC 3986 requires query string arguments to be in the ASCII range.
        # Rather than crash if this doesn't happen, we encode defensively.
        return '%s%s%s' % (
            escape_uri_path(path),
            '/' if force_append_slash and not path.endswith('/') else '',
            ('?' + iri_to_uri(self.META.get('QUERY_STRING', ''))) if self.META.get('QUERY_STRING', '') else ''
        )
```
### 18 - django/http/request.py:

Start line: 168, End line: 199

```python
class HttpRequest:

    def build_absolute_uri(self, location=None):
        """
        Build an absolute URI from the location and the variables available in
        this request. If no ``location`` is specified, build the absolute URI
        using request.get_full_path(). If the location is absolute, convert it
        to an RFC 3987 compliant URI and return it. If location is relative or
        is scheme-relative (i.e., ``//example.com/``), urljoin() it to a base
        URL constructed from the request variables.
        """
        if location is None:
            # Make it an absolute url (but schemeless and domainless) for the
            # edge case that the path starts with '//'.
            location = '//%s' % self.get_full_path()
        bits = urlsplit(location)
        if not (bits.scheme and bits.netloc):
            # Handle the simple, most common case. If the location is absolute
            # and a scheme or host (netloc) isn't provided, skip an expensive
            # urljoin() as long as no path segments are '.' or '..'.
            if (bits.path.startswith('/') and not bits.scheme and not bits.netloc and
                    '/./' not in bits.path and '/../' not in bits.path):
                # If location starts with '//' but has no netloc, reuse the
                # schema and netloc from the current request. Strip the double
                # slashes and continue as if it wasn't specified.
                if location.startswith('//'):
                    location = location[2:]
                location = self._current_scheme_host + location
            else:
                # Join the constructed URL with the provided location, which
                # allows the provided location to apply query strings to the
                # base path.
                location = urljoin(self._current_scheme_host + self.path, location)
        return iri_to_uri(location)
```
### 20 - django/http/request.py:

Start line: 1, End line: 35

```python
import copy
import re
from io import BytesIO
from itertools import chain
from urllib.parse import quote, urlencode, urljoin, urlsplit

from django.conf import settings
from django.core import signing
from django.core.exceptions import (
    DisallowedHost, ImproperlyConfigured, RequestDataTooBig,
)
from django.core.files import uploadhandler
from django.http.multipartparser import MultiPartParser, MultiPartParserError
from django.utils.datastructures import (
    CaseInsensitiveMapping, ImmutableList, MultiValueDict,
)
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, limited_parse_qsl

RAISE_ERROR = object()
host_validation_re = re.compile(r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:\d+)?$")


class UnreadablePostError(OSError):
    pass


class RawPostDataException(Exception):
    """
    You cannot access raw_post_data from a request that has
    multipart/* POST data if it has been accessed via POST,
    FILES, etc..
    """
    pass
```
### 23 - django/http/request.py:

Start line: 91, End line: 109

```python
class HttpRequest:

    def get_host(self):
        """Return the HTTP host using the environment or request headers."""
        host = self._get_raw_host()

        # Allow variants of localhost if ALLOWED_HOSTS is empty and DEBUG=True.
        allowed_hosts = settings.ALLOWED_HOSTS
        if settings.DEBUG and not allowed_hosts:
            allowed_hosts = ['localhost', '127.0.0.1', '[::1]']

        domain, port = split_domain_port(host)
        if domain and validate_host(domain, allowed_hosts):
            return host
        else:
            msg = "Invalid HTTP_HOST header: %r." % host
            if domain:
                msg += " You may need to add %r to ALLOWED_HOSTS." % domain
            else:
                msg += " The domain name provided is not valid according to RFC 1034/1035."
            raise DisallowedHost(msg)
```
### 29 - django/http/request.py:

Start line: 274, End line: 294

```python
class HttpRequest:

    @property
    def body(self):
        if not hasattr(self, '_body'):
            if self._read_started:
                raise RawPostDataException("You cannot access body after reading from request's data stream")

            # Limit the maximum request data size that will be handled in-memory.
            if (settings.DATA_UPLOAD_MAX_MEMORY_SIZE is not None and
                    int(self.META.get('CONTENT_LENGTH') or 0) > settings.DATA_UPLOAD_MAX_MEMORY_SIZE):
                raise RequestDataTooBig('Request body exceeded settings.DATA_UPLOAD_MAX_MEMORY_SIZE.')

            try:
                self._body = self.read()
            except OSError as e:
                raise UnreadablePostError(*e.args) from e
            self._stream = BytesIO(self._body)
        return self._body

    def _mark_post_parse_error(self):
        self._post = QueryDict()
        self._files = MultiValueDict()
```
### 35 - django/http/request.py:

Start line: 539, End line: 553

```python
# It's neither necessary nor appropriate to use
# django.utils.encoding.force_str() for parsing URLs and form inputs. Thus,
# this slightly more restricted function, used by QueryDict.
def bytes_to_text(s, encoding):
    """
    Convert bytes objects to strings, using the given encoding. Illegally
    encoded input characters are replaced with Unicode "unknown" codepoint
    (\ufffd).

    Return any non-bytes objects without change.
    """
    if isinstance(s, bytes):
        return str(s, encoding, 'replace')
    else:
        return s
```
### 41 - django/http/request.py:

Start line: 201, End line: 272

```python
class HttpRequest:

    @cached_property
    def _current_scheme_host(self):
        return '{}://{}'.format(self.scheme, self.get_host())

    def _get_scheme(self):
        """
        Hook for subclasses like WSGIRequest to implement. Return 'http' by
        default.
        """
        return 'http'

    @property
    def scheme(self):
        if settings.SECURE_PROXY_SSL_HEADER:
            try:
                header, value = settings.SECURE_PROXY_SSL_HEADER
            except ValueError:
                raise ImproperlyConfigured(
                    'The SECURE_PROXY_SSL_HEADER setting must be a tuple containing two values.'
                )
            if self.META.get(header) == value:
                return 'https'
        return self._get_scheme()

    def is_secure(self):
        return self.scheme == 'https'

    def is_ajax(self):
        return self.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, val):
        """
        Set the encoding used for GET/POST accesses. If the GET or POST
        dictionary has already been created, remove and recreate it on the
        next access (so that it is decoded correctly).
        """
        self._encoding = val
        if hasattr(self, 'GET'):
            del self.GET
        if hasattr(self, '_post'):
            del self._post

    def _initialize_handlers(self):
        self._upload_handlers = [uploadhandler.load_handler(handler, self)
                                 for handler in settings.FILE_UPLOAD_HANDLERS]

    @property
    def upload_handlers(self):
        if not self._upload_handlers:
            # If there are no upload handlers defined, initialize them from settings.
            self._initialize_handlers()
        return self._upload_handlers

    @upload_handlers.setter
    def upload_handlers(self, upload_handlers):
        if hasattr(self, '_files'):
            raise AttributeError("You cannot set the upload handlers after the upload has been processed.")
        self._upload_handlers = upload_handlers

    def parse_file_upload(self, META, post_data):
        """Return a tuple of (POST QueryDict, FILES MultiValueDict)."""
        self.upload_handlers = ImmutableList(
            self.upload_handlers,
            warning="You cannot alter upload handlers after the upload has been processed."
        )
        parser = MultiPartParser(META, post_data, self.upload_handlers, self.encoding)
        return parser.parse()
```
### 51 - django/http/request.py:

Start line: 38, End line: 70

```python
class HttpRequest:
    """A basic HTTP request."""

    # The encoding used in GET/POST dicts. None means use default setting.
    _encoding = None
    _upload_handlers = []

    def __init__(self):
        # WARNING: The `WSGIRequest` subclass doesn't call `super`.
        # Any variable assignment made here should also happen in
        # `WSGIRequest.__init__()`.

        self.GET = QueryDict(mutable=True)
        self.POST = QueryDict(mutable=True)
        self.COOKIES = {}
        self.META = {}
        self.FILES = MultiValueDict()

        self.path = ''
        self.path_info = ''
        self.method = None
        self.resolver_match = None
        self.content_type = None
        self.content_params = None

    def __repr__(self):
        if self.method is None or not self.get_full_path():
            return '<%s>' % self.__class__.__name__
        return '<%s: %s %r>' % (self.__class__.__name__, self.method, self.get_full_path())

    @cached_property
    def headers(self):
        return HttpHeaders(self.META)
```
### 60 - django/http/request.py:

Start line: 325, End line: 356

```python
class HttpRequest:

    def close(self):
        if hasattr(self, '_files'):
            for f in chain.from_iterable(l[1] for l in self._files.lists()):
                f.close()

    # File-like and iterator interface.
    #
    # Expects self._stream to be set to an appropriate source of bytes by
    # a corresponding request subclass (e.g. WSGIRequest).
    # Also when request data has already been read by request.POST or
    # request.body, self._stream points to a BytesIO instance
    # containing that data.

    def read(self, *args, **kwargs):
        self._read_started = True
        try:
            return self._stream.read(*args, **kwargs)
        except OSError as e:
            raise UnreadablePostError(*e.args) from e

    def readline(self, *args, **kwargs):
        self._read_started = True
        try:
            return self._stream.readline(*args, **kwargs)
        except OSError as e:
            raise UnreadablePostError(*e.args) from e

    def __iter__(self):
        return iter(self.readline, b'')

    def readlines(self):
        return list(self)
```
### 64 - django/http/request.py:

Start line: 509, End line: 536

```python
class QueryDict(MultiValueDict):

    def urlencode(self, safe=None):
        """
        Return an encoded string of all query string arguments.

        `safe` specifies characters which don't require quoting, for example::

            >>> q = QueryDict(mutable=True)
            >>> q['next'] = '/a&b/'
            >>> q.urlencode()
            'next=%2Fa%26b%2F'
            >>> q.urlencode(safe='/')
            'next=/a%26b/'
        """
        output = []
        if safe:
            safe = safe.encode(self.encoding)

            def encode(k, v):
                return '%s=%s' % ((quote(k, safe), quote(v, safe)))
        else:
            def encode(k, v):
                return urlencode({k: v})
        for k, list_ in self.lists():
            output.extend(
                encode(k.encode(self.encoding), str(v).encode(self.encoding))
                for v in list_
            )
        return '&'.join(output)
```
### 65 - django/http/request.py:

Start line: 134, End line: 166

```python
class HttpRequest:

    def get_signed_cookie(self, key, default=RAISE_ERROR, salt='', max_age=None):
        """
        Attempt to return a signed cookie. If the signature fails or the
        cookie has expired, raise an exception, unless the `default` argument
        is provided,  in which case return that value.
        """
        try:
            cookie_value = self.COOKIES[key]
        except KeyError:
            if default is not RAISE_ERROR:
                return default
            else:
                raise
        try:
            value = signing.get_cookie_signer(salt=key + salt).unsign(
                cookie_value, max_age=max_age)
        except signing.BadSignature:
            if default is not RAISE_ERROR:
                return default
            else:
                raise
        return value

    def get_raw_uri(self):
        """
        Return an absolute URI from variables available in this request. Skip
        allowed hosts protection, so may return insecure URI.
        """
        return '{scheme}://{host}{path}'.format(
            scheme=self.scheme,
            host=self._get_raw_host(),
            path=self.get_full_path(),
        )
```
### 140 - django/http/request.py:

Start line: 296, End line: 323

```python
class HttpRequest:

    def _load_post_and_files(self):
        """Populate self._post and self._files if the content-type is a form type"""
        if self.method != 'POST':
            self._post, self._files = QueryDict(encoding=self._encoding), MultiValueDict()
            return
        if self._read_started and not hasattr(self, '_body'):
            self._mark_post_parse_error()
            return

        if self.content_type == 'multipart/form-data':
            if hasattr(self, '_body'):
                # Use already read data
                data = BytesIO(self._body)
            else:
                data = self
            try:
                self._post, self._files = self.parse_file_upload(self.META, data)
            except MultiPartParserError:
                # An error occurred while parsing POST data. Since when
                # formatting the error the request handler might access
                # self.POST, set self._post and self._file to prevent
                # attempts to parse POST data again.
                self._mark_post_parse_error()
                raise
        elif self.content_type == 'application/x-www-form-urlencoded':
            self._post, self._files = QueryDict(self.body, encoding=self._encoding), MultiValueDict()
        else:
            self._post, self._files = QueryDict(encoding=self._encoding), MultiValueDict()
```
### 149 - django/http/request.py:

Start line: 487, End line: 507

```python
class QueryDict(MultiValueDict):

    def pop(self, key, *args):
        self._assert_mutable()
        return super().pop(key, *args)

    def popitem(self):
        self._assert_mutable()
        return super().popitem()

    def clear(self):
        self._assert_mutable()
        super().clear()

    def setdefault(self, key, default=None):
        self._assert_mutable()
        key = bytes_to_text(key, self.encoding)
        default = bytes_to_text(default, self.encoding)
        return super().setdefault(key, default)

    def copy(self):
        """Return a mutable copy of this object."""
        return self.__deepcopy__({})
```
### 181 - django/http/request.py:

Start line: 111, End line: 123

```python
class HttpRequest:

    def get_port(self):
        """Return the port number for the request as a string."""
        if settings.USE_X_FORWARDED_PORT and 'HTTP_X_FORWARDED_PORT' in self.META:
            port = self.META['HTTP_X_FORWARDED_PORT']
        else:
            port = self.META['SERVER_PORT']
        return str(port)

    def get_full_path(self, force_append_slash=False):
        return self._get_full_path(self.path, force_append_slash)

    def get_full_path_info(self, force_append_slash=False):
        return self._get_full_path(self.path_info, force_append_slash)
```
