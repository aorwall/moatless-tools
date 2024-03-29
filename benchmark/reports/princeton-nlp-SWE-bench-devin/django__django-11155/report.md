# django__django-11155

| **django/django** | `cef3f2d3c64055c9fc1757fd61dba24b557a2add` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 856 |
| **Any found context length** | 856 |
| **Avg pos** | 3.5 |
| **Min pos** | 1 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/conf/global_settings.py b/django/conf/global_settings.py
--- a/django/conf/global_settings.py
+++ b/django/conf/global_settings.py
@@ -154,6 +154,9 @@ def gettext_noop(s):
 LANGUAGE_COOKIE_AGE = None
 LANGUAGE_COOKIE_DOMAIN = None
 LANGUAGE_COOKIE_PATH = '/'
+LANGUAGE_COOKIE_SECURE = False
+LANGUAGE_COOKIE_HTTPONLY = False
+LANGUAGE_COOKIE_SAMESITE = None
 
 
 # If you set this to True, Django will format dates, numbers and calendars
diff --git a/django/views/i18n.py b/django/views/i18n.py
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -55,6 +55,9 @@ def set_language(request):
                 max_age=settings.LANGUAGE_COOKIE_AGE,
                 path=settings.LANGUAGE_COOKIE_PATH,
                 domain=settings.LANGUAGE_COOKIE_DOMAIN,
+                secure=settings.LANGUAGE_COOKIE_SECURE,
+                httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
+                samesite=settings.LANGUAGE_COOKIE_SAMESITE,
             )
     return response
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/conf/global_settings.py | 157 | 157 | 1 | 1 | 856
| django/views/i18n.py | 58 | 58 | 6 | 4 | 4204


## Problem Statement

```
Support setting Secure, HttpOnly, SameSite on the language cookie
Description
	
I propose to add the following settings, with the following default values:
LANGUAGE_COOKIE_SECURE = False
LANGUAGE_COOKIE_HTTPONLY = False
LANGUAGE_COOKIE_SAMESITE = None
The default values maintain the current behavior.
These settings do not provide much security value, since the language is not secret or sensitive. This was also discussed briefly here: ​https://github.com/django/django/pull/8380#discussion_r112448195. The reasons I'd like to add them are:
Sometimes auditors require them.
I personally prefer to set them unless I have a reason *not* to.
Browsers are starting to strongly nudge toward HttpOnly and Secure when possible, e.g. ​https://webkit.org/blog/8613/intelligent-tracking-prevention-2-1/.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/conf/global_settings.py** | 145 | 260| 856 | 856 | 5584 | 
| 2 | **1 django/conf/global_settings.py** | 496 | 635| 853 | 1709 | 5584 | 
| 3 | 2 django/core/checks/security/base.py | 1 | 86| 752 | 2461 | 7210 | 
| 4 | **2 django/conf/global_settings.py** | 395 | 494| 793 | 3254 | 7210 | 
| 5 | 3 django/core/checks/security/sessions.py | 1 | 98| 572 | 3826 | 7783 | 
| **-> 6 <-** | **4 django/views/i18n.py** | 23 | 59| 378 | 4204 | 10259 | 
| 7 | 5 django/http/response.py | 157 | 201| 447 | 4651 | 14517 | 
| 8 | 5 django/core/checks/security/base.py | 88 | 190| 747 | 5398 | 14517 | 
| 9 | 6 django/core/checks/security/csrf.py | 1 | 41| 299 | 5697 | 14816 | 
| 10 | 7 django/middleware/security.py | 1 | 28| 260 | 5957 | 15234 | 
| 11 | **7 django/conf/global_settings.py** | 261 | 343| 800 | 6757 | 15234 | 
| 12 | 8 django/http/__init__.py | 1 | 22| 197 | 6954 | 15431 | 
| 13 | 8 django/middleware/security.py | 30 | 47| 164 | 7118 | 15431 | 
| 14 | 8 django/http/response.py | 203 | 219| 193 | 7311 | 15431 | 
| 15 | 9 django/http/cookie.py | 1 | 27| 188 | 7499 | 15620 | 
| 16 | 10 django/middleware/clickjacking.py | 1 | 46| 361 | 7860 | 15981 | 
| 17 | 11 django/utils/http.py | 369 | 395| 326 | 8186 | 19934 | 
| 18 | 12 django/contrib/sessions/middleware.py | 1 | 76| 578 | 8764 | 20513 | 
| 19 | 13 django/views/debug.py | 72 | 95| 196 | 8960 | 24731 | 
| 20 | 14 django/conf/__init__.py | 132 | 185| 472 | 9432 | 26517 | 
| 21 | 14 django/views/debug.py | 154 | 177| 176 | 9608 | 26517 | 
| 22 | 15 django/contrib/messages/storage/cookie.py | 78 | 92| 127 | 9735 | 27849 | 
| 23 | 16 django/http/request.py | 201 | 272| 510 | 10245 | 32535 | 
| 24 | 17 django/middleware/locale.py | 28 | 62| 331 | 10576 | 33102 | 
| 25 | 18 django/middleware/csrf.py | 94 | 120| 224 | 10800 | 35977 | 
| 26 | 19 django/views/decorators/debug.py | 41 | 63| 139 | 10939 | 36450 | 
| 27 | 19 django/middleware/csrf.py | 159 | 180| 189 | 11128 | 36450 | 
| 28 | 19 django/middleware/csrf.py | 182 | 204| 230 | 11358 | 36450 | 
| 29 | 20 django/contrib/auth/views.py | 1 | 35| 272 | 11630 | 39102 | 
| 30 | 20 django/middleware/csrf.py | 206 | 328| 1189 | 12819 | 39102 | 
| 31 | 20 django/views/debug.py | 48 | 69| 160 | 12979 | 39102 | 
| 32 | 20 django/middleware/csrf.py | 1 | 42| 330 | 13309 | 39102 | 
| 33 | 20 django/http/request.py | 359 | 378| 151 | 13460 | 39102 | 
| 34 | 20 django/middleware/csrf.py | 123 | 157| 267 | 13727 | 39102 | 
| 35 | 21 django/core/checks/translation.py | 1 | 71| 508 | 14235 | 39611 | 
| 36 | 22 django/contrib/sessions/backends/signed_cookies.py | 28 | 83| 366 | 14601 | 40139 | 
| 37 | 23 django/middleware/common.py | 34 | 61| 257 | 14858 | 41650 | 
| 38 | 23 django/http/request.py | 134 | 166| 235 | 15093 | 41650 | 
| 39 | 23 django/contrib/messages/storage/cookie.py | 123 | 142| 182 | 15275 | 41650 | 
| 40 | 23 django/utils/http.py | 296 | 317| 221 | 15496 | 41650 | 
| 41 | 24 django/template/context_processors.py | 52 | 82| 143 | 15639 | 42139 | 
| 42 | 25 django/views/csrf.py | 1 | 13| 132 | 15771 | 43682 | 
| 43 | 26 django/core/handlers/wsgi.py | 66 | 127| 528 | 16299 | 45413 | 
| 44 | 26 django/views/csrf.py | 15 | 100| 835 | 17134 | 45413 | 
| 45 | **26 django/conf/global_settings.py** | 1 | 50| 366 | 17500 | 45413 | 
| 46 | 26 django/middleware/locale.py | 1 | 26| 242 | 17742 | 45413 | 
| 47 | 27 django/core/signing.py | 1 | 78| 722 | 18464 | 47122 | 
| 48 | 28 django/contrib/auth/admin.py | 1 | 22| 188 | 18652 | 48848 | 
| 49 | 28 django/middleware/common.py | 1 | 32| 247 | 18899 | 48848 | 
| 50 | 29 django/contrib/auth/__init__.py | 1 | 58| 393 | 19292 | 50415 | 
| 51 | **29 django/conf/global_settings.py** | 344 | 394| 826 | 20118 | 50415 | 
| 52 | 29 django/views/decorators/debug.py | 1 | 38| 218 | 20336 | 50415 | 
| 53 | 29 django/views/debug.py | 193 | 241| 462 | 20798 | 50415 | 
| 54 | 30 django/middleware/cache.py | 74 | 110| 340 | 21138 | 51991 | 
| 55 | 31 django/views/decorators/clickjacking.py | 22 | 54| 238 | 21376 | 52367 | 
| 56 | 31 django/conf/__init__.py | 65 | 95| 260 | 21636 | 52367 | 
| 57 | 32 django/contrib/sessions/models.py | 1 | 36| 250 | 21886 | 52617 | 
| 58 | 33 django/contrib/auth/forms.py | 209 | 234| 176 | 22062 | 55585 | 
| 59 | 34 django/views/decorators/csrf.py | 1 | 57| 460 | 22522 | 56045 | 
| 60 | 34 django/views/debug.py | 125 | 152| 248 | 22770 | 56045 | 
| 61 | 35 django/utils/crypto.py | 1 | 45| 350 | 23120 | 56754 | 
| 62 | 35 django/http/response.py | 134 | 155| 176 | 23296 | 56754 | 
| 63 | 35 django/conf/__init__.py | 188 | 237| 353 | 23649 | 56754 | 
| 64 | 35 django/middleware/csrf.py | 45 | 54| 112 | 23761 | 56754 | 
| 65 | 36 django/core/servers/basehttp.py | 156 | 175| 170 | 23931 | 58466 | 
| 66 | 36 django/middleware/common.py | 63 | 74| 117 | 24048 | 58466 | 
| 67 | 37 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 24337 | 58771 | 
| 68 | 38 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 24753 | 59232 | 
| 69 | 39 django/conf/locale/cs/formats.py | 5 | 43| 640 | 25393 | 59917 | 
| 70 | 40 django/utils/translation/__init__.py | 1 | 36| 281 | 25674 | 62201 | 
| 71 | 41 django/utils/cache.py | 116 | 131| 188 | 25862 | 65750 | 
| 72 | 41 django/core/checks/security/base.py | 193 | 211| 127 | 25989 | 65750 | 
| 73 | 42 django/contrib/admin/tests.py | 1 | 112| 812 | 26801 | 67166 | 
| 74 | 43 django/contrib/auth/middleware.py | 1 | 24| 193 | 26994 | 68182 | 
| 75 | 43 django/utils/http.py | 398 | 460| 318 | 27312 | 68182 | 
| 76 | 44 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 27574 | 68460 | 
| 77 | 45 django/conf/locale/fr/formats.py | 5 | 34| 489 | 28063 | 68994 | 
| 78 | 45 django/contrib/messages/storage/cookie.py | 144 | 167| 190 | 28253 | 68994 | 
| 79 | 46 django/utils/translation/trans_real.py | 444 | 484| 289 | 28542 | 72819 | 
| 80 | 47 django/contrib/sites/models.py | 1 | 22| 130 | 28672 | 73608 | 
| 81 | 47 django/utils/http.py | 1 | 72| 693 | 29365 | 73608 | 
| 82 | 48 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 29635 | 73894 | 
| 83 | 49 django/views/decorators/http.py | 1 | 52| 350 | 29985 | 74846 | 
| 84 | 49 django/utils/translation/trans_real.py | 364 | 389| 221 | 30206 | 74846 | 
| 85 | 49 django/utils/cache.py | 1 | 34| 269 | 30475 | 74846 | 
| 86 | 49 django/views/decorators/debug.py | 64 | 79| 126 | 30601 | 74846 | 
| 87 | 50 django/conf/locale/id/formats.py | 5 | 50| 708 | 31309 | 75599 | 
| 88 | 51 django/templatetags/i18n.py | 531 | 549| 121 | 31430 | 79518 | 
| 89 | 51 django/middleware/common.py | 149 | 175| 254 | 31684 | 79518 | 
| 90 | 52 django/conf/locale/es/formats.py | 5 | 31| 285 | 31969 | 79848 | 
| 91 | 53 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 32174 | 80301 | 
| 92 | 54 django/utils/formats.py | 1 | 57| 377 | 32551 | 82393 | 
| 93 | 55 django/contrib/sessions/backends/base.py | 127 | 207| 547 | 33098 | 84918 | 
| 94 | 56 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 33806 | 85671 | 
| 95 | 57 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 34081 | 85991 | 
| 96 | 58 django/templatetags/l10n.py | 41 | 64| 190 | 34271 | 86433 | 
| 97 | 59 django/conf/locale/de/formats.py | 5 | 29| 323 | 34594 | 86801 | 
| 98 | 60 django/middleware/http.py | 1 | 42| 335 | 34929 | 87136 | 
| 99 | 61 django/conf/locale/da/formats.py | 5 | 27| 250 | 35179 | 87431 | 
| 100 | 62 docs/conf.py | 97 | 199| 893 | 36072 | 90408 | 
| 101 | 63 django/conf/locale/ml/formats.py | 5 | 41| 663 | 36735 | 91116 | 
| 102 | **63 django/views/i18n.py** | 74 | 177| 711 | 37446 | 91116 | 
| 103 | 64 django/conf/locale/sk/formats.py | 5 | 30| 348 | 37794 | 91509 | 
| 104 | 65 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 38046 | 91777 | 
| 105 | 65 django/conf/__init__.py | 1 | 39| 240 | 38286 | 91777 | 
| 106 | 66 django/conf/locale/sr/formats.py | 23 | 44| 511 | 38797 | 92626 | 
| 107 | 67 django/conf/locale/cy/formats.py | 5 | 36| 582 | 39379 | 93253 | 
| 108 | 68 django/conf/locale/en/formats.py | 5 | 41| 663 | 40042 | 93961 | 
| 109 | 69 django/core/files/storage.py | 1 | 22| 158 | 40200 | 96797 | 
| 110 | 70 django/utils/safestring.py | 40 | 64| 159 | 40359 | 97185 | 
| 111 | 71 django/utils/translation/trans_null.py | 1 | 68| 269 | 40628 | 97454 | 
| 112 | 72 django/conf/locale/sv/formats.py | 5 | 39| 534 | 41162 | 98033 | 
| 113 | 73 django/conf/locale/az/formats.py | 5 | 33| 399 | 41561 | 98477 | 
| 114 | 74 django/conf/locale/sq/formats.py | 5 | 22| 128 | 41689 | 98649 | 
| 115 | **74 django/views/i18n.py** | 1 | 20| 117 | 41806 | 98649 | 
| 116 | 74 django/core/servers/basehttp.py | 119 | 154| 280 | 42086 | 98649 | 
| 117 | 74 django/utils/http.py | 193 | 215| 166 | 42252 | 98649 | 
| 118 | 75 django/template/defaultfilters.py | 324 | 408| 499 | 42751 | 104702 | 
| 119 | 76 django/conf/locale/ro/formats.py | 5 | 36| 262 | 43013 | 105009 | 
| 120 | 77 django/conf/locale/uk/formats.py | 5 | 38| 460 | 43473 | 105514 | 
| 121 | 77 django/utils/cache.py | 222 | 253| 253 | 43726 | 105514 | 
| 122 | 77 django/utils/cache.py | 37 | 83| 428 | 44154 | 105514 | 
| 123 | 78 django/conf/locale/sr_Latn/formats.py | 23 | 44| 511 | 44665 | 106363 | 
| 124 | 78 django/middleware/cache.py | 1 | 52| 431 | 45096 | 106363 | 
| 125 | 79 django/conf/locale/pt/formats.py | 5 | 39| 630 | 45726 | 107038 | 
| 126 | 80 django/conf/locale/pt_BR/formats.py | 5 | 34| 494 | 46220 | 107577 | 
| 127 | 81 django/db/utils.py | 160 | 179| 188 | 46408 | 109595 | 
| 128 | 82 django/conf/locale/fi/formats.py | 5 | 40| 470 | 46878 | 110110 | 
| 129 | 83 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 47586 | 110863 | 
| 130 | 84 django/conf/locale/ru/formats.py | 5 | 33| 402 | 47988 | 111310 | 
| 131 | 85 django/conf/locale/pl/formats.py | 5 | 30| 339 | 48327 | 111694 | 
| 132 | 86 django/conf/locale/bs/formats.py | 5 | 22| 139 | 48466 | 111877 | 
| 133 | 87 django/conf/locale/nn/formats.py | 5 | 41| 664 | 49130 | 112586 | 
| 134 | 88 django/conf/locale/et/formats.py | 5 | 22| 133 | 49263 | 112763 | 
| 135 | 89 django/conf/locale/lt/formats.py | 5 | 46| 711 | 49974 | 113519 | 
| 136 | 90 django/contrib/admin/options.py | 1 | 96| 769 | 50743 | 131847 | 
| 137 | 91 django/conf/locale/gl/formats.py | 5 | 22| 170 | 50913 | 132061 | 
| 138 | 92 django/conf/locale/el/formats.py | 5 | 36| 508 | 51421 | 132614 | 
| 139 | 93 django/conf/locale/is/formats.py | 5 | 22| 130 | 51551 | 132789 | 
| 140 | 94 django/contrib/auth/password_validation.py | 1 | 32| 206 | 51757 | 134278 | 
| 141 | 95 django/conf/locale/ca/formats.py | 5 | 31| 287 | 52044 | 134610 | 
| 142 | 96 django/conf/locale/sl/formats.py | 22 | 48| 596 | 52640 | 135459 | 
| 143 | 96 django/views/debug.py | 1 | 45| 306 | 52946 | 135459 | 
| 144 | 97 django/conf/locale/hu/formats.py | 5 | 32| 323 | 53269 | 135827 | 
| 145 | 98 django/conf/locale/gd/formats.py | 5 | 22| 144 | 53413 | 136015 | 
| 146 | 99 django/conf/locale/nb/formats.py | 5 | 40| 646 | 54059 | 136706 | 
| 147 | 100 django/conf/locale/bn/formats.py | 5 | 33| 294 | 54353 | 137044 | 
| 148 | 100 django/utils/cache.py | 276 | 296| 217 | 54570 | 137044 | 
| 149 | 100 django/contrib/auth/middleware.py | 113 | 124| 107 | 54677 | 137044 | 
| 150 | 100 django/contrib/sessions/backends/base.py | 38 | 107| 513 | 55190 | 137044 | 
| 151 | 101 django/conf/locale/ja/formats.py | 5 | 22| 149 | 55339 | 137237 | 
| 152 | 101 django/core/signing.py | 145 | 170| 255 | 55594 | 137237 | 
| 153 | 102 django/db/models/options.py | 1 | 36| 304 | 55898 | 144103 | 
| 154 | 102 django/utils/cache.py | 342 | 388| 531 | 56429 | 144103 | 
| 155 | 102 django/views/csrf.py | 101 | 155| 576 | 57005 | 144103 | 
| 156 | 102 django/conf/locale/sr/formats.py | 5 | 22| 293 | 57298 | 144103 | 
| 157 | 103 django/conf/locale/eu/formats.py | 5 | 22| 171 | 57469 | 144319 | 
| 158 | 103 django/conf/locale/sr_Latn/formats.py | 5 | 22| 293 | 57762 | 144319 | 
| 159 | 104 django/conf/locale/hi/formats.py | 5 | 22| 125 | 57887 | 144488 | 
| 160 | 104 django/http/response.py | 1 | 25| 144 | 58031 | 144488 | 
| 161 | 104 django/views/decorators/clickjacking.py | 1 | 19| 138 | 58169 | 144488 | 
| 162 | 105 django/conf/locale/nl/formats.py | 5 | 71| 479 | 58648 | 146519 | 
| 163 | 106 django/conf/locale/ga/formats.py | 5 | 22| 124 | 58772 | 146687 | 
| 164 | 106 django/contrib/auth/middleware.py | 47 | 83| 360 | 59132 | 146687 | 
| 165 | 107 django/conf/locale/eo/formats.py | 5 | 50| 742 | 59874 | 147474 | 
| 166 | 107 django/contrib/auth/forms.py | 1 | 20| 137 | 60011 | 147474 | 
| 167 | 108 django/conf/locale/lv/formats.py | 5 | 47| 735 | 60746 | 148254 | 
| 168 | 109 django/conf/locale/kn/formats.py | 5 | 22| 123 | 60869 | 148421 | 
| 169 | 109 django/middleware/csrf.py | 75 | 91| 195 | 61064 | 148421 | 
| 170 | **109 django/views/i18n.py** | 195 | 205| 137 | 61201 | 148421 | 
| 171 | 110 django/conf/urls/i18n.py | 23 | 40| 129 | 61330 | 148677 | 
| 172 | 111 django/conf/locale/mn/formats.py | 5 | 22| 120 | 61450 | 148841 | 
| 173 | 112 django/conf/locale/te/formats.py | 5 | 22| 123 | 61573 | 149008 | 
| 174 | 113 django/conf/locale/mk/formats.py | 5 | 43| 672 | 62245 | 149725 | 
| 175 | 114 django/conf/locale/he/formats.py | 5 | 22| 142 | 62387 | 149911 | 
| 176 | 115 django/conf/locale/bg/formats.py | 5 | 22| 131 | 62518 | 150086 | 
| 177 | 116 django/conf/locale/ar/formats.py | 5 | 22| 135 | 62653 | 150265 | 
| 178 | 117 django/conf/locale/hr/formats.py | 22 | 48| 620 | 63273 | 151148 | 
| 179 | 117 django/template/defaultfilters.py | 307 | 321| 111 | 63384 | 151148 | 
| 180 | 117 django/http/response.py | 450 | 468| 186 | 63570 | 151148 | 
| 181 | 118 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 64205 | 151828 | 
| 182 | 119 django/contrib/admin/sites.py | 1 | 29| 175 | 64380 | 156019 | 
| 183 | 120 django/contrib/admin/widgets.py | 344 | 370| 328 | 64708 | 159817 | 
| 184 | 121 django/conf/locale/ka/formats.py | 23 | 48| 564 | 65272 | 160724 | 
| 185 | 122 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 65907 | 161404 | 
| 186 | 122 django/http/response.py | 505 | 530| 159 | 66066 | 161404 | 
| 187 | 123 django/conf/locale/ko/formats.py | 32 | 53| 438 | 66504 | 162347 | 
| 188 | 123 django/conf/locale/hr/formats.py | 5 | 21| 218 | 66722 | 162347 | 
| 189 | 124 django/contrib/auth/hashers.py | 80 | 102| 167 | 66889 | 167134 | 
| 190 | 125 django/middleware/gzip.py | 1 | 53| 405 | 67294 | 167540 | 
| 191 | 126 django/core/mail/message.py | 184 | 192| 115 | 67409 | 171265 | 
| 192 | 127 django/views/defaults.py | 1 | 62| 485 | 67894 | 172209 | 
| 193 | 127 django/template/defaultfilters.py | 56 | 91| 203 | 68097 | 172209 | 
| 194 | 128 django/conf/locale/ta/formats.py | 5 | 22| 125 | 68222 | 172378 | 
| 195 | 128 django/http/response.py | 471 | 502| 153 | 68375 | 172378 | 
| 196 | 128 django/utils/translation/trans_real.py | 1 | 56| 482 | 68857 | 172378 | 
| 197 | 128 django/middleware/cache.py | 156 | 190| 272 | 69129 | 172378 | 
| 198 | 128 django/contrib/messages/storage/cookie.py | 94 | 121| 261 | 69390 | 172378 | 
| 199 | 129 django/conf/locale/it/formats.py | 21 | 46| 564 | 69954 | 173275 | 
| 200 | 129 django/conf/locale/sl/formats.py | 5 | 20| 208 | 70162 | 173275 | 
| 201 | 130 django/conf/locale/__init__.py | 1 | 576| 75 | 70237 | 177268 | 
| 202 | 130 django/contrib/sessions/backends/base.py | 1 | 35| 202 | 70439 | 177268 | 
| 203 | 130 django/middleware/csrf.py | 57 | 72| 161 | 70600 | 177268 | 
| 204 | 131 django/contrib/redirects/middleware.py | 1 | 51| 355 | 70955 | 177624 | 
| 205 | 132 django/utils/log.py | 1 | 76| 492 | 71447 | 179232 | 
| 206 | 132 django/utils/translation/__init__.py | 271 | 305| 262 | 71709 | 179232 | 
| 207 | 132 django/db/utils.py | 181 | 224| 295 | 72004 | 179232 | 
| 208 | 132 django/contrib/auth/views.py | 63 | 102| 292 | 72296 | 179232 | 
| 209 | 133 django/contrib/sites/middleware.py | 1 | 13| 0 | 72296 | 179291 | 
| 210 | **133 django/conf/global_settings.py** | 51 | 144| 1087 | 73383 | 179291 | 
| 211 | 133 django/contrib/auth/middleware.py | 27 | 45| 178 | 73561 | 179291 | 
| 212 | 133 django/contrib/messages/storage/cookie.py | 52 | 76| 235 | 73796 | 179291 | 
| 213 | 134 django/conf/locale/vi/formats.py | 5 | 22| 179 | 73975 | 179514 | 
| 214 | 134 django/conf/locale/ka/formats.py | 5 | 22| 298 | 74273 | 179514 | 
| 215 | 134 django/contrib/auth/__init__.py | 86 | 131| 392 | 74665 | 179514 | 
| 216 | 135 django/conf/locale/fa/formats.py | 5 | 22| 149 | 74814 | 179707 | 
| 217 | 135 django/contrib/admin/tests.py | 148 | 162| 160 | 74974 | 179707 | 
| 218 | 136 django/urls/base.py | 93 | 160| 381 | 75355 | 180901 | 
| 219 | 136 django/http/request.py | 72 | 89| 183 | 75538 | 180901 | 
| 220 | 136 django/http/response.py | 412 | 447| 339 | 75877 | 180901 | 


## Patch

```diff
diff --git a/django/conf/global_settings.py b/django/conf/global_settings.py
--- a/django/conf/global_settings.py
+++ b/django/conf/global_settings.py
@@ -154,6 +154,9 @@ def gettext_noop(s):
 LANGUAGE_COOKIE_AGE = None
 LANGUAGE_COOKIE_DOMAIN = None
 LANGUAGE_COOKIE_PATH = '/'
+LANGUAGE_COOKIE_SECURE = False
+LANGUAGE_COOKIE_HTTPONLY = False
+LANGUAGE_COOKIE_SAMESITE = None
 
 
 # If you set this to True, Django will format dates, numbers and calendars
diff --git a/django/views/i18n.py b/django/views/i18n.py
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -55,6 +55,9 @@ def set_language(request):
                 max_age=settings.LANGUAGE_COOKIE_AGE,
                 path=settings.LANGUAGE_COOKIE_PATH,
                 domain=settings.LANGUAGE_COOKIE_DOMAIN,
+                secure=settings.LANGUAGE_COOKIE_SECURE,
+                httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
+                samesite=settings.LANGUAGE_COOKIE_SAMESITE,
             )
     return response
 

```

## Test Patch

```diff
diff --git a/tests/view_tests/tests/test_i18n.py b/tests/view_tests/tests/test_i18n.py
--- a/tests/view_tests/tests/test_i18n.py
+++ b/tests/view_tests/tests/test_i18n.py
@@ -45,6 +45,9 @@ def test_setlang(self):
         self.assertEqual(language_cookie['domain'], '')
         self.assertEqual(language_cookie['path'], '/')
         self.assertEqual(language_cookie['max-age'], '')
+        self.assertEqual(language_cookie['httponly'], '')
+        self.assertEqual(language_cookie['samesite'], '')
+        self.assertEqual(language_cookie['secure'], '')
 
     def test_setlang_unsafe_next(self):
         """
@@ -175,6 +178,9 @@ def test_setlang_cookie(self):
             'LANGUAGE_COOKIE_AGE': 3600 * 7 * 2,
             'LANGUAGE_COOKIE_DOMAIN': '.example.com',
             'LANGUAGE_COOKIE_PATH': '/test/',
+            'LANGUAGE_COOKIE_HTTPONLY': True,
+            'LANGUAGE_COOKIE_SAMESITE': 'Strict',
+            'LANGUAGE_COOKIE_SECURE': True,
         }
         with self.settings(**test_settings):
             post_data = {'language': 'pl', 'next': '/views/'}
@@ -184,6 +190,9 @@ def test_setlang_cookie(self):
             self.assertEqual(language_cookie['domain'], '.example.com')
             self.assertEqual(language_cookie['path'], '/test/')
             self.assertEqual(language_cookie['max-age'], 3600 * 7 * 2)
+            self.assertEqual(language_cookie['httponly'], True)
+            self.assertEqual(language_cookie['samesite'], 'Strict')
+            self.assertEqual(language_cookie['secure'], True)
 
     def test_setlang_decodes_http_referer_url(self):
         """

```


## Code snippets

### 1 - django/conf/global_settings.py:

Start line: 145, End line: 260

```python
LANGUAGES_BIDI = ["he", "ar", "fa", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = 'django_language'
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = '/'


# If you set this to True, Django will format dates, numbers and calendars
# according to user current locale.
USE_L10N = False

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default charset to use for all HttpResponse objects, if a MIME type isn't
# manually specified. It's used to construct the Content-Type header.
DEFAULT_CHARSET = 'utf-8'

# Encoding of files read from disk (template and initial SQL files).
FILE_CHARSET = 'utf-8'

# Email address that error messages come from.
SERVER_EMAIL = 'root@localhost'

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'

# Host for sending email.
EMAIL_HOST = 'localhost'

# Port for sending email.
EMAIL_PORT = 25

# Whether to send SMTP 'Date' header in the local time zone or in UTC.
EMAIL_USE_LOCALTIME = False

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ''
EMAIL_HOST_PASSWORD = ''
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default form rendering class.
FORM_RENDERER = 'django.forms.renderers.DjangoTemplates'

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = 'webmaster@localhost'

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = '[Django] '

# Whether to append trailing slashes to URLs.
APPEND_SLASH = True

# Whether to prepend the "www." subdomain to URLs that don't have it.
PREPEND_WWW = False

# Override the server-derived value of SCRIPT_NAME
FORCE_SCRIPT_NAME = None

# List of compiled regular expression objects representing User-Agent strings
# that are not allowed to visit any page, systemwide. Use this for bad
# robots/crawlers. Here are a few examples:
#     import re
#     DISALLOWED_USER_AGENTS = [
#         re.compile(r'^NaverBot.*'),
#         re.compile(r'^EmailSiphon.*'),
#         re.compile(r'^SiteSucker.*'),
#         re.compile(r'^sohu-search'),
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$'),
#        re.compile(r'^/robots.txt$'),
#        re.compile(r'^/phpmyadmin/'),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
```
### 2 - django/conf/global_settings.py:

Start line: 496, End line: 635

```python
AUTH_USER_MODEL = 'auth.User'

AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend']

LOGIN_URL = '/accounts/login/'

LOGIN_REDIRECT_URL = '/accounts/profile/'

LOGOUT_REDIRECT_URL = None

# The number of days a password reset link is valid for
PASSWORD_RESET_TIMEOUT_DAYS = 3

# the first hasher in this list is the preferred algorithm.  any
# password using different algorithms will be converted automatically
# upon login
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]

AUTH_PASSWORD_VALIDATORS = []

###########
# SIGNING #
###########

SIGNING_BACKEND = 'django.core.signing.TimestampSigner'

########
# CSRF #
########

# Dotted path to callable to be used as view when a request is
# rejected by the CSRF middleware.
CSRF_FAILURE_VIEW = 'django.views.csrf.csrf_failure'

# Settings for CSRF cookie.
CSRF_COOKIE_NAME = 'csrftoken'
CSRF_COOKIE_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_DOMAIN = None
CSRF_COOKIE_PATH = '/'
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_HEADER_NAME = 'HTTP_X_CSRFTOKEN'
CSRF_TRUSTED_ORIGINS = []
CSRF_USE_SESSIONS = False

############
# MESSAGES #
############

# Class to use as messages backend
MESSAGE_STORAGE = 'django.contrib.messages.storage.fallback.FallbackStorage'

# Default values of MESSAGE_LEVEL and MESSAGE_TAGS are defined within
# django.contrib.messages to avoid imports in this settings file.

###########
# LOGGING #
###########

# The callable to use to configure logging
LOGGING_CONFIG = 'logging.config.dictConfig'

# Custom logging configuration.
LOGGING = {}

# Default exception reporter filter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER_FILTER = 'django.views.debug.SafeExceptionReporterFilter'

###########
# TESTING #
###########

# The name of the class to use to run the test suite
TEST_RUNNER = 'django.test.runner.DiscoverRunner'

# Apps that don't need to be serialized at test database creation time
# (only apps with migrations are to start with)
TEST_NON_SERIALIZED_APPS = []

############
# FIXTURES #
############

# The list of directories to search for fixtures
FIXTURE_DIRS = []

###############
# STATICFILES #
###############

# A list of locations of additional static files
STATICFILES_DIRS = []

# The default file storage backend used during the build process
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
]

##############
# MIGRATIONS #
##############

# Migration module overrides for apps, by app label.
MIGRATION_MODULES = {}

#################
# SYSTEM CHECKS #
#################

# List of all issues generated by system checks that should be silenced. Light
# issues like warnings, infos or debugs will not generate a message. Silencing
# serious issues like errors and criticals does not result in hiding the
# message, but Django will not stop you from e.g. running server.
SILENCED_SYSTEM_CHECKS = []

#######################
# SECURITY MIDDLEWARE #
#######################
SECURE_BROWSER_XSS_FILTER = False
SECURE_CONTENT_TYPE_NOSNIFF = False
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_HSTS_SECONDS = 0
SECURE_REDIRECT_EXEMPT = []
SECURE_SSL_HOST = None
SECURE_SSL_REDIRECT = False
```
### 3 - django/core/checks/security/base.py:

Start line: 1, End line: 86

```python
from django.conf import settings

from .. import Tags, Warning, register

SECRET_KEY_MIN_LENGTH = 50
SECRET_KEY_MIN_UNIQUE_CHARACTERS = 5

W001 = Warning(
    "You do not have 'django.middleware.security.SecurityMiddleware' "
    "in your MIDDLEWARE so the SECURE_HSTS_SECONDS, "
    "SECURE_CONTENT_TYPE_NOSNIFF, "
    "SECURE_BROWSER_XSS_FILTER, and SECURE_SSL_REDIRECT settings "
    "will have no effect.",
    id='security.W001',
)

W002 = Warning(
    "You do not have "
    "'django.middleware.clickjacking.XFrameOptionsMiddleware' in your "
    "MIDDLEWARE, so your pages will not be served with an "
    "'x-frame-options' header. Unless there is a good reason for your "
    "site to be served in a frame, you should consider enabling this "
    "header to help prevent clickjacking attacks.",
    id='security.W002',
)

W004 = Warning(
    "You have not set a value for the SECURE_HSTS_SECONDS setting. "
    "If your entire site is served only over SSL, you may want to consider "
    "setting a value and enabling HTTP Strict Transport Security. "
    "Be sure to read the documentation first; enabling HSTS carelessly "
    "can cause serious, irreversible problems.",
    id='security.W004',
)

W005 = Warning(
    "You have not set the SECURE_HSTS_INCLUDE_SUBDOMAINS setting to True. "
    "Without this, your site is potentially vulnerable to attack "
    "via an insecure connection to a subdomain. Only set this to True if "
    "you are certain that all subdomains of your domain should be served "
    "exclusively via SSL.",
    id='security.W005',
)

W006 = Warning(
    "Your SECURE_CONTENT_TYPE_NOSNIFF setting is not set to True, "
    "so your pages will not be served with an "
    "'X-Content-Type-Options: nosniff' header. "
    "You should consider enabling this header to prevent the "
    "browser from identifying content types incorrectly.",
    id='security.W006',
)

W007 = Warning(
    "Your SECURE_BROWSER_XSS_FILTER setting is not set to True, "
    "so your pages will not be served with an "
    "'X-XSS-Protection: 1; mode=block' header. "
    "You should consider enabling this header to activate the "
    "browser's XSS filtering and help prevent XSS attacks.",
    id='security.W007',
)

W008 = Warning(
    "Your SECURE_SSL_REDIRECT setting is not set to True. "
    "Unless your site should be available over both SSL and non-SSL "
    "connections, you may want to either set this setting True "
    "or configure a load balancer or reverse-proxy server "
    "to redirect all connections to HTTPS.",
    id='security.W008',
)

W009 = Warning(
    "Your SECRET_KEY has less than %(min_length)s characters or less than "
    "%(min_unique_chars)s unique characters. Please generate a long and random "
    "SECRET_KEY, otherwise many of Django's security-critical features will be "
    "vulnerable to attack." % {
        'min_length': SECRET_KEY_MIN_LENGTH,
        'min_unique_chars': SECRET_KEY_MIN_UNIQUE_CHARACTERS,
    },
    id='security.W009',
)

W018 = Warning(
    "You should not have DEBUG set to True in deployment.",
    id='security.W018',
)
```
### 4 - django/conf/global_settings.py:

Start line: 395, End line: 494

```python
FIRST_DAY_OF_WEEK = 0

# Decimal separator symbol
DECIMAL_SEPARATOR = '.'

# Boolean that sets whether to add thousand separator when formatting numbers
USE_THOUSAND_SEPARATOR = False

# Number of digits that will be together, when splitting them by
# THOUSAND_SEPARATOR. 0 means no grouping, 3 means splitting by thousands...
NUMBER_GROUPING = 0

# Thousand separator symbol
THOUSAND_SEPARATOR = ','

# The tablespaces to use for each model when not specified otherwise.
DEFAULT_TABLESPACE = ''
DEFAULT_INDEX_TABLESPACE = ''

# Default X-Frame-Options header value
X_FRAME_OPTIONS = 'SAMEORIGIN'

USE_X_FORWARDED_HOST = False
USE_X_FORWARDED_PORT = False

# The Python dotted path to the WSGI application that Django's internal server
# (runserver) will use. If `None`, the return value of
# 'django.core.wsgi.get_wsgi_application' is used, thus preserving the same
# behavior as previous versions of Django. Otherwise this should point to an
# actual WSGI application object.
WSGI_APPLICATION = None

# If your Django app is behind a proxy that sets a header to specify secure
# connections, AND that proxy ensures that user-submitted headers with the
# same name are ignored (so that people can't spoof it), set this value to
# a tuple of (header_name, header_value). For any requests that come in with
# that header/value, request.is_secure() will return True.
# WARNING! Only set this if you fully understand what you're doing. Otherwise,
# you may be opening yourself up to a security risk.
SECURE_PROXY_SSL_HEADER = None

##############
# MIDDLEWARE #
##############

# List of middleware to use. Order is important; in the request phase, these
# middleware will be applied in the order given, and in the response
# phase the middleware will be applied in reverse order.
MIDDLEWARE = []

############
# SESSIONS #
############

# Cache to store session data if using the cache session backend.
SESSION_CACHE_ALIAS = 'default'
# Cookie name. This can be whatever you want.
SESSION_COOKIE_NAME = 'sessionid'
# Age of cookie, in seconds (default: 2 weeks).
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7 * 2
# A string like "example.com", or None for standard domain cookie.
SESSION_COOKIE_DOMAIN = None
# Whether the session cookie should be secure (https:// only).
SESSION_COOKIE_SECURE = False
# The path of the session cookie.
SESSION_COOKIE_PATH = '/'
# Whether to use the HttpOnly flag.
SESSION_COOKIE_HTTPONLY = True
# Whether to set the flag restricting cookie leaks on cross-site requests.
# This can be 'Lax', 'Strict', or None to disable the flag.
SESSION_COOKIE_SAMESITE = 'Lax'
# Whether to save the session data on every request.
SESSION_SAVE_EVERY_REQUEST = False
# Whether a user's session cookie expires when the Web browser is closed.
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
# The module to store session data
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
# Directory to store session files if using the file session module. If None,
# the backend will use a sensible default.
SESSION_FILE_PATH = None
# class to serialize session data
SESSION_SERIALIZER = 'django.contrib.sessions.serializers.JSONSerializer'

#########
# CACHE #
#########

# The cache backends to use.
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}
CACHE_MIDDLEWARE_KEY_PREFIX = ''
CACHE_MIDDLEWARE_SECONDS = 600
CACHE_MIDDLEWARE_ALIAS = 'default'

##################
# AUTHENTICATION #
##################
```
### 5 - django/core/checks/security/sessions.py:

Start line: 1, End line: 98

```python
from django.conf import settings

from .. import Tags, Warning, register


def add_session_cookie_message(message):
    return message + (
        " Using a secure-only session cookie makes it more difficult for "
        "network traffic sniffers to hijack user sessions."
    )


W010 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_SECURE to True."
    ),
    id='security.W010',
)

W011 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_SECURE to True."
    ),
    id='security.W011',
)

W012 = Warning(
    add_session_cookie_message("SESSION_COOKIE_SECURE is not set to True."),
    id='security.W012',
)


def add_httponly_message(message):
    return message + (
        " Using an HttpOnly session cookie makes it more difficult for "
        "cross-site scripting attacks to hijack user sessions."
    )


W013 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_HTTPONLY to True.",
    ),
    id='security.W013',
)

W014 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_HTTPONLY to True."
    ),
    id='security.W014',
)

W015 = Warning(
    add_httponly_message("SESSION_COOKIE_HTTPONLY is not set to True."),
    id='security.W015',
)


@register(Tags.security, deploy=True)
def check_session_cookie_secure(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_SECURE:
        if _session_app():
            errors.append(W010)
        if _session_middleware():
            errors.append(W011)
        if len(errors) > 1:
            errors = [W012]
    return errors


@register(Tags.security, deploy=True)
def check_session_cookie_httponly(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_HTTPONLY:
        if _session_app():
            errors.append(W013)
        if _session_middleware():
            errors.append(W014)
        if len(errors) > 1:
            errors = [W015]
    return errors


def _session_middleware():
    return 'django.contrib.sessions.middleware.SessionMiddleware' in settings.MIDDLEWARE


def _session_app():
    return "django.contrib.sessions" in settings.INSTALLED_APPS
```
### 6 - django/views/i18n.py:

Start line: 23, End line: 59

```python
def set_language(request):
    """
    Redirect to a given URL while setting the chosen language in the session
    (if enabled) and in a cookie. The URL and the language code need to be
    specified in the request parameters.

    Since this view changes how the user will see the rest of the site, it must
    only be accessed as a POST request. If called as a GET request, it will
    redirect to the page in the request (the 'next' parameter) without changing
    any state.
    """
    next = request.POST.get('next', request.GET.get('next'))
    if ((next or not request.is_ajax()) and
            not is_safe_url(url=next, allowed_hosts={request.get_host()}, require_https=request.is_secure())):
        next = request.META.get('HTTP_REFERER')
        next = next and unquote(next)  # HTTP_REFERER may be encoded.
        if not is_safe_url(url=next, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
            next = '/'
    response = HttpResponseRedirect(next) if next else HttpResponse(status=204)
    if request.method == 'POST':
        lang_code = request.POST.get(LANGUAGE_QUERY_PARAMETER)
        if lang_code and check_for_language(lang_code):
            if next:
                next_trans = translate_url(next, lang_code)
                if next_trans != next:
                    response = HttpResponseRedirect(next_trans)
            if hasattr(request, 'session'):
                # Storing the language in the session is deprecated.
                # (RemovedInDjango40Warning)
                request.session[LANGUAGE_SESSION_KEY] = lang_code
            response.set_cookie(
                settings.LANGUAGE_COOKIE_NAME, lang_code,
                max_age=settings.LANGUAGE_COOKIE_AGE,
                path=settings.LANGUAGE_COOKIE_PATH,
                domain=settings.LANGUAGE_COOKIE_DOMAIN,
            )
    return response
```
### 7 - django/http/response.py:

Start line: 157, End line: 201

```python
class HttpResponseBase:

    def set_cookie(self, key, value='', max_age=None, expires=None, path='/',
                   domain=None, secure=False, httponly=False, samesite=None):
        """
        Set a cookie.

        ``expires`` can be:
        - a string in the correct format,
        - a naive ``datetime.datetime`` object in UTC,
        - an aware ``datetime.datetime`` object in any time zone.
        If it is a ``datetime.datetime`` object then calculate ``max_age``.
        """
        self.cookies[key] = value
        if expires is not None:
            if isinstance(expires, datetime.datetime):
                if timezone.is_aware(expires):
                    expires = timezone.make_naive(expires, timezone.utc)
                delta = expires - expires.utcnow()
                # Add one second so the date matches exactly (a fraction of
                # time gets lost between converting to a timedelta and
                # then the date string).
                delta = delta + datetime.timedelta(seconds=1)
                # Just set max_age - the max_age logic will set expires.
                expires = None
                max_age = max(0, delta.days * 86400 + delta.seconds)
            else:
                self.cookies[key]['expires'] = expires
        else:
            self.cookies[key]['expires'] = ''
        if max_age is not None:
            self.cookies[key]['max-age'] = max_age
            # IE requires expires, so set it if hasn't been already.
            if not expires:
                self.cookies[key]['expires'] = http_date(time.time() + max_age)
        if path is not None:
            self.cookies[key]['path'] = path
        if domain is not None:
            self.cookies[key]['domain'] = domain
        if secure:
            self.cookies[key]['secure'] = True
        if httponly:
            self.cookies[key]['httponly'] = True
        if samesite:
            if samesite.lower() not in ('lax', 'strict'):
                raise ValueError('samesite must be "lax" or "strict".')
            self.cookies[key]['samesite'] = samesite
```
### 8 - django/core/checks/security/base.py:

Start line: 88, End line: 190

```python
W019 = Warning(
    "You have "
    "'django.middleware.clickjacking.XFrameOptionsMiddleware' in your "
    "MIDDLEWARE, but X_FRAME_OPTIONS is not set to 'DENY'. "
    "The default is 'SAMEORIGIN', but unless there is a good reason for "
    "your site to serve other parts of itself in a frame, you should "
    "change it to 'DENY'.",
    id='security.W019',
)

W020 = Warning(
    "ALLOWED_HOSTS must not be empty in deployment.",
    id='security.W020',
)

W021 = Warning(
    "You have not set the SECURE_HSTS_PRELOAD setting to True. Without this, "
    "your site cannot be submitted to the browser preload list.",
    id='security.W021',
)


def _security_middleware():
    return 'django.middleware.security.SecurityMiddleware' in settings.MIDDLEWARE


def _xframe_middleware():
    return 'django.middleware.clickjacking.XFrameOptionsMiddleware' in settings.MIDDLEWARE


@register(Tags.security, deploy=True)
def check_security_middleware(app_configs, **kwargs):
    passed_check = _security_middleware()
    return [] if passed_check else [W001]


@register(Tags.security, deploy=True)
def check_xframe_options_middleware(app_configs, **kwargs):
    passed_check = _xframe_middleware()
    return [] if passed_check else [W002]


@register(Tags.security, deploy=True)
def check_sts(app_configs, **kwargs):
    passed_check = not _security_middleware() or settings.SECURE_HSTS_SECONDS
    return [] if passed_check else [W004]


@register(Tags.security, deploy=True)
def check_sts_include_subdomains(app_configs, **kwargs):
    passed_check = (
        not _security_middleware() or
        not settings.SECURE_HSTS_SECONDS or
        settings.SECURE_HSTS_INCLUDE_SUBDOMAINS is True
    )
    return [] if passed_check else [W005]


@register(Tags.security, deploy=True)
def check_sts_preload(app_configs, **kwargs):
    passed_check = (
        not _security_middleware() or
        not settings.SECURE_HSTS_SECONDS or
        settings.SECURE_HSTS_PRELOAD is True
    )
    return [] if passed_check else [W021]


@register(Tags.security, deploy=True)
def check_content_type_nosniff(app_configs, **kwargs):
    passed_check = (
        not _security_middleware() or
        settings.SECURE_CONTENT_TYPE_NOSNIFF is True
    )
    return [] if passed_check else [W006]


@register(Tags.security, deploy=True)
def check_xss_filter(app_configs, **kwargs):
    passed_check = (
        not _security_middleware() or
        settings.SECURE_BROWSER_XSS_FILTER is True
    )
    return [] if passed_check else [W007]


@register(Tags.security, deploy=True)
def check_ssl_redirect(app_configs, **kwargs):
    passed_check = (
        not _security_middleware() or
        settings.SECURE_SSL_REDIRECT is True
    )
    return [] if passed_check else [W008]


@register(Tags.security, deploy=True)
def check_secret_key(app_configs, **kwargs):
    passed_check = (
        getattr(settings, 'SECRET_KEY', None) and
        len(set(settings.SECRET_KEY)) >= SECRET_KEY_MIN_UNIQUE_CHARACTERS and
        len(settings.SECRET_KEY) >= SECRET_KEY_MIN_LENGTH
    )
    return [] if passed_check else [W009]
```
### 9 - django/core/checks/security/csrf.py:

Start line: 1, End line: 41

```python
from django.conf import settings

from .. import Tags, Warning, register

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
### 10 - django/middleware/security.py:

Start line: 1, End line: 28

```python
import re

from django.conf import settings
from django.http import HttpResponsePermanentRedirect
from django.utils.deprecation import MiddlewareMixin


class SecurityMiddleware(MiddlewareMixin):
    def __init__(self, get_response=None):
        self.sts_seconds = settings.SECURE_HSTS_SECONDS
        self.sts_include_subdomains = settings.SECURE_HSTS_INCLUDE_SUBDOMAINS
        self.sts_preload = settings.SECURE_HSTS_PRELOAD
        self.content_type_nosniff = settings.SECURE_CONTENT_TYPE_NOSNIFF
        self.xss_filter = settings.SECURE_BROWSER_XSS_FILTER
        self.redirect = settings.SECURE_SSL_REDIRECT
        self.redirect_host = settings.SECURE_SSL_HOST
        self.redirect_exempt = [re.compile(r) for r in settings.SECURE_REDIRECT_EXEMPT]
        self.get_response = get_response

    def process_request(self, request):
        path = request.path.lstrip("/")
        if (self.redirect and not request.is_secure() and
                not any(pattern.search(path)
                        for pattern in self.redirect_exempt)):
            host = self.redirect_host or request.get_host()
            return HttpResponsePermanentRedirect(
                "https://%s%s" % (host, request.get_full_path())
            )
```
### 11 - django/conf/global_settings.py:

Start line: 261, End line: 343

```python
SECRET_KEY = ''

# Default file storage mechanism that holds media.
DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/var/www/example.com/media/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT.
# Examples: "http://example.com/media/", "http://media.example.com/"
MEDIA_URL = ''

# Absolute path to the directory static files should be collected to.
# Example: "/var/www/example.com/static/"
STATIC_ROOT = None

# URL that handles the static files served from STATIC_ROOT.
# Example: "http://example.com/static/", "http://static.example.com/"
STATIC_URL = None

# List of upload handler classes to be applied in order.
FILE_UPLOAD_HANDLERS = [
    'django.core.files.uploadhandler.MemoryFileUploadHandler',
    'django.core.files.uploadhandler.TemporaryFileUploadHandler',
]

# Maximum size, in bytes, of a request before it will be streamed to the
# file system instead of into memory.
FILE_UPLOAD_MAX_MEMORY_SIZE = 2621440  # i.e. 2.5 MB

# Maximum size in bytes of request data (excluding file uploads) that will be
# read before a SuspiciousOperation (RequestDataTooBig) is raised.
DATA_UPLOAD_MAX_MEMORY_SIZE = 2621440  # i.e. 2.5 MB

# Maximum number of GET/POST parameters that will be read before a
# SuspiciousOperation (TooManyFieldsSent) is raised.
DATA_UPLOAD_MAX_NUMBER_FIELDS = 1000

# Directory in which upload streamed files will be temporarily saved. A value of
# `None` will make Django use the operating system's default temporary directory
# (i.e. "/tmp" on *nix systems).
FILE_UPLOAD_TEMP_DIR = None

# The numeric mode to set newly-uploaded files to. The value should be a mode
# you'd pass directly to os.chmod; see https://docs.python.org/library/os.html#files-and-directories.
FILE_UPLOAD_PERMISSIONS = 0o644

# The numeric mode to assign to newly-created directories, when uploading files.
# The value should be a mode as you'd pass to os.chmod;
# see https://docs.python.org/library/os.html#files-and-directories.
FILE_UPLOAD_DIRECTORY_PERMISSIONS = None

# Python module path where user will place custom format definition.
# The directory where this setting is pointing should contain subdirectories
# named as the locales, containing a formats.py file
# (i.e. "myproject.locale" for myproject/locale/en/formats.py etc. use)
FORMAT_MODULE_PATH = None

# Default formatting for date objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
DATE_FORMAT = 'N j, Y'

# Default formatting for datetime objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
DATETIME_FORMAT = 'N j, Y, P'

# Default formatting for time objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
TIME_FORMAT = 'P'

# Default formatting for date objects when only the year and month are relevant.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
YEAR_MONTH_FORMAT = 'F Y'

# Default formatting for date objects when only the month and day are relevant.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
MONTH_DAY_FORMAT = 'F j'

# Default short formatting for date objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
```
### 45 - django/conf/global_settings.py:

Start line: 1, End line: 50

```python
"""
Default Django settings. Override these with settings in the module pointed to
by the DJANGO_SETTINGS_MODULE environment variable.
"""


# This is defined here as a do-nothing function because we can't import
# django.utils.translation -- that module depends on the settings.
def gettext_noop(s):
    return s


####################
# CORE             #
####################

DEBUG = False

# Whether the framework should propagate raw exceptions rather than catching
# them. This is useful under some testing situations and should never be used
# on a live site.
DEBUG_PROPAGATE_EXCEPTIONS = False

# People who get code error notifications.
# In the format [('Full Name', 'email@example.com'), ('Full Name', 'anotheremail@example.com')]
ADMINS = []

# List of IP addresses, as strings, that:
#   * See debug comments, when DEBUG is true
#   * Receive x-headers
INTERNAL_IPS = []

# Hosts/domain names that are valid for this site.
# "*" matches anything, ".example.com" matches example.com and all subdomains
ALLOWED_HOSTS = []

# Local time zone for this installation. All choices can be found here:
# https://en.wikipedia.org/wiki/List_of_tz_zones_by_name (although not all
# systems may support all possibilities). When USE_TZ is True, this is
# interpreted as the default user time zone.
TIME_ZONE = 'America/Chicago'

# If you set this to True, Django will use timezone-aware datetimes.
USE_TZ = False

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

# Languages we provide translations for, out of the box.
```
### 51 - django/conf/global_settings.py:

Start line: 344, End line: 394

```python
SHORT_DATE_FORMAT = 'm/d/Y'

# Default short formatting for datetime objects.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
SHORT_DATETIME_FORMAT = 'm/d/Y P'

# Default formats to be used when parsing dates from input boxes, in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
DATE_INPUT_FORMATS = [
    '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y',  # '2006-10-25', '10/25/2006', '10/25/06'
    '%b %d %Y', '%b %d, %Y',             # 'Oct 25 2006', 'Oct 25, 2006'
    '%d %b %Y', '%d %b, %Y',             # '25 Oct 2006', '25 Oct, 2006'
    '%B %d %Y', '%B %d, %Y',             # 'October 25 2006', 'October 25, 2006'
    '%d %B %Y', '%d %B, %Y',             # '25 October 2006', '25 October, 2006'
]

# Default formats to be used when parsing times from input boxes, in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
TIME_INPUT_FORMATS = [
    '%H:%M:%S',     # '14:30:59'
    '%H:%M:%S.%f',  # '14:30:59.000200'
    '%H:%M',        # '14:30'
]

# Default formats to be used when parsing dates and times from input boxes,
# in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
DATETIME_INPUT_FORMATS = [
    '%Y-%m-%d %H:%M:%S',     # '2006-10-25 14:30:59'
    '%Y-%m-%d %H:%M:%S.%f',  # '2006-10-25 14:30:59.000200'
    '%Y-%m-%d %H:%M',        # '2006-10-25 14:30'
    '%Y-%m-%d',              # '2006-10-25'
    '%m/%d/%Y %H:%M:%S',     # '10/25/2006 14:30:59'
    '%m/%d/%Y %H:%M:%S.%f',  # '10/25/2006 14:30:59.000200'
    '%m/%d/%Y %H:%M',        # '10/25/2006 14:30'
    '%m/%d/%Y',              # '10/25/2006'
    '%m/%d/%y %H:%M:%S',     # '10/25/06 14:30:59'
    '%m/%d/%y %H:%M:%S.%f',  # '10/25/06 14:30:59.000200'
    '%m/%d/%y %H:%M',        # '10/25/06 14:30'
    '%m/%d/%y',              # '10/25/06'
]

# First day of week, to be used on calendars
# 0 means Sunday, 1 means Monday...
```
### 102 - django/views/i18n.py:

Start line: 74, End line: 177

```python
js_catalog_template = r"""
{% autoescape off %}
(function(globals) {

  var django = globals.django || (globals.django = {});

  {% if plural %}
  django.pluralidx = function(n) {
    var v={{ plural }};
    if (typeof(v) == 'boolean') {
      return v ? 1 : 0;
    } else {
      return v;
    }
  };
  {% else %}
  django.pluralidx = function(count) { return (count == 1) ? 0 : 1; };
  {% endif %}

  /* gettext library */

  django.catalog = django.catalog || {};
  {% if catalog_str %}
  var newcatalog = {{ catalog_str }};
  for (var key in newcatalog) {
    django.catalog[key] = newcatalog[key];
  }
  {% endif %}

  if (!django.jsi18n_initialized) {
    django.gettext = function(msgid) {
      var value = django.catalog[msgid];
      if (typeof(value) == 'undefined') {
        return msgid;
      } else {
        return (typeof(value) == 'string') ? value : value[0];
      }
    };

    django.ngettext = function(singular, plural, count) {
      var value = django.catalog[singular];
      if (typeof(value) == 'undefined') {
        return (count == 1) ? singular : plural;
      } else {
        return value.constructor === Array ? value[django.pluralidx(count)] : value;
      }
    };

    django.gettext_noop = function(msgid) { return msgid; };

    django.pgettext = function(context, msgid) {
      var value = django.gettext(context + '\x04' + msgid);
      if (value.indexOf('\x04') != -1) {
        value = msgid;
      }
      return value;
    };

    django.npgettext = function(context, singular, plural, count) {
      var value = django.ngettext(context + '\x04' + singular, context + '\x04' + plural, count);
      if (value.indexOf('\x04') != -1) {
        value = django.ngettext(singular, plural, count);
      }
      return value;
    };

    django.interpolate = function(fmt, obj, named) {
      if (named) {
        return fmt.replace(/%\(\w+\)s/g, function(match){return String(obj[match.slice(2,-2)])});
      } else {
        return fmt.replace(/%s/g, function(match){return String(obj.shift())});
      }
    };


    /* formatting library */

    django.formats = {{ formats_str }};

    django.get_format = function(format_type) {
      var value = django.formats[format_type];
      if (typeof(value) == 'undefined') {
        return format_type;
      } else {
        return value;
      }
    };

    /* add to global namespace */
    globals.pluralidx = django.pluralidx;
    globals.gettext = django.gettext;
    globals.ngettext = django.ngettext;
    globals.gettext_noop = django.gettext_noop;
    globals.pgettext = django.pgettext;
    globals.npgettext = django.npgettext;
    globals.interpolate = django.interpolate;
    globals.get_format = django.get_format;

    django.jsi18n_initialized = true;
  }

}(this));
{% endautoescape %}
"""
```
### 115 - django/views/i18n.py:

Start line: 1, End line: 20

```python
import itertools
import json
import os
import re
from urllib.parse import unquote

from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import is_safe_url
from django.utils.translation import (
    LANGUAGE_SESSION_KEY, check_for_language, get_language,
)
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View

LANGUAGE_QUERY_PARAMETER = 'language'
```
### 170 - django/views/i18n.py:

Start line: 195, End line: 205

```python
class JavaScriptCatalog(View):

    def get(self, request, *args, **kwargs):
        locale = get_language()
        domain = kwargs.get('domain', self.domain)
        # If packages are not provided, default to all installed packages, as
        # DjangoTranslation without localedirs harvests them all.
        packages = kwargs.get('packages', '')
        packages = packages.split('+') if packages else self.packages
        paths = self.get_paths(packages) if packages else None
        self.translation = DjangoTranslation(locale, domain=domain, localedirs=paths)
        context = self.get_context_data(**kwargs)
        return self.render_to_response(context)
```
### 210 - django/conf/global_settings.py:

Start line: 51, End line: 144

```python
LANGUAGES = [
    ('af', gettext_noop('Afrikaans')),
    ('ar', gettext_noop('Arabic')),
    ('ast', gettext_noop('Asturian')),
    ('az', gettext_noop('Azerbaijani')),
    ('bg', gettext_noop('Bulgarian')),
    ('be', gettext_noop('Belarusian')),
    ('bn', gettext_noop('Bengali')),
    ('br', gettext_noop('Breton')),
    ('bs', gettext_noop('Bosnian')),
    ('ca', gettext_noop('Catalan')),
    ('cs', gettext_noop('Czech')),
    ('cy', gettext_noop('Welsh')),
    ('da', gettext_noop('Danish')),
    ('de', gettext_noop('German')),
    ('dsb', gettext_noop('Lower Sorbian')),
    ('el', gettext_noop('Greek')),
    ('en', gettext_noop('English')),
    ('en-au', gettext_noop('Australian English')),
    ('en-gb', gettext_noop('British English')),
    ('eo', gettext_noop('Esperanto')),
    ('es', gettext_noop('Spanish')),
    ('es-ar', gettext_noop('Argentinian Spanish')),
    ('es-co', gettext_noop('Colombian Spanish')),
    ('es-mx', gettext_noop('Mexican Spanish')),
    ('es-ni', gettext_noop('Nicaraguan Spanish')),
    ('es-ve', gettext_noop('Venezuelan Spanish')),
    ('et', gettext_noop('Estonian')),
    ('eu', gettext_noop('Basque')),
    ('fa', gettext_noop('Persian')),
    ('fi', gettext_noop('Finnish')),
    ('fr', gettext_noop('French')),
    ('fy', gettext_noop('Frisian')),
    ('ga', gettext_noop('Irish')),
    ('gd', gettext_noop('Scottish Gaelic')),
    ('gl', gettext_noop('Galician')),
    ('he', gettext_noop('Hebrew')),
    ('hi', gettext_noop('Hindi')),
    ('hr', gettext_noop('Croatian')),
    ('hsb', gettext_noop('Upper Sorbian')),
    ('hu', gettext_noop('Hungarian')),
    ('hy', gettext_noop('Armenian')),
    ('ia', gettext_noop('Interlingua')),
    ('id', gettext_noop('Indonesian')),
    ('io', gettext_noop('Ido')),
    ('is', gettext_noop('Icelandic')),
    ('it', gettext_noop('Italian')),
    ('ja', gettext_noop('Japanese')),
    ('ka', gettext_noop('Georgian')),
    ('kab', gettext_noop('Kabyle')),
    ('kk', gettext_noop('Kazakh')),
    ('km', gettext_noop('Khmer')),
    ('kn', gettext_noop('Kannada')),
    ('ko', gettext_noop('Korean')),
    ('lb', gettext_noop('Luxembourgish')),
    ('lt', gettext_noop('Lithuanian')),
    ('lv', gettext_noop('Latvian')),
    ('mk', gettext_noop('Macedonian')),
    ('ml', gettext_noop('Malayalam')),
    ('mn', gettext_noop('Mongolian')),
    ('mr', gettext_noop('Marathi')),
    ('my', gettext_noop('Burmese')),
    ('nb', gettext_noop('Norwegian Bokmål')),
    ('ne', gettext_noop('Nepali')),
    ('nl', gettext_noop('Dutch')),
    ('nn', gettext_noop('Norwegian Nynorsk')),
    ('os', gettext_noop('Ossetic')),
    ('pa', gettext_noop('Punjabi')),
    ('pl', gettext_noop('Polish')),
    ('pt', gettext_noop('Portuguese')),
    ('pt-br', gettext_noop('Brazilian Portuguese')),
    ('ro', gettext_noop('Romanian')),
    ('ru', gettext_noop('Russian')),
    ('sk', gettext_noop('Slovak')),
    ('sl', gettext_noop('Slovenian')),
    ('sq', gettext_noop('Albanian')),
    ('sr', gettext_noop('Serbian')),
    ('sr-latn', gettext_noop('Serbian Latin')),
    ('sv', gettext_noop('Swedish')),
    ('sw', gettext_noop('Swahili')),
    ('ta', gettext_noop('Tamil')),
    ('te', gettext_noop('Telugu')),
    ('th', gettext_noop('Thai')),
    ('tr', gettext_noop('Turkish')),
    ('tt', gettext_noop('Tatar')),
    ('udm', gettext_noop('Udmurt')),
    ('uk', gettext_noop('Ukrainian')),
    ('ur', gettext_noop('Urdu')),
    ('vi', gettext_noop('Vietnamese')),
    ('zh-hans', gettext_noop('Simplified Chinese')),
    ('zh-hant', gettext_noop('Traditional Chinese')),
]

# Languages using BiDi (right-to-left) layout
```
