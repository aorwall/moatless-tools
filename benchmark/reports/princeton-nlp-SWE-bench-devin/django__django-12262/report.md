# django__django-12262

| **django/django** | `69331bb851c34f05bc77e9fc24020fe6908b9cd5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1129 |
| **Any found context length** | 1129 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/template/library.py b/django/template/library.py
--- a/django/template/library.py
+++ b/django/template/library.py
@@ -261,7 +261,7 @@ def parse_bits(parser, bits, params, varargs, varkw, defaults,
         if kwarg:
             # The kwarg was successfully extracted
             param, value = kwarg.popitem()
-            if param not in params and param not in unhandled_kwargs and varkw is None:
+            if param not in params and param not in kwonly and varkw is None:
                 # An unexpected keyword argument was supplied
                 raise TemplateSyntaxError(
                     "'%s' received unexpected keyword argument '%s'" %

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/template/library.py | 264 | 264 | 3 | 2 | 1129


## Problem Statement

```
Custom template tags raise TemplateSyntaxError when keyword-only arguments with defaults are provided.
Description
	 
		(last modified by P-Seebauer)
	 
When creating simple tags without variable keyword args, but an keyword argument with a default value. It's not possible to supply any other variable.
@register.simple_tag
def hello(*, greeting='hello'):
	return f'{greeting} world'
{% hello greeting='hi' %}
Raises “'hello' received unexpected keyword argument 'greeting'”
Also supplying a keyword argument a second time raises the wrong error message:
#tag
@register.simple_tag
def hi(*, greeting):
	return f'{greeting} world'
{% hi greeting='hi' greeting='hello' %}
Raises “'hi' received unexpected keyword argument 'greeting'”
instead of "'hi' received multiple values for keyword argument 'greeting'"
Same goes for inclusion tags (is the same code) I already have a fix ready, will push it after creating the ticket (that I have a ticket# for the commit).
Is actually for all versions since the offending line is from 2.0…

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/template/defaulttags.py | 1 | 48| 314 | 314 | 11041 | 
| 2 | 1 django/template/defaulttags.py | 517 | 540| 188 | 502 | 11041 | 
| **-> 3 <-** | **2 django/template/library.py** | 237 | 309| 627 | 1129 | 13579 | 
| 4 | 2 django/template/defaulttags.py | 1280 | 1312| 287 | 1416 | 13579 | 
| 5 | 2 django/template/defaulttags.py | 51 | 68| 152 | 1568 | 13579 | 
| 6 | 2 django/template/defaulttags.py | 839 | 888| 268 | 1836 | 13579 | 
| 7 | 2 django/template/defaulttags.py | 499 | 514| 142 | 1978 | 13579 | 
| 8 | 2 django/template/defaulttags.py | 822 | 836| 154 | 2132 | 13579 | 
| 9 | 2 django/template/defaulttags.py | 1223 | 1246| 176 | 2308 | 13579 | 
| 10 | 2 django/template/defaulttags.py | 156 | 217| 532 | 2840 | 13579 | 
| 11 | **2 django/template/library.py** | 1 | 52| 344 | 3184 | 13579 | 
| 12 | 2 django/template/defaulttags.py | 1315 | 1379| 504 | 3688 | 13579 | 
| 13 | 2 django/template/defaulttags.py | 1443 | 1476| 246 | 3934 | 13579 | 
| 14 | 2 django/template/defaulttags.py | 421 | 455| 270 | 4204 | 13579 | 
| 15 | 3 django/template/base.py | 1 | 94| 779 | 4983 | 21460 | 
| 16 | **3 django/template/library.py** | 164 | 181| 157 | 5140 | 21460 | 
| 17 | 3 django/template/defaulttags.py | 402 | 418| 121 | 5261 | 21460 | 
| 18 | **3 django/template/library.py** | 184 | 198| 119 | 5380 | 21460 | 
| 19 | 3 django/template/base.py | 507 | 535| 244 | 5624 | 21460 | 
| 20 | 3 django/template/defaulttags.py | 1382 | 1402| 121 | 5745 | 21460 | 
| 21 | **3 django/template/library.py** | 96 | 134| 324 | 6069 | 21460 | 
| 22 | 3 django/template/defaulttags.py | 220 | 248| 260 | 6329 | 21460 | 
| 23 | 3 django/template/defaulttags.py | 543 | 628| 803 | 7132 | 21460 | 
| 24 | 3 django/template/defaulttags.py | 315 | 333| 151 | 7283 | 21460 | 
| 25 | 3 django/template/defaulttags.py | 1053 | 1081| 225 | 7508 | 21460 | 
| 26 | 3 django/template/defaulttags.py | 71 | 93| 175 | 7683 | 21460 | 
| 27 | 4 django/template/backends/dummy.py | 1 | 53| 325 | 8008 | 21785 | 
| 28 | 4 django/template/defaulttags.py | 134 | 154| 183 | 8191 | 21785 | 
| 29 | 4 django/template/defaulttags.py | 891 | 975| 686 | 8877 | 21785 | 
| 30 | 5 django/contrib/admin/templatetags/base.py | 1 | 20| 173 | 9050 | 22084 | 
| 31 | 6 django/template/loader_tags.py | 272 | 318| 392 | 9442 | 24635 | 
| 32 | 6 django/template/defaulttags.py | 363 | 399| 211 | 9653 | 24635 | 
| 33 | 6 django/template/defaulttags.py | 96 | 131| 224 | 9877 | 24635 | 
| 34 | 6 django/template/base.py | 705 | 724| 179 | 10056 | 24635 | 
| 35 | 6 django/template/loader_tags.py | 250 | 269| 239 | 10295 | 24635 | 
| 36 | 7 django/template/defaultfilters.py | 640 | 668| 212 | 10507 | 30709 | 
| 37 | 7 django/template/defaulttags.py | 282 | 312| 171 | 10678 | 30709 | 
| 38 | 7 django/template/defaulttags.py | 458 | 496| 252 | 10930 | 30709 | 
| 39 | 7 django/template/defaulttags.py | 1249 | 1277| 164 | 11094 | 30709 | 
| 40 | 8 django/core/checks/templates.py | 1 | 36| 259 | 11353 | 30969 | 
| 41 | 9 django/template/backends/django.py | 79 | 111| 225 | 11578 | 31825 | 
| 42 | 10 django/template/utils.py | 1 | 62| 401 | 11979 | 32533 | 
| 43 | 11 django/template/exceptions.py | 1 | 43| 262 | 12241 | 32796 | 
| 44 | 11 django/template/base.py | 96 | 137| 208 | 12449 | 32796 | 
| 45 | 11 django/template/defaulttags.py | 631 | 678| 331 | 12780 | 32796 | 
| 46 | 11 django/template/base.py | 727 | 790| 531 | 13311 | 32796 | 
| 47 | 11 django/template/defaultfilters.py | 240 | 305| 427 | 13738 | 32796 | 
| 48 | 11 django/template/defaulttags.py | 681 | 727| 313 | 14051 | 32796 | 
| 49 | 11 django/template/defaulttags.py | 1084 | 1128| 370 | 14421 | 32796 | 
| 50 | 11 django/template/defaulttags.py | 250 | 260| 140 | 14561 | 32796 | 
| 51 | 11 django/template/defaulttags.py | 263 | 279| 187 | 14748 | 32796 | 
| 52 | 12 django/templatetags/cache.py | 1 | 49| 413 | 15161 | 33523 | 
| 53 | 12 django/template/defaulttags.py | 336 | 360| 231 | 15392 | 33523 | 
| 54 | 13 django/views/defaults.py | 100 | 119| 149 | 15541 | 34565 | 
| 55 | 14 django/core/checks/registry.py | 1 | 55| 312 | 15853 | 35193 | 
| 56 | 15 django/contrib/staticfiles/storage.py | 366 | 409| 328 | 16181 | 38723 | 
| 57 | 15 django/template/defaulttags.py | 978 | 1017| 364 | 16545 | 38723 | 
| 58 | 15 django/template/defaultfilters.py | 56 | 91| 203 | 16748 | 38723 | 
| 59 | 16 django/template/smartif.py | 150 | 209| 423 | 17171 | 40249 | 
| 60 | 16 django/template/defaultfilters.py | 221 | 237| 116 | 17287 | 40249 | 
| 61 | 17 django/template/response.py | 1 | 43| 383 | 17670 | 41328 | 
| 62 | 18 django/templatetags/static.py | 93 | 120| 195 | 17865 | 42292 | 
| 63 | 19 django/views/generic/base.py | 1 | 27| 148 | 18013 | 43895 | 
| 64 | 20 django/template/__init__.py | 1 | 69| 360 | 18373 | 44255 | 
| 65 | 21 django/template/engine.py | 1 | 53| 388 | 18761 | 45565 | 
| 66 | 21 django/template/base.py | 537 | 569| 207 | 18968 | 45565 | 
| 67 | 21 django/template/smartif.py | 114 | 147| 188 | 19156 | 45565 | 
| 68 | 21 django/template/defaultfilters.py | 31 | 53| 191 | 19347 | 45565 | 
| 69 | 21 django/template/defaulttags.py | 1131 | 1151| 160 | 19507 | 45565 | 
| 70 | 22 django/core/checks/security/base.py | 1 | 83| 732 | 20239 | 47346 | 
| 71 | 22 django/template/defaulttags.py | 1154 | 1220| 648 | 20887 | 47346 | 
| 72 | 23 django/utils/translation/template.py | 1 | 32| 376 | 21263 | 49303 | 
| 73 | 23 django/template/loader_tags.py | 1 | 38| 182 | 21445 | 49303 | 
| 74 | 23 django/template/smartif.py | 1 | 40| 299 | 21744 | 49303 | 
| 75 | 24 django/contrib/humanize/templatetags/humanize.py | 82 | 128| 578 | 22322 | 52463 | 
| 76 | 25 django/contrib/admindocs/views.py | 56 | 84| 285 | 22607 | 55771 | 
| 77 | 25 django/template/defaultfilters.py | 325 | 409| 499 | 23106 | 55771 | 
| 78 | 25 django/views/defaults.py | 1 | 24| 149 | 23255 | 55771 | 
| 79 | **25 django/template/library.py** | 136 | 161| 219 | 23474 | 55771 | 
| 80 | 26 django/utils/cache.py | 38 | 103| 557 | 24031 | 59519 | 
| 81 | 26 django/template/base.py | 140 | 172| 255 | 24286 | 59519 | 
| 82 | 26 django/template/base.py | 572 | 607| 361 | 24647 | 59519 | 
| 83 | 26 django/template/defaulttags.py | 1020 | 1050| 195 | 24842 | 59519 | 
| 84 | 26 django/template/base.py | 979 | 998| 143 | 24985 | 59519 | 
| 85 | 27 django/templatetags/i18n.py | 318 | 409| 643 | 25628 | 63494 | 
| 86 | 27 django/contrib/admin/templatetags/base.py | 22 | 34| 135 | 25763 | 63494 | 
| 87 | 27 django/utils/translation/template.py | 61 | 228| 1436 | 27199 | 63494 | 
| 88 | 28 django/templatetags/tz.py | 81 | 122| 246 | 27445 | 64679 | 
| 89 | 28 django/template/base.py | 486 | 505| 187 | 27632 | 64679 | 
| 90 | 28 django/templatetags/static.py | 57 | 90| 159 | 27791 | 64679 | 
| 91 | 28 django/template/base.py | 668 | 703| 272 | 28063 | 64679 | 
| 92 | 28 django/template/base.py | 381 | 404| 230 | 28293 | 64679 | 
| 93 | 28 django/template/utils.py | 64 | 90| 195 | 28488 | 64679 | 
| 94 | 29 django/templatetags/l10n.py | 41 | 64| 190 | 28678 | 65121 | 
| 95 | 30 django/contrib/admin/templatetags/admin_modify.py | 1 | 45| 372 | 29050 | 66089 | 
| 96 | 30 django/template/defaulttags.py | 1405 | 1440| 361 | 29411 | 66089 | 
| 97 | **30 django/template/library.py** | 201 | 234| 304 | 29715 | 66089 | 
| 98 | 30 django/template/loader_tags.py | 191 | 217| 285 | 30000 | 66089 | 
| 99 | 30 django/template/loader_tags.py | 109 | 124| 155 | 30155 | 66089 | 
| 100 | 31 django/views/debug.py | 194 | 242| 462 | 30617 | 70367 | 
| 101 | 31 django/templatetags/i18n.py | 518 | 533| 197 | 30814 | 70367 | 
| 102 | 31 django/template/defaultfilters.py | 412 | 421| 111 | 30925 | 70367 | 
| 103 | 32 django/template/backends/jinja2.py | 1 | 51| 341 | 31266 | 71189 | 
| 104 | 32 django/templatetags/tz.py | 125 | 145| 176 | 31442 | 71189 | 
| 105 | 32 django/template/base.py | 278 | 329| 359 | 31801 | 71189 | 
| 106 | 32 django/templatetags/tz.py | 148 | 170| 176 | 31977 | 71189 | 
| 107 | 33 django/views/csrf.py | 101 | 155| 577 | 32554 | 72733 | 
| 108 | 33 django/template/defaultfilters.py | 424 | 459| 233 | 32787 | 72733 | 
| 109 | 34 django/utils/html.py | 352 | 379| 212 | 32999 | 75835 | 
| 110 | 34 django/template/backends/django.py | 48 | 76| 210 | 33209 | 75835 | 
| 111 | 34 django/template/base.py | 332 | 378| 455 | 33664 | 75835 | 
| 112 | 35 django/core/management/templates.py | 210 | 241| 236 | 33900 | 78516 | 
| 113 | 35 django/views/generic/base.py | 153 | 185| 228 | 34128 | 78516 | 
| 114 | 36 django/contrib/syndication/views.py | 96 | 121| 180 | 34308 | 80242 | 
| 115 | 37 django/utils/jslex.py | 76 | 144| 763 | 35071 | 82029 | 
| 116 | 37 django/template/response.py | 60 | 145| 587 | 35658 | 82029 | 
| 117 | 37 django/views/defaults.py | 122 | 149| 214 | 35872 | 82029 | 
| 118 | 38 django/core/checks/security/sessions.py | 1 | 98| 572 | 36444 | 82602 | 
| 119 | 38 django/templatetags/cache.py | 52 | 94| 313 | 36757 | 82602 | 
| 120 | 39 django/core/checks/model_checks.py | 178 | 211| 332 | 37089 | 84389 | 
| 121 | 40 django/core/checks/translation.py | 1 | 62| 447 | 37536 | 84836 | 
| 122 | 40 django/template/response.py | 45 | 58| 120 | 37656 | 84836 | 
| 123 | 40 django/template/engine.py | 81 | 147| 457 | 38113 | 84836 | 
| 124 | 41 django/core/checks/security/csrf.py | 1 | 41| 299 | 38412 | 85135 | 
| 125 | 41 django/template/base.py | 816 | 881| 540 | 38952 | 85135 | 
| 126 | 41 django/template/defaultfilters.py | 308 | 322| 111 | 39063 | 85135 | 
| 127 | 41 django/template/defaultfilters.py | 462 | 489| 164 | 39227 | 85135 | 
| 128 | 41 django/views/defaults.py | 27 | 76| 401 | 39628 | 85135 | 
| 129 | 42 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 170 | 39798 | 88964 | 
| 130 | 42 django/template/base.py | 1001 | 1046| 351 | 40149 | 88964 | 
| 131 | 43 django/utils/safestring.py | 40 | 64| 159 | 40308 | 89350 | 
| 132 | 43 django/template/defaultfilters.py | 1 | 28| 207 | 40515 | 89350 | 
| 133 | 43 django/utils/translation/template.py | 35 | 59| 165 | 40680 | 89350 | 
| 134 | 43 django/views/csrf.py | 15 | 100| 835 | 41515 | 89350 | 
| 135 | 44 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 41920 | 89755 | 
| 136 | 44 django/template/defaultfilters.py | 692 | 769| 443 | 42363 | 89755 | 
| 137 | 44 django/core/management/templates.py | 58 | 117| 527 | 42890 | 89755 | 
| 138 | 44 django/template/backends/django.py | 1 | 45| 303 | 43193 | 89755 | 
| 139 | 44 django/template/defaulttags.py | 730 | 819| 759 | 43952 | 89755 | 
| 140 | 45 django/db/backends/ddl_references.py | 165 | 201| 253 | 44205 | 91102 | 
| 141 | 45 django/contrib/humanize/templatetags/humanize.py | 219 | 262| 731 | 44936 | 91102 | 
| 142 | 45 django/views/debug.py | 1 | 46| 319 | 45255 | 91102 | 
| 143 | 45 django/views/defaults.py | 79 | 97| 129 | 45384 | 91102 | 
| 144 | 45 django/template/loader_tags.py | 153 | 188| 292 | 45676 | 91102 | 
| 145 | 45 django/templatetags/l10n.py | 1 | 38| 251 | 45927 | 91102 | 
| 146 | 45 django/template/backends/jinja2.py | 54 | 88| 232 | 46159 | 91102 | 
| 147 | 45 django/template/base.py | 199 | 275| 503 | 46662 | 91102 | 
| 148 | 45 django/template/base.py | 407 | 424| 127 | 46789 | 91102 | 
| 149 | 45 django/contrib/admin/templatetags/admin_list.py | 197 | 211| 136 | 46925 | 91102 | 
| 150 | 46 django/template/loader.py | 1 | 49| 299 | 47224 | 91518 | 
| 151 | 46 django/templatetags/tz.py | 173 | 191| 149 | 47373 | 91518 | 
| 152 | 46 django/template/loader_tags.py | 80 | 107| 225 | 47598 | 91518 | 
| 153 | 46 django/templatetags/i18n.py | 412 | 517| 764 | 48362 | 91518 | 
| 154 | 46 django/core/management/templates.py | 119 | 182| 563 | 48925 | 91518 | 
| 155 | 47 django/core/cache/backends/base.py | 1 | 47| 245 | 49170 | 93675 | 
| 156 | 48 django/db/models/fields/mixins.py | 31 | 57| 173 | 49343 | 94018 | 
| 157 | 48 django/template/defaultfilters.py | 167 | 188| 206 | 49549 | 94018 | 
| 158 | 48 django/contrib/humanize/templatetags/humanize.py | 264 | 302| 370 | 49919 | 94018 | 
| 159 | 49 django/contrib/admin/templatetags/log.py | 26 | 60| 317 | 50236 | 94497 | 
| 160 | 50 django/utils/feedgenerator.py | 294 | 393| 926 | 51162 | 97830 | 
| 161 | 51 django/views/generic/edit.py | 103 | 126| 194 | 51356 | 99546 | 
| 162 | 52 docs/conf.py | 97 | 199| 893 | 52249 | 102523 | 
| 163 | 52 django/templatetags/static.py | 1 | 54| 351 | 52600 | 102523 | 
| 164 | 53 docs/_ext/djangodocs.py | 108 | 169| 526 | 53126 | 105568 | 
| 165 | 53 django/contrib/admin/templatetags/admin_list.py | 431 | 489| 343 | 53469 | 105568 | 
| 166 | 54 django/core/checks/urls.py | 71 | 111| 264 | 53733 | 106269 | 
| 167 | 54 django/core/management/templates.py | 40 | 56| 181 | 53914 | 106269 | 
| 168 | 55 django/forms/widgets.py | 1 | 42| 299 | 54213 | 114275 | 
| 169 | 55 django/templatetags/i18n.py | 1 | 28| 188 | 54401 | 114275 | 
| 170 | 55 django/template/base.py | 610 | 666| 440 | 54841 | 114275 | 
| 171 | 56 django/template/context.py | 233 | 260| 199 | 55040 | 116156 | 
| 172 | 56 django/utils/feedgenerator.py | 46 | 56| 114 | 55154 | 116156 | 
| 173 | 57 django/views/decorators/debug.py | 1 | 44| 274 | 55428 | 116745 | 
| 174 | 57 django/templatetags/i18n.py | 68 | 95| 229 | 55657 | 116745 | 
| 175 | 58 django/contrib/admindocs/utils.py | 115 | 139| 185 | 55842 | 118651 | 
| 176 | 59 django/utils/http.py | 1 | 73| 714 | 56556 | 122829 | 
| 177 | 59 django/templatetags/i18n.py | 536 | 554| 121 | 56677 | 122829 | 
| 178 | 59 django/core/checks/security/base.py | 213 | 226| 131 | 56808 | 122829 | 
| 179 | 59 django/templatetags/tz.py | 1 | 34| 148 | 56956 | 122829 | 
| 180 | 60 django/utils/deprecation.py | 1 | 27| 181 | 57137 | 123507 | 
| 181 | 61 django/template/backends/base.py | 1 | 82| 518 | 57655 | 124026 | 
| 182 | 62 django/views/i18n.py | 83 | 186| 711 | 58366 | 126553 | 
| 183 | 62 django/template/defaultfilters.py | 852 | 907| 539 | 58905 | 126553 | 
| 184 | 62 django/template/loader_tags.py | 220 | 247| 239 | 59144 | 126553 | 
| 185 | 62 django/template/defaultfilters.py | 772 | 802| 273 | 59417 | 126553 | 
| 186 | 62 django/utils/html.py | 179 | 197| 161 | 59578 | 126553 | 
| 187 | 62 django/utils/html.py | 1 | 75| 614 | 60192 | 126553 | 
| 188 | 63 django/template/loaders/cached.py | 62 | 93| 225 | 60417 | 127241 | 
| 189 | 63 django/contrib/admin/templatetags/admin_modify.py | 89 | 117| 203 | 60620 | 127241 | 
| 190 | 64 django/db/migrations/questioner.py | 109 | 141| 290 | 60910 | 129315 | 
| 191 | 64 django/template/context.py | 133 | 167| 288 | 61198 | 129315 | 
| 192 | 65 django/core/mail/message.py | 55 | 71| 171 | 61369 | 132853 | 
| 193 | 65 django/views/debug.py | 99 | 123| 155 | 61524 | 132853 | 
| 194 | 66 django/utils/inspect.py | 1 | 33| 215 | 61739 | 133249 | 
| 195 | 66 django/forms/widgets.py | 464 | 504| 275 | 62014 | 133249 | 
| 196 | 66 django/utils/html.py | 150 | 176| 149 | 62163 | 133249 | 
| 197 | 66 django/core/checks/security/base.py | 85 | 180| 710 | 62873 | 133249 | 
| 198 | 66 docs/_ext/djangodocs.py | 73 | 105| 255 | 63128 | 133249 | 
| 199 | 66 django/template/base.py | 792 | 814| 190 | 63318 | 133249 | 
| 200 | 67 django/db/migrations/exceptions.py | 1 | 55| 250 | 63568 | 133500 | 
| 201 | 67 django/views/debug.py | 126 | 153| 248 | 63816 | 133500 | 
| 202 | 67 django/template/defaultfilters.py | 94 | 164| 633 | 64449 | 133500 | 
| 203 | 67 django/utils/html.py | 118 | 135| 154 | 64603 | 133500 | 
| 204 | 67 django/utils/html.py | 92 | 115| 200 | 64803 | 133500 | 
| 205 | 67 django/template/engine.py | 55 | 79| 204 | 65007 | 133500 | 
| 206 | 68 django/views/generic/__init__.py | 1 | 23| 189 | 65196 | 133690 | 
| 207 | 68 django/views/csrf.py | 1 | 13| 132 | 65328 | 133690 | 


## Patch

```diff
diff --git a/django/template/library.py b/django/template/library.py
--- a/django/template/library.py
+++ b/django/template/library.py
@@ -261,7 +261,7 @@ def parse_bits(parser, bits, params, varargs, varkw, defaults,
         if kwarg:
             # The kwarg was successfully extracted
             param, value = kwarg.popitem()
-            if param not in params and param not in unhandled_kwargs and varkw is None:
+            if param not in params and param not in kwonly and varkw is None:
                 # An unexpected keyword argument was supplied
                 raise TemplateSyntaxError(
                     "'%s' received unexpected keyword argument '%s'" %

```

## Test Patch

```diff
diff --git a/tests/template_tests/templatetags/inclusion.py b/tests/template_tests/templatetags/inclusion.py
--- a/tests/template_tests/templatetags/inclusion.py
+++ b/tests/template_tests/templatetags/inclusion.py
@@ -136,6 +136,15 @@ def inclusion_one_default(one, two='hi'):
 inclusion_one_default.anything = "Expected inclusion_one_default __dict__"
 
 
+@register.inclusion_tag('inclusion.html')
+def inclusion_keyword_only_default(*, kwarg=42):
+    return {
+        'result': (
+            'inclusion_keyword_only_default - Expected result: %s' % kwarg
+        ),
+    }
+
+
 @register.inclusion_tag(engine.get_template('inclusion.html'))
 def inclusion_one_default_from_template(one, two='hi'):
     """Expected inclusion_one_default_from_template __doc__"""
diff --git a/tests/template_tests/test_custom.py b/tests/template_tests/test_custom.py
--- a/tests/template_tests/test_custom.py
+++ b/tests/template_tests/test_custom.py
@@ -62,6 +62,10 @@ def test_simple_tags(self):
                 'simple_keyword_only_param - Expected result: 37'),
             ('{% load custom %}{% simple_keyword_only_default %}',
                 'simple_keyword_only_default - Expected result: 42'),
+            (
+                '{% load custom %}{% simple_keyword_only_default kwarg=37 %}',
+                'simple_keyword_only_default - Expected result: 37',
+            ),
             ('{% load custom %}{% simple_one_default 37 %}', 'simple_one_default - Expected result: 37, hi'),
             ('{% load custom %}{% simple_one_default 37 two="hello" %}',
                 'simple_one_default - Expected result: 37, hello'),
@@ -97,6 +101,18 @@ def test_simple_tag_errors(self):
                 '{% load custom %}{% simple_one_default 37 42 56 %}'),
             ("'simple_keyword_only_param' did not receive value(s) for the argument(s): 'kwarg'",
                 '{% load custom %}{% simple_keyword_only_param %}'),
+            (
+                "'simple_keyword_only_param' received multiple values for "
+                "keyword argument 'kwarg'",
+                '{% load custom %}{% simple_keyword_only_param kwarg=42 '
+                'kwarg=37 %}',
+            ),
+            (
+                "'simple_keyword_only_default' received multiple values for "
+                "keyword argument 'kwarg'",
+                '{% load custom %}{% simple_keyword_only_default kwarg=42 '
+                'kwarg=37 %}',
+            ),
             ("'simple_unlimited_args_kwargs' received some positional argument(s) after some keyword argument(s)",
                 '{% load custom %}{% simple_unlimited_args_kwargs 37 40|add:2 eggs="scrambled" 56 four=1|add:3 %}'),
             ("'simple_unlimited_args_kwargs' received multiple values for keyword argument 'eggs'",
@@ -180,6 +196,10 @@ def test_inclusion_tags(self):
                 'inclusion_one_default - Expected result: 99, hello\n'),
             ('{% load inclusion %}{% inclusion_one_default 37 42 %}',
                 'inclusion_one_default - Expected result: 37, 42\n'),
+            (
+                '{% load inclusion %}{% inclusion_keyword_only_default kwarg=37 %}',
+                'inclusion_keyword_only_default - Expected result: 37\n',
+            ),
             ('{% load inclusion %}{% inclusion_unlimited_args 37 %}',
                 'inclusion_unlimited_args - Expected result: 37, hi\n'),
             ('{% load inclusion %}{% inclusion_unlimited_args 37 42 56 89 %}',
@@ -206,6 +226,12 @@ def test_inclusion_tag_errors(self):
                 '{% load inclusion %}{% inclusion_one_default 37 42 56 %}'),
             ("'inclusion_one_default' did not receive value(s) for the argument(s): 'one'",
                 '{% load inclusion %}{% inclusion_one_default %}'),
+            (
+                "'inclusion_keyword_only_default' received multiple values "
+                "for keyword argument 'kwarg'",
+                '{% load inclusion %}{% inclusion_keyword_only_default '
+                'kwarg=37 kwarg=42 %}',
+            ),
             ("'inclusion_unlimited_args' did not receive value(s) for the argument(s): 'one'",
                 '{% load inclusion %}{% inclusion_unlimited_args %}'),
             (

```


## Code snippets

### 1 - django/template/defaulttags.py:

Start line: 1, End line: 48

```python
"""Default tags used by the template system, available to all templates."""
import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle, groupby

from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe

from .base import (
    BLOCK_TAG_END, BLOCK_TAG_START, COMMENT_TAG_END, COMMENT_TAG_START,
    FILTER_SEPARATOR, SINGLE_BRACE_END, SINGLE_BRACE_START,
    VARIABLE_ATTRIBUTE_SEPARATOR, VARIABLE_TAG_END, VARIABLE_TAG_START, Node,
    NodeList, TemplateSyntaxError, VariableDoesNotExist, kwarg_re,
    render_value_in_context, token_kwargs,
)
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal

register = Library()


class AutoEscapeControlNode(Node):
    """Implement the actions of the autoescape tag."""
    def __init__(self, setting, nodelist):
        self.setting, self.nodelist = setting, nodelist

    def render(self, context):
        old_setting = context.autoescape
        context.autoescape = self.setting
        output = self.nodelist.render(context)
        context.autoescape = old_setting
        if self.setting:
            return mark_safe(output)
        else:
            return output


class CommentNode(Node):
    def render(self, context):
        return ''
```
### 2 - django/template/defaulttags.py:

Start line: 517, End line: 540

```python
@register.tag
def autoescape(parser, token):
    """
    Force autoescape behavior for this block.
    """
    # token.split_contents() isn't useful here because this tag doesn't accept variable as arguments
    args = token.contents.split()
    if len(args) != 2:
        raise TemplateSyntaxError("'autoescape' tag requires exactly one argument.")
    arg = args[1]
    if arg not in ('on', 'off'):
        raise TemplateSyntaxError("'autoescape' argument should be 'on' or 'off'")
    nodelist = parser.parse(('endautoescape',))
    parser.delete_first_token()
    return AutoEscapeControlNode((arg == 'on'), nodelist)


@register.tag
def comment(parser, token):
    """
    Ignore everything between ``{% comment %}`` and ``{% endcomment %}``.
    """
    parser.skip_past('endcomment')
    return CommentNode()
```
### 3 - django/template/library.py:

Start line: 237, End line: 309

```python
def parse_bits(parser, bits, params, varargs, varkw, defaults,
               kwonly, kwonly_defaults, takes_context, name):
    """
    Parse bits for template tag helpers simple_tag and inclusion_tag, in
    particular by detecting syntax errors and by extracting positional and
    keyword arguments.
    """
    if takes_context:
        if params[0] == 'context':
            params = params[1:]
        else:
            raise TemplateSyntaxError(
                "'%s' is decorated with takes_context=True so it must "
                "have a first argument of 'context'" % name)
    args = []
    kwargs = {}
    unhandled_params = list(params)
    unhandled_kwargs = [
        kwarg for kwarg in kwonly
        if not kwonly_defaults or kwarg not in kwonly_defaults
    ]
    for bit in bits:
        # First we try to extract a potential kwarg from the bit
        kwarg = token_kwargs([bit], parser)
        if kwarg:
            # The kwarg was successfully extracted
            param, value = kwarg.popitem()
            if param not in params and param not in unhandled_kwargs and varkw is None:
                # An unexpected keyword argument was supplied
                raise TemplateSyntaxError(
                    "'%s' received unexpected keyword argument '%s'" %
                    (name, param))
            elif param in kwargs:
                # The keyword argument has already been supplied once
                raise TemplateSyntaxError(
                    "'%s' received multiple values for keyword argument '%s'" %
                    (name, param))
            else:
                # All good, record the keyword argument
                kwargs[str(param)] = value
                if param in unhandled_params:
                    # If using the keyword syntax for a positional arg, then
                    # consume it.
                    unhandled_params.remove(param)
                elif param in unhandled_kwargs:
                    # Same for keyword-only arguments
                    unhandled_kwargs.remove(param)
        else:
            if kwargs:
                raise TemplateSyntaxError(
                    "'%s' received some positional argument(s) after some "
                    "keyword argument(s)" % name)
            else:
                # Record the positional argument
                args.append(parser.compile_filter(bit))
                try:
                    # Consume from the list of expected positional arguments
                    unhandled_params.pop(0)
                except IndexError:
                    if varargs is None:
                        raise TemplateSyntaxError(
                            "'%s' received too many positional arguments" %
                            name)
    if defaults is not None:
        # Consider the last n params handled, where n is the
        # number of defaults.
        unhandled_params = unhandled_params[:-len(defaults)]
    if unhandled_params or unhandled_kwargs:
        # Some positional arguments were not supplied
        raise TemplateSyntaxError(
            "'%s' did not receive value(s) for the argument(s): %s" %
            (name, ", ".join("'%s'" % p for p in unhandled_params + unhandled_kwargs)))
    return args, kwargs
```
### 4 - django/template/defaulttags.py:

Start line: 1280, End line: 1312

```python
@register.tag
def templatetag(parser, token):
    """
    Output one of the bits used to compose template tags.

    Since the template system has no concept of "escaping", to display one of
    the bits used in template tags, you must use the ``{% templatetag %}`` tag.

    The argument tells which template bit to output:

        ==================  =======
        Argument            Outputs
        ==================  =======
        ``openblock``       ``{%``
        ``closeblock``      ``%}``
        ``openvariable``    ``{{``
        ``closevariable``   ``}}``
        ``openbrace``       ``{``
        ``closebrace``      ``}``
        ``opencomment``     ``{#``
        ``closecomment``    ``#}``
        ==================  =======
    """
    # token.split_contents() isn't useful here because this tag doesn't accept variable as arguments
    bits = token.contents.split()
    if len(bits) != 2:
        raise TemplateSyntaxError("'templatetag' statement takes one argument")
    tag = bits[1]
    if tag not in TemplateTagNode.mapping:
        raise TemplateSyntaxError("Invalid templatetag argument: '%s'."
                                  " Must be one of: %s" %
                                  (tag, list(TemplateTagNode.mapping)))
    return TemplateTagNode(tag)
```
### 5 - django/template/defaulttags.py:

Start line: 51, End line: 68

```python
class CsrfTokenNode(Node):
    def render(self, context):
        csrf_token = context.get('csrf_token')
        if csrf_token:
            if csrf_token == 'NOTPROVIDED':
                return format_html("")
            else:
                return format_html('<input type="hidden" name="csrfmiddlewaretoken" value="{}">', csrf_token)
        else:
            # It's very probable that the token is missing because of
            # misconfiguration, so we raise a warning
            if settings.DEBUG:
                warnings.warn(
                    "A {% csrf_token %} was used in a template, but the context "
                    "did not provide the value.  This is usually caused by not "
                    "using RequestContext."
                )
            return ''
```
### 6 - django/template/defaulttags.py:

Start line: 839, End line: 888

```python
@register.tag
def ifequal(parser, token):
    """
    Output the contents of the block if the two arguments equal each other.

    Examples::

        {% ifequal user.id comment.user_id %}
            ...
        {% endifequal %}

        {% ifnotequal user.id comment.user_id %}
            ...
        {% else %}
            ...
        {% endifnotequal %}
    """
    return do_ifequal(parser, token, False)


@register.tag
def ifnotequal(parser, token):
    """
    Output the contents of the block if the two arguments are not equal.
    See ifequal.
    """
    return do_ifequal(parser, token, True)


class TemplateLiteral(Literal):
    def __init__(self, value, text):
        self.value = value
        self.text = text  # for better error messages

    def display(self):
        return self.text

    def eval(self, context):
        return self.value.resolve(context, ignore_failures=True)


class TemplateIfParser(IfParser):
    error_class = TemplateSyntaxError

    def __init__(self, parser, *args, **kwargs):
        self.template_parser = parser
        super().__init__(*args, **kwargs)

    def create_var(self, value):
        return TemplateLiteral(self.template_parser.compile_filter(value), value)
```
### 7 - django/template/defaulttags.py:

Start line: 499, End line: 514

```python
class WithNode(Node):
    def __init__(self, var, name, nodelist, extra_context=None):
        self.nodelist = nodelist
        # var and name are legacy attributes, being left in case they are used
        # by third-party subclasses of this Node.
        self.extra_context = extra_context or {}
        if name:
            self.extra_context[name] = var

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def render(self, context):
        values = {key: val.resolve(context) for key, val in self.extra_context.items()}
        with context.push(**values):
            return self.nodelist.render(context)
```
### 8 - django/template/defaulttags.py:

Start line: 822, End line: 836

```python
def do_ifequal(parser, token, negate):
    bits = list(token.split_contents())
    if len(bits) != 3:
        raise TemplateSyntaxError("%r takes two arguments" % bits[0])
    end_tag = 'end' + bits[0]
    nodelist_true = parser.parse(('else', end_tag))
    token = parser.next_token()
    if token.contents == 'else':
        nodelist_false = parser.parse((end_tag,))
        parser.delete_first_token()
    else:
        nodelist_false = NodeList()
    val1 = parser.compile_filter(bits[1])
    val2 = parser.compile_filter(bits[2])
    return IfEqualNode(val1, val2, nodelist_true, nodelist_false, negate)
```
### 9 - django/template/defaulttags.py:

Start line: 1223, End line: 1246

```python
@register.tag
def resetcycle(parser, token):
    """
    Reset a cycle tag.

    If an argument is given, reset the last rendered cycle tag whose name
    matches the argument, else reset the last rendered cycle tag (named or
    unnamed).
    """
    args = token.split_contents()

    if len(args) > 2:
        raise TemplateSyntaxError("%r tag accepts at most one argument." % args[0])

    if len(args) == 2:
        name = args[1]
        try:
            return ResetCycleNode(parser._named_cycle_nodes[name])
        except (AttributeError, KeyError):
            raise TemplateSyntaxError("Named cycle '%s' does not exist." % name)
    try:
        return ResetCycleNode(parser._last_cycle_node)
    except AttributeError:
        raise TemplateSyntaxError("No cycles in template.")
```
### 10 - django/template/defaulttags.py:

Start line: 156, End line: 217

```python
class ForNode(Node):

    def render(self, context):
        if 'forloop' in context:
            parentloop = context['forloop']
        else:
            parentloop = {}
        with context.push():
            values = self.sequence.resolve(context, ignore_failures=True)
            if values is None:
                values = []
            if not hasattr(values, '__len__'):
                values = list(values)
            len_values = len(values)
            if len_values < 1:
                return self.nodelist_empty.render(context)
            nodelist = []
            if self.is_reversed:
                values = reversed(values)
            num_loopvars = len(self.loopvars)
            unpack = num_loopvars > 1
            # Create a forloop value in the context.  We'll update counters on each
            # iteration just below.
            loop_dict = context['forloop'] = {'parentloop': parentloop}
            for i, item in enumerate(values):
                # Shortcuts for current loop iteration number.
                loop_dict['counter0'] = i
                loop_dict['counter'] = i + 1
                # Reverse counter iteration numbers.
                loop_dict['revcounter'] = len_values - i
                loop_dict['revcounter0'] = len_values - i - 1
                # Boolean values designating first and last times through loop.
                loop_dict['first'] = (i == 0)
                loop_dict['last'] = (i == len_values - 1)

                pop_context = False
                if unpack:
                    # If there are multiple loop variables, unpack the item into
                    # them.
                    try:
                        len_item = len(item)
                    except TypeError:  # not an iterable
                        len_item = 1
                    # Check loop variable count before unpacking
                    if num_loopvars != len_item:
                        raise ValueError(
                            "Need {} values to unpack in for loop; got {}. "
                            .format(num_loopvars, len_item),
                        )
                    unpacked_vars = dict(zip(self.loopvars, item))
                    pop_context = True
                    context.update(unpacked_vars)
                else:
                    context[self.loopvars[0]] = item

                for node in self.nodelist_loop:
                    nodelist.append(node.render_annotated(context))

                if pop_context:
                    # Pop the loop variables pushed on to the context to avoid
                    # the context ending up in an inconsistent state when other
                    # tags (e.g., include and with) push data to context.
                    context.pop()
        return mark_safe(''.join(nodelist))
```
### 11 - django/template/library.py:

Start line: 1, End line: 52

```python
import functools
from importlib import import_module
from inspect import getfullargspec, unwrap

from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable

from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError


class InvalidTemplateLibrary(Exception):
    pass


class Library:
    """
    A class for registering template tags and filters. Compiled filter and
    template tag functions are stored in the filters and tags attributes.
    The filter, simple_tag, and inclusion_tag methods provide a convenient
    way to register callables as tags.
    """
    def __init__(self):
        self.filters = {}
        self.tags = {}

    def tag(self, name=None, compile_function=None):
        if name is None and compile_function is None:
            # @register.tag()
            return self.tag_function
        elif name is not None and compile_function is None:
            if callable(name):
                # @register.tag
                return self.tag_function(name)
            else:
                # @register.tag('somename') or @register.tag(name='somename')
                def dec(func):
                    return self.tag(name, func)
                return dec
        elif name is not None and compile_function is not None:
            # register.tag('somename', somefunc)
            self.tags[name] = compile_function
            return compile_function
        else:
            raise ValueError(
                "Unsupported arguments to Library.tag: (%r, %r)" %
                (name, compile_function),
            )

    def tag_function(self, func):
        self.tags[getattr(func, "_decorated_function", func).__name__] = func
        return func
```
### 16 - django/template/library.py:

Start line: 164, End line: 181

```python
class TagHelperNode(Node):
    """
    Base class for tag helper nodes such as SimpleNode and InclusionNode.
    Manages the positional and keyword arguments to be passed to the decorated
    function.
    """
    def __init__(self, func, takes_context, args, kwargs):
        self.func = func
        self.takes_context = takes_context
        self.args = args
        self.kwargs = kwargs

    def get_resolved_arguments(self, context):
        resolved_args = [var.resolve(context) for var in self.args]
        if self.takes_context:
            resolved_args = [context] + resolved_args
        resolved_kwargs = {k: v.resolve(context) for k, v in self.kwargs.items()}
        return resolved_args, resolved_kwargs
```
### 18 - django/template/library.py:

Start line: 184, End line: 198

```python
class SimpleNode(TagHelperNode):

    def __init__(self, func, takes_context, args, kwargs, target_var):
        super().__init__(func, takes_context, args, kwargs)
        self.target_var = target_var

    def render(self, context):
        resolved_args, resolved_kwargs = self.get_resolved_arguments(context)
        output = self.func(*resolved_args, **resolved_kwargs)
        if self.target_var is not None:
            context[self.target_var] = output
            return ''
        if context.autoescape:
            output = conditional_escape(output)
        return output
```
### 21 - django/template/library.py:

Start line: 96, End line: 134

```python
class Library:

    def filter_function(self, func, **flags):
        name = getattr(func, "_decorated_function", func).__name__
        return self.filter(name, func, **flags)

    def simple_tag(self, func=None, takes_context=None, name=None):
        """
        Register a callable as a compiled template tag. Example:

        @register.simple_tag
        def hello(*args, **kwargs):
            return 'world'
        """
        def dec(func):
            params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
            function_name = (name or getattr(func, '_decorated_function', func).__name__)

            @functools.wraps(func)
            def compile_func(parser, token):
                bits = token.split_contents()[1:]
                target_var = None
                if len(bits) >= 2 and bits[-2] == 'as':
                    target_var = bits[-1]
                    bits = bits[:-2]
                args, kwargs = parse_bits(
                    parser, bits, params, varargs, varkw, defaults,
                    kwonly, kwonly_defaults, takes_context, function_name,
                )
                return SimpleNode(func, takes_context, args, kwargs, target_var)
            self.tag(function_name, compile_func)
            return func

        if func is None:
            # @register.simple_tag(...)
            return dec
        elif callable(func):
            # @register.simple_tag
            return dec(func)
        else:
            raise ValueError("Invalid arguments provided to simple_tag")
```
### 79 - django/template/library.py:

Start line: 136, End line: 161

```python
class Library:

    def inclusion_tag(self, filename, func=None, takes_context=None, name=None):
        """
        Register a callable as an inclusion tag:

        @register.inclusion_tag('results.html')
        def show_results(poll):
            choices = poll.choice_set.all()
            return {'choices': choices}
        """
        def dec(func):
            params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
            function_name = (name or getattr(func, '_decorated_function', func).__name__)

            @functools.wraps(func)
            def compile_func(parser, token):
                bits = token.split_contents()[1:]
                args, kwargs = parse_bits(
                    parser, bits, params, varargs, varkw, defaults,
                    kwonly, kwonly_defaults, takes_context, function_name,
                )
                return InclusionNode(
                    func, takes_context, args, kwargs, filename,
                )
            self.tag(function_name, compile_func)
            return func
        return dec
```
### 97 - django/template/library.py:

Start line: 201, End line: 234

```python
class InclusionNode(TagHelperNode):

    def __init__(self, func, takes_context, args, kwargs, filename):
        super().__init__(func, takes_context, args, kwargs)
        self.filename = filename

    def render(self, context):
        """
        Render the specified template and context. Cache the template object
        in render_context to avoid reparsing and loading when used in a for
        loop.
        """
        resolved_args, resolved_kwargs = self.get_resolved_arguments(context)
        _dict = self.func(*resolved_args, **resolved_kwargs)

        t = context.render_context.get(self)
        if t is None:
            if isinstance(self.filename, Template):
                t = self.filename
            elif isinstance(getattr(self.filename, 'template', None), Template):
                t = self.filename.template
            elif not isinstance(self.filename, str) and is_iterable(self.filename):
                t = context.template.engine.select_template(self.filename)
            else:
                t = context.template.engine.get_template(self.filename)
            context.render_context[self] = t
        new_context = context.new(_dict)
        # Copy across the CSRF token, if present, because inclusion tags are
        # often used for forms, and we need instructions for using CSRF
        # protection to be as simple as possible.
        csrf_token = context.get('csrf_token')
        if csrf_token is not None:
            new_context['csrf_token'] = csrf_token
        return t.render(new_context)
```
