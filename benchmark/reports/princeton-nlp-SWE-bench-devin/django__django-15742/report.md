# django__django-15742

| **django/django** | `7faf25d682b8e8f4fd2006eb7dfc71ed2a2193b7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 18166 |
| **Any found context length** | 1236 |
| **Avg pos** | 56.0 |
| **Min pos** | 2 |
| **Max pos** | 54 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/templatetags/i18n.py b/django/templatetags/i18n.py
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -5,7 +5,7 @@
 from django.template.base import TokenType, render_value_in_context
 from django.template.defaulttags import token_kwargs
 from django.utils import translation
-from django.utils.safestring import SafeData, mark_safe
+from django.utils.safestring import SafeData, SafeString, mark_safe
 
 register = Library()
 
@@ -198,7 +198,7 @@ def render_value(key):
             with translation.override(None):
                 result = self.render(context, nested=True)
         if self.asvar:
-            context[self.asvar] = result
+            context[self.asvar] = SafeString(result)
             return ""
         else:
             return result

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/templatetags/i18n.py | 8 | 8 | 54 | 1 | 18166
| django/templatetags/i18n.py | 201 | 201 | 2 | 1 | 1236


## Problem Statement

```
Blocktranslate asvar escapes variables, but stores the result as str instance, leading to double escaping
Description
	
In the docs, this snippet is given as an example usage of blocktranslate with the asvar argument (here: ​https://docs.djangoproject.com/en/4.0/topics/i18n/translation/#blocktranslate-template-tag:
{% blocktranslate asvar the_title %}The title is {{ title }}.{% endblocktranslate %}
<title>{{ the_title }}</title>
<meta name="description" content="{{ the_title }}">
However, this template is buggy when title is a string, which I'd argue is a common use case.
title will be escaped when formatting the content of the blocktranslate block, but the "was escaped" information is discarded, and the_title will be a str instance with escaped content.
When later using the the_title variable, it will be conditionally escaped. Since it is a str, it will be escaped, so control characters are escaped again, breaking their display on the final page.
Minimal example to reproduce (can be put in any view):
	from django.template import Template, Context
	template_content = """
{% blocktranslate asvar the_title %}The title is {{ title }}.{% endblocktranslate %}
<title>{{ the_title }}</title>
<meta name="description" content="{{ the_title }}">
"""
	rendered = Template(template_content).render(Context({"title": "<>& Title"}))
	assert "&amp;lt;" not in rendered, "> was escaped two times"
I'd argue that blocktranslate should:
Either assign a SafeString instance to prevent future escaping
or not escape the variables used within the translation, and store them marked as unsafe (= as str instance)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/templatetags/i18n.py** | 456 | 567| 774 | 774 | 4182 | 
| **-> 2 <-** | **1 django/templatetags/i18n.py** | 148 | 204| 462 | 1236 | 4182 | 
| 3 | **1 django/templatetags/i18n.py** | 568 | 596| 216 | 1452 | 4182 | 
| 4 | **1 django/templatetags/i18n.py** | 104 | 146| 283 | 1735 | 4182 | 
| 5 | 2 django/utils/translation/template.py | 65 | 247| 1459 | 3194 | 6170 | 
| 6 | **2 django/templatetags/i18n.py** | 358 | 453| 649 | 3843 | 6170 | 
| 7 | 2 django/utils/translation/template.py | 1 | 36| 384 | 4227 | 6170 | 
| 8 | 3 django/template/base.py | 776 | 841| 558 | 4785 | 14443 | 
| 9 | 3 django/template/base.py | 1037 | 1050| 127 | 4912 | 14443 | 
| 10 | **3 django/templatetags/i18n.py** | 72 | 101| 245 | 5157 | 14443 | 
| 11 | 4 django/utils/safestring.py | 1 | 22| 127 | 5284 | 14856 | 
| 12 | 5 django/template/defaulttags.py | 1395 | 1415| 121 | 5405 | 25629 | 
| 13 | 6 django/views/debug.py | 225 | 277| 471 | 5876 | 30375 | 
| 14 | 7 django/utils/html.py | 82 | 105| 200 | 6076 | 33636 | 
| 15 | 8 django/template/defaultfilters.py | 348 | 362| 111 | 6187 | 40116 | 
| 16 | 8 django/template/defaulttags.py | 545 | 569| 191 | 6378 | 40116 | 
| 17 | 9 django/utils/text.py | 378 | 396| 171 | 6549 | 43412 | 
| 18 | 9 django/template/defaultfilters.py | 280 | 345| 425 | 6974 | 43412 | 
| 19 | 9 django/template/base.py | 1 | 91| 754 | 7728 | 43412 | 
| 20 | 9 django/utils/safestring.py | 47 | 73| 166 | 7894 | 43412 | 
| 21 | 9 django/utils/safestring.py | 25 | 44| 118 | 8012 | 43412 | 
| 22 | 9 django/utils/html.py | 1 | 59| 440 | 8452 | 43412 | 
| 23 | 9 django/template/defaultfilters.py | 365 | 452| 504 | 8956 | 43412 | 
| 24 | 10 django/template/backends/dummy.py | 1 | 53| 327 | 9283 | 43739 | 
| 25 | 10 django/template/base.py | 843 | 865| 190 | 9473 | 43739 | 
| 26 | 11 docs/_ext/djangodocs.py | 111 | 175| 567 | 10040 | 46963 | 
| 27 | 12 django/views/i18n.py | 88 | 191| 702 | 10742 | 49466 | 
| 28 | 12 django/utils/html.py | 387 | 402| 103 | 10845 | 49466 | 
| 29 | 13 django/templatetags/l10n.py | 41 | 64| 190 | 11035 | 49908 | 
| 30 | 13 django/template/defaultfilters.py | 467 | 502| 234 | 11269 | 49908 | 
| 31 | 14 django/utils/translation/__init__.py | 44 | 60| 154 | 11423 | 51786 | 
| 32 | 15 django/template/smartif.py | 117 | 151| 188 | 11611 | 53314 | 
| 33 | 15 django/utils/text.py | 399 | 417| 172 | 11783 | 53314 | 
| 34 | 15 django/template/base.py | 609 | 644| 361 | 12144 | 53314 | 
| 35 | 15 django/templatetags/l10n.py | 1 | 38| 251 | 12395 | 53314 | 
| 36 | 15 django/template/defaulttags.py | 719 | 765| 312 | 12707 | 53314 | 
| 37 | 16 django/template/loader_tags.py | 1 | 39| 207 | 12914 | 56026 | 
| 38 | 16 django/template/base.py | 867 | 942| 575 | 13489 | 56026 | 
| 39 | 16 django/template/defaulttags.py | 1 | 64| 338 | 13827 | 56026 | 
| 40 | 17 django/utils/encoding.py | 56 | 75| 156 | 13983 | 58251 | 
| 41 | 17 django/utils/translation/template.py | 39 | 63| 165 | 14148 | 58251 | 
| 42 | 17 django/utils/translation/__init__.py | 1 | 43| 255 | 14403 | 58251 | 
| 43 | 17 django/template/defaultfilters.py | 207 | 228| 206 | 14609 | 58251 | 
| 44 | 17 django/template/loader_tags.py | 211 | 240| 292 | 14901 | 58251 | 
| 45 | 18 django/utils/translation/trans_real.py | 389 | 439| 358 | 15259 | 62708 | 
| 46 | 18 django/utils/translation/__init__.py | 62 | 81| 138 | 15397 | 62708 | 
| 47 | 18 docs/_ext/djangodocs.py | 288 | 380| 740 | 16137 | 62708 | 
| 48 | 18 django/template/defaultfilters.py | 261 | 277| 116 | 16253 | 62708 | 
| 49 | 18 django/utils/translation/trans_real.py | 1 | 63| 513 | 16766 | 62708 | 
| 50 | 18 django/template/defaulttags.py | 1289 | 1323| 293 | 17059 | 62708 | 
| 51 | 18 django/template/defaultfilters.py | 1 | 90| 559 | 17618 | 62708 | 
| 52 | 18 django/template/base.py | 1053 | 1074| 150 | 17768 | 62708 | 
| 53 | 19 django/urls/base.py | 160 | 188| 201 | 17969 | 63907 | 
| **-> 54 <-** | **19 django/templatetags/i18n.py** | 1 | 32| 197 | 18166 | 63907 | 
| 55 | 20 django/utils/http.py | 376 | 397| 178 | 18344 | 67669 | 
| 56 | 21 django/utils/translation/trans_null.py | 1 | 68| 275 | 18619 | 67944 | 
| 57 | 22 django/views/static.py | 56 | 79| 211 | 18830 | 68930 | 
| 58 | 22 django/template/base.py | 712 | 747| 270 | 19100 | 68930 | 
| 59 | 22 django/utils/html.py | 62 | 79| 165 | 19265 | 68930 | 
| 60 | 22 django/template/defaulttags.py | 665 | 716| 339 | 19604 | 68930 | 
| 61 | 22 django/template/loader_tags.py | 42 | 80| 321 | 19925 | 68930 | 
| 62 | **22 django/templatetags/i18n.py** | 35 | 69| 222 | 20147 | 68930 | 
| 63 | 23 django/db/models/functions/text.py | 42 | 64| 156 | 20303 | 71299 | 
| 64 | 23 django/utils/encoding.py | 195 | 209| 207 | 20510 | 71299 | 
| 65 | 23 django/utils/text.py | 420 | 436| 104 | 20614 | 71299 | 
| 66 | 23 django/db/models/functions/text.py | 238 | 255| 150 | 20764 | 71299 | 
| 67 | 23 django/utils/html.py | 277 | 327| 395 | 21159 | 71299 | 
| 68 | 23 django/template/defaulttags.py | 892 | 980| 694 | 21853 | 71299 | 
| 69 | 23 django/template/defaultfilters.py | 455 | 464| 111 | 21964 | 71299 | 
| 70 | 23 django/utils/html.py | 405 | 424| 168 | 22132 | 71299 | 
| 71 | 24 django/core/management/commands/makemessages.py | 166 | 178| 111 | 22243 | 77320 | 
| 72 | 24 django/template/defaultfilters.py | 183 | 204| 212 | 22455 | 77320 | 
| 73 | 24 django/utils/html.py | 129 | 138| 121 | 22576 | 77320 | 
| 74 | 24 django/utils/translation/__init__.py | 84 | 111| 136 | 22712 | 77320 | 
| 75 | 25 django/forms/utils.py | 81 | 107| 180 | 22892 | 79039 | 
| 76 | 25 django/template/base.py | 543 | 572| 245 | 23137 | 79039 | 
| 77 | 25 django/views/debug.py | 170 | 182| 148 | 23285 | 79039 | 
| 78 | 26 django/views/csrf.py | 1 | 13| 132 | 23417 | 80592 | 
| 79 | 26 django/template/defaulttags.py | 1326 | 1392| 508 | 23925 | 80592 | 
| 80 | 26 django/views/debug.py | 211 | 223| 143 | 24068 | 80592 | 
| 81 | 26 django/utils/translation/__init__.py | 114 | 163| 343 | 24411 | 80592 | 
| 82 | 26 django/template/smartif.py | 154 | 214| 426 | 24837 | 80592 | 
| 83 | 26 django/utils/translation/trans_real.py | 363 | 386| 179 | 25016 | 80592 | 
| 84 | 26 django/views/debug.py | 105 | 140| 262 | 25278 | 80592 | 
| 85 | 26 django/db/models/functions/text.py | 1 | 39| 266 | 25544 | 80592 | 
| 86 | 26 django/template/defaultfilters.py | 697 | 729| 217 | 25761 | 80592 | 
| 87 | 26 django/template/defaulttags.py | 1258 | 1286| 164 | 25925 | 80592 | 
| 88 | 26 docs/_ext/djangodocs.py | 383 | 402| 204 | 26129 | 80592 | 
| 89 | 26 django/utils/encoding.py | 1 | 53| 285 | 26414 | 80592 | 
| 90 | 27 django/views/defaults.py | 1 | 26| 151 | 26565 | 81575 | 
| 91 | 28 django/template/context_processors.py | 58 | 90| 143 | 26708 | 82070 | 
| 92 | 28 django/views/debug.py | 184 | 209| 181 | 26889 | 82070 | 
| 93 | 29 django/forms/boundfield.py | 106 | 154| 365 | 27254 | 84534 | 
| 94 | **29 django/templatetags/i18n.py** | 599 | 617| 121 | 27375 | 84534 | 
| 95 | 30 django/views/decorators/debug.py | 1 | 46| 273 | 27648 | 85125 | 
| 96 | 30 django/template/base.py | 574 | 606| 207 | 27855 | 85125 | 
| 97 | 30 django/template/base.py | 94 | 135| 228 | 28083 | 85125 | 
| 98 | 31 django/templatetags/cache.py | 1 | 55| 426 | 28509 | 85867 | 
| 99 | 31 django/forms/utils.py | 23 | 45| 195 | 28704 | 85867 | 
| 100 | 31 django/template/base.py | 138 | 177| 296 | 29000 | 85867 | 
| 101 | 31 django/template/defaulttags.py | 248 | 278| 263 | 29263 | 85867 | 
| 102 | 31 django/template/defaulttags.py | 869 | 889| 133 | 29396 | 85867 | 
| 103 | 31 django/utils/encoding.py | 78 | 107| 250 | 29646 | 85867 | 
| 104 | 31 django/views/csrf.py | 15 | 100| 839 | 30485 | 85867 | 
| 105 | 32 django/contrib/postgres/aggregates/general.py | 98 | 116| 178 | 30663 | 86687 | 
| 106 | 32 django/utils/html.py | 170 | 188| 159 | 30822 | 86687 | 
| 107 | 33 django/db/backends/oracle/base.py | 426 | 449| 200 | 31022 | 91836 | 
| 108 | 33 django/utils/translation/__init__.py | 268 | 302| 236 | 31258 | 91836 | 
| 109 | 34 django/template/backends/jinja2.py | 1 | 52| 344 | 31602 | 92652 | 
| 110 | 34 django/template/defaulttags.py | 455 | 483| 225 | 31827 | 92652 | 
| 111 | 35 django/shortcuts.py | 1 | 25| 161 | 31988 | 93776 | 
| 112 | 35 django/db/models/functions/text.py | 319 | 352| 250 | 32238 | 93776 | 
| 113 | 35 django/utils/html.py | 191 | 226| 311 | 32549 | 93776 | 
| 114 | 35 django/template/base.py | 945 | 996| 353 | 32902 | 93776 | 
| 115 | 36 django/contrib/humanize/templatetags/humanize.py | 86 | 115| 315 | 33217 | 96775 | 
| 116 | 36 django/template/base.py | 285 | 338| 359 | 33576 | 96775 | 
| 117 | 36 django/template/defaulttags.py | 437 | 453| 117 | 33693 | 96775 | 
| 118 | 37 django/template/library.py | 228 | 243| 126 | 33819 | 99327 | 
| 119 | 38 django/db/transaction.py | 136 | 179| 367 | 34186 | 101637 | 
| 120 | **38 django/templatetags/i18n.py** | 292 | 332| 246 | 34432 | 101637 | 
| 121 | 39 django/conf/__init__.py | 157 | 177| 184 | 34616 | 104039 | 
| 122 | 40 django/template/context.py | 133 | 168| 288 | 34904 | 105931 | 
| 123 | 40 django/utils/translation/__init__.py | 166 | 227| 345 | 35249 | 105931 | 
| 124 | 40 django/forms/boundfield.py | 1 | 34| 233 | 35482 | 105931 | 
| 125 | 40 django/template/defaulttags.py | 183 | 245| 532 | 36014 | 105931 | 
| 126 | 41 django/contrib/flatpages/views.py | 48 | 71| 191 | 36205 | 106521 | 
| 127 | 42 docs/conf.py | 129 | 232| 931 | 37136 | 110033 | 
| 128 | 43 django/templatetags/static.py | 59 | 92| 159 | 37295 | 111052 | 
| 129 | 44 django/contrib/auth/decorators.py | 1 | 40| 315 | 37610 | 111644 | 
| 130 | 45 django/contrib/syndication/views.py | 51 | 76| 202 | 37812 | 113502 | 
| 131 | 45 django/template/base.py | 402 | 433| 262 | 38074 | 113502 | 
| 132 | 45 django/template/defaulttags.py | 1460 | 1495| 250 | 38324 | 113502 | 
| 133 | 46 django/utils/jslex.py | 213 | 250| 281 | 38605 | 115330 | 
| 134 | 47 django/forms/forms.py | 226 | 317| 702 | 39307 | 119411 | 
| 135 | 47 django/template/defaulttags.py | 527 | 542| 142 | 39449 | 119411 | 
| 136 | **47 django/templatetags/i18n.py** | 242 | 262| 178 | 39627 | 119411 | 
| 137 | 48 django/core/mail/message.py | 55 | 75| 177 | 39804 | 123123 | 
| 138 | 48 django/utils/translation/trans_real.py | 66 | 121| 408 | 40212 | 123123 | 
| 139 | 49 django/db/backends/sqlite3/base.py | 343 | 365| 183 | 40395 | 126154 | 
| 140 | 49 django/db/backends/oracle/base.py | 519 | 550| 432 | 40827 | 126154 | 
| 141 | 49 django/contrib/humanize/templatetags/humanize.py | 118 | 142| 201 | 41028 | 126154 | 
| 142 | 49 django/template/defaulttags.py | 418 | 434| 121 | 41149 | 126154 | 
| 143 | 49 django/views/i18n.py | 300 | 323| 167 | 41316 | 126154 | 
| 144 | 49 django/db/backends/oracle/base.py | 402 | 423| 143 | 41459 | 126154 | 
| 145 | 49 django/core/management/commands/makemessages.py | 71 | 92| 146 | 41605 | 126154 | 
| 146 | 49 django/utils/jslex.py | 101 | 166| 585 | 42190 | 126154 | 
| 147 | 49 django/template/defaultfilters.py | 505 | 545| 248 | 42438 | 126154 | 
| 148 | 49 django/utils/text.py | 32 | 62| 247 | 42685 | 126154 | 
| 149 | 49 django/contrib/humanize/templatetags/humanize.py | 258 | 279| 232 | 42917 | 126154 | 
| 150 | 50 django/db/models/sql/compiler.py | 476 | 504| 242 | 43159 | 141696 | 
| 151 | 51 django/conf/global_settings.py | 56 | 156| 1171 | 44330 | 147552 | 
| 152 | **51 django/templatetags/i18n.py** | 335 | 355| 173 | 44503 | 147552 | 
| 153 | 52 django/template/__init__.py | 1 | 76| 394 | 44897 | 147946 | 
| 154 | 52 django/utils/translation/__init__.py | 230 | 265| 271 | 45168 | 147946 | 
| 155 | 53 django/utils/regex_helper.py | 1 | 38| 250 | 45418 | 150587 | 
| 156 | 53 django/template/defaulttags.py | 67 | 89| 164 | 45582 | 150587 | 
| 157 | 54 django/views/decorators/common.py | 1 | 17| 112 | 45694 | 150700 | 
| 158 | 54 django/views/debug.py | 1 | 56| 351 | 46045 | 150700 | 
| 159 | 55 django/utils/asyncio.py | 1 | 40| 221 | 46266 | 150922 | 
| 160 | 55 django/templatetags/cache.py | 58 | 101| 315 | 46581 | 150922 | 
| 161 | 55 django/template/defaulttags.py | 1094 | 1138| 370 | 46951 | 150922 | 
| 162 | 55 django/db/models/sql/compiler.py | 1456 | 1490| 331 | 47282 | 150922 | 
| 163 | 55 django/utils/html.py | 229 | 249| 253 | 47535 | 150922 | 
| 164 | 56 django/template/utils.py | 67 | 94| 197 | 47732 | 151638 | 
| 165 | 56 django/utils/translation/trans_real.py | 281 | 360| 458 | 48190 | 151638 | 
| 166 | 57 django/template/engine.py | 181 | 195| 138 | 48328 | 153136 | 
| 167 | 58 django/views/generic/base.py | 210 | 244| 228 | 48556 | 155009 | 
| 168 | 58 django/db/backends/oracle/base.py | 552 | 592| 322 | 48878 | 155009 | 
| 169 | 58 django/template/base.py | 367 | 399| 372 | 49250 | 155009 | 
| 170 | 58 django/contrib/humanize/templatetags/humanize.py | 207 | 257| 647 | 49897 | 155009 | 
| 171 | 59 django/templatetags/tz.py | 164 | 183| 175 | 50072 | 156361 | 
| 172 | 60 django/core/validators.py | 113 | 159| 474 | 50546 | 161073 | 
| 173 | 60 docs/_ext/djangodocs.py | 216 | 240| 151 | 50697 | 161073 | 
| 174 | 61 django/core/checks/templates.py | 1 | 47| 311 | 51008 | 161553 | 
| 175 | 61 django/forms/utils.py | 58 | 78| 150 | 51158 | 161553 | 
| 176 | 61 django/utils/text.py | 65 | 86| 189 | 51347 | 161553 | 
| 177 | 61 django/template/context.py | 171 | 212| 296 | 51643 | 161553 | 
| 178 | 61 django/template/defaultfilters.py | 231 | 258| 155 | 51798 | 161553 | 
| 179 | 61 django/template/defaultfilters.py | 869 | 914| 379 | 52177 | 161553 | 
| 180 | 62 django/db/models/functions/comparison.py | 104 | 119| 178 | 52355 | 163308 | 
| 181 | 62 django/utils/translation/trans_real.py | 229 | 244| 158 | 52513 | 163308 | 
| 182 | 62 django/utils/text.py | 359 | 375| 198 | 52711 | 163308 | 
| 183 | 63 django/db/models/fields/json.py | 310 | 341| 310 | 53021 | 167586 | 
| 184 | 63 django/template/engine.py | 1 | 63| 397 | 53418 | 167586 | 
| 185 | 64 django/utils/functional.py | 215 | 271| 323 | 53741 | 170869 | 
| 186 | 65 django/utils/dateformat.py | 38 | 51| 121 | 53862 | 173472 | 
| 187 | 65 django/utils/text.py | 251 | 280| 235 | 54097 | 173472 | 
| 188 | 65 django/template/defaulttags.py | 486 | 524| 252 | 54349 | 173472 | 
| 189 | 65 django/template/base.py | 749 | 773| 210 | 54559 | 173472 | 
| 190 | 66 django/http/request.py | 586 | 616| 208 | 54767 | 178771 | 
| 191 | **66 django/templatetags/i18n.py** | 265 | 289| 234 | 55001 | 178771 | 
| 192 | 66 django/utils/html.py | 141 | 167| 149 | 55150 | 178771 | 
| 193 | 66 django/views/csrf.py | 101 | 161| 581 | 55731 | 178771 | 
| 194 | 66 django/core/validators.py | 68 | 111| 609 | 56340 | 178771 | 
| 195 | 66 docs/conf.py | 54 | 127| 672 | 57012 | 178771 | 
| 196 | 67 django/db/backends/sqlite3/schema.py | 43 | 72| 245 | 57257 | 183344 | 
| 197 | 67 django/core/management/commands/makemessages.py | 109 | 126| 139 | 57396 | 183344 | 
| 198 | 67 django/templatetags/static.py | 95 | 131| 242 | 57638 | 183344 | 
| 199 | 68 django/contrib/messages/storage/base.py | 1 | 41| 264 | 57902 | 184583 | 
| 200 | 68 django/core/validators.py | 352 | 384| 231 | 58133 | 184583 | 
| 201 | 68 django/template/base.py | 521 | 541| 181 | 58314 | 184583 | 
| 202 | 68 django/utils/html.py | 329 | 368| 348 | 58662 | 184583 | 


### Hint

```
Hi Richard, thanks for the report. So this would be the way forward: ... assign a SafeString instance to prevent future escaping But it's not at all clear how feasible it would be to correctly mark the returned string as safe. Individual variables are run via render_value_in_context() which escapes them assuming autoescape is enabled, but then the final output is constructed after that, so it's not clear we can reliably mark it safe. Rather if, in your example, you know the_title is safe, declare it as so: {{ the_title|safe }}. The following test case passes: diff --git a/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py b/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py index 4a162362c6..967a7c1829 100644 --- a/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py +++ b/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py @@ -388,6 +388,23 @@ class I18nBlockTransTagTests(SimpleTestCase): output = self.engine.render_to_string("i18n39") self.assertEqual(output, ">Seite nicht gefunden<") + @setup( + { + "issue33631": ( + """ + {% load i18n %} + {% blocktranslate asvar the_title %}The title is {{title}}.{% endblocktranslate %} + < title > {{the_title|safe}} < / title > + < meta name="description" content="{{ the_title|safe }}" > + """ + ) + } + ) + def test_issue33631(self): + with translation.override("en"): + output = self.engine.render_to_string("issue33631", {"title": "<>& Title"}) + self.assertNotIn("&amp;lt;", output) + @setup( { ... and it avoids trying to resolve the difficulty above. As such, I'm going to say wontfix here initially, and ask that you follow-up on the ​Internationalization Topics section of the Django Forum to get a wider audience if you'd like to discuss it further. Thanks.
Reopening to assess a patch ​based on forum discussion.
PR ​https://github.com/django/django/pull/15742
```

## Patch

```diff
diff --git a/django/templatetags/i18n.py b/django/templatetags/i18n.py
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -5,7 +5,7 @@
 from django.template.base import TokenType, render_value_in_context
 from django.template.defaulttags import token_kwargs
 from django.utils import translation
-from django.utils.safestring import SafeData, mark_safe
+from django.utils.safestring import SafeData, SafeString, mark_safe
 
 register = Library()
 
@@ -198,7 +198,7 @@ def render_value(key):
             with translation.override(None):
                 result = self.render(context, nested=True)
         if self.asvar:
-            context[self.asvar] = result
+            context[self.asvar] = SafeString(result)
             return ""
         else:
             return result

```

## Test Patch

```diff
diff --git a/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py b/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py
--- a/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py
+++ b/tests/template_tests/syntax_tests/i18n/test_blocktranslate.py
@@ -416,6 +416,22 @@ def test_i18n41(self):
             output = self.engine.render_to_string("i18n41")
         self.assertEqual(output, ">Error: Seite nicht gefunden<")
 
+    @setup(
+        {
+            "i18n_asvar_safestring": (
+                "{% load i18n %}"
+                "{% blocktranslate asvar the_title %}"
+                "{{title}}other text"
+                "{% endblocktranslate %}"
+                "{{ the_title }}"
+            )
+        }
+    )
+    def test_i18n_asvar_safestring(self):
+        context = {"title": "<Main Title>"}
+        output = self.engine.render_to_string("i18n_asvar_safestring", context=context)
+        self.assertEqual(output, "&lt;Main Title&gt;other text")
+
     @setup(
         {
             "template": (

```


## Code snippets

### 1 - django/templatetags/i18n.py:

Start line: 456, End line: 567

```python
@register.tag("blocktranslate")
@register.tag("blocktrans")
def do_block_translate(parser, token):
    """
    Translate a block of text with parameters.

    Usage::

        {% blocktranslate with bar=foo|filter boo=baz|filter %}
        This is {{ bar }} and {{ boo }}.
        {% endblocktranslate %}

    Additionally, this supports pluralization::

        {% blocktranslate count count=var|length %}
        There is {{ count }} object.
        {% plural %}
        There are {{ count }} objects.
        {% endblocktranslate %}

    This is much like ngettext, only in template syntax.

    The "var as value" legacy format is still supported::

        {% blocktranslate with foo|filter as bar and baz|filter as boo %}
        {% blocktranslate count var|length as count %}

    The translated string can be stored in a variable using `asvar`::

        {% blocktranslate with bar=foo|filter boo=baz|filter asvar var %}
        This is {{ bar }} and {{ boo }}.
        {% endblocktranslate %}
        {{ var }}

    Contextual translations are supported::

        {% blocktranslate with bar=foo|filter context "greeting" %}
            This is {{ bar }}.
        {% endblocktranslate %}

    This is equivalent to calling pgettext/npgettext instead of
    (u)gettext/(u)ngettext.
    """
    bits = token.split_contents()

    options = {}
    remaining_bits = bits[1:]
    asvar = None
    while remaining_bits:
        option = remaining_bits.pop(0)
        if option in options:
            raise TemplateSyntaxError(
                "The %r option was specified more than once." % option
            )
        if option == "with":
            value = token_kwargs(remaining_bits, parser, support_legacy=True)
            if not value:
                raise TemplateSyntaxError(
                    '"with" in %r tag needs at least one keyword argument.' % bits[0]
                )
        elif option == "count":
            value = token_kwargs(remaining_bits, parser, support_legacy=True)
            if len(value) != 1:
                raise TemplateSyntaxError(
                    '"count" in %r tag expected exactly '
                    "one keyword argument." % bits[0]
                )
        elif option == "context":
            try:
                value = remaining_bits.pop(0)
                value = parser.compile_filter(value)
            except Exception:
                raise TemplateSyntaxError(
                    '"context" in %r tag expected exactly one argument.' % bits[0]
                )
        elif option == "trimmed":
            value = True
        elif option == "asvar":
            try:
                value = remaining_bits.pop(0)
            except IndexError:
                raise TemplateSyntaxError(
                    "No argument provided to the '%s' tag for the asvar option."
                    % bits[0]
                )
            asvar = value
        else:
            raise TemplateSyntaxError(
                "Unknown argument for %r tag: %r." % (bits[0], option)
            )
        options[option] = value

    if "count" in options:
        countervar, counter = next(iter(options["count"].items()))
    else:
        countervar, counter = None, None
    if "context" in options:
        message_context = options["context"]
    else:
        message_context = None
    extra_context = options.get("with", {})

    trimmed = options.get("trimmed", False)

    singular = []
    plural = []
    while parser.tokens:
        token = parser.next_token()
        if token.token_type in (TokenType.VAR, TokenType.TEXT):
            singular.append(token)
        else:
            break
    # ... other code
```
### 2 - django/templatetags/i18n.py:

Start line: 148, End line: 204

```python
class BlockTranslateNode(Node):

    def render(self, context, nested=False):
        if self.message_context:
            message_context = self.message_context.resolve(context)
        else:
            message_context = None
        # Update() works like a push(), so corresponding context.pop() is at
        # the end of function
        context.update(
            {var: val.resolve(context) for var, val in self.extra_context.items()}
        )
        singular, vars = self.render_token_list(self.singular)
        if self.plural and self.countervar and self.counter:
            count = self.counter.resolve(context)
            if not isinstance(count, (Decimal, float, int)):
                raise TemplateSyntaxError(
                    "%r argument to %r tag must be a number."
                    % (self.countervar, self.tag_name)
                )
            context[self.countervar] = count
            plural, plural_vars = self.render_token_list(self.plural)
            if message_context:
                result = translation.npgettext(message_context, singular, plural, count)
            else:
                result = translation.ngettext(singular, plural, count)
            vars.extend(plural_vars)
        else:
            if message_context:
                result = translation.pgettext(message_context, singular)
            else:
                result = translation.gettext(singular)
        default_value = context.template.engine.string_if_invalid

        def render_value(key):
            if key in context:
                val = context[key]
            else:
                val = default_value % key if "%s" in default_value else default_value
            return render_value_in_context(val, context)

        data = {v: render_value(v) for v in vars}
        context.pop()
        try:
            result = result % data
        except (KeyError, ValueError):
            if nested:
                # Either string is malformed, or it's a bug
                raise TemplateSyntaxError(
                    "%r is unable to format string returned by gettext: %r "
                    "using %r" % (self.tag_name, result, data)
                )
            with translation.override(None):
                result = self.render(context, nested=True)
        if self.asvar:
            context[self.asvar] = result
            return ""
        else:
            return result
```
### 3 - django/templatetags/i18n.py:

Start line: 568, End line: 596

```python
@register.tag("blocktranslate")
@register.tag("blocktrans")
def do_block_translate(parser, token):
    # ... other code
    if countervar and counter:
        if token.contents.strip() != "plural":
            raise TemplateSyntaxError(
                "%r doesn't allow other block tags inside it" % bits[0]
            )
        while parser.tokens:
            token = parser.next_token()
            if token.token_type in (TokenType.VAR, TokenType.TEXT):
                plural.append(token)
            else:
                break
    end_tag_name = "end%s" % bits[0]
    if token.contents.strip() != end_tag_name:
        raise TemplateSyntaxError(
            "%r doesn't allow other block tags (seen %r) inside it"
            % (bits[0], token.contents)
        )

    return BlockTranslateNode(
        extra_context,
        singular,
        plural,
        countervar,
        counter,
        message_context,
        trimmed=trimmed,
        asvar=asvar,
        tag_name=bits[0],
    )
```
### 4 - django/templatetags/i18n.py:

Start line: 104, End line: 146

```python
class BlockTranslateNode(Node):
    def __init__(
        self,
        extra_context,
        singular,
        plural=None,
        countervar=None,
        counter=None,
        message_context=None,
        trimmed=False,
        asvar=None,
        tag_name="blocktranslate",
    ):
        self.extra_context = extra_context
        self.singular = singular
        self.plural = plural
        self.countervar = countervar
        self.counter = counter
        self.message_context = message_context
        self.trimmed = trimmed
        self.asvar = asvar
        self.tag_name = tag_name

    def __repr__(self):
        return (
            f"<{self.__class__.__qualname__}: "
            f"extra_context={self.extra_context!r} "
            f"singular={self.singular!r} plural={self.plural!r}>"
        )

    def render_token_list(self, tokens):
        result = []
        vars = []
        for token in tokens:
            if token.token_type == TokenType.TEXT:
                result.append(token.contents.replace("%", "%%"))
            elif token.token_type == TokenType.VAR:
                result.append("%%(%s)s" % token.contents)
                vars.append(token.contents)
        msg = "".join(result)
        if self.trimmed:
            msg = translation.trim_whitespace(msg)
        return msg, vars
```
### 5 - django/utils/translation/template.py:

Start line: 65, End line: 247

```python
def templatize(src, origin=None):
    # ... other code

    for t in Lexer(src).tokenize():
        if incomment:
            if t.token_type == TokenType.BLOCK and t.contents == "endcomment":
                content = "".join(comment)
                translators_comment_start = None
                for lineno, line in enumerate(content.splitlines(True)):
                    if line.lstrip().startswith(TRANSLATOR_COMMENT_MARK):
                        translators_comment_start = lineno
                for lineno, line in enumerate(content.splitlines(True)):
                    if (
                        translators_comment_start is not None
                        and lineno >= translators_comment_start
                    ):
                        out.write(" # %s" % line)
                    else:
                        out.write(" #\n")
                incomment = False
                comment = []
            else:
                comment.append(t.contents)
        elif intrans:
            if t.token_type == TokenType.BLOCK:
                endbmatch = endblock_re.match(t.contents)
                pluralmatch = plural_re.match(t.contents)
                if endbmatch:
                    if inplural:
                        if message_context:
                            out.write(
                                " npgettext({p}{!r}, {p}{!r}, {p}{!r},count) ".format(
                                    message_context,
                                    join_tokens(singular, trimmed),
                                    join_tokens(plural, trimmed),
                                    p=raw_prefix,
                                )
                            )
                        else:
                            out.write(
                                " ngettext({p}{!r}, {p}{!r}, count) ".format(
                                    join_tokens(singular, trimmed),
                                    join_tokens(plural, trimmed),
                                    p=raw_prefix,
                                )
                            )
                        for part in singular:
                            out.write(blankout(part, "S"))
                        for part in plural:
                            out.write(blankout(part, "P"))
                    else:
                        if message_context:
                            out.write(
                                " pgettext({p}{!r}, {p}{!r}) ".format(
                                    message_context,
                                    join_tokens(singular, trimmed),
                                    p=raw_prefix,
                                )
                            )
                        else:
                            out.write(
                                " gettext({p}{!r}) ".format(
                                    join_tokens(singular, trimmed),
                                    p=raw_prefix,
                                )
                            )
                        for part in singular:
                            out.write(blankout(part, "S"))
                    message_context = None
                    intrans = False
                    inplural = False
                    singular = []
                    plural = []
                elif pluralmatch:
                    inplural = True
                else:
                    filemsg = ""
                    if origin:
                        filemsg = "file %s, " % origin
                    raise SyntaxError(
                        "Translation blocks must not include other block tags: "
                        "%s (%sline %d)" % (t.contents, filemsg, t.lineno)
                    )
            elif t.token_type == TokenType.VAR:
                if inplural:
                    plural.append("%%(%s)s" % t.contents)
                else:
                    singular.append("%%(%s)s" % t.contents)
            elif t.token_type == TokenType.TEXT:
                contents = t.contents.replace("%", "%%")
                if inplural:
                    plural.append(contents)
                else:
                    singular.append(contents)
        else:
            # Handle comment tokens (`{# ... #}`) plus other constructs on
            # the same line:
            if comment_lineno_cache is not None:
                cur_lineno = t.lineno + t.contents.count("\n")
                if comment_lineno_cache == cur_lineno:
                    if t.token_type != TokenType.COMMENT:
                        for c in lineno_comment_map[comment_lineno_cache]:
                            filemsg = ""
                            if origin:
                                filemsg = "file %s, " % origin
                            warn_msg = (
                                "The translator-targeted comment '%s' "
                                "(%sline %d) was ignored, because it wasn't "
                                "the last item on the line."
                            ) % (c, filemsg, comment_lineno_cache)
                            warnings.warn(warn_msg, TranslatorCommentWarning)
                        lineno_comment_map[comment_lineno_cache] = []
                else:
                    out.write(
                        "# %s" % " | ".join(lineno_comment_map[comment_lineno_cache])
                    )
                comment_lineno_cache = None

            if t.token_type == TokenType.BLOCK:
                imatch = inline_re.match(t.contents)
                bmatch = block_re.match(t.contents)
                cmatches = constant_re.findall(t.contents)
                if imatch:
                    g = imatch[1]
                    if g[0] == '"':
                        g = g.strip('"')
                    elif g[0] == "'":
                        g = g.strip("'")
                    g = g.replace("%", "%%")
                    if imatch[2]:
                        # A context is provided
                        context_match = context_re.match(imatch[2])
                        message_context = context_match[1]
                        if message_context[0] == '"':
                            message_context = message_context.strip('"')
                        elif message_context[0] == "'":
                            message_context = message_context.strip("'")
                        out.write(
                            " pgettext({p}{!r}, {p}{!r}) ".format(
                                message_context, g, p=raw_prefix
                            )
                        )
                        message_context = None
                    else:
                        out.write(" gettext({p}{!r}) ".format(g, p=raw_prefix))
                elif bmatch:
                    for fmatch in constant_re.findall(t.contents):
                        out.write(" _(%s) " % fmatch)
                    if bmatch[1]:
                        # A context is provided
                        context_match = context_re.match(bmatch[1])
                        message_context = context_match[1]
                        if message_context[0] == '"':
                            message_context = message_context.strip('"')
                        elif message_context[0] == "'":
                            message_context = message_context.strip("'")
                    intrans = True
                    inplural = False
                    trimmed = "trimmed" in t.split_contents()
                    singular = []
                    plural = []
                elif cmatches:
                    for cmatch in cmatches:
                        out.write(" _(%s) " % cmatch)
                elif t.contents == "comment":
                    incomment = True
                else:
                    out.write(blankout(t.contents, "B"))
            elif t.token_type == TokenType.VAR:
                parts = t.contents.split("|")
                cmatch = constant_re.match(parts[0])
                if cmatch:
                    out.write(" _(%s) " % cmatch[1])
                for p in parts[1:]:
                    if p.find(":_(") >= 0:
                        out.write(" %s " % p.split(":", 1)[1])
                    else:
                        out.write(blankout(p, "F"))
            elif t.token_type == TokenType.COMMENT:
                if t.contents.lstrip().startswith(TRANSLATOR_COMMENT_MARK):
                    lineno_comment_map.setdefault(t.lineno, []).append(t.contents)
                    comment_lineno_cache = t.lineno
            else:
                out.write(blankout(t.contents, "X"))
    return out.getvalue()
```
### 6 - django/templatetags/i18n.py:

Start line: 358, End line: 453

```python
@register.tag("translate")
@register.tag("trans")
def do_translate(parser, token):
    """
    Mark a string for translation and translate the string for the current
    language.

    Usage::

        {% translate "this is a test" %}

    This marks the string for translation so it will be pulled out by
    makemessages into the .po files and runs the string through the translation
    engine.

    There is a second form::

        {% translate "this is a test" noop %}

    This marks the string for translation, but returns the string unchanged.
    Use it when you need to store values into forms that should be translated
    later on.

    You can use variables instead of constant strings
    to translate stuff you marked somewhere else::

        {% translate variable %}

    This tries to translate the contents of the variable ``variable``. Make
    sure that the string in there is something that is in the .po file.

    It is possible to store the translated string into a variable::

        {% translate "this is a test" as var %}
        {{ var }}

    Contextual translations are also supported::

        {% translate "this is a test" context "greeting" %}

    This is equivalent to calling pgettext instead of (u)gettext.
    """
    bits = token.split_contents()
    if len(bits) < 2:
        raise TemplateSyntaxError("'%s' takes at least one argument" % bits[0])
    message_string = parser.compile_filter(bits[1])
    remaining = bits[2:]

    noop = False
    asvar = None
    message_context = None
    seen = set()
    invalid_context = {"as", "noop"}

    while remaining:
        option = remaining.pop(0)
        if option in seen:
            raise TemplateSyntaxError(
                "The '%s' option was specified more than once." % option,
            )
        elif option == "noop":
            noop = True
        elif option == "context":
            try:
                value = remaining.pop(0)
            except IndexError:
                raise TemplateSyntaxError(
                    "No argument provided to the '%s' tag for the context option."
                    % bits[0]
                )
            if value in invalid_context:
                raise TemplateSyntaxError(
                    "Invalid argument '%s' provided to the '%s' tag for the context "
                    "option" % (value, bits[0]),
                )
            message_context = parser.compile_filter(value)
        elif option == "as":
            try:
                value = remaining.pop(0)
            except IndexError:
                raise TemplateSyntaxError(
                    "No argument provided to the '%s' tag for the as option." % bits[0]
                )
            asvar = value
        else:
            raise TemplateSyntaxError(
                "Unknown argument for '%s' tag: '%s'. The only options "
                "available are 'noop', 'context' \"xxx\", and 'as VAR'."
                % (
                    bits[0],
                    option,
                )
            )
        seen.add(option)

    return TranslateNode(message_string, noop, asvar, message_context)
```
### 7 - django/utils/translation/template.py:

Start line: 1, End line: 36

```python
import warnings
from io import StringIO

from django.template.base import Lexer, TokenType
from django.utils.regex_helper import _lazy_re_compile

from . import TranslatorCommentWarning, trim_whitespace

TRANSLATOR_COMMENT_MARK = "Translators"

dot_re = _lazy_re_compile(r"\S")


def blankout(src, char):
    """
    Change every non-whitespace character to the given char.
    Used in the templatize function.
    """
    return dot_re.sub(char, src)


context_re = _lazy_re_compile(r"""^\s+.*context\s+((?:"[^"]*?")|(?:'[^']*?'))\s*""")
inline_re = _lazy_re_compile(
    # Match the trans/translate 'some text' part.
    r"""^\s*trans(?:late)?\s+((?:"[^"]*?")|(?:'[^']*?'))"""
    # Match and ignore optional filters
    r"""(?:\s*\|\s*[^\s:]+(?::(?:[^\s'":]+|(?:"[^"]*?")|(?:'[^']*?')))?)*"""
    # Match the optional context part
    r"""(\s+.*context\s+((?:"[^"]*?")|(?:'[^']*?')))?\s*"""
)
block_re = _lazy_re_compile(
    r"""^\s*blocktrans(?:late)?(\s+.*context\s+((?:"[^"]*?")|(?:'[^']*?')))?(?:\s+|$)"""
)
endblock_re = _lazy_re_compile(r"""^\s*endblocktrans(?:late)?$""")
plural_re = _lazy_re_compile(r"""^\s*plural$""")
constant_re = _lazy_re_compile(r"""_\(((?:".*?")|(?:'.*?'))\)""")
```
### 8 - django/template/base.py:

Start line: 776, End line: 841

```python
class Variable:
    """
    A template variable, resolvable against a given context. The variable may
    be a hard-coded string (if it begins and ends with single or double quote
    marks)::

        >>> c = {'article': {'section':'News'}}
        >>> Variable('article.section').resolve(c)
        'News'
        >>> Variable('article').resolve(c)
        {'section': 'News'}
        >>> class AClass: pass
        >>> c = AClass()
        >>> c.article = AClass()
        >>> c.article.section = 'News'

    (The example assumes VARIABLE_ATTRIBUTE_SEPARATOR is '.')
    """

    __slots__ = ("var", "literal", "lookups", "translate", "message_context")

    def __init__(self, var):
        self.var = var
        self.literal = None
        self.lookups = None
        self.translate = False
        self.message_context = None

        if not isinstance(var, str):
            raise TypeError("Variable must be a string or number, got %s" % type(var))
        try:
            # First try to treat this variable as a number.
            #
            # Note that this could cause an OverflowError here that we're not
            # catching. Since this should only happen at compile time, that's
            # probably OK.

            # Try to interpret values containing a period or an 'e'/'E'
            # (possibly scientific notation) as a float;  otherwise, try int.
            if "." in var or "e" in var.lower():
                self.literal = float(var)
                # "2." is invalid
                if var[-1] == ".":
                    raise ValueError
            else:
                self.literal = int(var)
        except ValueError:
            # A ValueError means that the variable isn't a number.
            if var[0:2] == "_(" and var[-1] == ")":
                # The result of the lookup should be translated at rendering
                # time.
                self.translate = True
                var = var[2:-1]
            # If it's wrapped with quotes (single or double), then
            # we're also dealing with a literal.
            try:
                self.literal = mark_safe(unescape_string_literal(var))
            except ValueError:
                # Otherwise we'll set self.lookups so that resolve() knows we're
                # dealing with a bonafide variable
                if VARIABLE_ATTRIBUTE_SEPARATOR + "_" in var or var[0] == "_":
                    raise TemplateSyntaxError(
                        "Variables and attributes may "
                        "not begin with underscores: '%s'" % var
                    )
                self.lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))
```
### 9 - django/template/base.py:

Start line: 1037, End line: 1050

```python
def render_value_in_context(value, context):
    """
    Convert any value to a string to become part of a rendered template. This
    means escaping, if required, and conversion to a string. If value is a
    string, it's expected to already be translated.
    """
    value = template_localtime(value, use_tz=context.use_tz)
    value = localize(value, use_l10n=context.use_l10n)
    if context.autoescape:
        if not issubclass(type(value), str):
            value = str(value)
        return conditional_escape(value)
    else:
        return str(value)
```
### 10 - django/templatetags/i18n.py:

Start line: 72, End line: 101

```python
class TranslateNode(Node):
    child_nodelists = ()

    def __init__(self, filter_expression, noop, asvar=None, message_context=None):
        self.noop = noop
        self.asvar = asvar
        self.message_context = message_context
        self.filter_expression = filter_expression
        if isinstance(self.filter_expression.var, str):
            self.filter_expression.is_var = True
            self.filter_expression.var = Variable("'%s'" % self.filter_expression.var)

    def render(self, context):
        self.filter_expression.var.translate = not self.noop
        if self.message_context:
            self.filter_expression.var.message_context = self.message_context.resolve(
                context
            )
        output = self.filter_expression.resolve(context)
        value = render_value_in_context(output, context)
        # Restore percent signs. Percent signs in template text are doubled
        # so they are not interpreted as string format flags.
        is_safe = isinstance(value, SafeData)
        value = value.replace("%%", "%")
        value = mark_safe(value) if is_safe else value
        if self.asvar:
            context[self.asvar] = value
            return ""
        else:
            return value
```
### 54 - django/templatetags/i18n.py:

Start line: 1, End line: 32

```python
from decimal import Decimal

from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, mark_safe

register = Library()


class GetAvailableLanguagesNode(Node):
    def __init__(self, variable):
        self.variable = variable

    def render(self, context):
        context[self.variable] = [
            (k, translation.gettext(v)) for k, v in settings.LANGUAGES
        ]
        return ""


class GetLanguageInfoNode(Node):
    def __init__(self, lang_code, variable):
        self.lang_code = lang_code
        self.variable = variable

    def render(self, context):
        lang_code = self.lang_code.resolve(context)
        context[self.variable] = translation.get_language_info(lang_code)
        return ""
```
### 62 - django/templatetags/i18n.py:

Start line: 35, End line: 69

```python
class GetLanguageInfoListNode(Node):
    def __init__(self, languages, variable):
        self.languages = languages
        self.variable = variable

    def get_language_info(self, language):
        # ``language`` is either a language code string or a sequence
        # with the language code as its first item
        if len(language[0]) > 1:
            return translation.get_language_info(language[0])
        else:
            return translation.get_language_info(str(language))

    def render(self, context):
        langs = self.languages.resolve(context)
        context[self.variable] = [self.get_language_info(lang) for lang in langs]
        return ""


class GetCurrentLanguageNode(Node):
    def __init__(self, variable):
        self.variable = variable

    def render(self, context):
        context[self.variable] = translation.get_language()
        return ""


class GetCurrentLanguageBidiNode(Node):
    def __init__(self, variable):
        self.variable = variable

    def render(self, context):
        context[self.variable] = translation.get_language_bidi()
        return ""
```
### 94 - django/templatetags/i18n.py:

Start line: 599, End line: 617

```python
@register.tag
def language(parser, token):
    """
    Enable the given language just for this block.

    Usage::

        {% language "de" %}
            This is {{ bar }} and {{ boo }}.
        {% endlanguage %}
    """
    bits = token.split_contents()
    if len(bits) != 2:
        raise TemplateSyntaxError("'%s' takes one argument (language)" % bits[0])
    language = parser.compile_filter(bits[1])
    nodelist = parser.parse(("endlanguage",))
    parser.delete_first_token()
    return LanguageNode(nodelist, language)
```
### 120 - django/templatetags/i18n.py:

Start line: 292, End line: 332

```python
@register.filter
def language_name(lang_code):
    return translation.get_language_info(lang_code)["name"]


@register.filter
def language_name_translated(lang_code):
    english_name = translation.get_language_info(lang_code)["name"]
    return translation.gettext(english_name)


@register.filter
def language_name_local(lang_code):
    return translation.get_language_info(lang_code)["name_local"]


@register.filter
def language_bidi(lang_code):
    return translation.get_language_info(lang_code)["bidi"]


@register.tag("get_current_language")
def do_get_current_language(parser, token):
    """
    Store the current language in the context.

    Usage::

        {% get_current_language as language %}

    This fetches the currently active language and puts its value into the
    ``language`` context variable.
    """
    # token.split_contents() isn't useful here because this tag doesn't accept
    # variable as arguments.
    args = token.contents.split()
    if len(args) != 3 or args[1] != "as":
        raise TemplateSyntaxError(
            "'get_current_language' requires 'as variable' (got %r)" % args
        )
    return GetCurrentLanguageNode(args[2])
```
### 136 - django/templatetags/i18n.py:

Start line: 242, End line: 262

```python
@register.tag("get_language_info")
def do_get_language_info(parser, token):
    """
    Store the language information dictionary for the given language code in a
    context variable.

    Usage::

        {% get_language_info for LANGUAGE_CODE as l %}
        {{ l.code }}
        {{ l.name }}
        {{ l.name_translated }}
        {{ l.name_local }}
        {{ l.bidi|yesno:"bi-directional,uni-directional" }}
    """
    args = token.split_contents()
    if len(args) != 5 or args[1] != "for" or args[3] != "as":
        raise TemplateSyntaxError(
            "'%s' requires 'for string as variable' (got %r)" % (args[0], args[1:])
        )
    return GetLanguageInfoNode(parser.compile_filter(args[2]), args[4])
```
### 152 - django/templatetags/i18n.py:

Start line: 335, End line: 355

```python
@register.tag("get_current_language_bidi")
def do_get_current_language_bidi(parser, token):
    """
    Store the current language layout in the context.

    Usage::

        {% get_current_language_bidi as bidi %}

    This fetches the currently active language's layout and puts its value into
    the ``bidi`` context variable. True indicates right-to-left layout,
    otherwise left-to-right.
    """
    # token.split_contents() isn't useful here because this tag doesn't accept
    # variable as arguments.
    args = token.contents.split()
    if len(args) != 3 or args[1] != "as":
        raise TemplateSyntaxError(
            "'get_current_language_bidi' requires 'as variable' (got %r)" % args
        )
    return GetCurrentLanguageBidiNode(args[2])
```
### 191 - django/templatetags/i18n.py:

Start line: 265, End line: 289

```python
@register.tag("get_language_info_list")
def do_get_language_info_list(parser, token):
    """
    Store a list of language information dictionaries for the given language
    codes in a context variable. The language codes can be specified either as
    a list of strings or a settings.LANGUAGES style list (or any sequence of
    sequences whose first items are language codes).

    Usage::

        {% get_language_info_list for LANGUAGES as langs %}
        {% for l in langs %}
          {{ l.code }}
          {{ l.name }}
          {{ l.name_translated }}
          {{ l.name_local }}
          {{ l.bidi|yesno:"bi-directional,uni-directional" }}
        {% endfor %}
    """
    args = token.split_contents()
    if len(args) != 5 or args[1] != "for" or args[3] != "as":
        raise TemplateSyntaxError(
            "'%s' requires 'for sequence as variable' (got %r)" % (args[0], args[1:])
        )
    return GetLanguageInfoListNode(parser.compile_filter(args[2]), args[4])
```
