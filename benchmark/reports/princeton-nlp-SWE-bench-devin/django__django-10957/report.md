# django__django-10957

| **django/django** | `48c17807a99f7a4341c74db19e16a37b010827c2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2442 |
| **Any found context length** | 180 |
| **Avg pos** | 17.0 |
| **Min pos** | 1 |
| **Max pos** | 9 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -2,9 +2,11 @@
 Internationalization support.
 """
 import re
+import warnings
 from contextlib import ContextDecorator
 
 from django.utils.autoreload import autoreload_started, file_changed
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.functional import lazy
 
 __all__ = [
@@ -72,23 +74,51 @@ def gettext_noop(message):
     return _trans.gettext_noop(message)
 
 
-ugettext_noop = gettext_noop
+def ugettext_noop(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of gettext_noop() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext_noop() is deprecated in favor of '
+        'django.utils.translation.gettext_noop().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext_noop(message)
 
 
 def gettext(message):
     return _trans.gettext(message)
 
 
-# An alias since Django 2.0
-ugettext = gettext
+def ugettext(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of gettext() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext() is deprecated in favor of '
+        'django.utils.translation.gettext().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext(message)
 
 
 def ngettext(singular, plural, number):
     return _trans.ngettext(singular, plural, number)
 
 
-# An alias since Django 2.0
-ungettext = ngettext
+def ungettext(singular, plural, number):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of ngettext() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ungettext() is deprecated in favor of '
+        'django.utils.translation.ngettext().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return ngettext(singular, plural, number)
 
 
 def pgettext(context, message):
@@ -99,10 +129,23 @@ def npgettext(context, singular, plural, number):
     return _trans.npgettext(context, singular, plural, number)
 
 
-gettext_lazy = ugettext_lazy = lazy(gettext, str)
+gettext_lazy = lazy(gettext, str)
 pgettext_lazy = lazy(pgettext, str)
 
 
+def ugettext_lazy(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2. Has been
+    Alias of gettext_lazy since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext_lazy() is deprecated in favor of '
+        'django.utils.translation.gettext_lazy().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext_lazy(message)
+
+
 def lazy_number(func, resultclass, number=None, **kwargs):
     if isinstance(number, int):
         kwargs['number'] = number
@@ -158,8 +201,17 @@ def ngettext_lazy(singular, plural, number=None):
     return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)
 
 
-# An alias since Django 2.0
-ungettext_lazy = ngettext_lazy
+def ungettext_lazy(singular, plural, number=None):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    An alias of ungettext_lazy() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ungettext_lazy() is deprecated in favor of '
+        'django.utils.translation.ngettext_lazy().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return ngettext_lazy(singular, plural, number)
 
 
 def npgettext_lazy(context, singular, plural, number=None):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/translation/__init__.py | 5 | 5 | 9 | 1 | 2442
| django/utils/translation/__init__.py | 75 | 91 | 1 | 1 | 180
| django/utils/translation/__init__.py | 102 | 102 | 1 | 1 | 180
| django/utils/translation/__init__.py | 161 | 162 | 6 | 1 | 1355


## Problem Statement

```
Deprecate ugettext(), ugettext_lazy(), ugettext_noop(), ungettext(), and ungettext_lazy()
Description
	
Along the lines of #27753 (Cleanups when no supported version of Django supports Python 2 anymore), the legacy functions in django.utils.translation -- ugettext(), ugettext_lazy(), ugettext_noop(), ungettext(), and ungettext_lazy() -- are simple aliases that remain for Python 2 Unicode backwards compatibility. As other compatibility layers have been cleaned up, these shims can be deprecated for removal.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/utils/translation/__init__.py** | 65 | 103| 180 | 180 | 1877 | 
| 2 | 2 django/utils/translation/trans_null.py | 1 | 68| 269 | 449 | 2146 | 
| 3 | 3 django/utils/http.py | 75 | 100| 195 | 644 | 6099 | 
| 4 | 4 django/utils/deprecation.py | 1 | 30| 195 | 839 | 6791 | 
| 5 | 4 django/utils/deprecation.py | 76 | 98| 158 | 997 | 6791 | 
| **-> 6 <-** | **4 django/utils/translation/__init__.py** | 153 | 216| 358 | 1355 | 6791 | 
| 7 | 4 django/utils/deprecation.py | 33 | 73| 336 | 1691 | 6791 | 
| 8 | 5 django/utils/translation/trans_real.py | 1 | 56| 486 | 2177 | 10648 | 
| **-> 9 <-** | **5 django/utils/translation/__init__.py** | 1 | 34| 265 | 2442 | 10648 | 
| 10 | 5 django/utils/http.py | 1 | 72| 693 | 3135 | 10648 | 
| 11 | 5 django/utils/translation/trans_real.py | 299 | 347| 354 | 3489 | 10648 | 
| 12 | 6 django/contrib/gis/views.py | 1 | 21| 155 | 3644 | 10803 | 
| 13 | 6 django/utils/translation/trans_real.py | 153 | 166| 154 | 3798 | 10803 | 
| 14 | **6 django/utils/translation/__init__.py** | 35 | 50| 154 | 3952 | 10803 | 
| 15 | 7 django/utils/formats.py | 1 | 57| 377 | 4329 | 12895 | 
| 16 | 8 django/views/i18n.py | 72 | 175| 711 | 5040 | 15349 | 
| 17 | 9 django/utils/encoding.py | 102 | 115| 130 | 5170 | 17645 | 
| 18 | **9 django/utils/translation/__init__.py** | 106 | 150| 335 | 5505 | 17645 | 
| 19 | 10 django/utils/text.py | 385 | 398| 153 | 5658 | 21018 | 
| 20 | 11 django/conf/global_settings.py | 51 | 144| 1087 | 6745 | 26610 | 
| 21 | 12 django/utils/timezone.py | 1 | 28| 154 | 6899 | 28488 | 
| 22 | 13 django/template/defaultfilters.py | 31 | 53| 191 | 7090 | 34541 | 
| 23 | 14 django/core/management/commands/makemessages.py | 117 | 152| 269 | 7359 | 40101 | 
| 24 | 14 django/utils/text.py | 364 | 382| 168 | 7527 | 40101 | 
| 25 | 14 django/conf/global_settings.py | 1 | 50| 366 | 7893 | 40101 | 
| 26 | 14 django/utils/encoding.py | 48 | 67| 156 | 8049 | 40101 | 
| 27 | 14 django/template/defaultfilters.py | 324 | 408| 499 | 8548 | 40101 | 
| 28 | 14 django/core/management/commands/makemessages.py | 394 | 416| 200 | 8748 | 40101 | 
| 29 | 15 django/db/models/fields/__init__.py | 348 | 374| 199 | 8947 | 56940 | 
| 30 | 16 django/utils/functional.py | 332 | 402| 513 | 9460 | 59906 | 
| 31 | 17 django/utils/dates.py | 1 | 50| 679 | 10139 | 60585 | 
| 32 | 18 scripts/manage_translations.py | 1 | 29| 200 | 10339 | 62278 | 
| 33 | **18 django/utils/translation/__init__.py** | 52 | 62| 127 | 10466 | 62278 | 
| 34 | 19 django/utils/html.py | 318 | 361| 442 | 10908 | 65527 | 
| 35 | **19 django/utils/translation/__init__.py** | 256 | 283| 197 | 11105 | 65527 | 
| 36 | 20 django/contrib/humanize/templatetags/humanize.py | 82 | 128| 578 | 11683 | 68668 | 
| 37 | 20 django/template/defaultfilters.py | 1 | 28| 207 | 11890 | 68668 | 
| 38 | 20 django/utils/formats.py | 141 | 162| 205 | 12095 | 68668 | 
| 39 | 20 django/utils/functional.py | 178 | 226| 299 | 12394 | 68668 | 
| 40 | 20 django/utils/html.py | 262 | 269| 140 | 12534 | 68668 | 
| 41 | 21 django/core/checks/model_checks.py | 134 | 167| 332 | 12866 | 70070 | 
| 42 | 21 django/utils/html.py | 271 | 301| 321 | 13187 | 70070 | 
| 43 | 22 django/utils/translation/template.py | 1 | 32| 343 | 13530 | 71994 | 
| 44 | 22 django/conf/global_settings.py | 145 | 260| 856 | 14386 | 71994 | 
| 45 | 22 django/template/defaultfilters.py | 190 | 217| 155 | 14541 | 71994 | 
| 46 | 23 django/core/management/base.py | 64 | 88| 161 | 14702 | 76358 | 
| 47 | 23 django/utils/translation/trans_real.py | 110 | 130| 186 | 14888 | 76358 | 
| 48 | 24 django/urls/__init__.py | 1 | 24| 239 | 15127 | 76597 | 
| 49 | 25 django/utils/version.py | 1 | 15| 129 | 15256 | 77391 | 
| 50 | 25 django/utils/encoding.py | 1 | 45| 290 | 15546 | 77391 | 
| 51 | 25 django/core/management/commands/makemessages.py | 97 | 115| 154 | 15700 | 77391 | 
| 52 | 26 django/contrib/admin/widgets.py | 344 | 370| 328 | 16028 | 81189 | 
| 53 | 26 django/core/management/commands/makemessages.py | 497 | 590| 744 | 16772 | 81189 | 
| 54 | 27 docs/_ext/djangodocs.py | 108 | 169| 524 | 17296 | 84262 | 
| 55 | 28 django/db/models/functions/__init__.py | 1 | 43| 619 | 17915 | 84881 | 
| 56 | 28 django/core/checks/model_checks.py | 85 | 109| 268 | 18183 | 84881 | 
| 57 | 28 django/core/management/commands/makemessages.py | 283 | 362| 816 | 18999 | 84881 | 
| 58 | 28 django/utils/html.py | 238 | 260| 254 | 19253 | 84881 | 
| 59 | 29 django/contrib/admin/utils.py | 285 | 303| 175 | 19428 | 88749 | 
| 60 | 30 django/db/models/functions/datetime.py | 241 | 309| 412 | 19840 | 91150 | 
| 61 | 30 django/contrib/humanize/templatetags/humanize.py | 218 | 261| 731 | 20571 | 91150 | 
| 62 | 30 django/utils/html.py | 1 | 78| 648 | 21219 | 91150 | 
| 63 | 31 django/contrib/postgres/utils.py | 1 | 30| 218 | 21437 | 91368 | 
| 64 | 32 django/db/models/query_utils.py | 154 | 218| 492 | 21929 | 94000 | 
| 65 | 33 django/utils/jslex.py | 185 | 221| 282 | 22211 | 95787 | 
| 66 | 33 django/core/management/commands/makemessages.py | 1 | 33| 247 | 22458 | 95787 | 
| 67 | 33 django/utils/translation/template.py | 35 | 59| 165 | 22623 | 95787 | 
| 68 | 34 django/utils/datetime_safe.py | 1 | 70| 451 | 23074 | 96553 | 
| 69 | 35 django/db/models/__init__.py | 1 | 49| 548 | 23622 | 97101 | 
| 70 | 35 django/utils/translation/trans_real.py | 191 | 270| 458 | 24080 | 97101 | 
| 71 | 36 django/utils/itercompat.py | 1 | 9| 0 | 24080 | 97141 | 
| 72 | 36 django/utils/html.py | 182 | 200| 159 | 24239 | 97141 | 
| 73 | 37 django/core/exceptions.py | 1 | 91| 381 | 24620 | 98146 | 
| 74 | 38 django/contrib/messages/utils.py | 1 | 13| 0 | 24620 | 98196 | 
| 75 | 39 django/core/paginator.py | 1 | 53| 305 | 24925 | 99511 | 
| 76 | **39 django/utils/translation/__init__.py** | 219 | 253| 262 | 25187 | 99511 | 
| 77 | 39 django/utils/functional.py | 113 | 175| 484 | 25671 | 99511 | 
| 78 | 39 django/contrib/humanize/templatetags/humanize.py | 263 | 301| 370 | 26041 | 99511 | 
| 79 | 39 django/utils/text.py | 56 | 76| 188 | 26229 | 99511 | 
| 80 | 40 django/db/utils.py | 1 | 48| 150 | 26379 | 101529 | 
| 81 | 41 django/db/migrations/serializer.py | 1 | 68| 407 | 26786 | 104065 | 
| 82 | 41 django/conf/global_settings.py | 490 | 635| 869 | 27655 | 104065 | 
| 83 | 42 django/contrib/auth/__init__.py | 1 | 59| 402 | 28057 | 105679 | 
| 84 | 43 django/db/models/options.py | 1 | 36| 304 | 28361 | 112545 | 
| 85 | 44 django/contrib/postgres/apps.py | 20 | 37| 188 | 28549 | 113111 | 
| 86 | 45 django/db/models/base.py | 1 | 44| 281 | 28830 | 127792 | 
| 87 | 45 django/db/models/functions/datetime.py | 190 | 217| 424 | 29254 | 127792 | 
| 88 | 46 django/template/backends/dummy.py | 1 | 54| 330 | 29584 | 128122 | 
| 89 | 47 django/utils/safestring.py | 40 | 64| 159 | 29743 | 128509 | 
| 90 | 48 django/contrib/admindocs/utils.py | 26 | 38| 144 | 29887 | 130411 | 
| 91 | 49 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 30104 | 130628 | 
| 92 | 50 django/views/debug.py | 193 | 241| 462 | 30566 | 134826 | 
| 93 | 50 django/db/models/functions/datetime.py | 219 | 238| 164 | 30730 | 134826 | 
| 94 | 51 django/utils/timesince.py | 1 | 24| 220 | 30950 | 135680 | 
| 95 | 51 django/template/defaultfilters.py | 804 | 848| 378 | 31328 | 135680 | 
| 96 | 51 django/db/models/functions/datetime.py | 169 | 188| 204 | 31532 | 135680 | 
| 97 | 52 django/core/checks/translation.py | 1 | 20| 0 | 31532 | 135783 | 
| 98 | 53 django/middleware/locale.py | 28 | 62| 331 | 31863 | 136350 | 
| 99 | 53 django/utils/functional.py | 93 | 111| 208 | 32071 | 136350 | 
| 100 | 53 django/utils/text.py | 101 | 119| 165 | 32236 | 136350 | 
| 101 | 53 django/core/management/commands/makemessages.py | 197 | 214| 177 | 32413 | 136350 | 
| 102 | 54 django/conf/__init__.py | 115 | 127| 120 | 32533 | 138109 | 
| 103 | 55 django/db/backends/mysql/operations.py | 1 | 31| 236 | 32769 | 141152 | 
| 104 | 55 django/views/i18n.py | 1 | 20| 117 | 32886 | 141152 | 
| 105 | 55 django/utils/encoding.py | 70 | 99| 250 | 33136 | 141152 | 
| 106 | 56 django/db/backends/base/operations.py | 482 | 523| 284 | 33420 | 146538 | 
| 107 | 57 django/templatetags/l10n.py | 41 | 64| 190 | 33610 | 146980 | 
| 108 | 57 django/conf/__init__.py | 1 | 39| 240 | 33850 | 146980 | 
| 109 | 57 django/utils/text.py | 337 | 361| 168 | 34018 | 146980 | 
| 110 | 57 django/template/defaultfilters.py | 56 | 91| 203 | 34221 | 146980 | 
| 111 | 57 django/core/management/commands/makemessages.py | 363 | 392| 231 | 34452 | 146980 | 
| 112 | 57 django/contrib/admin/utils.py | 1 | 24| 218 | 34670 | 146980 | 
| 113 | 57 django/utils/translation/template.py | 61 | 228| 1436 | 36106 | 146980 | 
| 114 | 57 django/template/defaultfilters.py | 423 | 458| 233 | 36339 | 146980 | 
| 115 | 57 django/utils/html.py | 364 | 391| 212 | 36551 | 146980 | 
| 116 | 57 django/template/defaultfilters.py | 307 | 321| 111 | 36662 | 146980 | 
| 117 | 57 django/utils/text.py | 232 | 259| 232 | 36894 | 146980 | 
| 118 | 57 django/template/defaultfilters.py | 851 | 908| 527 | 37421 | 146980 | 
| 119 | 57 django/utils/translation/trans_real.py | 132 | 151| 198 | 37619 | 146980 | 
| 120 | 58 django/utils/dateparse.py | 1 | 65| 731 | 38350 | 148428 | 
| 121 | 59 django/contrib/gis/utils/__init__.py | 1 | 15| 139 | 38489 | 148568 | 
| 122 | 60 django/contrib/gis/db/backends/mysql/operations.py | 50 | 69| 213 | 38702 | 149412 | 
| 123 | 61 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 39118 | 149873 | 
| 124 | 61 django/core/management/commands/makemessages.py | 155 | 167| 111 | 39229 | 149873 | 
| 125 | 62 django/core/checks/messages.py | 53 | 76| 161 | 39390 | 150446 | 
| 126 | 62 django/utils/text.py | 78 | 99| 202 | 39592 | 150446 | 
| 127 | 62 django/utils/translation/trans_real.py | 273 | 296| 180 | 39772 | 150446 | 
| 128 | 63 django/db/models/functions/window.py | 52 | 79| 182 | 39954 | 151089 | 
| 129 | 64 django/urls/converters.py | 1 | 67| 313 | 40267 | 151402 | 
| 130 | 65 django/forms/utils.py | 149 | 179| 229 | 40496 | 152639 | 
| 131 | 66 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 40625 | 156534 | 
| 132 | 67 django/db/backends/base/base.py | 1 | 21| 125 | 40750 | 161241 | 
| 133 | 67 django/core/management/commands/makemessages.py | 216 | 281| 633 | 41383 | 161241 | 
| 134 | 67 django/utils/text.py | 401 | 417| 104 | 41487 | 161241 | 
| 135 | 68 django/utils/duration.py | 1 | 45| 304 | 41791 | 161546 | 
| 136 | 69 django/conf/locale/id/formats.py | 5 | 50| 708 | 42499 | 162299 | 
| 137 | 70 django/contrib/auth/forms.py | 1 | 20| 137 | 42636 | 165234 | 
| 138 | 71 django/conf/locale/uk/formats.py | 5 | 38| 460 | 43096 | 165739 | 
| 139 | 72 django/conf/locale/hu/formats.py | 5 | 32| 323 | 43419 | 166107 | 
| 140 | 72 django/utils/encoding.py | 150 | 165| 182 | 43601 | 166107 | 
| 141 | 73 django/conf/locale/ml/formats.py | 5 | 41| 663 | 44264 | 166815 | 
| 142 | 74 django/template/base.py | 1 | 91| 725 | 44989 | 174681 | 
| 143 | 75 django/core/validators.py | 1 | 25| 190 | 45179 | 179001 | 
| 144 | 75 django/template/defaultfilters.py | 239 | 304| 427 | 45606 | 179001 | 
| 145 | 76 django/db/migrations/autodetector.py | 884 | 903| 184 | 45790 | 190671 | 
| 146 | 77 django/conf/locale/cs/formats.py | 5 | 43| 640 | 46430 | 191356 | 
| 147 | 77 django/utils/translation/trans_real.py | 59 | 108| 450 | 46880 | 191356 | 
| 148 | 77 django/db/models/functions/datetime.py | 1 | 28| 236 | 47116 | 191356 | 
| 149 | 78 django/conf/locale/bg/formats.py | 5 | 22| 131 | 47247 | 191531 | 
| 150 | 79 django/template/__init__.py | 1 | 69| 360 | 47607 | 191891 | 
| 151 | 80 django/utils/regex_helper.py | 1 | 34| 238 | 47845 | 194424 | 
| 152 | 81 django/conf/locale/ru/formats.py | 5 | 33| 402 | 48247 | 194871 | 
| 153 | 82 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 48536 | 195176 | 
| 154 | 83 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 49040 | 196442 | 
| 155 | 83 django/core/management/commands/makemessages.py | 36 | 57| 143 | 49183 | 196442 | 
| 156 | 84 django/contrib/humanize/__init__.py | 1 | 2| 0 | 49183 | 196458 | 
| 157 | 84 django/utils/functional.py | 229 | 329| 905 | 50088 | 196458 | 
| 158 | 84 django/core/checks/model_checks.py | 68 | 83| 176 | 50264 | 196458 | 
| 159 | 85 django/conf/locale/eo/formats.py | 5 | 50| 742 | 51006 | 197245 | 
| 160 | 86 django/conf/locale/bs/formats.py | 5 | 22| 139 | 51145 | 197428 | 
| 161 | 87 django/conf/locale/de/formats.py | 5 | 29| 323 | 51468 | 197796 | 
| 162 | 87 django/core/checks/model_checks.py | 111 | 132| 263 | 51731 | 197796 | 
| 163 | 88 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 51868 | 197933 | 
| 164 | 88 django/utils/translation/trans_real.py | 350 | 361| 110 | 51978 | 197933 | 
| 165 | 88 django/core/checks/model_checks.py | 45 | 66| 168 | 52146 | 197933 | 
| 166 | 89 django/contrib/gis/geos/libgeos.py | 134 | 175| 293 | 52439 | 199217 | 


## Patch

```diff
diff --git a/django/utils/translation/__init__.py b/django/utils/translation/__init__.py
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -2,9 +2,11 @@
 Internationalization support.
 """
 import re
+import warnings
 from contextlib import ContextDecorator
 
 from django.utils.autoreload import autoreload_started, file_changed
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.functional import lazy
 
 __all__ = [
@@ -72,23 +74,51 @@ def gettext_noop(message):
     return _trans.gettext_noop(message)
 
 
-ugettext_noop = gettext_noop
+def ugettext_noop(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of gettext_noop() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext_noop() is deprecated in favor of '
+        'django.utils.translation.gettext_noop().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext_noop(message)
 
 
 def gettext(message):
     return _trans.gettext(message)
 
 
-# An alias since Django 2.0
-ugettext = gettext
+def ugettext(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of gettext() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext() is deprecated in favor of '
+        'django.utils.translation.gettext().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext(message)
 
 
 def ngettext(singular, plural, number):
     return _trans.ngettext(singular, plural, number)
 
 
-# An alias since Django 2.0
-ungettext = ngettext
+def ungettext(singular, plural, number):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    Alias of ngettext() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ungettext() is deprecated in favor of '
+        'django.utils.translation.ngettext().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return ngettext(singular, plural, number)
 
 
 def pgettext(context, message):
@@ -99,10 +129,23 @@ def npgettext(context, singular, plural, number):
     return _trans.npgettext(context, singular, plural, number)
 
 
-gettext_lazy = ugettext_lazy = lazy(gettext, str)
+gettext_lazy = lazy(gettext, str)
 pgettext_lazy = lazy(pgettext, str)
 
 
+def ugettext_lazy(message):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2. Has been
+    Alias of gettext_lazy since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ugettext_lazy() is deprecated in favor of '
+        'django.utils.translation.gettext_lazy().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return gettext_lazy(message)
+
+
 def lazy_number(func, resultclass, number=None, **kwargs):
     if isinstance(number, int):
         kwargs['number'] = number
@@ -158,8 +201,17 @@ def ngettext_lazy(singular, plural, number=None):
     return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)
 
 
-# An alias since Django 2.0
-ungettext_lazy = ngettext_lazy
+def ungettext_lazy(singular, plural, number=None):
+    """
+    A legacy compatibility wrapper for Unicode handling on Python 2.
+    An alias of ungettext_lazy() since Django 2.0.
+    """
+    warnings.warn(
+        'django.utils.translation.ungettext_lazy() is deprecated in favor of '
+        'django.utils.translation.ngettext_lazy().',
+        RemovedInDjango40Warning, stacklevel=2,
+    )
+    return ngettext_lazy(singular, plural, number)
 
 
 def npgettext_lazy(context, singular, plural, number=None):

```

## Test Patch

```diff
diff --git a/tests/i18n/tests.py b/tests/i18n/tests.py
--- a/tests/i18n/tests.py
+++ b/tests/i18n/tests.py
@@ -23,6 +23,7 @@
     RequestFactory, SimpleTestCase, TestCase, override_settings,
 )
 from django.utils import translation
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.formats import (
     date_format, get_format, get_format_modules, iter_format_modules, localize,
     localize_input, reset_format_cache, sanitize_separators, time_format,
@@ -34,7 +35,8 @@
     get_language, get_language_bidi, get_language_from_request,
     get_language_info, gettext, gettext_lazy, ngettext, ngettext_lazy,
     npgettext, npgettext_lazy, pgettext, to_language, to_locale, trans_null,
-    trans_real, ugettext, ugettext_lazy, ungettext, ungettext_lazy,
+    trans_real, ugettext, ugettext_lazy, ugettext_noop, ungettext,
+    ungettext_lazy,
 )
 from django.utils.translation.reloader import (
     translation_file_changed, watch_for_translation_changes,
@@ -74,13 +76,39 @@ def test_legacy_aliases(self):
         """
         Pre-Django 2.0 aliases with u prefix are still available.
         """
-        self.assertEqual(ugettext("Image"), "Bild")
-        self.assertEqual(ugettext_lazy("Image"), gettext_lazy("Image"))
-        self.assertEqual(ungettext("%d year", "%d years", 0) % 0, "0 Jahre")
-        self.assertEqual(
-            ungettext_lazy("%d year", "%d years", 0) % 0,
-            ngettext_lazy("%d year", "%d years", 0) % 0,
+        msg = (
+            'django.utils.translation.ugettext_noop() is deprecated in favor '
+            'of django.utils.translation.gettext_noop().'
         )
+        with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+            self.assertEqual(ugettext_noop("Image"), "Image")
+        msg = (
+            'django.utils.translation.ugettext() is deprecated in favor of '
+            'django.utils.translation.gettext().'
+        )
+        with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+            self.assertEqual(ugettext("Image"), "Bild")
+        msg = (
+            'django.utils.translation.ugettext_lazy() is deprecated in favor '
+            'of django.utils.translation.gettext_lazy().'
+        )
+        with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+            self.assertEqual(ugettext_lazy("Image"), gettext_lazy("Image"))
+        msg = (
+            'django.utils.translation.ungettext() is deprecated in favor of '
+            'django.utils.translation.ngettext().'
+        )
+        with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+            self.assertEqual(ungettext("%d year", "%d years", 0) % 0, "0 Jahre")
+        msg = (
+            'django.utils.translation.ungettext_lazy() is deprecated in favor '
+            'of django.utils.translation.ngettext_lazy().'
+        )
+        with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+            self.assertEqual(
+                ungettext_lazy("%d year", "%d years", 0) % 0,
+                ngettext_lazy("%d year", "%d years", 0) % 0,
+            )
 
     @translation.override('fr')
     def test_plural(self):

```


## Code snippets

### 1 - django/utils/translation/__init__.py:

Start line: 65, End line: 103

```python
_trans = Trans()

# The Trans class is no more needed, so remove it from the namespace.
del Trans


def gettext_noop(message):
    return _trans.gettext_noop(message)


ugettext_noop = gettext_noop


def gettext(message):
    return _trans.gettext(message)


# An alias since Django 2.0
ugettext = gettext


def ngettext(singular, plural, number):
    return _trans.ngettext(singular, plural, number)


# An alias since Django 2.0
ungettext = ngettext


def pgettext(context, message):
    return _trans.pgettext(context, message)


def npgettext(context, singular, plural, number):
    return _trans.npgettext(context, singular, plural, number)


gettext_lazy = ugettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)
```
### 2 - django/utils/translation/trans_null.py:

Start line: 1, End line: 68

```python
# These are versions of the functions in django.utils.translation.trans_real
# that don't actually do anything. This is purely for performance, so that
# settings.USE_I18N = False can use this module rather than trans_real.py.

from django.conf import settings


def gettext(message):
    return message


gettext_noop = gettext_lazy = _ = gettext


def ngettext(singular, plural, number):
    if number == 1:
        return singular
    return plural


ngettext_lazy = ngettext


def pgettext(context, message):
    return gettext(message)


def npgettext(context, singular, plural, number):
    return ngettext(singular, plural, number)


def activate(x):
    return None


def deactivate():
    return None


deactivate_all = deactivate


def get_language():
    return settings.LANGUAGE_CODE


def get_language_bidi():
    return settings.LANGUAGE_CODE in settings.LANGUAGES_BIDI


def check_for_language(x):
    return True


def get_language_from_request(request, check_path=False):
    return settings.LANGUAGE_CODE


def get_language_from_path(request):
    return None


def get_supported_language_variant(lang_code, strict=False):
    if lang_code == settings.LANGUAGE_CODE:
        return lang_code
    else:
        raise LookupError(lang_code)
```
### 3 - django/utils/http.py:

Start line: 75, End line: 100

```python
@keep_lazy_text
def urlunquote(quoted_url):
    """
    A legacy compatibility wrapper to Python's urllib.parse.unquote() function.
    (was used for unicode handling on Python 2)
    """
    warnings.warn(
        'django.utils.http.urlunquote() is deprecated in favor of '
        'urllib.parse.unquote().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return unquote(quoted_url)


@keep_lazy_text
def urlunquote_plus(quoted_url):
    """
    A legacy compatibility wrapper to Python's urllib.parse.unquote_plus()
    function. (was used for unicode handling on Python 2)
    """
    warnings.warn(
        'django.utils.http.urlunquote_plus() is deprecated in favor of '
        'urllib.parse.unquote_plus().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return unquote_plus(quoted_url)
```
### 4 - django/utils/deprecation.py:

Start line: 1, End line: 30

```python
import inspect
import warnings


class RemovedInDjango31Warning(DeprecationWarning):
    pass


class RemovedInDjango40Warning(PendingDeprecationWarning):
    pass


RemovedInNextVersionWarning = RemovedInDjango31Warning


class warn_about_renamed_method:
    def __init__(self, class_name, old_method_name, new_method_name, deprecation_warning):
        self.class_name = class_name
        self.old_method_name = old_method_name
        self.new_method_name = new_method_name
        self.deprecation_warning = deprecation_warning

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            warnings.warn(
                "`%s.%s` is deprecated, use `%s` instead." %
                (self.class_name, self.old_method_name, self.new_method_name),
                self.deprecation_warning, 2)
            return f(*args, **kwargs)
        return wrapped
```
### 5 - django/utils/deprecation.py:

Start line: 76, End line: 98

```python
class DeprecationInstanceCheck(type):
    def __instancecheck__(self, instance):
        warnings.warn(
            "`%s` is deprecated, use `%s` instead." % (self.__name__, self.alternative),
            self.deprecation_warning, 2
        )
        return super().__instancecheck__(instance)


class MiddlewareMixin:
    def __init__(self, get_response=None):
        self.get_response = get_response
        super().__init__()

    def __call__(self, request):
        response = None
        if hasattr(self, 'process_request'):
            response = self.process_request(request)
        response = response or self.get_response(request)
        if hasattr(self, 'process_response'):
            response = self.process_response(request, response)
        return response
```
### 6 - django/utils/translation/__init__.py:

Start line: 153, End line: 216

```python
def _lazy_number_unpickle(func, resultclass, number, kwargs):
    return lazy_number(func, resultclass, number=number, **kwargs)


def ngettext_lazy(singular, plural, number=None):
    return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)


# An alias since Django 2.0
ungettext_lazy = ngettext_lazy


def npgettext_lazy(context, singular, plural, number=None):
    return lazy_number(npgettext, str, context=context, singular=singular, plural=plural, number=number)


def activate(language):
    return _trans.activate(language)


def deactivate():
    return _trans.deactivate()


class override(ContextDecorator):
    def __init__(self, language, deactivate=False):
        self.language = language
        self.deactivate = deactivate

    def __enter__(self):
        self.old_language = get_language()
        if self.language is not None:
            activate(self.language)
        else:
            deactivate_all()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_language is None:
            deactivate_all()
        elif self.deactivate:
            deactivate()
        else:
            activate(self.old_language)


def get_language():
    return _trans.get_language()


def get_language_bidi():
    return _trans.get_language_bidi()


def check_for_language(lang_code):
    return _trans.check_for_language(lang_code)


def to_language(locale):
    """Turn a locale name (en_US) into a language name (en-us)."""
    p = locale.find('_')
    if p >= 0:
        return locale[:p].lower() + '-' + locale[p + 1:].lower()
    else:
        return locale.lower()
```
### 7 - django/utils/deprecation.py:

Start line: 33, End line: 73

```python
class RenameMethodsBase(type):
    """
    Handles the deprecation paths when renaming a method.

    It does the following:
        1) Define the new method if missing and complain about it.
        2) Define the old method if missing.
        3) Complain whenever an old method is called.

    See #15363 for more details.
    """

    renamed_methods = ()

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        for base in inspect.getmro(new_class):
            class_name = base.__name__
            for renamed_method in cls.renamed_methods:
                old_method_name = renamed_method[0]
                old_method = base.__dict__.get(old_method_name)
                new_method_name = renamed_method[1]
                new_method = base.__dict__.get(new_method_name)
                deprecation_warning = renamed_method[2]
                wrapper = warn_about_renamed_method(class_name, *renamed_method)

                # Define the new method if missing and complain about it
                if not new_method and old_method:
                    warnings.warn(
                        "`%s.%s` method should be renamed `%s`." %
                        (class_name, old_method_name, new_method_name),
                        deprecation_warning, 2)
                    setattr(base, new_method_name, old_method)
                    setattr(base, old_method_name, wrapper(old_method))

                # Define the old method as a wrapped call to the new method.
                if not old_method and new_method:
                    setattr(base, old_method_name, wrapper(new_method))

        return new_class
```
### 8 - django/utils/translation/trans_real.py:

Start line: 1, End line: 56

```python
"""Translation helper functions."""
import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from threading import local

from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.safestring import SafeData, mark_safe

from . import LANGUAGE_SESSION_KEY, to_language, to_locale

# Translations are cached in a dictionary for every language.
# The active translations are stored by threadid to make them thread local.
_translations = {}
_active = local()

# The default translation is based on the settings file.
_default = None

# magic gettext number to separate context from message
CONTEXT_SEPARATOR = "\x04"

# Format of Accept-Language header values. From RFC 2616, section 14.4 and 3.9
# and RFC 3066, section 2.1
accept_language_re = re.compile(r'''
        ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\*)      # "en", "en-au", "x-y-z", "es-419", "*"
        (?:\s*;\s*q=(0(?:\.\d{,3})?|1(?:\.0{,3})?))?  # Optional "q=1.00", "q=0.8"
        (?:\s*,\s*|$)                                 # Multiple accepts per header.
        ''', re.VERBOSE)

language_code_re = re.compile(
    r'^[a-z]{1,8}(?:-[a-z0-9]{1,8})*(?:@[a-z0-9]{1,20})?$',
    re.IGNORECASE
)

language_code_prefix_re = re.compile(r'^/(\w+([@-]\w+)?)(/|$)')


@receiver(setting_changed)
def reset_cache(**kwargs):
    """
    Reset global state when LANGUAGES setting has been changed, as some
    languages should no longer be accepted.
    """
    if kwargs['setting'] in ('LANGUAGES', 'LANGUAGE_CODE'):
        check_for_language.cache_clear()
        get_languages.cache_clear()
        get_supported_language_variant.cache_clear()
```
### 9 - django/utils/translation/__init__.py:

Start line: 1, End line: 34

```python
"""
Internationalization support.
"""
import re
from contextlib import ContextDecorator

from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy

__all__ = [
    'activate', 'deactivate', 'override', 'deactivate_all',
    'get_language', 'get_language_from_request',
    'get_language_info', 'get_language_bidi',
    'check_for_language', 'to_language', 'to_locale', 'templatize',
    'gettext', 'gettext_lazy', 'gettext_noop',
    'ugettext', 'ugettext_lazy', 'ugettext_noop',
    'ngettext', 'ngettext_lazy',
    'ungettext', 'ungettext_lazy',
    'pgettext', 'pgettext_lazy',
    'npgettext', 'npgettext_lazy',
    'LANGUAGE_SESSION_KEY',
]

LANGUAGE_SESSION_KEY = '_language'


class TranslatorCommentWarning(SyntaxWarning):
    pass


# Here be dragons, so a short explanation of the logic won't hurt:
# We are trying to solve two problems: (1) access settings, in particular
# settings.USE_I18N, as late as possible, so that modules can be imported
# without having to first configure Django, and (2) if some other code creates
```
### 10 - django/utils/http.py:

Start line: 1, End line: 72

```python
import base64
import calendar
import datetime
import re
import unicodedata
import warnings
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import (
    ParseResult, SplitResult, _coerce_args, _splitnetloc, _splitparams, quote,
    quote_plus, scheme_chars, unquote, unquote_plus,
    urlencode as original_urlencode, uses_params,
)

from django.core.exceptions import TooManyFieldsSent
from django.utils.datastructures import MultiValueDict
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import keep_lazy_text

# based on RFC 7232, Appendix C
ETAG_MATCH = re.compile(r'''
    \A(      # start of string and capture group
    (?:W/)?  # optional weak indicator
    "        # opening quote
    [^"]*    # any sequence of non-quote characters
    "        # end quote
    )\Z      # end of string and capture group
''', re.X)

MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()
__D = r'(?P<day>\d{2})'
__D2 = r'(?P<day>[ \d]\d)'
__M = r'(?P<mon>\w{3})'
__Y = r'(?P<year>\d{4})'
__Y2 = r'(?P<year>\d{2})'
__T = r'(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'
RFC1123_DATE = re.compile(r'^\w{3}, %s %s %s %s GMT$' % (__D, __M, __Y, __T))
RFC850_DATE = re.compile(r'^\w{6,9}, %s-%s-%s %s GMT$' % (__D, __M, __Y2, __T))
ASCTIME_DATE = re.compile(r'^\w{3} %s %s %s %s$' % (__M, __D2, __T, __Y))

RFC3986_GENDELIMS = ":/?#[]@"
RFC3986_SUBDELIMS = "!$&'()*+,;="

FIELDS_MATCH = re.compile('[&;]')


@keep_lazy_text
def urlquote(url, safe='/'):
    """
    A legacy compatibility wrapper to Python's urllib.parse.quote() function.
    (was used for unicode handling on Python 2)
    """
    warnings.warn(
        'django.utils.http.urlquote() is deprecated in favor of '
        'urllib.parse.quote().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return quote(url, safe)


@keep_lazy_text
def urlquote_plus(url, safe=''):
    """
    A legacy compatibility wrapper to Python's urllib.parse.quote_plus()
    function. (was used for unicode handling on Python 2)
    """
    warnings.warn(
        'django.utils.http.urlquote_plus() is deprecated in favor of '
        'urllib.parse.quote_plus(),',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return quote_plus(url, safe)
```
### 14 - django/utils/translation/__init__.py:

Start line: 35, End line: 50

```python
# a reference to one of these functions, don't break that reference when we
# replace the functions with their real counterparts (once we do access the
# settings).

class Trans:
    """
    The purpose of this class is to store the actual translation function upon
    receiving the first call to that function. After this is done, changes to
    USE_I18N will have no effect to which function is served upon request. If
    your tests rely on changing USE_I18N, you can delete all the functions
    from _trans.__dict__.

    Note that storing the function with setattr will have a noticeable
    performance effect, as access to the function goes the normal path,
    instead of using __getattr__.
    """
```
### 18 - django/utils/translation/__init__.py:

Start line: 106, End line: 150

```python
def lazy_number(func, resultclass, number=None, **kwargs):
    if isinstance(number, int):
        kwargs['number'] = number
        proxy = lazy(func, resultclass)(**kwargs)
    else:
        original_kwargs = kwargs.copy()

        class NumberAwareString(resultclass):
            def __bool__(self):
                return bool(kwargs['singular'])

            def _get_number_value(self, values):
                try:
                    return values[number]
                except KeyError:
                    raise KeyError(
                        "Your dictionary lacks key '%s\'. Please provide "
                        "it, because it is required to determine whether "
                        "string is singular or plural." % number
                    )

            def _translate(self, number_value):
                kwargs['number'] = number_value
                return func(**kwargs)

            def format(self, *args, **kwargs):
                number_value = self._get_number_value(kwargs) if kwargs and number else args[0]
                return self._translate(number_value).format(*args, **kwargs)

            def __mod__(self, rhs):
                if isinstance(rhs, dict) and number:
                    number_value = self._get_number_value(rhs)
                else:
                    number_value = rhs
                translated = self._translate(number_value)
                try:
                    translated = translated % rhs
                except TypeError:
                    # String doesn't contain a placeholder for the number.
                    pass
                return translated

        proxy = lazy(lambda **kwargs: NumberAwareString(), NumberAwareString)(**kwargs)
        proxy.__reduce__ = lambda: (_lazy_number_unpickle, (func, resultclass, number, original_kwargs))
    return proxy
```
### 33 - django/utils/translation/__init__.py:

Start line: 52, End line: 62

```python
class Trans:

    def __getattr__(self, real_name):
        from django.conf import settings
        if settings.USE_I18N:
            from django.utils.translation import trans_real as trans
            from django.utils.translation.reloader import watch_for_translation_changes, translation_file_changed
            autoreload_started.connect(watch_for_translation_changes, dispatch_uid='translation_file_changed')
            file_changed.connect(translation_file_changed, dispatch_uid='translation_file_changed')
        else:
            from django.utils.translation import trans_null as trans
        setattr(self, real_name, getattr(trans, real_name))
        return getattr(trans, real_name)
```
### 35 - django/utils/translation/__init__.py:

Start line: 256, End line: 283

```python
def get_language_info(lang_code):
    from django.conf.locale import LANG_INFO
    try:
        lang_info = LANG_INFO[lang_code]
        if 'fallback' in lang_info and 'name' not in lang_info:
            info = get_language_info(lang_info['fallback'][0])
        else:
            info = lang_info
    except KeyError:
        if '-' not in lang_code:
            raise KeyError("Unknown language code %s." % lang_code)
        generic_lang_code = lang_code.split('-')[0]
        try:
            info = LANG_INFO[generic_lang_code]
        except KeyError:
            raise KeyError("Unknown language code %s and %s." % (lang_code, generic_lang_code))

    if info:
        info['name_translated'] = gettext_lazy(info['name'])
    return info


trim_whitespace_re = re.compile(r'\s*\n\s*')


def trim_whitespace(s):
    return trim_whitespace_re.sub(' ', s.strip())
```
### 76 - django/utils/translation/__init__.py:

Start line: 219, End line: 253

```python
def to_locale(language):
    """Turn a language name (en-us) into a locale name (en_US)."""
    language, _, country = language.lower().partition('-')
    if not country:
        return language
    # A language with > 2 characters after the dash only has its first
    # character after the dash capitalized; e.g. sr-latn becomes sr_Latn.
    # A language with 2 characters after the dash has both characters
    # capitalized; e.g. en-us becomes en_US.
    country, _, tail = country.partition('-')
    country = country.title() if len(country) > 2 else country.upper()
    if tail:
        country += '-' + tail
    return language + '_' + country


def get_language_from_request(request, check_path=False):
    return _trans.get_language_from_request(request, check_path)


def get_language_from_path(path):
    return _trans.get_language_from_path(path)


def get_supported_language_variant(lang_code, *, strict=False):
    return _trans.get_supported_language_variant(lang_code, strict)


def templatize(src, **kwargs):
    from .template import templatize
    return templatize(src, **kwargs)


def deactivate_all():
    return _trans.deactivate_all()
```
