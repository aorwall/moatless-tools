# django__django-11244

| **django/django** | `0c916255eb4d94e06e123fafec93efdba45b1259` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 508 |
| **Any found context length** | 508 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/checks/translation.py b/django/core/checks/translation.py
--- a/django/core/checks/translation.py
+++ b/django/core/checks/translation.py
@@ -24,12 +24,6 @@
     id='translation.E004',
 )
 
-E005 = Error(
-    'You have provided values in the LANGUAGES_BIDI setting that are not in '
-    'the LANGUAGES setting.',
-    id='translation.E005',
-)
-
 
 @register(Tags.translation)
 def check_setting_language_code(app_configs, **kwargs):
@@ -62,9 +56,6 @@ def check_setting_languages_bidi(app_configs, **kwargs):
 def check_language_settings_consistent(app_configs, **kwargs):
     """Error if language settings are not consistent with each other."""
     available_tags = {i for i, _ in settings.LANGUAGES} | {'en-us'}
-    messages = []
     if settings.LANGUAGE_CODE not in available_tags:
-        messages.append(E004)
-    if not available_tags.issuperset(settings.LANGUAGES_BIDI):
-        messages.append(E005)
-    return messages
+        return [E004]
+    return []

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/checks/translation.py | 27 | 32 | 1 | 1 | 508
| django/core/checks/translation.py | 65 | 70 | 1 | 1 | 508


## Problem Statement

```
Remove the LANGUAGES_BIDI<=LANGUAGES check.
Description
	
(Adding Nick Pope to Cc: as author of the commit referenced below)
Since ​https://github.com/django/django/commit/4400d8296d268f5a8523cd02ddc33b12219b2535 there is a system check which verifies that LANGUAGES_BIDI is a subset of LANGUAGES. This breaks almost all installations of Django using a custom LANGUAGES list which do not also override LANGUAGES_BIDI -- probably many installations.
All of them will either have to add a LANGUAGES_BIDI override or silence translation.E005 when updating. If this is intentional then this change has to be mentioned in the release notes and documented somewhere.
However, I don't really see the need to verify that LANGUAGES_BIDI is a subset of LANGUAGES and propose that the easiest and also the best way to solve this is to remove the translation.E005 check again.
Here's a test which currently fails but shouldn't in my opinion:
diff --git a/tests/check_framework/test_translation.py b/tests/check_framework/test_translation.py
index 9a34b65c06..cea844988d 100644
--- a/tests/check_framework/test_translation.py
+++ b/tests/check_framework/test_translation.py
@@ -92,3 +92,7 @@ class TranslationCheckTests(SimpleTestCase):
			 self.assertEqual(check_language_settings_consistent(None), [
				 Error(msg, id='translation.E005'),
			 ])
+
+	def test_languages_without_bidi(self):
+		with self.settings(LANGUAGE_CODE='en', LANGUAGES=[('en', 'English')]):
+			self.assertEqual(check_language_settings_consistent(None), [])
Remove the LANGUAGES_BIDI<=LANGUAGES check.
Description
	
(Adding Nick Pope to Cc: as author of the commit referenced below)
Since ​https://github.com/django/django/commit/4400d8296d268f5a8523cd02ddc33b12219b2535 there is a system check which verifies that LANGUAGES_BIDI is a subset of LANGUAGES. This breaks almost all installations of Django using a custom LANGUAGES list which do not also override LANGUAGES_BIDI -- probably many installations.
All of them will either have to add a LANGUAGES_BIDI override or silence translation.E005 when updating. If this is intentional then this change has to be mentioned in the release notes and documented somewhere.
However, I don't really see the need to verify that LANGUAGES_BIDI is a subset of LANGUAGES and propose that the easiest and also the best way to solve this is to remove the translation.E005 check again.
Here's a test which currently fails but shouldn't in my opinion:
diff --git a/tests/check_framework/test_translation.py b/tests/check_framework/test_translation.py
index 9a34b65c06..cea844988d 100644
--- a/tests/check_framework/test_translation.py
+++ b/tests/check_framework/test_translation.py
@@ -92,3 +92,7 @@ class TranslationCheckTests(SimpleTestCase):
			 self.assertEqual(check_language_settings_consistent(None), [
				 Error(msg, id='translation.E005'),
			 ])
+
+	def test_languages_without_bidi(self):
+		with self.settings(LANGUAGE_CODE='en', LANGUAGES=[('en', 'English')]):
+			self.assertEqual(check_language_settings_consistent(None), [])

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/checks/translation.py** | 1 | 71| 508 | 508 | 509 | 
| 2 | 2 django/utils/translation/__init__.py | 1 | 36| 281 | 789 | 2793 | 
| 3 | 3 django/utils/translation/trans_null.py | 1 | 68| 269 | 1058 | 3062 | 
| 4 | 4 django/conf/global_settings.py | 51 | 144| 1087 | 2145 | 8666 | 
| 5 | 5 django/utils/translation/trans_real.py | 364 | 389| 221 | 2366 | 12491 | 
| 6 | 5 django/utils/translation/trans_real.py | 1 | 56| 482 | 2848 | 12491 | 
| 7 | 5 django/utils/translation/__init__.py | 196 | 268| 440 | 3288 | 12491 | 
| 8 | 5 django/utils/translation/__init__.py | 67 | 146| 489 | 3777 | 12491 | 
| 9 | 6 django/views/i18n.py | 77 | 180| 711 | 4488 | 14998 | 
| 10 | 7 scripts/manage_translations.py | 83 | 103| 191 | 4679 | 16691 | 
| 11 | 8 django/contrib/auth/__init__.py | 1 | 58| 393 | 5072 | 18258 | 
| 12 | 8 django/utils/translation/__init__.py | 308 | 335| 197 | 5269 | 18258 | 
| 13 | 9 django/core/checks/messages.py | 53 | 76| 161 | 5430 | 18831 | 
| 14 | 9 django/core/checks/messages.py | 26 | 50| 259 | 5689 | 18831 | 
| 15 | 9 django/utils/translation/trans_real.py | 153 | 166| 154 | 5843 | 18831 | 
| 16 | 9 scripts/manage_translations.py | 1 | 29| 200 | 6043 | 18831 | 
| 17 | 9 django/utils/translation/trans_real.py | 191 | 270| 458 | 6501 | 18831 | 
| 18 | 9 scripts/manage_translations.py | 176 | 186| 116 | 6617 | 18831 | 
| 19 | 10 django/templatetags/i18n.py | 31 | 65| 222 | 6839 | 22750 | 
| 20 | 11 django/core/checks/templates.py | 1 | 36| 259 | 7098 | 23010 | 
| 21 | 12 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 7602 | 24276 | 
| 22 | 12 django/core/checks/messages.py | 1 | 24| 156 | 7758 | 24276 | 
| 23 | 13 django/core/checks/model_checks.py | 134 | 167| 332 | 8090 | 25678 | 
| 24 | 14 django/core/checks/__init__.py | 1 | 25| 254 | 8344 | 25932 | 
| 25 | 15 django/core/management/commands/makemessages.py | 283 | 362| 816 | 9160 | 31492 | 
| 26 | 16 django/contrib/admin/checks.py | 874 | 922| 416 | 9576 | 40453 | 
| 27 | 16 django/utils/translation/trans_real.py | 444 | 484| 289 | 9865 | 40453 | 
| 28 | 17 django/middleware/locale.py | 28 | 62| 331 | 10196 | 41020 | 
| 29 | 17 django/contrib/admin/checks.py | 57 | 121| 535 | 10731 | 41020 | 
| 30 | 18 django/core/checks/security/base.py | 1 | 86| 752 | 11483 | 42646 | 
| 31 | 18 django/contrib/admin/checks.py | 1082 | 1112| 188 | 11671 | 42646 | 
| 32 | 18 scripts/manage_translations.py | 106 | 136| 301 | 11972 | 42646 | 
| 33 | 18 django/contrib/admin/checks.py | 838 | 860| 217 | 12189 | 42646 | 
| 34 | 18 django/core/checks/security/base.py | 88 | 190| 747 | 12936 | 42646 | 
| 35 | 18 django/conf/global_settings.py | 145 | 263| 876 | 13812 | 42646 | 
| 36 | 18 django/utils/translation/trans_real.py | 132 | 151| 198 | 14010 | 42646 | 
| 37 | 19 django/contrib/auth/checks.py | 97 | 167| 525 | 14535 | 43819 | 
| 38 | 20 docs/_ext/djangodocs.py | 108 | 169| 524 | 15059 | 46892 | 
| 39 | 20 django/contrib/admin/checks.py | 1008 | 1035| 204 | 15263 | 46892 | 
| 40 | 21 django/db/models/constraints.py | 30 | 66| 309 | 15572 | 47891 | 
| 41 | 22 django/templatetags/l10n.py | 41 | 64| 190 | 15762 | 48333 | 
| 42 | 22 django/utils/translation/trans_real.py | 168 | 188| 168 | 15930 | 48333 | 
| 43 | 22 django/contrib/admin/checks.py | 615 | 634| 183 | 16113 | 48333 | 
| 44 | 23 django/contrib/gis/geos/prototypes/errcheck.py | 1 | 51| 373 | 16486 | 48948 | 
| 45 | 23 django/contrib/admin/checks.py | 1037 | 1079| 343 | 16829 | 48948 | 
| 46 | 23 django/utils/translation/__init__.py | 54 | 64| 127 | 16956 | 48948 | 
| 47 | 23 django/core/management/commands/makemessages.py | 1 | 33| 247 | 17203 | 48948 | 
| 48 | 24 django/db/models/fields/__init__.py | 1114 | 1143| 218 | 17421 | 65947 | 
| 49 | 25 django/utils/translation/template.py | 61 | 228| 1436 | 18857 | 67871 | 
| 50 | 25 django/contrib/admin/checks.py | 348 | 364| 134 | 18991 | 67871 | 
| 51 | 25 django/contrib/admin/checks.py | 763 | 784| 190 | 19181 | 67871 | 
| 52 | 26 django/views/debug.py | 193 | 241| 462 | 19643 | 72089 | 
| 53 | 26 django/core/checks/model_checks.py | 85 | 109| 268 | 19911 | 72089 | 
| 54 | 26 django/contrib/gis/geos/prototypes/errcheck.py | 54 | 84| 241 | 20152 | 72089 | 
| 55 | 26 django/contrib/admin/checks.py | 746 | 761| 182 | 20334 | 72089 | 
| 56 | 27 django/contrib/gis/gdal/prototypes/errcheck.py | 66 | 140| 541 | 20875 | 73077 | 
| 57 | 28 django/db/backends/mysql/validation.py | 1 | 27| 248 | 21123 | 73559 | 
| 58 | 29 django/core/management/base.py | 379 | 444| 614 | 21737 | 77914 | 
| 59 | 29 django/core/management/commands/compilemessages.py | 29 | 56| 231 | 21968 | 77914 | 
| 60 | 29 django/core/checks/model_checks.py | 45 | 66| 168 | 22136 | 77914 | 
| 61 | 29 django/utils/translation/trans_real.py | 299 | 347| 354 | 22490 | 77914 | 
| 62 | 30 django/db/models/base.py | 1616 | 1708| 673 | 23163 | 92804 | 
| 63 | 30 django/contrib/admin/checks.py | 701 | 711| 115 | 23278 | 92804 | 
| 64 | 30 django/contrib/admin/checks.py | 124 | 141| 155 | 23433 | 92804 | 
| 65 | 30 django/core/checks/model_checks.py | 111 | 132| 263 | 23696 | 92804 | 
| 66 | 30 django/core/management/commands/makemessages.py | 216 | 281| 633 | 24329 | 92804 | 
| 67 | 31 django/core/validators.py | 340 | 385| 308 | 24637 | 97124 | 
| 68 | 31 django/contrib/admin/checks.py | 636 | 663| 232 | 24869 | 97124 | 
| 69 | 31 django/utils/translation/trans_real.py | 392 | 424| 302 | 25171 | 97124 | 
| 70 | 31 django/core/management/commands/makemessages.py | 497 | 590| 744 | 25915 | 97124 | 
| 71 | 32 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 26052 | 97261 | 
| 72 | 32 django/core/management/commands/makemessages.py | 363 | 392| 231 | 26283 | 97261 | 
| 73 | 32 django/utils/translation/trans_real.py | 59 | 108| 450 | 26733 | 97261 | 
| 74 | 32 django/conf/global_settings.py | 499 | 638| 853 | 27586 | 97261 | 
| 75 | 32 django/contrib/admin/checks.py | 578 | 589| 128 | 27714 | 97261 | 
| 76 | 33 django/conf/locale/__init__.py | 1 | 576| 75 | 27789 | 101254 | 
| 77 | 33 django/core/management/commands/makemessages.py | 394 | 416| 200 | 27989 | 101254 | 
| 78 | 34 setup.py | 1 | 66| 508 | 28497 | 102271 | 
| 79 | 35 django/utils/deprecation.py | 1 | 30| 195 | 28692 | 102963 | 
| 80 | 35 django/contrib/admin/checks.py | 225 | 255| 229 | 28921 | 102963 | 
| 81 | 36 django/db/backends/base/features.py | 1 | 115| 900 | 29821 | 105416 | 
| 82 | 37 django/db/backends/mysql/features.py | 1 | 101| 843 | 30664 | 106396 | 
| 83 | 37 django/db/models/fields/__init__.py | 348 | 374| 199 | 30863 | 106396 | 
| 84 | 37 django/contrib/admin/checks.py | 541 | 576| 303 | 31166 | 106396 | 
| 85 | 37 django/core/management/commands/makemessages.py | 36 | 57| 143 | 31309 | 106396 | 
| 86 | 38 django/middleware/common.py | 149 | 175| 254 | 31563 | 107907 | 
| 87 | 38 django/views/i18n.py | 198 | 208| 137 | 31700 | 107907 | 
| 88 | 39 django/contrib/postgres/fields/mixins.py | 1 | 30| 179 | 31879 | 108087 | 
| 89 | 39 django/core/checks/security/base.py | 193 | 211| 127 | 32006 | 108087 | 
| 90 | 39 django/utils/translation/__init__.py | 271 | 305| 262 | 32268 | 108087 | 
| 91 | 40 django/db/backends/base/base.py | 413 | 490| 525 | 32793 | 112872 | 
| 92 | 40 django/contrib/admin/checks.py | 198 | 208| 127 | 32920 | 112872 | 
| 93 | 41 django/core/management/commands/check.py | 36 | 66| 214 | 33134 | 113307 | 
| 94 | 41 django/contrib/gis/gdal/prototypes/errcheck.py | 1 | 35| 221 | 33355 | 113307 | 
| 95 | 41 django/templatetags/i18n.py | 531 | 549| 121 | 33476 | 113307 | 
| 96 | 41 django/core/management/commands/compilemessages.py | 117 | 159| 382 | 33858 | 113307 | 
| 97 | 42 django/db/backends/mysql/base.py | 245 | 278| 241 | 34099 | 116232 | 
| 98 | 42 django/contrib/auth/checks.py | 1 | 94| 646 | 34745 | 116232 | 
| 99 | 42 django/middleware/locale.py | 1 | 26| 242 | 34987 | 116232 | 
| 100 | 42 django/templatetags/i18n.py | 316 | 406| 638 | 35625 | 116232 | 
| 101 | 42 django/utils/translation/trans_real.py | 350 | 361| 110 | 35735 | 116232 | 
| 102 | 43 django/contrib/admin/tests.py | 1 | 112| 812 | 36547 | 117648 | 
| 103 | 43 django/contrib/admin/checks.py | 862 | 872| 129 | 36676 | 117648 | 
| 104 | 43 django/core/validators.py | 233 | 277| 330 | 37006 | 117648 | 
| 105 | 43 django/views/i18n.py | 23 | 62| 409 | 37415 | 117648 | 
| 106 | 44 django/db/migrations/questioner.py | 227 | 240| 123 | 37538 | 119722 | 
| 107 | 44 django/core/checks/model_checks.py | 1 | 42| 282 | 37820 | 119722 | 
| 108 | 44 django/contrib/admin/checks.py | 408 | 417| 125 | 37945 | 119722 | 
| 109 | 44 django/contrib/admin/checks.py | 786 | 836| 443 | 38388 | 119722 | 
| 110 | 44 django/db/models/base.py | 1710 | 1781| 565 | 38953 | 119722 | 
| 111 | 45 django/contrib/admin/widgets.py | 344 | 370| 328 | 39281 | 123520 | 
| 112 | 45 django/contrib/admin/checks.py | 924 | 953| 243 | 39524 | 123520 | 
| 113 | 45 django/core/management/commands/compilemessages.py | 1 | 26| 157 | 39681 | 123520 | 
| 114 | 45 django/utils/translation/template.py | 1 | 32| 343 | 40024 | 123520 | 
| 115 | 45 django/db/models/base.py | 1230 | 1259| 242 | 40266 | 123520 | 
| 116 | 45 django/contrib/admin/checks.py | 143 | 153| 123 | 40389 | 123520 | 
| 117 | 46 django/db/backends/oracle/creation.py | 130 | 165| 399 | 40788 | 127415 | 
| 118 | 46 django/contrib/admin/checks.py | 665 | 699| 265 | 41053 | 127415 | 
| 119 | 46 django/views/i18n.py | 253 | 273| 189 | 41242 | 127415 | 
| 120 | 46 django/core/management/commands/makemessages.py | 97 | 115| 154 | 41396 | 127415 | 
| 121 | 47 django/core/checks/database.py | 1 | 12| 0 | 41396 | 127468 | 
| 122 | 47 django/core/management/commands/makemessages.py | 450 | 495| 474 | 41870 | 127468 | 
| 123 | 47 django/utils/deprecation.py | 33 | 73| 336 | 42206 | 127468 | 
| 124 | 48 django/db/migrations/executor.py | 298 | 377| 712 | 42918 | 130760 | 
| 125 | 49 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 43043 | 131009 | 
| 126 | 50 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 43186 | 131700 | 
| 127 | 50 django/templatetags/i18n.py | 409 | 513| 758 | 43944 | 131700 | 
| 128 | 50 django/db/models/base.py | 1568 | 1614| 324 | 44268 | 131700 | 
| 129 | 50 django/contrib/admin/checks.py | 210 | 223| 161 | 44429 | 131700 | 
| 130 | 50 django/templatetags/i18n.py | 178 | 207| 207 | 44636 | 131700 | 
| 131 | 51 django/core/management/commands/test.py | 25 | 57| 260 | 44896 | 132109 | 
| 132 | 51 django/contrib/admin/checks.py | 504 | 514| 134 | 45030 | 132109 | 
| 133 | 51 django/contrib/admin/checks.py | 994 | 1006| 116 | 45146 | 132109 | 
| 134 | 52 django/db/backends/sqlite3/base.py | 291 | 375| 829 | 45975 | 137661 | 
| 135 | 52 scripts/manage_translations.py | 139 | 173| 414 | 46389 | 137661 | 
| 136 | 53 django/core/checks/security/sessions.py | 1 | 98| 572 | 46961 | 138234 | 
| 137 | 54 django/forms/models.py | 306 | 345| 387 | 47348 | 149683 | 
| 138 | 54 django/core/management/commands/makemessages.py | 633 | 663| 286 | 47634 | 149683 | 
| 139 | 55 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 48249 | 150708 | 
| 140 | 55 django/contrib/admin/checks.py | 713 | 744| 227 | 48476 | 150708 | 
| 141 | 55 django/db/migrations/questioner.py | 162 | 185| 246 | 48722 | 150708 | 
| 142 | 55 django/contrib/admin/checks.py | 155 | 196| 325 | 49047 | 150708 | 
| 143 | 56 django/utils/translation/reloader.py | 1 | 30| 205 | 49252 | 150914 | 
| 144 | 56 django/contrib/admin/checks.py | 470 | 480| 149 | 49401 | 150914 | 
| 145 | 56 django/templatetags/i18n.py | 231 | 253| 231 | 49632 | 150914 | 
| 146 | 56 django/contrib/admin/checks.py | 591 | 612| 162 | 49794 | 150914 | 
| 147 | 56 django/templatetags/l10n.py | 1 | 38| 251 | 50045 | 150914 | 
| 148 | 57 django/db/utils.py | 181 | 224| 295 | 50340 | 152932 | 
| 149 | 58 docs/conf.py | 1 | 95| 746 | 51086 | 155909 | 
| 150 | 58 django/contrib/admin/checks.py | 516 | 539| 230 | 51316 | 155909 | 
| 151 | 58 django/contrib/admin/checks.py | 366 | 392| 281 | 51597 | 155909 | 
| 152 | 58 django/core/management/base.py | 64 | 88| 161 | 51758 | 155909 | 
| 153 | 58 django/views/debug.py | 125 | 152| 248 | 52006 | 155909 | 
| 154 | 58 django/contrib/admin/checks.py | 320 | 346| 221 | 52227 | 155909 | 
| 155 | 58 django/db/models/base.py | 1288 | 1317| 205 | 52432 | 155909 | 
| 156 | 58 docs/_ext/djangodocs.py | 73 | 105| 255 | 52687 | 155909 | 
| 157 | 59 django/db/migrations/loader.py | 277 | 301| 205 | 52892 | 158826 | 
| 158 | 60 django/db/migrations/autodetector.py | 1038 | 1058| 136 | 53028 | 170497 | 
| 159 | 60 django/db/models/fields/__init__.py | 322 | 346| 184 | 53212 | 170497 | 
| 160 | 61 django/conf/__init__.py | 132 | 185| 472 | 53684 | 172283 | 
| 161 | 62 django/core/checks/security/csrf.py | 1 | 41| 299 | 53983 | 172582 | 
| 162 | 62 django/core/validators.py | 466 | 501| 251 | 54234 | 172582 | 
| 163 | 63 django/core/mail/message.py | 55 | 71| 171 | 54405 | 176307 | 
| 164 | 63 django/core/management/commands/makemessages.py | 418 | 448| 263 | 54668 | 176307 | 
| 165 | 63 django/contrib/admin/checks.py | 442 | 468| 190 | 54858 | 176307 | 
| 166 | 63 django/db/migrations/questioner.py | 56 | 81| 220 | 55078 | 176307 | 
| 167 | 64 django/urls/resolvers.py | 280 | 311| 190 | 55268 | 181622 | 
| 168 | 64 django/db/backends/oracle/creation.py | 30 | 100| 722 | 55990 | 181622 | 
| 169 | 64 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 56112 | 181622 | 
| 170 | 64 django/contrib/admin/checks.py | 419 | 440| 191 | 56303 | 181622 | 
| 171 | 65 django/contrib/humanize/__init__.py | 1 | 2| 0 | 56303 | 181638 | 
| 172 | 65 django/core/validators.py | 504 | 540| 219 | 56522 | 181638 | 
| 173 | 66 django/utils/encoding.py | 102 | 115| 130 | 56652 | 183934 | 
| 174 | 66 django/core/management/commands/makemessages.py | 197 | 214| 177 | 56829 | 183934 | 
| 175 | 66 django/templatetags/i18n.py | 1 | 28| 188 | 57017 | 183934 | 
| 176 | 66 django/views/debug.py | 154 | 177| 176 | 57193 | 183934 | 
| 177 | 66 django/db/backends/mysql/base.py | 280 | 318| 404 | 57597 | 183934 | 
| 178 | 66 django/contrib/admin/checks.py | 257 | 270| 135 | 57732 | 183934 | 
| 179 | 67 django/utils/formats.py | 1 | 57| 377 | 58109 | 186026 | 
| 180 | 68 django/utils/text.py | 232 | 259| 232 | 58341 | 189399 | 
| 181 | 69 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 58491 | 189549 | 
| 182 | 70 django/db/models/__init__.py | 1 | 49| 548 | 59039 | 190097 | 
| 183 | 70 django/utils/deprecation.py | 76 | 98| 158 | 59197 | 190097 | 
| 184 | 70 django/db/migrations/executor.py | 281 | 296| 165 | 59362 | 190097 | 
| 185 | 70 django/core/management/commands/check.py | 1 | 34| 226 | 59588 | 190097 | 
| 186 | 71 django/core/checks/urls.py | 30 | 50| 165 | 59753 | 190798 | 
| 187 | 71 django/utils/encoding.py | 48 | 67| 156 | 59909 | 190798 | 
| 188 | 71 django/contrib/gis/gdal/prototypes/errcheck.py | 38 | 64| 224 | 60133 | 190798 | 
| 189 | 72 django/template/context_processors.py | 52 | 82| 143 | 60276 | 191287 | 
| 190 | 72 django/core/management/commands/makemessages.py | 592 | 631| 399 | 60675 | 191287 | 
| 191 | 73 django/contrib/admin/sites.py | 70 | 84| 129 | 60804 | 195478 | 
| 192 | 74 django/conf/locale/nb/formats.py | 5 | 40| 646 | 61450 | 196169 | 
| 193 | 74 django/templatetags/i18n.py | 256 | 293| 236 | 61686 | 196169 | 


### Hint

```
Thanks for the report, however there is not a valid use case for using LANGUAGES_BIDI that are not in LANGUAGES, that's why we added this check. I don't think that it is a big issue to override LANGUAGES_BIDI (when you override LANGUAGES) or silence this checks. We do not put system checks into release notes normally, IMO it's necessary.
Reopening, based on the ​django-developers discussion. I also think that backwards compatibility has priority over the check framework, so a solution has to be found.
Thanks for the report, however there is not a valid use case for using LANGUAGES_BIDI that are not in LANGUAGES, that's why we added this check. I don't think that it is a big issue to override LANGUAGES_BIDI (when you override LANGUAGES) or silence this checks. We do not put system checks into release notes normally, IMO it's necessary.
Reopening, based on the ​django-developers discussion. I also think that backwards compatibility has priority over the check framework, so a solution has to be found.
```

## Patch

```diff
diff --git a/django/core/checks/translation.py b/django/core/checks/translation.py
--- a/django/core/checks/translation.py
+++ b/django/core/checks/translation.py
@@ -24,12 +24,6 @@
     id='translation.E004',
 )
 
-E005 = Error(
-    'You have provided values in the LANGUAGES_BIDI setting that are not in '
-    'the LANGUAGES setting.',
-    id='translation.E005',
-)
-
 
 @register(Tags.translation)
 def check_setting_language_code(app_configs, **kwargs):
@@ -62,9 +56,6 @@ def check_setting_languages_bidi(app_configs, **kwargs):
 def check_language_settings_consistent(app_configs, **kwargs):
     """Error if language settings are not consistent with each other."""
     available_tags = {i for i, _ in settings.LANGUAGES} | {'en-us'}
-    messages = []
     if settings.LANGUAGE_CODE not in available_tags:
-        messages.append(E004)
-    if not available_tags.issuperset(settings.LANGUAGES_BIDI):
-        messages.append(E005)
-    return messages
+        return [E004]
+    return []

```

## Test Patch

```diff
diff --git a/tests/check_framework/test_translation.py b/tests/check_framework/test_translation.py
--- a/tests/check_framework/test_translation.py
+++ b/tests/check_framework/test_translation.py
@@ -80,15 +80,7 @@ def test_inconsistent_language_settings(self):
             'You have provided a value for the LANGUAGE_CODE setting that is '
             'not in the LANGUAGES setting.'
         )
-        with self.settings(LANGUAGE_CODE='fr', LANGUAGES=[('en', 'English')], LANGUAGES_BIDI=[]):
+        with self.settings(LANGUAGE_CODE='fr', LANGUAGES=[('en', 'English')]):
             self.assertEqual(check_language_settings_consistent(None), [
                 Error(msg, id='translation.E004'),
             ])
-        msg = (
-            'You have provided values in the LANGUAGES_BIDI setting that are '
-            'not in the LANGUAGES setting.'
-        )
-        with self.settings(LANGUAGE_CODE='en', LANGUAGES=[('en', 'English')], LANGUAGES_BIDI=['he']):
-            self.assertEqual(check_language_settings_consistent(None), [
-                Error(msg, id='translation.E005'),
-            ])

```


## Code snippets

### 1 - django/core/checks/translation.py:

Start line: 1, End line: 71

```python
from django.conf import settings
from django.utils.translation.trans_real import language_code_re

from . import Error, Tags, register

E001 = Error(
    'You have provided an invalid value for the LANGUAGE_CODE setting: {}.',
    id='translation.E001',
)

E002 = Error(
    'You have provided an invalid language code in the LANGUAGES setting: {}.',
    id='translation.E002',
)

E003 = Error(
    'You have provided an invalid language code in the LANGUAGES_BIDI setting: {}.',
    id='translation.E003',
)

E004 = Error(
    'You have provided a value for the LANGUAGE_CODE setting that is not in '
    'the LANGUAGES setting.',
    id='translation.E004',
)

E005 = Error(
    'You have provided values in the LANGUAGES_BIDI setting that are not in '
    'the LANGUAGES setting.',
    id='translation.E005',
)


@register(Tags.translation)
def check_setting_language_code(app_configs, **kwargs):
    """Error if LANGUAGE_CODE setting is invalid."""
    tag = settings.LANGUAGE_CODE
    if not isinstance(tag, str) or not language_code_re.match(tag):
        return [Error(E001.msg.format(tag), id=E001.id)]
    return []


@register(Tags.translation)
def check_setting_languages(app_configs, **kwargs):
    """Error if LANGUAGES setting is invalid."""
    return [
        Error(E002.msg.format(tag), id=E002.id)
        for tag, _ in settings.LANGUAGES if not isinstance(tag, str) or not language_code_re.match(tag)
    ]


@register(Tags.translation)
def check_setting_languages_bidi(app_configs, **kwargs):
    """Error if LANGUAGES_BIDI setting is invalid."""
    return [
        Error(E003.msg.format(tag), id=E003.id)
        for tag in settings.LANGUAGES_BIDI if not isinstance(tag, str) or not language_code_re.match(tag)
    ]


@register(Tags.translation)
def check_language_settings_consistent(app_configs, **kwargs):
    """Error if language settings are not consistent with each other."""
    available_tags = {i for i, _ in settings.LANGUAGES} | {'en-us'}
    messages = []
    if settings.LANGUAGE_CODE not in available_tags:
        messages.append(E004)
    if not available_tags.issuperset(settings.LANGUAGES_BIDI):
        messages.append(E005)
    return messages
```
### 2 - django/utils/translation/__init__.py:

Start line: 1, End line: 36

```python
"""
Internationalization support.
"""
import re
import warnings
from contextlib import ContextDecorator

from django.utils.autoreload import autoreload_started, file_changed
from django.utils.deprecation import RemovedInDjango40Warning
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
### 3 - django/utils/translation/trans_null.py:

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
### 4 - django/conf/global_settings.py:

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
### 5 - django/utils/translation/trans_real.py:

Start line: 364, End line: 389

```python
@functools.lru_cache(maxsize=1000)
def check_for_language(lang_code):
    """
    Check whether there is a global language file for the given language
    code. This is used to decide whether a user-provided language is
    available.

    lru_cache should have a maxsize to prevent from memory exhaustion attacks,
    as the provided language codes are taken from the HTTP request. See also
    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.
    """
    # First, a quick check to make sure lang_code is well-formed (#21458)
    if lang_code is None or not language_code_re.search(lang_code):
        return False
    return any(
        gettext_module.find('django', path, [to_locale(lang_code)]) is not None
        for path in all_locale_paths()
    )


@functools.lru_cache()
def get_languages():
    """
    Cache of settings.LANGUAGES in a dictionary for easy lookups by key.
    """
    return dict(settings.LANGUAGES)
```
### 6 - django/utils/translation/trans_real.py:

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

from . import to_language, to_locale

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
### 7 - django/utils/translation/__init__.py:

Start line: 196, End line: 268

```python
def _lazy_number_unpickle(func, resultclass, number, kwargs):
    return lazy_number(func, resultclass, number=number, **kwargs)


def ngettext_lazy(singular, plural, number=None):
    return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)


def ungettext_lazy(singular, plural, number=None):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    An alias of ungettext_lazy() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ungettext_lazy() is deprecated in favor of '
        'django.utils.translation.ngettext_lazy().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return ngettext_lazy(singular, plural, number)


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
### 8 - django/utils/translation/__init__.py:

Start line: 67, End line: 146

```python
_trans = Trans()

# The Trans class is no more needed, so remove it from the namespace.
del Trans


def gettext_noop(message):
    return _trans.gettext_noop(message)


def ugettext_noop(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of gettext_noop() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext_noop() is deprecated in favor of '
        'django.utils.translation.gettext_noop().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext_noop(message)


def gettext(message):
    return _trans.gettext(message)


def ugettext(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of gettext() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext() is deprecated in favor of '
        'django.utils.translation.gettext().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext(message)


def ngettext(singular, plural, number):
    return _trans.ngettext(singular, plural, number)


def ungettext(singular, plural, number):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of ngettext() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ungettext() is deprecated in favor of '
        'django.utils.translation.ngettext().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return ngettext(singular, plural, number)


def pgettext(context, message):
    return _trans.pgettext(context, message)


def npgettext(context, singular, plural, number):
    return _trans.npgettext(context, singular, plural, number)


gettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)


def ugettext_lazy(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2. Has been
    Alias of gettext_lazy since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext_lazy() is deprecated in favor of '
        'django.utils.translation.gettext_lazy().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext_lazy(message)
```
### 9 - django/views/i18n.py:

Start line: 77, End line: 180

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
### 10 - scripts/manage_translations.py:

Start line: 83, End line: 103

```python
def update_catalogs(resources=None, languages=None):
    """
    Update the en/LC_MESSAGES/django.po (main and contrib) files with
    new/updated translatable strings.
    """
    settings.configure()
    django.setup()
    if resources is not None:
        print("`update_catalogs` will always process all resources.")
    contrib_dirs = _get_locale_dirs(None, include_core=False)

    os.chdir(os.path.join(os.getcwd(), 'django'))
    print("Updating en catalogs for Django and contrib apps...")
    call_command('makemessages', locale=['en'])
    print("Updating en JS catalogs for Django and contrib apps...")
    call_command('makemessages', locale=['en'], domain='djangojs')

    # Output changed stats
    _check_diff('core', os.path.join(os.getcwd(), 'conf', 'locale'))
    for name, dir_ in contrib_dirs:
        _check_diff(name, dir_)
```
