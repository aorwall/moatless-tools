# django__django-14871

| **django/django** | `32b7ffc2bbfd1ae055bdbe287f8598de731adce1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 901 |
| **Any found context length** | 569 |
| **Avg pos** | 9.0 |
| **Min pos** | 2 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/widgets.py b/django/contrib/admin/widgets.py
--- a/django/contrib/admin/widgets.py
+++ b/django/contrib/admin/widgets.py
@@ -388,6 +388,7 @@ def __init__(self, field, admin_site, attrs=None, choices=(), using=None):
         self.db = using
         self.choices = choices
         self.attrs = {} if attrs is None else attrs.copy()
+        self.i18n_name = SELECT2_TRANSLATIONS.get(get_language())
 
     def get_url(self):
         return reverse(self.url_name % self.admin_site.name)
@@ -413,6 +414,7 @@ def build_attrs(self, base_attrs, extra_attrs=None):
             'data-theme': 'admin-autocomplete',
             'data-allow-clear': json.dumps(not self.is_required),
             'data-placeholder': '',  # Allows clearing of the input.
+            'lang': self.i18n_name,
             'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete',
         })
         return attrs
@@ -449,8 +451,7 @@ def optgroups(self, name, value, attr=None):
     @property
     def media(self):
         extra = '' if settings.DEBUG else '.min'
-        i18n_name = SELECT2_TRANSLATIONS.get(get_language())
-        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else ()
+        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % self.i18n_name,) if self.i18n_name else ()
         return forms.Media(
             js=(
                 'admin/js/vendor/jquery/jquery%s.js' % extra,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/widgets.py | 391 | 391 | 4 | 1 | 901
| django/contrib/admin/widgets.py | 416 | 416 | 2 | 1 | 569
| django/contrib/admin/widgets.py | 452 | 453 | 3 | 1 | 772


## Problem Statement

```
Select2 doesn't load translations with subtags.
Description
	
For example, when using the setting LANGUAGE_CODE="pt-BR", the translation of select2 is not applied, the static file i18n is not found. 
This is due to the fact that some languages are converted to lowercase. ​https://github.com/django/django/blob/main/django/contrib/admin/widgets.py#L366

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/contrib/admin/widgets.py** | 347 | 373| 328 | 328 | 3872 | 
| **-> 2 <-** | **1 django/contrib/admin/widgets.py** | 395 | 418| 241 | 569 | 3872 | 
| **-> 3 <-** | **1 django/contrib/admin/widgets.py** | 449 | 477| 203 | 772 | 3872 | 
| **-> 4 <-** | **1 django/contrib/admin/widgets.py** | 376 | 393| 129 | 901 | 3872 | 
| 5 | 2 django/views/i18n.py | 79 | 182| 702 | 1603 | 6333 | 
| 6 | 2 django/views/i18n.py | 200 | 210| 137 | 1740 | 6333 | 
| 7 | 3 django/forms/widgets.py | 672 | 703| 261 | 2001 | 14443 | 
| 8 | 4 django/utils/translation/__init__.py | 48 | 60| 131 | 2132 | 16284 | 
| 9 | 4 django/forms/widgets.py | 744 | 773| 219 | 2351 | 16284 | 
| 10 | 5 django/contrib/admin/tests.py | 139 | 155| 176 | 2527 | 17762 | 
| 11 | 6 django/templatetags/i18n.py | 70 | 99| 236 | 2763 | 21853 | 
| 12 | 6 django/utils/translation/__init__.py | 1 | 30| 242 | 3005 | 21853 | 
| 13 | 7 django/utils/translation/trans_real.py | 212 | 225| 154 | 3159 | 26236 | 
| 14 | 8 django/utils/translation/trans_null.py | 1 | 68| 269 | 3428 | 26505 | 
| 15 | 8 django/templatetags/i18n.py | 1 | 30| 193 | 3621 | 26505 | 
| 16 | 8 django/templatetags/i18n.py | 534 | 549| 197 | 3818 | 26505 | 
| 17 | 9 scripts/manage_translations.py | 176 | 186| 116 | 3934 | 28161 | 
| 18 | 9 django/templatetags/i18n.py | 33 | 67| 222 | 4156 | 28161 | 
| 19 | 9 django/utils/translation/__init__.py | 239 | 270| 232 | 4388 | 28161 | 
| 20 | 9 django/utils/translation/__init__.py | 202 | 236| 271 | 4659 | 28161 | 
| 21 | 9 django/templatetags/i18n.py | 552 | 570| 121 | 4780 | 28161 | 
| 22 | 10 django/templatetags/l10n.py | 41 | 64| 190 | 4970 | 28603 | 
| 23 | 10 django/templatetags/l10n.py | 1 | 38| 251 | 5221 | 28603 | 
| 24 | 10 scripts/manage_translations.py | 1 | 29| 197 | 5418 | 28603 | 
| 25 | 10 django/templatetags/i18n.py | 102 | 136| 273 | 5691 | 28603 | 
| 26 | 10 django/utils/translation/trans_real.py | 262 | 341| 458 | 6149 | 28603 | 
| 27 | 10 django/utils/translation/trans_real.py | 518 | 558| 289 | 6438 | 28603 | 
| 28 | 11 django/conf/__init__.py | 147 | 166| 172 | 6610 | 30862 | 
| 29 | 11 django/templatetags/i18n.py | 196 | 225| 207 | 6817 | 30862 | 
| 30 | 11 django/templatetags/i18n.py | 274 | 311| 236 | 7053 | 30862 | 
| 31 | 11 django/templatetags/i18n.py | 334 | 425| 643 | 7696 | 30862 | 
| 32 | 11 django/templatetags/i18n.py | 138 | 193| 459 | 8155 | 30862 | 
| 33 | 12 django/conf/global_settings.py | 56 | 155| 1160 | 9315 | 36618 | 
| 34 | 13 django/middleware/locale.py | 28 | 67| 388 | 9703 | 37239 | 
| 35 | 14 django/contrib/admin/options.py | 1 | 96| 756 | 10459 | 55921 | 
| 36 | 14 django/contrib/admin/tests.py | 173 | 194| 205 | 10664 | 55921 | 
| 37 | 14 django/forms/widgets.py | 776 | 790| 139 | 10803 | 55921 | 
| 38 | 14 django/views/i18n.py | 255 | 275| 189 | 10992 | 55921 | 
| 39 | 14 django/utils/translation/__init__.py | 140 | 199| 340 | 11332 | 55921 | 
| 40 | 14 django/views/i18n.py | 277 | 294| 158 | 11490 | 55921 | 
| 41 | 14 django/utils/translation/__init__.py | 63 | 90| 136 | 11626 | 55921 | 
| 42 | 14 django/views/i18n.py | 246 | 253| 142 | 11768 | 55921 | 
| 43 | **14 django/contrib/admin/widgets.py** | 92 | 117| 179 | 11947 | 55921 | 
| 44 | 14 django/views/i18n.py | 185 | 198| 127 | 12074 | 55921 | 
| 45 | 15 django/views/static.py | 57 | 80| 211 | 12285 | 56973 | 
| 46 | **15 django/contrib/admin/widgets.py** | 1 | 46| 330 | 12615 | 56973 | 
| 47 | 15 django/forms/widgets.py | 984 | 1018| 391 | 13006 | 56973 | 
| 48 | 15 django/forms/widgets.py | 706 | 741| 237 | 13243 | 56973 | 
| 49 | 15 django/forms/widgets.py | 1045 | 1064| 144 | 13387 | 56973 | 
| 50 | 15 django/utils/translation/trans_real.py | 463 | 498| 340 | 13727 | 56973 | 
| 51 | 16 django/contrib/admin/sites.py | 419 | 433| 129 | 13856 | 61402 | 
| 52 | 16 django/utils/translation/trans_real.py | 1 | 58| 503 | 14359 | 61402 | 
| 53 | **16 django/contrib/admin/widgets.py** | 420 | 447| 296 | 14655 | 61402 | 
| 54 | 17 django/core/checks/translation.py | 1 | 65| 445 | 15100 | 61847 | 
| 55 | 18 django/db/backends/mysql/features.py | 51 | 63| 130 | 15230 | 64099 | 
| 56 | 18 django/utils/translation/__init__.py | 31 | 46| 154 | 15384 | 64099 | 
| 57 | 19 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 186 | 15570 | 67791 | 
| 58 | 20 django/contrib/admin/models.py | 1 | 20| 118 | 15688 | 68914 | 
| 59 | 21 django/conf/locale/pt_BR/formats.py | 5 | 32| 459 | 16147 | 69418 | 
| 60 | 21 django/views/i18n.py | 67 | 76| 120 | 16267 | 69418 | 
| 61 | 21 django/templatetags/i18n.py | 428 | 533| 764 | 17031 | 69418 | 
| 62 | 21 django/utils/translation/trans_real.py | 435 | 460| 221 | 17252 | 69418 | 
| 63 | 22 django/contrib/admin/helpers.py | 1 | 32| 215 | 17467 | 72891 | 
| 64 | 22 scripts/manage_translations.py | 82 | 102| 191 | 17658 | 72891 | 
| 65 | 22 django/forms/widgets.py | 1066 | 1089| 250 | 17908 | 72891 | 
| 66 | 23 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 18197 | 73196 | 
| 67 | 24 django/template/backends/dummy.py | 1 | 53| 325 | 18522 | 73521 | 
| 68 | 25 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 18784 | 73799 | 
| 69 | 26 django/template/backends/jinja2.py | 1 | 51| 341 | 19125 | 74621 | 
| 70 | 26 django/middleware/locale.py | 1 | 26| 239 | 19364 | 74621 | 
| 71 | 26 django/forms/widgets.py | 621 | 638| 179 | 19543 | 74621 | 
| 72 | 27 django/utils/encoding.py | 46 | 65| 156 | 19699 | 76836 | 
| 73 | 28 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 19951 | 77104 | 
| 74 | 28 django/utils/translation/trans_real.py | 227 | 239| 127 | 20078 | 77104 | 
| 75 | 29 django/contrib/admin/templatetags/base.py | 22 | 34| 134 | 20212 | 77402 | 
| 76 | 30 django/db/models/fields/__init__.py | 2106 | 2139| 228 | 20440 | 95549 | 
| 77 | 31 django/template/context_processors.py | 53 | 83| 143 | 20583 | 96038 | 
| 78 | 32 django/conf/locale/pt/formats.py | 5 | 36| 577 | 21160 | 96660 | 
| 79 | 33 django/core/management/commands/makemessages.py | 37 | 58| 143 | 21303 | 102287 | 
| 80 | 34 django/urls/resolvers.py | 305 | 336| 190 | 21493 | 108079 | 
| 81 | 35 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 21763 | 108365 | 
| 82 | 35 django/utils/translation/trans_real.py | 241 | 259| 140 | 21903 | 108365 | 
| 83 | 36 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 22178 | 108685 | 
| 84 | 37 django/forms/boundfield.py | 35 | 50| 149 | 22327 | 110961 | 
| 85 | 37 django/contrib/admin/options.py | 1029 | 1046| 184 | 22511 | 110961 | 
| 86 | 38 django/views/csrf.py | 1 | 13| 132 | 22643 | 112505 | 
| 87 | 38 django/views/i18n.py | 212 | 221| 131 | 22774 | 112505 | 
| 88 | 39 django/conf/locale/bn/formats.py | 5 | 33| 294 | 23068 | 112843 | 
| 89 | **39 django/contrib/admin/widgets.py** | 311 | 323| 114 | 23182 | 112843 | 
| 90 | 39 django/contrib/admin/templatetags/admin_list.py | 166 | 180| 136 | 23318 | 112843 | 
| 91 | 40 django/contrib/admin/views/main.py | 1 | 44| 317 | 23635 | 117310 | 
| 92 | 41 django/conf/locale/bg/formats.py | 5 | 22| 131 | 23766 | 117485 | 
| 93 | 42 django/contrib/postgres/operations.py | 192 | 213| 207 | 23973 | 119871 | 
| 94 | 42 django/views/i18n.py | 223 | 244| 176 | 24149 | 119871 | 
| 95 | 43 django/conf/locale/sq/formats.py | 5 | 22| 128 | 24277 | 120043 | 
| 96 | 44 django/conf/locale/bs/formats.py | 5 | 22| 139 | 24416 | 120226 | 
| 97 | 45 django/core/checks/templates.py | 1 | 36| 259 | 24675 | 120486 | 
| 98 | 46 django/conf/locale/ta/formats.py | 5 | 22| 125 | 24800 | 120655 | 
| 99 | 47 django/forms/formsets.py | 1 | 24| 204 | 25004 | 124742 | 
| 100 | 48 django/conf/locale/da/formats.py | 5 | 27| 250 | 25254 | 125037 | 
| 101 | 48 django/contrib/admin/tests.py | 157 | 171| 160 | 25414 | 125037 | 
| 102 | 48 django/forms/widgets.py | 640 | 669| 238 | 25652 | 125037 | 
| 103 | 49 django/conf/locale/ig/formats.py | 5 | 33| 388 | 26040 | 125470 | 
| 104 | 50 django/conf/locale/it/formats.py | 5 | 41| 764 | 26804 | 126279 | 
| 105 | 50 scripts/manage_translations.py | 140 | 173| 390 | 27194 | 126279 | 
| 106 | 50 django/utils/translation/trans_real.py | 118 | 167| 452 | 27646 | 126279 | 
| 107 | 50 django/contrib/admin/sites.py | 1 | 35| 225 | 27871 | 126279 | 
| 108 | 51 django/conf/locale/pl/formats.py | 5 | 29| 321 | 28192 | 126645 | 
| 109 | 52 django/contrib/admin/filters.py | 280 | 304| 217 | 28409 | 130775 | 
| 110 | 53 django/utils/text.py | 384 | 398| 164 | 28573 | 134160 | 
| 111 | 54 django/conf/locale/ru/formats.py | 5 | 31| 367 | 28940 | 134572 | 
| 112 | 55 django/conf/locale/tk/formats.py | 5 | 33| 402 | 29342 | 135019 | 
| 113 | 56 django/db/backends/oracle/features.py | 1 | 121| 1032 | 30374 | 136052 | 
| 114 | 56 django/contrib/admin/options.py | 285 | 374| 636 | 31010 | 136052 | 
| 115 | 57 django/conf/locale/et/formats.py | 5 | 22| 133 | 31143 | 136229 | 
| 116 | 58 django/conf/locale/uz/formats.py | 5 | 31| 418 | 31561 | 136692 | 
| 117 | 59 django/utils/translation/template.py | 37 | 61| 165 | 31726 | 138646 | 
| 118 | 60 django/conf/locale/tg/formats.py | 5 | 33| 402 | 32128 | 139093 | 
| 119 | 61 django/conf/locale/es/formats.py | 5 | 31| 285 | 32413 | 139423 | 
| 120 | 62 django/conf/locale/sv/formats.py | 5 | 36| 481 | 32894 | 139949 | 
| 121 | 63 django/conf/locale/is/formats.py | 5 | 22| 130 | 33024 | 140124 | 
| 122 | 63 django/contrib/admin/options.py | 640 | 657| 128 | 33152 | 140124 | 
| 123 | 64 django/conf/locale/id/formats.py | 5 | 47| 670 | 33822 | 140839 | 
| 124 | 65 django/conf/locale/ro/formats.py | 5 | 36| 262 | 34084 | 141146 | 
| 125 | 65 django/utils/translation/trans_real.py | 61 | 115| 408 | 34492 | 141146 | 
| 126 | 66 django/conf/locale/en_GB/formats.py | 5 | 37| 655 | 35147 | 141846 | 
| 127 | **66 django/contrib/admin/widgets.py** | 161 | 192| 243 | 35390 | 141846 | 
| 128 | 67 django/contrib/admin/__init__.py | 1 | 25| 255 | 35645 | 142101 | 
| 129 | 68 django/contrib/auth/forms.py | 1 | 30| 228 | 35873 | 145227 | 
| 130 | 69 django/conf/locale/eo/formats.py | 5 | 48| 707 | 36580 | 145979 | 
| 131 | 69 django/forms/widgets.py | 551 | 585| 277 | 36857 | 145979 | 
| 132 | 70 django/conf/locale/fy/formats.py | 22 | 22| 0 | 36857 | 146131 | 
| 133 | 71 django/conf/locale/sk/formats.py | 5 | 29| 330 | 37187 | 146506 | 
| 134 | 72 django/conf/locale/uk/formats.py | 5 | 36| 425 | 37612 | 146976 | 
| 135 | 73 django/core/validators.py | 227 | 242| 135 | 37747 | 151487 | 
| 136 | 74 django/forms/renderers.py | 1 | 66| 380 | 38127 | 151867 | 
| 137 | 74 django/views/i18n.py | 297 | 316| 114 | 38241 | 151867 | 
| 138 | 75 django/conf/locale/te/formats.py | 5 | 22| 123 | 38364 | 152034 | 
| 139 | 76 django/forms/fields.py | 1191 | 1228| 199 | 38563 | 161455 | 
| 140 | 77 django/db/models/sql/compiler.py | 427 | 435| 133 | 38696 | 176205 | 
| 141 | 78 django/conf/locale/eu/formats.py | 5 | 22| 171 | 38867 | 176421 | 
| 142 | 78 django/utils/translation/trans_real.py | 421 | 432| 110 | 38977 | 176421 | 
| 143 | 79 django/conf/locale/sl/formats.py | 5 | 43| 708 | 39685 | 177174 | 
| 144 | 80 django/conf/locale/de/formats.py | 5 | 28| 305 | 39990 | 177524 | 
| 145 | 81 django/contrib/humanize/templatetags/humanize.py | 81 | 94| 287 | 40277 | 180408 | 
| 146 | 82 django/template/defaultfilters.py | 277 | 342| 425 | 40702 | 186829 | 
| 147 | 82 django/template/defaultfilters.py | 228 | 255| 155 | 40857 | 186829 | 
| 148 | 83 django/conf/locale/ar/formats.py | 5 | 22| 135 | 40992 | 187008 | 
| 149 | 84 django/contrib/gis/geoip2/base.py | 1 | 21| 145 | 41137 | 189024 | 
| 150 | 84 django/db/models/sql/compiler.py | 1076 | 1116| 337 | 41474 | 189024 | 
| 151 | 85 django/conf/locale/tr/formats.py | 5 | 29| 319 | 41793 | 189388 | 
| 152 | 86 django/conf/locale/az/formats.py | 5 | 31| 364 | 42157 | 189797 | 
| 153 | 87 django/conf/locale/en_AU/formats.py | 5 | 37| 655 | 42812 | 190497 | 
| 154 | 88 django/conf/locale/lt/formats.py | 5 | 44| 676 | 43488 | 191218 | 
| 155 | 89 django/conf/locale/hu/formats.py | 5 | 31| 305 | 43793 | 191568 | 
| 156 | 90 docs/_ext/djangodocs.py | 109 | 170| 526 | 44319 | 194724 | 
| 157 | 91 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 44538 | 196318 | 
| 158 | 92 django/conf/locale/de_CH/formats.py | 5 | 34| 403 | 44941 | 196766 | 


### Hint

```
I fixed it ​https://github.com/django/django/pull/14813
LANGUAGE_CODE should be lowercased, so pt-br instead of pt-BR, see ​Definitions, ​LANGUAGE_CODE docs, and the list of ​LANGUAGES.
Replying to Mariusz Felisiak: LANGUAGE_CODE should be lowercased, so pt-br instead of pt-BR, see ​Definitions, ​LANGUAGE_CODE docs, and the list of ​LANGUAGES. The translation of select2 only accepts pt-BR that way. The file is found but the translation doesn't work. It only works when the tag html has <html lang="pt-BR" dir="ltr" data-select2-id="14">...
Replying to Cleiton de Lima: Replying to Mariusz Felisiak: LANGUAGE_CODE should be lowercased, so pt-br instead of pt-BR, see ​Definitions, ​LANGUAGE_CODE docs, and the list of ​LANGUAGES. The translation of select2 only accepts pt-BR that way. The file is found but the translation doesn't work. It only works when the tag html has <html lang="pt-BR" dir="ltr" data-select2-id="14">... Thanks, I didn't notice that Select2 loads translations based on LANG. Lowercase when searching for a file will help only for pt-BR but not for zh-hans, pt-br etc. We could probably add lang to the attrs, e.g.: diff --git a/django/contrib/admin/widgets.py b/django/contrib/admin/widgets.py index aeb74773ac..f1002cac6c 100644 --- a/django/contrib/admin/widgets.py +++ b/django/contrib/admin/widgets.py @@ -388,6 +388,7 @@ class AutocompleteMixin: self.db = using self.choices = choices self.attrs = {} if attrs is None else attrs.copy() + self.i18n_name = SELECT2_TRANSLATIONS.get(get_language()) def get_url(self): return reverse(self.url_name % self.admin_site.name) @@ -413,6 +414,7 @@ class AutocompleteMixin: 'data-theme': 'admin-autocomplete', 'data-allow-clear': json.dumps(not self.is_required), 'data-placeholder': '', # Allows clearing of the input. + 'lang': self.i18n_name, 'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete', }) return attrs @@ -449,8 +451,7 @@ class AutocompleteMixin: @property def media(self): extra = '' if settings.DEBUG else '.min' - i18n_name = SELECT2_TRANSLATIONS.get(get_language()) - i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else () + i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % self.i18n_name,) if self.i18n_name else () return forms.Media( js=( 'admin/js/vendor/jquery/jquery%s.js' % extra, What do you think? It works for me.
It looks good to me too! Replying to Mariusz Felisiak: Replying to Cleiton de Lima: Replying to Mariusz Felisiak: LANGUAGE_CODE should be lowercased, so pt-br instead of pt-BR, see ​Definitions, ​LANGUAGE_CODE docs, and the list of ​LANGUAGES. The translation of select2 only accepts pt-BR that way. The file is found but the translation doesn't work. It only works when the tag html has <html lang="pt-BR" dir="ltr" data-select2-id="14">... Thanks, I didn't notice that Select2 loads translations based on LANG. Lowercase when searching for a file will help only for pt-BR but not for zh-hans, pt-br etc. We could probably add lang to the attrs, e.g.: diff --git a/django/contrib/admin/widgets.py b/django/contrib/admin/widgets.py index aeb74773ac..f1002cac6c 100644 --- a/django/contrib/admin/widgets.py +++ b/django/contrib/admin/widgets.py @@ -388,6 +388,7 @@ class AutocompleteMixin: self.db = using self.choices = choices self.attrs = {} if attrs is None else attrs.copy() + self.i18n_name = SELECT2_TRANSLATIONS.get(get_language()) def get_url(self): return reverse(self.url_name % self.admin_site.name) @@ -413,6 +414,7 @@ class AutocompleteMixin: 'data-theme': 'admin-autocomplete', 'data-allow-clear': json.dumps(not self.is_required), 'data-placeholder': '', # Allows clearing of the input. + 'lang': self.i18n_name, 'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete', }) return attrs @@ -449,8 +451,7 @@ class AutocompleteMixin: @property def media(self): extra = '' if settings.DEBUG else '.min' - i18n_name = SELECT2_TRANSLATIONS.get(get_language()) - i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else () + i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % self.i18n_name,) if self.i18n_name else () return forms.Media( js=( 'admin/js/vendor/jquery/jquery%s.js' % extra, What do you think? It works for me.
Cleiton, thanks for checking. Would you like to prepare a patch?
Yes, I will. Replying to Mariusz Felisiak: Cleiton, thanks for checking. Would you like to prepare a patch?
```

## Patch

```diff
diff --git a/django/contrib/admin/widgets.py b/django/contrib/admin/widgets.py
--- a/django/contrib/admin/widgets.py
+++ b/django/contrib/admin/widgets.py
@@ -388,6 +388,7 @@ def __init__(self, field, admin_site, attrs=None, choices=(), using=None):
         self.db = using
         self.choices = choices
         self.attrs = {} if attrs is None else attrs.copy()
+        self.i18n_name = SELECT2_TRANSLATIONS.get(get_language())
 
     def get_url(self):
         return reverse(self.url_name % self.admin_site.name)
@@ -413,6 +414,7 @@ def build_attrs(self, base_attrs, extra_attrs=None):
             'data-theme': 'admin-autocomplete',
             'data-allow-clear': json.dumps(not self.is_required),
             'data-placeholder': '',  # Allows clearing of the input.
+            'lang': self.i18n_name,
             'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete',
         })
         return attrs
@@ -449,8 +451,7 @@ def optgroups(self, name, value, attr=None):
     @property
     def media(self):
         extra = '' if settings.DEBUG else '.min'
-        i18n_name = SELECT2_TRANSLATIONS.get(get_language())
-        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else ()
+        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % self.i18n_name,) if self.i18n_name else ()
         return forms.Media(
             js=(
                 'admin/js/vendor/jquery/jquery%s.js' % extra,

```

## Test Patch

```diff
diff --git a/tests/admin_widgets/test_autocomplete_widget.py b/tests/admin_widgets/test_autocomplete_widget.py
--- a/tests/admin_widgets/test_autocomplete_widget.py
+++ b/tests/admin_widgets/test_autocomplete_widget.py
@@ -72,7 +72,8 @@ def test_build_attrs(self):
             'data-app-label': 'admin_widgets',
             'data-field-name': 'band',
             'data-model-name': 'album',
-            'data-placeholder': ''
+            'data-placeholder': '',
+            'lang': 'en',
         })
 
     def test_build_attrs_no_custom_class(self):

```


## Code snippets

### 1 - django/contrib/admin/widgets.py:

Start line: 347, End line: 373

```python
class AdminIntegerFieldWidget(forms.NumberInput):
    class_name = 'vIntegerField'

    def __init__(self, attrs=None):
        super().__init__(attrs={'class': self.class_name, **(attrs or {})})


class AdminBigIntegerFieldWidget(AdminIntegerFieldWidget):
    class_name = 'vBigIntegerField'


class AdminUUIDInputWidget(forms.TextInput):
    def __init__(self, attrs=None):
        super().__init__(attrs={'class': 'vUUIDField', **(attrs or {})})


# Mapping of lowercase language codes [returned by Django's get_language()] to
# language codes supported by select2.
# See django/contrib/admin/static/admin/js/vendor/select2/i18n/*
SELECT2_TRANSLATIONS = {x.lower(): x for x in [
    'ar', 'az', 'bg', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
    'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'id', 'is',
    'it', 'ja', 'km', 'ko', 'lt', 'lv', 'mk', 'ms', 'nb', 'nl', 'pl',
    'pt-BR', 'pt', 'ro', 'ru', 'sk', 'sr-Cyrl', 'sr', 'sv', 'th',
    'tr', 'uk', 'vi',
]}
SELECT2_TRANSLATIONS.update({'zh-hans': 'zh-CN', 'zh-hant': 'zh-TW'})
```
### 2 - django/contrib/admin/widgets.py:

Start line: 395, End line: 418

```python
class AutocompleteMixin:

    def build_attrs(self, base_attrs, extra_attrs=None):
        """
        Set select2's AJAX attributes.

        Attributes can be set using the html5 data attribute.
        Nested attributes require a double dash as per
        https://select2.org/configuration/data-attributes#nested-subkey-options
        """
        attrs = super().build_attrs(base_attrs, extra_attrs=extra_attrs)
        attrs.setdefault('class', '')
        attrs.update({
            'data-ajax--cache': 'true',
            'data-ajax--delay': 250,
            'data-ajax--type': 'GET',
            'data-ajax--url': self.get_url(),
            'data-app-label': self.field.model._meta.app_label,
            'data-model-name': self.field.model._meta.model_name,
            'data-field-name': self.field.name,
            'data-theme': 'admin-autocomplete',
            'data-allow-clear': json.dumps(not self.is_required),
            'data-placeholder': '',  # Allows clearing of the input.
            'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete',
        })
        return attrs
```
### 3 - django/contrib/admin/widgets.py:

Start line: 449, End line: 477

```python
class AutocompleteMixin:

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        i18n_name = SELECT2_TRANSLATIONS.get(get_language())
        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else ()
        return forms.Media(
            js=(
                'admin/js/vendor/jquery/jquery%s.js' % extra,
                'admin/js/vendor/select2/select2.full%s.js' % extra,
            ) + i18n_file + (
                'admin/js/jquery.init.js',
                'admin/js/autocomplete.js',
            ),
            css={
                'screen': (
                    'admin/css/vendor/select2/select2%s.css' % extra,
                    'admin/css/autocomplete.css',
                ),
            },
        )


class AutocompleteSelect(AutocompleteMixin, forms.Select):
    pass


class AutocompleteSelectMultiple(AutocompleteMixin, forms.SelectMultiple):
    pass
```
### 4 - django/contrib/admin/widgets.py:

Start line: 376, End line: 393

```python
class AutocompleteMixin:
    """
    Select widget mixin that loads options from AutocompleteJsonView via AJAX.

    Renders the necessary data attributes for select2 and adds the static form
    media.
    """
    url_name = '%s:autocomplete'

    def __init__(self, field, admin_site, attrs=None, choices=(), using=None):
        self.field = field
        self.admin_site = admin_site
        self.db = using
        self.choices = choices
        self.attrs = {} if attrs is None else attrs.copy()

    def get_url(self):
        return reverse(self.url_name % self.admin_site.name)
```
### 5 - django/views/i18n.py:

Start line: 79, End line: 182

```python
js_catalog_template = r"""
{% autoescape off %}
'use strict';
{
  const globals = this;
  const django = globals.django || (globals.django = {});

  {% if plural %}
  django.pluralidx = function(n) {
    const v = {{ plural }};
    if (typeof v === 'boolean') {
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
  const newcatalog = {{ catalog_str }};
  for (const key in newcatalog) {
    django.catalog[key] = newcatalog[key];
  }
  {% endif %}

  if (!django.jsi18n_initialized) {
    django.gettext = function(msgid) {
      const value = django.catalog[msgid];
      if (typeof value === 'undefined') {
        return msgid;
      } else {
        return (typeof value === 'string') ? value : value[0];
      }
    };

    django.ngettext = function(singular, plural, count) {
      const value = django.catalog[singular];
      if (typeof value === 'undefined') {
        return (count == 1) ? singular : plural;
      } else {
        return value.constructor === Array ? value[django.pluralidx(count)] : value;
      }
    };

    django.gettext_noop = function(msgid) { return msgid; };

    django.pgettext = function(context, msgid) {
      let value = django.gettext(context + '\x04' + msgid);
      if (value.includes('\x04')) {
        value = msgid;
      }
      return value;
    };

    django.npgettext = function(context, singular, plural, count) {
      let value = django.ngettext(context + '\x04' + singular, context + '\x04' + plural, count);
      if (value.includes('\x04')) {
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
      const value = django.formats[format_type];
      if (typeof value === 'undefined') {
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
};
{% endautoescape %}
"""
```
### 6 - django/views/i18n.py:

Start line: 200, End line: 210

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
### 7 - django/forms/widgets.py:

Start line: 672, End line: 703

```python
class Select(ChoiceWidget):
    input_type = 'select'
    template_name = 'django/forms/widgets/select.html'
    option_template_name = 'django/forms/widgets/select_option.html'
    add_id_index = False
    checked_attribute = {'selected': True}
    option_inherits_attrs = False

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.allow_multiple_selected:
            context['widget']['attrs']['multiple'] = True
        return context

    @staticmethod
    def _choice_has_empty_value(choice):
        """Return True if the choice's value is empty string or None."""
        value, _ = choice
        return value is None or value == ''

    def use_required_attribute(self, initial):
        """
        Don't render 'required' if the first <option> has a value, as that's
        invalid HTML.
        """
        use_required_attribute = super().use_required_attribute(initial)
        # 'required' is always okay for <select multiple>.
        if self.allow_multiple_selected:
            return use_required_attribute

        first_choice = next(iter(self.choices), None)
        return use_required_attribute and first_choice is not None and self._choice_has_empty_value(first_choice)
```
### 8 - django/utils/translation/__init__.py:

Start line: 48, End line: 60

```python
class Trans:

    def __getattr__(self, real_name):
        from django.conf import settings
        if settings.USE_I18N:
            from django.utils.translation import trans_real as trans
            from django.utils.translation.reloader import (
                translation_file_changed, watch_for_translation_changes,
            )
            autoreload_started.connect(watch_for_translation_changes, dispatch_uid='translation_file_changed')
            file_changed.connect(translation_file_changed, dispatch_uid='translation_file_changed')
        else:
            from django.utils.translation import trans_null as trans
        setattr(self, real_name, getattr(trans, real_name))
        return getattr(trans, real_name)
```
### 9 - django/forms/widgets.py:

Start line: 744, End line: 773

```python
class SelectMultiple(Select):
    allow_multiple_selected = True

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def value_omitted_from_data(self, data, files, name):
        # An unselected <select multiple> doesn't appear in POST data, so it's
        # never known if the value is actually omitted.
        return False


class RadioSelect(ChoiceWidget):
    input_type = 'radio'
    template_name = 'django/forms/widgets/radio.html'
    option_template_name = 'django/forms/widgets/radio_option.html'

    def id_for_label(self, id_, index=None):
        """
        Don't include for="field_0" in <label> to improve accessibility when
        using a screen reader, in addition clicking such a label would toggle
        the first input.
        """
        if index is None:
            return ''
        return super().id_for_label(id_, index)
```
### 10 - django/contrib/admin/tests.py:

Start line: 139, End line: 155

```python
@modify_settings(MIDDLEWARE={'append': 'django.contrib.admin.tests.CSPMiddleware'})
class AdminSeleniumTestCase(SeleniumTestCase, StaticLiveServerTestCase):

    def select_option(self, selector, value):
        """
        Select the <OPTION> with the value `value` inside the <SELECT> widget
        identified by the CSS selector `selector`.
        """
        from selenium.webdriver.support.ui import Select
        select = Select(self.selenium.find_element_by_css_selector(selector))
        select.select_by_value(value)

    def deselect_option(self, selector, value):
        """
        Deselect the <OPTION> with the value `value` inside the <SELECT> widget
        identified by the CSS selector `selector`.
        """
        from selenium.webdriver.support.ui import Select
        select = Select(self.selenium.find_element_by_css_selector(selector))
        select.deselect_by_value(value)
```
### 43 - django/contrib/admin/widgets.py:

Start line: 92, End line: 117

```python
class AdminRadioSelect(forms.RadioSelect):
    template_name = 'admin/widgets/radio.html'


class AdminFileWidget(forms.ClearableFileInput):
    template_name = 'admin/widgets/clearable_file_input.html'


def url_params_from_lookup_dict(lookups):
    """
    Convert the type of lookups specified in a ForeignKey limit_choices_to
    attribute to a dictionary of query parameters
    """
    params = {}
    if lookups and hasattr(lookups, 'items'):
        for k, v in lookups.items():
            if callable(v):
                v = v()
            if isinstance(v, (tuple, list)):
                v = ','.join(str(x) for x in v)
            elif isinstance(v, bool):
                v = ('0', '1')[v]
            else:
                v = str(v)
            params[k] = v
    return params
```
### 46 - django/contrib/admin/widgets.py:

Start line: 1, End line: 46

```python
"""
Form Widget classes specific to the Django admin site.
"""
import copy
import json

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language, gettext as _


class FilteredSelectMultiple(forms.SelectMultiple):
    """
    A SelectMultiple with a JavaScript filter interface.

    Note that the resulting JavaScript assumes that the jsi18n
    catalog has been loaded in the page
    """
    class Media:
        js = [
            'admin/js/core.js',
            'admin/js/SelectBox.js',
            'admin/js/SelectFilter2.js',
        ]

    def __init__(self, verbose_name, is_stacked, attrs=None, choices=()):
        self.verbose_name = verbose_name
        self.is_stacked = is_stacked
        super().__init__(attrs, choices)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context['widget']['attrs']['class'] = 'selectfilter'
        if self.is_stacked:
            context['widget']['attrs']['class'] += 'stacked'
        context['widget']['attrs']['data-field-name'] = self.verbose_name
        context['widget']['attrs']['data-is-stacked'] = int(self.is_stacked)
        return context
```
### 53 - django/contrib/admin/widgets.py:

Start line: 420, End line: 447

```python
class AutocompleteMixin:

    def optgroups(self, name, value, attr=None):
        """Return selected options based on the ModelChoiceIterator."""
        default = (None, [], 0)
        groups = [default]
        has_selected = False
        selected_choices = {
            str(v) for v in value
            if str(v) not in self.choices.field.empty_values
        }
        if not self.is_required and not self.allow_multiple_selected:
            default[1].append(self.create_option(name, '', '', False, 0))
        remote_model_opts = self.field.remote_field.model._meta
        to_field_name = getattr(self.field.remote_field, 'field_name', remote_model_opts.pk.attname)
        to_field_name = remote_model_opts.get_field(to_field_name).attname
        choices = (
            (getattr(obj, to_field_name), self.choices.field.label_from_instance(obj))
            for obj in self.choices.queryset.using(self.db).filter(**{'%s__in' % to_field_name: selected_choices})
        )
        for option_value, option_label in choices:
            selected = (
                str(option_value) in value and
                (has_selected is False or self.allow_multiple_selected)
            )
            has_selected |= selected
            index = len(default[1])
            subgroup = default[1]
            subgroup.append(self.create_option(name, option_value, option_label, selected_choices, index))
        return groups
```
### 89 - django/contrib/admin/widgets.py:

Start line: 311, End line: 323

```python
class AdminTextareaWidget(forms.Textarea):
    def __init__(self, attrs=None):
        super().__init__(attrs={'class': 'vLargeTextField', **(attrs or {})})


class AdminTextInputWidget(forms.TextInput):
    def __init__(self, attrs=None):
        super().__init__(attrs={'class': 'vTextField', **(attrs or {})})


class AdminEmailInputWidget(forms.EmailInput):
    def __init__(self, attrs=None):
        super().__init__(attrs={'class': 'vTextField', **(attrs or {})})
```
### 127 - django/contrib/admin/widgets.py:

Start line: 161, End line: 192

```python
class ForeignKeyRawIdWidget(forms.TextInput):

    def base_url_parameters(self):
        limit_choices_to = self.rel.limit_choices_to
        if callable(limit_choices_to):
            limit_choices_to = limit_choices_to()
        return url_params_from_lookup_dict(limit_choices_to)

    def url_parameters(self):
        from django.contrib.admin.views.main import TO_FIELD_VAR
        params = self.base_url_parameters()
        params.update({TO_FIELD_VAR: self.rel.get_related_field().name})
        return params

    def label_and_url_for_value(self, value):
        key = self.rel.get_related_field().name
        try:
            obj = self.rel.model._default_manager.using(self.db).get(**{key: value})
        except (ValueError, self.rel.model.DoesNotExist, ValidationError):
            return '', ''

        try:
            url = reverse(
                '%s:%s_%s_change' % (
                    self.admin_site.name,
                    obj._meta.app_label,
                    obj._meta.object_name.lower(),
                ),
                args=(obj.pk,)
            )
        except NoReverseMatch:
            url = ''  # Admin not registered for target model.

        return Truncator(obj).words(14), url
```
