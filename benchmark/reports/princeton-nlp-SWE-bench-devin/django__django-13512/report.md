# django__django-13512

| **django/django** | `b79088306513d5ed76d31ac40ab3c15f858946ea` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 351 |
| **Any found context length** | 351 |
| **Avg pos** | 23.5 |
| **Min pos** | 1 |
| **Max pos** | 32 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/utils.py b/django/contrib/admin/utils.py
--- a/django/contrib/admin/utils.py
+++ b/django/contrib/admin/utils.py
@@ -1,5 +1,6 @@
 import datetime
 import decimal
+import json
 from collections import defaultdict
 
 from django.core.exceptions import FieldDoesNotExist
@@ -400,7 +401,7 @@ def display_for_field(value, field, empty_value_display):
         return format_html('<a href="{}">{}</a>', value.url, value)
     elif isinstance(field, models.JSONField) and value:
         try:
-            return field.get_prep_value(value)
+            return json.dumps(value, ensure_ascii=False, cls=field.encoder)
         except TypeError:
             return display_for_value(value, empty_value_display)
     else:
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -1258,7 +1258,7 @@ def bound_data(self, data, initial):
     def prepare_value(self, value):
         if isinstance(value, InvalidJSONInput):
             return value
-        return json.dumps(value, cls=self.encoder)
+        return json.dumps(value, ensure_ascii=False, cls=self.encoder)
 
     def has_changed(self, initial, data):
         if super().has_changed(initial, data):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/utils.py | 3 | 3 | 32 | 5 | 9186
| django/contrib/admin/utils.py | 403 | 403 | 14 | 5 | 4248
| django/forms/fields.py | 1261 | 1261 | 1 | 1 | 351


## Problem Statement

```
Admin doesn't display properly unicode chars in JSONFields.
Description
	 
		(last modified by ZhaoQi99)
	 
>>> import json
>>> print json.dumps('中国')
"\u4e2d\u56fd"
json.dumps use ASCII encoding by default when serializing Chinese.
So when we edit a JsonField which contains Chinese character in Django admin,it will appear in ASCII characters.
I have try to fix this this problem in ​https://github.com/adamchainz/django-mysql/pull/714.And it works prefectly.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/forms/fields.py** | 1219 | 1272| 351 | 351 | 9347 | 
| 2 | 2 django/contrib/admin/models.py | 1 | 20| 118 | 469 | 10470 | 
| 3 | 3 django/db/models/fields/json.py | 1 | 40| 273 | 742 | 14554 | 
| 4 | 3 django/db/models/fields/json.py | 42 | 60| 125 | 867 | 14554 | 
| 5 | 3 django/db/models/fields/json.py | 62 | 112| 320 | 1187 | 14554 | 
| 6 | 4 django/contrib/admin/helpers.py | 1 | 32| 215 | 1402 | 17885 | 
| 7 | **5 django/contrib/admin/utils.py** | 493 | 551| 463 | 1865 | 22037 | 
| 8 | 5 django/contrib/admin/models.py | 96 | 134| 376 | 2241 | 22037 | 
| 9 | 6 django/contrib/admin/options.py | 1 | 97| 767 | 3008 | 40624 | 
| 10 | 7 django/db/models/fields/__init__.py | 1039 | 1075| 248 | 3256 | 58995 | 
| 11 | **7 django/forms/fields.py** | 1179 | 1216| 199 | 3455 | 58995 | 
| 12 | 8 django/contrib/admin/widgets.py | 347 | 373| 328 | 3783 | 62789 | 
| 13 | 8 django/db/models/fields/__init__.py | 1002 | 1016| 122 | 3905 | 62789 | 
| **-> 14 <-** | **8 django/contrib/admin/utils.py** | 368 | 407| 343 | 4248 | 62789 | 
| 15 | 9 django/contrib/admin/templatetags/admin_modify.py | 1 | 45| 372 | 4620 | 63757 | 
| 16 | 10 django/contrib/gis/admin/options.py | 52 | 63| 139 | 4759 | 64953 | 
| 17 | 11 django/contrib/postgres/fields/array.py | 1 | 15| 110 | 4869 | 67034 | 
| 18 | 12 django/contrib/gis/serializers/geojson.py | 38 | 68| 280 | 5149 | 67613 | 
| 19 | 12 django/contrib/gis/admin/options.py | 80 | 135| 555 | 5704 | 67613 | 
| 20 | 12 django/contrib/admin/helpers.py | 124 | 150| 220 | 5924 | 67613 | 
| 21 | 12 django/db/models/fields/json.py | 338 | 355| 170 | 6094 | 67613 | 
| 22 | 12 django/db/models/fields/json.py | 254 | 282| 236 | 6330 | 67613 | 
| 23 | 13 django/contrib/postgres/forms/hstore.py | 1 | 59| 339 | 6669 | 67952 | 
| 24 | 13 django/contrib/admin/options.py | 1540 | 1626| 760 | 7429 | 67952 | 
| 25 | 13 django/db/models/fields/__init__.py | 1077 | 1092| 173 | 7602 | 67952 | 
| 26 | 14 django/contrib/postgres/forms/jsonb.py | 1 | 17| 108 | 7710 | 68060 | 
| 27 | 15 django/contrib/postgres/fields/jsonb.py | 1 | 44| 312 | 8022 | 68372 | 
| 28 | 15 django/db/models/fields/__init__.py | 1018 | 1037| 128 | 8150 | 68372 | 
| 29 | 15 django/contrib/admin/options.py | 1654 | 1668| 173 | 8323 | 68372 | 
| 30 | 15 django/db/models/fields/json.py | 284 | 295| 153 | 8476 | 68372 | 
| 31 | 15 django/contrib/admin/options.py | 1127 | 1172| 482 | 8958 | 68372 | 
| **-> 32 <-** | **15 django/contrib/admin/utils.py** | 1 | 24| 228 | 9186 | 68372 | 
| 33 | **15 django/contrib/admin/utils.py** | 410 | 439| 203 | 9389 | 68372 | 
| 34 | 16 django/core/serializers/json.py | 1 | 59| 364 | 9753 | 69072 | 
| 35 | 17 django/core/serializers/jsonl.py | 1 | 39| 254 | 10007 | 69451 | 
| 36 | 18 django/contrib/auth/admin.py | 128 | 189| 465 | 10472 | 71177 | 
| 37 | 19 django/contrib/admin/forms.py | 1 | 31| 185 | 10657 | 71362 | 
| 38 | 19 django/contrib/admin/helpers.py | 383 | 399| 138 | 10795 | 71362 | 
| 39 | 19 django/db/models/fields/json.py | 297 | 310| 184 | 10979 | 71362 | 
| 40 | 20 django/contrib/admin/views/autocomplete.py | 37 | 52| 154 | 11133 | 71754 | 
| 41 | 20 django/contrib/admin/views/autocomplete.py | 1 | 35| 246 | 11379 | 71754 | 
| 42 | 20 django/contrib/admin/widgets.py | 49 | 70| 168 | 11547 | 71754 | 
| 43 | 21 django/contrib/admin/checks.py | 896 | 944| 416 | 11963 | 80891 | 
| 44 | **21 django/forms/fields.py** | 210 | 241| 274 | 12237 | 80891 | 
| 45 | 21 django/contrib/admin/widgets.py | 311 | 323| 114 | 12351 | 80891 | 
| 46 | 21 django/contrib/admin/options.py | 100 | 130| 223 | 12574 | 80891 | 
| 47 | 21 django/contrib/auth/admin.py | 1 | 22| 188 | 12762 | 80891 | 
| 48 | 21 django/contrib/admin/helpers.py | 35 | 69| 230 | 12992 | 80891 | 
| 49 | 21 django/contrib/admin/options.py | 1037 | 1059| 198 | 13190 | 80891 | 
| 50 | 21 django/contrib/admin/checks.py | 430 | 439| 125 | 13315 | 80891 | 
| 51 | 21 django/contrib/admin/options.py | 1627 | 1652| 291 | 13606 | 80891 | 
| 52 | 22 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 14241 | 81571 | 
| 53 | 23 django/contrib/postgres/fields/hstore.py | 1 | 69| 435 | 14676 | 82271 | 
| 54 | 23 django/contrib/admin/checks.py | 220 | 230| 127 | 14803 | 82271 | 
| 55 | 24 django/contrib/admin/views/main.py | 1 | 45| 324 | 15127 | 86667 | 
| 56 | 24 django/db/models/fields/json.py | 371 | 392| 215 | 15342 | 86667 | 
| 57 | 24 django/db/models/fields/json.py | 115 | 140| 204 | 15546 | 86667 | 
| 58 | 24 django/contrib/admin/checks.py | 492 | 502| 149 | 15695 | 86667 | 
| 59 | 24 django/db/models/fields/json.py | 210 | 251| 318 | 16013 | 86667 | 
| 60 | 25 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 16648 | 87347 | 
| 61 | 26 django/db/models/functions/text.py | 42 | 61| 153 | 16801 | 89683 | 
| 62 | 27 django/contrib/gis/db/backends/mysql/schema.py | 1 | 23| 203 | 17004 | 90314 | 
| 63 | 27 django/db/models/fields/__init__.py | 1095 | 1108| 104 | 17108 | 90314 | 
| 64 | 27 django/contrib/admin/helpers.py | 204 | 237| 318 | 17426 | 90314 | 
| 65 | 28 django/db/backends/mysql/schema.py | 51 | 87| 349 | 17775 | 91832 | 
| 66 | 28 django/contrib/admin/options.py | 286 | 375| 641 | 18416 | 91832 | 
| 67 | 28 django/contrib/admin/models.py | 39 | 72| 241 | 18657 | 91832 | 
| 68 | 28 django/contrib/admin/options.py | 1758 | 1840| 750 | 19407 | 91832 | 
| 69 | 28 django/core/serializers/json.py | 62 | 106| 336 | 19743 | 91832 | 
| 70 | 28 django/contrib/gis/admin/options.py | 1 | 50| 394 | 20137 | 91832 | 
| 71 | 29 django/conf/locale/uz/formats.py | 5 | 31| 418 | 20555 | 92295 | 
| 72 | 30 django/conf/locale/ar/formats.py | 5 | 22| 135 | 20690 | 92474 | 
| 73 | 31 django/views/i18n.py | 88 | 191| 702 | 21392 | 95011 | 
| 74 | 31 django/db/models/fields/__init__.py | 2387 | 2437| 339 | 21731 | 95011 | 
| 75 | 32 django/utils/html.py | 78 | 89| 117 | 21848 | 98113 | 
| 76 | 32 django/contrib/admin/checks.py | 613 | 635| 155 | 22003 | 98113 | 
| 77 | 32 django/contrib/admin/checks.py | 177 | 218| 325 | 22328 | 98113 | 
| 78 | 33 django/contrib/postgres/fields/citext.py | 1 | 25| 113 | 22441 | 98227 | 
| 79 | 34 django/contrib/admin/templatetags/admin_list.py | 1 | 25| 175 | 22616 | 101887 | 
| 80 | **34 django/contrib/admin/utils.py** | 261 | 284| 174 | 22790 | 101887 | 
| 81 | 34 django/contrib/auth/admin.py | 25 | 37| 128 | 22918 | 101887 | 
| 82 | 35 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 23170 | 102184 | 
| 83 | 35 django/db/models/fields/__init__.py | 1442 | 1465| 171 | 23341 | 102184 | 
| 84 | 36 django/db/migrations/serializer.py | 198 | 232| 281 | 23622 | 104855 | 
| 85 | 36 django/contrib/admin/checks.py | 724 | 734| 115 | 23737 | 104855 | 
| 86 | 36 django/db/backends/mysql/schema.py | 89 | 99| 138 | 23875 | 104855 | 
| 87 | 36 django/db/models/fields/json.py | 395 | 423| 301 | 24176 | 104855 | 
| 88 | 37 django/conf/global_settings.py | 51 | 150| 1160 | 25336 | 110615 | 
| 89 | 38 django/conf/locale/ko/formats.py | 32 | 50| 385 | 25721 | 111505 | 
| 90 | 38 django/contrib/admin/helpers.py | 323 | 349| 171 | 25892 | 111505 | 
| 91 | 38 django/contrib/admin/options.py | 207 | 218| 135 | 26027 | 111505 | 
| 92 | 39 django/core/serializers/xml_serializer.py | 65 | 91| 219 | 26246 | 115017 | 
| 93 | 39 django/contrib/admin/checks.py | 342 | 368| 221 | 26467 | 115017 | 
| 94 | 39 django/contrib/admin/checks.py | 165 | 175| 123 | 26590 | 115017 | 
| 95 | 39 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 26780 | 115017 | 
| 96 | 40 django/conf/locale/az/formats.py | 5 | 31| 364 | 27144 | 115426 | 
| 97 | 41 django/conf/locale/hu/formats.py | 5 | 31| 305 | 27449 | 115776 | 
| 98 | 42 django/conf/locale/ky/formats.py | 5 | 33| 414 | 27863 | 116235 | 
| 99 | 42 django/db/models/fields/__init__.py | 1637 | 1658| 183 | 28046 | 116235 | 
| 100 | 42 django/contrib/auth/admin.py | 40 | 99| 504 | 28550 | 116235 | 
| 101 | 42 django/contrib/admin/widgets.py | 73 | 89| 145 | 28695 | 116235 | 
| 102 | 42 django/contrib/admin/checks.py | 441 | 462| 191 | 28886 | 116235 | 
| 103 | 42 django/contrib/admin/options.py | 1461 | 1480| 133 | 29019 | 116235 | 
| 104 | 42 django/contrib/admin/checks.py | 638 | 657| 183 | 29202 | 116235 | 
| 105 | 42 django/contrib/admin/checks.py | 232 | 245| 161 | 29363 | 116235 | 
| 106 | 43 django/contrib/auth/forms.py | 403 | 455| 358 | 29721 | 119443 | 
| 107 | 43 django/contrib/admin/checks.py | 146 | 163| 155 | 29876 | 119443 | 
| 108 | 44 django/contrib/admin/filters.py | 1 | 17| 127 | 30003 | 123566 | 
| 109 | 44 django/contrib/admin/options.py | 635 | 652| 128 | 30131 | 123566 | 
| 110 | 44 django/contrib/admin/checks.py | 786 | 807| 190 | 30321 | 123566 | 
| 111 | 44 django/db/models/fields/__init__.py | 1286 | 1299| 157 | 30478 | 123566 | 
| 112 | 45 django/contrib/admindocs/views.py | 250 | 315| 573 | 31051 | 126862 | 
| 113 | 45 django/contrib/admin/checks.py | 464 | 490| 190 | 31241 | 126862 | 
| 114 | 45 django/contrib/admindocs/views.py | 183 | 249| 584 | 31825 | 126862 | 
| 115 | 45 django/contrib/admin/checks.py | 736 | 767| 219 | 32044 | 126862 | 
| 116 | 45 django/contrib/admin/widgets.py | 444 | 472| 203 | 32247 | 126862 | 
| 117 | 46 django/core/serializers/python.py | 62 | 77| 156 | 32403 | 128124 | 
| 118 | 46 django/contrib/admin/options.py | 1842 | 1910| 584 | 32987 | 128124 | 
| 119 | 47 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 33124 | 128261 | 
| 120 | 47 django/contrib/admin/options.py | 2088 | 2140| 451 | 33575 | 128261 | 
| 121 | 48 django/conf/locale/fa/formats.py | 5 | 22| 149 | 33724 | 128454 | 
| 122 | 49 django/core/mail/__init__.py | 90 | 104| 175 | 33899 | 129574 | 
| 123 | 50 django/conf/locale/ru/formats.py | 5 | 31| 367 | 34266 | 129986 | 
| 124 | 50 django/db/models/fields/__init__.py | 1898 | 1975| 567 | 34833 | 129986 | 
| 125 | 50 django/db/models/fields/json.py | 189 | 207| 232 | 35065 | 129986 | 
| 126 | 50 django/contrib/admin/options.py | 611 | 633| 280 | 35345 | 129986 | 
| 127 | 51 django/utils/encoding.py | 102 | 115| 130 | 35475 | 132348 | 
| 128 | 52 django/conf/locale/cs/formats.py | 5 | 41| 600 | 36075 | 132993 | 
| 129 | 53 django/conf/locale/eu/formats.py | 5 | 22| 171 | 36246 | 133209 | 
| 130 | 53 django/contrib/admin/options.py | 1022 | 1035| 169 | 36415 | 133209 | 
| 131 | 54 django/conf/locale/da/formats.py | 5 | 27| 250 | 36665 | 133504 | 
| 132 | 54 django/db/models/fields/json.py | 313 | 335| 181 | 36846 | 133504 | 
| 133 | 54 django/contrib/admin/checks.py | 538 | 561| 230 | 37076 | 133504 | 
| 134 | 55 django/conf/locale/bn/formats.py | 5 | 33| 294 | 37370 | 133842 | 
| 135 | 55 django/db/models/fields/__init__.py | 1395 | 1423| 281 | 37651 | 133842 | 
| 136 | 56 django/conf/locale/ca/formats.py | 5 | 31| 287 | 37938 | 134174 | 
| 137 | 57 django/conf/locale/he/formats.py | 5 | 22| 142 | 38080 | 134360 | 
| 138 | 57 django/db/models/fields/__init__.py | 1515 | 1574| 403 | 38483 | 134360 | 
| 139 | 58 django/forms/utils.py | 1 | 41| 287 | 38770 | 135617 | 
| 140 | 58 django/db/models/fields/__init__.py | 1143 | 1159| 175 | 38945 | 135617 | 
| 141 | 59 django/conf/locale/tg/formats.py | 5 | 33| 402 | 39347 | 136064 | 
| 142 | 59 django/contrib/admin/checks.py | 600 | 611| 128 | 39475 | 136064 | 
| 143 | 60 django/conf/locale/de_CH/formats.py | 5 | 34| 398 | 39873 | 136507 | 
| 144 | 61 django/contrib/gis/admin/__init__.py | 1 | 13| 130 | 40003 | 136637 | 
| 145 | 61 django/db/models/fields/__init__.py | 1491 | 1513| 119 | 40122 | 136637 | 
| 146 | 62 django/conf/locale/fr/formats.py | 5 | 32| 448 | 40570 | 137130 | 
| 147 | 63 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 40975 | 137535 | 
| 148 | 63 django/contrib/gis/db/backends/mysql/schema.py | 25 | 38| 146 | 41121 | 137535 | 
| 149 | 64 django/conf/locale/et/formats.py | 5 | 22| 133 | 41254 | 137712 | 
| 150 | **64 django/contrib/admin/utils.py** | 121 | 156| 303 | 41557 | 137712 | 
| 151 | 64 django/contrib/admin/checks.py | 279 | 292| 135 | 41692 | 137712 | 
| 152 | 64 django/db/models/fields/json.py | 143 | 155| 125 | 41817 | 137712 | 
| 153 | 65 django/conf/locale/hi/formats.py | 5 | 22| 125 | 41942 | 137881 | 
| 154 | 65 django/contrib/admin/options.py | 1952 | 1995| 403 | 42345 | 137881 | 
| 155 | 66 django/core/serializers/base.py | 273 | 298| 218 | 42563 | 140306 | 
| 156 | 67 django/contrib/admin/sites.py | 334 | 354| 182 | 42745 | 144504 | 
| 157 | 67 django/contrib/admin/options.py | 189 | 205| 171 | 42916 | 144504 | 
| 158 | 68 django/conf/locale/nn/formats.py | 5 | 37| 593 | 43509 | 145142 | 
| 159 | 68 django/db/models/fields/__init__.py | 1425 | 1439| 121 | 43630 | 145142 | 
| 160 | 69 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 43768 | 145280 | 
| 161 | 69 django/contrib/admin/sites.py | 1 | 29| 175 | 43943 | 145280 | 
| 162 | 69 django/contrib/admin/widgets.py | 326 | 344| 168 | 44111 | 145280 | 
| 163 | 69 django/contrib/admin/widgets.py | 161 | 192| 243 | 44354 | 145280 | 
| 164 | 69 django/contrib/admin/options.py | 1353 | 1418| 581 | 44935 | 145280 | 
| 165 | 69 django/contrib/admindocs/views.py | 156 | 180| 234 | 45169 | 145280 | 
| 166 | 70 django/db/backends/oracle/schema.py | 1 | 41| 427 | 45596 | 147315 | 
| 167 | 71 django/conf/locale/km/formats.py | 5 | 22| 164 | 45760 | 147523 | 
| 168 | 72 django/conf/locale/fi/formats.py | 5 | 38| 435 | 46195 | 148003 | 
| 169 | 73 django/conf/locale/hr/formats.py | 5 | 43| 742 | 46937 | 148790 | 
| 170 | 74 django/conf/locale/sq/formats.py | 5 | 22| 128 | 47065 | 148962 | 
| 171 | 75 django/conf/locale/gl/formats.py | 5 | 22| 170 | 47235 | 149176 | 
| 172 | 76 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 47645 | 150201 | 
| 173 | 76 django/conf/locale/ko/formats.py | 5 | 31| 460 | 48105 | 150201 | 
| 174 | 76 django/core/serializers/base.py | 232 | 249| 208 | 48313 | 150201 | 
| 175 | 77 django/conf/locale/ig/formats.py | 5 | 33| 388 | 48701 | 150634 | 
| 176 | 78 django/conf/locale/ga/formats.py | 5 | 22| 124 | 48825 | 150802 | 
| 177 | 79 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 49114 | 151107 | 
| 178 | 79 django/contrib/admin/helpers.py | 93 | 121| 249 | 49363 | 151107 | 
| 179 | 80 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 49606 | 155263 | 
| 180 | 80 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 50111 | 155263 | 
| 181 | 80 django/contrib/admin/options.py | 244 | 284| 378 | 50489 | 155263 | 
| 182 | 81 django/conf/locale/lv/formats.py | 5 | 45| 700 | 51189 | 156008 | 
| 183 | 82 django/conf/locale/gd/formats.py | 5 | 22| 144 | 51333 | 156196 | 
| 184 | 83 django/conf/locale/is/formats.py | 5 | 22| 130 | 51463 | 156371 | 
| 185 | 84 django/conf/locale/de/formats.py | 5 | 28| 305 | 51768 | 156721 | 
| 186 | 85 django/contrib/admin/__init__.py | 1 | 25| 245 | 52013 | 156966 | 
| 187 | 85 django/core/serializers/xml_serializer.py | 93 | 114| 192 | 52205 | 156966 | 
| 188 | 85 django/views/i18n.py | 1 | 20| 120 | 52325 | 156966 | 
| 189 | **85 django/contrib/admin/utils.py** | 287 | 305| 175 | 52500 | 156966 | 
| 190 | 85 django/db/backends/mysql/schema.py | 132 | 150| 192 | 52692 | 156966 | 
| 191 | 85 django/contrib/admin/options.py | 1686 | 1757| 653 | 53345 | 156966 | 
| 192 | 86 django/db/backends/mysql/operations.py | 1 | 35| 282 | 53627 | 160625 | 
| 193 | 86 django/contrib/auth/admin.py | 101 | 126| 286 | 53913 | 160625 | 
| 194 | 87 django/conf/locale/sk/formats.py | 5 | 29| 330 | 54243 | 161000 | 
| 195 | 87 django/contrib/admin/models.py | 23 | 36| 111 | 54354 | 161000 | 
| 196 | 88 django/conf/locale/tk/formats.py | 5 | 33| 402 | 54756 | 161447 | 
| 197 | 89 django/conf/locale/mn/formats.py | 5 | 22| 120 | 54876 | 161611 | 
| 198 | **89 django/forms/fields.py** | 1164 | 1176| 113 | 54989 | 161611 | 
| 199 | 90 django/conf/locale/uk/formats.py | 5 | 36| 425 | 55414 | 162081 | 
| 200 | 90 django/contrib/admin/options.py | 551 | 594| 297 | 55711 | 162081 | 


### Hint

```
As far as I'm aware, we cannot use ensure_ascii=False by default because it requires utf8mb4 encoding on MySQL, see #18392. It looks that you can use a custom encoder/decoder to make it works without changes in Django.
Replying to felixxm: As far as I'm aware, we cannot use ensure_ascii=False by default because it requires utf8mb4 encoding on MySQL, see #18392. It looks that you can use a custom encoder/decoder to make it works without changes in Django. No, this function is only used in Django admin's display, so it will not influence any operation about MySQL writing and reading. I just tested it using utf8 encoding on MySQL, and it works perfectly. In my view, If we save non-ASCII characters in a JsonField ,such as emoij,chinese,Japanese.... And when we want to edit it in Django's admin, it is really not good if it displays in ASCII characters. In order to fix this,we need to do many extra things...
We just need to modify this line.​https://github.com/django/django/blob/3d4ffd1ff0eb9343ee41de77caf6ae427b6e873c/django/forms/fields.py#L1261 Then I read the source code of django tests , and It seems that internationalization is not considered.(​https://github.com/django/django/blob/3d4ffd1ff0eb9343ee41de77caf6ae427b6e873c/tests/forms_tests/field_tests/test_jsonfield.py#L29)
No, this function is only used in Django admin's display, so it will not influence any operation about MySQL writing and reading. Sorry I missed that. Good catch. Would you like to provide a patch?
Replying to felixxm: Sorry I missed that. Good catch. Would you like to provide a patch? Yep!
```

## Patch

```diff
diff --git a/django/contrib/admin/utils.py b/django/contrib/admin/utils.py
--- a/django/contrib/admin/utils.py
+++ b/django/contrib/admin/utils.py
@@ -1,5 +1,6 @@
 import datetime
 import decimal
+import json
 from collections import defaultdict
 
 from django.core.exceptions import FieldDoesNotExist
@@ -400,7 +401,7 @@ def display_for_field(value, field, empty_value_display):
         return format_html('<a href="{}">{}</a>', value.url, value)
     elif isinstance(field, models.JSONField) and value:
         try:
-            return field.get_prep_value(value)
+            return json.dumps(value, ensure_ascii=False, cls=field.encoder)
         except TypeError:
             return display_for_value(value, empty_value_display)
     else:
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -1258,7 +1258,7 @@ def bound_data(self, data, initial):
     def prepare_value(self, value):
         if isinstance(value, InvalidJSONInput):
             return value
-        return json.dumps(value, cls=self.encoder)
+        return json.dumps(value, ensure_ascii=False, cls=self.encoder)
 
     def has_changed(self, initial, data):
         if super().has_changed(initial, data):

```

## Test Patch

```diff
diff --git a/tests/admin_utils/tests.py b/tests/admin_utils/tests.py
--- a/tests/admin_utils/tests.py
+++ b/tests/admin_utils/tests.py
@@ -186,6 +186,7 @@ def test_json_display_for_field(self):
             ({'a': {'b': 'c'}}, '{"a": {"b": "c"}}'),
             (['a', 'b'], '["a", "b"]'),
             ('a', '"a"'),
+            ({'a': '你好 世界'}, '{"a": "你好 世界"}'),
             ({('a', 'b'): 'c'}, "{('a', 'b'): 'c'}"),  # Invalid JSON.
         ]
         for value, display_value in tests:
diff --git a/tests/forms_tests/field_tests/test_jsonfield.py b/tests/forms_tests/field_tests/test_jsonfield.py
--- a/tests/forms_tests/field_tests/test_jsonfield.py
+++ b/tests/forms_tests/field_tests/test_jsonfield.py
@@ -29,6 +29,12 @@ def test_prepare_value(self):
         self.assertEqual(field.prepare_value({'a': 'b'}), '{"a": "b"}')
         self.assertEqual(field.prepare_value(None), 'null')
         self.assertEqual(field.prepare_value('foo'), '"foo"')
+        self.assertEqual(field.prepare_value('你好，世界'), '"你好，世界"')
+        self.assertEqual(field.prepare_value({'a': '😀🐱'}), '{"a": "😀🐱"}')
+        self.assertEqual(
+            field.prepare_value(["你好，世界", "jaźń"]),
+            '["你好，世界", "jaźń"]',
+        )
 
     def test_widget(self):
         field = JSONField()

```


## Code snippets

### 1 - django/forms/fields.py:

Start line: 1219, End line: 1272

```python
class JSONField(CharField):
    default_error_messages = {
        'invalid': _('Enter a valid JSON.'),
    }
    widget = Textarea

    def __init__(self, encoder=None, decoder=None, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(**kwargs)

    def to_python(self, value):
        if self.disabled:
            return value
        if value in self.empty_values:
            return None
        elif isinstance(value, (list, dict, int, float, JSONString)):
            return value
        try:
            converted = json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            raise ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )
        if isinstance(converted, str):
            return JSONString(converted)
        else:
            return converted

    def bound_data(self, data, initial):
        if self.disabled:
            return initial
        try:
            return json.loads(data, cls=self.decoder)
        except json.JSONDecodeError:
            return InvalidJSONInput(data)

    def prepare_value(self, value):
        if isinstance(value, InvalidJSONInput):
            return value
        return json.dumps(value, cls=self.encoder)

    def has_changed(self, initial, data):
        if super().has_changed(initial, data):
            return True
        # For purposes of seeing whether something has changed, True isn't the
        # same as 1 and the order of keys doesn't matter.
        return (
            json.dumps(initial, sort_keys=True, cls=self.encoder) !=
            json.dumps(self.to_python(data), sort_keys=True, cls=self.encoder)
        )
```
### 2 - django/contrib/admin/models.py:

Start line: 1, End line: 20

```python
import json

from django.conf import settings
from django.contrib.admin.utils import quote
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.urls import NoReverseMatch, reverse
from django.utils import timezone
from django.utils.text import get_text_list
from django.utils.translation import gettext, gettext_lazy as _

ADDITION = 1
CHANGE = 2
DELETION = 3

ACTION_FLAG_CHOICES = (
    (ADDITION, _('Addition')),
    (CHANGE, _('Change')),
    (DELETION, _('Deletion')),
)
```
### 3 - django/db/models/fields/json.py:

Start line: 1, End line: 40

```python
import json

from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import lookups
from django.db.models.lookups import PostgresOperatorLookup, Transform
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import CheckFieldDefaultMixin

__all__ = ['JSONField']


class JSONField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    description = _('A JSON object')
    default_error_messages = {
        'invalid': _('Value must be valid JSON.'),
    }
    _default_hint = ('dict', '{}')

    def __init__(
        self, verbose_name=None, name=None, encoder=None, decoder=None,
        **kwargs,
    ):
        if encoder and not callable(encoder):
            raise ValueError('The encoder parameter must be a callable object.')
        if decoder and not callable(decoder):
            raise ValueError('The decoder parameter must be a callable object.')
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        databases = kwargs.get('databases') or []
        errors.extend(self._check_supported(databases))
        return errors
```
### 4 - django/db/models/fields/json.py:

Start line: 42, End line: 60

```python
class JSONField(CheckFieldDefaultMixin, Field):

    def _check_supported(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (
                'supports_json_field' in self.model._meta.required_db_features or
                connection.features.supports_json_field
            ):
                errors.append(
                    checks.Error(
                        '%s does not support JSONFields.'
                        % connection.display_name,
                        obj=self.model,
                        id='fields.E180',
                    )
                )
        return errors
```
### 5 - django/db/models/fields/json.py:

Start line: 62, End line: 112

```python
class JSONField(CheckFieldDefaultMixin, Field):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.encoder is not None:
            kwargs['encoder'] = self.encoder
        if self.decoder is not None:
            kwargs['decoder'] = self.decoder
        return name, path, args, kwargs

    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        try:
            return json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            return value

    def get_internal_type(self):
        return 'JSONField'

    def get_prep_value(self, value):
        if value is None:
            return value
        return json.dumps(value, cls=self.encoder)

    def get_transform(self, name):
        transform = super().get_transform(name)
        if transform:
            return transform
        return KeyTransformFactory(name)

    def validate(self, value, model_instance):
        super().validate(value, model_instance)
        try:
            json.dumps(value, cls=self.encoder)
        except TypeError:
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def value_to_string(self, obj):
        return self.value_from_object(obj)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.JSONField,
            'encoder': self.encoder,
            'decoder': self.decoder,
            **kwargs,
        })
```
### 6 - django/contrib/admin/helpers.py:

Start line: 1, End line: 32

```python
import json

from django import forms
from django.contrib.admin.utils import (
    display_for_field, flatten_fieldsets, help_text_for_field, label_for_field,
    lookup_field, quote,
)
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
    ForeignObjectRel, ManyToManyRel, OneToOneField,
)
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext, gettext_lazy as _

ACTION_CHECKBOX_NAME = '_selected_action'


class ActionForm(forms.Form):
    action = forms.ChoiceField(label=_('Action:'))
    select_across = forms.BooleanField(
        label='',
        required=False,
        initial=0,
        widget=forms.HiddenInput({'class': 'select-across'}),
    )


checkbox = forms.CheckboxInput({'class': 'action-select'}, lambda value: False)
```
### 7 - django/contrib/admin/utils.py:

Start line: 493, End line: 551

```python
def construct_change_message(form, formsets, add):
    """
    Construct a JSON structure describing changes from a changed object.
    Translations are deactivated so that strings are stored untranslated.
    Translation happens later on LogEntry access.
    """
    # Evaluating `form.changed_data` prior to disabling translations is required
    # to avoid fields affected by localization from being included incorrectly,
    # e.g. where date formats differ such as MM/DD/YYYY vs DD/MM/YYYY.
    changed_data = form.changed_data
    with translation_override(None):
        # Deactivate translations while fetching verbose_name for form
        # field labels and using `field_name`, if verbose_name is not provided.
        # Translations will happen later on LogEntry access.
        changed_field_labels = _get_changed_field_labels_from_form(form, changed_data)

    change_message = []
    if add:
        change_message.append({'added': {}})
    elif form.changed_data:
        change_message.append({'changed': {'fields': changed_field_labels}})
    if formsets:
        with translation_override(None):
            for formset in formsets:
                for added_object in formset.new_objects:
                    change_message.append({
                        'added': {
                            'name': str(added_object._meta.verbose_name),
                            'object': str(added_object),
                        }
                    })
                for changed_object, changed_fields in formset.changed_objects:
                    change_message.append({
                        'changed': {
                            'name': str(changed_object._meta.verbose_name),
                            'object': str(changed_object),
                            'fields': _get_changed_field_labels_from_form(formset.forms[0], changed_fields),
                        }
                    })
                for deleted_object in formset.deleted_objects:
                    change_message.append({
                        'deleted': {
                            'name': str(deleted_object._meta.verbose_name),
                            'object': str(deleted_object),
                        }
                    })
    return change_message


def _get_changed_field_labels_from_form(form, changed_data):
    changed_field_labels = []
    for field_name in changed_data:
        try:
            verbose_field_name = form.fields[field_name].label or field_name
        except KeyError:
            verbose_field_name = field_name
        changed_field_labels.append(str(verbose_field_name))
    return changed_field_labels
```
### 8 - django/contrib/admin/models.py:

Start line: 96, End line: 134

```python
class LogEntry(models.Model):

    def get_change_message(self):
        """
        If self.change_message is a JSON structure, interpret it as a change
        string, properly translated.
        """
        if self.change_message and self.change_message[0] == '[':
            try:
                change_message = json.loads(self.change_message)
            except json.JSONDecodeError:
                return self.change_message
            messages = []
            for sub_message in change_message:
                if 'added' in sub_message:
                    if sub_message['added']:
                        sub_message['added']['name'] = gettext(sub_message['added']['name'])
                        messages.append(gettext('Added {name} “{object}”.').format(**sub_message['added']))
                    else:
                        messages.append(gettext('Added.'))

                elif 'changed' in sub_message:
                    sub_message['changed']['fields'] = get_text_list(
                        [gettext(field_name) for field_name in sub_message['changed']['fields']], gettext('and')
                    )
                    if 'name' in sub_message['changed']:
                        sub_message['changed']['name'] = gettext(sub_message['changed']['name'])
                        messages.append(gettext('Changed {fields} for {name} “{object}”.').format(
                            **sub_message['changed']
                        ))
                    else:
                        messages.append(gettext('Changed {fields}.').format(**sub_message['changed']))

                elif 'deleted' in sub_message:
                    sub_message['deleted']['name'] = gettext(sub_message['deleted']['name'])
                    messages.append(gettext('Deleted {name} “{object}”.').format(**sub_message['deleted']))

            change_message = ' '.join(msg[0].upper() + msg[1:] for msg in messages)
            return change_message or gettext('No fields changed.')
        else:
            return self.change_message
```
### 9 - django/contrib/admin/options.py:

Start line: 1, End line: 97

```python
import copy
import json
import operator
import re
from functools import partial, reduce, update_wrapper
from urllib.parse import quote as urlquote

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
    BaseModelAdminChecks, InlineModelAdminChecks, ModelAdminChecks,
)
from django.contrib.admin.exceptions import DisallowedModelAdminToField
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
    NestedObjects, construct_change_message, flatten_fieldsets,
    get_deleted_objects, lookup_needs_distinct, model_format_dict,
    model_ngettext, quote, unquote,
)
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.admin.widgets import (
    AutocompleteSelect, AutocompleteSelectMultiple,
)
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
    FieldDoesNotExist, FieldError, PermissionDenied, ValidationError,
)
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
    BaseInlineFormSet, inlineformset_factory, modelform_defines_fields,
    modelform_factory, modelformset_factory,
)
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
    capfirst, format_lazy, get_text_list, smart_split, unescape_string_literal,
)
from django.utils.translation import gettext as _, ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView

IS_POPUP_VAR = '_popup'
TO_FIELD_VAR = '_to_field'


HORIZONTAL, VERTICAL = 1, 2


def get_content_type_for_model(obj):
    # Since this module gets imported in the application's root package,
    # it cannot import models from other applications at the module level.
    from django.contrib.contenttypes.models import ContentType
    return ContentType.objects.get_for_model(obj, for_concrete_model=False)


def get_ul_class(radio_style):
    return 'radiolist' if radio_style == VERTICAL else 'radiolist inline'


class IncorrectLookupParameters(Exception):
    pass


# Defaults for formfield_overrides. ModelAdmin subclasses can change this
# by adding to ModelAdmin.formfield_overrides.

FORMFIELD_FOR_DBFIELD_DEFAULTS = {
    models.DateTimeField: {
        'form_class': forms.SplitDateTimeField,
        'widget': widgets.AdminSplitDateTime
    },
    models.DateField: {'widget': widgets.AdminDateWidget},
    models.TimeField: {'widget': widgets.AdminTimeWidget},
    models.TextField: {'widget': widgets.AdminTextareaWidget},
    models.URLField: {'widget': widgets.AdminURLFieldWidget},
    models.IntegerField: {'widget': widgets.AdminIntegerFieldWidget},
    models.BigIntegerField: {'widget': widgets.AdminBigIntegerFieldWidget},
    models.CharField: {'widget': widgets.AdminTextInputWidget},
    models.ImageField: {'widget': widgets.AdminFileWidget},
    models.FileField: {'widget': widgets.AdminFileWidget},
    models.EmailField: {'widget': widgets.AdminEmailInputWidget},
    models.UUIDField: {'widget': widgets.AdminUUIDInputWidget},
}

csrf_protect_m = method_decorator(csrf_protect)
```
### 10 - django/db/models/fields/__init__.py:

Start line: 1039, End line: 1075

```python
class CharField(Field):

    def _check_db_collation(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (
                self.db_collation is None or
                'supports_collation_on_charfield' in self.model._meta.required_db_features or
                connection.features.supports_collation_on_charfield
            ):
                errors.append(
                    checks.Error(
                        '%s does not support a database collation on '
                        'CharFields.' % connection.display_name,
                        obj=self,
                        id='fields.E190',
                    ),
                )
        return errors

    def cast_db_type(self, connection):
        if self.max_length is None:
            return connection.ops.cast_char_field_without_max_length
        return super().cast_db_type(connection)

    def get_internal_type(self):
        return "CharField"

    def to_python(self, value):
        if isinstance(value, str) or value is None:
            return value
        return str(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)
```
### 11 - django/forms/fields.py:

Start line: 1179, End line: 1216

```python
class SlugField(CharField):
    default_validators = [validators.validate_slug]

    def __init__(self, *, allow_unicode=False, **kwargs):
        self.allow_unicode = allow_unicode
        if self.allow_unicode:
            self.default_validators = [validators.validate_unicode_slug]
        super().__init__(**kwargs)


class UUIDField(CharField):
    default_error_messages = {
        'invalid': _('Enter a valid UUID.'),
    }

    def prepare_value(self, value):
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    def to_python(self, value):
        value = super().to_python(value)
        if value in self.empty_values:
            return None
        if not isinstance(value, uuid.UUID):
            try:
                value = uuid.UUID(value)
            except ValueError:
                raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value


class InvalidJSONInput(str):
    pass


class JSONString(str):
    pass
```
### 14 - django/contrib/admin/utils.py:

Start line: 368, End line: 407

```python
def help_text_for_field(name, model):
    help_text = ""
    try:
        field = _get_non_gfk_field(model._meta, name)
    except (FieldDoesNotExist, FieldIsAForeignKeyColumnName):
        pass
    else:
        if hasattr(field, 'help_text'):
            help_text = field.help_text
    return help_text


def display_for_field(value, field, empty_value_display):
    from django.contrib.admin.templatetags.admin_list import _boolean_icon

    if getattr(field, 'flatchoices', None):
        return dict(field.flatchoices).get(value, empty_value_display)
    # BooleanField needs special-case null-handling, so it comes before the
    # general null test.
    elif isinstance(field, models.BooleanField):
        return _boolean_icon(value)
    elif value is None:
        return empty_value_display
    elif isinstance(field, models.DateTimeField):
        return formats.localize(timezone.template_localtime(value))
    elif isinstance(field, (models.DateField, models.TimeField)):
        return formats.localize(value)
    elif isinstance(field, models.DecimalField):
        return formats.number_format(value, field.decimal_places)
    elif isinstance(field, (models.IntegerField, models.FloatField)):
        return formats.number_format(value)
    elif isinstance(field, models.FileField) and value:
        return format_html('<a href="{}">{}</a>', value.url, value)
    elif isinstance(field, models.JSONField) and value:
        try:
            return field.get_prep_value(value)
        except TypeError:
            return display_for_value(value, empty_value_display)
    else:
        return display_for_value(value, empty_value_display)
```
### 32 - django/contrib/admin/utils.py:

Start line: 1, End line: 24

```python
import datetime
import decimal
from collections import defaultdict

from django.core.exceptions import FieldDoesNotExist
from django.db import models, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import Collector
from django.forms.utils import pretty_name
from django.urls import NoReverseMatch, reverse
from django.utils import formats, timezone
from django.utils.html import format_html
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import capfirst
from django.utils.translation import ngettext, override as translation_override

QUOTE_MAP = {i: '_%02X' % i for i in b'":/_#?;@&=+$,"[]<>%\n\\'}
UNQUOTE_MAP = {v: chr(k) for k, v in QUOTE_MAP.items()}
UNQUOTE_RE = _lazy_re_compile('_(?:%s)' % '|'.join([x[1:] for x in UNQUOTE_MAP]))


class FieldIsAForeignKeyColumnName(Exception):
    """A field is a foreign key attname, i.e. <FK>_id."""
    pass
```
### 33 - django/contrib/admin/utils.py:

Start line: 410, End line: 439

```python
def display_for_value(value, empty_value_display, boolean=False):
    from django.contrib.admin.templatetags.admin_list import _boolean_icon

    if boolean:
        return _boolean_icon(value)
    elif value is None:
        return empty_value_display
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, datetime.datetime):
        return formats.localize(timezone.template_localtime(value))
    elif isinstance(value, (datetime.date, datetime.time)):
        return formats.localize(value)
    elif isinstance(value, (int, decimal.Decimal, float)):
        return formats.number_format(value)
    elif isinstance(value, (list, tuple)):
        return ', '.join(str(v) for v in value)
    else:
        return str(value)


class NotRelationField(Exception):
    pass


def get_model_from_relation(field):
    if hasattr(field, 'get_path_info'):
        return field.get_path_info()[-1].to_opts.model
    else:
        raise NotRelationField
```
### 44 - django/forms/fields.py:

Start line: 210, End line: 241

```python
class CharField(Field):
    def __init__(self, *, max_length=None, min_length=None, strip=True, empty_value='', **kwargs):
        self.max_length = max_length
        self.min_length = min_length
        self.strip = strip
        self.empty_value = empty_value
        super().__init__(**kwargs)
        if min_length is not None:
            self.validators.append(validators.MinLengthValidator(int(min_length)))
        if max_length is not None:
            self.validators.append(validators.MaxLengthValidator(int(max_length)))
        self.validators.append(validators.ProhibitNullCharactersValidator())

    def to_python(self, value):
        """Return a string."""
        if value not in self.empty_values:
            value = str(value)
            if self.strip:
                value = value.strip()
        if value in self.empty_values:
            return self.empty_value
        return value

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if self.max_length is not None and not widget.is_hidden:
            # The HTML attribute is maxlength, not max_length.
            attrs['maxlength'] = str(self.max_length)
        if self.min_length is not None and not widget.is_hidden:
            # The HTML attribute is minlength, not min_length.
            attrs['minlength'] = str(self.min_length)
        return attrs
```
### 80 - django/contrib/admin/utils.py:

Start line: 261, End line: 284

```python
def lookup_field(name, obj, model_admin=None):
    opts = obj._meta
    try:
        f = _get_non_gfk_field(opts, name)
    except (FieldDoesNotExist, FieldIsAForeignKeyColumnName):
        # For non-field values, the value is either a method, property or
        # returned via a callable.
        if callable(name):
            attr = name
            value = attr(obj)
        elif hasattr(model_admin, name) and name != '__str__':
            attr = getattr(model_admin, name)
            value = attr(obj)
        else:
            attr = getattr(obj, name)
            if callable(attr):
                value = attr()
            else:
                value = attr
        f = None
    else:
        attr = None
        value = getattr(obj, name)
    return f, attr, value
```
### 150 - django/contrib/admin/utils.py:

Start line: 121, End line: 156

```python
def get_deleted_objects(objs, request, admin_site):
    # ... other code

    def format_callback(obj):
        model = obj.__class__
        has_admin = model in admin_site._registry
        opts = obj._meta

        no_edit_link = '%s: %s' % (capfirst(opts.verbose_name), obj)

        if has_admin:
            if not admin_site._registry[model].has_delete_permission(request, obj):
                perms_needed.add(opts.verbose_name)
            try:
                admin_url = reverse('%s:%s_%s_change'
                                    % (admin_site.name,
                                       opts.app_label,
                                       opts.model_name),
                                    None, (quote(obj.pk),))
            except NoReverseMatch:
                # Change url doesn't exist -- don't display link to edit
                return no_edit_link

            # Display a link to the admin page.
            return format_html('{}: <a href="{}">{}</a>',
                               capfirst(opts.verbose_name),
                               admin_url,
                               obj)
        else:
            # Don't display link to edit, because it either has no
            # admin or is edited inline.
            return no_edit_link

    to_delete = collector.nested(format_callback)

    protected = [format_callback(obj) for obj in collector.protected]
    model_count = {model._meta.verbose_name_plural: len(objs) for model, objs in collector.model_objs.items()}

    return to_delete, model_count, perms_needed, protected
```
### 189 - django/contrib/admin/utils.py:

Start line: 287, End line: 305

```python
def _get_non_gfk_field(opts, name):
    """
    For historical reasons, the admin app relies on GenericForeignKeys as being
    "not found" by get_field(). This could likely be cleaned up.

    Reverse relations should also be excluded as these aren't attributes of the
    model (rather something like `foo_set`).
    """
    field = opts.get_field(name)
    if (field.is_relation and
            # Generic foreign keys OR reverse relations
            ((field.many_to_one and not field.related_model) or field.one_to_many)):
        raise FieldDoesNotExist()

    # Avoid coercing <FK>_id fields to FK
    if field.is_relation and not field.many_to_many and hasattr(field, 'attname') and field.attname == name:
        raise FieldIsAForeignKeyColumnName()

    return field
```
### 198 - django/forms/fields.py:

Start line: 1164, End line: 1176

```python
class GenericIPAddressField(CharField):
    def __init__(self, *, protocol='both', unpack_ipv4=False, **kwargs):
        self.unpack_ipv4 = unpack_ipv4
        self.default_validators = validators.ip_address_validators(protocol, unpack_ipv4)[0]
        super().__init__(**kwargs)

    def to_python(self, value):
        if value in self.empty_values:
            return ''
        value = value.strip()
        if value and ':' in value:
            return clean_ipv6_address(value, self.unpack_ipv4)
        return value
```
