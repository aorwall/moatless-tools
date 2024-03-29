# django__django-11893

| **django/django** | `0f843fdd5b9b2f2307148465cd60f4e1b2befbb4` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 54427 |
| **Any found context length** | 356 |
| **Avg pos** | 171.0 |
| **Min pos** | 2 |
| **Max pos** | 169 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -25,7 +25,7 @@
     URLInput,
 )
 from django.utils import formats
-from django.utils.dateparse import parse_duration
+from django.utils.dateparse import parse_datetime, parse_duration
 from django.utils.duration import duration_string
 from django.utils.ipv6 import clean_ipv6_address
 from django.utils.regex_helper import _lazy_re_compile
@@ -459,7 +459,12 @@ def to_python(self, value):
         if isinstance(value, datetime.date):
             result = datetime.datetime(value.year, value.month, value.day)
             return from_current_timezone(result)
-        result = super().to_python(value)
+        try:
+            result = parse_datetime(value.strip())
+        except ValueError:
+            raise ValidationError(self.error_messages['invalid'], code='invalid')
+        if not result:
+            result = super().to_python(value)
         return from_current_timezone(result)
 
     def strptime(self, value, format):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/fields.py | 28 | 28 | 169 | 2 | 54427
| django/forms/fields.py | 462 | 462 | 2 | 2 | 356


## Problem Statement

```
DateTimeField doesn't accept ISO 8601 formatted date string
Description
	
DateTimeField doesn't accept ISO 8601 formatted date string. Differene is that ISO format allows date and time separator to be capital T letter. (Format being YYYY-MM-DDTHH:MM:SS. Django expects to have only space as a date and time separator.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/__init__.py | 1227 | 1240| 157 | 157 | 17512 | 
| **-> 2 <-** | **2 django/forms/fields.py** | 438 | 466| 199 | 356 | 26459 | 
| 3 | 3 django/utils/dateformat.py | 209 | 280| 571 | 927 | 29143 | 
| 4 | 3 django/db/models/fields/__init__.py | 1285 | 1334| 342 | 1269 | 29143 | 
| 5 | 3 django/db/models/fields/__init__.py | 2063 | 2079| 185 | 1454 | 29143 | 
| 6 | 3 django/utils/dateformat.py | 282 | 293| 121 | 1575 | 29143 | 
| 7 | 3 django/utils/dateformat.py | 295 | 350| 452 | 2027 | 29143 | 
| 8 | 4 django/conf/locale/hr/formats.py | 5 | 21| 218 | 2245 | 30026 | 
| 9 | 4 django/db/models/fields/__init__.py | 1084 | 1100| 175 | 2420 | 30026 | 
| 10 | 5 django/conf/locale/is/formats.py | 5 | 22| 130 | 2550 | 30201 | 
| 11 | 6 django/conf/locale/et/formats.py | 5 | 22| 133 | 2683 | 30378 | 
| 12 | 7 django/conf/locale/cs/formats.py | 5 | 43| 640 | 3323 | 31063 | 
| 13 | 8 django/conf/locale/pt/formats.py | 5 | 39| 630 | 3953 | 31738 | 
| 14 | 9 django/conf/locale/hi/formats.py | 5 | 22| 125 | 4078 | 31907 | 
| 15 | 10 django/conf/locale/ta/formats.py | 5 | 22| 125 | 4203 | 32076 | 
| 16 | 11 django/conf/locale/eo/formats.py | 5 | 50| 742 | 4945 | 32863 | 
| 17 | 12 django/conf/locale/cy/formats.py | 5 | 36| 582 | 5527 | 33490 | 
| 18 | 12 django/db/models/fields/__init__.py | 1336 | 1364| 281 | 5808 | 33490 | 
| 19 | 13 django/conf/locale/it/formats.py | 21 | 46| 564 | 6372 | 34387 | 
| 20 | 14 django/conf/locale/da/formats.py | 5 | 27| 250 | 6622 | 34682 | 
| 21 | 15 django/conf/locale/te/formats.py | 5 | 22| 123 | 6745 | 34849 | 
| 22 | 16 django/conf/locale/id/formats.py | 5 | 50| 708 | 7453 | 35602 | 
| 23 | 16 django/conf/locale/hr/formats.py | 22 | 48| 620 | 8073 | 35602 | 
| 24 | 17 django/conf/locale/fi/formats.py | 5 | 40| 470 | 8543 | 36117 | 
| 25 | 18 django/conf/locale/nn/formats.py | 5 | 41| 664 | 9207 | 36826 | 
| 26 | 19 django/conf/locale/he/formats.py | 5 | 22| 142 | 9349 | 37012 | 
| 27 | 20 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 9638 | 37317 | 
| 28 | 21 django/conf/locale/pt_BR/formats.py | 5 | 34| 494 | 10132 | 37856 | 
| 29 | 22 django/conf/locale/sv/formats.py | 5 | 39| 534 | 10666 | 38435 | 
| 30 | 23 django/conf/locale/fr/formats.py | 5 | 34| 489 | 11155 | 38969 | 
| 31 | 24 django/conf/locale/en/formats.py | 5 | 41| 663 | 11818 | 39677 | 
| 32 | 25 django/conf/locale/bg/formats.py | 5 | 22| 131 | 11949 | 39852 | 
| 33 | 26 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 12219 | 40138 | 
| 34 | 26 django/db/models/fields/__init__.py | 1366 | 1380| 121 | 12340 | 40138 | 
| 35 | 27 django/conf/locale/lt/formats.py | 5 | 46| 711 | 13051 | 40894 | 
| 36 | 28 django/conf/locale/ru/formats.py | 5 | 33| 402 | 13453 | 41341 | 
| 37 | 29 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 13869 | 41802 | 
| 38 | 30 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 14504 | 42482 | 
| 39 | 31 django/conf/locale/az/formats.py | 5 | 33| 399 | 14903 | 42926 | 
| 40 | 32 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 15538 | 43606 | 
| 41 | 33 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 16246 | 44359 | 
| 42 | 34 django/conf/locale/lv/formats.py | 5 | 47| 735 | 16981 | 45139 | 
| 43 | 35 django/conf/locale/ml/formats.py | 5 | 41| 663 | 17644 | 45847 | 
| 44 | 36 django/conf/locale/de/formats.py | 5 | 29| 323 | 17967 | 46215 | 
| 45 | 37 django/conf/locale/bs/formats.py | 5 | 22| 139 | 18106 | 46398 | 
| 46 | 38 django/conf/locale/fa/formats.py | 5 | 22| 149 | 18255 | 46591 | 
| 47 | **38 django/forms/fields.py** | 1122 | 1155| 293 | 18548 | 46591 | 
| 48 | 39 django/conf/locale/ar/formats.py | 5 | 22| 135 | 18683 | 46770 | 
| 49 | 40 django/conf/locale/el/formats.py | 5 | 36| 508 | 19191 | 47323 | 
| 50 | 41 django/conf/locale/sl/formats.py | 5 | 20| 208 | 19399 | 48172 | 
| 51 | 42 django/conf/locale/uz/formats.py | 5 | 33| 461 | 19860 | 48678 | 
| 52 | 43 django/conf/locale/mk/formats.py | 5 | 43| 672 | 20532 | 49395 | 
| 53 | 44 django/conf/locale/sr/formats.py | 23 | 44| 511 | 21043 | 50244 | 
| 54 | 45 django/conf/locale/gd/formats.py | 5 | 22| 144 | 21187 | 50432 | 
| 55 | 46 django/conf/locale/uk/formats.py | 5 | 38| 460 | 21647 | 50937 | 
| 56 | 47 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 22355 | 51690 | 
| 57 | 48 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 22630 | 52010 | 
| 58 | 49 django/conf/locale/sk/formats.py | 5 | 30| 348 | 22978 | 52403 | 
| 59 | **49 django/forms/fields.py** | 392 | 413| 144 | 23122 | 52403 | 
| 60 | 50 django/conf/locale/sr_Latn/formats.py | 23 | 44| 511 | 23633 | 53252 | 
| 61 | 50 django/db/models/fields/__init__.py | 2124 | 2164| 289 | 23922 | 53252 | 
| 62 | 51 django/conf/locale/ro/formats.py | 5 | 36| 262 | 24184 | 53559 | 
| 63 | 52 django/conf/locale/es/formats.py | 5 | 31| 285 | 24469 | 53889 | 
| 64 | **52 django/forms/fields.py** | 416 | 435| 131 | 24600 | 53889 | 
| 65 | 53 django/utils/datetime_safe.py | 74 | 107| 313 | 24913 | 54665 | 
| 66 | 53 django/conf/locale/sr/formats.py | 5 | 22| 293 | 25206 | 54665 | 
| 67 | 54 django/conf/locale/hu/formats.py | 5 | 32| 323 | 25529 | 55033 | 
| 68 | 55 django/conf/locale/gl/formats.py | 5 | 22| 170 | 25699 | 55247 | 
| 69 | 56 django/conf/locale/nb/formats.py | 5 | 40| 646 | 26345 | 55938 | 
| 70 | 57 django/conf/locale/ka/formats.py | 23 | 48| 564 | 26909 | 56845 | 
| 71 | 58 django/conf/locale/ko/formats.py | 32 | 53| 438 | 27347 | 57788 | 
| 72 | 58 django/conf/locale/sl/formats.py | 22 | 48| 596 | 27943 | 57788 | 
| 73 | 58 django/conf/locale/ko/formats.py | 5 | 31| 460 | 28403 | 57788 | 
| 74 | 59 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 28655 | 58056 | 
| 75 | 59 django/conf/locale/it/formats.py | 5 | 20| 288 | 28943 | 58056 | 
| 76 | 60 django/conf/locale/ca/formats.py | 5 | 31| 287 | 29230 | 58388 | 
| 77 | 61 django/conf/locale/sq/formats.py | 5 | 22| 128 | 29358 | 58560 | 
| 78 | 62 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 29620 | 58838 | 
| 79 | 63 django/conf/locale/bn/formats.py | 5 | 33| 294 | 29914 | 59176 | 
| 80 | 64 django/conf/locale/ja/formats.py | 5 | 22| 149 | 30063 | 59369 | 
| 81 | 65 django/conf/locale/pl/formats.py | 5 | 30| 339 | 30402 | 59753 | 
| 82 | 65 django/conf/locale/sr_Latn/formats.py | 5 | 22| 293 | 30695 | 59753 | 
| 83 | 65 django/conf/locale/ka/formats.py | 5 | 22| 298 | 30993 | 59753 | 
| 84 | 66 django/conf/locale/nl/formats.py | 5 | 71| 479 | 31472 | 61784 | 
| 85 | 67 django/conf/locale/kn/formats.py | 5 | 22| 123 | 31595 | 61951 | 
| 86 | 68 django/conf/locale/ga/formats.py | 5 | 22| 124 | 31719 | 62119 | 
| 87 | 68 django/db/models/fields/__init__.py | 1142 | 1184| 287 | 32006 | 62119 | 
| 88 | 68 django/utils/dateformat.py | 47 | 123| 520 | 32526 | 62119 | 
| 89 | 68 django/utils/dateformat.py | 154 | 181| 203 | 32729 | 62119 | 
| 90 | 68 django/db/models/fields/__init__.py | 1206 | 1224| 149 | 32878 | 62119 | 
| 91 | 69 django/conf/locale/eu/formats.py | 5 | 22| 171 | 33049 | 62335 | 
| 92 | 70 django/conf/locale/vi/formats.py | 5 | 22| 179 | 33228 | 62558 | 
| 93 | 71 django/conf/locale/km/formats.py | 5 | 22| 164 | 33392 | 62766 | 
| 94 | 72 django/conf/locale/tr/formats.py | 5 | 30| 337 | 33729 | 63148 | 
| 95 | 73 django/conf/locale/ar_DZ/formats.py | 5 | 22| 153 | 33882 | 63346 | 
| 96 | 74 django/conf/locale/mn/formats.py | 5 | 22| 120 | 34002 | 63510 | 
| 97 | 74 django/db/models/fields/__init__.py | 2166 | 2192| 213 | 34215 | 63510 | 
| 98 | 75 django/db/models/functions/datetime.py | 87 | 180| 548 | 34763 | 66062 | 
| 99 | **75 django/forms/fields.py** | 371 | 389| 134 | 34897 | 66062 | 
| 100 | 76 django/contrib/gis/gdal/field.py | 163 | 175| 171 | 35068 | 67739 | 
| 101 | 76 django/db/models/functions/datetime.py | 1 | 28| 236 | 35304 | 67739 | 
| 102 | 76 django/utils/dateformat.py | 125 | 139| 127 | 35431 | 67739 | 
| 103 | 77 django/conf/locale/th/formats.py | 5 | 34| 355 | 35786 | 68139 | 
| 104 | 77 django/utils/dateformat.py | 31 | 44| 121 | 35907 | 68139 | 
| 105 | 78 django/conf/global_settings.py | 346 | 396| 826 | 36733 | 73798 | 
| 106 | 78 django/db/models/fields/__init__.py | 1242 | 1283| 332 | 37065 | 73798 | 
| 107 | 78 django/db/models/fields/__init__.py | 1102 | 1140| 293 | 37358 | 73798 | 
| 108 | 78 django/db/models/fields/__init__.py | 2081 | 2122| 325 | 37683 | 73798 | 
| 109 | 78 django/db/models/functions/datetime.py | 205 | 232| 424 | 38107 | 73798 | 
| 110 | 78 django/db/models/functions/datetime.py | 183 | 203| 216 | 38323 | 73798 | 
| 111 | 79 django/contrib/humanize/templatetags/humanize.py | 264 | 302| 370 | 38693 | 76958 | 
| 112 | 79 django/db/models/fields/__init__.py | 1186 | 1204| 180 | 38873 | 76958 | 
| 113 | 79 django/utils/dateformat.py | 1 | 28| 223 | 39096 | 76958 | 
| 114 | 79 django/contrib/gis/gdal/field.py | 60 | 71| 158 | 39254 | 76958 | 
| 115 | 80 django/utils/duration.py | 1 | 45| 304 | 39558 | 77263 | 
| 116 | 80 django/utils/dateformat.py | 141 | 152| 151 | 39709 | 77263 | 
| 117 | 81 django/utils/formats.py | 141 | 162| 205 | 39914 | 79355 | 
| 118 | 81 django/db/models/functions/datetime.py | 31 | 61| 300 | 40214 | 79355 | 
| 119 | **81 django/forms/fields.py** | 469 | 494| 174 | 40388 | 79355 | 
| 120 | 82 django/forms/utils.py | 149 | 179| 228 | 40616 | 80587 | 
| 121 | 82 django/db/models/functions/datetime.py | 234 | 253| 170 | 40786 | 80587 | 
| 122 | 83 django/utils/dateparse.py | 1 | 66| 761 | 41547 | 82067 | 
| 123 | 83 django/utils/dateformat.py | 183 | 206| 226 | 41773 | 82067 | 
| 124 | 83 django/db/models/functions/datetime.py | 63 | 84| 255 | 42028 | 82067 | 
| 125 | 83 django/db/models/fields/__init__.py | 1052 | 1081| 218 | 42246 | 82067 | 
| 126 | 84 django/utils/timesince.py | 1 | 24| 220 | 42466 | 82924 | 
| 127 | 84 django/utils/dateparse.py | 98 | 122| 258 | 42724 | 82924 | 
| 128 | 84 django/utils/datetime_safe.py | 1 | 71| 461 | 43185 | 82924 | 
| 129 | 85 django/db/backends/sqlite3/base.py | 411 | 430| 196 | 43381 | 88688 | 
| 130 | 86 django/utils/http.py | 145 | 156| 119 | 43500 | 92866 | 
| 131 | 87 django/forms/widgets.py | 1031 | 1050| 144 | 43644 | 100872 | 
| 132 | 87 django/contrib/humanize/templatetags/humanize.py | 219 | 262| 731 | 44375 | 100872 | 
| 133 | 87 django/utils/dateparse.py | 69 | 95| 222 | 44597 | 100872 | 
| 134 | **87 django/forms/fields.py** | 1173 | 1203| 182 | 44779 | 100872 | 
| 135 | 88 django/utils/dates.py | 1 | 50| 679 | 45458 | 101551 | 
| 136 | 88 django/db/backends/sqlite3/base.py | 498 | 519| 377 | 45835 | 101551 | 
| 137 | 89 django/contrib/admin/filters.py | 305 | 368| 627 | 46462 | 105644 | 
| 138 | 90 django/contrib/admin/widgets.py | 54 | 75| 168 | 46630 | 109510 | 
| 139 | 90 django/db/models/fields/__init__.py | 1036 | 1049| 104 | 46734 | 109510 | 
| 140 | 90 django/db/models/fields/__init__.py | 2282 | 2332| 339 | 47073 | 109510 | 
| 141 | 91 django/templatetags/tz.py | 125 | 145| 176 | 47249 | 110695 | 
| 142 | 92 django/template/defaultfilters.py | 692 | 769| 443 | 47692 | 116769 | 
| 143 | 92 django/templatetags/tz.py | 37 | 78| 288 | 47980 | 116769 | 
| 144 | 93 django/db/models/base.py | 1143 | 1158| 138 | 48118 | 132085 | 
| 145 | 93 django/utils/dateparse.py | 125 | 148| 239 | 48357 | 132085 | 
| 146 | 93 django/forms/widgets.py | 1052 | 1075| 248 | 48605 | 132085 | 
| 147 | 93 django/utils/formats.py | 1 | 57| 377 | 48982 | 132085 | 
| 148 | 94 django/contrib/admin/utils.py | 1 | 24| 228 | 49210 | 136202 | 
| 149 | 94 django/db/models/functions/datetime.py | 256 | 327| 425 | 49635 | 136202 | 
| 150 | 94 django/utils/http.py | 159 | 196| 380 | 50015 | 136202 | 
| 151 | 95 django/db/backends/utils.py | 151 | 175| 234 | 50249 | 138069 | 
| 152 | 96 django/template/defaulttags.py | 1131 | 1151| 160 | 50409 | 149110 | 
| 153 | 97 django/utils/timezone.py | 133 | 150| 148 | 50557 | 150780 | 
| 154 | 97 django/contrib/humanize/templatetags/humanize.py | 182 | 216| 280 | 50837 | 150780 | 
| 155 | 98 django/core/files/storage.py | 296 | 368| 483 | 51320 | 153651 | 
| 156 | 98 django/db/backends/sqlite3/base.py | 451 | 477| 229 | 51549 | 153651 | 
| 157 | 98 django/contrib/gis/gdal/field.py | 178 | 233| 330 | 51879 | 153651 | 
| 158 | 99 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 52028 | 154328 | 
| 159 | 99 django/utils/formats.py | 187 | 207| 237 | 52265 | 154328 | 
| 160 | 99 django/forms/widgets.py | 883 | 907| 172 | 52437 | 154328 | 
| 161 | 99 django/db/models/fields/__init__.py | 1383 | 1406| 171 | 52608 | 154328 | 
| 162 | 99 django/db/backends/sqlite3/base.py | 480 | 495| 156 | 52764 | 154328 | 
| 163 | **99 django/forms/fields.py** | 242 | 259| 164 | 52928 | 154328 | 
| 164 | 99 django/utils/timezone.py | 1 | 101| 608 | 53536 | 154328 | 
| 165 | 99 django/db/models/fields/__init__.py | 1808 | 1836| 191 | 53727 | 154328 | 
| 166 | 99 django/contrib/admin/widgets.py | 78 | 94| 145 | 53872 | 154328 | 
| 167 | 100 django/conf/locale/fy/formats.py | 22 | 22| 0 | 53872 | 154480 | 
| 168 | 100 django/db/models/fields/__init__.py | 973 | 1005| 208 | 54080 | 154480 | 
| **-> 169 <-** | **100 django/forms/fields.py** | 1 | 42| 347 | 54427 | 154480 | 
| 170 | 100 django/utils/formats.py | 210 | 232| 209 | 54636 | 154480 | 
| 171 | **100 django/forms/fields.py** | 351 | 368| 160 | 54796 | 154480 | 
| 172 | 100 django/db/backends/utils.py | 132 | 148| 141 | 54937 | 154480 | 
| 173 | 101 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 55282 | 154825 | 
| 174 | 101 django/template/defaultfilters.py | 805 | 849| 378 | 55660 | 154825 | 
| 175 | 102 django/views/generic/dates.py | 1 | 65| 420 | 56080 | 160188 | 
| 176 | 103 django/db/models/fields/files.py | 1 | 130| 905 | 56985 | 163910 | 
| 177 | 103 django/db/models/fields/__init__.py | 1645 | 1663| 134 | 57119 | 163910 | 
| 178 | 103 django/db/models/fields/__init__.py | 1456 | 1515| 398 | 57517 | 163910 | 
| 179 | 103 django/db/backends/sqlite3/base.py | 433 | 448| 199 | 57716 | 163910 | 
| 180 | 103 django/db/models/fields/__init__.py | 1432 | 1454| 119 | 57835 | 163910 | 
| 181 | 103 django/db/models/fields/__init__.py | 1408 | 1430| 121 | 57956 | 163910 | 
| 182 | 104 django/contrib/postgres/fields/ranges.py | 114 | 161| 262 | 58218 | 166027 | 
| 183 | 104 django/db/models/fields/__init__.py | 1518 | 1575| 350 | 58568 | 166027 | 
| 184 | 105 django/utils/log.py | 162 | 196| 290 | 58858 | 167652 | 
| 185 | **105 django/forms/fields.py** | 1158 | 1170| 113 | 58971 | 167652 | 
| 186 | **105 django/forms/fields.py** | 208 | 239| 274 | 59245 | 167652 | 
| 187 | 105 django/forms/widgets.py | 1006 | 1029| 239 | 59484 | 167652 | 
| 188 | 105 django/db/models/fields/__init__.py | 1706 | 1729| 146 | 59630 | 167652 | 
| 189 | 105 django/views/generic/dates.py | 229 | 271| 334 | 59964 | 167652 | 
| 190 | 106 django/contrib/postgres/forms/hstore.py | 1 | 59| 339 | 60303 | 167991 | 
| 191 | 106 django/db/models/fields/__init__.py | 1578 | 1599| 183 | 60486 | 167991 | 
| 192 | 106 django/utils/formats.py | 235 | 258| 202 | 60688 | 167991 | 
| 193 | **106 django/forms/fields.py** | 1078 | 1119| 353 | 61041 | 167991 | 
| 194 | 106 django/db/models/fields/__init__.py | 1760 | 1805| 279 | 61320 | 167991 | 
| 195 | **106 django/forms/fields.py** | 547 | 566| 171 | 61491 | 167991 | 
| 196 | **106 django/forms/fields.py** | 323 | 349| 227 | 61718 | 167991 | 
| 197 | 107 django/db/backends/base/base.py | 117 | 138| 186 | 61904 | 172845 | 
| 198 | 108 django/db/backends/oracle/utils.py | 41 | 82| 279 | 62183 | 173374 | 
| 199 | 108 django/db/models/fields/__init__.py | 1007 | 1033| 229 | 62412 | 173374 | 
| 200 | 109 django/core/serializers/json.py | 61 | 105| 336 | 62748 | 174063 | 
| 201 | 110 django/forms/boundfield.py | 1 | 33| 239 | 62987 | 176184 | 
| 202 | 110 django/views/generic/dates.py | 605 | 626| 214 | 63201 | 176184 | 
| 203 | 110 django/db/models/fields/__init__.py | 1839 | 1916| 567 | 63768 | 176184 | 
| 204 | 111 django/db/migrations/serializer.py | 1 | 73| 428 | 64196 | 178733 | 
| 205 | 112 django/contrib/postgres/fields/citext.py | 1 | 25| 113 | 64309 | 178847 | 
| 206 | 112 django/templatetags/tz.py | 173 | 191| 149 | 64458 | 178847 | 
| 207 | 113 django/db/backends/sqlite3/operations.py | 209 | 238| 198 | 64656 | 181738 | 
| 208 | 113 django/utils/timezone.py | 228 | 252| 233 | 64889 | 181738 | 
| 209 | 113 django/forms/widgets.py | 464 | 504| 275 | 65164 | 181738 | 
| 210 | 113 django/forms/widgets.py | 910 | 919| 107 | 65271 | 181738 | 
| 211 | 113 django/utils/timezone.py | 175 | 225| 349 | 65620 | 181738 | 


### Hint

```
ISO8601 is a good machine format, but not a particularly nice human readable format. Form processing is primarily about human-readable input. If you disagree, the DateTimeField input formats are configurable (DATETIME_INPUT_FORMATS), so you can add ISO8601 format in your own projects if you want.
I think the problem can't be resolved with DATETIME_INPUT_FORMATS tweaking. ISO8601 format allows timezone info: '2010-09-01T19:52:15+04:00'. Such strings can't be parsed with python's strptime because python's strptime doesn't support '%z' format char (​http://bugs.python.org/issue6641). So DATETIME_INPUT_FORMATS directive is not helpful for ISO8601 handling. The solution is probably to use custom form field.
Replying to russellm: ISO8601 is a good machine format, but not a particularly nice human readable format. Form processing is primarily about human-readable input. If you disagree, the DateTimeField input formats are configurable (DATETIME_INPUT_FORMATS), so you can add ISO8601 format in your own projects if you want. Hi Russell, I understand your reasoning at the time this was closed for not supporting the T separator. However, this is not relevant again because of the way HTML5 provides the new Input Types. By default, using the datetime-local Input Type results in the format of YYYY-MM-DDTHH:MM. It would definitely make it nice to allow for that Input Type default to work properly with DateTimeField.
Reopening, considering comment:3, Python issue 6641 being fixed on Python 3 and the presence of django.utils.formats.ISO_INPUT_FORMATS.
Note that this is not yet really reopened. An oversight?
As kmike mentioned above, customizing the form field with an input_formats containing the timezone marker %z doesn't help — at least on Python 2.7. For anyone hitting this, I worked around it by using a custom form field and overriding the strptime method: from django.utils.dateparse import parse_datetime from django.utils.encoding import force_str class ISODateTimeField(forms.DateTimeField): def strptime(self, value, format): return parse_datetime(force_str(value)) I use Django's own parse_datetime utility. Note this is limited to ISO datetimes, and effectively any input_formats are omitted.
Interest for this is revived by the HTML5 <input type="datetime-local"> which is sending input formatted with ISO 8601. ​https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/datetime-local ​https://djangotricks.blogspot.com/2019/10/working-with-dates-and-times-in-forms.html ​PR
```

## Patch

```diff
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -25,7 +25,7 @@
     URLInput,
 )
 from django.utils import formats
-from django.utils.dateparse import parse_duration
+from django.utils.dateparse import parse_datetime, parse_duration
 from django.utils.duration import duration_string
 from django.utils.ipv6 import clean_ipv6_address
 from django.utils.regex_helper import _lazy_re_compile
@@ -459,7 +459,12 @@ def to_python(self, value):
         if isinstance(value, datetime.date):
             result = datetime.datetime(value.year, value.month, value.day)
             return from_current_timezone(result)
-        result = super().to_python(value)
+        try:
+            result = parse_datetime(value.strip())
+        except ValueError:
+            raise ValidationError(self.error_messages['invalid'], code='invalid')
+        if not result:
+            result = super().to_python(value)
         return from_current_timezone(result)
 
     def strptime(self, value, format):

```

## Test Patch

```diff
diff --git a/tests/forms_tests/field_tests/test_datetimefield.py b/tests/forms_tests/field_tests/test_datetimefield.py
--- a/tests/forms_tests/field_tests/test_datetimefield.py
+++ b/tests/forms_tests/field_tests/test_datetimefield.py
@@ -2,6 +2,7 @@
 
 from django.forms import DateTimeField, ValidationError
 from django.test import SimpleTestCase
+from django.utils.timezone import get_fixed_timezone, utc
 
 
 class DateTimeFieldTest(SimpleTestCase):
@@ -31,6 +32,19 @@ def test_datetimefield_clean(self):
             ('10/25/06 14:30:00', datetime(2006, 10, 25, 14, 30)),
             ('10/25/06 14:30', datetime(2006, 10, 25, 14, 30)),
             ('10/25/06', datetime(2006, 10, 25, 0, 0)),
+            # ISO 8601 formats.
+            (
+                '2014-09-23T22:34:41.614804',
+                datetime(2014, 9, 23, 22, 34, 41, 614804),
+            ),
+            ('2014-09-23T22:34:41', datetime(2014, 9, 23, 22, 34, 41)),
+            ('2014-09-23T22:34', datetime(2014, 9, 23, 22, 34)),
+            ('2014-09-23', datetime(2014, 9, 23, 0, 0)),
+            ('2014-09-23T22:34Z', datetime(2014, 9, 23, 22, 34, tzinfo=utc)),
+            (
+                '2014-09-23T22:34+07:00',
+                datetime(2014, 9, 23, 22, 34, tzinfo=get_fixed_timezone(420)),
+            ),
             # Whitespace stripping.
             (' 2006-10-25   14:30:45 ', datetime(2006, 10, 25, 14, 30, 45)),
             (' 2006-10-25 ', datetime(2006, 10, 25, 0, 0)),
@@ -39,6 +53,11 @@ def test_datetimefield_clean(self):
             (' 10/25/2006 ', datetime(2006, 10, 25, 0, 0)),
             (' 10/25/06 14:30:45 ', datetime(2006, 10, 25, 14, 30, 45)),
             (' 10/25/06 ', datetime(2006, 10, 25, 0, 0)),
+            (
+                ' 2014-09-23T22:34:41.614804 ',
+                datetime(2014, 9, 23, 22, 34, 41, 614804),
+            ),
+            (' 2014-09-23T22:34Z ', datetime(2014, 9, 23, 22, 34, tzinfo=utc)),
         ]
         f = DateTimeField()
         for value, expected_datetime in tests:
@@ -54,9 +73,11 @@ def test_datetimefield_clean_invalid(self):
             f.clean('2006-10-25 4:30 p.m.')
         with self.assertRaisesMessage(ValidationError, msg):
             f.clean('   ')
+        with self.assertRaisesMessage(ValidationError, msg):
+            f.clean('2014-09-23T28:23')
         f = DateTimeField(input_formats=['%Y %m %d %I:%M %p'])
         with self.assertRaisesMessage(ValidationError, msg):
-            f.clean('2006-10-25 14:30:45')
+            f.clean('2006.10.25 14:30:45')
 
     def test_datetimefield_clean_input_formats(self):
         tests = [
@@ -72,6 +93,8 @@ def test_datetimefield_clean_input_formats(self):
                     datetime(2006, 10, 25, 14, 30, 59, 200),
                 ),
                 ('2006 10 25 2:30 PM', datetime(2006, 10, 25, 14, 30)),
+                # ISO-like formats are always accepted.
+                ('2006-10-25 14:30:45', datetime(2006, 10, 25, 14, 30, 45)),
             )),
             ('%Y.%m.%d %H:%M:%S.%f', (
                 (
diff --git a/tests/forms_tests/tests/test_input_formats.py b/tests/forms_tests/tests/test_input_formats.py
--- a/tests/forms_tests/tests/test_input_formats.py
+++ b/tests/forms_tests/tests/test_input_formats.py
@@ -703,7 +703,7 @@ def test_localized_dateTimeField_with_inputformat(self):
         f = forms.DateTimeField(input_formats=["%H.%M.%S %m.%d.%Y", "%H.%M %m-%d-%Y"], localize=True)
         # Parse a date in an unaccepted format; get an error
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
         with self.assertRaises(forms.ValidationError):
             f.clean('1:30:05 PM 21/12/2010')
         with self.assertRaises(forms.ValidationError):
@@ -711,8 +711,12 @@ def test_localized_dateTimeField_with_inputformat(self):
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('13.30.05 12.21.2010')
-        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
-
+        self.assertEqual(datetime(2010, 12, 21, 13, 30, 5), result)
+        # ISO format is always valid.
+        self.assertEqual(
+            f.clean('2010-12-21 13:30:05'),
+            datetime(2010, 12, 21, 13, 30, 5),
+        )
         # The parsed result does a round trip to the same format
         text = f.widget.format_value(result)
         self.assertEqual(text, "21.12.2010 13:30:05")
@@ -733,7 +737,7 @@ def test_dateTimeField(self):
         f = forms.DateTimeField()
         # Parse a date in an unaccepted format; get an error
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('1:30:05 PM 21/12/2010')
@@ -756,7 +760,7 @@ def test_localized_dateTimeField(self):
         f = forms.DateTimeField(localize=True)
         # Parse a date in an unaccepted format; get an error
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('1:30:05 PM 21/12/2010')
@@ -781,7 +785,7 @@ def test_dateTimeField_with_inputformat(self):
         with self.assertRaises(forms.ValidationError):
             f.clean('13:30:05 21.12.2010')
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('12.21.2010 13:30:05')
@@ -806,7 +810,7 @@ def test_localized_dateTimeField_with_inputformat(self):
         with self.assertRaises(forms.ValidationError):
             f.clean('13:30:05 21.12.2010')
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('12.21.2010 13:30:05')
@@ -877,7 +881,7 @@ def test_dateTimeField_with_inputformat(self):
         f = forms.DateTimeField(input_formats=["%I:%M:%S %p %d.%m.%Y", "%I:%M %p %d-%m-%Y"])
         # Parse a date in an unaccepted format; get an error
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('1:30:05 PM 21.12.2010')
@@ -900,7 +904,7 @@ def test_localized_dateTimeField_with_inputformat(self):
         f = forms.DateTimeField(input_formats=["%I:%M:%S %p %d.%m.%Y", "%I:%M %p %d-%m-%Y"], localize=True)
         # Parse a date in an unaccepted format; get an error
         with self.assertRaises(forms.ValidationError):
-            f.clean('2010-12-21 13:30:05')
+            f.clean('2010/12/21 13:30:05')
 
         # Parse a date in a valid format, get a parsed result
         result = f.clean('1:30:05 PM 21.12.2010')
diff --git a/tests/timezones/tests.py b/tests/timezones/tests.py
--- a/tests/timezones/tests.py
+++ b/tests/timezones/tests.py
@@ -1081,11 +1081,6 @@ def test_form_with_other_timezone(self):
             self.assertTrue(form.is_valid())
             self.assertEqual(form.cleaned_data['dt'], datetime.datetime(2011, 9, 1, 10, 20, 30, tzinfo=UTC))
 
-    def test_form_with_explicit_timezone(self):
-        form = EventForm({'dt': '2011-09-01 17:20:30+07:00'})
-        # Datetime inputs formats don't allow providing a time zone.
-        self.assertFalse(form.is_valid())
-
     def test_form_with_non_existent_time(self):
         with timezone.override(pytz.timezone('Europe/Paris')):
             form = EventForm({'dt': '2011-03-27 02:30:00'})

```


## Code snippets

### 1 - django/db/models/fields/__init__.py:

Start line: 1227, End line: 1240

```python
class DateTimeField(DateField):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value has an invalid format. It must be in '
                     'YYYY-MM-DD HH:MM[:ss[.uuuuuu]][TZ] format.'),
        'invalid_date': _("“%(value)s” value has the correct format "
                          "(YYYY-MM-DD) but it is an invalid date."),
        'invalid_datetime': _('“%(value)s” value has the correct format '
                              '(YYYY-MM-DD HH:MM[:ss[.uuuuuu]][TZ]) '
                              'but it is an invalid date/time.'),
    }
    description = _("Date (with time)")

    # __init__ is inherited from DateField
```
### 2 - django/forms/fields.py:

Start line: 438, End line: 466

```python
class DateTimeField(BaseTemporalField):
    widget = DateTimeInput
    input_formats = formats.get_format_lazy('DATETIME_INPUT_FORMATS')
    default_error_messages = {
        'invalid': _('Enter a valid date/time.'),
    }

    def prepare_value(self, value):
        if isinstance(value, datetime.datetime):
            value = to_current_timezone(value)
        return value

    def to_python(self, value):
        """
        Validate that the input can be converted to a datetime. Return a
        Python datetime.datetime object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.datetime):
            return from_current_timezone(value)
        if isinstance(value, datetime.date):
            result = datetime.datetime(value.year, value.month, value.day)
            return from_current_timezone(result)
        result = super().to_python(value)
        return from_current_timezone(result)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format)
```
### 3 - django/utils/dateformat.py:

Start line: 209, End line: 280

```python
class DateFormat(TimeFormat):
    def b(self):
        "Month, textual, 3 letters, lowercase; e.g. 'jan'"
        return MONTHS_3[self.data.month]

    def c(self):
        """
        ISO 8601 Format
        Example : '2008-01-02T10:30:00.000123'
        """
        return self.data.isoformat()

    def d(self):
        "Day of the month, 2 digits with leading zeros; i.e. '01' to '31'"
        return '%02d' % self.data.day

    def D(self):
        "Day of the week, textual, 3 letters; e.g. 'Fri'"
        return WEEKDAYS_ABBR[self.data.weekday()]

    def E(self):
        "Alternative month names as required by some locales. Proprietary extension."
        return MONTHS_ALT[self.data.month]

    def F(self):
        "Month, textual, long; e.g. 'January'"
        return MONTHS[self.data.month]

    def I(self):  # NOQA: E743
        "'1' if Daylight Savings Time, '0' otherwise."
        try:
            if self.timezone and self.timezone.dst(self.data):
                return '1'
            else:
                return '0'
        except Exception:
            # pytz raises AmbiguousTimeError during the autumn DST change.
            # This happens mainly when __init__ receives a naive datetime
            # and sets self.timezone = get_default_timezone().
            return ''

    def j(self):
        "Day of the month without leading zeros; i.e. '1' to '31'"
        return self.data.day

    def l(self):  # NOQA: E743
        "Day of the week, textual, long; e.g. 'Friday'"
        return WEEKDAYS[self.data.weekday()]

    def L(self):
        "Boolean for whether it is a leap year; i.e. True or False"
        return calendar.isleap(self.data.year)

    def m(self):
        "Month; i.e. '01' to '12'"
        return '%02d' % self.data.month

    def M(self):
        "Month, textual, 3 letters; e.g. 'Jan'"
        return MONTHS_3[self.data.month].title()

    def n(self):
        "Month without leading zeros; i.e. '1' to '12'"
        return self.data.month

    def N(self):
        "Month abbreviation in Associated Press style. Proprietary extension."
        return MONTHS_AP[self.data.month]

    def o(self):
        "ISO 8601 year number matching the ISO week number (W)"
        return self.data.isocalendar()[0]
```
### 4 - django/db/models/fields/__init__.py:

Start line: 1285, End line: 1334

```python
class DateTimeField(DateField):

    def get_internal_type(self):
        return "DateTimeField"

    def to_python(self, value):
        if value is None:
            return value
        if isinstance(value, datetime.datetime):
            return value
        if isinstance(value, datetime.date):
            value = datetime.datetime(value.year, value.month, value.day)
            if settings.USE_TZ:
                # For backwards compatibility, interpret naive datetimes in
                # local time. This won't work during DST change, but we can't
                # do much about it, so we let the exceptions percolate up the
                # call stack.
                warnings.warn("DateTimeField %s.%s received a naive datetime "
                              "(%s) while time zone support is active." %
                              (self.model.__name__, self.name, value),
                              RuntimeWarning)
                default_timezone = timezone.get_default_timezone()
                value = timezone.make_aware(value, default_timezone)
            return value

        try:
            parsed = parse_datetime(value)
            if parsed is not None:
                return parsed
        except ValueError:
            raise exceptions.ValidationError(
                self.error_messages['invalid_datetime'],
                code='invalid_datetime',
                params={'value': value},
            )

        try:
            parsed = parse_date(value)
            if parsed is not None:
                return datetime.datetime(parsed.year, parsed.month, parsed.day)
        except ValueError:
            raise exceptions.ValidationError(
                self.error_messages['invalid_date'],
                code='invalid_date',
                params={'value': value},
            )

        raise exceptions.ValidationError(
            self.error_messages['invalid'],
            code='invalid',
            params={'value': value},
        )
```
### 5 - django/db/models/fields/__init__.py:

Start line: 2063, End line: 2079

```python
class TimeField(DateTimeCheckMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value has an invalid format. It must be in '
                     'HH:MM[:ss[.uuuuuu]] format.'),
        'invalid_time': _('“%(value)s” value has the correct format '
                          '(HH:MM[:ss[.uuuuuu]]) but it is an invalid time.'),
    }
    description = _("Time")

    def __init__(self, verbose_name=None, name=None, auto_now=False,
                 auto_now_add=False, **kwargs):
        self.auto_now, self.auto_now_add = auto_now, auto_now_add
        if auto_now or auto_now_add:
            kwargs['editable'] = False
            kwargs['blank'] = True
        super().__init__(verbose_name, name, **kwargs)
```
### 6 - django/utils/dateformat.py:

Start line: 282, End line: 293

```python
class DateFormat(TimeFormat):

    def r(self):
        "RFC 5322 formatted date; e.g. 'Thu, 21 Dec 2000 16:01:07 +0200'"
        if type(self.data) is datetime.date:
            raise TypeError(
                "The format for date objects may not contain time-related "
                "format specifiers (found 'r')."
            )
        if is_naive(self.data):
            dt = make_aware(self.data, timezone=self.timezone)
        else:
            dt = self.data
        return format_datetime_rfc5322(dt)
```
### 7 - django/utils/dateformat.py:

Start line: 295, End line: 350

```python
class DateFormat(TimeFormat):

    def S(self):
        "English ordinal suffix for the day of the month, 2 characters; i.e. 'st', 'nd', 'rd' or 'th'"
        if self.data.day in (11, 12, 13):  # Special case
            return 'th'
        last = self.data.day % 10
        if last == 1:
            return 'st'
        if last == 2:
            return 'nd'
        if last == 3:
            return 'rd'
        return 'th'

    def t(self):
        "Number of days in the given month; i.e. '28' to '31'"
        return '%02d' % calendar.monthrange(self.data.year, self.data.month)[1]

    def U(self):
        "Seconds since the Unix epoch (January 1 1970 00:00:00 GMT)"
        if isinstance(self.data, datetime.datetime) and is_aware(self.data):
            return int(calendar.timegm(self.data.utctimetuple()))
        else:
            return int(time.mktime(self.data.timetuple()))

    def w(self):
        "Day of the week, numeric, i.e. '0' (Sunday) to '6' (Saturday)"
        return (self.data.weekday() + 1) % 7

    def W(self):
        "ISO-8601 week number of year, weeks starting on Monday"
        return self.data.isocalendar()[1]

    def y(self):
        "Year, 2 digits; e.g. '99'"
        return str(self.data.year)[2:]

    def Y(self):
        "Year, 4 digits; e.g. '1999'"
        return self.data.year

    def z(self):
        """Day of the year, i.e. 1 to 366."""
        return self.data.timetuple().tm_yday


def format(value, format_string):
    "Convenience function"
    df = DateFormat(value)
    return df.format(format_string)


def time_format(value, format_string):
    "Convenience function"
    tf = TimeFormat(value)
    return tf.format(format_string)
```
### 8 - django/conf/locale/hr/formats.py:

Start line: 5, End line: 21

```python
DATE_FORMAT = 'j. E Y.'
TIME_FORMAT = 'H:i'
DATETIME_FORMAT = 'j. E Y. H:i'
YEAR_MONTH_FORMAT = 'F Y.'
MONTH_DAY_FORMAT = 'j. F'
SHORT_DATE_FORMAT = 'j.m.Y.'
SHORT_DATETIME_FORMAT = 'j.m.Y. H:i'
FIRST_DAY_OF_WEEK = 1

# The *_INPUT_FORMATS strings use the Python strftime format syntax,
# see https://docs.python.org/library/datetime.html#strftime-strptime-behavior
# Kept ISO formats as they are in first position
DATE_INPUT_FORMATS = [
    '%Y-%m-%d',                     # '2006-10-25'
    '%d.%m.%Y.', '%d.%m.%y.',       # '25.10.2006.', '25.10.06.'
    '%d. %m. %Y.', '%d. %m. %y.',   # '25. 10. 2006.', '25. 10. 06.'
]
```
### 9 - django/db/models/fields/__init__.py:

Start line: 1084, End line: 1100

```python
class DateField(DateTimeCheckMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value has an invalid date format. It must be '
                     'in YYYY-MM-DD format.'),
        'invalid_date': _('“%(value)s” value has the correct format (YYYY-MM-DD) '
                          'but it is an invalid date.'),
    }
    description = _("Date (without time)")

    def __init__(self, verbose_name=None, name=None, auto_now=False,
                 auto_now_add=False, **kwargs):
        self.auto_now, self.auto_now_add = auto_now, auto_now_add
        if auto_now or auto_now_add:
            kwargs['editable'] = False
            kwargs['blank'] = True
        super().__init__(verbose_name, name, **kwargs)
```
### 10 - django/conf/locale/is/formats.py:

Start line: 5, End line: 22

```python
DATE_FORMAT = 'j. F Y'
TIME_FORMAT = 'H:i'
# DATETIME_FORMAT =
YEAR_MONTH_FORMAT = 'F Y'
MONTH_DAY_FORMAT = 'j. F'
SHORT_DATE_FORMAT = 'j.n.Y'
# SHORT_DATETIME_FORMAT =
# FIRST_DAY_OF_WEEK =

# The *_INPUT_FORMATS strings use the Python strftime format syntax,
# see https://docs.python.org/library/datetime.html#strftime-strptime-behavior
# DATE_INPUT_FORMATS =
# TIME_INPUT_FORMATS =
# DATETIME_INPUT_FORMATS =
DECIMAL_SEPARATOR = ','
THOUSAND_SEPARATOR = '.'
NUMBER_GROUPING = 3
```
### 47 - django/forms/fields.py:

Start line: 1122, End line: 1155

```python
class SplitDateTimeField(MultiValueField):
    widget = SplitDateTimeWidget
    hidden_widget = SplitHiddenDateTimeWidget
    default_error_messages = {
        'invalid_date': _('Enter a valid date.'),
        'invalid_time': _('Enter a valid time.'),
    }

    def __init__(self, *, input_date_formats=None, input_time_formats=None, **kwargs):
        errors = self.default_error_messages.copy()
        if 'error_messages' in kwargs:
            errors.update(kwargs['error_messages'])
        localize = kwargs.get('localize', False)
        fields = (
            DateField(input_formats=input_date_formats,
                      error_messages={'invalid': errors['invalid_date']},
                      localize=localize),
            TimeField(input_formats=input_time_formats,
                      error_messages={'invalid': errors['invalid_time']},
                      localize=localize),
        )
        super().__init__(fields, **kwargs)

    def compress(self, data_list):
        if data_list:
            # Raise a validation error if time or date is empty
            # (possible if SplitDateTimeField has required=False).
            if data_list[0] in self.empty_values:
                raise ValidationError(self.error_messages['invalid_date'], code='invalid_date')
            if data_list[1] in self.empty_values:
                raise ValidationError(self.error_messages['invalid_time'], code='invalid_time')
            result = datetime.datetime.combine(*data_list)
            return from_current_timezone(result)
        return None
```
### 59 - django/forms/fields.py:

Start line: 392, End line: 413

```python
class DateField(BaseTemporalField):
    widget = DateInput
    input_formats = formats.get_format_lazy('DATE_INPUT_FORMATS')
    default_error_messages = {
        'invalid': _('Enter a valid date.'),
    }

    def to_python(self, value):
        """
        Validate that the input can be converted to a date. Return a Python
        datetime.date object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        return super().to_python(value)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format).date()
```
### 64 - django/forms/fields.py:

Start line: 416, End line: 435

```python
class TimeField(BaseTemporalField):
    widget = TimeInput
    input_formats = formats.get_format_lazy('TIME_INPUT_FORMATS')
    default_error_messages = {
        'invalid': _('Enter a valid time.')
    }

    def to_python(self, value):
        """
        Validate that the input can be converted to a time. Return a Python
        datetime.time object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.time):
            return value
        return super().to_python(value)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format).time()
```
### 99 - django/forms/fields.py:

Start line: 371, End line: 389

```python
class BaseTemporalField(Field):

    def __init__(self, *, input_formats=None, **kwargs):
        super().__init__(**kwargs)
        if input_formats is not None:
            self.input_formats = input_formats

    def to_python(self, value):
        value = value.strip()
        # Try to strptime against each input format.
        for format in self.input_formats:
            try:
                return self.strptime(value, format)
            except (ValueError, TypeError):
                continue
        raise ValidationError(self.error_messages['invalid'], code='invalid')

    def strptime(self, value, format):
        raise NotImplementedError('Subclasses must define this method.')
```
### 119 - django/forms/fields.py:

Start line: 469, End line: 494

```python
class DurationField(Field):
    default_error_messages = {
        'invalid': _('Enter a valid duration.'),
        'overflow': _('The number of days must be between {min_days} and {max_days}.')
    }

    def prepare_value(self, value):
        if isinstance(value, datetime.timedelta):
            return duration_string(value)
        return value

    def to_python(self, value):
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.timedelta):
            return value
        try:
            value = parse_duration(str(value))
        except OverflowError:
            raise ValidationError(self.error_messages['overflow'].format(
                min_days=datetime.timedelta.min.days,
                max_days=datetime.timedelta.max.days,
            ), code='overflow')
        if value is None:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value
```
### 134 - django/forms/fields.py:

Start line: 1173, End line: 1203

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
```
### 163 - django/forms/fields.py:

Start line: 242, End line: 259

```python
class IntegerField(Field):
    widget = NumberInput
    default_error_messages = {
        'invalid': _('Enter a whole number.'),
    }
    re_decimal = _lazy_re_compile(r'\.0*\s*$')

    def __init__(self, *, max_value=None, min_value=None, **kwargs):
        self.max_value, self.min_value = max_value, min_value
        if kwargs.get('localize') and self.widget == NumberInput:
            # Localized number input is not well supported on most browsers
            kwargs.setdefault('widget', super().widget)
        super().__init__(**kwargs)

        if max_value is not None:
            self.validators.append(validators.MaxValueValidator(max_value))
        if min_value is not None:
            self.validators.append(validators.MinValueValidator(min_value))
```
### 169 - django/forms/fields.py:

Start line: 1, End line: 42

```python
"""
Field classes.
"""

import copy
import datetime
import math
import operator
import os
import re
import uuid
from decimal import Decimal, DecimalException
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

from django.core import validators
from django.core.exceptions import ValidationError
from django.forms.boundfield import BoundField
from django.forms.utils import from_current_timezone, to_current_timezone
from django.forms.widgets import (
    FILE_INPUT_CONTRADICTION, CheckboxInput, ClearableFileInput, DateInput,
    DateTimeInput, EmailInput, FileInput, HiddenInput, MultipleHiddenInput,
    NullBooleanSelect, NumberInput, Select, SelectMultiple,
    SplitDateTimeWidget, SplitHiddenDateTimeWidget, TextInput, TimeInput,
    URLInput,
)
from django.utils import formats
from django.utils.dateparse import parse_duration
from django.utils.duration import duration_string
from django.utils.ipv6 import clean_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _, ngettext_lazy

__all__ = (
    'Field', 'CharField', 'IntegerField',
    'DateField', 'TimeField', 'DateTimeField', 'DurationField',
    'RegexField', 'EmailField', 'FileField', 'ImageField', 'URLField',
    'BooleanField', 'NullBooleanField', 'ChoiceField', 'MultipleChoiceField',
    'ComboField', 'MultiValueField', 'FloatField', 'DecimalField',
    'SplitDateTimeField', 'GenericIPAddressField', 'FilePathField',
    'SlugField', 'TypedChoiceField', 'TypedMultipleChoiceField', 'UUIDField',
)
```
### 171 - django/forms/fields.py:

Start line: 351, End line: 368

```python
class DecimalField(IntegerField):

    def validate(self, value):
        super().validate(value)
        if value in self.empty_values:
            return
        if not value.is_finite():
            raise ValidationError(self.error_messages['invalid'], code='invalid')

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, NumberInput) and 'step' not in widget.attrs:
            if self.decimal_places is not None:
                # Use exponential notation for small values since they might
                # be parsed as 0 otherwise. ref #20765
                step = str(Decimal(1).scaleb(-self.decimal_places)).lower()
            else:
                step = 'any'
            attrs.setdefault('step', step)
        return attrs
```
### 185 - django/forms/fields.py:

Start line: 1158, End line: 1170

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
### 186 - django/forms/fields.py:

Start line: 208, End line: 239

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
### 193 - django/forms/fields.py:

Start line: 1078, End line: 1119

```python
class FilePathField(ChoiceField):
    def __init__(self, path, *, match=None, recursive=False, allow_files=True,
                 allow_folders=False, **kwargs):
        self.path, self.match, self.recursive = path, match, recursive
        self.allow_files, self.allow_folders = allow_files, allow_folders
        super().__init__(choices=(), **kwargs)

        if self.required:
            self.choices = []
        else:
            self.choices = [("", "---------")]

        if self.match is not None:
            self.match_re = re.compile(self.match)

        if recursive:
            for root, dirs, files in sorted(os.walk(self.path)):
                if self.allow_files:
                    for f in sorted(files):
                        if self.match is None or self.match_re.search(f):
                            f = os.path.join(root, f)
                            self.choices.append((f, f.replace(path, "", 1)))
                if self.allow_folders:
                    for f in sorted(dirs):
                        if f == '__pycache__':
                            continue
                        if self.match is None or self.match_re.search(f):
                            f = os.path.join(root, f)
                            self.choices.append((f, f.replace(path, "", 1)))
        else:
            choices = []
            for f in os.scandir(self.path):
                if f.name == '__pycache__':
                    continue
                if (((self.allow_files and f.is_file()) or
                        (self.allow_folders and f.is_dir())) and
                        (self.match is None or self.match_re.search(f.name))):
                    choices.append((f.path, f.name))
            choices.sort(key=operator.itemgetter(1))
            self.choices.extend(choices)

        self.widget.choices = self.choices
```
### 195 - django/forms/fields.py:

Start line: 547, End line: 566

```python
class FileField(Field):

    def to_python(self, data):
        if data in self.empty_values:
            return None

        # UploadedFile objects should have name and size attributes.
        try:
            file_name = data.name
            file_size = data.size
        except AttributeError:
            raise ValidationError(self.error_messages['invalid'], code='invalid')

        if self.max_length is not None and len(file_name) > self.max_length:
            params = {'max': self.max_length, 'length': len(file_name)}
            raise ValidationError(self.error_messages['max_length'], code='max_length', params=params)
        if not file_name:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        if not self.allow_empty_file and not file_size:
            raise ValidationError(self.error_messages['empty'], code='empty')

        return data
```
### 196 - django/forms/fields.py:

Start line: 323, End line: 349

```python
class DecimalField(IntegerField):
    default_error_messages = {
        'invalid': _('Enter a number.'),
    }

    def __init__(self, *, max_value=None, min_value=None, max_digits=None, decimal_places=None, **kwargs):
        self.max_digits, self.decimal_places = max_digits, decimal_places
        super().__init__(max_value=max_value, min_value=min_value, **kwargs)
        self.validators.append(validators.DecimalValidator(max_digits, decimal_places))

    def to_python(self, value):
        """
        Validate that the input is a decimal number. Return a Decimal
        instance or None for empty values. Ensure that there are no more
        than max_digits in the number and no more than decimal_places digits
        after the decimal point.
        """
        if value in self.empty_values:
            return None
        if self.localize:
            value = formats.sanitize_separators(value)
        value = str(value).strip()
        try:
            value = Decimal(value)
        except DecimalException:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value
```
