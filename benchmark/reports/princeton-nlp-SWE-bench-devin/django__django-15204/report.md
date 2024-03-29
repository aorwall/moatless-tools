# django__django-15204

| **django/django** | `b0d16d0129b7cc5978a8d55d2331a34cb369e6c7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 42717 |
| **Any found context length** | 42717 |
| **Avg pos** | 122.0 |
| **Min pos** | 122 |
| **Max pos** | 122 |
| **Top file pos** | 35 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/dateparse.py b/django/utils/dateparse.py
--- a/django/utils/dateparse.py
+++ b/django/utils/dateparse.py
@@ -42,11 +42,11 @@
 iso8601_duration_re = _lazy_re_compile(
     r'^(?P<sign>[-+]?)'
     r'P'
-    r'(?:(?P<days>\d+(.\d+)?)D)?'
+    r'(?:(?P<days>\d+([\.,]\d+)?)D)?'
     r'(?:T'
-    r'(?:(?P<hours>\d+(.\d+)?)H)?'
-    r'(?:(?P<minutes>\d+(.\d+)?)M)?'
-    r'(?:(?P<seconds>\d+(.\d+)?)S)?'
+    r'(?:(?P<hours>\d+([\.,]\d+)?)H)?'
+    r'(?:(?P<minutes>\d+([\.,]\d+)?)M)?'
+    r'(?:(?P<seconds>\d+([\.,]\d+)?)S)?'
     r')?'
     r'$'
 )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/dateparse.py | 45 | 49 | 122 | 35 | 42717


## Problem Statement

```
Durationfield.clean fails to handle broken data
Description
	 
		(last modified by Florian Apolloner)
	 
The actual input string was 'P3(3D' 
 === Uncaught Python exception: ===
	ValueError: could not convert string to float: '3(3'
	Traceback (most recent call last):
	 File "basic_fuzzer.py", line 22, in TestOneInput
	 File "fuzzers.py", line 294, in test_forms_DurationField
	 File "django/forms/fields.py", line 149, in clean
	 File "django/forms/fields.py", line 502, in to_python
	 File "django/utils/dateparse.py", line 154, in parse_duration
	 File "django/utils/dateparse.py", line 154, in <dictcomp>

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/forms/fields.py | 485 | 510| 174 | 174 | 9421 | 
| 2 | 1 django/forms/fields.py | 352 | 373| 172 | 346 | 9421 | 
| 3 | 1 django/forms/fields.py | 397 | 418| 144 | 490 | 9421 | 
| 4 | 1 django/forms/fields.py | 1140 | 1173| 293 | 783 | 9421 | 
| 5 | 1 django/forms/fields.py | 290 | 322| 228 | 1011 | 9421 | 
| 6 | 1 django/forms/fields.py | 325 | 350| 220 | 1231 | 9421 | 
| 7 | 1 django/forms/fields.py | 449 | 482| 225 | 1456 | 9421 | 
| 8 | 1 django/forms/fields.py | 376 | 394| 134 | 1590 | 9421 | 
| 9 | 2 django/db/models/fields/__init__.py | 1723 | 1760| 226 | 1816 | 27589 | 
| 10 | 3 django/conf/locale/hr/formats.py | 5 | 43| 742 | 2558 | 28376 | 
| 11 | 4 django/conf/locale/sl/formats.py | 5 | 43| 708 | 3266 | 29129 | 
| 12 | 5 django/conf/locale/sr/formats.py | 5 | 40| 726 | 3992 | 29900 | 
| 13 | 6 django/conf/locale/nl/formats.py | 32 | 67| 1371 | 5363 | 31783 | 
| 14 | 7 django/conf/locale/en_AU/formats.py | 5 | 37| 655 | 6018 | 32483 | 
| 15 | 8 django/conf/locale/en_GB/formats.py | 5 | 37| 655 | 6673 | 33183 | 
| 16 | 9 django/conf/locale/sr_Latn/formats.py | 5 | 40| 726 | 7399 | 33954 | 
| 17 | 10 django/conf/locale/ka/formats.py | 5 | 43| 773 | 8172 | 34772 | 
| 18 | 11 django/conf/locale/ml/formats.py | 5 | 38| 610 | 8782 | 35427 | 
| 19 | 12 django/conf/locale/fr/formats.py | 5 | 32| 448 | 9230 | 35920 | 
| 20 | 13 django/conf/locale/sv/formats.py | 5 | 36| 481 | 9711 | 36446 | 
| 21 | 14 django/conf/locale/cy/formats.py | 5 | 33| 529 | 10240 | 37020 | 
| 22 | 15 django/conf/locale/mk/formats.py | 5 | 39| 594 | 10834 | 37659 | 
| 23 | 15 django/db/models/fields/__init__.py | 1539 | 1557| 118 | 10952 | 37659 | 
| 24 | 15 django/db/models/fields/__init__.py | 1434 | 1457| 171 | 11123 | 37659 | 
| 25 | 16 django/conf/locale/lt/formats.py | 5 | 44| 676 | 11799 | 38380 | 
| 26 | 17 django/conf/locale/ms/formats.py | 5 | 36| 599 | 12398 | 39024 | 
| 27 | 18 django/conf/locale/fi/formats.py | 5 | 38| 435 | 12833 | 39504 | 
| 28 | 19 django/conf/locale/pt/formats.py | 5 | 36| 577 | 13410 | 40126 | 
| 29 | 19 django/db/models/fields/__init__.py | 1339 | 1385| 329 | 13739 | 40126 | 
| 30 | 19 django/db/models/fields/__init__.py | 1483 | 1505| 119 | 13858 | 40126 | 
| 31 | 19 django/forms/fields.py | 421 | 446| 168 | 14026 | 40126 | 
| 32 | 20 django/conf/locale/nn/formats.py | 5 | 37| 593 | 14619 | 40764 | 
| 33 | 20 django/db/models/fields/__init__.py | 1459 | 1481| 121 | 14740 | 40764 | 
| 34 | 21 django/conf/locale/el/formats.py | 5 | 33| 455 | 15195 | 41264 | 
| 35 | 22 django/conf/locale/lv/formats.py | 5 | 45| 700 | 15895 | 42009 | 
| 36 | 23 django/conf/locale/ko/formats.py | 32 | 50| 385 | 16280 | 42899 | 
| 37 | 24 django/conf/locale/en/formats.py | 7 | 60| 812 | 17092 | 43762 | 
| 38 | 25 django/conf/locale/id/formats.py | 5 | 47| 670 | 17762 | 44477 | 
| 39 | 26 django/conf/locale/cs/formats.py | 5 | 41| 600 | 18362 | 45122 | 
| 40 | 27 django/conf/locale/de_CH/formats.py | 5 | 34| 403 | 18765 | 45570 | 
| 41 | 28 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 19400 | 46250 | 
| 42 | 29 django/conf/locale/da/formats.py | 5 | 27| 250 | 19650 | 46545 | 
| 43 | 30 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 20285 | 47225 | 
| 44 | 31 django/conf/locale/nb/formats.py | 5 | 37| 593 | 20878 | 47863 | 
| 45 | 31 django/db/models/fields/__init__.py | 1507 | 1537| 211 | 21089 | 47863 | 
| 46 | 32 django/conf/locale/eo/formats.py | 5 | 48| 707 | 21796 | 48615 | 
| 47 | 32 django/forms/fields.py | 584 | 609| 243 | 22039 | 48615 | 
| 48 | 33 django/conf/locale/az/formats.py | 5 | 31| 364 | 22403 | 49024 | 
| 49 | 34 django/conf/locale/uz/formats.py | 5 | 31| 418 | 22821 | 49487 | 
| 50 | **35 django/utils/dateparse.py** | 134 | 159| 255 | 23076 | 51083 | 
| 51 | 36 django/conf/locale/uk/formats.py | 5 | 36| 425 | 23501 | 51553 | 
| 52 | 37 django/conf/locale/tg/formats.py | 5 | 33| 402 | 23903 | 52000 | 
| 53 | 38 django/conf/locale/ru/formats.py | 5 | 31| 367 | 24270 | 52412 | 
| 54 | 39 django/conf/locale/pt_BR/formats.py | 5 | 32| 459 | 24729 | 52916 | 
| 55 | 40 django/conf/locale/hu/formats.py | 5 | 31| 305 | 25034 | 53266 | 
| 56 | 40 django/forms/fields.py | 244 | 261| 164 | 25198 | 53266 | 
| 57 | 40 django/forms/fields.py | 1231 | 1286| 365 | 25563 | 53266 | 
| 58 | 41 django/conf/locale/ig/formats.py | 5 | 33| 388 | 25951 | 53699 | 
| 59 | 42 django/conf/locale/tk/formats.py | 5 | 33| 402 | 26353 | 54146 | 
| 60 | 43 django/conf/locale/de/formats.py | 5 | 28| 305 | 26658 | 54496 | 
| 61 | 44 django/conf/locale/pl/formats.py | 5 | 29| 321 | 26979 | 54862 | 
| 62 | 45 django/template/defaultfilters.py | 94 | 182| 802 | 27781 | 61289 | 
| 63 | 46 django/conf/locale/sk/formats.py | 5 | 29| 330 | 28111 | 61664 | 
| 64 | 47 django/conf/locale/ca/formats.py | 5 | 31| 287 | 28398 | 61996 | 
| 65 | 47 django/forms/fields.py | 563 | 582| 171 | 28569 | 61996 | 
| 66 | 47 django/db/models/fields/__init__.py | 1236 | 1264| 189 | 28758 | 61996 | 
| 67 | 48 django/conf/locale/bn/formats.py | 5 | 33| 294 | 29052 | 62334 | 
| 68 | 49 django/core/validators.py | 425 | 471| 440 | 29492 | 66845 | 
| 69 | 50 django/conf/locale/it/formats.py | 5 | 41| 764 | 30256 | 67654 | 
| 70 | 50 django/forms/fields.py | 1191 | 1228| 199 | 30455 | 67654 | 
| 71 | 50 django/db/models/fields/__init__.py | 1307 | 1337| 261 | 30716 | 67654 | 
| 72 | 51 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 30968 | 67951 | 
| 73 | 52 django/conf/locale/ky/formats.py | 5 | 33| 414 | 31382 | 68410 | 
| 74 | 53 django/conf/locale/ro/formats.py | 5 | 36| 262 | 31644 | 68717 | 
| 75 | 54 django/conf/locale/et/formats.py | 5 | 22| 133 | 31777 | 68894 | 
| 76 | 54 django/core/validators.py | 396 | 423| 224 | 32001 | 68894 | 
| 77 | 54 django/conf/locale/nl/formats.py | 5 | 31| 467 | 32468 | 68894 | 
| 78 | 55 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 32720 | 69162 | 
| 79 | 56 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 32995 | 69482 | 
| 80 | 57 django/conf/locale/es/formats.py | 5 | 31| 285 | 33280 | 69812 | 
| 81 | 58 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 33542 | 70090 | 
| 82 | 59 django/conf/locale/km/formats.py | 5 | 22| 164 | 33706 | 70298 | 
| 83 | 60 django/conf/locale/bg/formats.py | 5 | 22| 131 | 33837 | 70473 | 
| 84 | 61 django/contrib/gis/forms/fields.py | 62 | 85| 204 | 34041 | 71392 | 
| 85 | 62 django/conf/locale/ga/formats.py | 5 | 22| 124 | 34165 | 71560 | 
| 86 | 63 django/conf/locale/is/formats.py | 5 | 22| 130 | 34295 | 71735 | 
| 87 | 64 django/utils/formats.py | 242 | 272| 320 | 34615 | 74200 | 
| 88 | 65 django/contrib/postgres/forms/ranges.py | 84 | 106| 149 | 34764 | 74912 | 
| 89 | 65 django/forms/fields.py | 545 | 561| 180 | 34944 | 74912 | 
| 90 | 66 django/utils/dateformat.py | 31 | 44| 121 | 35065 | 77506 | 
| 91 | 66 django/db/models/fields/__init__.py | 1185 | 1201| 175 | 35240 | 77506 | 
| 92 | 67 django/conf/locale/kn/formats.py | 5 | 22| 123 | 35363 | 77673 | 
| 93 | 68 django/db/models/functions/mixins.py | 23 | 53| 257 | 35620 | 78091 | 
| 94 | 69 django/conf/locale/sq/formats.py | 5 | 22| 128 | 35748 | 78263 | 
| 95 | 69 django/utils/dateformat.py | 273 | 328| 460 | 36208 | 78263 | 
| 96 | 70 django/conf/locale/tr/formats.py | 5 | 29| 319 | 36527 | 78627 | 
| 97 | 71 django/conf/locale/gl/formats.py | 5 | 22| 170 | 36697 | 78841 | 
| 98 | 72 django/conf/locale/te/formats.py | 5 | 22| 123 | 36820 | 79008 | 
| 99 | 73 django/conf/locale/mn/formats.py | 5 | 22| 120 | 36940 | 79172 | 
| 100 | 73 django/conf/locale/ko/formats.py | 5 | 31| 460 | 37400 | 79172 | 
| 101 | 73 django/utils/formats.py | 275 | 298| 202 | 37602 | 79172 | 
| 102 | 74 django/conf/locale/eu/formats.py | 5 | 22| 171 | 37773 | 79388 | 
| 103 | 75 django/conf/locale/bs/formats.py | 5 | 22| 139 | 37912 | 79571 | 
| 104 | 76 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 38201 | 79876 | 
| 105 | 77 django/conf/locale/ta/formats.py | 5 | 22| 125 | 38326 | 80045 | 
| 106 | 78 django/conf/locale/hi/formats.py | 5 | 22| 125 | 38451 | 80214 | 
| 107 | 78 django/db/models/fields/__init__.py | 1559 | 1572| 116 | 38567 | 80214 | 
| 108 | 79 django/conf/locale/ar/formats.py | 5 | 22| 135 | 38702 | 80393 | 
| 109 | 80 django/conf/locale/he/formats.py | 5 | 22| 142 | 38844 | 80579 | 
| 110 | 80 django/db/models/fields/__init__.py | 2214 | 2254| 289 | 39133 | 80579 | 
| 111 | 81 django/conf/locale/fa/formats.py | 5 | 22| 149 | 39282 | 80772 | 
| 112 | 81 django/utils/formats.py | 1 | 59| 380 | 39662 | 80772 | 
| 113 | 82 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 39932 | 81058 | 
| 114 | 82 django/forms/fields.py | 1011 | 1063| 450 | 40382 | 81058 | 
| 115 | 82 django/db/models/fields/__init__.py | 998 | 1012| 122 | 40504 | 81058 | 
| 116 | 82 django/db/models/fields/__init__.py | 1203 | 1234| 230 | 40734 | 81058 | 
| 117 | 82 django/forms/fields.py | 1 | 44| 361 | 41095 | 81058 | 
| 118 | 83 django/utils/duration.py | 1 | 45| 304 | 41399 | 81363 | 
| 119 | 83 django/db/models/fields/__init__.py | 1035 | 1071| 248 | 41647 | 81363 | 
| 120 | 84 django/forms/forms.py | 408 | 422| 135 | 41782 | 85386 | 
| 121 | 84 django/forms/forms.py | 385 | 406| 172 | 41954 | 85386 | 
| **-> 122 <-** | **84 django/utils/dateparse.py** | 1 | 66| 763 | 42717 | 85386 | 
| 123 | 84 django/db/models/fields/__init__.py | 1286 | 1304| 149 | 42866 | 85386 | 
| 124 | 85 django/conf/locale/gd/formats.py | 5 | 22| 144 | 43010 | 85574 | 
| 125 | 85 django/db/models/fields/__init__.py | 2372 | 2422| 339 | 43349 | 85574 | 
| 126 | 85 django/db/models/fields/__init__.py | 2110 | 2143| 228 | 43577 | 85574 | 
| 127 | 86 django/db/models/functions/datetime.py | 215 | 243| 426 | 44003 | 88253 | 
| 128 | 86 django/forms/fields.py | 263 | 287| 195 | 44198 | 88253 | 
| 129 | 87 django/conf/locale/th/formats.py | 5 | 34| 355 | 44553 | 88653 | 
| 130 | 87 django/db/models/fields/__init__.py | 1763 | 1786| 146 | 44699 | 88653 | 
| 131 | 87 django/db/models/fields/__init__.py | 367 | 393| 199 | 44898 | 88653 | 
| 132 | 87 django/db/models/fields/__init__.py | 2174 | 2190| 185 | 45083 | 88653 | 
| 133 | 87 django/forms/fields.py | 130 | 173| 290 | 45373 | 88653 | 
| 134 | 88 django/contrib/gis/db/models/sql/conversion.py | 43 | 70| 203 | 45576 | 89136 | 
| 135 | 89 django/db/models/base.py | 1195 | 1210| 138 | 45714 | 106754 | 
| 136 | 90 django/conf/locale/vi/formats.py | 5 | 22| 179 | 45893 | 106977 | 
| 137 | 91 django/contrib/postgres/forms/array.py | 197 | 235| 271 | 46164 | 108571 | 
| 138 | 91 django/contrib/postgres/forms/ranges.py | 31 | 81| 360 | 46524 | 108571 | 
| 139 | 92 django/conf/locale/ja/formats.py | 5 | 22| 149 | 46673 | 108764 | 
| 140 | 93 django/db/backends/sqlite3/operations.py | 314 | 328| 148 | 46821 | 112036 | 
| 141 | 94 django/contrib/humanize/templatetags/humanize.py | 177 | 211| 615 | 47436 | 114920 | 
| 142 | 94 django/db/models/fields/__init__.py | 1014 | 1033| 128 | 47564 | 114920 | 
| 143 | 94 django/db/models/fields/__init__.py | 1817 | 1847| 186 | 47750 | 114920 | 
| 144 | 94 django/db/models/fields/__init__.py | 1091 | 1114| 152 | 47902 | 114920 | 
| 145 | 94 django/db/models/fields/__init__.py | 1073 | 1088| 173 | 48075 | 114920 | 
| 146 | 94 django/forms/fields.py | 837 | 861| 177 | 48252 | 114920 | 
| 147 | 94 django/template/defaultfilters.py | 843 | 887| 378 | 48630 | 114920 | 
| 148 | 94 django/utils/dateformat.py | 260 | 271| 121 | 48751 | 114920 | 
| 149 | 94 django/db/models/functions/datetime.py | 187 | 213| 292 | 49043 | 114920 | 
| 150 | 95 django/conf/global_settings.py | 356 | 406| 785 | 49828 | 120727 | 
| 151 | 95 django/forms/fields.py | 864 | 903| 298 | 50126 | 120727 | 
| 152 | 95 django/forms/fields.py | 671 | 710| 293 | 50419 | 120727 | 
| 153 | 95 django/db/models/functions/mixins.py | 1 | 20| 161 | 50580 | 120727 | 
| 154 | 95 django/contrib/humanize/templatetags/humanize.py | 212 | 220| 205 | 50785 | 120727 | 
| 155 | 95 django/db/models/fields/__init__.py | 1417 | 1431| 121 | 50906 | 120727 | 
| 156 | 95 django/contrib/humanize/templatetags/humanize.py | 222 | 260| 370 | 51276 | 120727 | 
| 157 | 95 django/db/models/functions/datetime.py | 245 | 264| 170 | 51446 | 120727 | 
| 158 | 96 django/contrib/gis/gdal/field.py | 60 | 71| 158 | 51604 | 122400 | 
| 159 | 96 django/db/models/fields/__init__.py | 2192 | 2212| 181 | 51785 | 122400 | 
| 160 | 96 django/db/models/fields/__init__.py | 664 | 688| 206 | 51991 | 122400 | 
| 161 | 96 django/db/models/fields/__init__.py | 244 | 306| 448 | 52439 | 122400 | 
| 162 | 96 django/template/defaultfilters.py | 729 | 806| 443 | 52882 | 122400 | 
| 163 | 96 django/utils/dateformat.py | 194 | 258| 536 | 53418 | 122400 | 
| 164 | 97 django/conf/locale/fy/formats.py | 22 | 22| 0 | 53418 | 122552 | 
| 165 | 97 django/db/models/fields/__init__.py | 1150 | 1182| 267 | 53685 | 122552 | 
| 166 | 98 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 53981 | 125787 | 
| 167 | 99 django/http/multipartparser.py | 1 | 40| 195 | 54176 | 131009 | 
| 168 | 100 django/db/models/functions/math.py | 161 | 198| 251 | 54427 | 132419 | 
| 169 | 101 django/utils/datetime_safe.py | 1 | 73| 489 | 54916 | 133219 | 
| 170 | 101 django/forms/fields.py | 210 | 241| 274 | 55190 | 133219 | 
| 171 | 102 django/forms/utils.py | 180 | 215| 272 | 55462 | 134712 | 
| 172 | 102 django/db/models/fields/__init__.py | 1117 | 1149| 237 | 55699 | 134712 | 
| 173 | 102 django/forms/fields.py | 906 | 939| 235 | 55934 | 134712 | 
| 174 | 103 django/forms/models.py | 765 | 786| 194 | 56128 | 146630 | 
| 175 | 103 django/contrib/postgres/forms/array.py | 1 | 37| 258 | 56386 | 146630 | 
| 176 | 104 django/utils/http.py | 1 | 37| 464 | 56850 | 149872 | 
| 177 | 105 django/forms/widgets.py | 1043 | 1062| 144 | 56994 | 157969 | 
| 178 | 105 django/utils/formats.py | 217 | 239| 209 | 57203 | 157969 | 
| 179 | 105 django/forms/widgets.py | 448 | 466| 195 | 57398 | 157969 | 
| 180 | 106 django/db/backends/mysql/validation.py | 33 | 70| 287 | 57685 | 158489 | 
| 181 | 106 django/utils/datetime_safe.py | 76 | 109| 309 | 57994 | 158489 | 
| 182 | 106 django/db/models/fields/__init__.py | 208 | 242| 234 | 58228 | 158489 | 
| 183 | 106 django/db/models/fields/__init__.py | 2145 | 2171| 216 | 58444 | 158489 | 
| 184 | 107 django/db/backends/oracle/functions.py | 1 | 23| 188 | 58632 | 158677 | 
| 185 | 107 django/db/models/fields/__init__.py | 1387 | 1415| 281 | 58913 | 158677 | 
| 186 | 107 django/contrib/humanize/templatetags/humanize.py | 59 | 78| 179 | 59092 | 158677 | 
| 187 | **107 django/utils/dateparse.py** | 69 | 103| 311 | 59403 | 158677 | 
| 188 | 107 django/db/models/fields/__init__.py | 1850 | 1869| 119 | 59522 | 158677 | 
| 189 | 108 django/forms/formsets.py | 339 | 396| 481 | 60003 | 162721 | 
| 190 | 108 django/contrib/postgres/forms/ranges.py | 1 | 28| 201 | 60204 | 162721 | 
| 191 | 109 django/forms/boundfield.py | 52 | 77| 184 | 60388 | 165114 | 
| 192 | 109 django/utils/dateformat.py | 47 | 126| 557 | 60945 | 165114 | 
| 193 | 109 django/db/models/base.py | 1278 | 1301| 172 | 61117 | 165114 | 
| 194 | 109 django/contrib/postgres/forms/array.py | 168 | 195| 226 | 61343 | 165114 | 
| 195 | 109 django/db/models/fields/__init__.py | 2285 | 2305| 163 | 61506 | 165114 | 
| 196 | 109 django/db/models/fields/__init__.py | 2256 | 2282| 213 | 61719 | 165114 | 


## Patch

```diff
diff --git a/django/utils/dateparse.py b/django/utils/dateparse.py
--- a/django/utils/dateparse.py
+++ b/django/utils/dateparse.py
@@ -42,11 +42,11 @@
 iso8601_duration_re = _lazy_re_compile(
     r'^(?P<sign>[-+]?)'
     r'P'
-    r'(?:(?P<days>\d+(.\d+)?)D)?'
+    r'(?:(?P<days>\d+([\.,]\d+)?)D)?'
     r'(?:T'
-    r'(?:(?P<hours>\d+(.\d+)?)H)?'
-    r'(?:(?P<minutes>\d+(.\d+)?)M)?'
-    r'(?:(?P<seconds>\d+(.\d+)?)S)?'
+    r'(?:(?P<hours>\d+([\.,]\d+)?)H)?'
+    r'(?:(?P<minutes>\d+([\.,]\d+)?)M)?'
+    r'(?:(?P<seconds>\d+([\.,]\d+)?)S)?'
     r')?'
     r'$'
 )

```

## Test Patch

```diff
diff --git a/tests/forms_tests/field_tests/test_durationfield.py b/tests/forms_tests/field_tests/test_durationfield.py
--- a/tests/forms_tests/field_tests/test_durationfield.py
+++ b/tests/forms_tests/field_tests/test_durationfield.py
@@ -30,6 +30,8 @@ def test_durationfield_clean(self):
         msg = 'Enter a valid duration.'
         with self.assertRaisesMessage(ValidationError, msg):
             f.clean('not_a_time')
+        with self.assertRaisesMessage(ValidationError, msg):
+            DurationField().clean('P3(3D')
 
     def test_durationfield_clean_not_required(self):
         f = DurationField(required=False)
diff --git a/tests/utils_tests/test_dateparse.py b/tests/utils_tests/test_dateparse.py
--- a/tests/utils_tests/test_dateparse.py
+++ b/tests/utils_tests/test_dateparse.py
@@ -161,6 +161,11 @@ def test_iso_8601(self):
             ('-PT0.000005S', timedelta(microseconds=-5)),
             ('-PT0,000005S', timedelta(microseconds=-5)),
             ('-P4DT1H', timedelta(days=-4, hours=-1)),
+            # Invalid separators for decimal fractions.
+            ('P3(3D', None),
+            ('PT3)3H', None),
+            ('PT3|3M', None),
+            ('PT3/3S', None),
         )
         for source, expected in test_values:
             with self.subTest(source=source):

```


## Code snippets

### 1 - django/forms/fields.py:

Start line: 485, End line: 510

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
### 2 - django/forms/fields.py:

Start line: 352, End line: 373

```python
class DecimalField(IntegerField):

    def validate(self, value):
        super().validate(value)
        if value in self.empty_values:
            return
        if not value.is_finite():
            raise ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

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
### 3 - django/forms/fields.py:

Start line: 397, End line: 418

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
### 4 - django/forms/fields.py:

Start line: 1140, End line: 1173

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
### 5 - django/forms/fields.py:

Start line: 290, End line: 322

```python
class FloatField(IntegerField):
    default_error_messages = {
        'invalid': _('Enter a number.'),
    }

    def to_python(self, value):
        """
        Validate that float() can be called on the input. Return the result
        of float() or None for empty values.
        """
        value = super(IntegerField, self).to_python(value)
        if value in self.empty_values:
            return None
        if self.localize:
            value = formats.sanitize_separators(value)
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value

    def validate(self, value):
        super().validate(value)
        if value in self.empty_values:
            return
        if not math.isfinite(value):
            raise ValidationError(self.error_messages['invalid'], code='invalid')

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, NumberInput) and 'step' not in widget.attrs:
            attrs.setdefault('step', 'any')
        return attrs
```
### 6 - django/forms/fields.py:

Start line: 325, End line: 350

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
        try:
            value = Decimal(str(value))
        except DecimalException:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value
```
### 7 - django/forms/fields.py:

Start line: 449, End line: 482

```python
class DateTimeField(BaseTemporalField):
    widget = DateTimeInput
    input_formats = DateTimeFormatsIterator()
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
        try:
            result = parse_datetime(value.strip())
        except ValueError:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        if not result:
            result = super().to_python(value)
        return from_current_timezone(result)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format)
```
### 8 - django/forms/fields.py:

Start line: 376, End line: 394

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
### 9 - django/db/models/fields/__init__.py:

Start line: 1723, End line: 1760

```python
class FloatField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be a float.'),
    }
    description = _("Floating point number")

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise e.__class__(
                "Field '%s' expected a number but got %r." % (self.name, value),
            ) from e

    def get_internal_type(self):
        return "FloatField"

    def to_python(self, value):
        if value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.FloatField,
            **kwargs,
        })
```
### 10 - django/conf/locale/hr/formats.py:

Start line: 5, End line: 43

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
DATETIME_INPUT_FORMATS = [
    '%Y-%m-%d %H:%M:%S',        # '2006-10-25 14:30:59'
    '%Y-%m-%d %H:%M:%S.%f',     # '2006-10-25 14:30:59.000200'
    '%Y-%m-%d %H:%M',           # '2006-10-25 14:30'
    '%d.%m.%Y. %H:%M:%S',       # '25.10.2006. 14:30:59'
    '%d.%m.%Y. %H:%M:%S.%f',    # '25.10.2006. 14:30:59.000200'
    '%d.%m.%Y. %H:%M',          # '25.10.2006. 14:30'
    '%d.%m.%y. %H:%M:%S',       # '25.10.06. 14:30:59'
    '%d.%m.%y. %H:%M:%S.%f',    # '25.10.06. 14:30:59.000200'
    '%d.%m.%y. %H:%M',          # '25.10.06. 14:30'
    '%d. %m. %Y. %H:%M:%S',     # '25. 10. 2006. 14:30:59'
    '%d. %m. %Y. %H:%M:%S.%f',  # '25. 10. 2006. 14:30:59.000200'
    '%d. %m. %Y. %H:%M',        # '25. 10. 2006. 14:30'
    '%d. %m. %y. %H:%M:%S',     # '25. 10. 06. 14:30:59'
    '%d. %m. %y. %H:%M:%S.%f',  # '25. 10. 06. 14:30:59.000200'
    '%d. %m. %y. %H:%M',        # '25. 10. 06. 14:30'
]

DECIMAL_SEPARATOR = ','
THOUSAND_SEPARATOR = '.'
NUMBER_GROUPING = 3
```
### 50 - django/utils/dateparse.py:

Start line: 134, End line: 159

```python
def parse_duration(value):
    """Parse a duration string and return a datetime.timedelta.

    The preferred format for durations in Django is '%d %H:%M:%S.%f'.

    Also supports ISO 8601 representation and PostgreSQL's day-time interval
    format.
    """
    match = (
        standard_duration_re.match(value) or
        iso8601_duration_re.match(value) or
        postgres_interval_re.match(value)
    )
    if match:
        kw = match.groupdict()
        sign = -1 if kw.pop('sign', '+') == '-' else 1
        if kw.get('microseconds'):
            kw['microseconds'] = kw['microseconds'].ljust(6, '0')
        if kw.get('seconds') and kw.get('microseconds') and kw['seconds'].startswith('-'):
            kw['microseconds'] = '-' + kw['microseconds']
        kw = {k: float(v.replace(',', '.')) for k, v in kw.items() if v is not None}
        days = datetime.timedelta(kw.pop('days', .0) or .0)
        if match.re == iso8601_duration_re:
            days *= sign
        return days + sign * datetime.timedelta(**kw)
```
### 122 - django/utils/dateparse.py:

Start line: 1, End line: 66

```python
"""Functions to parse datetime objects."""

# We're using regular expressions rather than time.strptime because:
# - They provide both validation and parsing.
# - They're more flexible for datetimes.
# - The date/datetime/time constructors produce friendlier error messages.

import datetime

from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_fixed_timezone, utc

date_re = _lazy_re_compile(
    r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$'
)

time_re = _lazy_re_compile(
    r'(?P<hour>\d{1,2}):(?P<minute>\d{1,2})'
    r'(?::(?P<second>\d{1,2})(?:[\.,](?P<microsecond>\d{1,6})\d{0,6})?)?$'
)

datetime_re = _lazy_re_compile(
    r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})'
    r'[T ](?P<hour>\d{1,2}):(?P<minute>\d{1,2})'
    r'(?::(?P<second>\d{1,2})(?:[\.,](?P<microsecond>\d{1,6})\d{0,6})?)?'
    r'\s*(?P<tzinfo>Z|[+-]\d{2}(?::?\d{2})?)?$'
)

standard_duration_re = _lazy_re_compile(
    r'^'
    r'(?:(?P<days>-?\d+) (days?, )?)?'
    r'(?P<sign>-?)'
    r'((?:(?P<hours>\d+):)(?=\d+:\d+))?'
    r'(?:(?P<minutes>\d+):)?'
    r'(?P<seconds>\d+)'
    r'(?:[\.,](?P<microseconds>\d{1,6})\d{0,6})?'
    r'$'
)

# Support the sections of ISO 8601 date representation that are accepted by
# timedelta
iso8601_duration_re = _lazy_re_compile(
    r'^(?P<sign>[-+]?)'
    r'P'
    r'(?:(?P<days>\d+(.\d+)?)D)?'
    r'(?:T'
    r'(?:(?P<hours>\d+(.\d+)?)H)?'
    r'(?:(?P<minutes>\d+(.\d+)?)M)?'
    r'(?:(?P<seconds>\d+(.\d+)?)S)?'
    r')?'
    r'$'
)

# Support PostgreSQL's day-time interval format, e.g. "3 days 04:05:06". The
# year-month and mixed intervals cannot be converted to a timedelta and thus
# aren't accepted.
postgres_interval_re = _lazy_re_compile(
    r'^'
    r'(?:(?P<days>-?\d+) (days? ?))?'
    r'(?:(?P<sign>[-+])?'
    r'(?P<hours>\d+):'
    r'(?P<minutes>\d\d):'
    r'(?P<seconds>\d\d)'
    r'(?:\.(?P<microseconds>\d{1,6}))?'
    r')?$'
)
```
### 187 - django/utils/dateparse.py:

Start line: 69, End line: 103

```python
def parse_date(value):
    """Parse a string and return a datetime.date.

    Raise ValueError if the input is well formatted but not a valid date.
    Return None if the input isn't well formatted.
    """
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        if match := date_re.match(value):
            kw = {k: int(v) for k, v in match.groupdict().items()}
            return datetime.date(**kw)


def parse_time(value):
    """Parse a string and return a datetime.time.

    This function doesn't support time zone offsets.

    Raise ValueError if the input is well formatted but not a valid time.
    Return None if the input isn't well formatted, in particular if it
    contains an offset.
    """
    try:
        # The fromisoformat() method takes time zone info into account and
        # returns a time with a tzinfo component, if possible. However, there
        # are no circumstances where aware datetime.time objects make sense, so
        # remove the time zone offset.
        return datetime.time.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        if match := time_re.match(value):
            kw = match.groupdict()
            kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
            kw = {k: int(v) for k, v in kw.items() if v is not None}
            return datetime.time(**kw)
```
