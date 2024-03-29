# django__django-10999

| **django/django** | `36300ef336e3f130a0dadc1143163ff3d23dc843` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 968 |
| **Any found context length** | 968 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/dateparse.py b/django/utils/dateparse.py
--- a/django/utils/dateparse.py
+++ b/django/utils/dateparse.py
@@ -29,9 +29,10 @@
 standard_duration_re = re.compile(
     r'^'
     r'(?:(?P<days>-?\d+) (days?, )?)?'
-    r'((?:(?P<hours>-?\d+):)(?=\d+:\d+))?'
-    r'(?:(?P<minutes>-?\d+):)?'
-    r'(?P<seconds>-?\d+)'
+    r'(?P<sign>-?)'
+    r'((?:(?P<hours>\d+):)(?=\d+:\d+))?'
+    r'(?:(?P<minutes>\d+):)?'
+    r'(?P<seconds>\d+)'
     r'(?:\.(?P<microseconds>\d{1,6})\d{0,6})?'
     r'$'
 )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/dateparse.py | 32 | 34 | 2 | 1 | 968


## Problem Statement

```
Fix parse_duration() for some negative durations
Description
	
The ​https://docs.djangoproject.com/en/2.1/_modules/django/utils/dateparse/ defines:
standard_duration_re = re.compile(
	r'^'
	r'(?:(?P<days>-?\d+) (days?, )?)?'
	r'((?:(?P<hours>-?\d+):)(?=\d+:\d+))?'
	r'(?:(?P<minutes>-?\d+):)?'
	r'(?P<seconds>-?\d+)'
	r'(?:\.(?P<microseconds>\d{1,6})\d{0,6})?'
	r'$'
)
that doesn't match to negative durations, because of the <hours> definition final (lookahead) part does not have '-?' in it. The following will work:
	r'((?:(?P<hours>-?\d+):)(?=-?\d+:-?\d+))?'
(Thanks to Konstantin Senichev for finding the fix.)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/dateparse.py** | 124 | 147| 237 | 237 | 1448 | 
| **-> 2 <-** | **1 django/utils/dateparse.py** | 1 | 65| 731 | 968 | 1448 | 
| 3 | 2 django/forms/fields.py | 469 | 494| 174 | 1142 | 10392 | 
| 4 | 3 django/utils/duration.py | 1 | 45| 304 | 1446 | 10697 | 
| 5 | 4 django/utils/http.py | 157 | 190| 319 | 1765 | 14650 | 
| 6 | **4 django/utils/dateparse.py** | 97 | 121| 258 | 2023 | 14650 | 
| 7 | **4 django/utils/dateparse.py** | 68 | 94| 222 | 2245 | 14650 | 
| 8 | 5 django/conf/locale/ka/formats.py | 23 | 48| 564 | 2809 | 15557 | 
| 9 | 6 django/conf/locale/hr/formats.py | 22 | 48| 620 | 3429 | 16440 | 
| 10 | 7 django/utils/dateformat.py | 184 | 207| 226 | 3655 | 19356 | 
| 11 | 7 django/utils/http.py | 193 | 215| 166 | 3821 | 19356 | 
| 12 | 8 django/conf/locale/it/formats.py | 21 | 46| 564 | 4385 | 20253 | 
| 13 | 9 django/conf/locale/sr/formats.py | 23 | 44| 511 | 4896 | 21102 | 
| 14 | 10 django/conf/locale/ko/formats.py | 32 | 53| 438 | 5334 | 22045 | 
| 15 | 11 django/conf/locale/sr_Latn/formats.py | 23 | 44| 511 | 5845 | 22894 | 
| 16 | 12 django/db/models/functions/mixins.py | 23 | 51| 262 | 6107 | 23317 | 
| 17 | 13 django/conf/locale/sl/formats.py | 22 | 48| 596 | 6703 | 24166 | 
| 18 | 14 django/conf/locale/ru/formats.py | 5 | 33| 402 | 7105 | 24613 | 
| 19 | 15 django/contrib/humanize/templatetags/humanize.py | 218 | 261| 731 | 7836 | 27754 | 
| 20 | 16 django/utils/datetime_safe.py | 1 | 70| 451 | 8287 | 28520 | 
| 21 | 17 django/conf/global_settings.py | 344 | 394| 826 | 9113 | 34115 | 
| 22 | 18 django/conf/locale/cy/formats.py | 5 | 36| 582 | 9695 | 34742 | 
| 23 | 19 django/conf/locale/uk/formats.py | 5 | 38| 460 | 10155 | 35247 | 
| 24 | 20 django/db/models/expressions.py | 452 | 491| 305 | 10460 | 45572 | 
| 25 | 21 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 11168 | 46325 | 
| 26 | 22 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 11803 | 47005 | 
| 27 | 23 django/conf/locale/cs/formats.py | 5 | 43| 640 | 12443 | 47690 | 
| 28 | 24 django/conf/locale/lt/formats.py | 5 | 46| 711 | 13154 | 48446 | 
| 29 | 25 django/conf/locale/sv/formats.py | 5 | 39| 534 | 13688 | 49025 | 
| 30 | 26 django/conf/locale/pt/formats.py | 5 | 39| 630 | 14318 | 49700 | 
| 31 | 27 django/conf/locale/mk/formats.py | 5 | 43| 672 | 14990 | 50417 | 
| 32 | 28 django/conf/locale/id/formats.py | 5 | 50| 708 | 15698 | 51170 | 
| 33 | 29 django/conf/locale/nn/formats.py | 5 | 41| 664 | 16362 | 51879 | 
| 34 | 30 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 16997 | 52559 | 
| 35 | 31 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 17705 | 53312 | 
| 36 | 31 django/contrib/humanize/templatetags/humanize.py | 263 | 301| 370 | 18075 | 53312 | 
| 37 | 32 django/conf/locale/fr/formats.py | 5 | 34| 489 | 18564 | 53846 | 
| 38 | 33 django/conf/locale/fi/formats.py | 5 | 40| 470 | 19034 | 54361 | 
| 39 | 34 django/conf/locale/az/formats.py | 5 | 33| 399 | 19433 | 54805 | 
| 40 | 35 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 19685 | 55073 | 
| 41 | 36 django/conf/locale/lv/formats.py | 5 | 47| 735 | 20420 | 55853 | 
| 42 | 37 django/conf/locale/sk/formats.py | 5 | 30| 348 | 20768 | 56246 | 
| 43 | 38 django/conf/locale/el/formats.py | 5 | 36| 508 | 21276 | 56799 | 
| 44 | 39 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 21538 | 57077 | 
| 45 | 40 django/conf/locale/pt_BR/formats.py | 5 | 34| 494 | 22032 | 57616 | 
| 46 | 41 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 22321 | 57921 | 
| 47 | 42 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 22591 | 58207 | 
| 48 | 43 django/conf/locale/hu/formats.py | 5 | 32| 323 | 22914 | 58575 | 
| 49 | 44 django/conf/locale/en/formats.py | 5 | 41| 663 | 23577 | 59283 | 
| 50 | 45 django/conf/locale/ml/formats.py | 5 | 41| 663 | 24240 | 59991 | 
| 51 | 46 django/conf/locale/ca/formats.py | 5 | 31| 287 | 24527 | 60323 | 
| 52 | 47 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 24943 | 60784 | 
| 53 | 48 django/conf/locale/nb/formats.py | 5 | 40| 646 | 25589 | 61475 | 
| 54 | 49 django/conf/locale/da/formats.py | 5 | 27| 250 | 25839 | 61770 | 
| 55 | 50 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 26114 | 62090 | 
| 56 | 50 django/contrib/humanize/templatetags/humanize.py | 181 | 215| 280 | 26394 | 62090 | 
| 57 | 51 django/conf/locale/pl/formats.py | 5 | 30| 339 | 26733 | 62474 | 
| 58 | 52 django/db/backends/sqlite3/base.py | 541 | 563| 143 | 26876 | 67799 | 
| 59 | 53 django/conf/locale/es/formats.py | 5 | 31| 285 | 27161 | 68129 | 
| 60 | 54 django/conf/locale/eo/formats.py | 5 | 50| 742 | 27903 | 68916 | 
| 61 | 55 django/conf/locale/de/formats.py | 5 | 29| 323 | 28226 | 69284 | 
| 62 | 55 django/utils/dateformat.py | 126 | 140| 127 | 28353 | 69284 | 
| 63 | 56 django/conf/locale/bn/formats.py | 5 | 33| 294 | 28647 | 69622 | 
| 64 | 57 django/template/defaultfilters.py | 691 | 768| 443 | 29090 | 75675 | 
| 65 | 58 django/conf/locale/nl/formats.py | 5 | 71| 479 | 29569 | 77706 | 
| 66 | 59 django/utils/timesince.py | 1 | 24| 220 | 29789 | 78560 | 
| 67 | 59 django/utils/dateformat.py | 302 | 315| 166 | 29955 | 78560 | 
| 68 | 59 django/conf/locale/ka/formats.py | 5 | 22| 298 | 30253 | 78560 | 
| 69 | 60 django/conf/locale/ro/formats.py | 5 | 36| 262 | 30515 | 78867 | 
| 70 | 60 django/utils/timesince.py | 27 | 92| 634 | 31149 | 78867 | 
| 71 | 60 django/conf/locale/hr/formats.py | 5 | 21| 218 | 31367 | 78867 | 
| 72 | 60 django/template/defaultfilters.py | 804 | 848| 378 | 31745 | 78867 | 
| 73 | 61 django/db/models/functions/datetime.py | 190 | 217| 424 | 32169 | 81268 | 
| 74 | 61 django/utils/http.py | 1 | 72| 693 | 32862 | 81268 | 
| 75 | 61 django/forms/fields.py | 1122 | 1155| 293 | 33155 | 81268 | 
| 76 | 61 django/conf/locale/ko/formats.py | 5 | 31| 460 | 33615 | 81268 | 
| 77 | 62 django/conf/locale/is/formats.py | 5 | 22| 130 | 33745 | 81443 | 
| 78 | 62 django/utils/datetime_safe.py | 73 | 106| 313 | 34058 | 81443 | 
| 79 | 62 django/db/backends/sqlite3/base.py | 470 | 491| 367 | 34425 | 81443 | 
| 80 | 62 django/conf/locale/sr/formats.py | 5 | 22| 293 | 34718 | 81443 | 
| 81 | 63 django/conf/locale/tr/formats.py | 5 | 30| 337 | 35055 | 81825 | 
| 82 | 64 django/conf/locale/he/formats.py | 5 | 22| 142 | 35197 | 82011 | 
| 83 | 64 django/conf/locale/sr_Latn/formats.py | 5 | 22| 293 | 35490 | 82011 | 
| 84 | 64 django/utils/dateformat.py | 28 | 41| 121 | 35611 | 82011 | 
| 85 | 65 django/conf/locale/bg/formats.py | 5 | 22| 131 | 35742 | 82186 | 
| 86 | 66 django/conf/locale/ar/formats.py | 5 | 22| 135 | 35877 | 82365 | 
| 87 | 67 django/conf/locale/kn/formats.py | 5 | 22| 123 | 36000 | 82532 | 
| 88 | 67 django/utils/dateformat.py | 317 | 340| 267 | 36267 | 82532 | 
| 89 | 68 django/conf/locale/ta/formats.py | 5 | 22| 125 | 36392 | 82701 | 
| 90 | 69 django/utils/timezone.py | 31 | 57| 182 | 36574 | 84579 | 
| 91 | 70 django/conf/locale/et/formats.py | 5 | 22| 133 | 36707 | 84756 | 
| 92 | 70 django/conf/locale/sl/formats.py | 5 | 20| 208 | 36915 | 84756 | 
| 93 | 71 django/conf/locale/km/formats.py | 5 | 22| 164 | 37079 | 84964 | 
| 94 | 72 django/conf/locale/gl/formats.py | 5 | 22| 170 | 37249 | 85178 | 
| 95 | 73 django/conf/locale/mn/formats.py | 5 | 22| 120 | 37369 | 85342 | 
| 96 | 74 django/conf/locale/hi/formats.py | 5 | 22| 125 | 37494 | 85511 | 
| 97 | 74 django/db/models/functions/datetime.py | 241 | 309| 412 | 37906 | 85511 | 
| 98 | 75 django/utils/dates.py | 1 | 50| 679 | 38585 | 86190 | 
| 99 | 76 django/db/models/fields/__init__.py | 1580 | 1637| 350 | 38935 | 103029 | 
| 100 | 77 django/conf/locale/vi/formats.py | 5 | 22| 179 | 39114 | 103252 | 
| 101 | 78 django/conf/locale/bs/formats.py | 5 | 22| 139 | 39253 | 103435 | 
| 102 | 78 django/utils/dateformat.py | 342 | 368| 176 | 39429 | 103435 | 
| 103 | 79 django/conf/locale/ga/formats.py | 5 | 22| 124 | 39553 | 103603 | 
| 104 | 79 django/utils/dateformat.py | 1 | 25| 191 | 39744 | 103603 | 
| 105 | 79 django/db/models/functions/datetime.py | 63 | 76| 180 | 39924 | 103603 | 
| 106 | 80 django/conf/locale/te/formats.py | 5 | 22| 123 | 40047 | 103770 | 
| 107 | 80 django/db/models/functions/datetime.py | 31 | 61| 300 | 40347 | 103770 | 
| 108 | 81 django/template/base.py | 572 | 607| 359 | 40706 | 111636 | 
| 109 | 82 django/conf/locale/ja/formats.py | 5 | 22| 149 | 40855 | 111829 | 
| 110 | 82 django/conf/locale/it/formats.py | 5 | 20| 288 | 41143 | 111829 | 
| 111 | 83 django/conf/locale/sq/formats.py | 5 | 22| 128 | 41271 | 112001 | 
| 112 | 83 django/db/backends/sqlite3/base.py | 440 | 467| 209 | 41480 | 112001 | 
| 113 | 83 django/db/models/fields/__init__.py | 1289 | 1302| 154 | 41634 | 112001 | 
| 114 | 84 django/conf/locale/th/formats.py | 5 | 34| 355 | 41989 | 112401 | 
| 115 | 85 django/conf/locale/eu/formats.py | 5 | 22| 171 | 42160 | 112617 | 
| 116 | 86 django/utils/formats.py | 1 | 57| 377 | 42537 | 114709 | 
| 117 | 87 django/forms/utils.py | 149 | 179| 229 | 42766 | 115946 | 
| 118 | 88 django/db/backends/oracle/functions.py | 1 | 23| 188 | 42954 | 116134 | 
| 119 | 89 django/contrib/postgres/fields/ranges.py | 167 | 180| 158 | 43112 | 117887 | 
| 120 | 90 django/conf/locale/gd/formats.py | 5 | 22| 144 | 43256 | 118075 | 
| 121 | 91 django/db/backends/utils.py | 134 | 150| 141 | 43397 | 119968 | 
| 122 | 91 django/utils/dateformat.py | 210 | 300| 778 | 44175 | 119968 | 
| 123 | 92 django/conf/locale/fa/formats.py | 5 | 22| 149 | 44324 | 120161 | 
| 124 | 92 django/db/models/functions/datetime.py | 1 | 28| 236 | 44560 | 120161 | 
| 125 | 92 django/db/models/fields/__init__.py | 1114 | 1143| 218 | 44778 | 120161 | 
| 126 | 92 django/contrib/postgres/fields/ranges.py | 152 | 165| 123 | 44901 | 120161 | 
| 127 | 92 django/db/backends/sqlite3/base.py | 407 | 422| 199 | 45100 | 120161 | 
| 128 | 92 django/db/backends/sqlite3/base.py | 494 | 522| 250 | 45350 | 120161 | 
| 129 | 92 django/db/backends/sqlite3/base.py | 525 | 538| 135 | 45485 | 120161 | 
| 130 | 92 django/db/models/expressions.py | 676 | 716| 244 | 45729 | 120161 | 
| 131 | 92 django/db/models/functions/datetime.py | 169 | 188| 204 | 45933 | 120161 | 
| 132 | 92 django/forms/fields.py | 438 | 466| 199 | 46132 | 120161 | 
| 133 | 92 django/db/models/fields/__init__.py | 1146 | 1162| 173 | 46305 | 120161 | 
| 134 | 92 django/utils/dateformat.py | 142 | 153| 151 | 46456 | 120161 | 
| 135 | 92 django/db/models/fields/__init__.py | 1304 | 1345| 332 | 46788 | 120161 | 
| 136 | 92 django/db/backends/utils.py | 153 | 178| 251 | 47039 | 120161 | 
| 137 | 92 django/utils/http.py | 398 | 460| 318 | 47357 | 120161 | 
| 138 | 92 django/utils/dateformat.py | 44 | 124| 544 | 47901 | 120161 | 
| 139 | 92 django/db/backends/sqlite3/base.py | 425 | 437| 127 | 48028 | 120161 | 
| 140 | 92 django/forms/fields.py | 392 | 413| 144 | 48172 | 120161 | 
| 141 | 92 django/db/models/fields/__init__.py | 1164 | 1202| 293 | 48465 | 120161 | 
| 142 | 92 django/db/models/functions/datetime.py | 219 | 238| 164 | 48629 | 120161 | 
| 143 | 93 django/db/models/base.py | 1087 | 1114| 286 | 48915 | 134842 | 
| 144 | 94 django/db/backends/mysql/operations.py | 265 | 280| 254 | 49169 | 137874 | 
| 145 | 94 django/db/backends/sqlite3/base.py | 372 | 404| 251 | 49420 | 137874 | 
| 146 | 95 django/db/backends/sqlite3/operations.py | 66 | 114| 490 | 49910 | 140730 | 
| 147 | 96 django/contrib/admindocs/views.py | 402 | 415| 127 | 50037 | 144040 | 
| 148 | 97 django/contrib/sessions/backends/base.py | 209 | 232| 194 | 50231 | 146565 | 
| 149 | 98 django/views/generic/dates.py | 120 | 163| 285 | 50516 | 151928 | 
| 150 | 98 django/utils/formats.py | 141 | 162| 205 | 50721 | 151928 | 
| 151 | 98 django/contrib/humanize/templatetags/humanize.py | 82 | 128| 578 | 51299 | 151928 | 
| 152 | 98 django/db/models/fields/__init__.py | 2090 | 2106| 183 | 51482 | 151928 | 
| 153 | 98 django/db/models/fields/__init__.py | 2108 | 2149| 325 | 51807 | 151928 | 
| 154 | 99 django/contrib/admindocs/utils.py | 41 | 64| 173 | 51980 | 153828 | 
| 155 | 100 django/core/cache/backends/memcached.py | 38 | 63| 283 | 52263 | 155546 | 
| 156 | 100 django/db/backends/mysql/operations.py | 114 | 124| 130 | 52393 | 155546 | 
| 157 | 101 django/templatetags/tz.py | 37 | 78| 288 | 52681 | 156731 | 
| 158 | 101 django/db/models/fields/__init__.py | 1204 | 1246| 287 | 52968 | 156731 | 
| 159 | 102 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 53252 | 157435 | 
| 160 | 103 django/template/defaulttags.py | 1130 | 1150| 160 | 53412 | 168475 | 
| 161 | 103 django/contrib/postgres/forms/ranges.py | 1 | 63| 419 | 53831 | 168475 | 
| 162 | 103 django/utils/dateformat.py | 155 | 182| 203 | 54034 | 168475 | 
| 163 | 103 django/views/generic/dates.py | 344 | 372| 238 | 54272 | 168475 | 
| 164 | 103 django/utils/http.py | 252 | 263| 138 | 54410 | 168475 | 
| 165 | 104 django/utils/regex_helper.py | 74 | 186| 962 | 55372 | 171008 | 
| 166 | 104 django/views/generic/dates.py | 629 | 725| 806 | 56178 | 171008 | 
| 167 | 105 django/views/i18n.py | 74 | 177| 711 | 56889 | 173484 | 
| 168 | 105 django/templatetags/tz.py | 125 | 145| 176 | 57065 | 173484 | 
| 169 | 106 django/urls/resolvers.py | 139 | 189| 376 | 57441 | 178799 | 
| 170 | 106 django/db/models/functions/datetime.py | 79 | 166| 503 | 57944 | 178799 | 
| 171 | 106 django/contrib/postgres/fields/ranges.py | 1 | 62| 421 | 58365 | 178799 | 
| 172 | 107 django/utils/baseconv.py | 72 | 102| 243 | 58608 | 179586 | 
| 173 | 107 django/utils/formats.py | 187 | 207| 237 | 58845 | 179586 | 
| 174 | 107 django/db/backends/mysql/operations.py | 88 | 112| 350 | 59195 | 179586 | 
| 175 | 108 django/http/multipartparser.py | 101 | 148| 368 | 59563 | 184589 | 
| 176 | 108 django/db/models/fields/__init__.py | 1347 | 1396| 342 | 59905 | 184589 | 
| 177 | 108 django/db/models/fields/__init__.py | 1268 | 1286| 149 | 60054 | 184589 | 
| 178 | 108 django/utils/http.py | 143 | 154| 119 | 60173 | 184589 | 
| 179 | 108 django/views/generic/dates.py | 1 | 65| 420 | 60593 | 184589 | 
| 180 | 109 django/contrib/gis/measure.py | 257 | 295| 438 | 61031 | 187501 | 
| 181 | 109 django/db/backends/mysql/operations.py | 33 | 50| 268 | 61299 | 187501 | 
| 182 | 109 django/template/defaultfilters.py | 94 | 163| 624 | 61923 | 187501 | 
| 183 | 109 django/contrib/sessions/backends/base.py | 234 | 253| 150 | 62073 | 187501 | 
| 184 | 110 django/utils/cache.py | 195 | 219| 235 | 62308 | 191050 | 
| 185 | 110 django/db/backends/sqlite3/operations.py | 208 | 237| 198 | 62506 | 191050 | 
| 186 | 110 django/db/models/fields/__init__.py | 1398 | 1426| 281 | 62787 | 191050 | 
| 187 | 110 django/db/backends/mysql/operations.py | 52 | 69| 204 | 62991 | 191050 | 
| 188 | 111 django/db/backends/base/operations.py | 482 | 523| 284 | 63275 | 196436 | 
| 189 | 111 django/utils/formats.py | 235 | 258| 202 | 63477 | 196436 | 
| 191 | 112 django/forms/fields.py | 371 | 389| 134 | 64135 | 199509 | 
| 192 | 112 django/contrib/postgres/fields/ranges.py | 208 | 270| 332 | 64467 | 199509 | 


### Hint

```
Please give an example valid that's not working. There are ​some tests for negative values.
Right, this should have been fixed by #27699 which is included in 1.11.x.
Example cases, can be discussed: parse_duration('-00:01:01') => plus 61 seconds, so it is not -(00:01:01) but (-00):(+01):(+01) parse_duration('00:-01:-01) => None , leading zeros will prevent parsing parse_duration('-01:01') => minus 59 seconds parse_duration('-01:-01') => minus 61 seconds The fix presented would allow the second line to be parsed (which would help with generated durations). And some instructions in the function/documentation/wiki would be useful, to clarify how the minus sign affects in duration.
The fix from #27699 may not be entirely correct. I agree with your first and third examples. I'd expect a leading minus sign to negate the entire value so they would be minus 61 seconds. I think the second and fourth examples are invalid. I don't think a minus sign after a colon is valid.
Thanks for the extra details. I agree with Tim that everything but a leading - seems like an invalid value that happened to work because of an inappropriate pattern as ​it was never tested.
```

## Patch

```diff
diff --git a/django/utils/dateparse.py b/django/utils/dateparse.py
--- a/django/utils/dateparse.py
+++ b/django/utils/dateparse.py
@@ -29,9 +29,10 @@
 standard_duration_re = re.compile(
     r'^'
     r'(?:(?P<days>-?\d+) (days?, )?)?'
-    r'((?:(?P<hours>-?\d+):)(?=\d+:\d+))?'
-    r'(?:(?P<minutes>-?\d+):)?'
-    r'(?P<seconds>-?\d+)'
+    r'(?P<sign>-?)'
+    r'((?:(?P<hours>\d+):)(?=\d+:\d+))?'
+    r'(?:(?P<minutes>\d+):)?'
+    r'(?P<seconds>\d+)'
     r'(?:\.(?P<microseconds>\d{1,6})\d{0,6})?'
     r'$'
 )

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_dateparse.py b/tests/utils_tests/test_dateparse.py
--- a/tests/utils_tests/test_dateparse.py
+++ b/tests/utils_tests/test_dateparse.py
@@ -113,9 +113,12 @@ def test_negative(self):
         test_values = (
             ('-4 15:30', timedelta(days=-4, minutes=15, seconds=30)),
             ('-172800', timedelta(days=-2)),
-            ('-15:30', timedelta(minutes=-15, seconds=30)),
-            ('-1:15:30', timedelta(hours=-1, minutes=15, seconds=30)),
+            ('-15:30', timedelta(minutes=-15, seconds=-30)),
+            ('-1:15:30', timedelta(hours=-1, minutes=-15, seconds=-30)),
             ('-30.1', timedelta(seconds=-30, milliseconds=-100)),
+            ('-00:01:01', timedelta(minutes=-1, seconds=-1)),
+            ('-01:01', timedelta(seconds=-61)),
+            ('-01:-01', None),
         )
         for source, expected in test_values:
             with self.subTest(source=source):

```


## Code snippets

### 1 - django/utils/dateparse.py:

Start line: 124, End line: 147

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
        days = datetime.timedelta(float(kw.pop('days', 0) or 0))
        sign = -1 if kw.pop('sign', '+') == '-' else 1
        if kw.get('microseconds'):
            kw['microseconds'] = kw['microseconds'].ljust(6, '0')
        if kw.get('seconds') and kw.get('microseconds') and kw['seconds'].startswith('-'):
            kw['microseconds'] = '-' + kw['microseconds']
        kw = {k: float(v) for k, v in kw.items() if v is not None}
        return days + sign * datetime.timedelta(**kw)
```
### 2 - django/utils/dateparse.py:

Start line: 1, End line: 65

```python
"""Functions to parse datetime objects."""

# We're using regular expressions rather than time.strptime because:
# - They provide both validation and parsing.
# - They're more flexible for datetimes.
# - The date/datetime/time constructors produce friendlier error messages.

import datetime
import re

from django.utils.timezone import get_fixed_timezone, utc

date_re = re.compile(
    r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$'
)

time_re = re.compile(
    r'(?P<hour>\d{1,2}):(?P<minute>\d{1,2})'
    r'(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?'
)

datetime_re = re.compile(
    r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})'
    r'[T ](?P<hour>\d{1,2}):(?P<minute>\d{1,2})'
    r'(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?'
    r'(?P<tzinfo>Z|[+-]\d{2}(?::?\d{2})?)?$'
)

standard_duration_re = re.compile(
    r'^'
    r'(?:(?P<days>-?\d+) (days?, )?)?'
    r'((?:(?P<hours>-?\d+):)(?=\d+:\d+))?'
    r'(?:(?P<minutes>-?\d+):)?'
    r'(?P<seconds>-?\d+)'
    r'(?:\.(?P<microseconds>\d{1,6})\d{0,6})?'
    r'$'
)

# Support the sections of ISO 8601 date representation that are accepted by
# timedelta
iso8601_duration_re = re.compile(
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
postgres_interval_re = re.compile(
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
### 3 - django/forms/fields.py:

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
### 4 - django/utils/duration.py:

Start line: 1, End line: 45

```python
import datetime


def _get_duration_components(duration):
    days = duration.days
    seconds = duration.seconds
    microseconds = duration.microseconds

    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes // 60
    minutes = minutes % 60

    return days, hours, minutes, seconds, microseconds


def duration_string(duration):
    """Version of str(timedelta) which is not English specific."""
    days, hours, minutes, seconds, microseconds = _get_duration_components(duration)

    string = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    if days:
        string = '{} '.format(days) + string
    if microseconds:
        string += '.{:06d}'.format(microseconds)

    return string


def duration_iso_string(duration):
    if duration < datetime.timedelta(0):
        sign = '-'
        duration *= -1
    else:
        sign = ''

    days, hours, minutes, seconds, microseconds = _get_duration_components(duration)
    ms = '.{:06d}'.format(microseconds) if microseconds else ""
    return '{}P{}DT{:02d}H{:02d}M{:02d}{}S'.format(sign, days, hours, minutes, seconds, ms)


def duration_microseconds(delta):
    return (24 * 60 * 60 * delta.days + delta.seconds) * 1000000 + delta.microseconds
```
### 5 - django/utils/http.py:

Start line: 157, End line: 190

```python
def parse_http_date(date):
    """
    Parse a date format as specified by HTTP RFC7231 section 7.1.1.1.

    The three formats allowed by the RFC are accepted, even if only the first
    one is still in widespread use.

    Return an integer expressed in seconds since the epoch, in UTC.
    """
    # email.utils.parsedate() does the job for RFC1123 dates; unfortunately
    # RFC7231 makes it mandatory to support RFC850 dates too. So we roll
    # our own RFC-compliant parsing.
    for regex in RFC1123_DATE, RFC850_DATE, ASCTIME_DATE:
        m = regex.match(date)
        if m is not None:
            break
    else:
        raise ValueError("%r is not in a valid HTTP date format" % date)
    try:
        year = int(m.group('year'))
        if year < 100:
            if year < 70:
                year += 2000
            else:
                year += 1900
        month = MONTHS.index(m.group('mon').lower()) + 1
        day = int(m.group('day'))
        hour = int(m.group('hour'))
        min = int(m.group('min'))
        sec = int(m.group('sec'))
        result = datetime.datetime(year, month, day, hour, min, sec)
        return calendar.timegm(result.utctimetuple())
    except Exception as exc:
        raise ValueError("%r is not a valid date" % date) from exc
```
### 6 - django/utils/dateparse.py:

Start line: 97, End line: 121

```python
def parse_datetime(value):
    """Parse a string and return a datetime.datetime.

    This function supports time zone offsets. When the input contains one,
    the output uses a timezone with a fixed offset from UTC.

    Raise ValueError if the input is well formatted but not a valid datetime.
    Return None if the input isn't well formatted.
    """
    match = datetime_re.match(value)
    if match:
        kw = match.groupdict()
        kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
        tzinfo = kw.pop('tzinfo')
        if tzinfo == 'Z':
            tzinfo = utc
        elif tzinfo is not None:
            offset_mins = int(tzinfo[-2:]) if len(tzinfo) > 3 else 0
            offset = 60 * int(tzinfo[1:3]) + offset_mins
            if tzinfo[0] == '-':
                offset = -offset
            tzinfo = get_fixed_timezone(offset)
        kw = {k: int(v) for k, v in kw.items() if v is not None}
        kw['tzinfo'] = tzinfo
        return datetime.datetime(**kw)
```
### 7 - django/utils/dateparse.py:

Start line: 68, End line: 94

```python
def parse_date(value):
    """Parse a string and return a datetime.date.

    Raise ValueError if the input is well formatted but not a valid date.
    Return None if the input isn't well formatted.
    """
    match = date_re.match(value)
    if match:
        kw = {k: int(v) for k, v in match.groupdict().items()}
        return datetime.date(**kw)


def parse_time(value):
    """Parse a string and return a datetime.time.

    This function doesn't support time zone offsets.

    Raise ValueError if the input is well formatted but not a valid time.
    Return None if the input isn't well formatted, in particular if it
    contains an offset.
    """
    match = time_re.match(value)
    if match:
        kw = match.groupdict()
        kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
        kw = {k: int(v) for k, v in kw.items() if v is not None}
        return datetime.time(**kw)
```
### 8 - django/conf/locale/ka/formats.py:

Start line: 23, End line: 48

```python
DATETIME_INPUT_FORMATS = [
    '%Y-%m-%d %H:%M:%S',     # '2006-10-25 14:30:59'
    '%Y-%m-%d %H:%M:%S.%f',  # '2006-10-25 14:30:59.000200'
    '%Y-%m-%d %H:%M',        # '2006-10-25 14:30'
    '%Y-%m-%d',              # '2006-10-25'
    '%d.%m.%Y %H:%M:%S',     # '25.10.2006 14:30:59'
    '%d.%m.%Y %H:%M:%S.%f',  # '25.10.2006 14:30:59.000200'
    '%d.%m.%Y %H:%M',        # '25.10.2006 14:30'
    '%d.%m.%Y',              # '25.10.2006'
    '%d.%m.%y %H:%M:%S',     # '25.10.06 14:30:59'
    '%d.%m.%y %H:%M:%S.%f',  # '25.10.06 14:30:59.000200'
    '%d.%m.%y %H:%M',        # '25.10.06 14:30'
    '%d.%m.%y',              # '25.10.06'
    '%m/%d/%Y %H:%M:%S',     # '10/25/2006 14:30:59'
    '%m/%d/%Y %H:%M:%S.%f',  # '10/25/2006 14:30:59.000200'
    '%m/%d/%Y %H:%M',        # '10/25/2006 14:30'
    '%m/%d/%Y',              # '10/25/2006'
    '%m/%d/%y %H:%M:%S',     # '10/25/06 14:30:59'
    '%m/%d/%y %H:%M:%S.%f',  # '10/25/06 14:30:59.000200'
    '%m/%d/%y %H:%M',        # '10/25/06 14:30'
    '%m/%d/%y',              # '10/25/06'
]
DECIMAL_SEPARATOR = '.'
THOUSAND_SEPARATOR = " "
NUMBER_GROUPING = 3
```
### 9 - django/conf/locale/hr/formats.py:

Start line: 22, End line: 48

```python
DATETIME_INPUT_FORMATS = [
    '%Y-%m-%d %H:%M:%S',        # '2006-10-25 14:30:59'
    '%Y-%m-%d %H:%M:%S.%f',     # '2006-10-25 14:30:59.000200'
    '%Y-%m-%d %H:%M',           # '2006-10-25 14:30'
    '%Y-%m-%d',                 # '2006-10-25'
    '%d.%m.%Y. %H:%M:%S',       # '25.10.2006. 14:30:59'
    '%d.%m.%Y. %H:%M:%S.%f',    # '25.10.2006. 14:30:59.000200'
    '%d.%m.%Y. %H:%M',          # '25.10.2006. 14:30'
    '%d.%m.%Y.',                # '25.10.2006.'
    '%d.%m.%y. %H:%M:%S',       # '25.10.06. 14:30:59'
    '%d.%m.%y. %H:%M:%S.%f',    # '25.10.06. 14:30:59.000200'
    '%d.%m.%y. %H:%M',          # '25.10.06. 14:30'
    '%d.%m.%y.',                # '25.10.06.'
    '%d. %m. %Y. %H:%M:%S',     # '25. 10. 2006. 14:30:59'
    '%d. %m. %Y. %H:%M:%S.%f',  # '25. 10. 2006. 14:30:59.000200'
    '%d. %m. %Y. %H:%M',        # '25. 10. 2006. 14:30'
    '%d. %m. %Y.',              # '25. 10. 2006.'
    '%d. %m. %y. %H:%M:%S',     # '25. 10. 06. 14:30:59'
    '%d. %m. %y. %H:%M:%S.%f',  # '25. 10. 06. 14:30:59.000200'
    '%d. %m. %y. %H:%M',        # '25. 10. 06. 14:30'
    '%d. %m. %y.',              # '25. 10. 06.'
]

DECIMAL_SEPARATOR = ','
THOUSAND_SEPARATOR = '.'
NUMBER_GROUPING = 3
```
### 10 - django/utils/dateformat.py:

Start line: 184, End line: 207

```python
class TimeFormat(Formatter):

    def Z(self):
        """
        Time zone offset in seconds (i.e. '-43200' to '43200'). The offset for
        timezones west of UTC is always negative, and for those east of UTC is
        always positive.

        If timezone information is not available, return an empty string.
        """
        if not self.timezone:
            return ""

        try:
            offset = self.timezone.utcoffset(self.data)
        except Exception:
            # pytz raises AmbiguousTimeError during the autumn DST change.
            # This happens mainly when __init__ receives a naive datetime
            # and sets self.timezone = get_default_timezone().
            return ""

        # `offset` is a datetime.timedelta. For negative values (to the west of
        # UTC) only days can be negative (days=-1) and seconds are always
        # positive. e.g. UTC-1 -> timedelta(days=-1, seconds=82800, microseconds=0)
        # Positive offsets have days=0
        return offset.days * 86400 + offset.seconds
```
