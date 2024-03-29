# pallets__flask-5014

| **pallets/flask** | `7ee9ceb71e868944a46e1ff00b506772a53a4f1d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 395 |
| **Any found context length** | 395 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -190,6 +190,9 @@ def __init__(
             root_path=root_path,
         )
 
+        if not name:
+            raise ValueError("'name' may not be empty.")
+
         if "." in name:
             raise ValueError("'name' may not contain a dot '.' character.")
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/flask/blueprints.py | 193 | 193 | 2 | 1 | 395


## Problem Statement

```
Require a non-empty name for Blueprints
Things do not work correctly if a Blueprint is given an empty name (e.g. #4944).
It would be helpful if a `ValueError` was raised when trying to do that.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 src/flask/blueprints.py** | 208 | 216| 110 | 110 | 5143 | 
| **-> 2 <-** | **1 src/flask/blueprints.py** | 172 | 206| 285 | 395 | 5143 | 
| 3 | **1 src/flask/blueprints.py** | 250 | 266| 166 | 561 | 5143 | 
| 4 | 2 src/flask/wrappers.py | 94 | 133| 258 | 819 | 6401 | 
| 5 | **2 src/flask/blueprints.py** | 117 | 170| 555 | 1374 | 6401 | 
| 6 | **2 src/flask/blueprints.py** | 364 | 402| 356 | 1730 | 6401 | 
| 7 | **2 src/flask/blueprints.py** | 469 | 486| 138 | 1868 | 6401 | 
| 8 | **2 src/flask/blueprints.py** | 587 | 599| 119 | 1987 | 6401 | 
| 9 | **2 src/flask/blueprints.py** | 32 | 83| 412 | 2399 | 6401 | 
| 10 | 2 src/flask/wrappers.py | 75 | 92| 124 | 2523 | 6401 | 
| 11 | **2 src/flask/blueprints.py** | 507 | 524| 138 | 2661 | 6401 | 
| 12 | **2 src/flask/blueprints.py** | 601 | 622| 193 | 2854 | 6401 | 
| 13 | **2 src/flask/blueprints.py** | 435 | 450| 128 | 2982 | 6401 | 
| 14 | **2 src/flask/blueprints.py** | 488 | 505| 156 | 3138 | 6401 | 
| 15 | **2 src/flask/blueprints.py** | 404 | 433| 239 | 3377 | 6401 | 
| 16 | 3 src/flask/templating.py | 36 | 46| 108 | 3485 | 8076 | 
| 17 | **3 src/flask/blueprints.py** | 452 | 467| 146 | 3631 | 8076 | 
| 18 | **3 src/flask/blueprints.py** | 545 | 585| 362 | 3993 | 8076 | 
| 19 | **3 src/flask/blueprints.py** | 526 | 543| 157 | 4150 | 8076 | 
| 20 | 4 src/flask/debughelpers.py | 1 | 38| 287 | 4437 | 9309 | 
| 21 | **4 src/flask/blueprints.py** | 1 | 29| 303 | 4740 | 9309 | 
| 22 | **4 src/flask/blueprints.py** | 218 | 248| 278 | 5018 | 9309 | 
| 23 | **4 src/flask/blueprints.py** | 268 | 362| 774 | 5792 | 9309 | 
| 24 | **4 src/flask/blueprints.py** | 85 | 114| 235 | 6027 | 9309 | 
| 25 | 5 src/flask/app.py | 984 | 1017| 324 | 6351 | 28071 | 
| 26 | 5 src/flask/debughelpers.py | 114 | 159| 407 | 6758 | 28071 | 
| 27 | 6 src/flask/scaffold.py | 76 | 222| 1470 | 8228 | 35781 | 
| 28 | 6 src/flask/scaffold.py | 54 | 74| 216 | 8444 | 35781 | 
| 29 | 7 src/flask/cli.py | 1 | 29| 139 | 8583 | 43370 | 
| 30 | 8 src/flask/signals.py | 1 | 57| 483 | 9066 | 43853 | 
| 31 | 8 src/flask/app.py | 1 | 102| 761 | 9827 | 43853 | 
| 32 | 9 src/flask/typing.py | 1 | 81| 736 | 10563 | 44589 | 
| 33 | 9 src/flask/app.py | 1651 | 1716| 587 | 11150 | 44589 | 
| 34 | 10 src/flask/__init__.py | 1 | 42| 440 | 11590 | 45322 | 
| 35 | 10 src/flask/templating.py | 1 | 18| 118 | 11708 | 45322 | 
| 36 | 10 src/flask/app.py | 1935 | 1967| 292 | 12000 | 45322 | 
| 37 | 10 src/flask/app.py | 522 | 532| 110 | 12110 | 45322 | 
| 38 | 11 examples/celery/src/task_app/views.py | 1 | 39| 235 | 12345 | 45557 | 
| 39 | 11 src/flask/app.py | 1141 | 1153| 112 | 12457 | 45557 | 
| 40 | 11 src/flask/scaffold.py | 669 | 711| 352 | 12809 | 45557 | 
| 41 | 11 src/flask/app.py | 1179 | 1191| 114 | 12923 | 45557 | 
| 42 | 12 src/flask/globals.py | 34 | 70| 371 | 13294 | 46274 | 
| 43 | 12 src/flask/app.py | 1112 | 1139| 201 | 13495 | 46274 | 
| 44 | 12 src/flask/__init__.py | 45 | 93| 293 | 13788 | 46274 | 
| 45 | 12 src/flask/cli.py | 85 | 108| 162 | 13950 | 46274 | 
| 46 | 12 src/flask/app.py | 1155 | 1177| 164 | 14114 | 46274 | 
| 47 | 13 src/flask/sessions.py | 91 | 105| 148 | 14262 | 49754 | 
| 48 | 13 src/flask/scaffold.py | 630 | 653| 232 | 14494 | 49754 | 
| 49 | 13 src/flask/app.py | 713 | 724| 120 | 14614 | 49754 | 
| 50 | 13 src/flask/app.py | 1363 | 1414| 439 | 15053 | 49754 | 
| 51 | 13 src/flask/app.py | 367 | 520| 1441 | 16494 | 49754 | 
| 52 | 13 src/flask/app.py | 700 | 711| 120 | 16614 | 49754 | 
| 53 | 13 src/flask/templating.py | 90 | 124| 256 | 16870 | 49754 | 
| 54 | 13 src/flask/globals.py | 73 | 95| 133 | 17003 | 49754 | 
| 55 | 13 src/flask/scaffold.py | 655 | 667| 144 | 17147 | 49754 | 
| 56 | 13 src/flask/app.py | 1227 | 1259| 253 | 17400 | 49754 | 
| 57 | 13 src/flask/templating.py | 49 | 62| 127 | 17527 | 49754 | 
| 58 | 13 src/flask/scaffold.py | 1 | 51| 407 | 17934 | 49754 | 
| 59 | 13 src/flask/app.py | 1969 | 1994| 217 | 18151 | 49754 | 
| 60 | 13 src/flask/app.py | 623 | 633| 108 | 18259 | 49754 | 
| 61 | 13 src/flask/templating.py | 64 | 88| 210 | 18469 | 49754 | 
| 62 | 13 src/flask/scaffold.py | 579 | 611| 303 | 18772 | 49754 | 
| 63 | 13 src/flask/cli.py | 32 | 82| 370 | 19142 | 49754 | 
| 64 | 13 src/flask/app.py | 2024 | 2055| 278 | 19420 | 49754 | 
| 65 | 13 src/flask/app.py | 551 | 576| 251 | 19671 | 49754 | 
| 66 | 13 src/flask/debughelpers.py | 71 | 94| 165 | 19836 | 49754 | 
| 67 | 13 src/flask/app.py | 1331 | 1361| 242 | 20078 | 49754 | 
| 68 | 13 src/flask/scaffold.py | 713 | 726| 118 | 20196 | 49754 | 
| 69 | 13 src/flask/app.py | 202 | 283| 822 | 21018 | 49754 | 
| 70 | 13 src/flask/globals.py | 1 | 31| 213 | 21231 | 49754 | 
| 71 | 13 src/flask/app.py | 1078 | 1110| 247 | 21478 | 49754 | 
| 72 | 13 src/flask/app.py | 1019 | 1076| 486 | 21964 | 49754 | 
| 73 | 13 src/flask/scaffold.py | 558 | 577| 214 | 22178 | 49754 | 
| 74 | 13 src/flask/app.py | 534 | 549| 162 | 22340 | 49754 | 
| 75 | 13 src/flask/app.py | 1296 | 1329| 277 | 22617 | 49754 | 
| 76 | 13 src/flask/scaffold.py | 531 | 556| 234 | 22851 | 49754 | 
| 77 | 14 src/flask/ctx.py | 25 | 72| 329 | 23180 | 53009 | 
| 78 | 15 src/flask/json/tag.py | 116 | 167| 332 | 23512 | 55126 | 
| 79 | 15 src/flask/templating.py | 127 | 147| 214 | 23726 | 55126 | 
| 80 | 15 src/flask/app.py | 1416 | 1431| 138 | 23864 | 55126 | 
| 81 | 15 src/flask/app.py | 1433 | 1459| 199 | 24063 | 55126 | 
| 82 | 15 src/flask/ctx.py | 74 | 86| 124 | 24187 | 55126 | 
| 83 | 15 src/flask/scaffold.py | 613 | 628| 153 | 24340 | 55126 | 
| 84 | 16 src/flask/testing.py | 23 | 91| 612 | 24952 | 57466 | 
| 85 | 16 src/flask/testing.py | 1 | 20| 118 | 25070 | 57466 | 
| 86 | 16 src/flask/app.py | 851 | 907| 460 | 25530 | 57466 | 
| 87 | 16 src/flask/app.py | 1261 | 1294| 277 | 25807 | 57466 | 
| 88 | 16 src/flask/app.py | 578 | 605| 188 | 25995 | 57466 | 
| 89 | 16 src/flask/app.py | 967 | 982| 136 | 26131 | 57466 | 
| 90 | 17 src/flask/helpers.py | 257 | 279| 212 | 26343 | 63195 | 
| 91 | 17 src/flask/helpers.py | 611 | 654| 349 | 26692 | 63195 | 
| 92 | 17 src/flask/debughelpers.py | 97 | 111| 133 | 26825 | 63195 | 
| 93 | 17 src/flask/app.py | 1593 | 1649| 617 | 27442 | 63195 | 
| 94 | 17 src/flask/app.py | 1533 | 1568| 293 | 27735 | 63195 | 
| 95 | 17 src/flask/app.py | 660 | 698| 304 | 28039 | 63195 | 
| 96 | 17 src/flask/templating.py | 181 | 197| 180 | 28219 | 63195 | 
| 97 | 17 src/flask/helpers.py | 282 | 301| 184 | 28403 | 63195 | 
| 98 | 18 src/flask/logging.py | 1 | 23| 172 | 28575 | 63729 | 
| 99 | 19 examples/tutorial/flaskr/auth.py | 1 | 43| 240 | 28815 | 64409 | 
| 100 | 19 src/flask/cli.py | 564 | 589| 226 | 29041 | 64409 | 
| 101 | 19 src/flask/cli.py | 216 | 237| 180 | 29221 | 64409 | 
| 102 | 20 docs/conf.py | 66 | 99| 213 | 29434 | 65223 | 
| 103 | 20 src/flask/scaffold.py | 320 | 331| 109 | 29543 | 65223 | 
| 104 | 20 src/flask/helpers.py | 657 | 685| 161 | 29704 | 65223 | 
| 105 | 21 examples/tutorial/flaskr/blog.py | 60 | 83| 149 | 29853 | 65999 | 
| 106 | 21 examples/tutorial/flaskr/blog.py | 1 | 25| 157 | 30010 | 65999 | 
| 107 | 21 src/flask/ctx.py | 88 | 110| 201 | 30211 | 65999 | 
| 108 | 21 src/flask/scaffold.py | 773 | 812| 319 | 30530 | 65999 | 
| 109 | 21 src/flask/helpers.py | 179 | 230| 455 | 30985 | 65999 | 
| 110 | 21 src/flask/app.py | 1872 | 1912| 338 | 31323 | 65999 | 
| 111 | 21 src/flask/cli.py | 111 | 184| 562 | 31885 | 65999 | 
| 112 | 21 src/flask/ctx.py | 1 | 22| 124 | 32009 | 65999 | 
| 113 | 21 src/flask/cli.py | 438 | 478| 254 | 32263 | 65999 | 
| 114 | 21 src/flask/app.py | 1821 | 1870| 452 | 32715 | 65999 | 
| 115 | 21 src/flask/cli.py | 421 | 435| 148 | 32863 | 65999 | 
| 116 | 21 src/flask/sessions.py | 158 | 168| 117 | 32980 | 65999 | 
| 117 | 21 src/flask/scaffold.py | 302 | 318| 167 | 33147 | 65999 | 
| 118 | 21 src/flask/scaffold.py | 224 | 276| 377 | 33524 | 65999 | 
| 119 | 21 src/flask/app.py | 1914 | 1933| 169 | 33693 | 65999 | 
| 120 | 21 src/flask/testing.py | 192 | 250| 436 | 34129 | 65999 | 
| 121 | 21 src/flask/helpers.py | 130 | 176| 421 | 34550 | 65999 | 
| 122 | 21 src/flask/cli.py | 393 | 418| 236 | 34786 | 65999 | 
| 123 | 22 examples/tutorial/flaskr/__init__.py | 1 | 51| 315 | 35101 | 66315 | 
| 124 | 22 src/flask/cli.py | 266 | 290| 246 | 35347 | 66315 | 
| 125 | 22 src/flask/testing.py | 175 | 190| 122 | 35469 | 66315 | 
| 126 | 22 src/flask/debughelpers.py | 41 | 68| 238 | 35707 | 66315 | 
| 127 | 22 src/flask/app.py | 635 | 646| 125 | 35832 | 66315 | 
| 128 | 22 src/flask/app.py | 607 | 621| 144 | 35976 | 66315 | 
| 129 | 22 src/flask/app.py | 909 | 965| 540 | 36516 | 66315 | 
| 130 | 22 src/flask/json/tag.py | 188 | 213| 158 | 36674 | 66315 | 
| 131 | 22 src/flask/cli.py | 715 | 772| 353 | 37027 | 66315 | 
| 132 | 22 src/flask/helpers.py | 1 | 33| 229 | 37256 | 66315 | 
| 133 | 23 examples/celery/src/task_app/tasks.py | 1 | 24| 127 | 37383 | 66442 | 
| 134 | 23 src/flask/json/tag.py | 90 | 113| 176 | 37559 | 66442 | 
| 135 | 23 src/flask/app.py | 1504 | 1531| 218 | 37777 | 66442 | 
| 136 | 24 src/flask/config.py | 77 | 99| 207 | 37984 | 69147 | 
| 137 | 24 src/flask/app.py | 1193 | 1225| 281 | 38265 | 69147 | 
| 138 | 24 src/flask/json/tag.py | 216 | 251| 225 | 38490 | 69147 | 
| 139 | 25 src/flask/__main__.py | 1 | 4| 0 | 38490 | 69155 | 
| 140 | 25 src/flask/cli.py | 591 | 610| 192 | 38682 | 69155 | 
| 141 | 25 src/flask/config.py | 1 | 26| 175 | 38857 | 69155 | 
| 142 | 25 examples/tutorial/flaskr/auth.py | 46 | 81| 225 | 39082 | 69155 | 
| 143 | 25 examples/tutorial/flaskr/blog.py | 28 | 57| 121 | 39203 | 69155 | 
| 144 | 25 src/flask/helpers.py | 36 | 48| 107 | 39310 | 69155 | 
| 145 | 25 src/flask/json/tag.py | 170 | 185| 141 | 39451 | 69155 | 
| 146 | 26 examples/celery/src/task_app/__init__.py | 1 | 40| 228 | 39679 | 69384 | 
| 147 | 27 examples/celery/make_celery.py | 1 | 5| 0 | 39679 | 69409 | 
| 148 | 27 src/flask/cli.py | 481 | 545| 599 | 40278 | 69409 | 
| 149 | 27 src/flask/app.py | 2119 | 2173| 497 | 40775 | 69409 | 
| 150 | 27 src/flask/scaffold.py | 439 | 505| 501 | 41276 | 69409 | 
| 151 | 27 src/flask/cli.py | 547 | 562| 135 | 41411 | 69409 | 
| 152 | 27 src/flask/cli.py | 292 | 331| 310 | 41721 | 69409 | 
| 153 | 27 src/flask/app.py | 1732 | 1820| 725 | 42446 | 69409 | 
| 154 | 27 src/flask/cli.py | 240 | 263| 129 | 42575 | 69409 | 
| 155 | 27 src/flask/app.py | 726 | 752| 236 | 42811 | 69409 | 
| 156 | 27 src/flask/scaffold.py | 407 | 437| 237 | 43048 | 69409 | 
| 157 | 27 src/flask/config.py | 101 | 163| 456 | 43504 | 69409 | 
| 158 | 27 src/flask/helpers.py | 99 | 127| 289 | 43793 | 69409 | 
| 159 | 27 src/flask/sessions.py | 170 | 181| 120 | 43913 | 69409 | 
| 160 | 27 src/flask/json/tag.py | 286 | 313| 212 | 44125 | 69409 | 
| 161 | 27 src/flask/json/tag.py | 57 | 87| 265 | 44390 | 69409 | 
| 162 | 27 src/flask/cli.py | 334 | 359| 203 | 44593 | 69409 | 
| 163 | 27 src/flask/app.py | 786 | 850| 702 | 45295 | 69409 | 
| 164 | 27 src/flask/templating.py | 21 | 33| 119 | 45414 | 69409 | 
| 165 | 27 src/flask/app.py | 284 | 365| 763 | 46177 | 69409 | 
| 166 | 28 examples/javascript/js_example/__init__.py | 1 | 6| 0 | 46177 | 69435 | 
| 167 | 28 src/flask/scaffold.py | 815 | 880| 565 | 46742 | 69435 | 
| 168 | 28 src/flask/cli.py | 775 | 821| 355 | 47097 | 69435 | 
| 169 | 28 src/flask/scaffold.py | 507 | 529| 143 | 47240 | 69435 | 
| 170 | 28 src/flask/scaffold.py | 356 | 405| 444 | 47684 | 69435 | 
| 171 | 28 examples/tutorial/flaskr/blog.py | 86 | 126| 236 | 47920 | 69435 | 
| 172 | 28 src/flask/helpers.py | 51 | 97| 344 | 48264 | 69435 | 
| 173 | 28 src/flask/config.py | 165 | 192| 273 | 48537 | 69435 | 
| 174 | 28 src/flask/testing.py | 94 | 118| 239 | 48776 | 69435 | 
| 175 | 28 src/flask/config.py | 280 | 297| 133 | 48909 | 69435 | 
| 176 | 28 src/flask/wrappers.py | 1 | 73| 594 | 49503 | 69435 | 
| 177 | 28 src/flask/logging.py | 53 | 75| 170 | 49673 | 69435 | 
| 178 | 28 src/flask/json/tag.py | 1 | 54| 365 | 50038 | 69435 | 
| 179 | 28 src/flask/app.py | 2082 | 2101| 148 | 50186 | 69435 | 
| 180 | 28 src/flask/cli.py | 362 | 390| 255 | 50441 | 69435 | 
| 181 | 28 src/flask/cli.py | 187 | 213| 177 | 50618 | 69435 | 
| 182 | 28 src/flask/config.py | 29 | 75| 406 | 51024 | 69435 | 
| 183 | 29 examples/tutorial/flaskr/db.py | 1 | 53| 277 | 51301 | 69712 | 
| 184 | 29 src/flask/app.py | 648 | 658| 130 | 51431 | 69712 | 
| 185 | 29 src/flask/json/tag.py | 253 | 284| 252 | 51683 | 69712 | 
| 186 | 29 src/flask/app.py | 105 | 200| 1013 | 52696 | 69712 | 
| 187 | 29 src/flask/app.py | 1718 | 1730| 120 | 52816 | 69712 | 
| 188 | 29 src/flask/cli.py | 936 | 985| 371 | 53187 | 69712 | 
| 189 | 29 src/flask/sessions.py | 1 | 46| 315 | 53502 | 69712 | 
| 190 | 29 src/flask/testing.py | 253 | 287| 329 | 53831 | 69712 | 
| 191 | 29 src/flask/helpers.py | 562 | 608| 416 | 54247 | 69712 | 
| 192 | 30 src/flask/json/provider.py | 123 | 165| 413 | 54660 | 71516 | 
| 193 | 30 src/flask/scaffold.py | 278 | 300| 191 | 54851 | 71516 | 
| 194 | 30 src/flask/scaffold.py | 333 | 354| 209 | 55060 | 71516 | 
| 195 | 30 src/flask/ctx.py | 187 | 216| 227 | 55287 | 71516 | 
| 196 | 30 src/flask/cli.py | 824 | 933| 664 | 55951 | 71516 | 
| 197 | 30 src/flask/app.py | 1486 | 1502| 120 | 56071 | 71516 | 
| 198 | 30 src/flask/app.py | 754 | 784| 247 | 56318 | 71516 | 
| 199 | 31 src/flask/views.py | 1 | 80| 676 | 56994 | 73035 | 
| 200 | 31 src/flask/helpers.py | 304 | 333| 293 | 57287 | 73035 | 
| 201 | 31 src/flask/ctx.py | 146 | 184| 271 | 57558 | 73035 | 
| 202 | 32 examples/javascript/js_example/views.py | 1 | 19| 113 | 57671 | 73148 | 
| 203 | 32 src/flask/logging.py | 26 | 50| 190 | 57861 | 73148 | 
| 204 | 32 src/flask/views.py | 82 | 132| 459 | 58320 | 73148 | 
| 205 | 32 src/flask/helpers.py | 233 | 254| 207 | 58527 | 73148 | 
| 206 | 32 src/flask/ctx.py | 113 | 143| 226 | 58753 | 73148 | 
| 207 | 32 src/flask/app.py | 2103 | 2117| 151 | 58904 | 73148 | 
| 208 | 32 src/flask/app.py | 1461 | 1484| 262 | 59166 | 73148 | 
| 209 | 32 src/flask/scaffold.py | 728 | 770| 325 | 59491 | 73148 | 
| 210 | 32 src/flask/sessions.py | 297 | 323| 222 | 59713 | 73148 | 
| 211 | 32 src/flask/testing.py | 120 | 173| 498 | 60211 | 73148 | 
| 212 | 32 src/flask/ctx.py | 219 | 245| 230 | 60441 | 73148 | 
| 213 | 32 src/flask/app.py | 2057 | 2080| 196 | 60637 | 73148 | 
| 214 | 32 src/flask/scaffold.py | 883 | 922| 318 | 60955 | 73148 | 
| 215 | 32 src/flask/config.py | 232 | 278| 384 | 61339 | 73148 | 
| 216 | 32 src/flask/cli.py | 612 | 651| 406 | 61745 | 73148 | 
| 217 | 32 src/flask/config.py | 194 | 230| 348 | 62093 | 73148 | 
| 218 | 32 src/flask/helpers.py | 391 | 516| 1261 | 63354 | 73148 | 
| 219 | 32 src/flask/ctx.py | 385 | 420| 290 | 63644 | 73148 | 
| 220 | 32 src/flask/views.py | 135 | 189| 386 | 64030 | 73148 | 
| 221 | 32 examples/tutorial/flaskr/auth.py | 84 | 117| 215 | 64245 | 73148 | 
| 222 | 32 src/flask/app.py | 1996 | 2022| 248 | 64493 | 73148 | 
| 223 | 32 src/flask/ctx.py | 422 | 439| 117 | 64610 | 73148 | 
| 224 | 32 src/flask/app.py | 1570 | 1591| 164 | 64774 | 73148 | 
| 225 | 32 src/flask/json/provider.py | 88 | 120| 269 | 65043 | 73148 | 
| 226 | 32 src/flask/app.py | 2175 | 2229| 461 | 65504 | 73148 | 
| 227 | 32 src/flask/ctx.py | 326 | 354| 242 | 65746 | 73148 | 
| 228 | 32 src/flask/sessions.py | 108 | 156| 468 | 66214 | 73148 | 
| 229 | 32 src/flask/cli.py | 988 | 1055| 497 | 66711 | 73148 | 
| 230 | 32 src/flask/json/provider.py | 1 | 86| 656 | 67367 | 73148 | 
| 231 | 32 src/flask/ctx.py | 247 | 275| 212 | 67579 | 73148 | 
| 232 | 32 src/flask/config.py | 299 | 344| 346 | 67925 | 73148 | 
| 233 | 32 src/flask/templating.py | 200 | 213| 139 | 68064 | 73148 | 
| 234 | 32 src/flask/ctx.py | 356 | 383| 243 | 68307 | 73148 | 
| 235 | 32 src/flask/cli.py | 654 | 712| 439 | 68746 | 73148 | 
| 236 | 32 src/flask/templating.py | 150 | 178| 218 | 68964 | 73148 | 
| 237 | 32 src/flask/sessions.py | 281 | 295| 144 | 69108 | 73148 | 
| 238 | 33 src/flask/json/__init__.py | 138 | 171| 311 | 69419 | 74529 | 
| 239 | 33 src/flask/json/provider.py | 190 | 217| 272 | 69691 | 74529 | 
| 240 | 33 src/flask/helpers.py | 519 | 559| 349 | 70040 | 74529 | 
| 241 | 33 src/flask/sessions.py | 360 | 374| 127 | 70167 | 74529 | 
| 242 | 33 src/flask/ctx.py | 278 | 324| 444 | 70611 | 74529 | 
| 243 | 33 src/flask/sessions.py | 376 | 420| 318 | 70929 | 74529 | 
| 244 | 33 src/flask/json/__init__.py | 1 | 44| 316 | 71245 | 74529 | 


## Patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -190,6 +190,9 @@ def __init__(
             root_path=root_path,
         )
 
+        if not name:
+            raise ValueError("'name' may not be empty.")
+
         if "." in name:
             raise ValueError("'name' may not contain a dot '.' character.")
 

```

## Test Patch

```diff
diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -256,6 +256,11 @@ def test_dotted_name_not_allowed(app, client):
         flask.Blueprint("app.ui", __name__)
 
 
+def test_empty_name_not_allowed(app, client):
+    with pytest.raises(ValueError):
+        flask.Blueprint("", __name__)
+
+
 def test_dotted_names_from_app(app, client):
     test = flask.Blueprint("test", __name__)
 

```


## Code snippets

### 1 - src/flask/blueprints.py:

Start line: 208, End line: 216

```python
class Blueprint(Scaffold):

    def _check_setup_finished(self, f_name: str) -> None:
        if self._got_registered_once:
            raise AssertionError(
                f"The setup method '{f_name}' can no longer be called on the blueprint"
                f" '{self.name}'. It has already been registered at least once, any"
                " changes will not be applied consistently.\n"
                "Make sure all imports, decorators, functions, etc. needed to set up"
                " the blueprint are done before registering it."
            )
```
### 2 - src/flask/blueprints.py:

Start line: 172, End line: 206

```python
class Blueprint(Scaffold):

    def __init__(
        self,
        name: str,
        import_name: str,
        static_folder: t.Optional[t.Union[str, os.PathLike]] = None,
        static_url_path: t.Optional[str] = None,
        template_folder: t.Optional[t.Union[str, os.PathLike]] = None,
        url_prefix: t.Optional[str] = None,
        subdomain: t.Optional[str] = None,
        url_defaults: t.Optional[dict] = None,
        root_path: t.Optional[str] = None,
        cli_group: t.Optional[str] = _sentinel,  # type: ignore
    ):
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )

        if "." in name:
            raise ValueError("'name' may not contain a dot '.' character.")

        self.name = name
        self.url_prefix = url_prefix
        self.subdomain = subdomain
        self.deferred_functions: t.List[DeferredSetupFunction] = []

        if url_defaults is None:
            url_defaults = {}

        self.url_values_defaults = url_defaults
        self.cli_group = cli_group
        self._blueprints: t.List[t.Tuple["Blueprint", dict]] = []
```
### 3 - src/flask/blueprints.py:

Start line: 250, End line: 266

```python
class Blueprint(Scaffold):

    @setupmethod
    def register_blueprint(self, blueprint: "Blueprint", **options: t.Any) -> None:
        """Register a :class:`~flask.Blueprint` on this blueprint. Keyword
        arguments passed to this method will override the defaults set
        on the blueprint.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.

        .. versionadded:: 2.0
        """
        if blueprint is self:
            raise ValueError("Cannot register a blueprint on itself")
        self._blueprints.append((blueprint, options))
```
### 4 - src/flask/wrappers.py:

Start line: 94, End line: 133

```python
class Request(RequestBase):

    @property
    def blueprints(self) -> t.List[str]:
        """The registered names of the current blueprint upwards through
        parent blueprints.

        This will be an empty list if there is no current blueprint, or
        if URL matching failed.

        .. versionadded:: 2.0.1
        """
        name = self.blueprint

        if name is None:
            return []

        return _split_blueprint_path(name)

    def _load_form_data(self) -> None:
        super()._load_form_data()

        # In debug mode we're replacing the files multidict with an ad-hoc
        # subclass that raises a different error for key errors.
        if (
            current_app
            and current_app.debug
            and self.mimetype != "multipart/form-data"
            and not self.files
        ):
            from .debughelpers import attach_enctype_error_multidict

            attach_enctype_error_multidict(self)

    def on_json_loading_failed(self, e: t.Optional[ValueError]) -> t.Any:
        try:
            return super().on_json_loading_failed(e)
        except BadRequest as e:
            if current_app and current_app.debug:
                raise

            raise BadRequest() from e
```
### 5 - src/flask/blueprints.py:

Start line: 117, End line: 170

```python
class Blueprint(Scaffold):
    """Represents a blueprint, a collection of routes and other
    app-related functions that can be registered on a real application
    later.

    A blueprint is an object that allows defining application functions
    without requiring an application object ahead of time. It uses the
    same decorators as :class:`~flask.Flask`, but defers the need for an
    application by recording them for later registration.

    Decorating a function with a blueprint creates a deferred function
    that is called with :class:`~flask.blueprints.BlueprintSetupState`
    when the blueprint is registered on an application.

    See :doc:`/blueprints` for more information.

    :param name: The name of the blueprint. Will be prepended to each
        endpoint name.
    :param import_name: The name of the blueprint package, usually
        ``__name__``. This helps locate the ``root_path`` for the
        blueprint.
    :param static_folder: A folder with static files that should be
        served by the blueprint's static route. The path is relative to
        the blueprint's root path. Blueprint static files are disabled
        by default.
    :param static_url_path: The url to serve static files from.
        Defaults to ``static_folder``. If the blueprint does not have
        a ``url_prefix``, the app's static route will take precedence,
        and the blueprint's static files won't be accessible.
    :param template_folder: A folder with templates that should be added
        to the app's template search path. The path is relative to the
        blueprint's root path. Blueprint templates are disabled by
        default. Blueprint templates have a lower precedence than those
        in the app's templates folder.
    :param url_prefix: A path to prepend to all of the blueprint's URLs,
        to make them distinct from the rest of the app's routes.
    :param subdomain: A subdomain that blueprint routes will match on by
        default.
    :param url_defaults: A dict of default values that blueprint routes
        will receive by default.
    :param root_path: By default, the blueprint will automatically set
        this based on ``import_name``. In certain situations this
        automatic detection can fail, so the path can be specified
        manually instead.

    .. versionchanged:: 1.1.0
        Blueprints have a ``cli`` group to register nested CLI commands.
        The ``cli_group`` parameter controls the name of the group under
        the ``flask`` command.

    .. versionadded:: 0.7
    """

    _got_registered_once = False
```
### 6 - src/flask/blueprints.py:

Start line: 364, End line: 402

```python
class Blueprint(Scaffold):

    def register(self, app: "Flask", options: dict) -> None:
        # ... other code

        if self.cli.commands:
            if cli_resolved_group is None:
                app.cli.commands.update(self.cli.commands)
            elif cli_resolved_group is _sentinel:
                self.cli.name = name
                app.cli.add_command(self.cli)
            else:
                self.cli.name = cli_resolved_group
                app.cli.add_command(self.cli)

        for blueprint, bp_options in self._blueprints:
            bp_options = bp_options.copy()
            bp_url_prefix = bp_options.get("url_prefix")
            bp_subdomain = bp_options.get("subdomain")

            if bp_subdomain is None:
                bp_subdomain = blueprint.subdomain

            if state.subdomain is not None and bp_subdomain is not None:
                bp_options["subdomain"] = bp_subdomain + "." + state.subdomain
            elif bp_subdomain is not None:
                bp_options["subdomain"] = bp_subdomain
            elif state.subdomain is not None:
                bp_options["subdomain"] = state.subdomain

            if bp_url_prefix is None:
                bp_url_prefix = blueprint.url_prefix

            if state.url_prefix is not None and bp_url_prefix is not None:
                bp_options["url_prefix"] = (
                    state.url_prefix.rstrip("/") + "/" + bp_url_prefix.lstrip("/")
                )
            elif bp_url_prefix is not None:
                bp_options["url_prefix"] = bp_url_prefix
            elif state.url_prefix is not None:
                bp_options["url_prefix"] = state.url_prefix

            bp_options["name_prefix"] = name
            blueprint.register(app, bp_options)
```
### 7 - src/flask/blueprints.py:

Start line: 469, End line: 486

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_test(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_test], T_template_test]:
        """Register a template test, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_test`.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_test) -> T_template_test:
            self.add_app_template_test(f, name=name)
            return f

        return decorator
```
### 8 - src/flask/blueprints.py:

Start line: 587, End line: 599

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_errorhandler(
        self, code: t.Union[t.Type[Exception], int]
    ) -> t.Callable[[T_error_handler], T_error_handler]:
        """Like :meth:`errorhandler`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.errorhandler`.
        """

        def decorator(f: T_error_handler) -> T_error_handler:
            self.record_once(lambda s: s.app.errorhandler(code)(f))
            return f

        return decorator
```
### 9 - src/flask/blueprints.py:

Start line: 32, End line: 83

```python
class BlueprintSetupState:
    """Temporary holder object for registering a blueprint with the
    application.  An instance of this class is created by the
    :meth:`~flask.Blueprint.make_setup_state` method and later passed
    to all register callback functions.
    """

    def __init__(
        self,
        blueprint: "Blueprint",
        app: "Flask",
        options: t.Any,
        first_registration: bool,
    ) -> None:
        #: a reference to the current application
        self.app = app

        #: a reference to the blueprint that created this setup state.
        self.blueprint = blueprint

        #: a dictionary with all options that were passed to the
        #: :meth:`~flask.Flask.register_blueprint` method.
        self.options = options

        #: as blueprints can be registered multiple times with the
        #: application and not everything wants to be registered
        #: multiple times on it, this attribute can be used to figure
        #: out if the blueprint was registered in the past already.
        self.first_registration = first_registration

        subdomain = self.options.get("subdomain")
        if subdomain is None:
            subdomain = self.blueprint.subdomain

        #: The subdomain that the blueprint should be active for, ``None``
        #: otherwise.
        self.subdomain = subdomain

        url_prefix = self.options.get("url_prefix")
        if url_prefix is None:
            url_prefix = self.blueprint.url_prefix
        #: The prefix that should be used for all URLs defined on the
        #: blueprint.
        self.url_prefix = url_prefix

        self.name = self.options.get("name", blueprint.name)
        self.name_prefix = self.options.get("name_prefix", "")

        #: A dictionary with URL defaults that is added to each and every
        #: URL that was defined with the blueprint.
        self.url_defaults = dict(self.blueprint.url_values_defaults)
        self.url_defaults.update(self.options.get("url_defaults", ()))
```
### 10 - src/flask/wrappers.py:

Start line: 75, End line: 92

```python
class Request(RequestBase):

    @property
    def blueprint(self) -> t.Optional[str]:
        """The registered name of the current blueprint.

        This will be ``None`` if the endpoint is not part of a
        blueprint, or if URL matching failed or has not been performed
        yet.

        This does not necessarily match the name the blueprint was
        created with. It may have been nested, or registered with a
        different name.
        """
        endpoint = self.endpoint

        if endpoint is not None and "." in endpoint:
            return endpoint.rpartition(".")[0]

        return None
```
### 11 - src/flask/blueprints.py:

Start line: 507, End line: 524

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_global(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_global], T_template_global]:
        """Register a template global, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_global`.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_global) -> T_template_global:
            self.add_app_template_global(f, name=name)
            return f

        return decorator
```
### 12 - src/flask/blueprints.py:

Start line: 601, End line: 622

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_url_value_preprocessor(
        self, f: T_url_value_preprocessor
    ) -> T_url_value_preprocessor:
        """Like :meth:`url_value_preprocessor`, but for every request, not only those
        handled by the blueprint. Equivalent to :meth:`.Flask.url_value_preprocessor`.
        """
        self.record_once(
            lambda s: s.app.url_value_preprocessors.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_url_defaults(self, f: T_url_defaults) -> T_url_defaults:
        """Like :meth:`url_defaults`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.url_defaults`.
        """
        self.record_once(
            lambda s: s.app.url_default_functions.setdefault(None, []).append(f)
        )
        return f
```
### 13 - src/flask/blueprints.py:

Start line: 435, End line: 450

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_filter(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_filter], T_template_filter]:
        """Register a template filter, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_filter`.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_filter) -> T_template_filter:
            self.add_app_template_filter(f, name=name)
            return f

        return decorator
```
### 14 - src/flask/blueprints.py:

Start line: 488, End line: 505

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_test(
        self, f: ft.TemplateTestCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a template test, available in any template rendered by the
        application. Works like the :meth:`app_template_test` decorator. Equivalent to
        :meth:`.Flask.add_template_test`.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.tests[name or f.__name__] = f

        self.record_once(register_template)
```
### 15 - src/flask/blueprints.py:

Start line: 404, End line: 433

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_url_rule(
        self,
        rule: str,
        endpoint: t.Optional[str] = None,
        view_func: t.Optional[ft.RouteCallable] = None,
        provide_automatic_options: t.Optional[bool] = None,
        **options: t.Any,
    ) -> None:
        """Register a URL rule with the blueprint. See :meth:`.Flask.add_url_rule` for
        full documentation.

        The URL rule is prefixed with the blueprint's URL prefix. The endpoint name,
        used with :func:`url_for`, is prefixed with the blueprint's name.
        """
        if endpoint and "." in endpoint:
            raise ValueError("'endpoint' may not contain a dot '.' character.")

        if view_func and hasattr(view_func, "__name__") and "." in view_func.__name__:
            raise ValueError("'view_func' name may not contain a dot '.' character.")

        self.record(
            lambda s: s.add_url_rule(
                rule,
                endpoint,
                view_func,
                provide_automatic_options=provide_automatic_options,
                **options,
            )
        )
```
### 17 - src/flask/blueprints.py:

Start line: 452, End line: 467

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_filter(
        self, f: ft.TemplateFilterCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a template filter, available in any template rendered by the
        application. Works like the :meth:`app_template_filter` decorator. Equivalent to
        :meth:`.Flask.add_template_filter`.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.filters[name or f.__name__] = f

        self.record_once(register_template)
```
### 18 - src/flask/blueprints.py:

Start line: 545, End line: 585

```python
class Blueprint(Scaffold):

    @setupmethod
    def before_app_request(self, f: T_before_request) -> T_before_request:
        """Like :meth:`before_request`, but before every request, not only those handled
        by the blueprint. Equivalent to :meth:`.Flask.before_request`.
        """
        self.record_once(
            lambda s: s.app.before_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def after_app_request(self, f: T_after_request) -> T_after_request:
        """Like :meth:`after_request`, but after every request, not only those handled
        by the blueprint. Equivalent to :meth:`.Flask.after_request`.
        """
        self.record_once(
            lambda s: s.app.after_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def teardown_app_request(self, f: T_teardown) -> T_teardown:
        """Like :meth:`teardown_request`, but after every request, not only those
        handled by the blueprint. Equivalent to :meth:`.Flask.teardown_request`.
        """
        self.record_once(
            lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_context_processor(
        self, f: T_template_context_processor
    ) -> T_template_context_processor:
        """Like :meth:`context_processor`, but for templates rendered by every view, not
        only by the blueprint. Equivalent to :meth:`.Flask.context_processor`.
        """
        self.record_once(
            lambda s: s.app.template_context_processors.setdefault(None, []).append(f)
        )
        return f
```
### 19 - src/flask/blueprints.py:

Start line: 526, End line: 543

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_global(
        self, f: ft.TemplateGlobalCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a template global, available in any template rendered by the
        application. Works like the :meth:`app_template_global` decorator. Equivalent to
        :meth:`.Flask.add_template_global`.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.globals[name or f.__name__] = f

        self.record_once(register_template)
```
### 21 - src/flask/blueprints.py:

Start line: 1, End line: 29

```python
import os
import typing as t
from collections import defaultdict
from functools import update_wrapper

from . import typing as ft
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod

if t.TYPE_CHECKING:  # pragma: no cover
    from .app import Flask

DeferredSetupFunction = t.Callable[["BlueprintSetupState"], t.Callable]
T_after_request = t.TypeVar("T_after_request", bound=ft.AfterRequestCallable)
T_before_request = t.TypeVar("T_before_request", bound=ft.BeforeRequestCallable)
T_error_handler = t.TypeVar("T_error_handler", bound=ft.ErrorHandlerCallable)
T_teardown = t.TypeVar("T_teardown", bound=ft.TeardownCallable)
T_template_context_processor = t.TypeVar(
    "T_template_context_processor", bound=ft.TemplateContextProcessorCallable
)
T_template_filter = t.TypeVar("T_template_filter", bound=ft.TemplateFilterCallable)
T_template_global = t.TypeVar("T_template_global", bound=ft.TemplateGlobalCallable)
T_template_test = t.TypeVar("T_template_test", bound=ft.TemplateTestCallable)
T_url_defaults = t.TypeVar("T_url_defaults", bound=ft.URLDefaultCallable)
T_url_value_preprocessor = t.TypeVar(
    "T_url_value_preprocessor", bound=ft.URLValuePreprocessorCallable
)
```
### 22 - src/flask/blueprints.py:

Start line: 218, End line: 248

```python
class Blueprint(Scaffold):

    @setupmethod
    def record(self, func: t.Callable) -> None:
        """Registers a function that is called when the blueprint is
        registered on the application.  This function is called with the
        state as argument as returned by the :meth:`make_setup_state`
        method.
        """
        self.deferred_functions.append(func)

    @setupmethod
    def record_once(self, func: t.Callable) -> None:
        """Works like :meth:`record` but wraps the function in another
        function that will ensure the function is only called once.  If the
        blueprint is registered a second time on the application, the
        function passed is not called.
        """

        def wrapper(state: BlueprintSetupState) -> None:
            if state.first_registration:
                func(state)

        self.record(update_wrapper(wrapper, func))

    def make_setup_state(
        self, app: "Flask", options: dict, first_registration: bool = False
    ) -> BlueprintSetupState:
        """Creates an instance of :meth:`~flask.blueprints.BlueprintSetupState`
        object that is later passed to the register callback functions.
        Subclasses can override this to return a subclass of the setup state.
        """
        return BlueprintSetupState(self, app, options, first_registration)
```
### 23 - src/flask/blueprints.py:

Start line: 268, End line: 362

```python
class Blueprint(Scaffold):

    def register(self, app: "Flask", options: dict) -> None:
        """Called by :meth:`Flask.register_blueprint` to register all
        views and callbacks registered on the blueprint with the
        application. Creates a :class:`.BlueprintSetupState` and calls
        each :meth:`record` callback with it.

        :param app: The application this blueprint is being registered
            with.
        :param options: Keyword arguments forwarded from
            :meth:`~Flask.register_blueprint`.

        .. versionchanged:: 2.3
            Nested blueprints now correctly apply subdomains.

        .. versionchanged:: 2.1
            Registering the same blueprint with the same name multiple
            times is an error.

        .. versionchanged:: 2.0.1
            Nested blueprints are registered with their dotted name.
            This allows different blueprints with the same name to be
            nested at different locations.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.
        """
        name_prefix = options.get("name_prefix", "")
        self_name = options.get("name", self.name)
        name = f"{name_prefix}.{self_name}".lstrip(".")

        if name in app.blueprints:
            bp_desc = "this" if app.blueprints[name] is self else "a different"
            existing_at = f" '{name}'" if self_name != name else ""

            raise ValueError(
                f"The name '{self_name}' is already registered for"
                f" {bp_desc} blueprint{existing_at}. Use 'name=' to"
                f" provide a unique name."
            )

        first_bp_registration = not any(bp is self for bp in app.blueprints.values())
        first_name_registration = name not in app.blueprints

        app.blueprints[name] = self
        self._got_registered_once = True
        state = self.make_setup_state(app, options, first_bp_registration)

        if self.has_static_folder:
            state.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                view_func=self.send_static_file,
                endpoint="static",
            )

        # Merge blueprint data into parent.
        if first_bp_registration or first_name_registration:

            def extend(bp_dict, parent_dict):
                for key, values in bp_dict.items():
                    key = name if key is None else f"{name}.{key}"
                    parent_dict[key].extend(values)

            for key, value in self.error_handler_spec.items():
                key = name if key is None else f"{name}.{key}"
                value = defaultdict(
                    dict,
                    {
                        code: {
                            exc_class: func for exc_class, func in code_values.items()
                        }
                        for code, code_values in value.items()
                    },
                )
                app.error_handler_spec[key] = value

            for endpoint, func in self.view_functions.items():
                app.view_functions[endpoint] = func

            extend(self.before_request_funcs, app.before_request_funcs)
            extend(self.after_request_funcs, app.after_request_funcs)
            extend(
                self.teardown_request_funcs,
                app.teardown_request_funcs,
            )
            extend(self.url_default_functions, app.url_default_functions)
            extend(self.url_value_preprocessors, app.url_value_preprocessors)
            extend(self.template_context_processors, app.template_context_processors)

        for deferred in self.deferred_functions:
            deferred(state)

        cli_resolved_group = options.get("cli_group", self.cli_group)
        # ... other code
```
### 24 - src/flask/blueprints.py:

Start line: 85, End line: 114

```python
class BlueprintSetupState:

    def add_url_rule(
        self,
        rule: str,
        endpoint: t.Optional[str] = None,
        view_func: t.Optional[t.Callable] = None,
        **options: t.Any,
    ) -> None:
        """A helper method to register a rule (and optionally a view function)
        to the application.  The endpoint is automatically prefixed with the
        blueprint's name.
        """
        if self.url_prefix is not None:
            if rule:
                rule = "/".join((self.url_prefix.rstrip("/"), rule.lstrip("/")))
            else:
                rule = self.url_prefix
        options.setdefault("subdomain", self.subdomain)
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)  # type: ignore
        defaults = self.url_defaults
        if "defaults" in options:
            defaults = dict(defaults, **options.pop("defaults"))

        self.app.add_url_rule(
            rule,
            f"{self.name_prefix}.{self.name}.{endpoint}".lstrip("."),
            view_func,
            defaults=defaults,
            **options,
        )
```
