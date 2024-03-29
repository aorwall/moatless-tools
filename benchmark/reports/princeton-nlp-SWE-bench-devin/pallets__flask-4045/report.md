# pallets__flask-4045

| **pallets/flask** | `d8c37f43724cd9fb0870f77877b7c4c7e38a19e0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1088 |
| **Any found context length** | 172 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -188,6 +188,10 @@ def __init__(
             template_folder=template_folder,
             root_path=root_path,
         )
+
+        if "." in name:
+            raise ValueError("'name' may not contain a dot '.' character.")
+
         self.name = name
         self.url_prefix = url_prefix
         self.subdomain = subdomain
@@ -360,12 +364,12 @@ def add_url_rule(
         """Like :meth:`Flask.add_url_rule` but for a blueprint.  The endpoint for
         the :func:`url_for` function is prefixed with the name of the blueprint.
         """
-        if endpoint:
-            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
-        if view_func and hasattr(view_func, "__name__"):
-            assert (
-                "." not in view_func.__name__
-            ), "Blueprint view function name should not contain dots"
+        if endpoint and "." in endpoint:
+            raise ValueError("'endpoint' may not contain a dot '.' character.")
+
+        if view_func and hasattr(view_func, "__name__") and "." in view_func.__name__:
+            raise ValueError("'view_func' name may not contain a dot '.' character.")
+
         self.record(lambda s: s.add_url_rule(rule, endpoint, view_func, **options))
 
     def app_template_filter(self, name: t.Optional[str] = None) -> t.Callable:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/flask/blueprints.py | 191 | 191 | 3 | 1 | 1088
| src/flask/blueprints.py | 363 | 368 | 1 | 1 | 172


## Problem Statement

```
Raise error when blueprint name contains a dot
This is required since every dot is now significant since blueprints can be nested. An error was already added for endpoint names in 1.0, but should have been added for this as well.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/flask/blueprints.py** | 353 | 369| 172 | 172 | 4417 | 
| 2 | **1 src/flask/blueprints.py** | 108 | 169| 641 | 813 | 4417 | 
| **-> 3 <-** | **1 src/flask/blueprints.py** | 171 | 204| 275 | 1088 | 4417 | 
| 4 | **1 src/flask/blueprints.py** | 77 | 105| 233 | 1321 | 4417 | 
| 5 | **1 src/flask/blueprints.py** | 435 | 449| 121 | 1442 | 4417 | 
| 6 | **1 src/flask/blueprints.py** | 371 | 383| 111 | 1553 | 4417 | 
| 7 | **1 src/flask/blueprints.py** | 401 | 415| 121 | 1674 | 4417 | 
| 8 | **1 src/flask/blueprints.py** | 1 | 22| 155 | 1829 | 4417 | 
| 9 | **1 src/flask/blueprints.py** | 451 | 467| 153 | 1982 | 4417 | 
| 10 | **1 src/flask/blueprints.py** | 255 | 351| 754 | 2736 | 4417 | 
| 11 | **1 src/flask/blueprints.py** | 206 | 222| 131 | 2867 | 4417 | 
| 12 | **1 src/flask/blueprints.py** | 385 | 399| 142 | 3009 | 4417 | 
| 13 | **1 src/flask/blueprints.py** | 417 | 433| 152 | 3161 | 4417 | 
| 14 | 2 src/flask/templating.py | 33 | 43| 108 | 3269 | 5642 | 
| 15 | **2 src/flask/blueprints.py** | 469 | 543| 637 | 3906 | 5642 | 
| 16 | 3 src/flask/debughelpers.py | 1 | 40| 297 | 4203 | 7019 | 
| 17 | **3 src/flask/blueprints.py** | 224 | 253| 279 | 4482 | 7019 | 
| 18 | **3 src/flask/blueprints.py** | 25 | 75| 399 | 4881 | 7019 | 
| 19 | 4 src/flask/app.py | 1003 | 1030| 264 | 5145 | 24479 | 
| 20 | 5 src/flask/__init__.py | 1 | 47| 471 | 5616 | 24950 | 
| 21 | 5 src/flask/app.py | 1791 | 1812| 151 | 5767 | 24950 | 
| 22 | 6 src/flask/scaffold.py | 89 | 235| 1437 | 7204 | 31956 | 
| 23 | 6 src/flask/scaffold.py | 59 | 87| 316 | 7520 | 31956 | 
| 24 | 7 src/flask/helpers.py | 267 | 339| 550 | 8070 | 38825 | 
| 25 | 7 src/flask/helpers.py | 191 | 266| 850 | 8920 | 38825 | 
| 26 | 7 src/flask/app.py | 386 | 522| 1263 | 10183 | 38825 | 
| 27 | 7 src/flask/app.py | 1445 | 1462| 150 | 10333 | 38825 | 
| 28 | 7 src/flask/debughelpers.py | 75 | 93| 149 | 10482 | 38825 | 
| 29 | 7 src/flask/scaffold.py | 642 | 679| 274 | 10756 | 38825 | 
| 30 | 7 src/flask/app.py | 1 | 98| 662 | 11418 | 38825 | 
| 31 | 7 src/flask/app.py | 1032 | 1089| 485 | 11903 | 38825 | 
| 32 | 7 src/flask/debughelpers.py | 113 | 172| 513 | 12416 | 38825 | 
| 33 | 7 src/flask/app.py | 1246 | 1275| 233 | 12649 | 38825 | 
| 34 | 8 src/flask/typing.py | 1 | 47| 425 | 13074 | 39250 | 
| 35 | 8 src/flask/scaffold.py | 681 | 709| 223 | 13297 | 39250 | 
| 36 | 9 src/flask/cli.py | 1 | 34| 141 | 13438 | 46488 | 
| 37 | 9 src/flask/app.py | 1347 | 1377| 241 | 13679 | 46488 | 
| 38 | 9 src/flask/app.py | 1428 | 1443| 138 | 13817 | 46488 | 
| 39 | 10 src/flask/signals.py | 1 | 57| 480 | 14297 | 46968 | 
| 40 | 10 src/flask/scaffold.py | 1 | 34| 211 | 14508 | 46968 | 
| 41 | 11 docs/conf.py | 1 | 64| 586 | 15094 | 47767 | 
| 42 | 11 src/flask/scaffold.py | 506 | 527| 129 | 15223 | 47767 | 
| 43 | 11 src/flask/app.py | 1814 | 1843| 271 | 15494 | 47767 | 
| 44 | 11 src/flask/app.py | 1379 | 1426| 412 | 15906 | 47767 | 
| 45 | 11 docs/conf.py | 66 | 99| 213 | 16119 | 47767 | 
| 46 | 11 src/flask/app.py | 1164 | 1184| 152 | 16271 | 47767 | 
| 47 | 11 src/flask/app.py | 1277 | 1310| 279 | 16550 | 47767 | 
| 48 | 11 src/flask/app.py | 711 | 732| 207 | 16757 | 47767 | 
| 49 | 12 setup.py | 1 | 17| 105 | 16862 | 47872 | 
| 50 | 12 src/flask/app.py | 283 | 370| 811 | 17673 | 47872 | 
| 51 | 12 src/flask/app.py | 1150 | 1162| 111 | 17784 | 47872 | 
| 52 | 13 src/flask/json/tag.py | 116 | 167| 332 | 18116 | 49989 | 
| 53 | 13 src/flask/app.py | 1871 | 1904| 310 | 18426 | 49989 | 
| 54 | 14 src/flask/config.py | 1 | 25| 172 | 18598 | 52139 | 
| 55 | 14 src/flask/scaffold.py | 438 | 504| 502 | 19100 | 52139 | 
| 56 | 14 src/flask/app.py | 1186 | 1211| 200 | 19300 | 52139 | 
| 57 | 14 src/flask/app.py | 1777 | 1789| 132 | 19432 | 52139 | 
| 58 | 14 src/flask/app.py | 569 | 594| 252 | 19684 | 52139 | 
| 59 | 14 src/flask/helpers.py | 439 | 490| 407 | 20091 | 52139 | 
| 60 | 15 src/flask/globals.py | 1 | 60| 399 | 20490 | 52539 | 
| 61 | 15 src/flask/debughelpers.py | 43 | 72| 281 | 20771 | 52539 | 
| 62 | 15 src/flask/templating.py | 46 | 59| 127 | 20898 | 52539 | 
| 63 | 15 src/flask/app.py | 1091 | 1121| 234 | 21132 | 52539 | 
| 64 | 15 src/flask/app.py | 657 | 674| 155 | 21287 | 52539 | 
| 65 | 15 src/flask/app.py | 1312 | 1345| 277 | 21564 | 52539 | 
| 66 | 15 src/flask/json/tag.py | 188 | 213| 158 | 21722 | 52539 | 
| 67 | 16 examples/tutorial/flaskr/blog.py | 1 | 25| 157 | 21879 | 53315 | 
| 68 | 16 src/flask/app.py | 1123 | 1148| 189 | 22068 | 53315 | 
| 69 | 16 src/flask/json/tag.py | 90 | 113| 176 | 22244 | 53315 | 
| 70 | 16 src/flask/templating.py | 87 | 121| 256 | 22500 | 53315 | 
| 71 | 16 src/flask/json/tag.py | 216 | 251| 225 | 22725 | 53315 | 
| 72 | 17 examples/javascript/js_example/__init__.py | 1 | 6| 0 | 22725 | 53338 | 
| 73 | 18 src/flask/__main__.py | 1 | 4| 0 | 22725 | 53346 | 
| 74 | 18 src/flask/scaffold.py | 237 | 289| 365 | 23090 | 53346 | 
| 75 | 18 src/flask/app.py | 198 | 282| 857 | 23947 | 53346 | 
| 76 | 18 src/flask/json/tag.py | 170 | 185| 141 | 24088 | 53346 | 
| 77 | 18 src/flask/helpers.py | 627 | 647| 165 | 24253 | 53346 | 
| 78 | 18 src/flask/scaffold.py | 37 | 56| 186 | 24439 | 53346 | 
| 79 | 19 src/flask/sessions.py | 90 | 104| 148 | 24587 | 56696 | 
| 80 | 19 src/flask/scaffold.py | 407 | 436| 228 | 24815 | 56696 | 
| 81 | 19 src/flask/app.py | 1534 | 1585| 399 | 25214 | 56696 | 
| 82 | 19 src/flask/json/tag.py | 57 | 87| 265 | 25479 | 56696 | 
| 83 | 20 src/flask/logging.py | 1 | 23| 165 | 25644 | 57223 | 
| 84 | 20 src/flask/json/tag.py | 253 | 284| 252 | 25896 | 57223 | 
| 85 | 21 examples/tutorial/flaskr/auth.py | 1 | 43| 240 | 26136 | 57908 | 
| 86 | 21 src/flask/cli.py | 538 | 554| 153 | 26289 | 57908 | 
| 87 | 21 src/flask/cli.py | 223 | 249| 177 | 26466 | 57908 | 
| 88 | 21 src/flask/json/tag.py | 1 | 54| 365 | 26831 | 57908 | 
| 89 | 22 examples/javascript/setup.py | 1 | 4| 0 | 26831 | 57915 | 
| 90 | 22 src/flask/app.py | 615 | 630| 153 | 26984 | 57915 | 
| 91 | 22 src/flask/cli.py | 981 | 995| 118 | 27102 | 57915 | 
| 92 | 23 src/flask/testing.py | 1 | 91| 735 | 27837 | 60178 | 
| 93 | 23 examples/tutorial/flaskr/blog.py | 60 | 83| 149 | 27986 | 60178 | 
| 94 | 23 src/flask/scaffold.py | 606 | 640| 291 | 28277 | 60178 | 
| 95 | 23 src/flask/scaffold.py | 567 | 604| 308 | 28585 | 60178 | 
| 96 | 23 src/flask/app.py | 1735 | 1775| 338 | 28923 | 60178 | 
| 97 | 23 src/flask/app.py | 524 | 542| 185 | 29108 | 60178 | 
| 98 | 23 src/flask/templating.py | 61 | 85| 210 | 29318 | 60178 | 
| 99 | 24 examples/tutorial/setup.py | 1 | 4| 0 | 29318 | 60185 | 
| 100 | 24 src/flask/scaffold.py | 312 | 328| 167 | 29485 | 60185 | 
| 101 | 24 examples/tutorial/flaskr/auth.py | 46 | 81| 230 | 29715 | 60185 | 
| 102 | 24 src/flask/app.py | 596 | 613| 128 | 29843 | 60185 | 
| 103 | 24 src/flask/cli.py | 122 | 145| 162 | 30005 | 60185 | 
| 104 | 24 src/flask/config.py | 205 | 220| 123 | 30128 | 60185 | 
| 105 | 25 src/flask/wrappers.py | 1 | 99| 792 | 30920 | 61258 | 
| 106 | 25 src/flask/scaffold.py | 738 | 768| 263 | 31183 | 61258 | 
| 107 | 25 src/flask/cli.py | 692 | 733| 249 | 31432 | 61258 | 
| 108 | 25 src/flask/app.py | 879 | 926| 402 | 31834 | 61258 | 
| 109 | 25 src/flask/cli.py | 736 | 776| 337 | 32171 | 61258 | 
| 110 | 25 src/flask/app.py | 2065 | 2077| 128 | 32299 | 61258 | 
| 111 | 26 src/flask/ctx.py | 1 | 71| 445 | 32744 | 65058 | 
| 112 | 26 src/flask/scaffold.py | 529 | 549| 168 | 32912 | 65058 | 
| 113 | 26 src/flask/app.py | 676 | 709| 277 | 33189 | 65058 | 
| 114 | 27 examples/javascript/js_example/views.py | 1 | 19| 114 | 33303 | 65172 | 
| 115 | 27 src/flask/scaffold.py | 366 | 405| 376 | 33679 | 65172 | 
| 116 | 27 src/flask/scaffold.py | 551 | 565| 148 | 33827 | 65172 | 
| 117 | 27 src/flask/app.py | 1692 | 1733| 401 | 34228 | 65172 | 
| 118 | 27 src/flask/cli.py | 476 | 536| 477 | 34705 | 65172 | 
| 119 | 27 src/flask/cli.py | 606 | 660| 405 | 35110 | 65172 | 
| 120 | 27 src/flask/cli.py | 663 | 689| 198 | 35308 | 65172 | 
| 121 | 28 src/flask/json/__init__.py | 70 | 88| 162 | 35470 | 67996 | 
| 122 | 28 src/flask/cli.py | 556 | 575| 192 | 35662 | 67996 | 
| 123 | 28 src/flask/json/__init__.py | 252 | 281| 315 | 35977 | 67996 | 
| 124 | 28 src/flask/debughelpers.py | 96 | 110| 133 | 36110 | 67996 | 
| 125 | 28 src/flask/helpers.py | 342 | 361| 184 | 36294 | 67996 | 
| 126 | 28 src/flask/cli.py | 252 | 275| 198 | 36492 | 67996 | 
| 127 | 28 src/flask/helpers.py | 142 | 188| 416 | 36908 | 67996 | 
| 128 | 28 src/flask/testing.py | 174 | 221| 332 | 37240 | 67996 | 
| 129 | 28 src/flask/cli.py | 37 | 86| 395 | 37635 | 67996 | 
| 130 | 28 src/flask/helpers.py | 751 | 782| 260 | 37895 | 67996 | 
| 131 | 28 src/flask/app.py | 1488 | 1503| 120 | 38015 | 67996 | 
| 132 | 28 src/flask/ctx.py | 73 | 85| 124 | 38139 | 67996 | 
| 133 | 28 examples/tutorial/flaskr/blog.py | 86 | 126| 236 | 38375 | 67996 | 
| 134 | 28 src/flask/app.py | 1213 | 1244| 242 | 38617 | 67996 | 
| 135 | 28 src/flask/sessions.py | 1 | 45| 303 | 38920 | 67996 | 
| 136 | 28 src/flask/scaffold.py | 330 | 341| 110 | 39030 | 67996 | 
| 137 | 28 src/flask/app.py | 632 | 643| 125 | 39155 | 67996 | 
| 138 | 28 src/flask/testing.py | 223 | 244| 173 | 39328 | 67996 | 
| 139 | 28 src/flask/json/__init__.py | 91 | 106| 125 | 39453 | 67996 | 
| 140 | 28 src/flask/app.py | 372 | 384| 159 | 39612 | 67996 | 
| 141 | 28 src/flask/scaffold.py | 291 | 310| 178 | 39790 | 67996 | 
| 142 | 28 src/flask/cli.py | 304 | 360| 448 | 40238 | 67996 | 
| 143 | 28 src/flask/app.py | 101 | 196| 1013 | 41251 | 67996 | 
| 144 | 29 examples/tutorial/flaskr/__init__.py | 1 | 51| 315 | 41566 | 68312 | 
| 145 | 29 src/flask/json/__init__.py | 1 | 40| 306 | 41872 | 68312 | 
| 146 | 29 src/flask/templating.py | 1 | 30| 204 | 42076 | 68312 | 
| 147 | 29 src/flask/cli.py | 961 | 978| 138 | 42214 | 68312 | 
| 148 | 29 src/flask/app.py | 986 | 1001| 141 | 42355 | 68312 | 
| 149 | 29 src/flask/helpers.py | 364 | 393| 293 | 42648 | 68312 | 
| 150 | 29 src/flask/logging.py | 53 | 75| 170 | 42818 | 68312 | 
| 151 | 29 src/flask/scaffold.py | 771 | 820| 434 | 43252 | 68312 | 
| 152 | 29 src/flask/logging.py | 26 | 50| 190 | 43442 | 68312 | 
| 153 | 29 src/flask/scaffold.py | 343 | 364| 209 | 43651 | 68312 | 
| 154 | 29 src/flask/app.py | 544 | 567| 180 | 43831 | 68312 | 
| 155 | 29 src/flask/cli.py | 445 | 473| 255 | 44086 | 68312 | 
| 156 | 29 src/flask/json/__init__.py | 284 | 351| 561 | 44647 | 68312 | 
| 157 | 29 src/flask/cli.py | 577 | 603| 256 | 44903 | 68312 | 
| 158 | 29 examples/tutorial/flaskr/blog.py | 28 | 57| 121 | 45024 | 68312 | 
| 159 | 30 src/flask/views.py | 105 | 129| 174 | 45198 | 69606 | 
| 160 | 30 src/flask/helpers.py | 50 | 62| 106 | 45304 | 69606 | 
| 161 | 30 src/flask/config.py | 76 | 98| 213 | 45517 | 69606 | 
| 162 | 30 src/flask/json/tag.py | 286 | 313| 212 | 45729 | 69606 | 
| 163 | 30 src/flask/app.py | 645 | 655| 130 | 45859 | 69606 | 
| 164 | 30 src/flask/app.py | 784 | 804| 209 | 46068 | 69606 | 
| 165 | 30 src/flask/scaffold.py | 711 | 735| 197 | 46265 | 69606 | 
| 166 | 30 src/flask/cli.py | 427 | 442| 134 | 46399 | 69606 | 
| 167 | 30 src/flask/json/__init__.py | 144 | 180| 308 | 46707 | 69606 | 
| 168 | 30 src/flask/cli.py | 912 | 958| 409 | 47116 | 69606 | 
| 169 | 30 src/flask/config.py | 100 | 126| 259 | 47375 | 69606 | 
| 170 | 30 src/flask/app.py | 734 | 759| 256 | 47631 | 69606 | 
| 171 | 30 src/flask/helpers.py | 650 | 699| 406 | 48037 | 69606 | 
| 172 | 30 src/flask/cli.py | 278 | 301| 129 | 48166 | 69606 | 
| 173 | 30 src/flask/ctx.py | 112 | 134| 180 | 48346 | 69606 | 
| 174 | 30 src/flask/app.py | 928 | 984| 545 | 48891 | 69606 | 
| 175 | 31 examples/tutorial/flaskr/db.py | 1 | 55| 290 | 49181 | 69896 | 
| 176 | 31 src/flask/helpers.py | 702 | 748| 413 | 49594 | 69896 | 
| 177 | 31 src/flask/helpers.py | 36 | 47| 109 | 49703 | 69896 | 
| 178 | 31 src/flask/ctx.py | 137 | 174| 274 | 49977 | 69896 | 
| 179 | 31 src/flask/scaffold.py | 823 | 863| 322 | 50299 | 69896 | 
| 180 | 31 src/flask/testing.py | 94 | 117| 224 | 50523 | 69896 | 
| 181 | 31 examples/tutorial/flaskr/auth.py | 84 | 117| 215 | 50738 | 69896 | 
| 182 | 31 src/flask/helpers.py | 111 | 139| 278 | 51016 | 69896 | 
| 183 | 31 src/flask/config.py | 222 | 267| 346 | 51362 | 69896 | 
| 184 | 31 src/flask/app.py | 1505 | 1532| 217 | 51579 | 69896 | 
| 185 | 31 src/flask/app.py | 806 | 877| 765 | 52344 | 69896 | 
| 186 | 31 src/flask/helpers.py | 65 | 109| 330 | 52674 | 69896 | 
| 187 | 31 src/flask/app.py | 1845 | 1869| 268 | 52942 | 69896 | 
| 188 | 31 src/flask/config.py | 28 | 74| 408 | 53350 | 69896 | 
| 189 | 31 src/flask/config.py | 166 | 203| 301 | 53651 | 69896 | 
| 190 | 31 src/flask/ctx.py | 87 | 109| 202 | 53853 | 69896 | 
| 191 | 31 src/flask/config.py | 128 | 164| 348 | 54201 | 69896 | 
| 192 | 31 src/flask/cli.py | 779 | 858| 558 | 54759 | 69896 | 
| 193 | 31 src/flask/cli.py | 148 | 220| 552 | 55311 | 69896 | 
| 194 | 31 src/flask/app.py | 1906 | 1927| 196 | 55507 | 69896 | 
| 195 | 31 src/flask/app.py | 1929 | 1948| 148 | 55655 | 69896 | 
| 196 | 31 src/flask/sessions.py | 292 | 313| 188 | 55843 | 69896 | 
| 197 | 31 src/flask/json/__init__.py | 109 | 141| 266 | 56109 | 69896 | 
| 198 | 31 src/flask/templating.py | 124 | 151| 219 | 56328 | 69896 | 
| 199 | 31 src/flask/app.py | 1966 | 2020| 496 | 56824 | 69896 | 
| 200 | 31 src/flask/cli.py | 363 | 382| 214 | 57038 | 69896 | 
| 201 | 31 src/flask/sessions.py | 178 | 234| 473 | 57511 | 69896 | 
| 202 | 31 src/flask/views.py | 132 | 158| 236 | 57747 | 69896 | 
| 203 | 31 src/flask/app.py | 1617 | 1691| 610 | 58357 | 69896 | 
| 204 | 31 src/flask/cli.py | 861 | 909| 382 | 58739 | 69896 | 
| 205 | 31 src/flask/ctx.py | 342 | 370| 242 | 58981 | 69896 | 
| 206 | 31 src/flask/ctx.py | 372 | 410| 383 | 59364 | 69896 | 
| 207 | 31 src/flask/helpers.py | 1 | 33| 191 | 59555 | 69896 | 
| 208 | 31 src/flask/ctx.py | 451 | 479| 253 | 59808 | 69896 | 
| 209 | 31 src/flask/cli.py | 89 | 119| 251 | 60059 | 69896 | 
| 210 | 31 src/flask/views.py | 68 | 102| 358 | 60417 | 69896 | 
| 211 | 31 src/flask/ctx.py | 209 | 241| 275 | 60692 | 69896 | 
| 212 | 31 src/flask/testing.py | 247 | 281| 324 | 61016 | 69896 | 
| 213 | 31 src/flask/helpers.py | 493 | 624| 1305 | 62321 | 69896 | 
| 214 | 31 src/flask/templating.py | 154 | 166| 117 | 62438 | 69896 | 
| 215 | 31 src/flask/sessions.py | 150 | 160| 117 | 62555 | 69896 | 
| 216 | 31 src/flask/json/__init__.py | 42 | 67| 220 | 62775 | 69896 | 
| 217 | 31 src/flask/sessions.py | 48 | 87| 381 | 63156 | 69896 | 
| 218 | 31 src/flask/sessions.py | 276 | 290| 144 | 63300 | 69896 | 
| 219 | 31 src/flask/views.py | 1 | 66| 527 | 63827 | 69896 | 
| 220 | 31 src/flask/app.py | 1464 | 1486| 233 | 64060 | 69896 | 
| 221 | 31 src/flask/app.py | 1950 | 1964| 151 | 64211 | 69896 | 
| 222 | 31 src/flask/cli.py | 384 | 424| 324 | 64535 | 69896 | 
| 223 | 31 src/flask/ctx.py | 177 | 206| 226 | 64761 | 69896 | 
| 224 | 31 src/flask/sessions.py | 366 | 405| 305 | 65066 | 69896 | 
| 225 | 31 src/flask/app.py | 1587 | 1615| 214 | 65280 | 69896 | 
| 226 | 31 src/flask/ctx.py | 243 | 263| 191 | 65471 | 69896 | 
| 227 | 31 src/flask/testing.py | 119 | 172| 489 | 65960 | 69896 | 
| 228 | 31 src/flask/app.py | 2022 | 2063| 338 | 66298 | 69896 | 
| 229 | 31 src/flask/sessions.py | 162 | 176| 126 | 66424 | 69896 | 
| 230 | 31 src/flask/ctx.py | 266 | 340| 717 | 67141 | 69896 | 
| 231 | 31 src/flask/json/__init__.py | 219 | 249| 283 | 67424 | 69896 | 
| 232 | 31 src/flask/ctx.py | 412 | 449| 312 | 67736 | 69896 | 
| 233 | 31 src/flask/app.py | 761 | 782| 200 | 67936 | 69896 | 
| 234 | 31 src/flask/sessions.py | 350 | 364| 127 | 68063 | 69896 | 
| 235 | 31 src/flask/json/__init__.py | 183 | 216| 284 | 68347 | 69896 | 
| 236 | 31 src/flask/helpers.py | 785 | 824| 253 | 68600 | 69896 | 
| 237 | 31 src/flask/sessions.py | 107 | 148| 393 | 68993 | 69896 | 
| 238 | 31 src/flask/wrappers.py | 102 | 136| 280 | 69273 | 69896 | 
| 239 | 31 src/flask/sessions.py | 316 | 348| 276 | 69549 | 69896 | 
| 240 | 31 src/flask/sessions.py | 236 | 274| 403 | 69952 | 69896 | 
| 241 | 31 src/flask/helpers.py | 396 | 436| 406 | 70358 | 69896 | 


## Patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -188,6 +188,10 @@ def __init__(
             template_folder=template_folder,
             root_path=root_path,
         )
+
+        if "." in name:
+            raise ValueError("'name' may not contain a dot '.' character.")
+
         self.name = name
         self.url_prefix = url_prefix
         self.subdomain = subdomain
@@ -360,12 +364,12 @@ def add_url_rule(
         """Like :meth:`Flask.add_url_rule` but for a blueprint.  The endpoint for
         the :func:`url_for` function is prefixed with the name of the blueprint.
         """
-        if endpoint:
-            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
-        if view_func and hasattr(view_func, "__name__"):
-            assert (
-                "." not in view_func.__name__
-            ), "Blueprint view function name should not contain dots"
+        if endpoint and "." in endpoint:
+            raise ValueError("'endpoint' may not contain a dot '.' character.")
+
+        if view_func and hasattr(view_func, "__name__") and "." in view_func.__name__:
+            raise ValueError("'view_func' name may not contain a dot '.' character.")
+
         self.record(lambda s: s.add_url_rule(rule, endpoint, view_func, **options))
 
     def app_template_filter(self, name: t.Optional[str] = None) -> t.Callable:

```

## Test Patch

```diff
diff --git a/tests/test_basic.py b/tests/test_basic.py
--- a/tests/test_basic.py
+++ b/tests/test_basic.py
@@ -1631,7 +1631,7 @@ def something_else():
 
 
 def test_inject_blueprint_url_defaults(app):
-    bp = flask.Blueprint("foo.bar.baz", __name__, template_folder="template")
+    bp = flask.Blueprint("foo", __name__, template_folder="template")
 
     @bp.url_defaults
     def bp_defaults(endpoint, values):
@@ -1644,12 +1644,12 @@ def view(page):
     app.register_blueprint(bp)
 
     values = dict()
-    app.inject_url_defaults("foo.bar.baz.view", values)
+    app.inject_url_defaults("foo.view", values)
     expected = dict(page="login")
     assert values == expected
 
     with app.test_request_context("/somepage"):
-        url = flask.url_for("foo.bar.baz.view")
+        url = flask.url_for("foo.view")
     expected = "/login"
     assert url == expected
 
diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -1,5 +1,3 @@
-import functools
-
 import pytest
 from jinja2 import TemplateNotFound
 from werkzeug.http import parse_cache_control_header
@@ -253,28 +251,9 @@ def test_templates_list(test_apps):
     assert templates == ["admin/index.html", "frontend/index.html"]
 
 
-def test_dotted_names(app, client):
-    frontend = flask.Blueprint("myapp.frontend", __name__)
-    backend = flask.Blueprint("myapp.backend", __name__)
-
-    @frontend.route("/fe")
-    def frontend_index():
-        return flask.url_for("myapp.backend.backend_index")
-
-    @frontend.route("/fe2")
-    def frontend_page2():
-        return flask.url_for(".frontend_index")
-
-    @backend.route("/be")
-    def backend_index():
-        return flask.url_for("myapp.frontend.frontend_index")
-
-    app.register_blueprint(frontend)
-    app.register_blueprint(backend)
-
-    assert client.get("/fe").data.strip() == b"/be"
-    assert client.get("/fe2").data.strip() == b"/fe"
-    assert client.get("/be").data.strip() == b"/fe"
+def test_dotted_name_not_allowed(app, client):
+    with pytest.raises(ValueError):
+        flask.Blueprint("app.ui", __name__)
 
 
 def test_dotted_names_from_app(app, client):
@@ -343,62 +322,19 @@ def index():
 def test_route_decorator_custom_endpoint_with_dots(app, client):
     bp = flask.Blueprint("bp", __name__)
 
-    @bp.route("/foo")
-    def foo():
-        return flask.request.endpoint
-
-    try:
-
-        @bp.route("/bar", endpoint="bar.bar")
-        def foo_bar():
-            return flask.request.endpoint
-
-    except AssertionError:
-        pass
-    else:
-        raise AssertionError("expected AssertionError not raised")
-
-    try:
-
-        @bp.route("/bar/123", endpoint="bar.123")
-        def foo_bar_foo():
-            return flask.request.endpoint
-
-    except AssertionError:
-        pass
-    else:
-        raise AssertionError("expected AssertionError not raised")
-
-    def foo_foo_foo():
-        pass
-
-    pytest.raises(
-        AssertionError,
-        lambda: bp.add_url_rule("/bar/123", endpoint="bar.123", view_func=foo_foo_foo),
-    )
-
-    pytest.raises(
-        AssertionError, bp.route("/bar/123", endpoint="bar.123"), lambda: None
-    )
-
-    foo_foo_foo.__name__ = "bar.123"
+    with pytest.raises(ValueError):
+        bp.route("/", endpoint="a.b")(lambda: "")
 
-    pytest.raises(
-        AssertionError, lambda: bp.add_url_rule("/bar/123", view_func=foo_foo_foo)
-    )
+    with pytest.raises(ValueError):
+        bp.add_url_rule("/", endpoint="a.b")
 
-    bp.add_url_rule(
-        "/bar/456", endpoint="foofoofoo", view_func=functools.partial(foo_foo_foo)
-    )
+    def view():
+        return ""
 
-    app.register_blueprint(bp, url_prefix="/py")
+    view.__name__ = "a.b"
 
-    assert client.get("/py/foo").data == b"bp.foo"
-    # The rule's didn't actually made it through
-    rv = client.get("/py/bar")
-    assert rv.status_code == 404
-    rv = client.get("/py/bar/123")
-    assert rv.status_code == 404
+    with pytest.raises(ValueError):
+        bp.add_url_rule("/", view_func=view)
 
 
 def test_endpoint_decorator(app, client):

```


## Code snippets

### 1 - src/flask/blueprints.py:

Start line: 353, End line: 369

```python
class Blueprint(Scaffold):

    def add_url_rule(
        self,
        rule: str,
        endpoint: t.Optional[str] = None,
        view_func: t.Optional[t.Callable] = None,
        **options: t.Any,
    ) -> None:
        """Like :meth:`Flask.add_url_rule` but for a blueprint.  The endpoint for
        the :func:`url_for` function is prefixed with the name of the blueprint.
        """
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert (
                "." not in view_func.__name__
            ), "Blueprint view function name should not contain dots"
        self.record(lambda s: s.add_url_rule(rule, endpoint, view_func, **options))
```
### 2 - src/flask/blueprints.py:

Start line: 108, End line: 169

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

    warn_on_modifications = False
    _got_registered_once = False

    #: Blueprint local JSON encoder class to use. Set to ``None`` to use
    #: the app's :class:`~flask.Flask.json_encoder`.
    json_encoder = None
    #: Blueprint local JSON decoder class to use. Set to ``None`` to use
    #: the app's :class:`~flask.Flask.json_decoder`.
    json_decoder = None
```
### 3 - src/flask/blueprints.py:

Start line: 171, End line: 204

```python
class Blueprint(Scaffold):

    def __init__(
        self,
        name: str,
        import_name: str,
        static_folder: t.Optional[str] = None,
        static_url_path: t.Optional[str] = None,
        template_folder: t.Optional[str] = None,
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
        self.name = name
        self.url_prefix = url_prefix
        self.subdomain = subdomain
        self.deferred_functions: t.List[DeferredSetupFunction] = []

        if url_defaults is None:
            url_defaults = {}

        self.url_values_defaults = url_defaults
        self.cli_group = cli_group
        self._blueprints: t.List[t.Tuple["Blueprint", dict]] = []

    def _is_setup_finished(self) -> bool:
        return self.warn_on_modifications and self._got_registered_once
```
### 4 - src/flask/blueprints.py:

Start line: 77, End line: 105

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
            f"{self.name_prefix}{self.blueprint.name}.{endpoint}",
            view_func,
            defaults=defaults,
            **options,
        )
```
### 5 - src/flask/blueprints.py:

Start line: 435, End line: 449

```python
class Blueprint(Scaffold):

    def app_template_global(self, name: t.Optional[str] = None) -> t.Callable:
        """Register a custom template global, available application wide.  Like
        :meth:`Flask.template_global` but for a blueprint.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def decorator(f: TemplateGlobalCallable) -> TemplateGlobalCallable:
            self.add_app_template_global(f, name=name)
            return f

        return decorator
```
### 6 - src/flask/blueprints.py:

Start line: 371, End line: 383

```python
class Blueprint(Scaffold):

    def app_template_filter(self, name: t.Optional[str] = None) -> t.Callable:
        """Register a custom template filter, available application wide.  Like
        :meth:`Flask.template_filter` but for a blueprint.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def decorator(f: TemplateFilterCallable) -> TemplateFilterCallable:
            self.add_app_template_filter(f, name=name)
            return f

        return decorator
```
### 7 - src/flask/blueprints.py:

Start line: 401, End line: 415

```python
class Blueprint(Scaffold):

    def app_template_test(self, name: t.Optional[str] = None) -> t.Callable:
        """Register a custom template test, available application wide.  Like
        :meth:`Flask.template_test` but for a blueprint.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def decorator(f: TemplateTestCallable) -> TemplateTestCallable:
            self.add_app_template_test(f, name=name)
            return f

        return decorator
```
### 8 - src/flask/blueprints.py:

Start line: 1, End line: 22

```python
import typing as t
from collections import defaultdict
from functools import update_wrapper

from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .typing import AfterRequestCallable
from .typing import BeforeRequestCallable
from .typing import ErrorHandlerCallable
from .typing import TeardownCallable
from .typing import TemplateContextProcessorCallable
from .typing import TemplateFilterCallable
from .typing import TemplateGlobalCallable
from .typing import TemplateTestCallable
from .typing import URLDefaultCallable
from .typing import URLValuePreprocessorCallable

if t.TYPE_CHECKING:
    from .app import Flask

DeferredSetupFunction = t.Callable[["BlueprintSetupState"], t.Callable]
```
### 9 - src/flask/blueprints.py:

Start line: 451, End line: 467

```python
class Blueprint(Scaffold):

    def add_app_template_global(
        self, f: TemplateGlobalCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a custom template global, available application wide.  Like
        :meth:`Flask.add_template_global` but for a blueprint.  Works exactly
        like the :meth:`app_template_global` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.globals[name or f.__name__] = f

        self.record_once(register_template)
```
### 10 - src/flask/blueprints.py:

Start line: 255, End line: 351

```python
class Blueprint(Scaffold):

    def register(self, app: "Flask", options: dict) -> None:
        """Called by :meth:`Flask.register_blueprint` to register all
        views and callbacks registered on the blueprint with the
        application. Creates a :class:`.BlueprintSetupState` and calls
        each :meth:`record` callbackwith it.

        :param app: The application this blueprint is being registered
            with.
        :param options: Keyword arguments forwarded from
            :meth:`~Flask.register_blueprint`.
        :param first_registration: Whether this is the first time this
            blueprint has been registered on the application.
        """
        first_registration = False

        if self.name in app.blueprints:
            assert app.blueprints[self.name] is self, (
                "A name collision occurred between blueprints"
                f" {self!r} and {app.blueprints[self.name]!r}."
                f" Both share the same name {self.name!r}."
                f" Blueprints that are created on the fly need unique"
                f" names."
            )
        else:
            app.blueprints[self.name] = self
            first_registration = True

        self._got_registered_once = True
        state = self.make_setup_state(app, options, first_registration)

        if self.has_static_folder:
            state.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                view_func=self.send_static_file,
                endpoint="static",
            )

        # Merge blueprint data into parent.
        if first_registration:

            def extend(bp_dict, parent_dict):
                for key, values in bp_dict.items():
                    key = self.name if key is None else f"{self.name}.{key}"

                    parent_dict[key].extend(values)

            for key, value in self.error_handler_spec.items():
                key = self.name if key is None else f"{self.name}.{key}"
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

        if self.cli.commands:
            if cli_resolved_group is None:
                app.cli.commands.update(self.cli.commands)
            elif cli_resolved_group is _sentinel:
                self.cli.name = self.name
                app.cli.add_command(self.cli)
            else:
                self.cli.name = cli_resolved_group
                app.cli.add_command(self.cli)

        for blueprint, bp_options in self._blueprints:
            url_prefix = options.get("url_prefix", "")
            if "url_prefix" in bp_options:
                url_prefix = (
                    url_prefix.rstrip("/") + "/" + bp_options["url_prefix"].lstrip("/")
                )

            bp_options["url_prefix"] = url_prefix
            bp_options["name_prefix"] = options.get("name_prefix", "") + self.name + "."
            blueprint.register(app, bp_options)
```
### 11 - src/flask/blueprints.py:

Start line: 206, End line: 222

```python
class Blueprint(Scaffold):

    def record(self, func: t.Callable) -> None:
        """Registers a function that is called when the blueprint is
        registered on the application.  This function is called with the
        state as argument as returned by the :meth:`make_setup_state`
        method.
        """
        if self._got_registered_once and self.warn_on_modifications:
            from warnings import warn

            warn(
                Warning(
                    "The blueprint was already registered once but is"
                    " getting modified now. These changes will not show"
                    " up."
                )
            )
        self.deferred_functions.append(func)
```
### 12 - src/flask/blueprints.py:

Start line: 385, End line: 399

```python
class Blueprint(Scaffold):

    def add_app_template_filter(
        self, f: TemplateFilterCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a custom template filter, available application wide.  Like
        :meth:`Flask.add_template_filter` but for a blueprint.  Works exactly
        like the :meth:`app_template_filter` decorator.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.filters[name or f.__name__] = f

        self.record_once(register_template)
```
### 13 - src/flask/blueprints.py:

Start line: 417, End line: 433

```python
class Blueprint(Scaffold):

    def add_app_template_test(
        self, f: TemplateTestCallable, name: t.Optional[str] = None
    ) -> None:
        """Register a custom template test, available application wide.  Like
        :meth:`Flask.add_template_test` but for a blueprint.  Works exactly
        like the :meth:`app_template_test` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.tests[name or f.__name__] = f

        self.record_once(register_template)
```
### 15 - src/flask/blueprints.py:

Start line: 469, End line: 543

```python
class Blueprint(Scaffold):

    def before_app_request(self, f: BeforeRequestCallable) -> BeforeRequestCallable:
        """Like :meth:`Flask.before_request`.  Such a function is executed
        before each request, even if outside of a blueprint.
        """
        self.record_once(
            lambda s: s.app.before_request_funcs.setdefault(None, []).append(f)
        )
        return f

    def before_app_first_request(
        self, f: BeforeRequestCallable
    ) -> BeforeRequestCallable:
        """Like :meth:`Flask.before_first_request`.  Such a function is
        executed before the first request to the application.
        """
        self.record_once(lambda s: s.app.before_first_request_funcs.append(f))
        return f

    def after_app_request(self, f: AfterRequestCallable) -> AfterRequestCallable:
        """Like :meth:`Flask.after_request` but for a blueprint.  Such a function
        is executed after each request, even if outside of the blueprint.
        """
        self.record_once(
            lambda s: s.app.after_request_funcs.setdefault(None, []).append(f)
        )
        return f

    def teardown_app_request(self, f: TeardownCallable) -> TeardownCallable:
        """Like :meth:`Flask.teardown_request` but for a blueprint.  Such a
        function is executed when tearing down each request, even if outside of
        the blueprint.
        """
        self.record_once(
            lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f)
        )
        return f

    def app_context_processor(
        self, f: TemplateContextProcessorCallable
    ) -> TemplateContextProcessorCallable:
        """Like :meth:`Flask.context_processor` but for a blueprint.  Such a
        function is executed each request, even if outside of the blueprint.
        """
        self.record_once(
            lambda s: s.app.template_context_processors.setdefault(None, []).append(f)
        )
        return f

    def app_errorhandler(self, code: t.Union[t.Type[Exception], int]) -> t.Callable:
        """Like :meth:`Flask.errorhandler` but for a blueprint.  This
        handler is used for all requests, even if outside of the blueprint.
        """

        def decorator(f: ErrorHandlerCallable) -> ErrorHandlerCallable:
            self.record_once(lambda s: s.app.errorhandler(code)(f))
            return f

        return decorator

    def app_url_value_preprocessor(
        self, f: URLValuePreprocessorCallable
    ) -> URLValuePreprocessorCallable:
        """Same as :meth:`url_value_preprocessor` but application wide."""
        self.record_once(
            lambda s: s.app.url_value_preprocessors.setdefault(None, []).append(f)
        )
        return f

    def app_url_defaults(self, f: URLDefaultCallable) -> URLDefaultCallable:
        """Same as :meth:`url_defaults` but application wide."""
        self.record_once(
            lambda s: s.app.url_default_functions.setdefault(None, []).append(f)
        )
        return f
```
### 17 - src/flask/blueprints.py:

Start line: 224, End line: 253

```python
class Blueprint(Scaffold):

    def record_once(self, func: t.Callable) -> None:
        """Works like :meth:`record` but wraps the function in another
        function that will ensure the function is only called once.  If the
        blueprint is registered a second time on the application, the
        function passed is not called.
        """

        def wrapper(state: BlueprintSetupState) -> None:
            if state.first_registration:
                func(state)

        return self.record(update_wrapper(wrapper, func))

    def make_setup_state(
        self, app: "Flask", options: dict, first_registration: bool = False
    ) -> BlueprintSetupState:
        """Creates an instance of :meth:`~flask.blueprints.BlueprintSetupState`
        object that is later passed to the register callback functions.
        Subclasses can override this to return a subclass of the setup state.
        """
        return BlueprintSetupState(self, app, options, first_registration)

    def register_blueprint(self, blueprint: "Blueprint", **options: t.Any) -> None:
        """Register a :class:`~flask.Blueprint` on this blueprint. Keyword
        arguments passed to this method will override the defaults set
        on the blueprint.

        .. versionadded:: 2.0
        """
        self._blueprints.append((blueprint, options))
```
### 18 - src/flask/blueprints.py:

Start line: 25, End line: 75

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

        self.name_prefix = self.options.get("name_prefix", "")

        #: A dictionary with URL defaults that is added to each and every
        #: URL that was defined with the blueprint.
        self.url_defaults = dict(self.blueprint.url_values_defaults)
        self.url_defaults.update(self.options.get("url_defaults", ()))
```
