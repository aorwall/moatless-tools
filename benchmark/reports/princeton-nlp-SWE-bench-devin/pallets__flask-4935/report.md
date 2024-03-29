# pallets__flask-4935

| **pallets/flask** | `fa1ee7066807c21256e90089731c548b313394d2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6995 |
| **Any found context length** | 846 |
| **Avg pos** | 21.0 |
| **Min pos** | 2 |
| **Max pos** | 19 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -358,6 +358,9 @@ def register(self, app: "Flask", options: dict) -> None:
         :param options: Keyword arguments forwarded from
             :meth:`~Flask.register_blueprint`.
 
+        .. versionchanged:: 2.3
+            Nested blueprints now correctly apply subdomains.
+
         .. versionchanged:: 2.0.1
             Nested blueprints are registered with their dotted name.
             This allows different blueprints with the same name to be
@@ -453,6 +456,17 @@ def extend(bp_dict, parent_dict):
         for blueprint, bp_options in self._blueprints:
             bp_options = bp_options.copy()
             bp_url_prefix = bp_options.get("url_prefix")
+            bp_subdomain = bp_options.get("subdomain")
+
+            if bp_subdomain is None:
+                bp_subdomain = blueprint.subdomain
+
+            if state.subdomain is not None and bp_subdomain is not None:
+                bp_options["subdomain"] = bp_subdomain + "." + state.subdomain
+            elif bp_subdomain is not None:
+                bp_options["subdomain"] = bp_subdomain
+            elif state.subdomain is not None:
+                bp_options["subdomain"] = state.subdomain
 
             if bp_url_prefix is None:
                 bp_url_prefix = blueprint.url_prefix

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/flask/blueprints.py | 361 | 361 | 19 | 1 | 6995
| src/flask/blueprints.py | 456 | 456 | 2 | 1 | 846


## Problem Statement

```
Nested blueprints are not respected when mounted on subdomains
Hello, and thanks for all your work 🙏🏻 

Nested blueprints [as described in the docs](https://flask.palletsprojects.com/en/2.2.x/blueprints/#nesting-blueprints) work perfectly fine when using `url_prefix`. However, when mounting the parent blueprint using a subdomain, the child routes are not accessible.

\`\`\`python
from flask import Flask
from flask import Blueprint

app = Flask(__name__)
app.config["SERVER_NAME"] = "localhost:5000"
parent = Blueprint("parent", __name__)
child = Blueprint("child", __name__)

@app.route('/')
def index():
    return "index"

@parent.route('/')
def parent_index():
    return "parent"

@child.route('/child/')
def child_index():
    return "child"

parent.register_blueprint(child)
app.register_blueprint(parent, subdomain="api")


if __name__ == '__main__':
    app.run(debug=True)
\`\`\`

The index route works as expected:

\`\`\`
❯ http http://localhost:5000/
HTTP/1.1 200 OK
Connection: close
Content-Length: 5
Content-Type: text/html; charset=utf-8
Date: Tue, 04 Oct 2022 10:44:10 GMT
Server: Werkzeug/2.2.2 Python/3.10.4

index
\`\`\`

So does the parent route in the subdomain:

\`\`\`
❯ http http://api.localhost:5000/
HTTP/1.1 200 OK
Connection: close
Content-Length: 6
Content-Type: text/html; charset=utf-8
Date: Tue, 04 Oct 2022 10:44:06 GMT
Server: Werkzeug/2.2.2 Python/3.10.4

parent
\`\`\`

But the child responds with a 404:

\`\`\`
❯ http http://api.localhost:5000/child/
HTTP/1.1 404 NOT FOUND
Connection: close
Content-Length: 207
Content-Type: text/html; charset=utf-8
Date: Tue, 04 Oct 2022 10:45:42 GMT
Server: Werkzeug/2.2.2 Python/3.10.4

<!doctype html>
<html lang=en>
<title>404 Not Found</title>
<h1>Not Found</h1>
<p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p>
\`\`\`

If we change the `subdomain="api"` for `url_prefix="/api"` when registering the blueprint however everything works as expected:

\`\`\`
❯ http http://localhost:5000/api/
HTTP/1.1 200 OK
Connection: close
Content-Length: 6
Content-Type: text/html; charset=utf-8
Date: Tue, 04 Oct 2022 10:46:53 GMT
Server: Werkzeug/2.2.2 Python/3.10.4

parent

❯ http http://localhost:5000/api/child/
HTTP/1.1 200 OK
Connection: close
Content-Length: 5
Content-Type: text/html; charset=utf-8
Date: Tue, 04 Oct 2022 10:46:59 GMT
Server: Werkzeug/2.2.2 Python/3.10.4

child
\`\`\`

This was surprising to me as I expected the same nesting to apply regardless of whether the parent is mounted using a subdomain or a URL prefix. Am I missing something?

Environment:

- Python version: 3.10
- Flask version: 2.2.2


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 src/flask/blueprints.py** | 121 | 177| 595 | 595 | 5740 | 
| **-> 2 <-** | **1 src/flask/blueprints.py** | 443 | 470| 251 | 846 | 5740 | 
| 3 | 2 src/flask/app.py | 1273 | 1306| 324 | 1170 | 26987 | 
| 4 | **2 src/flask/blueprints.py** | 235 | 281| 379 | 1549 | 26987 | 
| 5 | 3 src/flask/wrappers.py | 94 | 133| 258 | 1807 | 28245 | 
| 6 | **3 src/flask/blueprints.py** | 690 | 707| 145 | 1952 | 28245 | 
| 7 | 4 src/flask/scaffold.py | 91 | 237| 1470 | 3422 | 35620 | 
| 8 | 4 src/flask/app.py | 1975 | 2039| 567 | 3989 | 35620 | 
| 9 | 5 src/flask/templating.py | 36 | 46| 108 | 4097 | 37295 | 
| 10 | **5 src/flask/blueprints.py** | 472 | 498| 218 | 4315 | 37295 | 
| 11 | **5 src/flask/blueprints.py** | 283 | 298| 146 | 4461 | 37295 | 
| 12 | **5 src/flask/blueprints.py** | 332 | 348| 166 | 4627 | 37295 | 
| 13 | **5 src/flask/blueprints.py** | 643 | 674| 276 | 4903 | 37295 | 
| 14 | 5 src/flask/scaffold.py | 55 | 89| 376 | 5279 | 37295 | 
| 15 | **5 src/flask/blueprints.py** | 201 | 233| 254 | 5533 | 37295 | 
| 16 | **5 src/flask/blueprints.py** | 676 | 688| 118 | 5651 | 37295 | 
| 17 | 6 src/flask/__init__.py | 1 | 45| 454 | 6105 | 37896 | 
| 18 | 6 src/flask/wrappers.py | 75 | 92| 124 | 6229 | 37896 | 
| **-> 19 <-** | **6 src/flask/blueprints.py** | 350 | 441| 766 | 6995 | 37896 | 
| 20 | **6 src/flask/blueprints.py** | 1 | 33| 331 | 7326 | 37896 | 
| 21 | **6 src/flask/blueprints.py** | 572 | 589| 138 | 7464 | 37896 | 
| 22 | 7 examples/tutorial/flaskr/blog.py | 1 | 25| 157 | 7621 | 38672 | 
| 23 | **7 src/flask/blueprints.py** | 89 | 118| 235 | 7856 | 38672 | 
| 24 | 8 src/flask/typing.py | 1 | 81| 736 | 8592 | 39408 | 
| 25 | **8 src/flask/blueprints.py** | 534 | 551| 138 | 8730 | 39408 | 
| 26 | 8 src/flask/app.py | 553 | 718| 1564 | 10294 | 39408 | 
| 27 | **8 src/flask/blueprints.py** | 36 | 87| 412 | 10706 | 39408 | 
| 28 | **8 src/flask/blueprints.py** | 620 | 641| 178 | 10884 | 39408 | 
| 29 | 8 src/flask/app.py | 1132 | 1196| 540 | 11424 | 39408 | 
| 30 | 8 src/flask/app.py | 1308 | 1365| 486 | 11910 | 39408 | 
| 31 | **8 src/flask/blueprints.py** | 500 | 515| 128 | 12038 | 39408 | 
| 32 | **8 src/flask/blueprints.py** | 591 | 618| 240 | 12278 | 39408 | 
| 33 | 9 src/flask/globals.py | 47 | 83| 371 | 12649 | 40193 | 
| 34 | **9 src/flask/blueprints.py** | 553 | 570| 158 | 12807 | 40193 | 
| 35 | **9 src/flask/blueprints.py** | 179 | 199| 160 | 12967 | 40193 | 
| 36 | 9 src/flask/templating.py | 1 | 18| 118 | 13085 | 40193 | 
| 37 | 9 src/flask/app.py | 1 | 107| 792 | 13877 | 40193 | 
| 38 | 9 src/flask/templating.py | 90 | 124| 256 | 14133 | 40193 | 
| 39 | **9 src/flask/blueprints.py** | 517 | 532| 148 | 14281 | 40193 | 
| 40 | 10 src/flask/debughelpers.py | 114 | 159| 407 | 14688 | 41426 | 
| 41 | 10 src/flask/templating.py | 49 | 62| 127 | 14815 | 41426 | 
| 42 | 10 src/flask/app.py | 1748 | 1774| 199 | 15014 | 41426 | 
| 43 | **10 src/flask/blueprints.py** | 300 | 330| 278 | 15292 | 41426 | 
| 44 | 10 src/flask/app.py | 207 | 275| 689 | 15981 | 41426 | 
| 45 | 10 src/flask/scaffold.py | 1 | 52| 410 | 16391 | 41426 | 
| 46 | 10 src/flask/app.py | 2292 | 2317| 217 | 16608 | 41426 | 
| 47 | 10 src/flask/scaffold.py | 239 | 291| 377 | 16985 | 41426 | 
| 48 | 10 src/flask/app.py | 2144 | 2193| 452 | 17437 | 41426 | 
| 49 | 10 src/flask/app.py | 947 | 958| 120 | 17557 | 41426 | 
| 50 | 10 src/flask/app.py | 871 | 905| 285 | 17842 | 41426 | 
| 51 | 10 src/flask/debughelpers.py | 1 | 38| 287 | 18129 | 41426 | 
| 52 | 11 src/flask/cli.py | 988 | 1055| 497 | 18626 | 49015 | 
| 53 | 11 src/flask/app.py | 110 | 205| 1013 | 19639 | 49015 | 
| 54 | 11 src/flask/app.py | 371 | 384| 142 | 19781 | 49015 | 
| 55 | 11 src/flask/templating.py | 64 | 88| 210 | 19991 | 49015 | 
| 56 | 11 src/flask/scaffold.py | 317 | 333| 167 | 20158 | 49015 | 
| 57 | 12 src/flask/testing.py | 192 | 250| 436 | 20594 | 51356 | 
| 58 | 13 docs/conf.py | 66 | 99| 213 | 20807 | 52170 | 
| 59 | 13 src/flask/app.py | 1776 | 1799| 262 | 21069 | 52170 | 
| 60 | 13 src/flask/app.py | 2237 | 2256| 169 | 21238 | 52170 | 
| 61 | 13 src/flask/app.py | 452 | 551| 912 | 22150 | 52170 | 
| 62 | 13 src/flask/app.py | 798 | 815| 128 | 22278 | 52170 | 
| 63 | 13 src/flask/app.py | 1801 | 1826| 208 | 22486 | 52170 | 
| 64 | 13 src/flask/testing.py | 23 | 91| 612 | 23098 | 52170 | 
| 65 | 13 src/flask/testing.py | 175 | 190| 122 | 23220 | 52170 | 
| 66 | 13 src/flask/app.py | 846 | 857| 125 | 23345 | 52170 | 
| 67 | 13 src/flask/app.py | 1067 | 1131| 702 | 24047 | 52170 | 
| 68 | 13 src/flask/scaffold.py | 454 | 520| 501 | 24548 | 52170 | 
| 69 | 14 src/flask/__main__.py | 1 | 4| 0 | 24548 | 52178 | 
| 70 | 14 src/flask/testing.py | 1 | 20| 119 | 24667 | 52178 | 
| 71 | 14 src/flask/cli.py | 1 | 29| 139 | 24806 | 52178 | 
| 72 | 15 examples/tutorial/flaskr/auth.py | 1 | 43| 240 | 25046 | 52858 | 
| 73 | 15 src/flask/app.py | 2195 | 2235| 338 | 25384 | 52858 | 
| 74 | 15 src/flask/app.py | 720 | 730| 110 | 25494 | 52858 | 
| 75 | 15 src/flask/app.py | 2347 | 2378| 278 | 25772 | 52858 | 
| 76 | 15 src/flask/app.py | 907 | 945| 304 | 26076 | 52858 | 
| 77 | 15 src/flask/app.py | 2258 | 2290| 292 | 26368 | 52858 | 
| 78 | 15 src/flask/scaffold.py | 371 | 420| 444 | 26812 | 52858 | 
| 79 | 15 src/flask/app.py | 1035 | 1065| 240 | 27052 | 52858 | 
| 80 | 15 src/flask/__init__.py | 48 | 72| 147 | 27199 | 52858 | 
| 81 | 15 src/flask/globals.py | 1 | 44| 281 | 27480 | 52858 | 
| 82 | 15 src/flask/app.py | 1576 | 1609| 277 | 27757 | 52858 | 
| 83 | 15 src/flask/app.py | 732 | 747| 163 | 27920 | 52858 | 
| 84 | 15 src/flask/debughelpers.py | 41 | 68| 238 | 28158 | 52858 | 
| 85 | 15 src/flask/scaffold.py | 335 | 346| 110 | 28268 | 52858 | 
| 86 | 16 src/flask/helpers.py | 1 | 25| 152 | 28420 | 58708 | 
| 87 | 16 src/flask/app.py | 817 | 832| 162 | 28582 | 58708 | 
| 88 | 17 src/flask/signals.py | 1 | 57| 483 | 29065 | 59191 | 
| 89 | 17 src/flask/scaffold.py | 860 | 899| 318 | 29383 | 59191 | 
| 90 | 17 src/flask/app.py | 1468 | 1480| 114 | 29497 | 59191 | 
| 91 | 17 src/flask/scaffold.py | 422 | 452| 237 | 29734 | 59191 | 
| 92 | 18 examples/tutorial/flaskr/__init__.py | 1 | 51| 315 | 30049 | 59507 | 
| 93 | 18 src/flask/cli.py | 564 | 589| 226 | 30275 | 59507 | 
| 94 | 18 src/flask/app.py | 1001 | 1033| 222 | 30497 | 59507 | 
| 95 | 18 src/flask/app.py | 1256 | 1271| 141 | 30638 | 59507 | 
| 96 | 18 src/flask/app.py | 1678 | 1729| 437 | 31075 | 59507 | 
| 97 | 18 src/flask/app.py | 1917 | 1973| 617 | 31692 | 59507 | 
| 98 | 18 src/flask/app.py | 1857 | 1892| 293 | 31985 | 59507 | 
| 99 | 18 src/flask/app.py | 2498 | 2552| 461 | 32446 | 59507 | 
| 100 | 18 src/flask/cli.py | 824 | 933| 664 | 33110 | 59507 | 
| 101 | 18 src/flask/globals.py | 86 | 108| 133 | 33243 | 59507 | 
| 102 | 18 src/flask/cli.py | 32 | 82| 370 | 33613 | 59507 | 
| 103 | 18 src/flask/app.py | 295 | 316| 212 | 33825 | 59507 | 
| 104 | 19 src/flask/ctx.py | 1 | 22| 124 | 33949 | 62762 | 
| 105 | 19 src/flask/app.py | 1542 | 1574| 253 | 34202 | 62762 | 
| 106 | 19 src/flask/scaffold.py | 750 | 789| 319 | 34521 | 62762 | 
| 107 | 20 examples/javascript/js_example/views.py | 1 | 19| 113 | 34634 | 62875 | 
| 108 | 20 src/flask/app.py | 749 | 769| 152 | 34786 | 62875 | 
| 109 | 20 src/flask/app.py | 960 | 971| 120 | 34906 | 62875 | 
| 110 | 20 src/flask/cli.py | 481 | 545| 599 | 35505 | 62875 | 
| 111 | 20 src/flask/cli.py | 362 | 390| 255 | 35760 | 62875 | 
| 112 | 21 setup.py | 1 | 18| 131 | 35891 | 63006 | 
| 113 | 21 src/flask/app.py | 1198 | 1254| 545 | 36436 | 63006 | 
| 114 | 21 examples/tutorial/flaskr/blog.py | 60 | 83| 149 | 36585 | 63006 | 
| 115 | 21 src/flask/cli.py | 547 | 562| 135 | 36720 | 63006 | 
| 116 | 21 src/flask/wrappers.py | 1 | 73| 594 | 37314 | 63006 | 
| 117 | 21 src/flask/app.py | 1444 | 1466| 164 | 37478 | 63006 | 
| 118 | 22 src/flask/config.py | 101 | 163| 456 | 37934 | 65659 | 
| 119 | 23 examples/tutorial/setup.py | 1 | 4| 0 | 37934 | 65666 | 
| 120 | 23 src/flask/app.py | 859 | 869| 130 | 38064 | 65666 | 
| 121 | 23 src/flask/app.py | 1646 | 1676| 242 | 38306 | 65666 | 
| 122 | 24 src/flask/views.py | 1 | 80| 676 | 38982 | 67179 | 
| 123 | 24 src/flask/app.py | 2442 | 2496| 497 | 39479 | 67179 | 
| 124 | 24 examples/tutorial/flaskr/blog.py | 28 | 57| 121 | 39600 | 67179 | 
| 125 | 24 src/flask/cli.py | 216 | 237| 180 | 39780 | 67179 | 
| 126 | 24 src/flask/testing.py | 94 | 118| 239 | 40019 | 67179 | 
| 127 | 24 src/flask/app.py | 1611 | 1644| 277 | 40296 | 67179 | 
| 128 | 24 src/flask/app.py | 277 | 293| 133 | 40429 | 67179 | 
| 129 | 25 src/flask/json/tag.py | 116 | 167| 332 | 40761 | 69296 | 
| 130 | 25 src/flask/scaffold.py | 613 | 649| 293 | 41054 | 69296 | 
| 131 | 25 src/flask/views.py | 82 | 132| 453 | 41507 | 69296 | 
| 132 | 25 src/flask/cli.py | 438 | 478| 254 | 41761 | 69296 | 
| 133 | 25 src/flask/cli.py | 612 | 651| 406 | 42167 | 69296 | 
| 134 | 25 examples/tutorial/flaskr/blog.py | 86 | 126| 236 | 42403 | 69296 | 
| 135 | 25 src/flask/app.py | 353 | 369| 162 | 42565 | 69296 | 
| 136 | 25 src/flask/app.py | 973 | 999| 236 | 42801 | 69296 | 
| 137 | 25 src/flask/scaffold.py | 348 | 369| 209 | 43010 | 69296 | 
| 138 | 25 src/flask/views.py | 135 | 189| 386 | 43396 | 69296 | 
| 139 | 25 src/flask/app.py | 341 | 351| 117 | 43513 | 69296 | 
| 140 | 25 src/flask/app.py | 1430 | 1442| 112 | 43625 | 69296 | 
| 141 | 26 examples/javascript/js_example/__init__.py | 1 | 6| 0 | 43625 | 69319 | 
| 142 | 27 examples/javascript/setup.py | 1 | 4| 0 | 43625 | 69326 | 
| 143 | 27 src/flask/cli.py | 591 | 610| 192 | 43817 | 69326 | 
| 144 | 27 src/flask/app.py | 771 | 796| 252 | 44069 | 69326 | 
| 145 | 27 src/flask/app.py | 1482 | 1506| 180 | 44249 | 69326 | 
| 146 | 27 src/flask/cli.py | 334 | 359| 203 | 44452 | 69326 | 
| 147 | 27 src/flask/cli.py | 936 | 985| 371 | 44823 | 69326 | 
| 148 | 27 src/flask/helpers.py | 678 | 706| 161 | 44984 | 69326 | 
| 149 | 27 src/flask/app.py | 1401 | 1428| 201 | 45185 | 69326 | 
| 150 | 27 src/flask/scaffold.py | 522 | 544| 143 | 45328 | 69326 | 
| 151 | 27 src/flask/helpers.py | 84 | 130| 344 | 45672 | 69326 | 
| 152 | 27 src/flask/app.py | 2319 | 2345| 248 | 45920 | 69326 | 
| 153 | 27 src/flask/app.py | 1367 | 1399| 247 | 46167 | 69326 | 
| 154 | 27 src/flask/scaffold.py | 792 | 857| 565 | 46732 | 69326 | 
| 155 | 27 src/flask/helpers.py | 212 | 263| 455 | 47187 | 69326 | 
| 156 | 27 src/flask/config.py | 29 | 75| 406 | 47593 | 69326 | 
| 157 | 28 examples/tutorial/flaskr/db.py | 1 | 53| 277 | 47870 | 69603 | 
| 158 | 29 src/flask/sessions.py | 183 | 239| 473 | 48343 | 73083 | 
| 159 | 29 src/flask/templating.py | 21 | 33| 119 | 48462 | 73083 | 
| 160 | 29 src/flask/ctx.py | 25 | 72| 329 | 48791 | 73083 | 
| 161 | 29 src/flask/app.py | 1731 | 1746| 138 | 48929 | 73083 | 
| 162 | 29 examples/tutorial/flaskr/auth.py | 46 | 81| 225 | 49154 | 73083 | 
| 163 | 29 src/flask/json/tag.py | 188 | 213| 158 | 49312 | 73083 | 
| 164 | 29 src/flask/app.py | 1828 | 1855| 218 | 49530 | 73083 | 
| 165 | 29 src/flask/cli.py | 292 | 331| 310 | 49840 | 73083 | 
| 166 | 29 src/flask/app.py | 413 | 450| 269 | 50109 | 73083 | 
| 167 | 29 src/flask/scaffold.py | 584 | 611| 236 | 50345 | 73083 | 
| 168 | 29 src/flask/app.py | 1894 | 1915| 164 | 50509 | 73083 | 
| 169 | 29 src/flask/helpers.py | 595 | 641| 416 | 50925 | 73083 | 
| 170 | 29 src/flask/app.py | 834 | 844| 108 | 51033 | 73083 | 
| 171 | 29 src/flask/ctx.py | 356 | 383| 243 | 51276 | 73083 | 
| 172 | 29 src/flask/ctx.py | 422 | 439| 117 | 51393 | 73083 | 
| 173 | 29 src/flask/helpers.py | 290 | 312| 212 | 51605 | 73083 | 
| 174 | 29 src/flask/config.py | 294 | 339| 346 | 51951 | 73083 | 
| 175 | 29 src/flask/app.py | 1508 | 1540| 281 | 52232 | 73083 | 
| 176 | 29 src/flask/ctx.py | 219 | 245| 230 | 52462 | 73083 | 
| 177 | 30 src/flask/logging.py | 1 | 23| 172 | 52634 | 73617 | 
| 178 | 30 src/flask/scaffold.py | 568 | 582| 148 | 52782 | 73617 | 
| 179 | 30 src/flask/scaffold.py | 546 | 566| 168 | 52950 | 73617 | 
| 180 | 30 src/flask/ctx.py | 326 | 354| 242 | 53192 | 73617 | 
| 181 | 30 src/flask/helpers.py | 132 | 160| 289 | 53481 | 73617 | 
| 182 | 30 src/flask/app.py | 2041 | 2053| 120 | 53601 | 73617 | 
| 183 | 30 examples/tutorial/flaskr/auth.py | 84 | 117| 215 | 53816 | 73617 | 
| 184 | 30 src/flask/templating.py | 127 | 147| 214 | 54030 | 73617 | 
| 185 | 30 src/flask/testing.py | 253 | 287| 329 | 54359 | 73617 | 
| 186 | 30 src/flask/app.py | 2055 | 2143| 725 | 55084 | 73617 | 
| 187 | 30 src/flask/app.py | 2426 | 2440| 151 | 55235 | 73617 | 
| 188 | 30 src/flask/app.py | 2380 | 2403| 196 | 55431 | 73617 | 
| 189 | 30 src/flask/cli.py | 715 | 772| 353 | 55784 | 73617 | 
| 190 | 30 src/flask/app.py | 318 | 339| 207 | 55991 | 73617 | 
| 191 | 30 src/flask/app.py | 386 | 411| 180 | 56171 | 73617 | 
| 192 | 30 src/flask/cli.py | 266 | 290| 246 | 56417 | 73617 | 
| 193 | 30 src/flask/cli.py | 111 | 184| 562 | 56979 | 73617 | 
| 194 | 30 src/flask/config.py | 275 | 292| 133 | 57112 | 73617 | 
| 195 | 31 src/flask/json/__init__.py | 58 | 81| 180 | 57292 | 76387 | 
| 196 | 31 src/flask/app.py | 2405 | 2424| 148 | 57440 | 76387 | 
| 197 | 31 src/flask/cli.py | 240 | 263| 129 | 57569 | 76387 | 
| 198 | 31 src/flask/cli.py | 654 | 712| 439 | 58008 | 76387 | 
| 199 | 31 src/flask/helpers.py | 163 | 209| 421 | 58429 | 76387 | 
| 200 | 31 src/flask/ctx.py | 247 | 275| 212 | 58641 | 76387 | 
| 201 | 31 src/flask/helpers.py | 552 | 592| 349 | 58990 | 76387 | 
| 202 | 31 src/flask/scaffold.py | 293 | 315| 191 | 59181 | 76387 | 
| 203 | 31 src/flask/ctx.py | 278 | 324| 444 | 59625 | 76387 | 
| 204 | 31 src/flask/cli.py | 393 | 418| 236 | 59861 | 76387 | 
| 205 | 31 src/flask/helpers.py | 424 | 549| 1261 | 61122 | 76387 | 
| 206 | 31 src/flask/debughelpers.py | 71 | 94| 165 | 61287 | 76387 | 
| 207 | 31 src/flask/json/tag.py | 1 | 54| 365 | 61652 | 76387 | 
| 208 | 31 src/flask/cli.py | 187 | 213| 177 | 61829 | 76387 | 
| 209 | 31 src/flask/cli.py | 85 | 108| 162 | 61991 | 76387 | 
| 210 | 31 src/flask/json/tag.py | 170 | 185| 141 | 62132 | 76387 | 
| 211 | 31 src/flask/ctx.py | 146 | 184| 271 | 62403 | 76387 | 
| 212 | 31 src/flask/helpers.py | 28 | 44| 138 | 62541 | 76387 | 
| 213 | 31 src/flask/debughelpers.py | 97 | 111| 133 | 62674 | 76387 | 
| 214 | 31 src/flask/helpers.py | 315 | 334| 184 | 62858 | 76387 | 
| 215 | 31 src/flask/ctx.py | 385 | 420| 290 | 63148 | 76387 | 
| 216 | 32 src/flask/json/provider.py | 1 | 88| 662 | 63810 | 78811 | 
| 217 | 32 src/flask/scaffold.py | 651 | 688| 284 | 64094 | 78811 | 
| 218 | 32 src/flask/helpers.py | 337 | 366| 293 | 64387 | 78811 | 
| 219 | 32 src/flask/ctx.py | 113 | 143| 226 | 64613 | 78811 | 
| 220 | 32 src/flask/templating.py | 181 | 197| 180 | 64793 | 78811 | 
| 221 | 32 src/flask/testing.py | 120 | 173| 498 | 65291 | 78811 | 
| 222 | 32 src/flask/helpers.py | 644 | 675| 260 | 65551 | 78811 | 
| 223 | 32 src/flask/scaffold.py | 690 | 703| 118 | 65669 | 78811 | 
| 224 | 32 src/flask/sessions.py | 158 | 168| 117 | 65786 | 78811 | 
| 225 | 32 src/flask/wrappers.py | 136 | 172| 288 | 66074 | 78811 | 
| 226 | 32 src/flask/helpers.py | 369 | 421| 499 | 66573 | 78811 | 
| 227 | 32 src/flask/sessions.py | 376 | 420| 318 | 66891 | 78811 | 
| 228 | 32 src/flask/json/__init__.py | 174 | 214| 328 | 67219 | 78811 | 
| 229 | 32 src/flask/logging.py | 53 | 75| 170 | 67389 | 78811 | 
| 230 | 32 src/flask/cli.py | 775 | 821| 355 | 67744 | 78811 | 
| 231 | 32 src/flask/sessions.py | 297 | 323| 222 | 67966 | 78811 | 
| 232 | 32 src/flask/helpers.py | 266 | 287| 207 | 68173 | 78811 | 
| 233 | 32 src/flask/json/provider.py | 125 | 167| 413 | 68586 | 78811 | 
| 234 | 32 src/flask/sessions.py | 108 | 156| 468 | 69054 | 78811 | 
| 235 | 32 src/flask/json/__init__.py | 256 | 288| 323 | 69377 | 78811 | 
| 236 | 32 src/flask/helpers.py | 69 | 81| 107 | 69484 | 78811 | 
| 237 | 32 src/flask/config.py | 77 | 99| 207 | 69691 | 78811 | 
| 238 | 32 src/flask/scaffold.py | 705 | 747| 325 | 70016 | 78811 | 
| 239 | 32 src/flask/sessions.py | 1 | 46| 315 | 70331 | 78811 | 
| 240 | 32 src/flask/helpers.py | 47 | 66| 148 | 70479 | 78811 | 
| 241 | 32 src/flask/cli.py | 421 | 435| 148 | 70627 | 78811 | 
| 242 | 32 src/flask/config.py | 1 | 26| 175 | 70802 | 78811 | 
| 243 | 32 src/flask/ctx.py | 88 | 110| 201 | 71003 | 78811 | 
| 244 | 32 src/flask/sessions.py | 360 | 374| 127 | 71130 | 78811 | 
| 245 | 32 src/flask/ctx.py | 187 | 216| 227 | 71357 | 78811 | 
| 246 | 32 src/flask/sessions.py | 49 | 88| 381 | 71738 | 78811 | 
| 247 | 32 src/flask/json/__init__.py | 291 | 307| 158 | 71896 | 78811 | 
| 248 | 32 src/flask/config.py | 194 | 230| 348 | 72244 | 78811 | 
| 249 | 32 src/flask/json/__init__.py | 84 | 127| 347 | 72591 | 78811 | 
| 250 | 32 src/flask/sessions.py | 170 | 181| 120 | 72711 | 78811 | 


### Hint

```
It looks like if you request `http://localhost:5000/child/`, you'll get 200 OK.

It means that when registering child blueprints, they don't respect the subdomain set by the parent.

I submitted a PR at #4855.
```

## Patch

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -358,6 +358,9 @@ def register(self, app: "Flask", options: dict) -> None:
         :param options: Keyword arguments forwarded from
             :meth:`~Flask.register_blueprint`.
 
+        .. versionchanged:: 2.3
+            Nested blueprints now correctly apply subdomains.
+
         .. versionchanged:: 2.0.1
             Nested blueprints are registered with their dotted name.
             This allows different blueprints with the same name to be
@@ -453,6 +456,17 @@ def extend(bp_dict, parent_dict):
         for blueprint, bp_options in self._blueprints:
             bp_options = bp_options.copy()
             bp_url_prefix = bp_options.get("url_prefix")
+            bp_subdomain = bp_options.get("subdomain")
+
+            if bp_subdomain is None:
+                bp_subdomain = blueprint.subdomain
+
+            if state.subdomain is not None and bp_subdomain is not None:
+                bp_options["subdomain"] = bp_subdomain + "." + state.subdomain
+            elif bp_subdomain is not None:
+                bp_options["subdomain"] = bp_subdomain
+            elif state.subdomain is not None:
+                bp_options["subdomain"] = state.subdomain
 
             if bp_url_prefix is None:
                 bp_url_prefix = blueprint.url_prefix

```

## Test Patch

```diff
diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -950,6 +950,55 @@ def index():
     assert response.status_code == 200
 
 
+def test_nesting_subdomains(app, client) -> None:
+    subdomain = "api"
+    parent = flask.Blueprint("parent", __name__)
+    child = flask.Blueprint("child", __name__)
+
+    @child.route("/child/")
+    def index():
+        return "child"
+
+    parent.register_blueprint(child)
+    app.register_blueprint(parent, subdomain=subdomain)
+
+    client.allow_subdomain_redirects = True
+
+    domain_name = "domain.tld"
+    app.config["SERVER_NAME"] = domain_name
+    response = client.get("/child/", base_url="http://api." + domain_name)
+
+    assert response.status_code == 200
+
+
+def test_child_and_parent_subdomain(app, client) -> None:
+    child_subdomain = "api"
+    parent_subdomain = "parent"
+    parent = flask.Blueprint("parent", __name__)
+    child = flask.Blueprint("child", __name__, subdomain=child_subdomain)
+
+    @child.route("/")
+    def index():
+        return "child"
+
+    parent.register_blueprint(child)
+    app.register_blueprint(parent, subdomain=parent_subdomain)
+
+    client.allow_subdomain_redirects = True
+
+    domain_name = "domain.tld"
+    app.config["SERVER_NAME"] = domain_name
+    response = client.get(
+        "/", base_url=f"http://{child_subdomain}.{parent_subdomain}.{domain_name}"
+    )
+
+    assert response.status_code == 200
+
+    response = client.get("/", base_url=f"http://{parent_subdomain}.{domain_name}")
+
+    assert response.status_code == 404
+
+
 def test_unique_blueprint_names(app, client) -> None:
     bp = flask.Blueprint("bp", __name__)
     bp2 = flask.Blueprint("bp", __name__)

```


## Code snippets

### 1 - src/flask/blueprints.py:

Start line: 121, End line: 177

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

    _json_encoder: t.Union[t.Type[json.JSONEncoder], None] = None
    _json_decoder: t.Union[t.Type[json.JSONDecoder], None] = None
```
### 2 - src/flask/blueprints.py:

Start line: 443, End line: 470

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
### 3 - src/flask/app.py:

Start line: 1273, End line: 1306

```python
class Flask(Scaffold):

    @setupmethod
    def register_blueprint(self, blueprint: "Blueprint", **options: t.Any) -> None:
        """Register a :class:`~flask.Blueprint` on the application. Keyword
        arguments passed to this method will override the defaults set on the
        blueprint.

        Calls the blueprint's :meth:`~flask.Blueprint.register` method after
        recording the blueprint in the application's :attr:`blueprints`.

        :param blueprint: The blueprint to register.
        :param url_prefix: Blueprint routes will be prefixed with this.
        :param subdomain: Blueprint routes will match on this subdomain.
        :param url_defaults: Blueprint routes will use these default values for
            view arguments.
        :param options: Additional keyword arguments are passed to
            :class:`~flask.blueprints.BlueprintSetupState`. They can be
            accessed in :meth:`~flask.Blueprint.record` callbacks.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.

        .. versionadded:: 0.7
        """
        blueprint.register(self, options)

    def iter_blueprints(self) -> t.ValuesView["Blueprint"]:
        """Iterates over all blueprints by the order they were registered.

        .. versionadded:: 0.11
        """
        return self.blueprints.values()
```
### 4 - src/flask/blueprints.py:

Start line: 235, End line: 281

```python
class Blueprint(Scaffold):

    @json_decoder.setter
    def json_decoder(self, value: t.Union[t.Type[json.JSONDecoder], None]) -> None:
        import warnings

        warnings.warn(
            "'bp.json_decoder' is deprecated and will be removed in Flask 2.3."
            " Customize 'app.json_provider_class' or 'app.json' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._json_decoder = value

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
### 5 - src/flask/wrappers.py:

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
### 6 - src/flask/blueprints.py:

Start line: 690, End line: 707

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_url_value_preprocessor(
        self, f: T_url_value_preprocessor
    ) -> T_url_value_preprocessor:
        """Same as :meth:`url_value_preprocessor` but application wide."""
        self.record_once(
            lambda s: s.app.url_value_preprocessors.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_url_defaults(self, f: T_url_defaults) -> T_url_defaults:
        """Same as :meth:`url_defaults` but application wide."""
        self.record_once(
            lambda s: s.app.url_default_functions.setdefault(None, []).append(f)
        )
        return f
```
### 7 - src/flask/scaffold.py:

Start line: 91, End line: 237

```python
class Scaffold:

    def __init__(
        self,
        import_name: str,
        static_folder: t.Optional[t.Union[str, os.PathLike]] = None,
        static_url_path: t.Optional[str] = None,
        template_folder: t.Optional[t.Union[str, os.PathLike]] = None,
        root_path: t.Optional[str] = None,
    ):
        #: The name of the package or module that this object belongs
        #: to. Do not change this once it is set by the constructor.
        self.import_name = import_name

        self.static_folder = static_folder  # type: ignore
        self.static_url_path = static_url_path

        #: The path to the templates folder, relative to
        #: :attr:`root_path`, to add to the template loader. ``None`` if
        #: templates should not be added.
        self.template_folder = template_folder

        if root_path is None:
            root_path = get_root_path(self.import_name)

        #: Absolute path to the package on the filesystem. Used to look
        #: up resources contained in the package.
        self.root_path = root_path

        #: The Click command group for registering CLI commands for this
        #: object. The commands are available from the ``flask`` command
        #: once the application has been discovered and blueprints have
        #: been registered.
        self.cli = AppGroup()

        #: A dictionary mapping endpoint names to view functions.
        #:
        #: To register a view function, use the :meth:`route` decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.view_functions: t.Dict[str, t.Callable] = {}

        #: A data structure of registered error handlers, in the format
        #: ``{scope: {code: {class: handler}}}``. The ``scope`` key is
        #: the name of a blueprint the handlers are active for, or
        #: ``None`` for all requests. The ``code`` key is the HTTP
        #: status code for ``HTTPException``, or ``None`` for
        #: other exceptions. The innermost dictionary maps exception
        #: classes to handler functions.
        #:
        #: To register an error handler, use the :meth:`errorhandler`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.error_handler_spec: t.Dict[
            ft.AppOrBlueprintKey,
            t.Dict[t.Optional[int], t.Dict[t.Type[Exception], ft.ErrorHandlerCallable]],
        ] = defaultdict(lambda: defaultdict(dict))

        #: A data structure of functions to call at the beginning of
        #: each request, in the format ``{scope: [functions]}``. The
        #: ``scope`` key is the name of a blueprint the functions are
        #: active for, or ``None`` for all requests.
        #:
        #: To register a function, use the :meth:`before_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.before_request_funcs: t.Dict[
            ft.AppOrBlueprintKey, t.List[ft.BeforeRequestCallable]
        ] = defaultdict(list)

        #: A data structure of functions to call at the end of each
        #: request, in the format ``{scope: [functions]}``. The
        #: ``scope`` key is the name of a blueprint the functions are
        #: active for, or ``None`` for all requests.
        #:
        #: To register a function, use the :meth:`after_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.after_request_funcs: t.Dict[
            ft.AppOrBlueprintKey, t.List[ft.AfterRequestCallable]
        ] = defaultdict(list)

        #: A data structure of functions to call at the end of each
        #: request even if an exception is raised, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`teardown_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.teardown_request_funcs: t.Dict[
            ft.AppOrBlueprintKey, t.List[ft.TeardownCallable]
        ] = defaultdict(list)

        #: A data structure of functions to call to pass extra context
        #: values when rendering templates, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`context_processor`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.template_context_processors: t.Dict[
            ft.AppOrBlueprintKey, t.List[ft.TemplateContextProcessorCallable]
        ] = defaultdict(list, {None: [_default_template_ctx_processor]})

        #: A data structure of functions to call to modify the keyword
        #: arguments passed to the view function, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the
        #: :meth:`url_value_preprocessor` decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.url_value_preprocessors: t.Dict[
            ft.AppOrBlueprintKey,
            t.List[ft.URLValuePreprocessorCallable],
        ] = defaultdict(list)

        #: A data structure of functions to call to modify the keyword
        #: arguments when generating URLs, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`url_defaults`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.url_default_functions: t.Dict[
            ft.AppOrBlueprintKey, t.List[ft.URLDefaultCallable]
        ] = defaultdict(list)
```
### 8 - src/flask/app.py:

Start line: 1975, End line: 2039

```python
class Flask(Scaffold):

    def url_for(
        self,
        endpoint: str,
        *,
        _anchor: t.Optional[str] = None,
        _method: t.Optional[str] = None,
        _scheme: t.Optional[str] = None,
        _external: t.Optional[bool] = None,
        **values: t.Any,
    ) -> str:
        # ... other code

        if req_ctx is not None:
            url_adapter = req_ctx.url_adapter
            blueprint_name = req_ctx.request.blueprint

            # If the endpoint starts with "." and the request matches a
            # blueprint, the endpoint is relative to the blueprint.
            if endpoint[:1] == ".":
                if blueprint_name is not None:
                    endpoint = f"{blueprint_name}{endpoint}"
                else:
                    endpoint = endpoint[1:]

            # When in a request, generate a URL without scheme and
            # domain by default, unless a scheme is given.
            if _external is None:
                _external = _scheme is not None
        else:
            app_ctx = _cv_app.get(None)

            # If called by helpers.url_for, an app context is active,
            # use its url_adapter. Otherwise, app.url_for was called
            # directly, build an adapter.
            if app_ctx is not None:
                url_adapter = app_ctx.url_adapter
            else:
                url_adapter = self.create_url_adapter(None)

            if url_adapter is None:
                raise RuntimeError(
                    "Unable to build URLs outside an active request"
                    " without 'SERVER_NAME' configured. Also configure"
                    " 'APPLICATION_ROOT' and 'PREFERRED_URL_SCHEME' as"
                    " needed."
                )

            # When outside a request, generate a URL with scheme and
            # domain by default.
            if _external is None:
                _external = True

        # It is an error to set _scheme when _external=False, in order
        # to avoid accidental insecure URLs.
        if _scheme is not None and not _external:
            raise ValueError("When specifying '_scheme', '_external' must be True.")

        self.inject_url_defaults(endpoint, values)

        try:
            rv = url_adapter.build(  # type: ignore[union-attr]
                endpoint,
                values,
                method=_method,
                url_scheme=_scheme,
                force_external=_external,
            )
        except BuildError as error:
            values.update(
                _anchor=_anchor, _method=_method, _scheme=_scheme, _external=_external
            )
            return self.handle_url_build_error(error, endpoint, values)

        if _anchor is not None:
            rv = f"{rv}#{url_quote(_anchor)}"

        return rv
```
### 9 - src/flask/templating.py:

Start line: 36, End line: 46

```python
class Environment(BaseEnvironment):
    """Works like a regular Jinja2 environment but has some additional
    knowledge of how Flask's blueprint works so that it can prepend the
    name of the blueprint to referenced templates if necessary.
    """

    def __init__(self, app: "Flask", **options: t.Any) -> None:
        if "loader" not in options:
            options["loader"] = app.create_global_jinja_loader()
        BaseEnvironment.__init__(self, **options)
        self.app = app
```
### 10 - src/flask/blueprints.py:

Start line: 472, End line: 498

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
        """Like :meth:`Flask.add_url_rule` but for a blueprint.  The endpoint for
        the :func:`url_for` function is prefixed with the name of the blueprint.
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
### 11 - src/flask/blueprints.py:

Start line: 283, End line: 298

```python
class Blueprint(Scaffold):

    def _check_setup_finished(self, f_name: str) -> None:
        if self._got_registered_once:
            import warnings

            warnings.warn(
                f"The setup method '{f_name}' can no longer be called on"
                f" the blueprint '{self.name}'. It has already been"
                " registered at least once, any changes will not be"
                " applied consistently.\n"
                "Make sure all imports, decorators, functions, etc."
                " needed to set up the blueprint are done before"
                " registering it.\n"
                "This warning will become an exception in Flask 2.3.",
                UserWarning,
                stacklevel=3,
            )
```
### 12 - src/flask/blueprints.py:

Start line: 332, End line: 348

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
### 13 - src/flask/blueprints.py:

Start line: 643, End line: 674

```python
class Blueprint(Scaffold):

    @setupmethod
    def after_app_request(self, f: T_after_request) -> T_after_request:
        """Like :meth:`Flask.after_request` but for a blueprint.  Such a function
        is executed after each request, even if outside of the blueprint.
        """
        self.record_once(
            lambda s: s.app.after_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def teardown_app_request(self, f: T_teardown) -> T_teardown:
        """Like :meth:`Flask.teardown_request` but for a blueprint.  Such a
        function is executed when tearing down each request, even if outside of
        the blueprint.
        """
        self.record_once(
            lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_context_processor(
        self, f: T_template_context_processor
    ) -> T_template_context_processor:
        """Like :meth:`Flask.context_processor` but for a blueprint.  Such a
        function is executed each request, even if outside of the blueprint.
        """
        self.record_once(
            lambda s: s.app.template_context_processors.setdefault(None, []).append(f)
        )
        return f
```
### 15 - src/flask/blueprints.py:

Start line: 201, End line: 233

```python
class Blueprint(Scaffold):

    @json_encoder.setter
    def json_encoder(self, value: t.Union[t.Type[json.JSONEncoder], None]) -> None:
        import warnings

        warnings.warn(
            "'bp.json_encoder' is deprecated and will be removed in Flask 2.3."
            " Customize 'app.json_provider_class' or 'app.json' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._json_encoder = value

    @property
    def json_decoder(
        self,
    ) -> t.Union[t.Type[json.JSONDecoder], None]:
        """Blueprint-local JSON decoder class to use. Set to ``None`` to use the app's.

        .. deprecated:: 2.2
             Will be removed in Flask 2.3. Customize
             :attr:`json_provider_class` instead.

        .. versionadded:: 0.10
        """
        import warnings

        warnings.warn(
            "'bp.json_decoder' is deprecated and will be removed in Flask 2.3."
            " Customize 'app.json_provider_class' or 'app.json' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._json_decoder
```
### 16 - src/flask/blueprints.py:

Start line: 676, End line: 688

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_errorhandler(
        self, code: t.Union[t.Type[Exception], int]
    ) -> t.Callable[[T_error_handler], T_error_handler]:
        """Like :meth:`Flask.errorhandler` but for a blueprint.  This
        handler is used for all requests, even if outside of the blueprint.
        """

        def decorator(f: T_error_handler) -> T_error_handler:
            self.record_once(lambda s: s.app.errorhandler(code)(f))
            return f

        return decorator
```
### 19 - src/flask/blueprints.py:

Start line: 350, End line: 441

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

        .. versionchanged:: 2.0.1
            Nested blueprints are registered with their dotted name.
            This allows different blueprints with the same name to be
            nested at different locations.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.

        .. versionchanged:: 2.0.1
            Registering the same blueprint with the same name multiple
            times is deprecated and will become an error in Flask 2.1.
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
### 20 - src/flask/blueprints.py:

Start line: 1, End line: 33

```python
import json
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
T_before_first_request = t.TypeVar(
    "T_before_first_request", bound=ft.BeforeFirstRequestCallable
)
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
### 21 - src/flask/blueprints.py:

Start line: 572, End line: 589

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_global(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_global], T_template_global]:
        """Register a custom template global, available application wide.  Like
        :meth:`Flask.template_global` but for a blueprint.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_global) -> T_template_global:
            self.add_app_template_global(f, name=name)
            return f

        return decorator
```
### 23 - src/flask/blueprints.py:

Start line: 89, End line: 118

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
### 25 - src/flask/blueprints.py:

Start line: 534, End line: 551

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_test(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_test], T_template_test]:
        """Register a custom template test, available application wide.  Like
        :meth:`Flask.template_test` but for a blueprint.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_test) -> T_template_test:
            self.add_app_template_test(f, name=name)
            return f

        return decorator
```
### 27 - src/flask/blueprints.py:

Start line: 36, End line: 87

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
### 28 - src/flask/blueprints.py:

Start line: 620, End line: 641

```python
class Blueprint(Scaffold):

    @setupmethod
    def before_app_first_request(
        self, f: T_before_first_request
    ) -> T_before_first_request:
        """Like :meth:`Flask.before_first_request`.  Such a function is
        executed before the first request to the application.

        .. deprecated:: 2.2
            Will be removed in Flask 2.3. Run setup code when creating
            the application instead.
        """
        import warnings

        warnings.warn(
            "'before_app_first_request' is deprecated and will be"
            " removed in Flask 2.3. Use 'record_once' instead to run"
            " setup code when registering the blueprint.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.record_once(lambda s: s.app.before_first_request_funcs.append(f))
        return f
```
### 31 - src/flask/blueprints.py:

Start line: 500, End line: 515

```python
class Blueprint(Scaffold):

    @setupmethod
    def app_template_filter(
        self, name: t.Optional[str] = None
    ) -> t.Callable[[T_template_filter], T_template_filter]:
        """Register a custom template filter, available application wide.  Like
        :meth:`Flask.template_filter` but for a blueprint.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_filter) -> T_template_filter:
            self.add_app_template_filter(f, name=name)
            return f

        return decorator
```
### 32 - src/flask/blueprints.py:

Start line: 591, End line: 618

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_global(
        self, f: ft.TemplateGlobalCallable, name: t.Optional[str] = None
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

    @setupmethod
    def before_app_request(self, f: T_before_request) -> T_before_request:
        """Like :meth:`Flask.before_request`.  Such a function is executed
        before each request, even if outside of a blueprint.
        """
        self.record_once(
            lambda s: s.app.before_request_funcs.setdefault(None, []).append(f)
        )
        return f
```
### 34 - src/flask/blueprints.py:

Start line: 553, End line: 570

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_test(
        self, f: ft.TemplateTestCallable, name: t.Optional[str] = None
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
### 35 - src/flask/blueprints.py:

Start line: 179, End line: 199

```python
class Blueprint(Scaffold):

    @property
    def json_encoder(
        self,
    ) -> t.Union[t.Type[json.JSONEncoder], None]:
        """Blueprint-local JSON encoder class to use. Set to ``None`` to use the app's.

        .. deprecated:: 2.2
             Will be removed in Flask 2.3. Customize
             :attr:`json_provider_class` instead.

        .. versionadded:: 0.10
        """
        import warnings

        warnings.warn(
            "'bp.json_encoder' is deprecated and will be removed in Flask 2.3."
            " Customize 'app.json_provider_class' or 'app.json' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._json_encoder
```
### 39 - src/flask/blueprints.py:

Start line: 517, End line: 532

```python
class Blueprint(Scaffold):

    @setupmethod
    def add_app_template_filter(
        self, f: ft.TemplateFilterCallable, name: t.Optional[str] = None
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
### 43 - src/flask/blueprints.py:

Start line: 300, End line: 330

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
