# django__django-16136

| **django/django** | `19e6efa50b603af325e7f62058364f278596758f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 176 |
| **Any found context length** | 176 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -148,7 +148,16 @@ def http_method_not_allowed(self, request, *args, **kwargs):
             request.path,
             extra={"status_code": 405, "request": request},
         )
-        return HttpResponseNotAllowed(self._allowed_methods())
+        response = HttpResponseNotAllowed(self._allowed_methods())
+
+        if self.view_is_async:
+
+            async def func():
+                return response
+
+            return func()
+        else:
+            return response
 
     def options(self, request, *args, **kwargs):
         """Handle responding to requests for the OPTIONS HTTP verb."""

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/generic/base.py | 151 | 151 | 1 | 1 | 176


## Problem Statement

```
object HttpResponseNotAllowed can't be used in 'await' expression
Description
	
When defining a simple View subclass with only an async "post" method, GET requests to this view cause the following exception:
[29/Sep/2022 07:50:48] "GET /demo HTTP/1.1" 500 81134
Method Not Allowed (GET): /demo
Internal Server Error: /demo
Traceback (most recent call last):
 File "/home/alorence/.cache/pypoetry/virtualenvs/dj-bug-demo-FlhD0jMY-py3.10/lib/python3.10/site-packages/django/core/handlers/exception.py", line 55, in inner
	response = get_response(request)
 File "/home/alorence/.cache/pypoetry/virtualenvs/dj-bug-demo-FlhD0jMY-py3.10/lib/python3.10/site-packages/django/core/handlers/base.py", line 197, in _get_response
	response = wrapped_callback(request, *callback_args, **callback_kwargs)
 File "/home/alorence/.cache/pypoetry/virtualenvs/dj-bug-demo-FlhD0jMY-py3.10/lib/python3.10/site-packages/asgiref/sync.py", line 218, in __call__
	return call_result.result()
 File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
	return self.__get_result()
 File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
	raise self._exception
 File "/home/alorence/.cache/pypoetry/virtualenvs/dj-bug-demo-FlhD0jMY-py3.10/lib/python3.10/site-packages/asgiref/sync.py", line 284, in main_wrap
	result = await self.awaitable(*args, **kwargs)
TypeError: object HttpResponseNotAllowed can't be used in 'await' expression
This can be easily reproduced with an empty project (no external dependencies) started with Django 4.1.1 and python 3.10.6.
Basic view to reproduce the bug:
from django.views import View
from django.http import HttpResponse
class Demo(View):
	"""This basic view supports only POST requests"""
	async def post(self, request):
		return HttpResponse("ok")
URL pattern to access it:
from django.urls import path
from views import Demo
urlpatterns = [
	path("demo", Demo.as_view()),
]
Start the local dev server (manage.py runserver) and open ​http://127.0.0.1:8000/demo in the browser.
Server crash with 500 error with the given traceback.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/views/generic/base.py** | 144 | 169| 176 | 176 | 1873 | 
| 2 | **1 django/views/generic/base.py** | 62 | 77| 133 | 309 | 1873 | 
| 3 | 2 django/http/response.py | 650 | 675| 159 | 468 | 7019 | 
| 4 | 3 django/views/csrf.py | 15 | 100| 839 | 1307 | 8572 | 
| 5 | 4 django/views/generic/__init__.py | 1 | 40| 204 | 1511 | 8777 | 
| 6 | 5 django/contrib/auth/password_validation.py | 217 | 267| 386 | 1897 | 10676 | 
| 7 | 6 django/__init__.py | 1 | 25| 173 | 2070 | 10849 | 
| 8 | 6 django/views/csrf.py | 101 | 161| 581 | 2651 | 10849 | 
| 9 | 6 django/http/response.py | 409 | 443| 210 | 2861 | 10849 | 
| 10 | 6 django/http/response.py | 318 | 358| 286 | 3147 | 10849 | 
| 11 | 7 django/contrib/admin/options.py | 1320 | 1408| 689 | 3836 | 30073 | 
| 12 | 8 django/views/decorators/http.py | 1 | 59| 358 | 4194 | 31053 | 
| 13 | 9 django/core/handlers/base.py | 136 | 172| 257 | 4451 | 33703 | 
| 14 | 10 django/views/defaults.py | 1 | 26| 151 | 4602 | 34686 | 
| 15 | 10 django/contrib/admin/options.py | 1410 | 1507| 710 | 5312 | 34686 | 
| 16 | 11 django/http/request.py | 336 | 365| 210 | 5522 | 40112 | 
| 17 | **11 django/views/generic/base.py** | 35 | 60| 147 | 5669 | 40112 | 
| 18 | 11 django/http/request.py | 405 | 436| 255 | 5924 | 40112 | 
| 19 | 12 django/views/debug.py | 556 | 614| 454 | 6378 | 44858 | 
| 20 | 12 django/views/defaults.py | 124 | 150| 190 | 6568 | 44858 | 
| 21 | 12 django/views/defaults.py | 102 | 121| 144 | 6712 | 44858 | 
| 22 | 13 django/contrib/auth/admin.py | 216 | 231| 185 | 6897 | 46629 | 
| 23 | 13 django/core/handlers/base.py | 317 | 341| 219 | 7116 | 46629 | 
| 24 | 13 django/contrib/admin/options.py | 1606 | 1652| 322 | 7438 | 46629 | 
| 25 | 13 django/views/debug.py | 59 | 72| 133 | 7571 | 46629 | 
| 26 | 13 django/http/request.py | 1 | 44| 276 | 7847 | 46629 | 
| 27 | 14 django/middleware/csrf.py | 420 | 475| 576 | 8423 | 50797 | 
| 28 | 15 django/views/generic/edit.py | 1 | 75| 494 | 8917 | 52801 | 
| 29 | **15 django/views/generic/base.py** | 79 | 122| 376 | 9293 | 52801 | 
| 30 | 15 django/views/generic/edit.py | 248 | 295| 341 | 9634 | 52801 | 
| 31 | 15 django/views/generic/edit.py | 163 | 214| 340 | 9974 | 52801 | 
| 32 | 15 django/views/generic/edit.py | 139 | 160| 182 | 10156 | 52801 | 
| 33 | 16 django/template/response.py | 1 | 116| 850 | 11006 | 53913 | 
| 34 | 17 django/contrib/auth/views.py | 1 | 34| 271 | 11277 | 56780 | 
| 35 | 17 django/views/generic/edit.py | 217 | 245| 192 | 11469 | 56780 | 
| 36 | 17 django/views/defaults.py | 82 | 99| 121 | 11590 | 56780 | 
| 37 | 18 django/contrib/sites/requests.py | 1 | 21| 131 | 11721 | 56911 | 
| 38 | 18 django/http/response.py | 614 | 647| 157 | 11878 | 56911 | 
| 39 | **18 django/views/generic/base.py** | 1 | 32| 156 | 12034 | 56911 | 
| 40 | 19 django/core/handlers/exception.py | 63 | 158| 600 | 12634 | 58029 | 
| 41 | 20 django/contrib/admindocs/views.py | 1 | 36| 252 | 12886 | 61519 | 
| 42 | 20 django/middleware/csrf.py | 206 | 225| 151 | 13037 | 61519 | 
| 43 | 21 django/core/servers/basehttp.py | 151 | 187| 279 | 13316 | 63428 | 
| 44 | 22 django/utils/deprecation.py | 87 | 140| 382 | 13698 | 64499 | 
| 45 | 23 django/contrib/syndication/views.py | 1 | 26| 223 | 13921 | 66357 | 
| 46 | 23 django/contrib/admindocs/views.py | 39 | 62| 163 | 14084 | 66357 | 
| 47 | 23 django/http/response.py | 151 | 164| 150 | 14234 | 66357 | 
| 48 | 24 django/views/static.py | 1 | 14| 109 | 14343 | 67343 | 
| 49 | 25 django/contrib/flatpages/views.py | 1 | 45| 399 | 14742 | 67933 | 
| 50 | 25 django/middleware/csrf.py | 303 | 353| 450 | 15192 | 67933 | 
| 51 | 26 django/contrib/auth/decorators.py | 1 | 40| 315 | 15507 | 68525 | 
| 52 | 26 django/views/debug.py | 392 | 421| 247 | 15754 | 68525 | 
| 53 | 26 django/contrib/syndication/views.py | 29 | 49| 193 | 15947 | 68525 | 
| 54 | 27 django/views/generic/list.py | 150 | 175| 205 | 16152 | 70127 | 
| 55 | 28 django/contrib/gis/views.py | 1 | 23| 160 | 16312 | 70287 | 
| 56 | 28 django/views/csrf.py | 1 | 13| 132 | 16444 | 70287 | 
| 57 | 28 django/middleware/csrf.py | 355 | 418| 585 | 17029 | 70287 | 
| 58 | 29 django/core/exceptions.py | 124 | 240| 756 | 17785 | 71480 | 
| 59 | 29 django/http/response.py | 588 | 611| 195 | 17980 | 71480 | 
| 60 | 30 django/core/checks/security/csrf.py | 45 | 68| 159 | 18139 | 71945 | 
| 61 | 31 django/views/decorators/clickjacking.py | 1 | 22| 140 | 18279 | 72330 | 
| 62 | 32 django/views/generic/dates.py | 389 | 410| 140 | 18419 | 77840 | 
| 63 | 33 django/contrib/admin/sites.py | 443 | 457| 129 | 18548 | 82292 | 
| 64 | 33 django/contrib/admin/sites.py | 229 | 250| 221 | 18769 | 82292 | 
| 65 | 33 django/http/request.py | 47 | 123| 528 | 19297 | 82292 | 
| 66 | 33 django/contrib/auth/views.py | 267 | 309| 382 | 19679 | 82292 | 
| 67 | 33 django/http/response.py | 300 | 316| 181 | 19860 | 82292 | 
| 68 | 33 django/views/debug.py | 211 | 223| 143 | 20003 | 82292 | 
| 69 | 33 django/contrib/auth/views.py | 363 | 395| 239 | 20242 | 82292 | 
| 70 | 33 django/views/decorators/clickjacking.py | 25 | 63| 243 | 20485 | 82292 | 
| 71 | 34 django/utils/log.py | 210 | 251| 262 | 20747 | 83966 | 
| 72 | 35 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 21172 | 84807 | 
| 73 | 35 django/views/generic/dates.py | 235 | 280| 338 | 21510 | 84807 | 
| 74 | 35 django/contrib/admindocs/views.py | 141 | 161| 172 | 21682 | 84807 | 
| 75 | 35 django/middleware/csrf.py | 166 | 189| 159 | 21841 | 84807 | 
| 76 | 36 django/contrib/staticfiles/views.py | 1 | 40| 270 | 22111 | 85077 | 
| 77 | 37 django/views/__init__.py | 1 | 4| 0 | 22111 | 85092 | 
| 78 | 37 django/http/response.py | 99 | 149| 385 | 22496 | 85092 | 
| 79 | 38 django/core/handlers/asgi.py | 1 | 24| 116 | 22612 | 87476 | 
| 80 | 38 django/views/generic/dates.py | 626 | 667| 308 | 22920 | 87476 | 
| 81 | 39 django/core/handlers/wsgi.py | 66 | 130| 575 | 23495 | 89322 | 
| 82 | 39 django/views/generic/dates.py | 670 | 697| 221 | 23716 | 89322 | 
| 83 | 39 django/contrib/auth/views.py | 244 | 264| 163 | 23879 | 89322 | 
| 84 | 39 django/views/debug.py | 498 | 553| 446 | 24325 | 89322 | 
| 85 | 39 django/views/defaults.py | 29 | 79| 377 | 24702 | 89322 | 
| 86 | 39 django/contrib/admindocs/views.py | 164 | 182| 187 | 24889 | 89322 | 
| 87 | 39 django/views/generic/dates.py | 302 | 326| 204 | 25093 | 89322 | 
| 88 | 39 django/core/handlers/base.py | 343 | 374| 214 | 25307 | 89322 | 
| 89 | 39 django/views/generic/dates.py | 1 | 68| 422 | 25729 | 89322 | 
| 90 | 40 django/urls/conf.py | 61 | 96| 266 | 25995 | 90039 | 
| 91 | 40 django/http/request.py | 179 | 188| 113 | 26108 | 90039 | 
| 92 | 40 django/contrib/auth/views.py | 311 | 360| 325 | 26433 | 90039 | 
| 93 | 41 django/views/decorators/cache.py | 29 | 46| 129 | 26562 | 90514 | 
| 94 | 41 django/core/handlers/base.py | 228 | 298| 494 | 27056 | 90514 | 
| 95 | 41 django/contrib/admindocs/views.py | 184 | 210| 238 | 27294 | 90514 | 
| 96 | 41 django/views/generic/list.py | 1 | 47| 333 | 27627 | 90514 | 
| 97 | 41 django/http/response.py | 1 | 26| 165 | 27792 | 90514 | 
| 98 | 42 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 27939 | 90662 | 
| 99 | 43 django/contrib/auth/urls.py | 1 | 37| 253 | 28192 | 90915 | 
| 100 | 43 django/views/debug.py | 280 | 312| 236 | 28428 | 90915 | 
| 101 | 43 django/core/handlers/asgi.py | 212 | 239| 237 | 28665 | 90915 | 
| 102 | 43 django/core/servers/basehttp.py | 189 | 208| 170 | 28835 | 90915 | 
| 103 | 43 django/utils/deprecation.py | 142 | 160| 122 | 28957 | 90915 | 
| 104 | 43 django/contrib/flatpages/views.py | 48 | 71| 191 | 29148 | 90915 | 
| 105 | 44 django/contrib/messages/views.py | 1 | 20| 0 | 29148 | 91011 | 
| 106 | 45 django/contrib/sitemaps/views.py | 1 | 38| 249 | 29397 | 92170 | 
| 107 | 45 django/views/generic/dates.py | 413 | 456| 298 | 29695 | 92170 | 
| 108 | 45 django/http/response.py | 361 | 407| 309 | 30004 | 92170 | 
| 109 | 46 django/contrib/admindocs/urls.py | 1 | 51| 307 | 30311 | 92477 | 
| 110 | 46 django/contrib/admin/sites.py | 205 | 227| 167 | 30478 | 92477 | 
| 111 | 46 django/views/debug.py | 459 | 496| 301 | 30779 | 92477 | 
| 112 | 46 django/views/generic/dates.py | 328 | 355| 234 | 31013 | 92477 | 
| 113 | 46 django/views/decorators/cache.py | 49 | 67| 133 | 31146 | 92477 | 
| 114 | 47 django/http/__init__.py | 1 | 51| 233 | 31379 | 92710 | 
| 115 | **47 django/views/generic/base.py** | 172 | 207| 243 | 31622 | 92710 | 
| 116 | 48 django/contrib/admin/actions.py | 1 | 97| 647 | 32269 | 93357 | 
| 117 | 49 django/views/decorators/csrf.py | 1 | 59| 462 | 32731 | 93820 | 
| 118 | 49 django/middleware/csrf.py | 277 | 301| 176 | 32907 | 93820 | 
| 119 | 49 django/contrib/auth/views.py | 228 | 242| 133 | 33040 | 93820 | 
| 120 | 49 django/contrib/admindocs/views.py | 395 | 428| 211 | 33251 | 93820 | 
| 121 | 50 django/contrib/admindocs/middleware.py | 1 | 34| 257 | 33508 | 94078 | 
| 122 | 50 django/template/response.py | 144 | 164| 106 | 33614 | 94078 | 
| 123 | 50 django/contrib/admin/options.py | 1748 | 1850| 782 | 34396 | 94078 | 
| 124 | 50 django/http/request.py | 282 | 334| 351 | 34747 | 94078 | 
| 125 | 50 django/views/generic/dates.py | 124 | 168| 285 | 35032 | 94078 | 
| 126 | 51 django/core/checks/security/base.py | 1 | 79| 691 | 35723 | 96267 | 
| 127 | 51 django/core/checks/security/csrf.py | 1 | 42| 305 | 36028 | 96267 | 
| 128 | **51 django/views/generic/base.py** | 210 | 244| 228 | 36256 | 96267 | 
| 129 | 51 django/views/generic/dates.py | 459 | 504| 304 | 36560 | 96267 | 
| 130 | 51 django/middleware/csrf.py | 477 | 490| 163 | 36723 | 96267 | 
| 131 | 51 django/views/generic/dates.py | 171 | 218| 320 | 37043 | 96267 | 
| 132 | 51 django/views/generic/dates.py | 71 | 121| 342 | 37385 | 96267 | 
| 133 | 51 django/views/generic/dates.py | 606 | 623| 123 | 37508 | 96267 | 
| 134 | 51 django/views/debug.py | 75 | 102| 178 | 37686 | 96267 | 
| 135 | **51 django/views/generic/base.py** | 124 | 142| 180 | 37866 | 96267 | 
| 136 | 52 django/urls/exceptions.py | 1 | 10| 0 | 37866 | 96292 | 
| 137 | 52 django/contrib/auth/views.py | 126 | 183| 458 | 38324 | 96292 | 
| 138 | 52 django/views/generic/edit.py | 78 | 110| 269 | 38593 | 96292 | 
| 139 | 52 django/contrib/admindocs/views.py | 213 | 297| 615 | 39208 | 96292 | 
| 140 | 52 django/utils/deprecation.py | 1 | 39| 228 | 39436 | 96292 | 
| 141 | 53 django/core/checks/async_checks.py | 1 | 17| 0 | 39436 | 96386 | 
| 142 | 53 django/views/debug.py | 1 | 56| 351 | 39787 | 96386 | 
| 143 | 53 django/core/handlers/exception.py | 161 | 186| 167 | 39954 | 96386 | 
| 144 | 53 django/views/generic/dates.py | 507 | 556| 370 | 40324 | 96386 | 
| 145 | 54 django/middleware/common.py | 153 | 179| 255 | 40579 | 97932 | 
| 146 | 55 django/views/generic/detail.py | 1 | 59| 433 | 41012 | 99262 | 
| 147 | 55 django/core/servers/basehttp.py | 230 | 247| 210 | 41222 | 99262 | 
| 148 | 55 django/contrib/syndication/views.py | 180 | 235| 480 | 41702 | 99262 | 
| 149 | 55 django/contrib/admin/sites.py | 383 | 403| 152 | 41854 | 99262 | 
| 150 | 55 django/middleware/csrf.py | 259 | 275| 186 | 42040 | 99262 | 
| 151 | 55 django/contrib/admin/options.py | 577 | 593| 198 | 42238 | 99262 | 
| 152 | 55 django/contrib/sitemaps/views.py | 103 | 154| 369 | 42607 | 99262 | 
| 153 | 55 django/http/request.py | 367 | 403| 294 | 42901 | 99262 | 
| 154 | 55 django/http/response.py | 79 | 96| 133 | 43034 | 99262 | 
| 155 | 55 django/contrib/admin/sites.py | 1 | 34| 227 | 43261 | 99262 | 
| 156 | 55 django/contrib/admin/views/autocomplete.py | 1 | 42| 241 | 43502 | 99262 | 
| 157 | 55 django/core/handlers/base.py | 174 | 226| 385 | 43887 | 99262 | 
| 158 | 56 django/core/serializers/xml_serializer.py | 444 | 462| 126 | 44013 | 102896 | 
| 159 | 56 django/contrib/admindocs/views.py | 298 | 392| 640 | 44653 | 102896 | 
| 160 | 57 django/core/management/commands/runserver.py | 122 | 184| 522 | 45175 | 104412 | 
| 161 | 57 django/contrib/admin/options.py | 1913 | 2001| 676 | 45851 | 104412 | 
| 162 | 57 django/views/debug.py | 184 | 209| 181 | 46032 | 104412 | 
| 163 | 58 django/contrib/admindocs/utils.py | 1 | 28| 185 | 46217 | 106367 | 
| 164 | 58 django/contrib/auth/views.py | 67 | 90| 185 | 46402 | 106367 | 
| 165 | 58 django/http/response.py | 446 | 495| 371 | 46773 | 106367 | 
| 166 | 58 django/core/handlers/asgi.py | 27 | 133| 888 | 47661 | 106367 | 
| 167 | 58 django/contrib/sitemaps/views.py | 53 | 100| 423 | 48084 | 106367 | 
| 168 | 58 django/http/request.py | 255 | 280| 191 | 48275 | 106367 | 
| 169 | 58 django/http/request.py | 439 | 462| 187 | 48462 | 106367 | 
| 170 | 59 django/urls/resolvers.py | 728 | 739| 120 | 48582 | 112451 | 
| 171 | 59 django/core/servers/basehttp.py | 131 | 148| 179 | 48761 | 112451 | 
| 172 | 59 django/urls/resolvers.py | 320 | 338| 163 | 48924 | 112451 | 
| 173 | 59 django/core/servers/basehttp.py | 113 | 129| 152 | 49076 | 112451 | 
| 174 | 60 docs/_ext/djangodocs.py | 26 | 71| 398 | 49474 | 115675 | 
| 175 | 61 django/shortcuts.py | 64 | 89| 224 | 49698 | 116799 | 
| 176 | 62 django/core/management/commands/testserver.py | 38 | 66| 237 | 49935 | 117245 | 
| 177 | 63 django/views/decorators/common.py | 1 | 17| 112 | 50047 | 117358 | 
| 178 | 63 django/core/handlers/asgi.py | 161 | 189| 257 | 50304 | 117358 | 
| 179 | 63 django/contrib/admin/options.py | 2170 | 2228| 444 | 50748 | 117358 | 
| 180 | 63 django/urls/resolvers.py | 150 | 182| 235 | 50983 | 117358 | 
| 181 | 63 django/contrib/admin/options.py | 1 | 114| 776 | 51759 | 117358 | 
| 182 | 63 django/core/management/commands/runserver.py | 1 | 22| 204 | 51963 | 117358 | 
| 183 | 64 django/db/models/query.py | 546 | 563| 153 | 52116 | 137808 | 
| 184 | 64 django/contrib/admin/sites.py | 568 | 590| 163 | 52279 | 137808 | 
| 185 | 64 django/core/servers/basehttp.py | 55 | 82| 189 | 52468 | 137808 | 
| 186 | 65 django/utils/asyncio.py | 1 | 40| 221 | 52689 | 138030 | 
| 187 | 65 django/contrib/admindocs/views.py | 102 | 138| 301 | 52990 | 138030 | 
| 188 | 65 django/core/servers/basehttp.py | 1 | 24| 170 | 53160 | 138030 | 
| 189 | 65 django/contrib/admindocs/views.py | 65 | 99| 297 | 53457 | 138030 | 
| 190 | 65 django/core/management/commands/runserver.py | 80 | 120| 401 | 53858 | 138030 | 
| 191 | 66 django/forms/utils.py | 58 | 78| 150 | 54008 | 139749 | 
| 192 | 66 django/shortcuts.py | 1 | 25| 161 | 54169 | 139749 | 
| 193 | 66 django/http/response.py | 498 | 523| 203 | 54372 | 139749 | 
| 194 | 66 django/views/generic/dates.py | 559 | 603| 290 | 54662 | 139749 | 
| 195 | 66 django/core/management/commands/runserver.py | 25 | 66| 284 | 54946 | 139749 | 
| 196 | 66 docs/_ext/djangodocs.py | 74 | 108| 257 | 55203 | 139749 | 
| 197 | 66 django/middleware/csrf.py | 1 | 56| 485 | 55688 | 139749 | 
| 198 | 67 django/utils/http.py | 1 | 51| 532 | 56220 | 143511 | 
| 199 | 67 django/urls/resolvers.py | 499 | 528| 292 | 56512 | 143511 | 
| 200 | 67 django/contrib/auth/admin.py | 1 | 25| 195 | 56707 | 143511 | 
| 201 | 67 django/http/response.py | 166 | 208| 250 | 56957 | 143511 | 
| 202 | 68 django/template/context.py | 1 | 24| 128 | 57085 | 145403 | 
| 203 | 68 django/views/generic/list.py | 122 | 147| 208 | 57293 | 145403 | 


### Hint

```
Yes, looks right. http_method_not_allowed() needs to be adjusted to handle both sync and async cases in the same way as options() Do you have capacity to do a patch quickly? (Otherwise I'll take it on.) Thanks for the report! Regression in 9ffd4eae2ce7a7100c98f681e2b6ab818df384a4.
Thank you very much for your confirmation. I've never contributed to Django codebase, but the fix for this issue seems obvious. I think this is a good occasion to contribute for the first tme. I will follow contributions guide and try to provide a regression test and a patch for that issue.
Great, welcome aboard Antoine! I pushed a draft here ​https://github.com/django/django/compare/main...carltongibson:django:4.1.2/ticket-34062 that you can use for inspiration. If you open a PR on GitHub, I'm happy to advise. Normally there's no rush here, but we have releases due for the beginning of next week, and as a regression this needs to go in. As such, if you hit any barriers please reach out so I can help out. Thanks 🏅
Wow, your draft is almost exactly what I already wrote in my local fork. I ended with the exact same modification in View.http_method_not_allowed() (to the letter). I also written the almost same test in "async/tests" (I didn't use RequestFactory, then your version is better). I was currently looking into "asgi/tests" to check if I can add a full request-response lifecycle test in such case, but this appear to be more challenging. Do you think this is feasible / required ?
It's feasible, but I don't think it's required. (We already test the full dispatch in many places elsewhere. What we're looking for here is the http_method_not_allowed() is correctly adapted when the view is async (i.e. has async handlers). Make sense?
That totally makes sense. What is the next step now ? Your draft is perfect, more complete than mine (I didn't write into release notes). I think you can merge your branch django:4.1.2/ticket-34062.
```

## Patch

```diff
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -148,7 +148,16 @@ def http_method_not_allowed(self, request, *args, **kwargs):
             request.path,
             extra={"status_code": 405, "request": request},
         )
-        return HttpResponseNotAllowed(self._allowed_methods())
+        response = HttpResponseNotAllowed(self._allowed_methods())
+
+        if self.view_is_async:
+
+            async def func():
+                return response
+
+            return func()
+        else:
+            return response
 
     def options(self, request, *args, **kwargs):
         """Handle responding to requests for the OPTIONS HTTP verb."""

```

## Test Patch

```diff
diff --git a/tests/async/tests.py b/tests/async/tests.py
--- a/tests/async/tests.py
+++ b/tests/async/tests.py
@@ -6,8 +6,8 @@
 
 from django.core.cache import DEFAULT_CACHE_ALIAS, caches
 from django.core.exceptions import ImproperlyConfigured, SynchronousOnlyOperation
-from django.http import HttpResponse
-from django.test import SimpleTestCase
+from django.http import HttpResponse, HttpResponseNotAllowed
+from django.test import RequestFactory, SimpleTestCase
 from django.utils.asyncio import async_unsafe
 from django.views.generic.base import View
 
@@ -119,6 +119,25 @@ def test_options_handler_responds_correctly(self):
 
                 self.assertIsInstance(response, HttpResponse)
 
+    def test_http_method_not_allowed_responds_correctly(self):
+        request_factory = RequestFactory()
+        tests = [
+            (SyncView, False),
+            (AsyncView, True),
+        ]
+        for view_cls, is_coroutine in tests:
+            with self.subTest(view_cls=view_cls, is_coroutine=is_coroutine):
+                instance = view_cls()
+                response = instance.http_method_not_allowed(request_factory.post("/"))
+                self.assertIs(
+                    asyncio.iscoroutine(response),
+                    is_coroutine,
+                )
+                if is_coroutine:
+                    response = asyncio.run(response)
+
+                self.assertIsInstance(response, HttpResponseNotAllowed)
+
     def test_base_view_class_is_sync(self):
         """
         View and by extension any subclasses that don't define handlers are

```


## Code snippets

### 1 - django/views/generic/base.py:

Start line: 144, End line: 169

```python
class View:

    def http_method_not_allowed(self, request, *args, **kwargs):
        logger.warning(
            "Method Not Allowed (%s): %s",
            request.method,
            request.path,
            extra={"status_code": 405, "request": request},
        )
        return HttpResponseNotAllowed(self._allowed_methods())

    def options(self, request, *args, **kwargs):
        """Handle responding to requests for the OPTIONS HTTP verb."""
        response = HttpResponse()
        response.headers["Allow"] = ", ".join(self._allowed_methods())
        response.headers["Content-Length"] = "0"

        if self.view_is_async:

            async def func():
                return response

            return func()
        else:
            return response

    def _allowed_methods(self):
        return [m.upper() for m in self.http_method_names if hasattr(self, m)]
```
### 2 - django/views/generic/base.py:

Start line: 62, End line: 77

```python
class View:

    @classproperty
    def view_is_async(cls):
        handlers = [
            getattr(cls, method)
            for method in cls.http_method_names
            if (method != "options" and hasattr(cls, method))
        ]
        if not handlers:
            return False
        is_async = asyncio.iscoroutinefunction(handlers[0])
        if not all(asyncio.iscoroutinefunction(h) == is_async for h in handlers[1:]):
            raise ImproperlyConfigured(
                f"{cls.__qualname__} HTTP handlers must either be all sync or all "
                "async."
            )
        return is_async
```
### 3 - django/http/response.py:

Start line: 650, End line: 675

```python
class HttpResponseNotAllowed(HttpResponse):
    status_code = 405

    def __init__(self, permitted_methods, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["Allow"] = ", ".join(permitted_methods)

    def __repr__(self):
        return "<%(cls)s [%(methods)s] status_code=%(status_code)d%(content_type)s>" % {
            "cls": self.__class__.__name__,
            "status_code": self.status_code,
            "content_type": self._content_type_for_repr,
            "methods": self["Allow"],
        }


class HttpResponseGone(HttpResponse):
    status_code = 410


class HttpResponseServerError(HttpResponse):
    status_code = 500


class Http404(Exception):
    pass
```
### 4 - django/views/csrf.py:

Start line: 15, End line: 100

```python
CSRF_FAILURE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta http-equiv="content-type" content="text/html; charset=utf-8">
  <meta name="robots" content="NONE,NOARCHIVE">
  <title>403 Forbidden</title>
  <style type="text/css">
    html * { padding:0; margin:0; }
    body * { padding:10px 20px; }
    body * * { padding:0; }
    body { font:small sans-serif; background:#eee; color:#000; }
    body>div { border-bottom:1px solid #ddd; }
    h1 { font-weight:normal; margin-bottom:.4em; }
    h1 span { font-size:60%; color:#666; font-weight:normal; }
    #info { background:#f6f6f6; }
    #info ul { margin: 0.5em 4em; }
    #info p, #summary p { padding-top:10px; }
    #summary { background: #ffc; }
    #explanation { background:#eee; border-bottom: 0px none; }
  </style>
</head>
<body>
<div id="summary">
  <h1>{{ title }} <span>(403)</span></h1>
  <p>{{ main }}</p>
{% if no_referer %}
  <p>{{ no_referer1 }}</p>
  <p>{{ no_referer2 }}</p>
  <p>{{ no_referer3 }}</p>
{% endif %}
{% if no_cookie %}
  <p>{{ no_cookie1 }}</p>
  <p>{{ no_cookie2 }}</p>
{% endif %}
</div>
{% if DEBUG %}
<div id="info">
  <h2>Help</h2>
    {% if reason %}
    <p>Reason given for failure:</p>
    <pre>
    {{ reason }}
    </pre>
    {% endif %}

  <p>In general, this can occur when there is a genuine Cross Site Request Forgery, or when
  <a
  href="https://docs.djangoproject.com/en/{{ docs_version }}/ref/csrf/">Django’s
  CSRF mechanism</a> has not been used correctly.  For POST forms, you need to
  ensure:</p>

  <ul>
    <li>Your browser is accepting cookies.</li>

    <li>The view function passes a <code>request</code> to the template’s <a
    href="https://docs.djangoproject.com/en/dev/topics/templates/#django.template.backends.base.Template.render"><code>render</code></a>
    method.</li>

    <li>In the template, there is a <code>{% templatetag openblock %} csrf_token
    {% templatetag closeblock %}</code> template tag inside each POST form that
    targets an internal URL.</li>

    <li>If you are not using <code>CsrfViewMiddleware</code>, then you must use
    <code>csrf_protect</code> on any views that use the <code>csrf_token</code>
    template tag, as well as those that accept the POST data.</li>

    <li>The form has a valid CSRF token. After logging in in another browser
    tab or hitting the back button after a login, you may need to reload the
    page with the form, because the token is rotated after a login.</li>
  </ul>

  <p>You’re seeing the help section of this page because you have <code>DEBUG =
  True</code> in your Django settings file. Change that to <code>False</code>,
  and only the initial error message will be displayed.  </p>

  <p>You can customize this page using the CSRF_FAILURE_VIEW setting.</p>
</div>
{% else %}
<div id="explanation">
  <p><small>{{ more }}</small></p>
</div>
{% endif %}
</body>
</html>
"""  # NOQA
```
### 5 - django/views/generic/__init__.py:

Start line: 1, End line: 40

```python
from django.views.generic.base import RedirectView, TemplateView, View
from django.views.generic.dates import (
    ArchiveIndexView,
    DateDetailView,
    DayArchiveView,
    MonthArchiveView,
    TodayArchiveView,
    WeekArchiveView,
    YearArchiveView,
)
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, DeleteView, FormView, UpdateView
from django.views.generic.list import ListView

__all__ = [
    "View",
    "TemplateView",
    "RedirectView",
    "ArchiveIndexView",
    "YearArchiveView",
    "MonthArchiveView",
    "WeekArchiveView",
    "DayArchiveView",
    "TodayArchiveView",
    "DateDetailView",
    "DetailView",
    "FormView",
    "CreateView",
    "UpdateView",
    "DeleteView",
    "ListView",
    "GenericViewError",
]


class GenericViewError(Exception):
    """A problem in a generic view."""

    pass
```
### 6 - django/contrib/auth/password_validation.py:

Start line: 217, End line: 267

```python
class CommonPasswordValidator:
    """
    Validate that the password is not a common password.

    The password is rejected if it occurs in a provided list of passwords,
    which may be gzipped. The list Django ships with contains 20000 common
    passwords (lowercased and deduplicated), created by Royce Williams:
    https://gist.github.com/roycewilliams/226886fd01572964e1431ac8afc999ce
    The password list must be lowercased to match the comparison in validate().
    """

    @cached_property
    def DEFAULT_PASSWORD_LIST_PATH(self):
        return Path(__file__).resolve().parent / "common-passwords.txt.gz"

    def __init__(self, password_list_path=DEFAULT_PASSWORD_LIST_PATH):
        if password_list_path is CommonPasswordValidator.DEFAULT_PASSWORD_LIST_PATH:
            password_list_path = self.DEFAULT_PASSWORD_LIST_PATH
        try:
            with gzip.open(password_list_path, "rt", encoding="utf-8") as f:
                self.passwords = {x.strip() for x in f}
        except OSError:
            with open(password_list_path) as f:
                self.passwords = {x.strip() for x in f}

    def validate(self, password, user=None):
        if password.lower().strip() in self.passwords:
            raise ValidationError(
                _("This password is too common."),
                code="password_too_common",
            )

    def get_help_text(self):
        return _("Your password can’t be a commonly used password.")


class NumericPasswordValidator:
    """
    Validate that the password is not entirely numeric.
    """

    def validate(self, password, user=None):
        if password.isdigit():
            raise ValidationError(
                _("This password is entirely numeric."),
                code="password_entirely_numeric",
            )

    def get_help_text(self):
        return _("Your password can’t be entirely numeric.")
```
### 7 - django/__init__.py:

Start line: 1, End line: 25

```python
from django.utils.version import get_version

VERSION = (4, 2, 0, "alpha", 0)

__version__ = get_version(VERSION)


def setup(set_prefix=True):
    """
    Configure the settings (this happens as a side effect of accessing the
    first setting), configure logging and populate the app registry.
    Set the thread-local urlresolvers script prefix if `set_prefix` is True.
    """
    from django.apps import apps
    from django.conf import settings
    from django.urls import set_script_prefix
    from django.utils.log import configure_logging

    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
    if set_prefix:
        set_script_prefix(
            "/" if settings.FORCE_SCRIPT_NAME is None else settings.FORCE_SCRIPT_NAME
        )
    apps.populate(settings.INSTALLED_APPS)
```
### 8 - django/views/csrf.py:

Start line: 101, End line: 161

```python
CSRF_FAILURE_TEMPLATE_NAME = "403_csrf.html"


def csrf_failure(request, reason="", template_name=CSRF_FAILURE_TEMPLATE_NAME):
    """
    Default view used when request fails CSRF protection
    """
    from django.middleware.csrf import REASON_NO_CSRF_COOKIE, REASON_NO_REFERER

    c = {
        "title": _("Forbidden"),
        "main": _("CSRF verification failed. Request aborted."),
        "reason": reason,
        "no_referer": reason == REASON_NO_REFERER,
        "no_referer1": _(
            "You are seeing this message because this HTTPS site requires a "
            "“Referer header” to be sent by your web browser, but none was "
            "sent. This header is required for security reasons, to ensure "
            "that your browser is not being hijacked by third parties."
        ),
        "no_referer2": _(
            "If you have configured your browser to disable “Referer” headers, "
            "please re-enable them, at least for this site, or for HTTPS "
            "connections, or for “same-origin” requests."
        ),
        "no_referer3": _(
            'If you are using the <meta name="referrer" '
            'content="no-referrer"> tag or including the “Referrer-Policy: '
            "no-referrer” header, please remove them. The CSRF protection "
            "requires the “Referer” header to do strict referer checking. If "
            "you’re concerned about privacy, use alternatives like "
            '<a rel="noreferrer" …> for links to third-party sites.'
        ),
        "no_cookie": reason == REASON_NO_CSRF_COOKIE,
        "no_cookie1": _(
            "You are seeing this message because this site requires a CSRF "
            "cookie when submitting forms. This cookie is required for "
            "security reasons, to ensure that your browser is not being "
            "hijacked by third parties."
        ),
        "no_cookie2": _(
            "If you have configured your browser to disable cookies, please "
            "re-enable them, at least for this site, or for “same-origin” "
            "requests."
        ),
        "DEBUG": settings.DEBUG,
        "docs_version": get_docs_version(),
        "more": _("More information is available with DEBUG=True."),
    }
    try:
        t = loader.get_template(template_name)
    except TemplateDoesNotExist:
        if template_name == CSRF_FAILURE_TEMPLATE_NAME:
            # If the default template doesn't exist, use the string template.
            t = Engine().from_string(CSRF_FAILURE_TEMPLATE)
            c = Context(c)
        else:
            # Raise if a developer-specified template doesn't exist.
            raise
    return HttpResponseForbidden(t.render(c))
```
### 9 - django/http/response.py:

Start line: 409, End line: 443

```python
class HttpResponse(HttpResponseBase):

    @content.setter
    def content(self, value):
        # Consume iterators upon assignment to allow repeated iteration.
        if hasattr(value, "__iter__") and not isinstance(
            value, (bytes, memoryview, str)
        ):
            content = b"".join(self.make_bytes(chunk) for chunk in value)
            if hasattr(value, "close"):
                try:
                    value.close()
                except Exception:
                    pass
        else:
            content = self.make_bytes(value)
        # Create a list of properly encoded bytestrings to support write().
        self._container = [content]

    def __iter__(self):
        return iter(self._container)

    def write(self, content):
        self._container.append(self.make_bytes(content))

    def tell(self):
        return len(self.content)

    def getvalue(self):
        return self.content

    def writable(self):
        return True

    def writelines(self, lines):
        for line in lines:
            self.write(line)
```
### 10 - django/http/response.py:

Start line: 318, End line: 358

```python
class HttpResponseBase:

    # These methods partially implement the file-like object interface.
    # See https://docs.python.org/library/io.html#io.IOBase

    # The WSGI server must call this method upon completion of the request.
    # See http://blog.dscpl.com.au/2012/10/obligations-for-calling-close-on.html
    def close(self):
        for closer in self._resource_closers:
            try:
                closer()
            except Exception:
                pass
        # Free resources that were still referenced.
        self._resource_closers.clear()
        self.closed = True
        signals.request_finished.send(sender=self._handler_class)

    def write(self, content):
        raise OSError("This %s instance is not writable" % self.__class__.__name__)

    def flush(self):
        pass

    def tell(self):
        raise OSError(
            "This %s instance cannot tell its position" % self.__class__.__name__
        )

    # These methods partially implement a stream-like object interface.
    # See https://docs.python.org/library/io.html#io.IOBase

    def readable(self):
        return False

    def seekable(self):
        return False

    def writable(self):
        return False

    def writelines(self, lines):
        raise OSError("This %s instance is not writable" % self.__class__.__name__)
```
### 17 - django/views/generic/base.py:

Start line: 35, End line: 60

```python
class View:
    """
    Intentionally simple parent class for all views. Only implements
    dispatch-by-method and simple sanity checking.
    """

    http_method_names = [
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "head",
        "options",
        "trace",
    ]

    def __init__(self, **kwargs):
        """
        Constructor. Called in the URLconf; can contain helpful extra
        keyword arguments, and other things.
        """
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        for key, value in kwargs.items():
            setattr(self, key, value)
```
### 29 - django/views/generic/base.py:

Start line: 79, End line: 122

```python
class View:

    @classonlymethod
    def as_view(cls, **initkwargs):
        """Main entry point for a request-response process."""
        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError(
                    "The method name %s is not accepted as a keyword argument "
                    "to %s()." % (key, cls.__name__)
                )
            if not hasattr(cls, key):
                raise TypeError(
                    "%s() received an invalid keyword %r. as_view "
                    "only accepts arguments that are already "
                    "attributes of the class." % (cls.__name__, key)
                )

        def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            self.setup(request, *args, **kwargs)
            if not hasattr(self, "request"):
                raise AttributeError(
                    "%s instance has no 'request' attribute. Did you override "
                    "setup() and forget to call super()?" % cls.__name__
                )
            return self.dispatch(request, *args, **kwargs)

        view.view_class = cls
        view.view_initkwargs = initkwargs

        # __name__ and __qualname__ are intentionally left unchanged as
        # view_class should be used to robustly determine the name of the view
        # instead.
        view.__doc__ = cls.__doc__
        view.__module__ = cls.__module__
        view.__annotations__ = cls.dispatch.__annotations__
        # Copy possible attributes set by decorators, e.g. @csrf_exempt, from
        # the dispatch method.
        view.__dict__.update(cls.dispatch.__dict__)

        # Mark the callback if the view class is async.
        if cls.view_is_async:
            view._is_coroutine = asyncio.coroutines._is_coroutine

        return view
```
### 39 - django/views/generic/base.py:

Start line: 1, End line: 32

```python
import asyncio
import logging

from django.core.exceptions import ImproperlyConfigured
from django.http import (
    HttpResponse,
    HttpResponseGone,
    HttpResponseNotAllowed,
    HttpResponsePermanentRedirect,
    HttpResponseRedirect,
)
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.decorators import classonlymethod
from django.utils.functional import classproperty

logger = logging.getLogger("django.request")


class ContextMixin:
    """
    A default context mixin that passes the keyword arguments received by
    get_context_data() as the template context.
    """

    extra_context = None

    def get_context_data(self, **kwargs):
        kwargs.setdefault("view", self)
        if self.extra_context is not None:
            kwargs.update(self.extra_context)
        return kwargs
```
### 115 - django/views/generic/base.py:

Start line: 172, End line: 207

```python
class TemplateResponseMixin:
    """A mixin that can be used to render a template."""

    template_name = None
    template_engine = None
    response_class = TemplateResponse
    content_type = None

    def render_to_response(self, context, **response_kwargs):
        """
        Return a response, using the `response_class` for this view, with a
        template rendered with the given context.

        Pass response_kwargs to the constructor of the response class.
        """
        response_kwargs.setdefault("content_type", self.content_type)
        return self.response_class(
            request=self.request,
            template=self.get_template_names(),
            context=context,
            using=self.template_engine,
            **response_kwargs,
        )

    def get_template_names(self):
        """
        Return a list of template names to be used for the request. Must return
        a list. May not be called if render_to_response() is overridden.
        """
        if self.template_name is None:
            raise ImproperlyConfigured(
                "TemplateResponseMixin requires either a definition of "
                "'template_name' or an implementation of 'get_template_names()'"
            )
        else:
            return [self.template_name]
```
### 128 - django/views/generic/base.py:

Start line: 210, End line: 244

```python
class TemplateView(TemplateResponseMixin, ContextMixin, View):
    """
    Render a template. Pass keyword arguments from the URLconf to the context.
    """

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        return self.render_to_response(context)


class RedirectView(View):
    """Provide a redirect on any GET request."""

    permanent = False
    url = None
    pattern_name = None
    query_string = False

    def get_redirect_url(self, *args, **kwargs):
        """
        Return the URL redirect to. Keyword arguments from the URL pattern
        match generating the redirect request are provided as kwargs to this
        method.
        """
        if self.url:
            url = self.url % kwargs
        elif self.pattern_name:
            url = reverse(self.pattern_name, args=args, kwargs=kwargs)
        else:
            return None

        args = self.request.META.get("QUERY_STRING", "")
        if args and self.query_string:
            url = "%s?%s" % (url, args)
        return url
```
### 135 - django/views/generic/base.py:

Start line: 124, End line: 142

```python
class View:

    def setup(self, request, *args, **kwargs):
        """Initialize attributes shared by all view methods."""
        if hasattr(self, "get") and not hasattr(self, "head"):
            self.head = self.get
        self.request = request
        self.args = args
        self.kwargs = kwargs

    def dispatch(self, request, *args, **kwargs):
        # Try to dispatch to the right method; if a method doesn't exist,
        # defer to the error handler. Also defer to the error handler if the
        # request method isn't on the approved list.
        if request.method.lower() in self.http_method_names:
            handler = getattr(
                self, request.method.lower(), self.http_method_not_allowed
            )
        else:
            handler = self.http_method_not_allowed
        return handler(request, *args, **kwargs)
```
