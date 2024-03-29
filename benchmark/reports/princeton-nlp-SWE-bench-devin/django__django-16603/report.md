# django__django-16603

| **django/django** | `4e4eda6d6c8a5867dafd2ba9167ad8c064bb644a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1596 |
| **Any found context length** | 423 |
| **Avg pos** | 7.0 |
| **Min pos** | 2 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/handlers/asgi.py b/django/core/handlers/asgi.py
--- a/django/core/handlers/asgi.py
+++ b/django/core/handlers/asgi.py
@@ -1,3 +1,4 @@
+import asyncio
 import logging
 import sys
 import tempfile
@@ -177,15 +178,49 @@ async def handle(self, scope, receive, send):
             body_file.close()
             await self.send_response(error_response, send)
             return
-        # Get the response, using the async mode of BaseHandler.
+        # Try to catch a disconnect while getting response.
+        tasks = [
+            asyncio.create_task(self.run_get_response(request)),
+            asyncio.create_task(self.listen_for_disconnect(receive)),
+        ]
+        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
+        done, pending = done.pop(), pending.pop()
+        # Allow views to handle cancellation.
+        pending.cancel()
+        try:
+            await pending
+        except asyncio.CancelledError:
+            # Task re-raised the CancelledError as expected.
+            pass
+        try:
+            response = done.result()
+        except RequestAborted:
+            body_file.close()
+            return
+        except AssertionError:
+            body_file.close()
+            raise
+        # Send the response.
+        await self.send_response(response, send)
+
+    async def listen_for_disconnect(self, receive):
+        """Listen for disconnect from the client."""
+        message = await receive()
+        if message["type"] == "http.disconnect":
+            raise RequestAborted()
+        # This should never happen.
+        assert False, "Invalid ASGI message after request body: %s" % message["type"]
+
+    async def run_get_response(self, request):
+        """Get async response."""
+        # Use the async mode of BaseHandler.
         response = await self.get_response_async(request)
         response._handler_class = self.__class__
         # Increase chunk size on file responses (ASGI servers handles low-level
         # chunking).
         if isinstance(response, FileResponse):
             response.block_size = self.chunk_size
-        # Send the response.
-        await self.send_response(response, send)
+        return response
 
     async def read_body(self, receive):
         """Reads an HTTP body from an ASGI connection."""

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/handlers/asgi.py | 1 | 1 | 5 | 1 | 1596
| django/core/handlers/asgi.py | 180 | 188 | 2 | 1 | 423


## Problem Statement

```
ASGI http.disconnect not handled on requests with body.
Description
	
Noticed whilst reviewing ​PR 15704 for #33699, we're not handling the ASGI http.disconnect message correctly. Since it's only dealt with whilst reading the request body, http.disconnect is not processed on a request that includes a body. 
​https://github.com/django/django/blob/241fe59b74bb6031fa644f3ad55e6ad6a9187510/django/core/handlers/asgi.py#L189
	async def read_body(self, receive):
		"""Reads an HTTP body from an ASGI connection."""
		# Use the tempfile that auto rolls-over to a disk file as it fills up.
		body_file = tempfile.SpooledTemporaryFile(
			max_size=settings.FILE_UPLOAD_MAX_MEMORY_SIZE, mode="w+b"
		)
		while True:
			message = await receive()
			if message["type"] == "http.disconnect":	# This is the only place `http.disconnect` is handled. 
				body_file.close()
				# Early client disconnect.
				raise RequestAborted()
			# Add a body chunk from the message, if provided.
			if "body" in message:
				body_file.write(message["body"])
			# Quit out if that's the end.
			if not message.get("more_body", False):
				break
		body_file.seek(0)
		return body_file
http.disconnect is designed for long-polling — so we imagine a client opening a request, with a request body, and then disconnecting before the response is generated. 
The protocol server (Daphne/uvicorn/...) will send the http.diconnect message, but it's not handled.
This test fails on main (at 9f5548952906c6ea97200c016734b4f519520a64 — 4.2 pre-alpha)
diff --git a/tests/asgi/tests.py b/tests/asgi/tests.py
index ef7b55724e..a68ca8a473 100644
--- a/tests/asgi/tests.py
+++ b/tests/asgi/tests.py
@@ -188,6 +188,18 @@ class ASGITest(SimpleTestCase):
		 with self.assertRaises(asyncio.TimeoutError):
			 await communicator.receive_output()
 
+	async def test_disconnect_with_body(self):
+		application = get_asgi_application()
+		scope = self.async_request_factory._base_scope(path="/")
+		communicator = ApplicationCommunicator(application, scope)
+		await communicator.send_input({
+			"type": "http.request",
+			"body": b"some body",
+		})
+		await communicator.send_input({"type": "http.disconnect"})
+		with self.assertRaises(asyncio.TimeoutError):
+			await communicator.receive_output()
+
	 async def test_wrong_connection_type(self):
		 application = get_asgi_application()
		 scope = self.async_request_factory._base_scope(path="/", type="other")
To handle this correctly it looks like we'd need something like Channel's ​`await_many_dispatch()` to keep receiving from the input queue whilst dispatching the request. 🤔

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/core/handlers/asgi.py** | 190 | 209| 177 | 177 | 2421 | 
| **-> 2 <-** | **1 django/core/handlers/asgi.py** | 162 | 188| 246 | 423 | 2421 | 
| 3 | **1 django/core/handlers/asgi.py** | 28 | 134| 879 | 1302 | 2421 | 
| 4 | **1 django/core/handlers/asgi.py** | 137 | 160| 171 | 1473 | 2421 | 
| **-> 5 <-** | **1 django/core/handlers/asgi.py** | 1 | 25| 123 | 1596 | 2421 | 
| 6 | 2 django/http/request.py | 338 | 367| 210 | 1806 | 7999 | 
| 7 | 3 django/contrib/staticfiles/handlers.py | 84 | 116| 264 | 2070 | 8827 | 
| 8 | **3 django/core/handlers/asgi.py** | 240 | 294| 458 | 2528 | 8827 | 
| 9 | 4 django/dispatch/dispatcher.py | 119 | 157| 260 | 2788 | 12224 | 
| 10 | 4 django/dispatch/dispatcher.py | 332 | 410| 547 | 3335 | 12224 | 
| 11 | 5 django/utils/deprecation.py | 86 | 137| 362 | 3697 | 13282 | 
| 12 | 5 django/dispatch/dispatcher.py | 269 | 330| 433 | 4130 | 13282 | 
| 13 | 5 django/utils/deprecation.py | 139 | 157| 122 | 4252 | 13282 | 
| 14 | 6 django/core/servers/basehttp.py | 155 | 173| 171 | 4423 | 15423 | 
| 15 | 6 django/http/request.py | 407 | 438| 255 | 4678 | 15423 | 
| 16 | 6 django/core/servers/basehttp.py | 214 | 233| 170 | 4848 | 15423 | 
| 17 | 6 django/core/servers/basehttp.py | 114 | 130| 152 | 5000 | 15423 | 
| 18 | 6 django/core/servers/basehttp.py | 176 | 212| 279 | 5279 | 15423 | 
| 19 | 6 django/core/servers/basehttp.py | 132 | 153| 208 | 5487 | 15423 | 
| 20 | 7 django/db/utils.py | 138 | 194| 517 | 6004 | 17322 | 
| 21 | 8 django/http/response.py | 1 | 29| 186 | 6190 | 22725 | 
| 22 | 9 django/core/handlers/exception.py | 1 | 21| 118 | 6308 | 23855 | 
| 23 | 9 django/http/response.py | 529 | 545| 122 | 6430 | 23855 | 
| 24 | 9 django/dispatch/dispatcher.py | 205 | 267| 415 | 6845 | 23855 | 
| 25 | 9 django/core/servers/basehttp.py | 56 | 83| 189 | 7034 | 23855 | 
| 26 | 10 django/utils/connection.py | 34 | 86| 326 | 7360 | 24374 | 
| 27 | 10 django/core/servers/basehttp.py | 255 | 282| 245 | 7605 | 24374 | 
| 28 | 10 django/core/servers/basehttp.py | 1 | 25| 175 | 7780 | 24374 | 
| 29 | 11 django/core/handlers/wsgi.py | 56 | 120| 568 | 8348 | 26151 | 
| 30 | **11 django/core/handlers/asgi.py** | 211 | 238| 237 | 8585 | 26151 | 
| 31 | 11 django/core/servers/basehttp.py | 86 | 111| 204 | 8789 | 26151 | 
| 32 | **11 django/core/handlers/asgi.py** | 296 | 320| 169 | 8958 | 26151 | 
| 33 | 11 django/contrib/staticfiles/handlers.py | 1 | 64| 436 | 9394 | 26151 | 
| 34 | 12 django/core/handlers/base.py | 343 | 374| 213 | 9607 | 28801 | 
| 35 | 12 django/http/request.py | 1 | 48| 286 | 9893 | 28801 | 
| 36 | 12 django/http/response.py | 321 | 361| 286 | 10179 | 28801 | 
| 37 | 12 django/dispatch/dispatcher.py | 412 | 465| 511 | 10690 | 28801 | 
| 38 | 12 django/dispatch/dispatcher.py | 50 | 117| 503 | 11193 | 28801 | 
| 39 | 13 django/core/management/commands/testserver.py | 38 | 66| 237 | 11430 | 29247 | 
| 40 | 14 django/core/asgi.py | 1 | 14| 0 | 11430 | 29332 | 
| 41 | 14 django/http/response.py | 510 | 527| 127 | 11557 | 29332 | 
| 42 | 15 django/core/checks/async_checks.py | 1 | 17| 0 | 11557 | 29426 | 
| 43 | 15 django/core/handlers/exception.py | 63 | 158| 605 | 12162 | 29426 | 
| 44 | 15 django/core/handlers/base.py | 136 | 172| 257 | 12419 | 29426 | 
| 45 | 16 django/core/checks/model_checks.py | 161 | 185| 267 | 12686 | 31236 | 
| 46 | 16 django/dispatch/dispatcher.py | 159 | 203| 314 | 13000 | 31236 | 
| 47 | 17 django/db/transaction.py | 224 | 313| 635 | 13635 | 33560 | 
| 48 | 18 django/http/multipartparser.py | 606 | 645| 245 | 13880 | 38813 | 
| 49 | 19 django/middleware/common.py | 153 | 179| 255 | 14135 | 40359 | 
| 50 | 20 django/contrib/auth/middleware.py | 1 | 22| 143 | 14278 | 41438 | 
| 51 | 20 django/http/request.py | 663 | 691| 229 | 14507 | 41438 | 
| 52 | 20 django/http/response.py | 692 | 717| 159 | 14666 | 41438 | 
| 53 | 20 django/http/request.py | 51 | 127| 528 | 15194 | 41438 | 
| 54 | 21 django/views/generic/base.py | 145 | 179| 204 | 15398 | 43344 | 
| 55 | 21 django/utils/connection.py | 1 | 31| 192 | 15590 | 43344 | 
| 56 | 22 django/core/management/commands/sendtestemail.py | 32 | 47| 120 | 15710 | 43655 | 
| 57 | 22 django/http/request.py | 257 | 282| 191 | 15901 | 43655 | 
| 58 | 22 django/core/servers/basehttp.py | 235 | 252| 165 | 16066 | 43655 | 
| 59 | 23 django/utils/cache.py | 234 | 258| 237 | 16303 | 47457 | 
| 60 | 23 django/http/response.py | 412 | 446| 210 | 16513 | 47457 | 
| 61 | 23 django/http/response.py | 630 | 653| 195 | 16708 | 47457 | 
| 62 | 24 django/db/backends/postgresql/features.py | 1 | 112| 895 | 17603 | 48502 | 
| 63 | 24 django/core/handlers/wsgi.py | 1 | 53| 375 | 17978 | 48502 | 
| 64 | 25 django/core/mail/message.py | 1 | 52| 349 | 18327 | 52216 | 
| 65 | 25 django/views/generic/base.py | 63 | 78| 131 | 18458 | 52216 | 
| 66 | 26 django/core/mail/backends/smtp.py | 100 | 141| 269 | 18727 | 53335 | 
| 67 | 26 django/db/transaction.py | 316 | 341| 181 | 18908 | 53335 | 
| 68 | 27 django/contrib/admin/tests.py | 1 | 37| 265 | 19173 | 54977 | 
| 69 | 28 django/db/migrations/executor.py | 307 | 411| 862 | 20035 | 58418 | 
| 70 | 29 django/db/__init__.py | 1 | 62| 299 | 20334 | 58717 | 
| 71 | 29 django/http/multipartparser.py | 184 | 363| 1354 | 21688 | 58717 | 
| 72 | 30 django/core/mail/backends/locmem.py | 1 | 32| 183 | 21871 | 58901 | 
| 73 | 30 django/http/request.py | 441 | 489| 338 | 22209 | 58901 | 
| 74 | 31 django/conf/global_settings.py | 597 | 668| 453 | 22662 | 64739 | 
| 75 | 31 django/http/response.py | 656 | 689| 157 | 22819 | 64739 | 
| 76 | 32 django/contrib/sessions/middleware.py | 1 | 78| 586 | 23405 | 65326 | 
| 77 | 33 django/db/backends/base/base.py | 1 | 27| 169 | 23574 | 70912 | 
| 78 | 34 django/core/mail/backends/console.py | 1 | 45| 285 | 23859 | 71198 | 
| 79 | 34 django/http/multipartparser.py | 365 | 411| 402 | 24261 | 71198 | 
| 80 | 35 django/core/cache/__init__.py | 38 | 67| 189 | 24450 | 71615 | 
| 81 | 35 django/http/multipartparser.py | 467 | 506| 258 | 24708 | 71615 | 
| 82 | 35 django/http/response.py | 169 | 211| 250 | 24958 | 71615 | 
| 83 | 35 django/db/transaction.py | 182 | 222| 333 | 25291 | 71615 | 
| 84 | 35 django/http/response.py | 82 | 99| 133 | 25424 | 71615 | 
| 85 | 35 django/contrib/auth/middleware.py | 59 | 96| 362 | 25786 | 71615 | 
| 86 | 35 django/dispatch/dispatcher.py | 1 | 48| 291 | 26077 | 71615 | 
| 87 | 35 django/dispatch/dispatcher.py | 468 | 491| 139 | 26216 | 71615 | 
| 88 | 36 django/core/files/uploadhandler.py | 125 | 157| 209 | 26425 | 73071 | 
| 89 | 36 django/utils/cache.py | 261 | 295| 264 | 26689 | 73071 | 
| 90 | 37 django/db/models/query.py | 542 | 559| 157 | 26846 | 93561 | 
| 91 | 37 django/http/request.py | 183 | 192| 113 | 26959 | 93561 | 
| 92 | 37 django/http/request.py | 369 | 405| 300 | 27259 | 93561 | 
| 93 | 38 django/core/checks/security/base.py | 1 | 79| 691 | 27950 | 95750 | 
| 94 | 39 django/utils/decorators.py | 166 | 191| 136 | 28086 | 97162 | 
| 95 | 40 django/contrib/auth/mixins.py | 112 | 136| 150 | 28236 | 98057 | 
| 96 | 40 django/http/multipartparser.py | 508 | 530| 211 | 28447 | 98057 | 
| 97 | 40 django/http/multipartparser.py | 441 | 465| 169 | 28616 | 98057 | 
| 98 | 41 django/db/migrations/questioner.py | 291 | 342| 367 | 28983 | 100753 | 
| 99 | 41 django/http/response.py | 154 | 167| 150 | 29133 | 100753 | 
| 100 | 42 django/core/management/commands/runserver.py | 80 | 120| 401 | 29534 | 102280 | 
| 101 | 42 django/core/management/commands/runserver.py | 1 | 22| 204 | 29738 | 102280 | 
| 102 | 42 django/db/utils.py | 237 | 279| 322 | 30060 | 102280 | 
| 103 | 42 django/db/backends/base/base.py | 490 | 585| 609 | 30669 | 102280 | 
| 104 | 42 django/core/management/commands/testserver.py | 1 | 36| 214 | 30883 | 102280 | 
| 105 | 43 django/db/backends/base/features.py | 223 | 374| 1315 | 32198 | 105494 | 
| 106 | 44 django/utils/asyncio.py | 1 | 40| 221 | 32419 | 105716 | 
| 107 | 44 django/contrib/admin/tests.py | 49 | 130| 580 | 32999 | 105716 | 
| 108 | 44 django/core/handlers/base.py | 228 | 298| 492 | 33491 | 105716 | 
| 109 | 45 django/middleware/csrf.py | 413 | 468| 577 | 34068 | 109827 | 
| 110 | 45 django/core/handlers/base.py | 1 | 102| 755 | 34823 | 109827 | 
| 111 | 45 django/core/management/commands/sendtestemail.py | 1 | 30| 196 | 35019 | 109827 | 
| 112 | 45 django/db/transaction.py | 1 | 82| 442 | 35461 | 109827 | 
| 113 | 45 django/http/response.py | 364 | 410| 309 | 35770 | 109827 | 
| 114 | 45 django/contrib/staticfiles/handlers.py | 67 | 81| 127 | 35897 | 109827 | 
| 115 | 45 django/core/mail/message.py | 178 | 187| 118 | 36015 | 109827 | 
| 116 | 45 django/contrib/auth/mixins.py | 46 | 73| 232 | 36247 | 109827 | 
| 117 | 45 django/core/handlers/exception.py | 161 | 186| 167 | 36414 | 109827 | 
| 118 | 45 django/http/multipartparser.py | 576 | 604| 220 | 36634 | 109827 | 
| 119 | 45 django/core/checks/security/base.py | 81 | 180| 732 | 37366 | 109827 | 
| 120 | 46 django/db/models/base.py | 1 | 66| 361 | 37727 | 128554 | 
| 121 | 47 django/core/management/commands/migrate.py | 270 | 368| 813 | 38540 | 132480 | 
| 122 | 47 django/core/checks/security/base.py | 183 | 223| 306 | 38846 | 132480 | 
| 123 | 47 django/core/files/uploadhandler.py | 31 | 65| 203 | 39049 | 132480 | 
| 124 | 47 django/http/multipartparser.py | 681 | 744| 468 | 39517 | 132480 | 
| 125 | 48 django/core/checks/messages.py | 59 | 82| 161 | 39678 | 133060 | 
| 126 | 49 django/db/backends/oracle/creation.py | 29 | 124| 768 | 40446 | 137075 | 
| 127 | 49 django/contrib/admin/tests.py | 39 | 47| 138 | 40584 | 137075 | 
| 128 | 49 django/core/checks/security/base.py | 259 | 284| 211 | 40795 | 137075 | 
| 129 | 49 django/http/response.py | 303 | 319| 181 | 40976 | 137075 | 
| 130 | 49 django/core/management/commands/runserver.py | 122 | 166| 361 | 41337 | 137075 | 
| 131 | 50 django/contrib/auth/__init__.py | 147 | 162| 129 | 41466 | 138789 | 
| 132 | 50 django/db/backends/base/base.py | 263 | 337| 521 | 41987 | 138789 | 
| 133 | 50 django/middleware/csrf.py | 296 | 346| 450 | 42437 | 138789 | 
| 134 | 50 django/http/multipartparser.py | 647 | 678| 249 | 42686 | 138789 | 
| 135 | 50 django/db/backends/oracle/creation.py | 159 | 201| 411 | 43097 | 138789 | 
| 136 | 50 django/core/handlers/base.py | 104 | 134| 227 | 43324 | 138789 | 
| 137 | 50 django/core/mail/message.py | 258 | 283| 322 | 43646 | 138789 | 
| 138 | 51 django/db/backends/utils.py | 146 | 172| 144 | 43790 | 140966 | 
| 139 | 51 django/db/backends/base/base.py | 236 | 261| 247 | 44037 | 140966 | 
| 140 | 51 django/contrib/auth/middleware.py | 25 | 36| 107 | 44144 | 140966 | 
| 141 | 52 django/db/backends/sqlite3/features.py | 63 | 130| 528 | 44672 | 142310 | 
| 142 | 53 django/utils/autoreload.py | 384 | 428| 273 | 44945 | 147436 | 
| 143 | 53 django/http/multipartparser.py | 115 | 129| 133 | 45078 | 147436 | 
| 144 | 53 django/utils/autoreload.py | 585 | 608| 177 | 45255 | 147436 | 
| 145 | 53 django/middleware/common.py | 100 | 115| 165 | 45420 | 147436 | 
| 146 | 54 django/db/backends/base/creation.py | 328 | 352| 282 | 45702 | 150377 | 
| 147 | 54 django/middleware/csrf.py | 1 | 55| 480 | 46182 | 150377 | 
| 148 | 55 django/http/__init__.py | 1 | 53| 241 | 46423 | 150618 | 
| 149 | 56 django/middleware/security.py | 33 | 67| 251 | 46674 | 151144 | 
| 150 | 56 django/utils/autoreload.py | 311 | 327| 162 | 46836 | 151144 | 
| 151 | 56 django/contrib/admin/tests.py | 184 | 202| 177 | 47013 | 151144 | 
| 152 | 56 django/db/transaction.py | 137 | 180| 367 | 47380 | 151144 | 
| 153 | 57 django/db/backends/postgresql/client.py | 1 | 65| 467 | 47847 | 151611 | 
| 154 | 58 django/core/checks/security/csrf.py | 1 | 42| 305 | 48152 | 152076 | 
| 155 | 58 django/http/request.py | 284 | 336| 351 | 48503 | 152076 | 
| 156 | 58 django/utils/autoreload.py | 1 | 56| 293 | 48796 | 152076 | 
| 157 | 59 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 36 | 113| 598 | 49394 | 152876 | 
| 158 | 59 django/core/mail/backends/smtp.py | 67 | 98| 260 | 49654 | 152876 | 
| 159 | 59 django/core/mail/message.py | 157 | 175| 218 | 49872 | 152876 | 
| 160 | 60 django/contrib/auth/password_validation.py | 217 | 267| 386 | 50258 | 154770 | 
| 161 | 60 django/views/generic/base.py | 256 | 286| 246 | 50504 | 154770 | 
| 162 | 61 django/contrib/syndication/views.py | 1 | 26| 223 | 50727 | 156628 | 
| 163 | 62 django/__init__.py | 1 | 25| 173 | 50900 | 156801 | 
| 164 | 63 django/contrib/gis/views.py | 1 | 23| 160 | 51060 | 156961 | 
| 165 | 64 django/contrib/messages/middleware.py | 1 | 27| 174 | 51234 | 157136 | 
| 166 | 64 django/core/checks/security/base.py | 242 | 256| 121 | 51355 | 157136 | 
| 167 | 65 django/db/backends/oracle/features.py | 89 | 153| 604 | 51959 | 158479 | 
| 168 | 66 django/contrib/gis/geos/prototypes/errcheck.py | 1 | 58| 386 | 52345 | 159116 | 
| 169 | 66 django/middleware/common.py | 34 | 60| 265 | 52610 | 159116 | 
| 170 | 67 django/core/checks/__init__.py | 1 | 48| 327 | 52937 | 159443 | 
| 171 | 67 django/core/checks/messages.py | 40 | 56| 115 | 53052 | 159443 | 
| 172 | 68 django/utils/log.py | 79 | 142| 475 | 53527 | 161117 | 
| 173 | 69 django/contrib/sites/requests.py | 1 | 21| 131 | 53658 | 161248 | 
| 174 | 70 django/core/mail/backends/dummy.py | 1 | 11| 0 | 53658 | 161291 | 
| 175 | 71 django/core/checks/security/sessions.py | 1 | 100| 580 | 54238 | 161872 | 
| 176 | 72 django/contrib/contenttypes/fields.py | 1 | 22| 171 | 54409 | 167700 | 
| 177 | 73 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 54556 | 167848 | 
| 178 | 73 django/contrib/auth/middleware.py | 98 | 125| 195 | 54751 | 167848 | 
| 179 | 73 django/db/backends/base/creation.py | 1 | 104| 754 | 55505 | 167848 | 
| 180 | 73 django/utils/cache.py | 213 | 231| 185 | 55690 | 167848 | 
| 181 | 73 django/db/backends/base/base.py | 587 | 612| 205 | 55895 | 167848 | 
| 182 | 74 django/middleware/gzip.py | 1 | 75| 537 | 56432 | 168386 | 
| 183 | 74 django/db/migrations/questioner.py | 1 | 55| 469 | 56901 | 168386 | 
| 184 | 74 django/views/generic/base.py | 1 | 33| 170 | 57071 | 168386 | 
| 185 | 74 django/middleware/common.py | 118 | 151| 284 | 57355 | 168386 | 
| 186 | 74 django/db/migrations/questioner.py | 90 | 107| 163 | 57518 | 168386 | 
| 187 | 75 django/db/backends/postgresql/psycopg_any.py | 1 | 103| 860 | 58378 | 169247 | 
| 188 | 75 django/db/backends/base/base.py | 339 | 357| 139 | 58517 | 169247 | 
| 189 | 75 django/utils/autoreload.py | 663 | 677| 121 | 58638 | 169247 | 
| 190 | 76 django/db/backends/postgresql/base.py | 246 | 286| 375 | 59013 | 173123 | 
| 191 | 76 django/db/utils.py | 1 | 50| 177 | 59190 | 173123 | 
| 192 | 77 django/contrib/auth/backends.py | 163 | 181| 146 | 59336 | 174879 | 
| 193 | 77 django/core/management/commands/runserver.py | 25 | 66| 284 | 59620 | 174879 | 
| 194 | 77 django/contrib/auth/backends.py | 183 | 234| 377 | 59997 | 174879 | 
| 195 | 78 django/utils/http.py | 1 | 39| 460 | 60457 | 178081 | 
| 196 | 79 django/core/signals.py | 1 | 7| 0 | 60457 | 178108 | 
| 197 | 79 django/utils/autoreload.py | 329 | 344| 146 | 60603 | 178108 | 
| 198 | 80 django/contrib/postgres/signals.py | 1 | 81| 640 | 61243 | 178748 | 
| 199 | 81 django/dispatch/__init__.py | 1 | 10| 0 | 61243 | 178813 | 
| 200 | 82 django/middleware/http.py | 1 | 41| 331 | 61574 | 179144 | 
| 201 | 82 django/utils/deprecation.py | 1 | 38| 235 | 61809 | 179144 | 
| 202 | 83 docs/_ext/djangodocs.py | 26 | 71| 398 | 62207 | 182368 | 
| 203 | 83 django/core/mail/message.py | 356 | 371| 127 | 62334 | 182368 | 
| 204 | 83 django/db/backends/postgresql/base.py | 346 | 372| 228 | 62562 | 182368 | 
| 205 | 83 django/core/checks/model_checks.py | 187 | 228| 345 | 62907 | 182368 | 
| 206 | 83 django/http/request.py | 194 | 216| 166 | 63073 | 182368 | 
| 207 | 83 django/utils/autoreload.py | 90 | 106| 161 | 63234 | 182368 | 
| 208 | 84 django/core/management/commands/makemigrations.py | 405 | 515| 927 | 64161 | 186316 | 


### Hint

```
Thanks! Don't you think it's a bug? 🤔
Don't you think it's a bug? 🤔 I had it down as such, but it's never worked, and once I started thinking about the fix I thought it's probably a non-minor adjustment so... (But happy if you want to re-classify :)
Replying to Carlton Gibson: async def read_body(self, receive): """Reads an HTTP body from an ASGI connection.""" # Use the tempfile that auto rolls-over to a disk file as it fills up. body_file = tempfile.SpooledTemporaryFile( max_size=settings.FILE_UPLOAD_MAX_MEMORY_SIZE, mode="w+b" ) while True: message = await receive() if message["type"] == "http.disconnect": # This is the only place `http.disconnect` is handled. body_file.close() # Early client disconnect. raise RequestAborted() # Add a body chunk from the message, if provided. if "body" in message: body_file.write(message["body"]) # Quit out if that's the end. if not message.get("more_body", False): break body_file.seek(0) return body_file I'm following this ticket for some days and trying to understand the code base learned more about the asynchronous processes and long-polling, IDK this is a solution or not, if we perform a loop only while receiving the message and break it after X interval of time like while True: message = await receive() message_queue.append(message) if time_waited == X : break and do store it in a queue and then perform a loop on the queue data to read the message? Please correct me if I'm wrong or if this sounds immature, thanks
Hi Rahul. ... if we perform a loop only while receiving the message and break it after X interval of time... This won't work I'm afraid. We still need to be able to handle the incoming http.disconnect at any time. It's not 100% straightforward but something like Channels' await_many_dispatch() will be needed. See ​the usage here: it's able to route multiple messages. (I don't think the example there is 100% properly handling the disconnect, as a kind of interrupt there either, since the dispatch task will only process one message at a time, where we're looking to handle a disconnect whilst processing a long-running request… 🤔)
Hi Carlton, I'm not sure if the description of this bug is 100% accurate. From what I can see, http.disconnect is indeed only handled while receiving the body, so we're not going to handle this event once the (potentially empty) body is fully received. Adding "more_body": False to your proposed test makes it pass. I had a look for how Starlette handles http.disconnect, and from what I understand the pattern seems to rely on anyio.create_taskgroup() to tear down request processing if an http.disconnect is received.
```

## Patch

```diff
diff --git a/django/core/handlers/asgi.py b/django/core/handlers/asgi.py
--- a/django/core/handlers/asgi.py
+++ b/django/core/handlers/asgi.py
@@ -1,3 +1,4 @@
+import asyncio
 import logging
 import sys
 import tempfile
@@ -177,15 +178,49 @@ async def handle(self, scope, receive, send):
             body_file.close()
             await self.send_response(error_response, send)
             return
-        # Get the response, using the async mode of BaseHandler.
+        # Try to catch a disconnect while getting response.
+        tasks = [
+            asyncio.create_task(self.run_get_response(request)),
+            asyncio.create_task(self.listen_for_disconnect(receive)),
+        ]
+        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
+        done, pending = done.pop(), pending.pop()
+        # Allow views to handle cancellation.
+        pending.cancel()
+        try:
+            await pending
+        except asyncio.CancelledError:
+            # Task re-raised the CancelledError as expected.
+            pass
+        try:
+            response = done.result()
+        except RequestAborted:
+            body_file.close()
+            return
+        except AssertionError:
+            body_file.close()
+            raise
+        # Send the response.
+        await self.send_response(response, send)
+
+    async def listen_for_disconnect(self, receive):
+        """Listen for disconnect from the client."""
+        message = await receive()
+        if message["type"] == "http.disconnect":
+            raise RequestAborted()
+        # This should never happen.
+        assert False, "Invalid ASGI message after request body: %s" % message["type"]
+
+    async def run_get_response(self, request):
+        """Get async response."""
+        # Use the async mode of BaseHandler.
         response = await self.get_response_async(request)
         response._handler_class = self.__class__
         # Increase chunk size on file responses (ASGI servers handles low-level
         # chunking).
         if isinstance(response, FileResponse):
             response.block_size = self.chunk_size
-        # Send the response.
-        await self.send_response(response, send)
+        return response
 
     async def read_body(self, receive):
         """Reads an HTTP body from an ASGI connection."""

```

## Test Patch

```diff
diff --git a/tests/asgi/tests.py b/tests/asgi/tests.py
--- a/tests/asgi/tests.py
+++ b/tests/asgi/tests.py
@@ -7,8 +7,10 @@
 
 from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
 from django.core.asgi import get_asgi_application
+from django.core.handlers.asgi import ASGIHandler, ASGIRequest
 from django.core.signals import request_finished, request_started
 from django.db import close_old_connections
+from django.http import HttpResponse
 from django.test import (
     AsyncRequestFactory,
     SimpleTestCase,
@@ -16,6 +18,7 @@
     modify_settings,
     override_settings,
 )
+from django.urls import path
 from django.utils.http import http_date
 
 from .urls import sync_waiter, test_filename
@@ -234,6 +237,34 @@ async def test_disconnect(self):
         with self.assertRaises(asyncio.TimeoutError):
             await communicator.receive_output()
 
+    async def test_disconnect_with_body(self):
+        application = get_asgi_application()
+        scope = self.async_request_factory._base_scope(path="/")
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({"type": "http.request", "body": b"some body"})
+        await communicator.send_input({"type": "http.disconnect"})
+        with self.assertRaises(asyncio.TimeoutError):
+            await communicator.receive_output()
+
+    async def test_assert_in_listen_for_disconnect(self):
+        application = get_asgi_application()
+        scope = self.async_request_factory._base_scope(path="/")
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({"type": "http.request"})
+        await communicator.send_input({"type": "http.not_a_real_message"})
+        msg = "Invalid ASGI message after request body: http.not_a_real_message"
+        with self.assertRaisesMessage(AssertionError, msg):
+            await communicator.receive_output()
+
+    async def test_delayed_disconnect_with_body(self):
+        application = get_asgi_application()
+        scope = self.async_request_factory._base_scope(path="/delayed_hello/")
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({"type": "http.request", "body": b"some body"})
+        await communicator.send_input({"type": "http.disconnect"})
+        with self.assertRaises(asyncio.TimeoutError):
+            await communicator.receive_output()
+
     async def test_wrong_connection_type(self):
         application = get_asgi_application()
         scope = self.async_request_factory._base_scope(path="/", type="other")
@@ -318,3 +349,56 @@ async def test_concurrent_async_uses_multiple_thread_pools(self):
         self.assertEqual(len(sync_waiter.active_threads), 2)
 
         sync_waiter.active_threads.clear()
+
+    async def test_asyncio_cancel_error(self):
+        # Flag to check if the view was cancelled.
+        view_did_cancel = False
+
+        # A view that will listen for the cancelled error.
+        async def view(request):
+            nonlocal view_did_cancel
+            try:
+                await asyncio.sleep(0.2)
+                return HttpResponse("Hello World!")
+            except asyncio.CancelledError:
+                # Set the flag.
+                view_did_cancel = True
+                raise
+
+        # Request class to use the view.
+        class TestASGIRequest(ASGIRequest):
+            urlconf = (path("cancel/", view),)
+
+        # Handler to use request class.
+        class TestASGIHandler(ASGIHandler):
+            request_class = TestASGIRequest
+
+        # Request cycle should complete since no disconnect was sent.
+        application = TestASGIHandler()
+        scope = self.async_request_factory._base_scope(path="/cancel/")
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({"type": "http.request"})
+        response_start = await communicator.receive_output()
+        self.assertEqual(response_start["type"], "http.response.start")
+        self.assertEqual(response_start["status"], 200)
+        response_body = await communicator.receive_output()
+        self.assertEqual(response_body["type"], "http.response.body")
+        self.assertEqual(response_body["body"], b"Hello World!")
+        # Give response.close() time to finish.
+        await communicator.wait()
+        self.assertIs(view_did_cancel, False)
+
+        # Request cycle with a disconnect before the view can respond.
+        application = TestASGIHandler()
+        scope = self.async_request_factory._base_scope(path="/cancel/")
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({"type": "http.request"})
+        # Let the view actually start.
+        await asyncio.sleep(0.1)
+        # Disconnect the client.
+        await communicator.send_input({"type": "http.disconnect"})
+        # The handler should not send a response.
+        with self.assertRaises(asyncio.TimeoutError):
+            await communicator.receive_output()
+        await communicator.wait()
+        self.assertIs(view_did_cancel, True)
diff --git a/tests/asgi/urls.py b/tests/asgi/urls.py
--- a/tests/asgi/urls.py
+++ b/tests/asgi/urls.py
@@ -1,4 +1,5 @@
 import threading
+import time
 
 from django.http import FileResponse, HttpResponse
 from django.urls import path
@@ -10,6 +11,12 @@ def hello(request):
     return HttpResponse("Hello %s!" % name)
 
 
+def hello_with_delay(request):
+    name = request.GET.get("name") or "World"
+    time.sleep(1)
+    return HttpResponse(f"Hello {name}!")
+
+
 def hello_meta(request):
     return HttpResponse(
         "From %s" % request.META.get("HTTP_REFERER") or "",
@@ -46,4 +53,5 @@ def post_echo(request):
     path("meta/", hello_meta),
     path("post/", post_echo),
     path("wait/", sync_waiter),
+    path("delayed_hello/", hello_with_delay),
 ]

```


## Code snippets

### 1 - django/core/handlers/asgi.py:

Start line: 190, End line: 209

```python
class ASGIHandler(base.BaseHandler):

    async def read_body(self, receive):
        """Reads an HTTP body from an ASGI connection."""
        # Use the tempfile that auto rolls-over to a disk file as it fills up.
        body_file = tempfile.SpooledTemporaryFile(
            max_size=settings.FILE_UPLOAD_MAX_MEMORY_SIZE, mode="w+b"
        )
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                body_file.close()
                # Early client disconnect.
                raise RequestAborted()
            # Add a body chunk from the message, if provided.
            if "body" in message:
                body_file.write(message["body"])
            # Quit out if that's the end.
            if not message.get("more_body", False):
                break
        body_file.seek(0)
        return body_file
```
### 2 - django/core/handlers/asgi.py:

Start line: 162, End line: 188

```python
class ASGIHandler(base.BaseHandler):

    async def handle(self, scope, receive, send):
        """
        Handles the ASGI request. Called via the __call__ method.
        """
        # Receive the HTTP request body as a stream object.
        try:
            body_file = await self.read_body(receive)
        except RequestAborted:
            return
        # Request is complete and can be served.
        set_script_prefix(self.get_script_prefix(scope))
        await signals.request_started.asend(sender=self.__class__, scope=scope)
        # Get the request and check for basic issues.
        request, error_response = self.create_request(scope, body_file)
        if request is None:
            body_file.close()
            await self.send_response(error_response, send)
            return
        # Get the response, using the async mode of BaseHandler.
        response = await self.get_response_async(request)
        response._handler_class = self.__class__
        # Increase chunk size on file responses (ASGI servers handles low-level
        # chunking).
        if isinstance(response, FileResponse):
            response.block_size = self.chunk_size
        # Send the response.
        await self.send_response(response, send)
```
### 3 - django/core/handlers/asgi.py:

Start line: 28, End line: 134

```python
class ASGIRequest(HttpRequest):
    """
    Custom request subclass that decodes from an ASGI-standard request dict
    and wraps request body handling.
    """

    # Number of seconds until a Request gives up on trying to read a request
    # body and aborts.
    body_receive_timeout = 60

    def __init__(self, scope, body_file):
        self.scope = scope
        self._post_parse_error = False
        self._read_started = False
        self.resolver_match = None
        self.script_name = self.scope.get("root_path", "")
        if self.script_name:
            # TODO: Better is-prefix checking, slash handling?
            self.path_info = scope["path"].removeprefix(self.script_name)
        else:
            self.path_info = scope["path"]
        # The Django path is different from ASGI scope path args, it should
        # combine with script name.
        if self.script_name:
            self.path = "%s/%s" % (
                self.script_name.rstrip("/"),
                self.path_info.replace("/", "", 1),
            )
        else:
            self.path = scope["path"]
        # HTTP basics.
        self.method = self.scope["method"].upper()
        # Ensure query string is encoded correctly.
        query_string = self.scope.get("query_string", "")
        if isinstance(query_string, bytes):
            query_string = query_string.decode()
        self.META = {
            "REQUEST_METHOD": self.method,
            "QUERY_STRING": query_string,
            "SCRIPT_NAME": self.script_name,
            "PATH_INFO": self.path_info,
            # WSGI-expecting code will need these for a while
            "wsgi.multithread": True,
            "wsgi.multiprocess": True,
        }
        if self.scope.get("client"):
            self.META["REMOTE_ADDR"] = self.scope["client"][0]
            self.META["REMOTE_HOST"] = self.META["REMOTE_ADDR"]
            self.META["REMOTE_PORT"] = self.scope["client"][1]
        if self.scope.get("server"):
            self.META["SERVER_NAME"] = self.scope["server"][0]
            self.META["SERVER_PORT"] = str(self.scope["server"][1])
        else:
            self.META["SERVER_NAME"] = "unknown"
            self.META["SERVER_PORT"] = "0"
        # Headers go into META.
        for name, value in self.scope.get("headers", []):
            name = name.decode("latin1")
            if name == "content-length":
                corrected_name = "CONTENT_LENGTH"
            elif name == "content-type":
                corrected_name = "CONTENT_TYPE"
            else:
                corrected_name = "HTTP_%s" % name.upper().replace("-", "_")
            # HTTP/2 say only ASCII chars are allowed in headers, but decode
            # latin1 just in case.
            value = value.decode("latin1")
            if corrected_name in self.META:
                value = self.META[corrected_name] + "," + value
            self.META[corrected_name] = value
        # Pull out request encoding, if provided.
        self._set_content_type_params(self.META)
        # Directly assign the body file to be our stream.
        self._stream = body_file
        # Other bits.
        self.resolver_match = None

    @cached_property
    def GET(self):
        return QueryDict(self.META["QUERY_STRING"])

    def _get_scheme(self):
        return self.scope.get("scheme") or super()._get_scheme()

    def _get_post(self):
        if not hasattr(self, "_post"):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    def _get_files(self):
        if not hasattr(self, "_files"):
            self._load_post_and_files()
        return self._files

    POST = property(_get_post, _set_post)
    FILES = property(_get_files)

    @cached_property
    def COOKIES(self):
        return parse_cookie(self.META.get("HTTP_COOKIE", ""))

    def close(self):
        super().close()
        self._stream.close()
```
### 4 - django/core/handlers/asgi.py:

Start line: 137, End line: 160

```python
class ASGIHandler(base.BaseHandler):
    """Handler for ASGI requests."""

    request_class = ASGIRequest
    # Size to chunk response bodies into for multiple response messages.
    chunk_size = 2**16

    def __init__(self):
        super().__init__()
        self.load_middleware(is_async=True)

    async def __call__(self, scope, receive, send):
        """
        Async entrypoint - parses the request and hands off to get_response.
        """
        # Serve only HTTP connections.
        # FIXME: Allow to override this.
        if scope["type"] != "http":
            raise ValueError(
                "Django can only handle ASGI/HTTP connections, not %s." % scope["type"]
            )

        async with ThreadSensitiveContext():
            await self.handle(scope, receive, send)
```
### 5 - django/core/handlers/asgi.py:

Start line: 1, End line: 25

```python
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing

from asgiref.sync import ThreadSensitiveContext, sync_to_async

from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseServerError,
    QueryDict,
    parse_cookie,
)
from django.urls import set_script_prefix
from django.utils.functional import cached_property

logger = logging.getLogger("django.request")
```
### 6 - django/http/request.py:

Start line: 338, End line: 367

```python
class HttpRequest:

    @property
    def body(self):
        if not hasattr(self, "_body"):
            if self._read_started:
                raise RawPostDataException(
                    "You cannot access body after reading from request's data stream"
                )

            # Limit the maximum request data size that will be handled in-memory.
            if (
                settings.DATA_UPLOAD_MAX_MEMORY_SIZE is not None
                and int(self.META.get("CONTENT_LENGTH") or 0)
                > settings.DATA_UPLOAD_MAX_MEMORY_SIZE
            ):
                raise RequestDataTooBig(
                    "Request body exceeded settings.DATA_UPLOAD_MAX_MEMORY_SIZE."
                )

            try:
                self._body = self.read()
            except OSError as e:
                raise UnreadablePostError(*e.args) from e
            finally:
                self._stream.close()
            self._stream = BytesIO(self._body)
        return self._body

    def _mark_post_parse_error(self):
        self._post = QueryDict()
        self._files = MultiValueDict()
```
### 7 - django/contrib/staticfiles/handlers.py:

Start line: 84, End line: 116

```python
class ASGIStaticFilesHandler(StaticFilesHandlerMixin, ASGIHandler):
    """
    ASGI application which wraps another and intercepts requests for static
    files, passing them off to Django's static file serving.
    """

    def __init__(self, application):
        self.application = application
        self.base_url = urlparse(self.get_base_url())

    async def __call__(self, scope, receive, send):
        # Only even look at HTTP requests
        if scope["type"] == "http" and self._should_handle(scope["path"]):
            # Serve static content
            # (the one thing super() doesn't do is __call__, apparently)
            return await super().__call__(scope, receive, send)
        # Hand off to the main app
        return await self.application(scope, receive, send)

    async def get_response_async(self, request):
        response = await super().get_response_async(request)
        response._resource_closers.append(request.close)
        # FileResponse is not async compatible.
        if response.streaming and not response.is_async:
            _iterator = response.streaming_content

            async def awrapper():
                for part in await sync_to_async(list)(_iterator):
                    yield part

            response.streaming_content = awrapper()
        return response
```
### 8 - django/core/handlers/asgi.py:

Start line: 240, End line: 294

```python
class ASGIHandler(base.BaseHandler):

    async def send_response(self, response, send):
        """Encode and send a response out over ASGI."""
        # Collect cookies into headers. Have to preserve header case as there
        # are some non-RFC compliant clients that require e.g. Content-Type.
        response_headers = []
        for header, value in response.items():
            if isinstance(header, str):
                header = header.encode("ascii")
            if isinstance(value, str):
                value = value.encode("latin1")
            response_headers.append((bytes(header), bytes(value)))
        for c in response.cookies.values():
            response_headers.append(
                (b"Set-Cookie", c.output(header="").encode("ascii").strip())
            )
        # Initial response message.
        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": response_headers,
            }
        )
        # Streaming responses need to be pinned to their iterator.
        if response.streaming:
            # - Consume via `__aiter__` and not `streaming_content` directly, to
            #   allow mapping of a sync iterator.
            # - Use aclosing() when consuming aiter.
            #   See https://github.com/python/cpython/commit/6e8dcda
            async with aclosing(aiter(response)) as content:
                async for part in content:
                    for chunk, _ in self.chunk_bytes(part):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                # Ignore "more" as there may be more parts; instead,
                                # use an empty final closing message with False.
                                "more_body": True,
                            }
                        )
            # Final closing message.
            await send({"type": "http.response.body"})
        # Other responses just need chunking.
        else:
            # Yield chunks of response.
            for chunk, last in self.chunk_bytes(response.content):
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": not last,
                    }
                )
        await sync_to_async(response.close, thread_sensitive=True)()
```
### 9 - django/dispatch/dispatcher.py:

Start line: 119, End line: 157

```python
class Signal:

    def disconnect(self, receiver=None, sender=None, dispatch_uid=None):
        """
        Disconnect receiver from sender for signal.

        If weak references are used, disconnect need not be called. The receiver
        will be removed from dispatch automatically.

        Arguments:

            receiver
                The registered receiver to disconnect. May be none if
                dispatch_uid is specified.

            sender
                The registered sender to disconnect

            dispatch_uid
                the unique identifier of the receiver to disconnect
        """
        if dispatch_uid:
            lookup_key = (dispatch_uid, _make_id(sender))
        else:
            lookup_key = (_make_id(receiver), _make_id(sender))

        disconnected = False
        with self.lock:
            self._clear_dead_receivers()
            for index in range(len(self.receivers)):
                r_key, *_ = self.receivers[index]
                if r_key == lookup_key:
                    disconnected = True
                    del self.receivers[index]
                    break
            self.sender_receivers_cache.clear()
        return disconnected

    def has_listeners(self, sender=None):
        sync_receivers, async_receivers = self._live_receivers(sender)
        return bool(sync_receivers) or bool(async_receivers)
```
### 10 - django/dispatch/dispatcher.py:

Start line: 332, End line: 410

```python
class Signal:

    async def asend_robust(self, sender, **named):
        """
        Send signal from sender to all connected receivers catching errors.

        If any receivers are synchronous, they are grouped and called behind a
        sync_to_async() adaption before executing any asynchronous receivers.

        If any receivers are asynchronous, they are grouped and executed
        concurrently with asyncio.gather.

        Arguments:

            sender
                The sender of the signal. Can be any Python object (normally one
                registered with a connect if you actually want something to
                occur).

            named
                Named arguments which will be passed to receivers.

        Return a list of tuple pairs [(receiver, response), ... ].

        If any receiver raises an error (specifically any subclass of
        Exception), return the error instance as the result for that receiver.
        """
        if (
            not self.receivers
            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
        ):
            return []

        # Call each receiver with whatever arguments it can accept.
        # Return a list of tuple pairs [(receiver, response), ... ].
        sync_receivers, async_receivers = self._live_receivers(sender)

        if sync_receivers:

            @sync_to_async
            def sync_send():
                responses = []
                for receiver in sync_receivers:
                    try:
                        response = receiver(signal=self, sender=sender, **named)
                    except Exception as err:
                        self._log_robust_failure(receiver, err)
                        responses.append((receiver, err))
                    else:
                        responses.append((receiver, response))
                return responses

        else:
            sync_send = list

        async def asend_and_wrap_exception(receiver):
            try:
                response = await receiver(signal=self, sender=sender, **named)
            except Exception as err:
                self._log_robust_failure(receiver, err)
                return err
            return response

        responses, async_responses = await asyncio.gather(
            sync_send(),
            asyncio.gather(
                *(asend_and_wrap_exception(receiver) for receiver in async_receivers),
            ),
        )
        responses.extend(zip(async_receivers, async_responses))
        return responses

    def _clear_dead_receivers(self):
        # Note: caller is assumed to hold self.lock.
        if self._dead_receivers:
            self._dead_receivers = False
            self.receivers = [
                r
                for r in self.receivers
                if not (isinstance(r[1], weakref.ReferenceType) and r[1]() is None)
            ]
```
### 30 - django/core/handlers/asgi.py:

Start line: 211, End line: 238

```python
class ASGIHandler(base.BaseHandler):

    def create_request(self, scope, body_file):
        """
        Create the Request object and returns either (request, None) or
        (None, response) if there is an error response.
        """
        try:
            return self.request_class(scope, body_file), None
        except UnicodeDecodeError:
            logger.warning(
                "Bad Request (UnicodeDecodeError)",
                exc_info=sys.exc_info(),
                extra={"status_code": 400},
            )
            return None, HttpResponseBadRequest()
        except RequestDataTooBig:
            return None, HttpResponse("413 Payload too large", status=413)

    def handle_uncaught_exception(self, request, resolver, exc_info):
        """Last-chance handler for exceptions."""
        # There's no WSGI server to catch the exception further up
        # if this fails, so translate it into a plain text response.
        try:
            return super().handle_uncaught_exception(request, resolver, exc_info)
        except Exception:
            return HttpResponseServerError(
                traceback.format_exc() if settings.DEBUG else "Internal Server Error",
                content_type="text/plain",
            )
```
### 32 - django/core/handlers/asgi.py:

Start line: 296, End line: 320

```python
class ASGIHandler(base.BaseHandler):

    @classmethod
    def chunk_bytes(cls, data):
        """
        Chunks some data up so it can be sent in reasonable size messages.
        Yields (chunk, last_chunk) tuples.
        """
        position = 0
        if not data:
            yield data, True
            return
        while position < len(data):
            yield (
                data[position : position + cls.chunk_size],
                (position + cls.chunk_size) >= len(data),
            )
            position += cls.chunk_size

    def get_script_prefix(self, scope):
        """
        Return the script prefix to use from either the scope or a setting.
        """
        if settings.FORCE_SCRIPT_NAME:
            return settings.FORCE_SCRIPT_NAME
        return scope.get("root_path", "") or ""
```
