# django__django-13321

| **django/django** | `35b03788b0607c1f8d2b64e4fa9e1669b0907ea4` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 8114 |
| **Any found context length** | 8114 |
| **Avg pos** | 18.0 |
| **Min pos** | 18 |
| **Max pos** | 18 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/sessions/backends/base.py b/django/contrib/sessions/backends/base.py
--- a/django/contrib/sessions/backends/base.py
+++ b/django/contrib/sessions/backends/base.py
@@ -121,6 +121,15 @@ def decode(self, session_data):
             return signing.loads(session_data, salt=self.key_salt, serializer=self.serializer)
         # RemovedInDjango40Warning: when the deprecation ends, handle here
         # exceptions similar to what _legacy_decode() does now.
+        except signing.BadSignature:
+            try:
+                # Return an empty session if data is not in the pre-Django 3.1
+                # format.
+                return self._legacy_decode(session_data)
+            except Exception:
+                logger = logging.getLogger('django.security.SuspiciousSession')
+                logger.warning('Session data corrupted')
+                return {}
         except Exception:
             return self._legacy_decode(session_data)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/sessions/backends/base.py | 124 | 124 | 18 | 1 | 8114


## Problem Statement

```
Decoding an invalid session data crashes.
Description
	 
		(last modified by Matt Hegarty)
	 
Hi
I recently upgraded my staging server to 3.1. I think that there was an old session which was still active.
On browsing to any URL, I get the crash below. It looks similar to ​this issue.
I cannot login at all with Chrome - each attempt to access the site results in a crash. Login with Firefox works fine.
This is only happening on my Staging site, which is running Gunicorn behind nginx proxy.
Internal Server Error: /overview/
Traceback (most recent call last):
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 215, in _get_session
return self._session_cache
AttributeError: 'SessionStore' object has no attribute '_session_cache'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 118, in decode
return signing.loads(session_data, salt=self.key_salt, serializer=self.serializer)
File "/usr/local/lib/python3.8/site-packages/django/core/signing.py", line 135, in loads
base64d = TimestampSigner(key, salt=salt).unsign(s, max_age=max_age).encode()
File "/usr/local/lib/python3.8/site-packages/django/core/signing.py", line 201, in unsign
result = super().unsign(value)
File "/usr/local/lib/python3.8/site-packages/django/core/signing.py", line 184, in unsign
raise BadSignature('Signature "%s" does not match' % sig)
django.core.signing.BadSignature: Signature "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" does not match
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
File "/usr/local/lib/python3.8/site-packages/django/core/handlers/exception.py", line 47, in inner
response = get_response(request)
File "/usr/local/lib/python3.8/site-packages/django/core/handlers/base.py", line 179, in _get_response
response = wrapped_callback(request, *callback_args, **callback_kwargs)
File "/usr/local/lib/python3.8/site-packages/django/views/generic/base.py", line 73, in view
return self.dispatch(request, *args, **kwargs)
File "/usr/local/lib/python3.8/site-packages/django/contrib/auth/mixins.py", line 50, in dispatch
if not request.user.is_authenticated:
File "/usr/local/lib/python3.8/site-packages/django/utils/functional.py", line 240, in inner
self._setup()
File "/usr/local/lib/python3.8/site-packages/django/utils/functional.py", line 376, in _setup
self._wrapped = self._setupfunc()
File "/usr/local/lib/python3.8/site-packages/django_otp/middleware.py", line 38, in _verify_user
user.otp_device = None
File "/usr/local/lib/python3.8/site-packages/django/utils/functional.py", line 270, in __setattr__
self._setup()
File "/usr/local/lib/python3.8/site-packages/django/utils/functional.py", line 376, in _setup
self._wrapped = self._setupfunc()
File "/usr/local/lib/python3.8/site-packages/django/contrib/auth/middleware.py", line 23, in <lambda>
request.user = SimpleLazyObject(lambda: get_user(request))
File "/usr/local/lib/python3.8/site-packages/django/contrib/auth/middleware.py", line 11, in get_user
request._cached_user = auth.get_user(request)
File "/usr/local/lib/python3.8/site-packages/django/contrib/auth/__init__.py", line 174, in get_user
user_id = _get_user_session_key(request)
File "/usr/local/lib/python3.8/site-packages/django/contrib/auth/__init__.py", line 58, in _get_user_session_key
return get_user_model()._meta.pk.to_python(request.session[SESSION_KEY])
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 65, in __getitem__
return self._session[key]
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 220, in _get_session
self._session_cache = self.load()
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/db.py", line 44, in load
return self.decode(s.session_data) if s else {}
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 122, in decode
return self._legacy_decode(session_data)
File "/usr/local/lib/python3.8/site-packages/django/contrib/sessions/backends/base.py", line 126, in _legacy_decode
encoded_data = base64.b64decode(session_data.encode('ascii'))
File "/usr/local/lib/python3.8/base64.py", line 87, in b64decode
return binascii.a2b_base64(s)
binascii.Error: Incorrect padding

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/contrib/sessions/backends/base.py** | 133 | 150| 196 | 196 | 2760 | 
| 2 | **1 django/contrib/sessions/backends/base.py** | 1 | 36| 208 | 404 | 2760 | 
| 3 | 2 django/views/csrf.py | 15 | 100| 835 | 1239 | 4304 | 
| 4 | 3 django/core/checks/security/base.py | 1 | 84| 743 | 1982 | 6196 | 
| 5 | 4 django/contrib/sessions/middleware.py | 1 | 80| 635 | 2617 | 6832 | 
| 6 | **4 django/contrib/sessions/backends/base.py** | 152 | 235| 563 | 3180 | 6832 | 
| 7 | 5 django/contrib/sessions/base_session.py | 26 | 48| 139 | 3319 | 7121 | 
| 8 | 6 django/core/checks/security/sessions.py | 1 | 98| 572 | 3891 | 7694 | 
| 9 | 7 django/contrib/sessions/backends/signed_cookies.py | 1 | 25| 160 | 4051 | 8215 | 
| 10 | 8 django/contrib/auth/__init__.py | 1 | 58| 393 | 4444 | 9814 | 
| 11 | 9 django/core/signing.py | 1 | 78| 741 | 5185 | 11675 | 
| 12 | 10 django/contrib/sessions/backends/file.py | 75 | 109| 253 | 5438 | 13176 | 
| 13 | 11 django/contrib/sessions/backends/cache.py | 36 | 52| 167 | 5605 | 13743 | 
| 14 | 12 django/conf/global_settings.py | 401 | 491| 794 | 6399 | 19503 | 
| 15 | 13 django/http/request.py | 1 | 41| 285 | 6684 | 24761 | 
| 16 | **13 django/contrib/sessions/backends/base.py** | 320 | 387| 459 | 7143 | 24761 | 
| 17 | 14 django/contrib/messages/storage/cookie.py | 155 | 189| 258 | 7401 | 26259 | 
| **-> 18 <-** | **14 django/contrib/sessions/backends/base.py** | 39 | 131| 713 | 8114 | 26259 | 
| 19 | 15 django/contrib/auth/admin.py | 1 | 22| 188 | 8302 | 27985 | 
| 20 | 16 django/contrib/sessions/backends/cached_db.py | 43 | 66| 172 | 8474 | 28406 | 
| 21 | 16 django/contrib/sessions/backends/cache.py | 54 | 82| 198 | 8672 | 28406 | 
| 22 | 16 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 8882 | 28406 | 
| 23 | 17 django/http/response.py | 1 | 26| 157 | 9039 | 32775 | 
| 24 | 18 django/contrib/sessions/exceptions.py | 1 | 12| 0 | 9039 | 32826 | 
| 25 | 18 django/contrib/sessions/backends/file.py | 111 | 170| 530 | 9569 | 32826 | 
| 26 | 18 django/core/checks/security/base.py | 86 | 186| 742 | 10311 | 32826 | 
| 27 | 19 django/contrib/auth/views.py | 1 | 37| 278 | 10589 | 35490 | 
| 28 | 19 django/core/signing.py | 146 | 184| 379 | 10968 | 35490 | 
| 29 | 20 django/core/checks/security/csrf.py | 1 | 41| 299 | 11267 | 35789 | 
| 30 | 21 django/core/servers/basehttp.py | 122 | 157| 280 | 11547 | 37534 | 
| 31 | 22 django/contrib/sessions/backends/db.py | 1 | 72| 461 | 12008 | 38258 | 
| 32 | 23 django/contrib/auth/middleware.py | 1 | 23| 171 | 12179 | 39252 | 
| 33 | 23 django/contrib/sessions/backends/db.py | 74 | 110| 269 | 12448 | 39252 | 
| 34 | 24 django/middleware/security.py | 34 | 59| 252 | 12700 | 39796 | 
| 35 | 24 django/conf/global_settings.py | 151 | 266| 859 | 13559 | 39796 | 
| 36 | 25 django/contrib/sessions/models.py | 1 | 36| 250 | 13809 | 40046 | 
| 37 | 26 django/core/handlers/exception.py | 54 | 115| 499 | 14308 | 41050 | 
| 38 | 26 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 14561 | 41050 | 
| 39 | 27 django/http/__init__.py | 1 | 22| 197 | 14758 | 41247 | 
| 40 | 28 django/contrib/sessions/serializers.py | 1 | 21| 0 | 14758 | 41334 | 
| 41 | 28 django/contrib/sessions/backends/signed_cookies.py | 27 | 82| 366 | 15124 | 41334 | 
| 42 | 28 django/contrib/sessions/backends/file.py | 1 | 39| 261 | 15385 | 41334 | 
| 43 | 29 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 15547 | 41496 | 
| 44 | 29 django/contrib/messages/storage/cookie.py | 130 | 153| 234 | 15781 | 41496 | 
| 45 | 29 django/core/signing.py | 127 | 143| 170 | 15951 | 41496 | 
| 46 | 30 django/contrib/messages/storage/session.py | 1 | 48| 323 | 16274 | 41820 | 
| 47 | 30 django/contrib/sessions/backends/cache.py | 1 | 34| 213 | 16487 | 41820 | 
| 48 | 30 django/conf/global_settings.py | 492 | 627| 812 | 17299 | 41820 | 
| 49 | 30 django/contrib/auth/middleware.py | 46 | 82| 360 | 17659 | 41820 | 
| 50 | 31 django/db/backends/signals.py | 1 | 4| 0 | 17659 | 41831 | 
| 51 | 31 django/conf/global_settings.py | 267 | 349| 800 | 18459 | 41831 | 
| 52 | 32 django/contrib/auth/signals.py | 1 | 6| 0 | 18459 | 41855 | 
| 53 | 33 django/contrib/sessions/apps.py | 1 | 8| 0 | 18459 | 41892 | 
| 54 | 34 django/views/defaults.py | 1 | 24| 149 | 18608 | 42934 | 
| 55 | 35 django/core/exceptions.py | 102 | 214| 770 | 19378 | 44123 | 
| 56 | 35 django/conf/global_settings.py | 628 | 652| 180 | 19558 | 44123 | 
| 57 | 36 django/db/utils.py | 1 | 49| 154 | 19712 | 46269 | 
| 58 | 37 django/contrib/auth/hashers.py | 617 | 658| 283 | 19995 | 51407 | 
| 59 | 38 django/utils/http.py | 418 | 480| 318 | 20313 | 55578 | 
| 60 | 39 django/utils/log.py | 1 | 75| 484 | 20797 | 57220 | 
| 61 | 39 django/core/servers/basehttp.py | 97 | 119| 211 | 21008 | 57220 | 
| 62 | 39 django/views/defaults.py | 100 | 119| 149 | 21157 | 57220 | 
| 63 | 40 django/middleware/csrf.py | 158 | 179| 173 | 21330 | 60106 | 
| 64 | 41 django/contrib/sessions/management/commands/clearsessions.py | 1 | 22| 124 | 21454 | 60230 | 
| 65 | 41 django/http/request.py | 312 | 332| 189 | 21643 | 60230 | 
| 66 | 42 django/core/signals.py | 1 | 7| 0 | 21643 | 60257 | 
| 67 | 42 django/views/csrf.py | 101 | 155| 577 | 22220 | 60257 | 
| 68 | 42 django/core/servers/basehttp.py | 159 | 178| 170 | 22390 | 60257 | 
| 69 | 42 django/utils/log.py | 162 | 198| 308 | 22698 | 60257 | 
| 70 | 43 django/db/__init__.py | 1 | 18| 141 | 22839 | 60650 | 
| 71 | 44 django/core/checks/messages.py | 53 | 76| 161 | 23000 | 61223 | 
| 72 | 44 django/middleware/csrf.py | 205 | 330| 1222 | 24222 | 61223 | 
| 73 | 44 django/views/csrf.py | 1 | 13| 132 | 24354 | 61223 | 
| 74 | 44 django/contrib/messages/storage/cookie.py | 81 | 99| 144 | 24498 | 61223 | 
| 75 | 45 django/views/debug.py | 155 | 178| 177 | 24675 | 65685 | 
| 76 | 46 django/contrib/admin/sites.py | 1 | 29| 175 | 24850 | 69895 | 
| 77 | 46 django/middleware/csrf.py | 1 | 42| 330 | 25180 | 69895 | 
| 78 | 47 django/utils/autoreload.py | 1 | 45| 217 | 25397 | 74767 | 
| 79 | 48 django/core/handlers/asgi.py | 1 | 123| 979 | 26376 | 77051 | 
| 80 | 48 django/contrib/auth/hashers.py | 441 | 479| 331 | 26707 | 77051 | 
| 81 | 48 django/contrib/sessions/base_session.py | 1 | 23| 149 | 26856 | 77051 | 
| 82 | 48 django/middleware/security.py | 1 | 32| 316 | 27172 | 77051 | 
| 83 | 48 django/views/defaults.py | 79 | 97| 129 | 27301 | 77051 | 
| 84 | 48 django/core/signing.py | 81 | 124| 368 | 27669 | 77051 | 
| 85 | 49 django/contrib/auth/base_user.py | 48 | 153| 694 | 28363 | 78050 | 
| 86 | 49 django/contrib/auth/hashers.py | 1 | 27| 187 | 28550 | 78050 | 
| 87 | 50 django/contrib/auth/models.py | 1 | 32| 216 | 28766 | 81322 | 
| 88 | 51 django/core/handlers/wsgi.py | 64 | 119| 486 | 29252 | 83081 | 
| 89 | 52 django/core/management/commands/loaddata.py | 1 | 35| 177 | 29429 | 86021 | 
| 90 | 52 django/views/debug.py | 396 | 470| 631 | 30060 | 86021 | 
| 91 | 52 django/views/debug.py | 50 | 61| 135 | 30195 | 86021 | 
| 92 | 52 django/http/response.py | 245 | 283| 282 | 30477 | 86021 | 
| 93 | 52 django/utils/autoreload.py | 354 | 393| 266 | 30743 | 86021 | 
| 94 | 53 django/db/backends/base/base.py | 1 | 23| 138 | 30881 | 90904 | 
| 95 | 53 django/http/response.py | 481 | 512| 153 | 31034 | 90904 | 
| 96 | 54 django/db/models/base.py | 507 | 552| 382 | 31416 | 107548 | 
| 97 | 54 django/contrib/auth/views.py | 65 | 104| 295 | 31711 | 107548 | 
| 98 | 55 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 32560 | 108397 | 
| 99 | 55 django/core/checks/security/base.py | 223 | 244| 184 | 32744 | 108397 | 
| 100 | 55 django/conf/global_settings.py | 51 | 150| 1160 | 33904 | 108397 | 
| 101 | 55 django/core/checks/security/base.py | 189 | 220| 223 | 34127 | 108397 | 
| 102 | 56 django/utils/encoding.py | 102 | 115| 130 | 34257 | 110759 | 
| 103 | 56 django/utils/http.py | 1 | 73| 714 | 34971 | 110759 | 
| 104 | 56 django/utils/log.py | 78 | 134| 463 | 35434 | 110759 | 
| 105 | 56 django/contrib/auth/hashers.py | 429 | 439| 127 | 35561 | 110759 | 
| 106 | 57 django/core/files/storage.py | 1 | 22| 158 | 35719 | 113630 | 
| 107 | 57 django/views/debug.py | 141 | 153| 148 | 35867 | 113630 | 
| 108 | 58 django/contrib/redirects/middleware.py | 1 | 51| 355 | 36222 | 113986 | 
| 109 | 58 django/contrib/auth/hashers.py | 342 | 357| 138 | 36360 | 113986 | 
| 110 | 58 django/views/debug.py | 194 | 242| 467 | 36827 | 113986 | 
| 111 | 58 django/middleware/csrf.py | 181 | 203| 230 | 37057 | 113986 | 
| 112 | 59 django/middleware/clickjacking.py | 1 | 46| 359 | 37416 | 114345 | 
| 113 | 59 django/core/handlers/exception.py | 118 | 143| 167 | 37583 | 114345 | 
| 114 | 60 django/utils/baseconv.py | 40 | 70| 273 | 37856 | 115132 | 
| 115 | 61 django/contrib/auth/mixins.py | 44 | 71| 235 | 38091 | 115996 | 
| 116 | 61 django/contrib/auth/middleware.py | 84 | 109| 192 | 38283 | 115996 | 
| 117 | 61 django/contrib/auth/hashers.py | 575 | 614| 257 | 38540 | 115996 | 
| 118 | 62 django/contrib/auth/password_validation.py | 1 | 32| 206 | 38746 | 117482 | 
| 119 | 63 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 38746 | 117552 | 
| 120 | 63 django/middleware/csrf.py | 93 | 119| 224 | 38970 | 117552 | 
| 121 | 64 django/contrib/admin/options.py | 1 | 97| 767 | 39737 | 136121 | 
| 122 | 64 django/contrib/auth/views.py | 129 | 163| 269 | 40006 | 136121 | 
| 123 | 64 django/utils/autoreload.py | 299 | 314| 146 | 40152 | 136121 | 
| 124 | 65 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 40399 | 136368 | 
| 125 | 66 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 41025 | 137980 | 
| 126 | 66 django/conf/global_settings.py | 350 | 400| 785 | 41810 | 137980 | 
| 127 | 66 django/core/servers/basehttp.py | 200 | 217| 210 | 42020 | 137980 | 
| 128 | 67 django/core/serializers/json.py | 62 | 106| 336 | 42356 | 138680 | 
| 129 | 67 django/core/management/commands/loaddata.py | 87 | 157| 640 | 42996 | 138680 | 
| 130 | 67 django/contrib/auth/__init__.py | 135 | 163| 232 | 43228 | 138680 | 
| 131 | 68 django/contrib/auth/urls.py | 1 | 21| 224 | 43452 | 138904 | 
| 132 | 69 django/contrib/messages/middleware.py | 1 | 27| 174 | 43626 | 139079 | 
| 133 | 69 django/db/__init__.py | 40 | 62| 118 | 43744 | 139079 | 
| 134 | 70 django/utils/cache.py | 154 | 191| 447 | 44191 | 142809 | 
| 135 | 71 django/core/mail/message.py | 1 | 52| 346 | 44537 | 146418 | 
| 136 | 72 django/contrib/gis/views.py | 1 | 21| 155 | 44692 | 146573 | 
| 137 | 72 django/http/response.py | 515 | 540| 159 | 44851 | 146573 | 
| 138 | 72 django/contrib/auth/admin.py | 128 | 189| 465 | 45316 | 146573 | 
| 139 | 72 django/contrib/messages/storage/cookie.py | 101 | 128| 261 | 45577 | 146573 | 
| 140 | 72 django/contrib/sessions/backends/file.py | 41 | 55| 131 | 45708 | 146573 | 
| 141 | 72 django/http/response.py | 460 | 478| 186 | 45894 | 146573 | 
| 142 | 73 django/core/validators.py | 1 | 16| 127 | 46021 | 151119 | 
| 143 | 74 django/template/backends/dummy.py | 1 | 53| 325 | 46346 | 151444 | 
| 144 | 75 django/core/cache/backends/memcached.py | 138 | 157| 185 | 46531 | 153356 | 
| 145 | 76 django/core/handlers/base.py | 318 | 349| 212 | 46743 | 155946 | 
| 146 | 77 django/views/generic/base.py | 209 | 240| 247 | 46990 | 157724 | 
| 147 | 78 django/contrib/auth/management/__init__.py | 89 | 149| 441 | 47431 | 158834 | 
| 148 | 79 django/core/mail/backends/dummy.py | 1 | 11| 0 | 47431 | 158877 | 
| 149 | 79 django/http/response.py | 227 | 243| 181 | 47612 | 158877 | 
| 150 | 79 django/contrib/messages/storage/cookie.py | 50 | 64| 149 | 47761 | 158877 | 
| 151 | 80 django/contrib/auth/forms.py | 1 | 30| 228 | 47989 | 162085 | 
| 152 | 81 django/contrib/gis/geos/error.py | 1 | 4| 0 | 47989 | 162109 | 
| 153 | 81 django/contrib/auth/hashers.py | 133 | 151| 189 | 48178 | 162109 | 
| 154 | 82 django/contrib/auth/apps.py | 1 | 29| 213 | 48391 | 162322 | 
| 155 | 83 django/core/cache/backends/base.py | 280 | 293| 112 | 48503 | 164505 | 
| 156 | 84 django/core/cache/backends/dummy.py | 1 | 40| 255 | 48758 | 164761 | 
| 157 | 85 django/contrib/auth/backends.py | 1 | 28| 171 | 48929 | 166523 | 
| 158 | 86 django/core/management/commands/runserver.py | 67 | 105| 397 | 49326 | 167970 | 
| 159 | 86 django/utils/autoreload.py | 48 | 76| 156 | 49482 | 167970 | 
| 160 | 86 django/contrib/auth/hashers.py | 661 | 706| 314 | 49796 | 167970 | 
| 161 | 86 django/contrib/admin/sites.py | 221 | 240| 221 | 50017 | 167970 | 
| 162 | 87 django/db/backends/base/client.py | 1 | 13| 0 | 50017 | 168077 | 
| 163 | 87 django/views/defaults.py | 122 | 149| 214 | 50231 | 168077 | 
| 164 | 88 django/conf/__init__.py | 167 | 226| 546 | 50777 | 170264 | 
| 165 | 88 django/core/cache/backends/memcached.py | 119 | 136| 174 | 50951 | 170264 | 
| 166 | 89 django/views/decorators/gzip.py | 1 | 6| 0 | 50951 | 170315 | 
| 167 | 90 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 51271 | 170635 | 
| 168 | 91 django/core/checks/translation.py | 1 | 65| 445 | 51716 | 171080 | 
| 169 | 92 django/conf/locale/en_GB/formats.py | 5 | 37| 655 | 52371 | 171780 | 
| 170 | 93 django/utils/formats.py | 1 | 57| 377 | 52748 | 173886 | 
| 171 | 93 django/core/management/commands/dumpdata.py | 180 | 204| 224 | 52972 | 173886 | 
| 172 | 93 django/db/models/base.py | 554 | 571| 142 | 53114 | 173886 | 
| 173 | 94 django/core/mail/backends/__init__.py | 1 | 2| 0 | 53114 | 173894 | 
| 174 | 94 django/views/debug.py | 331 | 360| 267 | 53381 | 173894 | 
| 175 | 94 django/utils/autoreload.py | 608 | 620| 117 | 53498 | 173894 | 
| 176 | 95 django/contrib/gis/db/models/proxy.py | 1 | 47| 363 | 53861 | 174563 | 
| 177 | 96 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 53861 | 174641 | 
| 178 | 97 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 53998 | 174778 | 
| 179 | 97 django/core/handlers/asgi.py | 264 | 288| 169 | 54167 | 174778 | 
| 180 | 97 django/core/management/commands/runserver.py | 107 | 159| 502 | 54669 | 174778 | 
| 181 | 98 django/views/decorators/debug.py | 77 | 92| 132 | 54801 | 175367 | 
| 182 | 98 django/core/signing.py | 187 | 213| 202 | 55003 | 175367 | 
| 183 | 99 django/db/migrations/loader.py | 152 | 178| 291 | 55294 | 178437 | 
| 184 | 100 django/conf/locale/en_AU/formats.py | 5 | 37| 655 | 55949 | 179137 | 
| 185 | 101 django/conf/urls/__init__.py | 1 | 23| 152 | 56101 | 179289 | 
| 186 | 102 django/conf/locale/id/formats.py | 5 | 47| 670 | 56771 | 180004 | 
| 187 | 102 django/utils/baseconv.py | 72 | 102| 243 | 57014 | 180004 | 
| 188 | 103 django/contrib/messages/views.py | 1 | 19| 0 | 57014 | 180100 | 
| 189 | 104 django/db/backends/mysql/validation.py | 1 | 31| 239 | 57253 | 180620 | 
| 190 | 105 django/contrib/sitemaps/views.py | 1 | 19| 131 | 57384 | 181394 | 
| 191 | 106 django/contrib/postgres/serializers.py | 1 | 11| 0 | 57384 | 181495 | 
| 192 | 107 django/db/migrations/state.py | 165 | 189| 213 | 57597 | 186617 | 
| 193 | 108 django/conf/locale/sr/formats.py | 5 | 40| 726 | 58323 | 187388 | 
| 194 | 109 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 58323 | 187466 | 
| 195 | 110 django/contrib/messages/__init__.py | 1 | 3| 0 | 58323 | 187490 | 
| 196 | 111 django/contrib/admin/models.py | 39 | 72| 241 | 58564 | 188613 | 
| 197 | 111 django/contrib/sessions/backends/file.py | 57 | 73| 143 | 58707 | 188613 | 
| 198 | 112 django/core/checks/templates.py | 1 | 36| 259 | 58966 | 188873 | 
| 199 | 113 django/contrib/syndication/views.py | 1 | 26| 220 | 59186 | 190612 | 
| 200 | 114 django/middleware/common.py | 149 | 175| 254 | 59440 | 192123 | 
| 201 | 114 django/middleware/common.py | 34 | 61| 257 | 59697 | 192123 | 
| 202 | 115 django/contrib/gis/gdal/prototypes/srs.py | 63 | 84| 317 | 60014 | 193188 | 
| 203 | 115 django/db/models/base.py | 1 | 50| 328 | 60342 | 193188 | 
| 204 | 115 django/utils/http.py | 224 | 255| 244 | 60586 | 193188 | 
| 205 | 116 django/template/response.py | 1 | 43| 383 | 60969 | 194267 | 
| 206 | 117 django/contrib/sites/middleware.py | 1 | 13| 0 | 60969 | 194326 | 
| 207 | 118 django/core/checks/__init__.py | 1 | 26| 270 | 61239 | 194596 | 
| 208 | 118 django/contrib/auth/base_user.py | 1 | 45| 304 | 61543 | 194596 | 
| 209 | 118 django/http/request.py | 615 | 629| 118 | 61661 | 194596 | 
| 210 | 119 django/conf/locale/sr_Latn/formats.py | 5 | 40| 726 | 62387 | 195367 | 


### Hint

```
I tried to run clearsessions, but that didn't help. The only workaround was to delete all rows in the django_session table.
Thanks for this report, however I cannot reproduce this issue. Can you provide a sample project? Support for user sessions created by older versions of Django remains until Django 4.0. See similar tickets #31864, #31592, and #31274, this can be a duplicate of one of them.
Thanks for the response. It does look similar to the other issues you posted. I don't have a reproducible instance at present. The only way I can think to reproduce would be to start up a 3.0 site, login, wait for the session to expire, then upgrade to 3.1. These are the steps that would have happened on the environment where I encountered the issue.
Thanks I was able to reproduce this issue with an invalid session data. Regression in d4fff711d4c97356bd6ba1273d2a5e349326eb5f.
```

## Patch

```diff
diff --git a/django/contrib/sessions/backends/base.py b/django/contrib/sessions/backends/base.py
--- a/django/contrib/sessions/backends/base.py
+++ b/django/contrib/sessions/backends/base.py
@@ -121,6 +121,15 @@ def decode(self, session_data):
             return signing.loads(session_data, salt=self.key_salt, serializer=self.serializer)
         # RemovedInDjango40Warning: when the deprecation ends, handle here
         # exceptions similar to what _legacy_decode() does now.
+        except signing.BadSignature:
+            try:
+                # Return an empty session if data is not in the pre-Django 3.1
+                # format.
+                return self._legacy_decode(session_data)
+            except Exception:
+                logger = logging.getLogger('django.security.SuspiciousSession')
+                logger.warning('Session data corrupted')
+                return {}
         except Exception:
             return self._legacy_decode(session_data)
 

```

## Test Patch

```diff
diff --git a/tests/sessions_tests/tests.py b/tests/sessions_tests/tests.py
--- a/tests/sessions_tests/tests.py
+++ b/tests/sessions_tests/tests.py
@@ -333,11 +333,16 @@ def test_default_hashing_algorith_legacy_decode(self):
             self.assertEqual(self.session._legacy_decode(encoded), data)
 
     def test_decode_failure_logged_to_security(self):
-        bad_encode = base64.b64encode(b'flaskdj:alkdjf').decode('ascii')
-        with self.assertLogs('django.security.SuspiciousSession', 'WARNING') as cm:
-            self.assertEqual({}, self.session.decode(bad_encode))
-        # The failed decode is logged.
-        self.assertIn('corrupted', cm.output[0])
+        tests = [
+            base64.b64encode(b'flaskdj:alkdjf').decode('ascii'),
+            'bad:encoded:value',
+        ]
+        for encoded in tests:
+            with self.subTest(encoded=encoded):
+                with self.assertLogs('django.security.SuspiciousSession', 'WARNING') as cm:
+                    self.assertEqual(self.session.decode(encoded), {})
+                # The failed decode is logged.
+                self.assertIn('Session data corrupted', cm.output[0])
 
     def test_actual_expiry(self):
         # this doesn't work with JSONSerializer (serializing timedelta)

```


## Code snippets

### 1 - django/contrib/sessions/backends/base.py:

Start line: 133, End line: 150

```python
class SessionBase:

    def _legacy_decode(self, session_data):
        # RemovedInDjango40Warning: pre-Django 3.1 format will be invalid.
        encoded_data = base64.b64decode(session_data.encode('ascii'))
        try:
            # could produce ValueError if there is no ':'
            hash, serialized = encoded_data.split(b':', 1)
            expected_hash = self._hash(serialized)
            if not constant_time_compare(hash.decode(), expected_hash):
                raise SuspiciousSession("Session data corrupted")
            else:
                return self.serializer().loads(serialized)
        except Exception as e:
            # ValueError, SuspiciousOperation, unpickling exceptions. If any of
            # these happen, just return an empty dictionary (an empty session).
            if isinstance(e, SuspiciousOperation):
                logger = logging.getLogger('django.security.%s' % e.__class__.__name__)
                logger.warning(str(e))
            return {}
```
### 2 - django/contrib/sessions/backends/base.py:

Start line: 1, End line: 36

```python
import base64
import logging
import string
import warnings
from datetime import datetime, timedelta

from django.conf import settings
from django.contrib.sessions.exceptions import SuspiciousSession
from django.core import signing
from django.core.exceptions import SuspiciousOperation
from django.utils import timezone
from django.utils.crypto import (
    constant_time_compare, get_random_string, salted_hmac,
)
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.module_loading import import_string
from django.utils.translation import LANGUAGE_SESSION_KEY

# session_key should not be case sensitive because some backends can store it
# on case insensitive file systems.
VALID_KEY_CHARS = string.ascii_lowercase + string.digits


class CreateError(Exception):
    """
    Used internally as a consistent exception type to catch from save (see the
    docstring for SessionBase.save() for details).
    """
    pass


class UpdateError(Exception):
    """
    Occurs if Django tries to update a session that was deleted.
    """
    pass
```
### 3 - django/views/csrf.py:

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
"""
```
### 4 - django/core/checks/security/base.py:

Start line: 1, End line: 84

```python
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .. import Error, Tags, Warning, register

REFERRER_POLICY_VALUES = {
    'no-referrer', 'no-referrer-when-downgrade', 'origin',
    'origin-when-cross-origin', 'same-origin', 'strict-origin',
    'strict-origin-when-cross-origin', 'unsafe-url',
}

SECRET_KEY_MIN_LENGTH = 50
SECRET_KEY_MIN_UNIQUE_CHARACTERS = 5

W001 = Warning(
    "You do not have 'django.middleware.security.SecurityMiddleware' "
    "in your MIDDLEWARE so the SECURE_HSTS_SECONDS, "
    "SECURE_CONTENT_TYPE_NOSNIFF, SECURE_BROWSER_XSS_FILTER, "
    "SECURE_REFERRER_POLICY, and SECURE_SSL_REDIRECT settings will have no "
    "effect.",
    id='security.W001',
)

W002 = Warning(
    "You do not have "
    "'django.middleware.clickjacking.XFrameOptionsMiddleware' in your "
    "MIDDLEWARE, so your pages will not be served with an "
    "'x-frame-options' header. Unless there is a good reason for your "
    "site to be served in a frame, you should consider enabling this "
    "header to help prevent clickjacking attacks.",
    id='security.W002',
)

W004 = Warning(
    "You have not set a value for the SECURE_HSTS_SECONDS setting. "
    "If your entire site is served only over SSL, you may want to consider "
    "setting a value and enabling HTTP Strict Transport Security. "
    "Be sure to read the documentation first; enabling HSTS carelessly "
    "can cause serious, irreversible problems.",
    id='security.W004',
)

W005 = Warning(
    "You have not set the SECURE_HSTS_INCLUDE_SUBDOMAINS setting to True. "
    "Without this, your site is potentially vulnerable to attack "
    "via an insecure connection to a subdomain. Only set this to True if "
    "you are certain that all subdomains of your domain should be served "
    "exclusively via SSL.",
    id='security.W005',
)

W006 = Warning(
    "Your SECURE_CONTENT_TYPE_NOSNIFF setting is not set to True, "
    "so your pages will not be served with an "
    "'X-Content-Type-Options: nosniff' header. "
    "You should consider enabling this header to prevent the "
    "browser from identifying content types incorrectly.",
    id='security.W006',
)

W008 = Warning(
    "Your SECURE_SSL_REDIRECT setting is not set to True. "
    "Unless your site should be available over both SSL and non-SSL "
    "connections, you may want to either set this setting True "
    "or configure a load balancer or reverse-proxy server "
    "to redirect all connections to HTTPS.",
    id='security.W008',
)

W009 = Warning(
    "Your SECRET_KEY has less than %(min_length)s characters or less than "
    "%(min_unique_chars)s unique characters. Please generate a long and random "
    "SECRET_KEY, otherwise many of Django's security-critical features will be "
    "vulnerable to attack." % {
        'min_length': SECRET_KEY_MIN_LENGTH,
        'min_unique_chars': SECRET_KEY_MIN_UNIQUE_CHARACTERS,
    },
    id='security.W009',
)

W018 = Warning(
    "You should not have DEBUG set to True in deployment.",
    id='security.W018',
)
```
### 5 - django/contrib/sessions/middleware.py:

Start line: 1, End line: 80

```python
import time
from importlib import import_module

from django.conf import settings
from django.contrib.sessions.backends.base import UpdateError
from django.core.exceptions import SuspiciousOperation
from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import http_date


class SessionMiddleware(MiddlewareMixin):
    # RemovedInDjango40Warning: when the deprecation ends, replace with:
    #   def __init__(self, get_response):
    def __init__(self, get_response=None):
        self._get_response_none_deprecation(get_response)
        self.get_response = get_response
        self._async_check()
        engine = import_module(settings.SESSION_ENGINE)
        self.SessionStore = engine.SessionStore

    def process_request(self, request):
        session_key = request.COOKIES.get(settings.SESSION_COOKIE_NAME)
        request.session = self.SessionStore(session_key)

    def process_response(self, request, response):
        """
        If request.session was modified, or if the configuration is to save the
        session every time, save the changes and set a session cookie or delete
        the session cookie if the session has been emptied.
        """
        try:
            accessed = request.session.accessed
            modified = request.session.modified
            empty = request.session.is_empty()
        except AttributeError:
            return response
        # First check if we need to delete this cookie.
        # The session should be deleted only if the session is entirely empty.
        if settings.SESSION_COOKIE_NAME in request.COOKIES and empty:
            response.delete_cookie(
                settings.SESSION_COOKIE_NAME,
                path=settings.SESSION_COOKIE_PATH,
                domain=settings.SESSION_COOKIE_DOMAIN,
                samesite=settings.SESSION_COOKIE_SAMESITE,
            )
            patch_vary_headers(response, ('Cookie',))
        else:
            if accessed:
                patch_vary_headers(response, ('Cookie',))
            if (modified or settings.SESSION_SAVE_EVERY_REQUEST) and not empty:
                if request.session.get_expire_at_browser_close():
                    max_age = None
                    expires = None
                else:
                    max_age = request.session.get_expiry_age()
                    expires_time = time.time() + max_age
                    expires = http_date(expires_time)
                # Save the session data and refresh the client cookie.
                # Skip session save for 500 responses, refs #3881.
                if response.status_code != 500:
                    try:
                        request.session.save()
                    except UpdateError:
                        raise SuspiciousOperation(
                            "The request's session was deleted before the "
                            "request completed. The user may have logged "
                            "out in a concurrent request, for example."
                        )
                    response.set_cookie(
                        settings.SESSION_COOKIE_NAME,
                        request.session.session_key, max_age=max_age,
                        expires=expires, domain=settings.SESSION_COOKIE_DOMAIN,
                        path=settings.SESSION_COOKIE_PATH,
                        secure=settings.SESSION_COOKIE_SECURE or None,
                        httponly=settings.SESSION_COOKIE_HTTPONLY or None,
                        samesite=settings.SESSION_COOKIE_SAMESITE,
                    )
        return response
```
### 6 - django/contrib/sessions/backends/base.py:

Start line: 152, End line: 235

```python
class SessionBase:

    def update(self, dict_):
        self._session.update(dict_)
        self.modified = True

    def has_key(self, key):
        return key in self._session

    def keys(self):
        return self._session.keys()

    def values(self):
        return self._session.values()

    def items(self):
        return self._session.items()

    def clear(self):
        # To avoid unnecessary persistent storage accesses, we set up the
        # internals directly (loading data wastes time, since we are going to
        # set it to an empty dict anyway).
        self._session_cache = {}
        self.accessed = True
        self.modified = True

    def is_empty(self):
        "Return True when there is no session_key and the session is empty."
        try:
            return not self._session_key and not self._session_cache
        except AttributeError:
            return True

    def _get_new_session_key(self):
        "Return session key that isn't being used."
        while True:
            session_key = get_random_string(32, VALID_KEY_CHARS)
            if not self.exists(session_key):
                return session_key

    def _get_or_create_session_key(self):
        if self._session_key is None:
            self._session_key = self._get_new_session_key()
        return self._session_key

    def _validate_session_key(self, key):
        """
        Key must be truthy and at least 8 characters long. 8 characters is an
        arbitrary lower bound for some minimal key security.
        """
        return key and len(key) >= 8

    def _get_session_key(self):
        return self.__session_key

    def _set_session_key(self, value):
        """
        Validate session key on assignment. Invalid values will set to None.
        """
        if self._validate_session_key(value):
            self.__session_key = value
        else:
            self.__session_key = None

    session_key = property(_get_session_key)
    _session_key = property(_get_session_key, _set_session_key)

    def _get_session(self, no_load=False):
        """
        Lazily load session from storage (unless "no_load" is True, when only
        an empty dict is stored) and store it in the current instance.
        """
        self.accessed = True
        try:
            return self._session_cache
        except AttributeError:
            if self.session_key is None or no_load:
                self._session_cache = {}
            else:
                self._session_cache = self.load()
        return self._session_cache

    _session = property(_get_session)

    def get_session_cookie_age(self):
        return settings.SESSION_COOKIE_AGE
```
### 7 - django/contrib/sessions/base_session.py:

Start line: 26, End line: 48

```python
class AbstractBaseSession(models.Model):
    session_key = models.CharField(_('session key'), max_length=40, primary_key=True)
    session_data = models.TextField(_('session data'))
    expire_date = models.DateTimeField(_('expire date'), db_index=True)

    objects = BaseSessionManager()

    class Meta:
        abstract = True
        verbose_name = _('session')
        verbose_name_plural = _('sessions')

    def __str__(self):
        return self.session_key

    @classmethod
    def get_session_store_class(cls):
        raise NotImplementedError

    def get_decoded(self):
        session_store_class = self.get_session_store_class()
        return session_store_class().decode(self.session_data)
```
### 8 - django/core/checks/security/sessions.py:

Start line: 1, End line: 98

```python
from django.conf import settings

from .. import Tags, Warning, register


def add_session_cookie_message(message):
    return message + (
        " Using a secure-only session cookie makes it more difficult for "
        "network traffic sniffers to hijack user sessions."
    )


W010 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_SECURE to True."
    ),
    id='security.W010',
)

W011 = Warning(
    add_session_cookie_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_SECURE to True."
    ),
    id='security.W011',
)

W012 = Warning(
    add_session_cookie_message("SESSION_COOKIE_SECURE is not set to True."),
    id='security.W012',
)


def add_httponly_message(message):
    return message + (
        " Using an HttpOnly session cookie makes it more difficult for "
        "cross-site scripting attacks to hijack user sessions."
    )


W013 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions' in your INSTALLED_APPS, "
        "but you have not set SESSION_COOKIE_HTTPONLY to True.",
    ),
    id='security.W013',
)

W014 = Warning(
    add_httponly_message(
        "You have 'django.contrib.sessions.middleware.SessionMiddleware' "
        "in your MIDDLEWARE, but you have not set "
        "SESSION_COOKIE_HTTPONLY to True."
    ),
    id='security.W014',
)

W015 = Warning(
    add_httponly_message("SESSION_COOKIE_HTTPONLY is not set to True."),
    id='security.W015',
)


@register(Tags.security, deploy=True)
def check_session_cookie_secure(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_SECURE:
        if _session_app():
            errors.append(W010)
        if _session_middleware():
            errors.append(W011)
        if len(errors) > 1:
            errors = [W012]
    return errors


@register(Tags.security, deploy=True)
def check_session_cookie_httponly(app_configs, **kwargs):
    errors = []
    if not settings.SESSION_COOKIE_HTTPONLY:
        if _session_app():
            errors.append(W013)
        if _session_middleware():
            errors.append(W014)
        if len(errors) > 1:
            errors = [W015]
    return errors


def _session_middleware():
    return 'django.contrib.sessions.middleware.SessionMiddleware' in settings.MIDDLEWARE


def _session_app():
    return "django.contrib.sessions" in settings.INSTALLED_APPS
```
### 9 - django/contrib/sessions/backends/signed_cookies.py:

Start line: 1, End line: 25

```python
from django.contrib.sessions.backends.base import SessionBase
from django.core import signing


class SessionStore(SessionBase):

    def load(self):
        """
        Load the data from the key itself instead of fetching from some
        external data store. Opposite of _get_session_key(), raise BadSignature
        if signature fails.
        """
        try:
            return signing.loads(
                self.session_key,
                serializer=self.serializer,
                # This doesn't handle non-default expiry dates, see #19201
                max_age=self.get_session_cookie_age(),
                salt='django.contrib.sessions.backends.signed_cookies',
            )
        except Exception:
            # BadSignature, ValueError, or unpickling exceptions. If any of
            # these happen, reset the session.
            self.create()
        return {}
```
### 10 - django/contrib/auth/__init__.py:

Start line: 1, End line: 58

```python
import inspect
import re

from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.middleware.csrf import rotate_token
from django.utils.crypto import constant_time_compare
from django.utils.module_loading import import_string

from .signals import user_logged_in, user_logged_out, user_login_failed

SESSION_KEY = '_auth_user_id'
BACKEND_SESSION_KEY = '_auth_user_backend'
HASH_SESSION_KEY = '_auth_user_hash'
REDIRECT_FIELD_NAME = 'next'


def load_backend(path):
    return import_string(path)()


def _get_backends(return_tuples=False):
    backends = []
    for backend_path in settings.AUTHENTICATION_BACKENDS:
        backend = load_backend(backend_path)
        backends.append((backend, backend_path) if return_tuples else backend)
    if not backends:
        raise ImproperlyConfigured(
            'No authentication backends have been defined. Does '
            'AUTHENTICATION_BACKENDS contain anything?'
        )
    return backends


def get_backends():
    return _get_backends(return_tuples=False)


def _clean_credentials(credentials):
    """
    Clean a dictionary of credentials of potentially sensitive info before
    sending to less secure functions.

    Not comprehensive - intended for user_login_failed signal
    """
    SENSITIVE_CREDENTIALS = re.compile('api|token|key|secret|password|signature', re.I)
    CLEANSED_SUBSTITUTE = '********************'
    for key in credentials:
        if SENSITIVE_CREDENTIALS.search(key):
            credentials[key] = CLEANSED_SUBSTITUTE
    return credentials


def _get_user_session_key(request):
    # This value in the session is always serialized to a string, so we need
    # to convert it back to Python whenever we access it.
    return get_user_model()._meta.pk.to_python(request.session[SESSION_KEY])
```
### 16 - django/contrib/sessions/backends/base.py:

Start line: 320, End line: 387

```python
class SessionBase:

    def flush(self):
        """
        Remove the current session data from the database and regenerate the
        key.
        """
        self.clear()
        self.delete()
        self._session_key = None

    def cycle_key(self):
        """
        Create a new session key, while retaining the current session data.
        """
        data = self._session
        key = self.session_key
        self.create()
        self._session_cache = data
        if key:
            self.delete(key)

    # Methods that child classes must implement.

    def exists(self, session_key):
        """
        Return True if the given session_key already exists.
        """
        raise NotImplementedError('subclasses of SessionBase must provide an exists() method')

    def create(self):
        """
        Create a new session instance. Guaranteed to create a new object with
        a unique key and will have saved the result once (with empty data)
        before the method returns.
        """
        raise NotImplementedError('subclasses of SessionBase must provide a create() method')

    def save(self, must_create=False):
        """
        Save the session data. If 'must_create' is True, create a new session
        object (or raise CreateError). Otherwise, only update an existing
        object and don't create one (raise UpdateError if needed).
        """
        raise NotImplementedError('subclasses of SessionBase must provide a save() method')

    def delete(self, session_key=None):
        """
        Delete the session data under this key. If the key is None, use the
        current session key value.
        """
        raise NotImplementedError('subclasses of SessionBase must provide a delete() method')

    def load(self):
        """
        Load the session data and return a dictionary.
        """
        raise NotImplementedError('subclasses of SessionBase must provide a load() method')

    @classmethod
    def clear_expired(cls):
        """
        Remove expired sessions from the session store.

        If this operation isn't possible on a given backend, it should raise
        NotImplementedError. If it isn't necessary, because the backend has
        a built-in expiration mechanism, it should be a no-op.
        """
        raise NotImplementedError('This backend does not support clear_expired().')
```
### 18 - django/contrib/sessions/backends/base.py:

Start line: 39, End line: 131

```python
class SessionBase:
    """
    Base class for all Session classes.
    """
    TEST_COOKIE_NAME = 'testcookie'
    TEST_COOKIE_VALUE = 'worked'

    __not_given = object()

    def __init__(self, session_key=None):
        self._session_key = session_key
        self.accessed = False
        self.modified = False
        self.serializer = import_string(settings.SESSION_SERIALIZER)

    def __contains__(self, key):
        return key in self._session

    def __getitem__(self, key):
        if key == LANGUAGE_SESSION_KEY:
            warnings.warn(
                'The user language will no longer be stored in '
                'request.session in Django 4.0. Read it from '
                'request.COOKIES[settings.LANGUAGE_COOKIE_NAME] instead.',
                RemovedInDjango40Warning, stacklevel=2,
            )
        return self._session[key]

    def __setitem__(self, key, value):
        self._session[key] = value
        self.modified = True

    def __delitem__(self, key):
        del self._session[key]
        self.modified = True

    @property
    def key_salt(self):
        return 'django.contrib.sessions.' + self.__class__.__qualname__

    def get(self, key, default=None):
        return self._session.get(key, default)

    def pop(self, key, default=__not_given):
        self.modified = self.modified or key in self._session
        args = () if default is self.__not_given else (default,)
        return self._session.pop(key, *args)

    def setdefault(self, key, value):
        if key in self._session:
            return self._session[key]
        else:
            self.modified = True
            self._session[key] = value
            return value

    def set_test_cookie(self):
        self[self.TEST_COOKIE_NAME] = self.TEST_COOKIE_VALUE

    def test_cookie_worked(self):
        return self.get(self.TEST_COOKIE_NAME) == self.TEST_COOKIE_VALUE

    def delete_test_cookie(self):
        del self[self.TEST_COOKIE_NAME]

    def _hash(self, value):
        # RemovedInDjango40Warning: pre-Django 3.1 format will be invalid.
        key_salt = "django.contrib.sessions" + self.__class__.__name__
        return salted_hmac(key_salt, value).hexdigest()

    def encode(self, session_dict):
        "Return the given session dictionary serialized and encoded as a string."
        # RemovedInDjango40Warning: DEFAULT_HASHING_ALGORITHM will be removed.
        if settings.DEFAULT_HASHING_ALGORITHM == 'sha1':
            return self._legacy_encode(session_dict)
        return signing.dumps(
            session_dict, salt=self.key_salt, serializer=self.serializer,
            compress=True,
        )

    def decode(self, session_data):
        try:
            return signing.loads(session_data, salt=self.key_salt, serializer=self.serializer)
        # RemovedInDjango40Warning: when the deprecation ends, handle here
        # exceptions similar to what _legacy_decode() does now.
        except Exception:
            return self._legacy_decode(session_data)

    def _legacy_encode(self, session_dict):
        # RemovedInDjango40Warning.
        serialized = self.serializer().dumps(session_dict)
        hash = self._hash(serialized)
        return base64.b64encode(hash.encode() + b':' + serialized).decode('ascii')
```
