# django__django-11592

| **django/django** | `806ba19bbff311b7d567857ae61db6ff84af4a2c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2686 |
| **Any found context length** | 2686 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/handlers/wsgi.py b/django/core/handlers/wsgi.py
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -141,7 +141,7 @@ def __call__(self, environ, start_response):
         ]
         start_response(status, response_headers)
         if getattr(response, 'file_to_stream', None) is not None and environ.get('wsgi.file_wrapper'):
-            response = environ['wsgi.file_wrapper'](response.file_to_stream)
+            response = environ['wsgi.file_wrapper'](response.file_to_stream, response.block_size)
         return response
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/handlers/wsgi.py | 144 | 144 | 11 | 2 | 2686


## Problem Statement

```
Start passing FileResponse.block_size to wsgi.file_wrapper.
Description
	 
		(last modified by Chris Jerdonek)
	 
I noticed that Django's FileResponse class has a block_size attribute which can be customized by subclassing: ​https://github.com/django/django/blob/415e899dc46c2f8d667ff11d3e54eff759eaded4/django/http/response.py#L393
but it's not passed to wsgi.file_wrapper. Only the filelike object is passed:
response = environ['wsgi.file_wrapper'](response.file_to_stream)
(from: ​https://github.com/django/django/blob/415e899dc46c2f8d667ff11d3e54eff759eaded4/django/core/handlers/wsgi.py#L144 )

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/http/response.py | 425 | 438| 142 | 142 | 4500 | 
| 2 | 1 django/http/response.py | 392 | 423| 201 | 343 | 4500 | 
| 3 | 1 django/http/response.py | 440 | 477| 365 | 708 | 4500 | 
| 4 | 1 django/http/response.py | 239 | 278| 319 | 1027 | 4500 | 
| 5 | **2 django/core/handlers/wsgi.py** | 64 | 119| 486 | 1513 | 6183 | 
| 6 | **2 django/core/handlers/wsgi.py** | 44 | 61| 139 | 1652 | 6183 | 
| 7 | **2 django/core/handlers/wsgi.py** | 1 | 42| 298 | 1950 | 6183 | 
| 8 | 3 django/core/files/base.py | 31 | 46| 129 | 2079 | 7235 | 
| 9 | 3 django/core/files/base.py | 48 | 73| 177 | 2256 | 7235 | 
| 10 | 4 django/core/servers/basehttp.py | 159 | 178| 170 | 2426 | 8980 | 
| **-> 11 <-** | **4 django/core/handlers/wsgi.py** | 122 | 152| 260 | 2686 | 8980 | 
| 12 | 5 django/core/handlers/asgi.py | 274 | 298| 169 | 2855 | 11345 | 
| 13 | 6 django/middleware/security.py | 30 | 47| 164 | 3019 | 11763 | 
| 14 | 7 django/contrib/staticfiles/handlers.py | 56 | 69| 127 | 3146 | 12429 | 
| 15 | 8 django/core/files/uploadhandler.py | 107 | 128| 165 | 3311 | 13760 | 
| 16 | 8 django/http/response.py | 312 | 344| 203 | 3514 | 13760 | 
| 17 | 8 django/core/files/base.py | 1 | 29| 174 | 3688 | 13760 | 
| 18 | 8 django/core/servers/basehttp.py | 122 | 157| 280 | 3968 | 13760 | 
| 19 | 8 django/http/response.py | 221 | 237| 181 | 4149 | 13760 | 
| 20 | 8 django/core/files/uploadhandler.py | 61 | 91| 199 | 4348 | 13760 | 
| 21 | 9 django/http/request.py | 214 | 286| 528 | 4876 | 18590 | 
| 22 | 10 django/template/defaultfilters.py | 804 | 848| 378 | 5254 | 24655 | 
| 23 | 10 django/core/handlers/asgi.py | 227 | 272| 387 | 5641 | 24655 | 
| 24 | 11 django/core/files/utils.py | 1 | 53| 378 | 6019 | 25033 | 
| 25 | 11 django/core/files/base.py | 75 | 118| 303 | 6322 | 25033 | 
| 26 | 12 django/core/wsgi.py | 1 | 14| 0 | 6322 | 25123 | 
| 27 | 12 django/http/request.py | 288 | 308| 189 | 6511 | 25123 | 
| 28 | 13 django/forms/fields.py | 547 | 566| 171 | 6682 | 34067 | 
| 29 | 14 django/core/files/uploadedfile.py | 38 | 52| 134 | 6816 | 34918 | 
| 30 | 15 django/db/models/fields/files.py | 212 | 324| 867 | 7683 | 38639 | 
| 31 | 15 django/contrib/staticfiles/handlers.py | 1 | 53| 368 | 8051 | 38639 | 
| 32 | 15 django/core/files/uploadhandler.py | 151 | 206| 382 | 8433 | 38639 | 
| 33 | 15 django/core/files/uploadhandler.py | 131 | 148| 138 | 8571 | 38639 | 
| 34 | 16 django/conf/global_settings.py | 264 | 346| 800 | 9371 | 44243 | 
| 35 | **16 django/core/handlers/wsgi.py** | 186 | 207| 176 | 9547 | 44243 | 
| 36 | 17 django/middleware/common.py | 99 | 115| 166 | 9713 | 45754 | 
| 37 | 17 django/conf/global_settings.py | 398 | 497| 793 | 10506 | 45754 | 
| 38 | 17 django/http/request.py | 339 | 370| 254 | 10760 | 45754 | 
| 39 | 17 django/db/models/fields/files.py | 150 | 209| 645 | 11405 | 45754 | 
| 40 | 18 django/contrib/gis/geos/prototypes/io.py | 254 | 285| 271 | 11676 | 48705 | 
| 41 | 19 django/core/handlers/exception.py | 1 | 38| 256 | 11932 | 49646 | 
| 42 | 19 django/core/files/uploadedfile.py | 78 | 96| 149 | 12081 | 49646 | 
| 43 | 20 django/utils/cache.py | 116 | 131| 188 | 12269 | 53198 | 
| 44 | 20 django/http/response.py | 347 | 389| 309 | 12578 | 53198 | 
| 45 | 21 django/middleware/csrf.py | 181 | 203| 230 | 12808 | 56068 | 
| 46 | 21 django/core/handlers/exception.py | 41 | 102| 499 | 13307 | 56068 | 
| 47 | 21 django/http/response.py | 28 | 105| 614 | 13921 | 56068 | 
| 48 | 21 django/http/response.py | 535 | 560| 159 | 14080 | 56068 | 
| 49 | 22 django/http/multipartparser.py | 339 | 363| 169 | 14249 | 61095 | 
| 50 | 22 django/forms/fields.py | 529 | 545| 180 | 14429 | 61095 | 
| 51 | 23 django/views/decorators/clickjacking.py | 22 | 54| 238 | 14667 | 61471 | 
| 52 | 23 django/http/request.py | 1 | 37| 251 | 14918 | 61471 | 
| 53 | 24 django/middleware/clickjacking.py | 1 | 46| 361 | 15279 | 61832 | 
| 54 | 24 django/core/files/base.py | 121 | 145| 154 | 15433 | 61832 | 
| 55 | 25 django/middleware/gzip.py | 1 | 53| 405 | 15838 | 62238 | 
| 56 | 25 django/utils/cache.py | 86 | 113| 178 | 16016 | 62238 | 
| 57 | 26 django/contrib/staticfiles/storage.py | 257 | 327| 574 | 16590 | 66154 | 
| 58 | 27 django/core/files/storage.py | 226 | 287| 524 | 17114 | 68990 | 
| 59 | 28 django/http/__init__.py | 1 | 22| 197 | 17311 | 69187 | 
| 60 | 28 django/views/decorators/clickjacking.py | 1 | 19| 138 | 17449 | 69187 | 
| 61 | 28 django/http/response.py | 281 | 310| 212 | 17661 | 69187 | 
| 62 | 28 django/core/servers/basehttp.py | 76 | 95| 184 | 17845 | 69187 | 
| 63 | 28 django/core/servers/basehttp.py | 180 | 197| 165 | 18010 | 69187 | 
| 64 | 28 django/core/handlers/asgi.py | 24 | 125| 872 | 18882 | 69187 | 
| 65 | 28 django/http/response.py | 501 | 532| 153 | 19035 | 69187 | 
| 66 | 28 django/contrib/staticfiles/storage.py | 49 | 82| 275 | 19310 | 69187 | 
| 67 | 28 django/core/handlers/asgi.py | 198 | 225| 237 | 19547 | 69187 | 
| 68 | 28 django/db/models/fields/files.py | 1 | 130| 905 | 20452 | 69187 | 
| 69 | 28 django/core/handlers/asgi.py | 128 | 174| 408 | 20860 | 69187 | 
| 70 | 28 django/db/models/fields/files.py | 133 | 148| 112 | 20972 | 69187 | 
| 71 | 28 django/core/servers/basehttp.py | 200 | 217| 210 | 21182 | 69187 | 
| 72 | 28 django/core/handlers/asgi.py | 1 | 21| 116 | 21298 | 69187 | 
| 73 | 29 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 21445 | 69335 | 
| 74 | 29 django/contrib/staticfiles/handlers.py | 72 | 89| 171 | 21616 | 69335 | 
| 75 | 29 django/http/multipartparser.py | 428 | 466| 222 | 21838 | 69335 | 
| 76 | 29 django/conf/global_settings.py | 145 | 263| 876 | 22714 | 69335 | 
| 77 | 30 django/core/cache/backends/filebased.py | 46 | 59| 145 | 22859 | 70501 | 
| 78 | **30 django/core/handlers/wsgi.py** | 155 | 183| 326 | 23185 | 70501 | 
| 79 | 31 django/contrib/sessions/backends/file.py | 111 | 170| 530 | 23715 | 72002 | 
| 80 | 32 django/contrib/sites/middleware.py | 1 | 13| 0 | 23715 | 72061 | 
| 81 | 32 django/http/response.py | 480 | 498| 186 | 23901 | 72061 | 
| 82 | 32 django/middleware/common.py | 76 | 97| 227 | 24128 | 72061 | 
| 83 | 32 django/core/files/storage.py | 198 | 224| 215 | 24343 | 72061 | 
| 84 | 32 django/http/multipartparser.py | 1 | 40| 195 | 24538 | 72061 | 
| 85 | 32 django/http/response.py | 134 | 155| 176 | 24714 | 72061 | 
| 86 | 32 django/middleware/csrf.py | 205 | 327| 1189 | 25903 | 72061 | 
| 87 | 32 django/http/multipartparser.py | 499 | 538| 245 | 26148 | 72061 | 
| 88 | 32 django/core/files/storage.py | 25 | 61| 270 | 26418 | 72061 | 
| 89 | 32 django/core/handlers/asgi.py | 176 | 196| 207 | 26625 | 72061 | 
| 90 | 32 django/http/multipartparser.py | 365 | 404| 258 | 26883 | 72061 | 
| 91 | 33 django/core/mail/message.py | 310 | 324| 127 | 27010 | 75600 | 
| 92 | 34 django/contrib/sessions/middleware.py | 1 | 75| 576 | 27586 | 76177 | 
| 93 | 34 django/core/files/uploadhandler.py | 27 | 58| 205 | 27791 | 76177 | 
| 94 | 34 django/http/request.py | 138 | 145| 111 | 27902 | 76177 | 
| 95 | 35 django/utils/text.py | 276 | 317| 287 | 28189 | 79607 | 
| 96 | 35 django/db/models/fields/files.py | 405 | 467| 551 | 28740 | 79607 | 
| 97 | 35 django/http/request.py | 310 | 337| 279 | 29019 | 79607 | 
| 98 | 35 django/conf/global_settings.py | 499 | 638| 853 | 29872 | 79607 | 
| 99 | 35 django/core/files/uploadhandler.py | 1 | 24| 126 | 29998 | 79607 | 
| 100 | 35 django/db/models/fields/files.py | 357 | 403| 373 | 30371 | 79607 | 
| 101 | 35 django/http/request.py | 373 | 396| 185 | 30556 | 79607 | 
| 102 | 36 django/views/defaults.py | 100 | 119| 149 | 30705 | 80649 | 
| 103 | 37 django/core/files/images.py | 1 | 30| 147 | 30852 | 81188 | 
| 104 | 37 django/core/servers/basehttp.py | 1 | 23| 164 | 31016 | 81188 | 
| 105 | 38 django/contrib/gis/geos/mutable_list.py | 283 | 293| 131 | 31147 | 83442 | 
| 106 | 38 django/core/files/storage.py | 169 | 184| 151 | 31298 | 83442 | 
| 107 | 39 django/views/generic/base.py | 1 | 27| 148 | 31446 | 85045 | 
| 108 | 39 django/core/files/uploadhandler.py | 93 | 105| 120 | 31566 | 85045 | 
| 109 | 39 django/utils/cache.py | 134 | 171| 447 | 32013 | 85045 | 
| 110 | 40 django/core/files/locks.py | 19 | 114| 773 | 32786 | 85996 | 
| 111 | 41 django/middleware/locale.py | 28 | 62| 331 | 33117 | 86563 | 
| 112 | 41 django/contrib/staticfiles/storage.py | 347 | 368| 230 | 33347 | 86563 | 
| 113 | 41 django/core/files/storage.py | 289 | 361| 483 | 33830 | 86563 | 
| 114 | 41 django/middleware/security.py | 1 | 28| 260 | 34090 | 86563 | 
| 115 | 41 django/contrib/sessions/backends/file.py | 75 | 109| 253 | 34343 | 86563 | 
| 116 | 42 django/contrib/auth/middleware.py | 27 | 45| 178 | 34521 | 87579 | 
| 117 | 42 django/middleware/common.py | 34 | 61| 257 | 34778 | 87579 | 
| 118 | 42 django/contrib/staticfiles/storage.py | 84 | 116| 346 | 35124 | 87579 | 
| 119 | 42 django/core/files/uploadedfile.py | 55 | 75| 181 | 35305 | 87579 | 
| 120 | 42 django/utils/cache.py | 256 | 273| 206 | 35511 | 87579 | 
| 121 | 42 django/core/cache/backends/filebased.py | 1 | 44| 284 | 35795 | 87579 | 
| 122 | 42 django/middleware/csrf.py | 158 | 179| 189 | 35984 | 87579 | 
| 123 | 42 django/core/servers/basehttp.py | 53 | 73| 170 | 36154 | 87579 | 
| 124 | 43 django/core/handlers/base.py | 64 | 83| 168 | 36322 | 88757 | 
| 125 | 43 django/http/response.py | 203 | 219| 193 | 36515 | 88757 | 
| 126 | 43 django/middleware/common.py | 1 | 32| 247 | 36762 | 88757 | 
| 127 | 43 django/utils/cache.py | 222 | 253| 256 | 37018 | 88757 | 
| 128 | 43 django/http/response.py | 157 | 201| 447 | 37465 | 88757 | 
| 129 | 43 django/contrib/staticfiles/storage.py | 155 | 207| 425 | 37890 | 88757 | 
| 130 | 44 django/core/mail/backends/filebased.py | 1 | 68| 547 | 38437 | 89305 | 
| 131 | 44 django/contrib/gis/geos/mutable_list.py | 261 | 281| 188 | 38625 | 89305 | 
| 132 | 45 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 38625 | 89398 | 
| 133 | 45 django/http/request.py | 40 | 83| 329 | 38954 | 89398 | 
| 134 | 46 django/core/files/__init__.py | 1 | 4| 0 | 38954 | 89413 | 
| 135 | 46 django/middleware/common.py | 118 | 147| 277 | 39231 | 89413 | 
| 136 | 47 django/template/response.py | 60 | 145| 587 | 39818 | 90492 | 
| 137 | 48 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 39818 | 90570 | 
| 138 | 49 django/core/validators.py | 467 | 502| 249 | 40067 | 94887 | 
| 139 | 49 django/contrib/sessions/backends/file.py | 57 | 73| 143 | 40210 | 94887 | 
| 140 | 49 django/core/files/uploadedfile.py | 99 | 118| 148 | 40358 | 94887 | 
| 141 | 50 django/core/files/temp.py | 1 | 75| 517 | 40875 | 95405 | 
| 142 | 51 django/core/management/commands/runserver.py | 54 | 64| 120 | 40995 | 96856 | 
| 143 | 51 django/contrib/auth/middleware.py | 47 | 83| 360 | 41355 | 96856 | 
| 144 | 51 django/http/response.py | 563 | 589| 264 | 41619 | 96856 | 
| 145 | 51 django/core/servers/basehttp.py | 97 | 119| 211 | 41830 | 96856 | 
| 146 | 52 django/middleware/cache.py | 74 | 110| 340 | 42170 | 98432 | 
| 147 | 52 django/db/models/fields/files.py | 327 | 354| 276 | 42446 | 98432 | 
| 148 | 52 django/http/response.py | 1 | 25| 144 | 42590 | 98432 | 
| 149 | 52 django/template/response.py | 45 | 58| 120 | 42710 | 98432 | 
| 150 | 53 django/core/asgi.py | 1 | 14| 0 | 42710 | 98517 | 
| 151 | 53 django/http/request.py | 124 | 136| 133 | 42843 | 98517 | 
| 152 | 54 django/utils/log.py | 1 | 76| 492 | 43335 | 100125 | 
| 153 | 55 django/db/backends/base/base.py | 528 | 559| 227 | 43562 | 104983 | 
| 154 | 56 django/utils/itercompat.py | 1 | 9| 0 | 43562 | 105023 | 
| 155 | 56 django/middleware/cache.py | 55 | 72| 168 | 43730 | 105023 | 
| 156 | 57 django/db/models/fields/__init__.py | 2256 | 2317| 425 | 44155 | 122040 | 
| 157 | 57 django/contrib/staticfiles/storage.py | 21 | 46| 213 | 44368 | 122040 | 
| 158 | 57 django/core/cache/backends/filebased.py | 61 | 95| 253 | 44621 | 122040 | 
| 159 | 58 django/contrib/messages/storage/cookie.py | 78 | 92| 127 | 44748 | 123372 | 
| 160 | 58 django/contrib/sessions/backends/file.py | 41 | 55| 131 | 44879 | 123372 | 
| 161 | 59 django/core/checks/security/csrf.py | 1 | 41| 299 | 45178 | 123671 | 
| 162 | 59 django/core/files/storage.py | 1 | 22| 158 | 45336 | 123671 | 
| 163 | 60 django/views/decorators/gzip.py | 1 | 6| 0 | 45336 | 123722 | 
| 164 | 60 django/contrib/staticfiles/storage.py | 462 | 473| 110 | 45446 | 123722 | 
| 165 | 60 django/utils/log.py | 199 | 231| 259 | 45705 | 123722 | 
| 166 | 60 django/http/multipartparser.py | 406 | 425| 205 | 45910 | 123722 | 
| 167 | 61 django/core/management/base.py | 114 | 145| 220 | 46130 | 128109 | 
| 168 | 61 django/utils/cache.py | 37 | 83| 428 | 46558 | 128109 | 
| 169 | 61 django/contrib/gis/geos/prototypes/io.py | 212 | 232| 196 | 46754 | 128109 | 
| 170 | 61 django/http/multipartparser.py | 289 | 310| 213 | 46967 | 128109 | 
| 171 | 61 django/contrib/gis/geos/mutable_list.py | 295 | 311| 133 | 47100 | 128109 | 
| 172 | 61 django/template/response.py | 1 | 43| 383 | 47483 | 128109 | 
| 173 | 61 django/contrib/auth/middleware.py | 1 | 24| 193 | 47676 | 128109 | 
| 174 | 62 django/utils/deprecation.py | 76 | 98| 158 | 47834 | 128801 | 
| 175 | 62 django/core/files/images.py | 33 | 85| 391 | 48225 | 128801 | 
| 176 | 63 django/template/backends/utils.py | 1 | 15| 0 | 48225 | 128890 | 
| 177 | 64 django/contrib/sessions/serializers.py | 1 | 21| 0 | 48225 | 128977 | 
| 178 | 65 django/core/cache/utils.py | 1 | 13| 0 | 48225 | 129064 | 
| 179 | 65 django/http/multipartparser.py | 469 | 497| 220 | 48445 | 129064 | 
| 180 | 66 django/middleware/http.py | 1 | 42| 335 | 48780 | 129399 | 
| 181 | 67 django/views/csrf.py | 15 | 100| 835 | 49615 | 130943 | 
| 182 | 68 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 49615 | 131021 | 
| 183 | 69 django/core/management/templates.py | 243 | 295| 403 | 50018 | 133689 | 
| 184 | 69 django/core/handlers/base.py | 85 | 167| 580 | 50598 | 133689 | 
| 185 | 70 django/views/debug.py | 72 | 95| 196 | 50794 | 137907 | 
| 186 | 71 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 50794 | 138004 | 
| 187 | 71 django/http/multipartparser.py | 153 | 287| 1112 | 51906 | 138004 | 
| 188 | 71 django/views/debug.py | 154 | 177| 176 | 52082 | 138004 | 
| 189 | 71 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 52211 | 138004 | 
| 190 | 71 django/contrib/sessions/backends/file.py | 1 | 39| 261 | 52472 | 138004 | 
| 191 | 71 django/conf/global_settings.py | 347 | 397| 826 | 53298 | 138004 | 
| 192 | 71 django/forms/fields.py | 568 | 593| 243 | 53541 | 138004 | 
| 193 | 71 django/utils/log.py | 162 | 196| 290 | 53831 | 138004 | 
| 194 | 71 django/core/files/storage.py | 63 | 92| 343 | 54174 | 138004 | 
| 195 | 71 django/http/multipartparser.py | 574 | 642| 494 | 54668 | 138004 | 
| 196 | 71 django/core/handlers/base.py | 1 | 62| 436 | 55104 | 138004 | 
| 197 | 72 django/utils/http.py | 398 | 460| 318 | 55422 | 141960 | 
| 198 | 72 django/views/defaults.py | 1 | 24| 149 | 55571 | 141960 | 
| 199 | 73 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 55571 | 142028 | 
| 200 | 74 django/template/loader_tags.py | 191 | 217| 285 | 55856 | 144579 | 
| 201 | 75 django/contrib/staticfiles/__init__.py | 1 | 2| 0 | 55856 | 144593 | 
| 202 | 75 django/views/debug.py | 193 | 241| 462 | 56318 | 144593 | 
| 203 | 76 django/core/management/commands/makemessages.py | 36 | 57| 143 | 56461 | 150153 | 
| 204 | 77 django/contrib/admin/options.py | 1996 | 2017| 221 | 56682 | 168519 | 
| 205 | 78 django/forms/widgets.py | 374 | 391| 116 | 56798 | 176585 | 
| 206 | 79 django/conf/__init__.py | 188 | 237| 353 | 57151 | 178371 | 
| 207 | 79 django/contrib/gis/geos/prototypes/io.py | 234 | 243| 147 | 57298 | 178371 | 
| 208 | 80 django/contrib/sessions/exceptions.py | 1 | 12| 0 | 57298 | 178422 | 
| 209 | 81 django/contrib/gis/gdal/raster/band.py | 180 | 233| 481 | 57779 | 180235 | 
| 210 | 81 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 57989 | 180235 | 
| 211 | 82 django/contrib/gis/shortcuts.py | 1 | 41| 248 | 58237 | 180483 | 
| 212 | 83 django/views/static.py | 108 | 136| 206 | 58443 | 181535 | 
| 213 | 83 django/views/generic/base.py | 117 | 150| 241 | 58684 | 181535 | 
| 214 | 83 django/core/cache/backends/filebased.py | 97 | 113| 174 | 58858 | 181535 | 


### Hint

```
Seems reasonable, Thanks (see ​example-of-wrapper-usage).
```

## Patch

```diff
diff --git a/django/core/handlers/wsgi.py b/django/core/handlers/wsgi.py
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -141,7 +141,7 @@ def __call__(self, environ, start_response):
         ]
         start_response(status, response_headers)
         if getattr(response, 'file_to_stream', None) is not None and environ.get('wsgi.file_wrapper'):
-            response = environ['wsgi.file_wrapper'](response.file_to_stream)
+            response = environ['wsgi.file_wrapper'](response.file_to_stream, response.block_size)
         return response
 
 

```

## Test Patch

```diff
diff --git a/tests/wsgi/tests.py b/tests/wsgi/tests.py
--- a/tests/wsgi/tests.py
+++ b/tests/wsgi/tests.py
@@ -3,6 +3,7 @@
 from django.core.signals import request_started
 from django.core.wsgi import get_wsgi_application
 from django.db import close_old_connections
+from django.http import FileResponse
 from django.test import SimpleTestCase, override_settings
 from django.test.client import RequestFactory
 
@@ -51,7 +52,8 @@ def test_file_wrapper(self):
         FileResponse uses wsgi.file_wrapper.
         """
         class FileWrapper:
-            def __init__(self, filelike, blksize=8192):
+            def __init__(self, filelike, block_size=None):
+                self.block_size = block_size
                 filelike.close()
         application = get_wsgi_application()
         environ = self.request_factory._base_environ(
@@ -67,6 +69,7 @@ def start_response(status, headers):
         response = application(environ, start_response)
         self.assertEqual(response_data['status'], '200 OK')
         self.assertIsInstance(response, FileWrapper)
+        self.assertEqual(response.block_size, FileResponse.block_size)
 
 
 class GetInternalWSGIApplicationTest(SimpleTestCase):

```


## Code snippets

### 1 - django/http/response.py:

Start line: 425, End line: 438

```python
class FileResponse(StreamingHttpResponse):

    def _set_streaming_content(self, value):
        if not hasattr(value, 'read'):
            self.file_to_stream = None
            return super()._set_streaming_content(value)

        self.file_to_stream = filelike = value
        # Add to closable objects before wrapping close(), since the filelike
        # might not have close().
        if hasattr(filelike, 'close'):
            self._closable_objects.append(filelike)
        self._wrap_file_to_stream_close(filelike)
        value = iter(lambda: filelike.read(self.block_size), b'')
        self.set_headers(filelike)
        super()._set_streaming_content(value)
```
### 2 - django/http/response.py:

Start line: 392, End line: 423

```python
class FileResponse(StreamingHttpResponse):
    """
    A streaming HTTP response class optimized for files.
    """
    block_size = 4096

    def __init__(self, *args, as_attachment=False, filename='', **kwargs):
        self.as_attachment = as_attachment
        self.filename = filename
        super().__init__(*args, **kwargs)

    def _wrap_file_to_stream_close(self, filelike):
        """
        Wrap the file-like close() with a version that calls
        FileResponse.close().
        """
        closing = False
        filelike_close = getattr(filelike, 'close', lambda: None)

        def file_wrapper_close():
            nonlocal closing
            # Prevent an infinite loop since FileResponse.close() tries to
            # close the objects in self._closable_objects.
            if closing:
                return
            closing = True
            try:
                filelike_close()
            finally:
                self.close()

        filelike.close = file_wrapper_close
```
### 3 - django/http/response.py:

Start line: 440, End line: 477

```python
class FileResponse(StreamingHttpResponse):

    def set_headers(self, filelike):
        """
        Set some common response headers (Content-Length, Content-Type, and
        Content-Disposition) based on the `filelike` response content.
        """
        encoding_map = {
            'bzip2': 'application/x-bzip',
            'gzip': 'application/gzip',
            'xz': 'application/x-xz',
        }
        filename = getattr(filelike, 'name', None)
        filename = filename if (isinstance(filename, str) and filename) else self.filename
        if os.path.isabs(filename):
            self['Content-Length'] = os.path.getsize(filelike.name)
        elif hasattr(filelike, 'getbuffer'):
            self['Content-Length'] = filelike.getbuffer().nbytes

        if self.get('Content-Type', '').startswith('text/html'):
            if filename:
                content_type, encoding = mimetypes.guess_type(filename)
                # Encoding isn't set to prevent browsers from automatically
                # uncompressing files.
                content_type = encoding_map.get(encoding, content_type)
                self['Content-Type'] = content_type or 'application/octet-stream'
            else:
                self['Content-Type'] = 'application/octet-stream'

        filename = self.filename or os.path.basename(filename)
        if filename:
            disposition = 'attachment' if self.as_attachment else 'inline'
            try:
                filename.encode('ascii')
                file_expr = 'filename="{}"'.format(filename)
            except UnicodeEncodeError:
                file_expr = "filename*=utf-8''{}".format(quote(filename))
            self['Content-Disposition'] = '{}; {}'.format(disposition, file_expr)
        elif self.as_attachment:
            self['Content-Disposition'] = 'attachment'
```
### 4 - django/http/response.py:

Start line: 239, End line: 278

```python
class HttpResponseBase:

    # These methods partially implement the file-like object interface.
    # See https://docs.python.org/library/io.html#io.IOBase

    # The WSGI server must call this method upon completion of the request.
    # See http://blog.dscpl.com.au/2012/10/obligations-for-calling-close-on.html
    # When wsgi.file_wrapper is used, the WSGI server instead calls close()
    # on the file-like object. Django ensures this method is called in this
    # case by replacing self.file_to_stream.close() with a wrapped version.
    def close(self):
        for closable in self._closable_objects:
            try:
                closable.close()
            except Exception:
                pass
        self.closed = True
        signals.request_finished.send(sender=self._handler_class)

    def write(self, content):
        raise OSError('This %s instance is not writable' % self.__class__.__name__)

    def flush(self):
        pass

    def tell(self):
        raise OSError('This %s instance cannot tell its position' % self.__class__.__name__)

    # These methods partially implement a stream-like object interface.
    # See https://docs.python.org/library/io.html#io.IOBase

    def readable(self):
        return False

    def seekable(self):
        return False

    def writable(self):
        return False

    def writelines(self, lines):
        raise OSError('This %s instance is not writable' % self.__class__.__name__)
```
### 5 - django/core/handlers/wsgi.py:

Start line: 64, End line: 119

```python
class WSGIRequest(HttpRequest):
    def __init__(self, environ):
        script_name = get_script_name(environ)
        # If PATH_INFO is empty (e.g. accessing the SCRIPT_NAME URL without a
        # trailing slash), operate as if '/' was requested.
        path_info = get_path_info(environ) or '/'
        self.environ = environ
        self.path_info = path_info
        # be careful to only replace the first slash in the path because of
        # http://test/something and http://test//something being different as
        # stated in https://www.ietf.org/rfc/rfc2396.txt
        self.path = '%s/%s' % (script_name.rstrip('/'),
                               path_info.replace('/', '', 1))
        self.META = environ
        self.META['PATH_INFO'] = path_info
        self.META['SCRIPT_NAME'] = script_name
        self.method = environ['REQUEST_METHOD'].upper()
        # Set content_type, content_params, and encoding.
        self._set_content_type_params(environ)
        try:
            content_length = int(environ.get('CONTENT_LENGTH'))
        except (ValueError, TypeError):
            content_length = 0
        self._stream = LimitedStream(self.environ['wsgi.input'], content_length)
        self._read_started = False
        self.resolver_match = None

    def _get_scheme(self):
        return self.environ.get('wsgi.url_scheme')

    @cached_property
    def GET(self):
        # The WSGI spec says 'QUERY_STRING' may be absent.
        raw_query_string = get_bytes_from_wsgi(self.environ, 'QUERY_STRING', '')
        return QueryDict(raw_query_string, encoding=self._encoding)

    def _get_post(self):
        if not hasattr(self, '_post'):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    @cached_property
    def COOKIES(self):
        raw_cookie = get_str_from_wsgi(self.environ, 'HTTP_COOKIE', '')
        return parse_cookie(raw_cookie)

    @property
    def FILES(self):
        if not hasattr(self, '_files'):
            self._load_post_and_files()
        return self._files

    POST = property(_get_post, _set_post)
```
### 6 - django/core/handlers/wsgi.py:

Start line: 44, End line: 61

```python
class LimitedStream:

    def readline(self, size=None):
        while b'\n' not in self.buffer and \
              (size is None or len(self.buffer) < size):
            if size:
                # since size is not None here, len(self.buffer) < size
                chunk = self._read_limited(size - len(self.buffer))
            else:
                chunk = self._read_limited()
            if not chunk:
                break
            self.buffer += chunk
        sio = BytesIO(self.buffer)
        if size:
            line = sio.readline(size)
        else:
            line = sio.readline()
        self.buffer = sio.read()
        return line
```
### 7 - django/core/handlers/wsgi.py:

Start line: 1, End line: 42

```python
import re
from io import BytesIO

from django.conf import settings
from django.core import signals
from django.core.handlers import base
from django.http import HttpRequest, QueryDict, parse_cookie
from django.urls import set_script_prefix
from django.utils.encoding import repercent_broken_unicode
from django.utils.functional import cached_property

_slashes_re = re.compile(br'/+')


class LimitedStream:
    """Wrap another stream to disallow reading it past a number of bytes."""
    def __init__(self, stream, limit, buf_size=64 * 1024 * 1024):
        self.stream = stream
        self.remaining = limit
        self.buffer = b''
        self.buf_size = buf_size

    def _read_limited(self, size=None):
        if size is None or size > self.remaining:
            size = self.remaining
        if size == 0:
            return b''
        result = self.stream.read(size)
        self.remaining -= len(result)
        return result

    def read(self, size=None):
        if size is None:
            result = self.buffer + self._read_limited()
            self.buffer = b''
        elif size < len(self.buffer):
            result = self.buffer[:size]
            self.buffer = self.buffer[size:]
        else:  # size >= len(self.buffer)
            result = self.buffer + self._read_limited(size - len(self.buffer))
            self.buffer = b''
        return result
```
### 8 - django/core/files/base.py:

Start line: 31, End line: 46

```python
class File(FileProxyMixin):

    @cached_property
    def size(self):
        if hasattr(self.file, 'size'):
            return self.file.size
        if hasattr(self.file, 'name'):
            try:
                return os.path.getsize(self.file.name)
            except (OSError, TypeError):
                pass
        if hasattr(self.file, 'tell') and hasattr(self.file, 'seek'):
            pos = self.file.tell()
            self.file.seek(0, os.SEEK_END)
            size = self.file.tell()
            self.file.seek(pos)
            return size
        raise AttributeError("Unable to determine the file's size.")
```
### 9 - django/core/files/base.py:

Start line: 48, End line: 73

```python
class File(FileProxyMixin):

    def chunks(self, chunk_size=None):
        """
        Read the file and yield chunks of ``chunk_size`` bytes (defaults to
        ``File.DEFAULT_CHUNK_SIZE``).
        """
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        try:
            self.seek(0)
        except (AttributeError, UnsupportedOperation):
            pass

        while True:
            data = self.read(chunk_size)
            if not data:
                break
            yield data

    def multiple_chunks(self, chunk_size=None):
        """
        Return ``True`` if you can expect multiple chunks.

        NB: If a particular file representation is in memory, subclasses should
        always return ``False`` -- there's no good reason to read from memory in
        chunks.
        """
        return self.size > (chunk_size or self.DEFAULT_CHUNK_SIZE)
```
### 10 - django/core/servers/basehttp.py:

Start line: 159, End line: 178

```python
class WSGIRequestHandler(simple_server.WSGIRequestHandler):

    def get_environ(self):
        # Strip all headers with underscores in the name before constructing
        # the WSGI environ. This prevents header-spoofing based on ambiguity
        # between underscores and dashes both normalized to underscores in WSGI
        # env vars. Nginx and Apache 2.4+ both do this as well.
        for k in self.headers:
            if '_' in k:
                del self.headers[k]

        return super().get_environ()

    def handle(self):
        self.close_connection = True
        self.handle_one_request()
        while not self.close_connection:
            self.handle_one_request()
        try:
            self.connection.shutdown(socket.SHUT_WR)
        except (AttributeError, OSError):
            pass
```
### 11 - django/core/handlers/wsgi.py:

Start line: 122, End line: 152

```python
class WSGIHandler(base.BaseHandler):
    request_class = WSGIRequest

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_middleware()

    def __call__(self, environ, start_response):
        set_script_prefix(get_script_name(environ))
        signals.request_started.send(sender=self.__class__, environ=environ)
        request = self.request_class(environ)
        response = self.get_response(request)

        response._handler_class = self.__class__

        status = '%d %s' % (response.status_code, response.reason_phrase)
        response_headers = [
            *response.items(),
            *(('Set-Cookie', c.output(header='')) for c in response.cookies.values()),
        ]
        start_response(status, response_headers)
        if getattr(response, 'file_to_stream', None) is not None and environ.get('wsgi.file_wrapper'):
            response = environ['wsgi.file_wrapper'](response.file_to_stream)
        return response


def get_path_info(environ):
    """Return the HTTP request's PATH_INFO as a string."""
    path_info = get_bytes_from_wsgi(environ, 'PATH_INFO', '/')

    return repercent_broken_unicode(path_info).decode()
```
### 35 - django/core/handlers/wsgi.py:

Start line: 186, End line: 207

```python
def get_bytes_from_wsgi(environ, key, default):
    """
    Get a value from the WSGI environ dictionary as bytes.

    key and default should be strings.
    """
    value = environ.get(key, default)
    # Non-ASCII values in the WSGI environ are arbitrarily decoded with
    # ISO-8859-1. This is wrong for Django websites where UTF-8 is the default.
    # Re-encode to recover the original bytestring.
    return value.encode('iso-8859-1')


def get_str_from_wsgi(environ, key, default):
    """
    Get a value from the WSGI environ dictionary as str.

    key and default should be str objects.
    """
    value = get_bytes_from_wsgi(environ, key, default)
    return value.decode(errors='replace')
```
### 78 - django/core/handlers/wsgi.py:

Start line: 155, End line: 183

```python
def get_script_name(environ):
    """
    Return the equivalent of the HTTP request's SCRIPT_NAME environment
    variable. If Apache mod_rewrite is used, return what would have been
    the script name prior to any rewriting (so it's the script name as seen
    from the client's perspective), unless the FORCE_SCRIPT_NAME setting is
    set (to anything).
    """
    if settings.FORCE_SCRIPT_NAME is not None:
        return settings.FORCE_SCRIPT_NAME

    # If Apache's mod_rewrite had a whack at the URL, Apache set either
    # SCRIPT_URL or REDIRECT_URL to the full resource URL before applying any
    # rewrites. Unfortunately not every Web server (lighttpd!) passes this
    # information through all the time, so FORCE_SCRIPT_NAME, above, is still
    # needed.
    script_url = get_bytes_from_wsgi(environ, 'SCRIPT_URL', '') or get_bytes_from_wsgi(environ, 'REDIRECT_URL', '')

    if script_url:
        if b'//' in script_url:
            # mod_wsgi squashes multiple successive slashes in PATH_INFO,
            # do the same with script_url before manipulating paths (#17133).
            script_url = _slashes_re.sub(b'/', script_url)
        path_info = get_bytes_from_wsgi(environ, 'PATH_INFO', '')
        script_name = script_url[:-len(path_info)] if path_info else script_url
    else:
        script_name = get_bytes_from_wsgi(environ, 'SCRIPT_NAME', '')

    return script_name.decode()
```
