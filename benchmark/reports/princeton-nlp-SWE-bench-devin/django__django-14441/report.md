# django__django-14441

| **django/django** | `c1d50b901b50672a46e7e5fe473c14da1616fc4e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 391 |
| **Any found context length** | 391 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/files/images.py b/django/core/files/images.py
--- a/django/core/files/images.py
+++ b/django/core/files/images.py
@@ -44,7 +44,10 @@ def get_image_dimensions(file_or_path, close=False):
         file_pos = file.tell()
         file.seek(0)
     else:
-        file = open(file_or_path, 'rb')
+        try:
+            file = open(file_or_path, 'rb')
+        except OSError:
+            return (None, None)
         close = True
     try:
         # Most of the time Pillow only needs a small chunk to parse the image

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/files/images.py | 47 | 47 | 1 | 1 | 391


## Problem Statement

```
Prevent get_image_dimensions() crash on nonexistent images.
Description
	
When using the get_image_dimensions(), If a non existing file/path is passed, the function crashes

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/files/images.py** | 33 | 85| 391 | 391 | 539 | 
| 2 | 2 django/db/models/fields/files.py | 372 | 418| 373 | 764 | 4372 | 
| 3 | **2 django/core/files/images.py** | 1 | 30| 147 | 911 | 4372 | 
| 4 | 2 django/db/models/fields/files.py | 420 | 482| 551 | 1462 | 4372 | 
| 5 | 2 django/db/models/fields/files.py | 342 | 369| 277 | 1739 | 4372 | 
| 6 | 3 django/forms/fields.py | 612 | 668| 438 | 2177 | 13793 | 
| 7 | 4 django/contrib/gis/gdal/prototypes/raster.py | 53 | 102| 804 | 2981 | 15342 | 
| 8 | 5 django/core/validators.py | 542 | 578| 225 | 3206 | 20041 | 
| 9 | 6 django/contrib/gis/geos/prototypes/errcheck.py | 54 | 84| 241 | 3447 | 20656 | 
| 10 | 7 django/contrib/gis/geos/error.py | 1 | 4| 0 | 3447 | 20680 | 
| 11 | 8 django/contrib/gis/gdal/error.py | 1 | 62| 457 | 3904 | 21137 | 
| 12 | 9 django/contrib/gis/geos/prototypes/io.py | 67 | 75| 111 | 4015 | 24088 | 
| 13 | 10 django/core/checks/files.py | 1 | 20| 0 | 4015 | 24192 | 
| 14 | 11 django/core/files/storage.py | 106 | 115| 116 | 4131 | 27144 | 
| 15 | 11 django/core/files/storage.py | 240 | 301| 534 | 4665 | 27144 | 
| 16 | 12 django/contrib/gis/gdal/base.py | 1 | 7| 0 | 4665 | 27186 | 
| 17 | 13 django/core/files/uploadhandler.py | 27 | 58| 204 | 4869 | 28602 | 
| 18 | 14 django/contrib/gis/gdal/prototypes/errcheck.py | 66 | 139| 536 | 5405 | 29585 | 
| 19 | 15 django/contrib/gis/gdal/libgdal.py | 93 | 124| 245 | 5650 | 30531 | 
| 20 | 16 django/contrib/gis/gdal/prototypes/generation.py | 136 | 170| 237 | 5887 | 31705 | 
| 21 | 17 django/core/files/utils.py | 1 | 24| 192 | 6079 | 32276 | 
| 22 | 18 django/core/cache/utils.py | 1 | 13| 0 | 6079 | 32355 | 
| 23 | 19 django/contrib/gis/gdal/prototypes/ds.py | 51 | 85| 637 | 6716 | 33743 | 
| 24 | 20 django/core/checks/caches.py | 59 | 73| 109 | 6825 | 34264 | 
| 25 | 20 django/contrib/gis/gdal/libgdal.py | 1 | 90| 701 | 7526 | 34264 | 
| 26 | 21 django/core/files/__init__.py | 1 | 4| 0 | 7526 | 34279 | 
| 27 | 22 django/contrib/gis/db/backends/oracle/features.py | 1 | 15| 0 | 7526 | 34391 | 
| 28 | 22 django/contrib/gis/geos/prototypes/errcheck.py | 1 | 51| 373 | 7899 | 34391 | 
| 29 | 23 django/contrib/gis/db/backends/postgis/features.py | 1 | 14| 0 | 7899 | 34487 | 
| 30 | 23 django/contrib/gis/gdal/prototypes/raster.py | 1 | 51| 745 | 8644 | 34487 | 
| 31 | 24 django/contrib/gis/geos/base.py | 1 | 7| 0 | 8644 | 34529 | 
| 32 | 25 django/contrib/gis/geos/libgeos.py | 21 | 70| 513 | 9157 | 35784 | 
| 33 | 26 django/contrib/postgres/functions.py | 1 | 12| 0 | 9157 | 35837 | 
| 34 | 27 django/contrib/gis/geos/geometry.py | 251 | 342| 763 | 9920 | 41752 | 
| 35 | 28 django/core/files/base.py | 31 | 46| 129 | 10049 | 42804 | 
| 36 | 29 django/contrib/gis/gdal/raster/const.py | 48 | 80| 468 | 10517 | 43806 | 
| 37 | 30 django/contrib/admin/utils.py | 289 | 307| 175 | 10692 | 47964 | 
| 38 | 31 django/contrib/staticfiles/checks.py | 1 | 15| 0 | 10692 | 48040 | 
| 39 | 31 django/contrib/gis/gdal/prototypes/errcheck.py | 38 | 64| 224 | 10916 | 48040 | 
| 40 | 31 django/core/checks/caches.py | 22 | 56| 291 | 11207 | 48040 | 
| 41 | 32 django/utils/itercompat.py | 1 | 9| 0 | 11207 | 48080 | 
| 42 | 33 django/contrib/gis/views.py | 1 | 21| 155 | 11362 | 48235 | 
| 43 | 33 django/contrib/gis/gdal/prototypes/errcheck.py | 1 | 35| 221 | 11583 | 48235 | 
| 44 | 34 django/contrib/gis/db/models/functions.py | 443 | 459| 180 | 11763 | 52180 | 
| 45 | 34 django/core/files/storage.py | 183 | 198| 151 | 11914 | 52180 | 
| 46 | 35 django/contrib/gis/db/backends/oracle/base.py | 1 | 17| 0 | 11914 | 52283 | 
| 47 | 36 django/contrib/gis/db/backends/mysql/base.py | 1 | 17| 0 | 11914 | 52392 | 
| 48 | 36 django/contrib/gis/geos/libgeos.py | 73 | 131| 345 | 12259 | 52392 | 
| 49 | 37 django/views/decorators/gzip.py | 1 | 6| 0 | 12259 | 52443 | 
| 50 | 38 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 12259 | 52536 | 
| 51 | 39 django/contrib/gis/gdal/raster/source.py | 218 | 234| 134 | 12393 | 56566 | 
| 52 | 40 django/core/checks/security/csrf.py | 45 | 68| 157 | 12550 | 57028 | 
| 53 | 41 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 12550 | 57051 | 
| 54 | 42 django/http/request.py | 151 | 158| 111 | 12661 | 62257 | 
| 55 | 43 django/utils/_os.py | 35 | 60| 190 | 12851 | 62762 | 
| 56 | 44 django/utils/autoreload.py | 59 | 87| 156 | 13007 | 67860 | 
| 57 | 44 django/contrib/admin/utils.py | 310 | 367| 466 | 13473 | 67860 | 
| 58 | 44 django/contrib/gis/gdal/prototypes/generation.py | 40 | 60| 149 | 13622 | 67860 | 
| 59 | 44 django/core/files/storage.py | 212 | 238| 215 | 13837 | 67860 | 
| 60 | 44 django/contrib/gis/geos/prototypes/io.py | 212 | 232| 196 | 14033 | 67860 | 
| 61 | 45 django/utils/cache.py | 135 | 150| 190 | 14223 | 71590 | 
| 62 | 46 django/db/backends/oracle/creation.py | 130 | 165| 399 | 14622 | 75483 | 
| 63 | 47 django/contrib/gis/gdal/prototypes/geom.py | 67 | 110| 694 | 15316 | 76957 | 
| 64 | 47 django/contrib/gis/gdal/raster/source.py | 191 | 216| 216 | 15532 | 76957 | 
| 65 | 47 django/core/files/storage.py | 73 | 104| 341 | 15873 | 76957 | 
| 66 | 48 django/db/backends/dummy/features.py | 1 | 7| 0 | 15873 | 76989 | 
| 67 | 49 django/contrib/gis/apps.py | 1 | 13| 0 | 15873 | 77071 | 
| 68 | 49 django/core/files/uploadhandler.py | 1 | 24| 129 | 16002 | 77071 | 
| 69 | 50 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 16002 | 77141 | 
| 70 | 51 django/template/defaultfilters.py | 821 | 865| 378 | 16380 | 83371 | 
| 71 | 52 django/contrib/messages/views.py | 1 | 19| 0 | 16380 | 83467 | 
| 72 | 53 django/core/files/uploadedfile.py | 39 | 55| 142 | 16522 | 84336 | 
| 73 | 54 django/contrib/staticfiles/apps.py | 1 | 14| 0 | 16522 | 84425 | 
| 74 | 55 django/core/checks/database.py | 1 | 15| 0 | 16522 | 84494 | 
| 75 | 56 django/contrib/gis/db/backends/spatialite/client.py | 1 | 6| 0 | 16522 | 84523 | 
| 76 | 56 django/utils/autoreload.py | 633 | 645| 117 | 16639 | 84523 | 
| 77 | 57 django/contrib/gis/db/models/proxy.py | 1 | 47| 363 | 17002 | 85192 | 
| 78 | 57 django/contrib/gis/geos/prototypes/io.py | 288 | 340| 407 | 17409 | 85192 | 
| 79 | 57 django/contrib/gis/geos/prototypes/io.py | 142 | 155| 131 | 17540 | 85192 | 
| 80 | 58 django/contrib/gis/forms/__init__.py | 1 | 9| 0 | 17540 | 85266 | 
| 81 | 59 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 17678 | 85405 | 
| 82 | 60 django/contrib/gis/db/backends/postgis/schema.py | 49 | 72| 195 | 17873 | 86064 | 
| 83 | 61 django/contrib/admin/exceptions.py | 1 | 12| 0 | 17873 | 86131 | 
| 84 | 62 django/contrib/staticfiles/management/commands/collectstatic.py | 244 | 292| 477 | 18350 | 88949 | 
| 85 | 63 django/contrib/gis/gdal/prototypes/srs.py | 63 | 84| 317 | 18667 | 90014 | 
| 86 | 64 django/template/backends/utils.py | 1 | 15| 0 | 18667 | 90103 | 
| 87 | 65 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 18872 | 90552 | 
| 88 | 66 django/urls/exceptions.py | 1 | 10| 0 | 18872 | 90577 | 
| 89 | 66 django/contrib/gis/gdal/raster/const.py | 1 | 47| 532 | 19404 | 90577 | 
| 90 | 67 django/views/static.py | 1 | 16| 110 | 19514 | 91629 | 
| 91 | 68 django/shortcuts.py | 57 | 78| 216 | 19730 | 92726 | 
| 92 | 68 django/core/files/storage.py | 1 | 24| 171 | 19901 | 92726 | 
| 93 | 68 django/core/validators.py | 1 | 16| 127 | 20028 | 92726 | 
| 94 | 68 django/contrib/gis/gdal/prototypes/ds.py | 1 | 50| 750 | 20778 | 92726 | 
| 95 | 68 django/contrib/gis/db/models/functions.py | 1 | 15| 120 | 20898 | 92726 | 
| 96 | 69 django/contrib/gis/geos/prototypes/threadsafe.py | 29 | 78| 386 | 21284 | 93281 | 
| 97 | 70 django/contrib/messages/utils.py | 1 | 13| 0 | 21284 | 93331 | 
| 98 | 71 django/core/checks/templates.py | 1 | 36| 259 | 21543 | 93591 | 
| 99 | 71 django/contrib/gis/db/models/functions.py | 380 | 420| 294 | 21837 | 93591 | 
| 100 | 72 django/core/files/locks.py | 19 | 119| 779 | 22616 | 94548 | 
| 101 | 72 django/core/files/storage.py | 56 | 71| 138 | 22754 | 94548 | 
| 102 | 72 django/core/files/base.py | 1 | 29| 174 | 22928 | 94548 | 
| 103 | 73 django/views/__init__.py | 1 | 4| 0 | 22928 | 94563 | 
| 104 | 74 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 22928 | 94583 | 
| 105 | 75 django/core/checks/urls.py | 30 | 50| 165 | 23093 | 95284 | 
| 106 | 76 django/contrib/messages/constants.py | 1 | 22| 0 | 23093 | 95380 | 
| 107 | 77 django/contrib/gis/geos/coordseq.py | 66 | 162| 830 | 23923 | 97145 | 
| 108 | 78 django/contrib/staticfiles/storage.py | 130 | 165| 307 | 24230 | 101051 | 
| 109 | 78 django/contrib/gis/geos/prototypes/io.py | 158 | 209| 377 | 24607 | 101051 | 
| 110 | 79 django/contrib/staticfiles/finders.py | 70 | 100| 251 | 24858 | 103154 | 
| 111 | 80 django/contrib/gis/gdal/geometries.py | 340 | 352| 144 | 25002 | 108971 | 
| 112 | 81 django/contrib/gis/db/models/fields.py | 143 | 170| 205 | 25207 | 112031 | 
| 113 | 82 django/contrib/admin/checks.py | 232 | 253| 208 | 25415 | 121213 | 
| 114 | 83 django/contrib/auth/forms.py | 57 | 75| 124 | 25539 | 124339 | 
| 115 | 84 django/contrib/sites/checks.py | 1 | 14| 0 | 25539 | 124418 | 
| 116 | 84 django/utils/autoreload.py | 1 | 56| 287 | 25826 | 124418 | 
| 117 | 85 django/core/management/base.py | 22 | 43| 165 | 25991 | 129190 | 
| 118 | 86 django/contrib/gis/db/backends/mysql/operations.py | 57 | 77| 225 | 26216 | 130078 | 
| 119 | 86 django/utils/cache.py | 217 | 241| 235 | 26451 | 130078 | 
| 120 | 86 django/core/files/storage.py | 303 | 374| 470 | 26921 | 130078 | 
| 121 | 87 django/views/defaults.py | 102 | 121| 149 | 27070 | 131106 | 
| 122 | 88 django/db/backends/postgresql/creation.py | 39 | 54| 173 | 27243 | 131775 | 
| 123 | 89 django/views/debug.py | 1 | 55| 351 | 27594 | 136528 | 
| 124 | 90 django/contrib/gis/db/backends/oracle/introspection.py | 1 | 48| 391 | 27985 | 136920 | 
| 125 | 91 django/contrib/gis/gdal/raster/band.py | 134 | 147| 125 | 28110 | 138704 | 
| 126 | 91 django/utils/autoreload.py | 213 | 248| 376 | 28486 | 138704 | 
| 127 | 91 django/core/checks/caches.py | 1 | 19| 119 | 28605 | 138704 | 
| 128 | 91 django/contrib/gis/geos/prototypes/io.py | 254 | 285| 271 | 28876 | 138704 | 
| 129 | 92 django/contrib/admin/helpers.py | 160 | 209| 437 | 29313 | 142087 | 
| 130 | 92 django/contrib/gis/geos/geometry.py | 563 | 578| 177 | 29490 | 142087 | 
| 131 | 92 django/contrib/gis/geos/prototypes/io.py | 78 | 139| 591 | 30081 | 142087 | 
| 132 | 92 django/views/debug.py | 189 | 201| 143 | 30224 | 142087 | 
| 133 | 93 django/template/loaders/app_directories.py | 1 | 15| 0 | 30224 | 142146 | 
| 134 | 94 django/core/cache/backends/base.py | 280 | 293| 112 | 30336 | 144367 | 
| 135 | 95 django/db/migrations/graph.py | 44 | 58| 121 | 30457 | 146970 | 
| 136 | 96 django/core/signals.py | 1 | 7| 0 | 30457 | 146997 | 
| 137 | 96 django/contrib/gis/db/models/fields.py | 243 | 254| 148 | 30605 | 146997 | 
| 138 | 97 django/contrib/gis/forms/fields.py | 62 | 85| 204 | 30809 | 147916 | 
| 139 | 97 django/contrib/gis/gdal/raster/source.py | 325 | 354| 219 | 31028 | 147916 | 
| 140 | 98 django/contrib/gis/db/backends/spatialite/adapter.py | 1 | 10| 0 | 31028 | 147982 | 
| 141 | 98 django/core/validators.py | 503 | 539| 255 | 31283 | 147982 | 
| 142 | 99 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 31423 | 148833 | 
| 143 | 99 django/views/debug.py | 427 | 464| 298 | 31721 | 148833 | 
| 144 | 100 django/contrib/gis/db/models/sql/__init__.py | 1 | 8| 0 | 31721 | 148868 | 
| 145 | 100 django/views/debug.py | 466 | 510| 429 | 32150 | 148868 | 
| 146 | 101 django/contrib/gis/db/backends/postgis/operations.py | 1 | 25| 216 | 32366 | 152471 | 
| 147 | 101 django/contrib/gis/gdal/raster/source.py | 376 | 429| 519 | 32885 | 152471 | 
| 148 | 102 django/contrib/sessions/exceptions.py | 1 | 17| 0 | 32885 | 152542 | 
| 149 | 103 django/contrib/gis/geos/prototypes/prepared.py | 1 | 29| 282 | 33167 | 152824 | 
| 150 | 103 django/core/files/uploadedfile.py | 58 | 78| 181 | 33348 | 152824 | 
| 151 | 104 django/contrib/gis/geos/prototypes/geom.py | 1 | 82| 814 | 34162 | 153638 | 
| 152 | 105 django/db/utils.py | 1 | 49| 177 | 34339 | 155645 | 
| 153 | 105 django/utils/cache.py | 153 | 193| 453 | 34792 | 155645 | 
| 154 | 106 django/views/csrf.py | 101 | 155| 577 | 35369 | 157189 | 
| 155 | 106 django/contrib/gis/geos/geometry.py | 116 | 131| 169 | 35538 | 157189 | 
| 156 | 107 django/core/exceptions.py | 107 | 218| 752 | 36290 | 158378 | 
| 157 | 107 django/core/checks/urls.py | 71 | 111| 264 | 36554 | 158378 | 
| 158 | 107 django/contrib/gis/geos/prototypes/io.py | 234 | 243| 147 | 36701 | 158378 | 
| 159 | 107 django/contrib/gis/geos/libgeos.py | 1 | 18| 126 | 36827 | 158378 | 
| 160 | 108 django/db/backends/mysql/features.py | 65 | 128| 640 | 37467 | 160630 | 
| 161 | 109 django/core/mail/backends/dummy.py | 1 | 11| 0 | 37467 | 160673 | 
| 162 | 110 django/core/checks/model_checks.py | 155 | 176| 263 | 37730 | 162458 | 
| 163 | 111 django/contrib/postgres/forms/__init__.py | 1 | 4| 0 | 37730 | 162489 | 
| 164 | 112 django/contrib/gis/utils/__init__.py | 1 | 17| 144 | 37874 | 162634 | 
| 165 | 112 django/contrib/staticfiles/storage.py | 374 | 395| 230 | 38104 | 162634 | 
| 166 | 112 django/core/files/uploadedfile.py | 81 | 99| 149 | 38253 | 162634 | 
| 167 | 113 django/contrib/gis/geos/prototypes/__init__.py | 1 | 27| 352 | 38605 | 162987 | 
| 168 | 113 django/contrib/gis/gdal/raster/source.py | 356 | 374| 159 | 38764 | 162987 | 
| 169 | 114 django/core/checks/messages.py | 54 | 77| 161 | 38925 | 163564 | 
| 170 | 115 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 38925 | 163661 | 
| 171 | 116 django/contrib/gis/geoip2/base.py | 142 | 162| 258 | 39183 | 165677 | 
| 172 | 117 django/db/backends/mysql/validation.py | 1 | 31| 239 | 39422 | 166197 | 
| 173 | 117 django/core/checks/model_checks.py | 129 | 153| 268 | 39690 | 166197 | 
| 174 | 118 django/db/backends/oracle/base.py | 60 | 99| 328 | 40018 | 171265 | 
| 175 | 119 django/contrib/gis/db/backends/postgis/const.py | 1 | 53| 620 | 40638 | 171886 | 
| 176 | 120 django/core/files/move.py | 1 | 27| 148 | 40786 | 172570 | 
| 177 | 120 django/core/checks/security/csrf.py | 1 | 42| 304 | 41090 | 172570 | 
| 178 | 120 django/db/models/fields/files.py | 1 | 142| 952 | 42042 | 172570 | 
| 179 | 121 django/utils/text.py | 224 | 238| 156 | 42198 | 175955 | 
| 180 | 122 django/contrib/gis/geos/prototypes/misc.py | 1 | 32| 308 | 42506 | 176263 | 
| 181 | 123 django/db/models/fields/__init__.py | 2188 | 2208| 181 | 42687 | 194410 | 
| 182 | 124 django/utils/baseconv.py | 1 | 55| 147 | 42834 | 195252 | 
| 183 | 125 django/utils/archive.py | 24 | 51| 118 | 42952 | 196862 | 
| 184 | 125 django/db/models/fields/__init__.py | 1652 | 1677| 206 | 43158 | 196862 | 


## Patch

```diff
diff --git a/django/core/files/images.py b/django/core/files/images.py
--- a/django/core/files/images.py
+++ b/django/core/files/images.py
@@ -44,7 +44,10 @@ def get_image_dimensions(file_or_path, close=False):
         file_pos = file.tell()
         file.seek(0)
     else:
-        file = open(file_or_path, 'rb')
+        try:
+            file = open(file_or_path, 'rb')
+        except OSError:
+            return (None, None)
         close = True
     try:
         # Most of the time Pillow only needs a small chunk to parse the image

```

## Test Patch

```diff
diff --git a/tests/files/tests.py b/tests/files/tests.py
--- a/tests/files/tests.py
+++ b/tests/files/tests.py
@@ -369,6 +369,10 @@ def test_valid_image(self):
                 size = images.get_image_dimensions(fh)
                 self.assertEqual(size, (None, None))
 
+    def test_missing_file(self):
+        size = images.get_image_dimensions('missing.png')
+        self.assertEqual(size, (None, None))
+
     @unittest.skipUnless(HAS_WEBP, 'WEBP not installed')
     def test_webp(self):
         img_path = os.path.join(os.path.dirname(__file__), 'test.webp')

```


## Code snippets

### 1 - django/core/files/images.py:

Start line: 33, End line: 85

```python
def get_image_dimensions(file_or_path, close=False):
    """
    Return the (width, height) of an image, given an open file or a path.  Set
    'close' to True to close the file at the end if it is initially in an open
    state.
    """
    from PIL import ImageFile as PillowImageFile

    p = PillowImageFile.Parser()
    if hasattr(file_or_path, 'read'):
        file = file_or_path
        file_pos = file.tell()
        file.seek(0)
    else:
        file = open(file_or_path, 'rb')
        close = True
    try:
        # Most of the time Pillow only needs a small chunk to parse the image
        # and get the dimensions, but with some TIFF files Pillow needs to
        # parse the whole file.
        chunk_size = 1024
        while 1:
            data = file.read(chunk_size)
            if not data:
                break
            try:
                p.feed(data)
            except zlib.error as e:
                # ignore zlib complaining on truncated stream, just feed more
                # data to parser (ticket #19457).
                if e.args[0].startswith("Error -5"):
                    pass
                else:
                    raise
            except struct.error:
                # Ignore PIL failing on a too short buffer when reads return
                # less bytes than expected. Skip and feed more data to the
                # parser (ticket #24544).
                pass
            except RuntimeError:
                # e.g. "RuntimeError: could not create decoder object" for
                # WebP files. A different chunk_size may work.
                pass
            if p.image:
                return p.image.size
            chunk_size *= 2
        return (None, None)
    finally:
        if close:
            file.close()
        else:
            file.seek(file_pos)
```
### 2 - django/db/models/fields/files.py:

Start line: 372, End line: 418

```python
class ImageField(FileField):
    attr_class = ImageFieldFile
    descriptor_class = ImageFileDescriptor
    description = _("Image")

    def __init__(self, verbose_name=None, name=None, width_field=None, height_field=None, **kwargs):
        self.width_field, self.height_field = width_field, height_field
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_image_library_installed(),
        ]

    def _check_image_library_installed(self):
        try:
            from PIL import Image  # NOQA
        except ImportError:
            return [
                checks.Error(
                    'Cannot use ImageField because Pillow is not installed.',
                    hint=('Get Pillow at https://pypi.org/project/Pillow/ '
                          'or run command "python -m pip install Pillow".'),
                    obj=self,
                    id='fields.E210',
                )
            ]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.width_field:
            kwargs['width_field'] = self.width_field
        if self.height_field:
            kwargs['height_field'] = self.height_field
        return name, path, args, kwargs

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        # Attach update_dimension_fields so that dimension fields declared
        # after their corresponding image field don't stay cleared by
        # Model.__init__, see bug #11196.
        # Only run post-initialization dimension update on non-abstract models
        if not cls._meta.abstract:
            signals.post_init.connect(self.update_dimension_fields, sender=cls)
```
### 3 - django/core/files/images.py:

Start line: 1, End line: 30

```python
"""
Utility functions for handling images.

Requires Pillow as you might imagine.
"""
import struct
import zlib

from django.core.files import File


class ImageFile(File):
    """
    A mixin for use alongside django.core.files.base.File, which provides
    additional features for dealing with images.
    """
    @property
    def width(self):
        return self._get_image_dimensions()[0]

    @property
    def height(self):
        return self._get_image_dimensions()[1]

    def _get_image_dimensions(self):
        if not hasattr(self, '_dimensions_cache'):
            close = self.closed
            self.open()
            self._dimensions_cache = get_image_dimensions(self, close=close)
        return self._dimensions_cache
```
### 4 - django/db/models/fields/files.py:

Start line: 420, End line: 482

```python
class ImageField(FileField):

    def update_dimension_fields(self, instance, force=False, *args, **kwargs):
        """
        Update field's width and height fields, if defined.

        This method is hooked up to model's post_init signal to update
        dimensions after instantiating a model instance.  However, dimensions
        won't be updated if the dimensions fields are already populated.  This
        avoids unnecessary recalculation when loading an object from the
        database.

        Dimensions can be forced to update with force=True, which is how
        ImageFileDescriptor.__set__ calls this method.
        """
        # Nothing to update if the field doesn't have dimension fields or if
        # the field is deferred.
        has_dimension_fields = self.width_field or self.height_field
        if not has_dimension_fields or self.attname not in instance.__dict__:
            return

        # getattr will call the ImageFileDescriptor's __get__ method, which
        # coerces the assigned value into an instance of self.attr_class
        # (ImageFieldFile in this case).
        file = getattr(instance, self.attname)

        # Nothing to update if we have no file and not being forced to update.
        if not file and not force:
            return

        dimension_fields_filled = not(
            (self.width_field and not getattr(instance, self.width_field)) or
            (self.height_field and not getattr(instance, self.height_field))
        )
        # When both dimension fields have values, we are most likely loading
        # data from the database or updating an image field that already had
        # an image stored.  In the first case, we don't want to update the
        # dimension fields because we are already getting their values from the
        # database.  In the second case, we do want to update the dimensions
        # fields and will skip this return because force will be True since we
        # were called from ImageFileDescriptor.__set__.
        if dimension_fields_filled and not force:
            return

        # file should be an instance of ImageFieldFile or should be None.
        if file:
            width = file.width
            height = file.height
        else:
            # No file, so clear dimensions fields.
            width = None
            height = None

        # Update the width and height fields.
        if self.width_field:
            setattr(instance, self.width_field, width)
        if self.height_field:
            setattr(instance, self.height_field, height)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.ImageField,
            **kwargs,
        })
```
### 5 - django/db/models/fields/files.py:

Start line: 342, End line: 369

```python
class ImageFileDescriptor(FileDescriptor):
    """
    Just like the FileDescriptor, but for ImageFields. The only difference is
    assigning the width/height to the width_field/height_field, if appropriate.
    """
    def __set__(self, instance, value):
        previous_file = instance.__dict__.get(self.field.attname)
        super().__set__(instance, value)

        # To prevent recalculating image dimensions when we are instantiating
        # an object from the database (bug #11084), only update dimensions if
        # the field had a value before this assignment.  Since the default
        # value for FileField subclasses is an instance of field.attr_class,
        # previous_file will only be None when we are called from
        # Model.__init__().  The ImageField.update_dimension_fields method
        # hooked up to the post_init signal handles the Model.__init__() cases.
        # Assignment happening outside of Model.__init__() will trigger the
        # update right here.
        if previous_file is not None:
            self.field.update_dimension_fields(instance, force=True)


class ImageFieldFile(ImageFile, FieldFile):
    def delete(self, save=True):
        # Clear the image dimensions cache
        if hasattr(self, '_dimensions_cache'):
            del self._dimensions_cache
        super().delete(save)
```
### 6 - django/forms/fields.py:

Start line: 612, End line: 668

```python
class ImageField(FileField):
    default_validators = [validators.validate_image_file_extension]
    default_error_messages = {
        'invalid_image': _(
            "Upload a valid image. The file you uploaded was either not an "
            "image or a corrupted image."
        ),
    }

    def to_python(self, data):
        """
        Check that the file-upload field data contains a valid image (GIF, JPG,
        PNG, etc. -- whatever Pillow supports).
        """
        f = super().to_python(data)
        if f is None:
            return None

        from PIL import Image

        # We need to get a file object for Pillow. We might have a path or we might
        # have to read the data into memory.
        if hasattr(data, 'temporary_file_path'):
            file = data.temporary_file_path()
        else:
            if hasattr(data, 'read'):
                file = BytesIO(data.read())
            else:
                file = BytesIO(data['content'])

        try:
            # load() could spot a truncated JPEG, but it loads the entire
            # image in memory, which is a DoS vector. See #3848 and #18520.
            image = Image.open(file)
            # verify() must be called immediately after the constructor.
            image.verify()

            # Annotating so subclasses can reuse it for their own validation
            f.image = image
            # Pillow doesn't detect the MIME type of all formats. In those
            # cases, content_type will be None.
            f.content_type = Image.MIME.get(image.format)
        except Exception as exc:
            # Pillow doesn't recognize it as an image.
            raise ValidationError(
                self.error_messages['invalid_image'],
                code='invalid_image',
            ) from exc
        if hasattr(f, 'seek') and callable(f.seek):
            f.seek(0)
        return f

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, FileInput) and 'accept' not in widget.attrs:
            attrs.setdefault('accept', 'image/*')
        return attrs
```
### 7 - django/contrib/gis/gdal/prototypes/raster.py:

Start line: 53, End line: 102

```python
get_ds_metadata = chararray_output(std_call('GDALGetMetadata'), [c_void_p, c_char_p], errcheck=False)
set_ds_metadata = void_output(std_call('GDALSetMetadata'), [c_void_p, POINTER(c_char_p), c_char_p])
get_ds_metadata_domain_list = chararray_output(std_call('GDALGetMetadataDomainList'), [c_void_p], errcheck=False)
get_ds_metadata_item = const_string_output(std_call('GDALGetMetadataItem'), [c_void_p, c_char_p, c_char_p])
set_ds_metadata_item = const_string_output(std_call('GDALSetMetadataItem'), [c_void_p, c_char_p, c_char_p, c_char_p])
free_dsl = void_output(std_call('CSLDestroy'), [POINTER(c_char_p)], errcheck=False)

# Raster Band Routines
band_io = void_output(
    std_call('GDALRasterIO'),
    [c_void_p, c_int, c_int, c_int, c_int, c_int, c_void_p, c_int, c_int, c_int, c_int, c_int]
)
get_band_xsize = int_output(std_call('GDALGetRasterBandXSize'), [c_void_p])
get_band_ysize = int_output(std_call('GDALGetRasterBandYSize'), [c_void_p])
get_band_index = int_output(std_call('GDALGetBandNumber'), [c_void_p])
get_band_description = const_string_output(std_call('GDALGetDescription'), [c_void_p])
get_band_ds = voidptr_output(std_call('GDALGetBandDataset'), [c_void_p])
get_band_datatype = int_output(std_call('GDALGetRasterDataType'), [c_void_p])
get_band_color_interp = int_output(std_call('GDALGetRasterColorInterpretation'), [c_void_p])
get_band_nodata_value = double_output(std_call('GDALGetRasterNoDataValue'), [c_void_p, POINTER(c_int)])
set_band_nodata_value = void_output(std_call('GDALSetRasterNoDataValue'), [c_void_p, c_double])
delete_band_nodata_value = void_output(std_call('GDALDeleteRasterNoDataValue'), [c_void_p])
get_band_statistics = void_output(
    std_call('GDALGetRasterStatistics'),
    [
        c_void_p, c_int, c_int, POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), c_void_p, c_void_p,
    ],
)
compute_band_statistics = void_output(
    std_call('GDALComputeRasterStatistics'),
    [c_void_p, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_void_p, c_void_p],
)

# Reprojection routine
reproject_image = void_output(
    std_call('GDALReprojectImage'),
    [c_void_p, c_char_p, c_void_p, c_char_p, c_int, c_double, c_double, c_void_p, c_void_p, c_void_p]
)
auto_create_warped_vrt = voidptr_output(
    std_call('GDALAutoCreateWarpedVRT'),
    [c_void_p, c_char_p, c_char_p, c_int, c_double, c_void_p]
)

# Create VSI gdal raster files from in-memory buffers.
# https://gdal.org/api/cpl.html#cpl-vsi-h
create_vsi_file_from_mem_buffer = voidptr_output(std_call('VSIFileFromMemBuffer'), [c_char_p, c_void_p, c_int, c_int])
get_mem_buffer_from_vsi_file = voidptr_output(std_call('VSIGetMemFileBuffer'), [c_char_p, POINTER(c_int), c_bool])
unlink_vsi_file = int_output(std_call('VSIUnlink'), [c_char_p])
```
### 8 - django/core/validators.py:

Start line: 542, End line: 578

```python
def get_available_image_extensions():
    try:
        from PIL import Image
    except ImportError:
        return []
    else:
        Image.init()
        return [ext.lower()[1:] for ext in Image.EXTENSION]


def validate_image_file_extension(value):
    return FileExtensionValidator(allowed_extensions=get_available_image_extensions())(value)


@deconstructible
class ProhibitNullCharactersValidator:
    """Validate that the string doesn't contain the null character."""
    message = _('Null characters are not allowed.')
    code = 'null_characters_not_allowed'

    def __init__(self, message=None, code=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code

    def __call__(self, value):
        if '\x00' in str(value):
            raise ValidationError(self.message, code=self.code, params={'value': value})

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self.message == other.message and
            self.code == other.code
        )
```
### 9 - django/contrib/gis/geos/prototypes/errcheck.py:

Start line: 54, End line: 84

```python
def check_sized_string(result, func, cargs):
    """
    Error checking for routines that return explicitly sized strings.

    This frees the memory allocated by GEOS at the result pointer.
    """
    if not result:
        raise GEOSException('Invalid string pointer returned by GEOS C function "%s"' % func.__name__)
    # A c_size_t object is passed in by reference for the second
    # argument on these routines, and its needed to determine the
    # correct size.
    s = string_at(result, last_arg_byref(cargs))
    # Freeing the memory allocated within GEOS
    free(result)
    return s


def check_string(result, func, cargs):
    """
    Error checking for routines that return strings.

    This frees the memory allocated by GEOS at the result pointer.
    """
    if not result:
        raise GEOSException('Error encountered checking string return value in GEOS C function "%s".' % func.__name__)
    # Getting the string value at the pointer address.
    s = string_at(result)
    # Freeing the memory allocated within GEOS
    free(result)
    return s
```
### 10 - django/contrib/gis/geos/error.py:

Start line: 1, End line: 4

```python

```
