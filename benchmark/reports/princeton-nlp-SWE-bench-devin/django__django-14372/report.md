# django__django-14372

| **django/django** | `b81c7562fc33f50166d5120138d6398dc42b13c3` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 1920 |
| **Any found context length** | 1920 |
| **Avg pos** | 62.5 |
| **Min pos** | 2 |
| **Max pos** | 123 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/files/utils.py b/django/core/files/utils.py
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -1,16 +1,26 @@
 import os
+import pathlib
 
 from django.core.exceptions import SuspiciousFileOperation
 
 
-def validate_file_name(name):
-    if name != os.path.basename(name):
-        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
-
+def validate_file_name(name, allow_relative_path=False):
     # Remove potentially dangerous names
-    if name in {'', '.', '..'}:
+    if os.path.basename(name) in {'', '.', '..'}:
         raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
 
+    if allow_relative_path:
+        # Use PurePosixPath() because this branch is checked only in
+        # FileField.generate_filename() where all file paths are expected to be
+        # Unix style (with forward slashes).
+        path = pathlib.PurePosixPath(name)
+        if path.is_absolute() or '..' in path.parts:
+            raise SuspiciousFileOperation(
+                "Detected path traversal attempt in '%s'" % name
+            )
+    elif name != os.path.basename(name):
+        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
+
     return name
 
 
diff --git a/django/db/models/fields/files.py b/django/db/models/fields/files.py
--- a/django/db/models/fields/files.py
+++ b/django/db/models/fields/files.py
@@ -313,12 +313,12 @@ def generate_filename(self, instance, filename):
         Until the storage layer, all file paths are expected to be Unix style
         (with forward slashes).
         """
-        filename = validate_file_name(filename)
         if callable(self.upload_to):
             filename = self.upload_to(instance, filename)
         else:
             dirname = datetime.datetime.now().strftime(str(self.upload_to))
             filename = posixpath.join(dirname, filename)
+        filename = validate_file_name(filename, allow_relative_path=True)
         return self.storage.generate_filename(filename)
 
     def save_form_data(self, instance, data):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/files/utils.py | 2 | 7 | 123 | 32 | 32608
| django/db/models/fields/files.py | 316 | 316 | 2 | 1 | 1920


## Problem Statement

```
Saving a FileField raises SuspiciousFileOperation in some scenarios.
Description
	
I came across this issue today when I was updating Django from 3.2.0 -> 3.2.1.
It's directly caused by: ​https://docs.djangoproject.com/en/3.2/releases/3.2.1/#cve-2021-31542-potential-directory-traversal-via-uploaded-files
Starting from 3.2.1, Django requires that only the basename is passed to FieldFile.save method, because otherwise it raises a new exception:
SuspiciousFileOperation: File name ... includes path elements
The issue is that in FileField.pre_save, a full path is passed to FieldFile.save, causing the exception to be raised.
Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself.
Steps to reproduce:
model_instance.file_attribute = File(open(path, 'rb'))
model_instance.save()
I also created a PR with the fix: ​https://github.com/django/django/pull/14354

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/files.py** | 1 | 142| 952 | 952 | 3828 | 
| **-> 2 <-** | **1 django/db/models/fields/files.py** | 217 | 339| 968 | 1920 | 3828 | 
| 3 | 2 django/db/migrations/operations/fields.py | 1 | 37| 241 | 2161 | 6951 | 
| 4 | 3 django/core/files/uploadedfile.py | 39 | 55| 142 | 2303 | 7820 | 
| 5 | 4 django/core/files/storage.py | 240 | 301| 534 | 2837 | 10764 | 
| 6 | 5 django/db/models/base.py | 915 | 947| 385 | 3222 | 28068 | 
| 7 | 6 django/core/files/base.py | 1 | 29| 174 | 3396 | 29120 | 
| 8 | 7 django/db/models/fields/__init__.py | 1660 | 1685| 206 | 3602 | 47544 | 
| 9 | **7 django/db/models/fields/files.py** | 372 | 418| 373 | 3975 | 47544 | 
| 10 | 8 django/forms/fields.py | 552 | 571| 171 | 4146 | 56888 | 
| 11 | 8 django/core/files/base.py | 75 | 118| 303 | 4449 | 56888 | 
| 12 | 8 django/forms/fields.py | 573 | 598| 243 | 4692 | 56888 | 
| 13 | 8 django/db/migrations/operations/fields.py | 39 | 61| 183 | 4875 | 56888 | 
| 14 | 8 django/forms/fields.py | 534 | 550| 180 | 5055 | 56888 | 
| 15 | 8 django/db/migrations/operations/fields.py | 85 | 95| 124 | 5179 | 56888 | 
| 16 | 8 django/db/migrations/operations/fields.py | 301 | 344| 410 | 5589 | 56888 | 
| 17 | 8 django/db/migrations/operations/fields.py | 216 | 234| 185 | 5774 | 56888 | 
| 18 | 8 django/core/files/storage.py | 1 | 24| 171 | 5945 | 56888 | 
| 19 | **8 django/db/models/fields/files.py** | 145 | 214| 711 | 6656 | 56888 | 
| 20 | 8 django/db/migrations/operations/fields.py | 236 | 246| 146 | 6802 | 56888 | 
| 21 | 8 django/db/migrations/operations/fields.py | 273 | 299| 158 | 6960 | 56888 | 
| 22 | 8 django/db/migrations/operations/fields.py | 346 | 381| 335 | 7295 | 56888 | 
| 23 | 8 django/db/migrations/operations/fields.py | 97 | 109| 130 | 7425 | 56888 | 
| 24 | 8 django/db/models/fields/__init__.py | 1687 | 1701| 144 | 7569 | 56888 | 
| 25 | 9 django/db/migrations/serializer.py | 198 | 232| 281 | 7850 | 59559 | 
| 26 | 9 django/db/migrations/operations/fields.py | 64 | 83| 128 | 7978 | 59559 | 
| 27 | 9 django/db/models/fields/__init__.py | 1244 | 1262| 180 | 8158 | 59559 | 
| 28 | 10 django/core/checks/files.py | 1 | 20| 0 | 8158 | 59663 | 
| 29 | 10 django/db/models/fields/__init__.py | 1703 | 1721| 134 | 8292 | 59663 | 
| 30 | 10 django/db/migrations/operations/fields.py | 111 | 121| 127 | 8419 | 59663 | 
| 31 | 11 django/db/models/fields/related.py | 108 | 125| 155 | 8574 | 73486 | 
| 32 | 11 django/db/models/fields/related.py | 127 | 154| 201 | 8775 | 73486 | 
| 33 | 11 django/db/migrations/operations/fields.py | 248 | 270| 188 | 8963 | 73486 | 
| 34 | 11 django/db/models/fields/__init__.py | 1394 | 1422| 281 | 9244 | 73486 | 
| 35 | 11 django/db/models/fields/related.py | 171 | 184| 140 | 9384 | 73486 | 
| 36 | 11 django/core/files/storage.py | 106 | 115| 116 | 9500 | 73486 | 
| 37 | 11 django/db/models/fields/related.py | 156 | 169| 144 | 9644 | 73486 | 
| 38 | 11 django/db/migrations/operations/fields.py | 123 | 143| 129 | 9773 | 73486 | 
| 39 | 11 django/forms/fields.py | 1083 | 1124| 353 | 10126 | 73486 | 
| 40 | 11 django/core/files/storage.py | 212 | 238| 215 | 10341 | 73486 | 
| 41 | 11 django/db/models/fields/related.py | 935 | 948| 126 | 10467 | 73486 | 
| 42 | 11 django/db/models/fields/__init__.py | 2333 | 2394| 425 | 10892 | 73486 | 
| 43 | 11 django/db/models/fields/__init__.py | 2310 | 2330| 163 | 11055 | 73486 | 
| 44 | 12 django/db/migrations/autodetector.py | 915 | 995| 871 | 11926 | 85066 | 
| 45 | 12 django/db/models/fields/related.py | 253 | 282| 271 | 12197 | 85066 | 
| 46 | 13 django/core/files/__init__.py | 1 | 4| 0 | 12197 | 85081 | 
| 47 | 13 django/db/migrations/operations/fields.py | 383 | 403| 160 | 12357 | 85081 | 
| 48 | 14 django/core/validators.py | 490 | 526| 255 | 12612 | 89663 | 
| 49 | 14 django/db/models/base.py | 999 | 1027| 230 | 12842 | 89663 | 
| 50 | 14 django/db/models/fields/__init__.py | 366 | 392| 199 | 13041 | 89663 | 
| 51 | 14 django/core/files/base.py | 31 | 46| 129 | 13170 | 89663 | 
| 52 | 14 django/db/models/fields/__init__.py | 1160 | 1198| 293 | 13463 | 89663 | 
| 53 | 15 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 13658 | 89858 | 
| 54 | 15 django/db/models/fields/related.py | 984 | 995| 128 | 13786 | 89858 | 
| 55 | 16 django/contrib/postgres/fields/ranges.py | 43 | 91| 362 | 14148 | 91950 | 
| 56 | 16 django/db/migrations/operations/fields.py | 146 | 189| 394 | 14542 | 91950 | 
| 57 | 16 django/db/models/fields/__init__.py | 589 | 630| 287 | 14829 | 91950 | 
| 58 | **16 django/db/models/fields/files.py** | 420 | 482| 551 | 15380 | 91950 | 
| 59 | 17 django/core/files/uploadhandler.py | 93 | 105| 120 | 15500 | 93366 | 
| 60 | 17 django/db/models/fields/related.py | 284 | 318| 293 | 15793 | 93366 | 
| 61 | **17 django/db/models/fields/files.py** | 342 | 369| 277 | 16070 | 93366 | 
| 62 | 17 django/db/models/base.py | 676 | 731| 499 | 16569 | 93366 | 
| 63 | 18 django/contrib/sessions/backends/file.py | 109 | 168| 530 | 17099 | 94855 | 
| 64 | 19 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 17099 | 94908 | 
| 65 | 19 django/db/models/fields/__init__.py | 1142 | 1158| 175 | 17274 | 94908 | 
| 66 | 19 django/db/models/fields/related.py | 1235 | 1352| 963 | 18237 | 94908 | 
| 67 | 19 django/db/models/fields/related.py | 487 | 507| 138 | 18375 | 94908 | 
| 68 | 20 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 18375 | 94931 | 
| 69 | 21 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 18880 | 99105 | 
| 70 | 21 django/db/models/fields/__init__.py | 494 | 505| 184 | 19064 | 99105 | 
| 71 | 21 django/db/models/base.py | 1721 | 1769| 348 | 19412 | 99105 | 
| 72 | 22 django/http/multipartparser.py | 296 | 336| 366 | 19778 | 104327 | 
| 73 | 22 django/db/models/fields/related.py | 913 | 933| 178 | 19956 | 104327 | 
| 74 | 22 django/forms/fields.py | 601 | 657| 438 | 20394 | 104327 | 
| 75 | 23 django/core/cache/backends/filebased.py | 1 | 44| 284 | 20678 | 105553 | 
| 76 | 23 django/db/models/fields/__init__.py | 1300 | 1341| 332 | 21010 | 105553 | 
| 77 | 23 django/core/files/storage.py | 56 | 71| 138 | 21148 | 105553 | 
| 78 | 23 django/db/models/fields/__init__.py | 2081 | 2111| 252 | 21400 | 105553 | 
| 79 | 23 django/db/migrations/autodetector.py | 894 | 913| 184 | 21584 | 105553 | 
| 80 | 23 django/core/files/uploadhandler.py | 27 | 58| 204 | 21788 | 105553 | 
| 81 | 23 django/db/models/fields/__init__.py | 1441 | 1464| 171 | 21959 | 105553 | 
| 82 | 23 django/db/models/fields/__init__.py | 2397 | 2447| 339 | 22298 | 105553 | 
| 83 | 24 django/contrib/contenttypes/fields.py | 1 | 17| 134 | 22432 | 111004 | 
| 84 | 24 django/db/models/fields/related.py | 864 | 890| 240 | 22672 | 111004 | 
| 85 | 24 django/db/models/fields/related.py | 1202 | 1233| 180 | 22852 | 111004 | 
| 86 | 24 django/core/files/storage.py | 183 | 198| 151 | 23003 | 111004 | 
| 87 | 24 django/db/models/fields/__init__.py | 207 | 241| 234 | 23237 | 111004 | 
| 88 | 24 django/db/models/fields/related.py | 1354 | 1426| 616 | 23853 | 111004 | 
| 89 | 24 django/core/files/uploadedfile.py | 81 | 99| 149 | 24002 | 111004 | 
| 90 | 24 django/db/models/fields/__init__.py | 2281 | 2307| 213 | 24215 | 111004 | 
| 91 | 24 django/db/models/fields/__init__.py | 543 | 567| 225 | 24440 | 111004 | 
| 92 | 24 django/db/models/fields/__init__.py | 632 | 661| 234 | 24674 | 111004 | 
| 93 | 24 django/db/models/fields/related.py | 841 | 862| 169 | 24843 | 111004 | 
| 94 | 24 django/core/files/storage.py | 73 | 104| 341 | 25184 | 111004 | 
| 95 | 25 django/core/serializers/base.py | 232 | 249| 208 | 25392 | 113430 | 
| 96 | 25 django/db/models/fields/__init__.py | 2196 | 2237| 325 | 25717 | 113430 | 
| 97 | 25 django/db/migrations/autodetector.py | 856 | 892| 352 | 26069 | 113430 | 
| 98 | 25 django/core/files/uploadedfile.py | 58 | 78| 181 | 26250 | 113430 | 
| 99 | 25 django/core/files/storage.py | 27 | 54| 211 | 26461 | 113430 | 
| 100 | 25 django/db/models/fields/__init__.py | 1466 | 1488| 121 | 26582 | 113430 | 
| 101 | 25 django/db/models/fields/__init__.py | 1514 | 1573| 403 | 26985 | 113430 | 
| 102 | 25 django/db/models/fields/__init__.py | 1285 | 1298| 157 | 27142 | 113430 | 
| 103 | 25 django/db/models/base.py | 732 | 781| 456 | 27598 | 113430 | 
| 104 | 25 django/db/models/base.py | 968 | 982| 212 | 27810 | 113430 | 
| 105 | 25 django/core/files/storage.py | 303 | 373| 462 | 28272 | 113430 | 
| 106 | 25 django/db/models/fields/related.py | 1 | 34| 246 | 28518 | 113430 | 
| 107 | 26 django/contrib/admin/utils.py | 1 | 25| 231 | 28749 | 117588 | 
| 108 | 27 django/core/checks/caches.py | 59 | 73| 109 | 28858 | 118109 | 
| 109 | 27 django/db/models/fields/__init__.py | 1264 | 1282| 149 | 29007 | 118109 | 
| 110 | 27 django/core/cache/backends/filebased.py | 46 | 59| 145 | 29152 | 118109 | 
| 111 | 28 django/contrib/gis/db/models/fields.py | 172 | 198| 239 | 29391 | 121169 | 
| 112 | 28 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 29813 | 121169 | 
| 113 | 28 django/db/migrations/operations/fields.py | 192 | 214| 147 | 29960 | 121169 | 
| 114 | 28 django/db/models/fields/__init__.py | 1724 | 1761| 226 | 30186 | 121169 | 
| 115 | 28 django/db/models/fields/__init__.py | 337 | 364| 203 | 30389 | 121169 | 
| 116 | 28 django/db/models/fields/__init__.py | 1636 | 1657| 183 | 30572 | 121169 | 
| 117 | 29 django/core/management/commands/makemessages.py | 37 | 58| 143 | 30715 | 126756 | 
| 118 | 30 django/contrib/staticfiles/storage.py | 265 | 335| 575 | 31290 | 130416 | 
| 119 | 31 django/db/backends/sqlite3/operations.py | 246 | 275| 198 | 31488 | 133688 | 
| 120 | 31 django/core/serializers/base.py | 301 | 323| 207 | 31695 | 133688 | 
| 121 | 31 django/db/models/base.py | 1208 | 1242| 230 | 31925 | 133688 | 
| 122 | 31 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 32143 | 133688 | 
| **-> 123 <-** | **32 django/core/files/utils.py** | 1 | 69| 465 | 32608 | 134153 | 
| 124 | 32 django/db/models/fields/related.py | 997 | 1024| 215 | 32823 | 134153 | 
| 125 | 32 django/db/models/base.py | 949 | 966| 177 | 33000 | 134153 | 
| 126 | 32 django/core/files/uploadhandler.py | 1 | 24| 129 | 33129 | 134153 | 
| 127 | 32 django/db/models/fields/related.py | 652 | 668| 163 | 33292 | 134153 | 
| 128 | 32 django/db/models/fields/__init__.py | 1076 | 1091| 173 | 33465 | 134153 | 
| 129 | 32 django/core/cache/backends/filebased.py | 61 | 96| 260 | 33725 | 134153 | 
| 130 | 32 django/db/models/base.py | 1161 | 1176| 138 | 33863 | 134153 | 
| 131 | 33 django/db/models/options.py | 252 | 287| 341 | 34204 | 141520 | 
| 132 | 33 django/db/models/fields/related.py | 509 | 574| 492 | 34696 | 141520 | 
| 133 | 33 django/db/migrations/autodetector.py | 1105 | 1142| 317 | 35013 | 141520 | 
| 134 | 33 django/db/models/fields/related.py | 186 | 252| 674 | 35687 | 141520 | 
| 135 | 34 django/conf/global_settings.py | 267 | 349| 800 | 36487 | 147207 | 
| 136 | 34 django/db/models/fields/__init__.py | 1424 | 1438| 121 | 36608 | 147207 | 
| 137 | 34 django/db/models/fields/related.py | 630 | 650| 168 | 36776 | 147207 | 
| 138 | 35 django/db/backends/base/schema.py | 1161 | 1183| 199 | 36975 | 159902 | 
| 139 | 36 django/utils/archive.py | 24 | 51| 118 | 37093 | 161512 | 
| 140 | 36 django/contrib/staticfiles/storage.py | 355 | 376| 230 | 37323 | 161512 | 
| 141 | 36 django/db/models/fields/related.py | 892 | 911| 145 | 37468 | 161512 | 
| 142 | 37 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 37764 | 165601 | 
| 143 | 37 django/db/models/fields/related.py | 611 | 628| 197 | 37961 | 165601 | 
| 144 | 38 django/db/migrations/questioner.py | 55 | 80| 220 | 38181 | 167658 | 
| 145 | 39 django/db/models/query.py | 786 | 809| 207 | 38388 | 185177 | 
| 146 | 39 django/db/models/fields/related.py | 750 | 768| 222 | 38610 | 185177 | 
| 147 | 39 django/db/models/fields/__init__.py | 524 | 541| 182 | 38792 | 185177 | 
| 148 | 39 django/db/models/fields/__init__.py | 1343 | 1392| 342 | 39134 | 185177 | 
| 149 | 39 django/db/models/fields/__init__.py | 307 | 335| 205 | 39339 | 185177 | 
| 150 | 39 django/db/models/fields/__init__.py | 1017 | 1036| 128 | 39467 | 185177 | 
| 151 | 40 django/db/migrations/operations/models.py | 602 | 618| 215 | 39682 | 192161 | 
| 152 | 41 django/core/files/move.py | 1 | 27| 148 | 39830 | 192845 | 
| 153 | 41 django/db/migrations/operations/models.py | 317 | 342| 290 | 40120 | 192845 | 
| 154 | 41 django/db/models/fields/related.py | 576 | 609| 334 | 40454 | 192845 | 
| 155 | 42 django/db/backends/oracle/schema.py | 60 | 80| 249 | 40703 | 194896 | 
| 156 | 42 django/core/files/base.py | 48 | 73| 177 | 40880 | 194896 | 
| 157 | 42 django/db/models/fields/__init__.py | 1001 | 1015| 122 | 41002 | 194896 | 
| 158 | 42 django/db/models/base.py | 1244 | 1267| 172 | 41174 | 194896 | 
| 159 | 42 django/forms/fields.py | 1178 | 1215| 199 | 41373 | 194896 | 
| 160 | 42 django/db/models/fields/related.py | 1661 | 1695| 266 | 41639 | 194896 | 
| 161 | 42 django/core/serializers/base.py | 219 | 230| 157 | 41796 | 194896 | 
| 162 | 43 django/db/backends/mysql/schema.py | 102 | 115| 148 | 41944 | 196437 | 
| 163 | 43 django/core/files/move.py | 30 | 88| 535 | 42479 | 196437 | 
| 164 | 43 django/db/models/fields/__init__.py | 1490 | 1512| 119 | 42598 | 196437 | 
| 165 | 44 django/core/files/locks.py | 19 | 119| 779 | 43377 | 197394 | 
| 166 | 44 django/db/models/fields/__init__.py | 2450 | 2499| 311 | 43688 | 197394 | 


### Hint

```
I am also experiencing this issue on 2.2.21. It's just as you described it. I'm going to apply your PR onto my 2.2.21 checkout and see if it resolves it for me also.
Oh, good catch Brian! I forgot to mention the bug is also present in 3.1.9 and 2.2.21 as they contain the CVE-2021-31542 fix, too.
Hi, I came across this as well. I had to make changes like this one all over my project to get things to work after the CV in 2.2.21. merged_path = '/tmp/merged.png' with open(img_path, mode='rb') as f: - field.merged_map = ImageFile(f) + field.merged_map = ImageFile(f, name=os.path.basename(merged_path)) field.save() # raises SuspiciousFileOperation("File name '/tmp/merged.png' includes path elements") unless change is made on previous line (this hotfix still works with my upload_to functions, so the file still goes to the right place) This shouldn't have had to be the case, as the CV was about "empty file names and paths with dot segments", and that doesn't apply in any of my cases. To me, it looks like Jakub's patch is sound and would solve this.
Correct me if I'm wrong, but this fix and also this ​commit breaks the Django file storage completely... For instance - Having a file field with an upload_to="users" Create a new file with name = "steve/file.txt" Create another file with name = "john/file.txt" Both files will be written (possibly overwritten!!) to: "users/file.txt" as you are only using the file basename?? Further the commit above (now released in 3.2.1) added: if name != os.path.basename(name): now invalidates the use of paths in the filename, which previously worked...
Correct me if I'm wrong, but this fix and also this ​commit breaks the Django file storage completely... It's ​documented that the FieldField.save()'s name argument "is the name of the file" not a path. You can define upload_to as a function to use subdirectories, see an example in FileField.upload_to ​docs.
Replying to Mariusz Felisiak: Correct me if I'm wrong, but this fix and also this ​commit breaks the Django file storage completely... It's ​documented that the FieldField.save()'s name argument "is the name of the file" not a path. You can define upload_to as a function to use subdirectories, see an example in FileField.upload_to ​docs. Outch, sadly it is not that simple. When you look at your documentation link about the .save() method it says: Note that the content argument should be an instance of django.core.files.File, not Python’s built-in file object. You can construct a File from an existing Python file object like this: This one links to ​https://docs.djangoproject.com/en/3.2/ref/files/file/#django.core.files.File which says name is The name of the file including the relative path from MEDIA_ROOT. And now all fixes are getting ugly… So now the question becomes: If you do File(…, name='path/to/something.txt') and then call .save(…), what should be the final path? Should it add the full relative path to upload_to, should it take the basename and set that in conjunction to upload_to or should it just take the path as is?
Replying to Florian Apolloner: Replying to Mariusz Felisiak: Thanks Mariusz Felisiak, you're correct - these changes will break Django's file system and cause files to possibly be overwritten. It must not go ahead. The core issue was the addition of an unneccessary check added here: ​https://github.com/django/django/blob/main/django/core/files/utils.py#L7 def validate_file_name(name): if name != os.path.basename(name): raise SuspiciousFileOperation("File name '%s' includes path elements" % name) It currently appends (joins) the file name to upload_to.
Replying to Florian Apolloner: Florian - it was your commit that has caused this break - surely this can be simply removed/refactored? ​https://github.com/django/django/commit/0b79eb36915d178aef5c6a7bbce71b1e76d376d3 This will affect thousands of applications and systems... many not frequently maintained. For the sake of 2 lines of code.
In my opinion, this ticket should get special care and urgency. I also had to update applications. I'm not convinced such a change is appropriate in stable releases.
Replying to Phillip Marshall: Hi, I came across this as well. I had to make changes like this one all over my project to get things to work after the CV in 2.2.21. merged_path = '/tmp/merged.png' with open(img_path, mode='rb') as f: - field.merged_map = ImageFile(f) + field.merged_map = ImageFile(f, name=os.path.basename(merged_path)) field.save() # raises SuspiciousFileOperation("File name '/tmp/merged.png' includes path elements") unless change is made on previous line Hi Phillip, I have tried your original code (without the basename) on 3.2 & 2.2.20 and it yielded this for me: SuspiciousFileOperation: The joined path (/tmp/merged.png) is located outside of the base path component (/home/florian/sources/django.git/testing) This was my code (functionally equal to yours): In [1]: from django.core.files import File In [2]: from testabc.models import * In [3]: f = open('/tmp/merged.png') In [4]: f2 = File(f) In [5]: x = Test() In [6]: x.file = f2 In [7]: x.save() So your code already fails for me before any changes. Can you please provide a reproducer and full traceback?
Replying to Jakub Kleň: Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself. This is true, but according to my tests on earlier versions a full (absolute) path did fail already because it would usually be outside the MEDIA_ROOT. Can you provide some more details?
Replying to Florian Apolloner: Replying to Jakub Kleň: Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself. This is true, but according to my tests on earlier versions a full (absolute) path did fail already because it would usually be outside the MEDIA_ROOT. Can you provide some more details? I am setting the images as I described in the initial post, with absolute paths which are outside of MEDIA_ROOT, and am not having any issues with it. I'm currently actually using v3.2.1, because I had to update. I initially updated to 3.2.0, but that caused some crashes because of a bug in legacy cookie decoding, so I really had no other option than to update to 3.2.1. So I temporarily monkey patched django.core.files.utils.validate_file_name to just return name, and everything works perfectly. Regarding the FileField.save method and the parameter it takes, to me it kind of makes sense to only pass in the basename of the file. I'm not completely sure if we should be passing in a path in the case of pre_save. It doesn't make sense to me to derive the final path of the file from that. The full path should be generated in a custom upload_to, and that parameter should only be used to e.g. reuse the same extension of the original file. But not to append that full path to upload_to. Note I'm talking only about this special case with pre_save, where files set to the model fields are handled. It would still be possible to manually call field.save('some/path/file.png. Although I'm not sure why someone would do that, because the path should be provided in upload_to I think. But I know we also have to think about backwards compatibility, so I'm not quite sure what is a viable solution here. Imagine the following scenario (assuming the FileField has upload_to='media'): model.file_field = File(open('/folder/file.png', 'rb')) model.save() Should the file be written to media/folder/file.png? I don't think the final path in MEDIA_ROOT should depend on the original path of the file. And we can't even provide File(open(), name='file.png') manually, because it would break File.open() where the name is being used. If you want to provide a custom path during save, you can still do that by manually calling file_field.save(path, but in this special case of pre_save, where the name is the full path, I think it would make sense to only use the basename. I'm not completely sure what Django does by default when upload_to is a path (string), as I'm using a callable where the only thing I take from the filename is the extension, which I reuse. So both ways will work for me.
Florian, You can also see some other use case here: ​https://gitlab.gnome.org/Infrastructure/damned-lies/-/blob/085328022/vertimus/models.py#L751, where we copy the file attached to a model to a new model. I had to wrap action.file.name with basename to make it work with 3.2.1. The tests failed otherwise.
Replying to Jakub Kleň: Replying to Florian Apolloner: Replying to Jakub Kleň: Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself. This is true, but according to my tests on earlier versions a full (absolute) path did fail already because it would usually be outside the MEDIA_ROOT. Can you provide some more details? I am setting the images as I described in the initial post, with absolute paths which are outside of MEDIA_ROOT, and am not having any issues with it. I'm currently actually using v3.2.1, because I had to update. I initially updated to 3.2.0, but that caused some crashes because of a bug in legacy cookie decoding, so I really had no other option than to update to 3.2.1. So I temporarily monkey patched django.core.files.utils.validate_file_name to just return name, and everything works perfectly. Regarding the FileField.save method and the parameter it takes, to me it kind of makes sense to only pass in the basename of the file. I'm not completely sure if we should be passing in a path in the case of pre_save. It doesn't make sense to me to derive the final path of the file from that. The full path should be generated in a custom upload_to, and that parameter should only be used to e.g. reuse the same extension of the original file. But not to append that full path to upload_to. Note I'm talking only about this special case with pre_save, where files set to the model fields are handled. It would still be possible to manually call field.save('some/path/file.png. Although I'm not sure why someone would do that, because the path should be provided in upload_to I think. But I know we also have to think about backwards compatibility, so I'm not quite sure what is a viable solution here. Imagine the following scenario (assuming the FileField has upload_to='media'): model.file_field = File(open('/folder/file.png', 'rb')) model.save() Should the file be written to media/folder/file.png? I don't think the final path in MEDIA_ROOT should depend on the original path of the file. And we can't even provide File(open(), name='file.png') manually, because it would break File.open() where the name is being used. If you want to provide a custom path during save, you can still do that by manually calling file_field.save(path, but in this special case of pre_save, where the name is the full path, I think it would make sense to only use the basename. I'm not completely sure what Django does by default when upload_to is a path (string), as I'm using a callable where the only thing I take from the filename is the extension, which I reuse. So both ways will work for me. Sorry, I was mistaken. It wouldn't be possible to even do it manually using file_field.save(path, because the check if made inside the FileField.save method. But it still kind of makes sense to me for the FileField.save method to only take the basename. But I'm not sure everyone would agree, and it would be a backwards incompatible change. And even if we allow FileField.save to take the full path, shouldn't we still be calling it with just the basename from the pre_save method?
Replying to carderm: Replying to Florian Apolloner: Replying to Mariusz Felisiak: The core issue was the addition of an unneccessary check added here: ​https://github.com/django/django/blob/main/django/core/files/utils.py#L7 def validate_file_name(name): if name != os.path.basename(name): raise SuspiciousFileOperation("File name '%s' includes path elements" % name) These two lines broke our application which uses the 2.2 LTS. Our application is shipped, so that's thousands of installations out there all with various Y versions of our software over the past few years. Even if we can port our application to use the new name=... argument, we would have to backport and release many older versions, or force our userbase to upgrade. I agree with @cardem; I see the impact of these two lines as huge. Can someone identify what is the motivation for having these two lines? I don't see how they are required for the CVE, but even if they are, since the CVE is graded as low, is this appropriate for the LTS? One of my other concerns is that what I see happening is folks are pinning to 2.2.20, which is going to prevent them from receiving the moderate CVE fix for Python 3.9 environments with 2.2.22. Thank you for all the effort you all put into Django. We couldn't do what we do without you.
First off, can we please stop with arguments like "this will break thousand installations". It is simply not true because I do not assume that those installations will automatically update. And bumping the Django dependency should result in your tests discovering the issue. If not, well we are in the same boat, Django didn't catch it either even though our test coverage is not that bad… Replying to Brian Bouterse: These two lines broke our application which uses the 2.2 LTS. Our application is shipped, so that's thousands of installations out there all with various Y versions of our software over the past few years. Even if we can port our application to use the new name=... argument, we would have to backport and release many older versions, or force our userbase to upgrade. I agree with @cardem; I see the impact of these two lines as huge. I fully understand that this change might have impacts for you. That said, assume this was a different security fix, all your users would have to upgrade anyways; how is it much different whether the upgrade just Django or your app + Django? Please be aware that literally every point release has the possibility to break your app because you might have used some undocumented feature. I am trying to understand how your normal upgrade process looks like. Can someone identify what is the motivation for having these two lines? I don't see how they are required for the CVE, but even if they are, since the CVE is graded as low, is this appropriate for the LTS? It is an in-depth defense against path traversal issues inside MEDIA_ROOT. The CVE could have been worded differently, sorry about that. The fallout seems to be larger than expected. That said, please be aware that we do develop security fixes in private (in a rather small group) to not allow early exploits. As such issues & mistakes like this are more likely to occur with security fixes than with regular fixes (where there is at least the chance of a larger review). As for whether or not this is appropriate for a LTS. The point of a LTS is exactly to get those security fixes (and a few normal high profile issues). Depending on how exactly your application works, this low issue could just as well be a high for you (although many things have to go wrong for that). One of my other concerns is that what I see happening is folks are pinning to 2.2.20, which is going to prevent them from receiving the moderate CVE fix for Python 3.9 environments with 2.2.22. That is fine, as with all things, we do recommend pinning. If one of the versions causes problems there is always the possibility to maintain your own forks as needed or monkeypatch around issue. And yes I really think people should have this as a last resort option because we simply cannot be perfect and will fail from time to time. Rushing out a fix for this to break yet another valid usecase will make things just worse. Thank you for all the effort you all put into Django. We couldn't do what we do without you. Thank you for the kind words. The are highly appreciated especially in heated tickets like this one. As said above already I am having a difficulty in reproducing all the mentioned problems. Some of the issues mentioned (especially absolute path) occur for me even on unpatched Django versions and I am yet trying to understand what the difference between our systems is. The more we know the faster we can fix this.
Replying to Florian Apolloner: Replying to Jakub Kleň: Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself. This is true, but according to my tests on earlier versions a full (absolute) path did fail already because it would usually be outside the MEDIA_ROOT. Can you provide some more details? Where exactly is it crashing for you Florian with absolute path when you comment out the checks in validate_file_name (3.2.1)? Maybe I could look into the code to search for some difference I might have in my setup. A stack trace would be awesome.
I think I see why some installations broke with this but other's haven't. Look at this reproducer: from django.core.files import File from django.db import models class MyModelWithUploadTo(models.Model): def upload_to_func(self, name): return 'qqq' file = models.FileField(upload_to=upload_to_func) class MyModel(models.Model): file = models.FileField() with open('/tmp/somefile', 'w') as f: f.write('asdfasdfasdf') f.flush() with open('/tmp/anotherfile', 'w') as f: f.write('fsfoisufsiofdsoiufd') f.flush() model_instance_with_upload_to = MyModelWithUploadTo() model_instance_with_upload_to.file = File(open('/tmp/somefile', 'rb')) model_instance_with_upload_to.save() print('I saved the one with upload_to()\n\n') model_instance = MyModel() model_instance.file = File(open('/tmp/anotherfile', 'rb')) model_instance.save() On 2.2.20 when I makemigrations for those two models, apply them, and the run the reproducer I get: I saved the one with upload_to() Traceback (most recent call last): File "/home/vagrant/devel/django_file_name_2_2_21_reproducer.py", line 30, in <module> model_instance.save() File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 743, in save self.save_base(using=using, force_insert=force_insert, File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 780, in save_base updated = self._save_table( File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 873, in _save_table result = self._do_insert(cls._base_manager, using, fields, update_pk, raw) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 910, in _do_insert return manager._insert([self], fields=fields, return_id=update_pk, File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/manager.py", line 82, in manager_method return getattr(self.get_queryset(), name)(*args, **kwargs) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/query.py", line 1186, in _insert return query.get_compiler(using=using).execute_sql(return_id) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1376, in execute_sql for sql, params in self.as_sql(): File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1318, in as_sql value_rows = [ File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1319, in <listcomp> [self.prepare_value(field, self.pre_save_val(field, obj)) for field in fields] File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1319, in <listcomp> [self.prepare_value(field, self.pre_save_val(field, obj)) for field in fields] File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1270, in pre_save_val return field.pre_save(obj, add=True) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/fields/files.py", line 288, in pre_save file.save(file.name, file.file, save=False) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/fields/files.py", line 87, in save self.name = self.storage.save(name, content, max_length=self.field.max_length) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/core/files/storage.py", line 52, in save return self._save(name, content) File "/home/vagrant/devel/pulpcore/pulpcore/app/models/storage.py", line 46, in _save full_path = self.path(name) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/core/files/storage.py", line 323, in path return safe_join(self.location, name) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/utils/_os.py", line 44, in safe_join raise SuspiciousFileOperation( django.core.exceptions.SuspiciousFileOperation: The joined path (/tmp/anotherfile) is located outside of the base path component (/var/lib/pulp/media) Notice how the output includes the I saved the one with upload_to(). That shows this was working with 2.2.20. Now run that same reproducer, migrations, etc against 2.2.21. I see: Traceback (most recent call last): File "/home/vagrant/devel/django_file_name_2_2_21_reproducer.py", line 25, in <module> model_instance_with_upload_to.save() File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 743, in save self.save_base(using=using, force_insert=force_insert, File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 780, in save_base updated = self._save_table( File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 873, in _save_table result = self._do_insert(cls._base_manager, using, fields, update_pk, raw) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/base.py", line 910, in _do_insert return manager._insert([self], fields=fields, return_id=update_pk, File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/manager.py", line 82, in manager_method return getattr(self.get_queryset(), name)(*args, **kwargs) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/query.py", line 1186, in _insert return query.get_compiler(using=using).execute_sql(return_id) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1376, in execute_sql for sql, params in self.as_sql(): File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1318, in as_sql value_rows = [ File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1319, in <listcomp> [self.prepare_value(field, self.pre_save_val(field, obj)) for field in fields] File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1319, in <listcomp> [self.prepare_value(field, self.pre_save_val(field, obj)) for field in fields] File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/sql/compiler.py", line 1270, in pre_save_val return field.pre_save(obj, add=True) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/fields/files.py", line 289, in pre_save file.save(file.name, file.file, save=False) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/fields/files.py", line 87, in save name = self.field.generate_filename(self.instance, name) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/db/models/fields/files.py", line 303, in generate_filename filename = validate_file_name(filename) File "/usr/local/lib/pulp/lib64/python3.9/site-packages/django/core/files/utils.py", line 8, in validate_file_name raise SuspiciousFileOperation("File name '%s' includes path elements" % name) django.core.exceptions.SuspiciousFileOperation: File name '/tmp/somefile' includes path elements Here the SuspiciousFileOperation is also raised on the saving of MyModelWithUploadTo. Do you all know why this difference is significant? Regarding non reproducer discussion... Yes let's take how many folks are broken out of the equation; we have to do what's secure, even if its inconvenient. I agree with that, and thank you for saying that. I think what is contributing to the issue is confusion on why these 2 lines are necessary given how the CVE reads. I'm not here to second guess the good work of the security response team or it's review process, but I am confused. As a practical matter, I'm still trying to figure out if another release removing these two lines will occur, or if the CVE description needs revision. Do you have some advice for me on which you think will happen? Also I'd like to exchange perspectives on version pinning for shipped django apps depending on the Django LTS, but not until we resolve the matter at hand first. Thank you for everything. Our project and users really appreciate it. Please let me know how we can help.
Replying to Florian Apolloner: Replying to Jakub Kleň: Correct me if I'm wrong, but file-like objects always contain the full path to the file in the name attribute (the built-in Django File class even uses it to reopen the file if it was closed), and so it seems to be a bug in Django itself. This is true, but according to my tests on earlier versions a full (absolute) path did fail already because it would usually be outside the MEDIA_ROOT. Can you provide some more details? I believe an absolute path outside of Media Root should fail, that is correct (in my opinion) - as an absolute path might be able to write shell scripts around the file system... It should however, work for a relative file path. Hack example here # Some model class FileModel(models.Model): file_field = models.FileField(_('UsersFile'), storage='users', max_length=1000) model_inst = FileModel() # Add some file new_file = ContentFile(some_file.read()) new_file.name = f"{user.name}/file.txt" # E.g. "steve/file.txt" <<< This file should only be for 'steve' model_inst.file = new_file # Save the Model model_inst.save() #<<< in Django 3.2.0 this creates file "MEDIA_ROOT/users/steve/file.txt ! which is correct model_inst.save() #<<< in Django 3.2.1 this FAILS with Path issue ! Incorrect model_inst.save() #<<< in Fix above [https://github.com/django/django/pull/14354/commits/1c78e83791163b034a7f1689673bff02f9969368 commit] will create "MEDIA_ROOT/users/file.txt ! which is incorrect - missing path! # What we need to stop: bad_file = ContentFile(malicious_script.read()) bad_file.name = "/etc/init.d/nameofscript.sh" # or bad_file.name = "../../../../../usr/local/lib/bad.file" model_inst.file = bad_file # On Save we need to protect here - imho model_inst.save() # <<< This *should Fail* with Security Error
Replying to Florian Apolloner: Can you please provide a reproducer and full traceback? I hope this isn't too overboard, but I went ahead and copied out a chunk of my production code into a new project. please look at the readme to see my full comments and traceback: ​https://github.com/wizpig64/django_32718_repro disclaimer: i don't use upload_to strings, just upload_to functions, so i dont know how to account for those users. thanks for your time and your hard work.
Replying to Brian Bouterse: Here the SuspiciousFileOperation is also raised on the saving of MyModelWithUploadTo. Do you all know why this difference is significant? This is perfect! The difference can be significant due to this: ​https://github.com/django/django/blob/c4ee3b208a2c95a5102b5e4fa789b10f8ee29b84/django/db/models/fields/files.py#L309-L322 -- This means when upload_to is set it is supposed to return the final filename (including a path relative path). Since you are just returning 'qqq' there it will be valid in 2.2.20. 2.2.21 validates the name beforehand and will break that. This really helps. Phillip had similar code (ie upload_to in a function as well). As a practical matter, I'm still trying to figure out if another release removing these two lines will occur, or if the CVE description needs revision. Do you have some advice for me on which you think will happen? There will be another release fixing this and the CVE will probably get adjusted to drop the sentence "Specifically, empty file names and paths with dot segments will be rejected.". Does the wording make more sense for you then? Thank you for everything. Our project and users really appreciate it. Please let me know how we can help. Will do, testing will certainly help once we have a suitable PR :)
Replying to carderm: model_inst.save() #<<< in Django Latest Git commit will create "MEDIA_ROOT/users/file.txt ! which is incorrect - missing path! I cannot reproduce this, this fails also with SuspiciousFileOperation: File name 'steve/file.txt' includes path elements for me. Please provide an actual reproducer.
Replying to Phillip Marshall: please look at the readme to see my full comments and traceback: ​https://github.com/wizpig64/django_32718_repro disclaimer: i don't use upload_to strings, just upload_to functions, so i dont know how to account for those users. Perfect, this explains a lot! Your upload_to basically ignores the passed filename (aside from the extension). Now we have a check before you even get the filename which ensures that there is no full path.
Replying to carderm: I believe an absolute path outside of Media Root should fail, that is correct (in my opinion) - as an absolute path might be able to write shell scripts around the file system... It should however, work for a relative file path. Absolute paths wouldn't be able to write shell scripts around the file system, because the path always starts with MEDIA_ROOT, and ../ and such are disallowed. Also, I'm not sure if we should be forced to override the name attribute of File, because overriding it breaks the File.open function from working properly.
I'm still not completely sure if we should disallow absolute paths in the File. When I'm for example setting a file from /tmp, which I'm doing in my project, it would force me to override the File.name like this: file = File(open('/tmp/image.png', 'rb'), name='image.png') # which would then break: file.close() file.open() I know that the open method is not called when using the File to update a model, but it doesn't seem to be the right thing to do. Should we be using the path from File.name and append it to upload_to? In my project, I use a callable upload_to, which takes care of the path and filename, and only takes the extension of the original filename. Isn't that a better solution if we want a custom path that depends on the model instance? def custom_upload_to(instance, filename): extension = os.path.splitext(filename)[1][1:].lower() actual_filename = 'image.' + extension return os.path.join('users', instance.username, actual_filename) model.file = File(open('/tmp/sth.png', 'rb')) My point here is, shouldn't we be using upload_to for the purpose of constructing a custom file path?
Thanks y'all for various test projects and detailed reports of encountered issue with saving FileFields. After some discussions, we decided to prepare a patch for ​FileField.generate_filename() that should solve most of these issues. Please see the following points. If filename passed to the FileField.generate_filename() is an absolute path, it will be converted to the os.path.basename(filename). Validate filename returned by FileField.upload_to() not a filename passed to the FileField.generate_filename() (upload_to() may completely ignored passed filename). Allow relative paths (without dot segments) in the generated filename. We're going to prepare a patch in the next few days.
Replying to Mariusz Felisiak: Thanks y'all for various test projects and detailed reports of encountered issue with saving FileFields. After some discussions, we decided to prepare a patch for ​FileField.generate_filename() that should solve most of these issues. Please see the following points. If filename passed to the FileField.generate_filename() is an absolute path, it will be converted to the os.path.basename(filename). Validate filename returned by FileField.upload_to() not a filename passed to the FileField.generate_filename() (upload_to() may completely ignored passed filename). Allow relative paths (without dot segments) in the generated filename. We're going to prepare a patch in the next few days. Thanks a lot Mariusz! I was thinking about this for a while now, and it seems like a great solution which will both be backwards compatible and will also fit the use cases for all of us! It also takes my concerns into account, which I'm super happy about! Thanks a lot for the great work all of you guys are doing for Django. Love the framework!
We're going to prepare a patch in the next few days. The sooner the better! :) An Open edX release is hinging on this. We've always loved Django's stability and the trust we've had in the security patches.
Replying to Florian Apolloner: There will be another release fixing this and the CVE will probably get adjusted to drop the sentence "Specifically, empty file names and paths with dot segments will be rejected.". Does the wording make more sense for you then? It does! Thank you so much for making this clearer. Will do, testing will certainly help once we have a suitable PR :) I can do that. The description of the plan in Comment 29 sounds great!
```

## Patch

```diff
diff --git a/django/core/files/utils.py b/django/core/files/utils.py
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -1,16 +1,26 @@
 import os
+import pathlib
 
 from django.core.exceptions import SuspiciousFileOperation
 
 
-def validate_file_name(name):
-    if name != os.path.basename(name):
-        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
-
+def validate_file_name(name, allow_relative_path=False):
     # Remove potentially dangerous names
-    if name in {'', '.', '..'}:
+    if os.path.basename(name) in {'', '.', '..'}:
         raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
 
+    if allow_relative_path:
+        # Use PurePosixPath() because this branch is checked only in
+        # FileField.generate_filename() where all file paths are expected to be
+        # Unix style (with forward slashes).
+        path = pathlib.PurePosixPath(name)
+        if path.is_absolute() or '..' in path.parts:
+            raise SuspiciousFileOperation(
+                "Detected path traversal attempt in '%s'" % name
+            )
+    elif name != os.path.basename(name):
+        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
+
     return name
 
 
diff --git a/django/db/models/fields/files.py b/django/db/models/fields/files.py
--- a/django/db/models/fields/files.py
+++ b/django/db/models/fields/files.py
@@ -313,12 +313,12 @@ def generate_filename(self, instance, filename):
         Until the storage layer, all file paths are expected to be Unix style
         (with forward slashes).
         """
-        filename = validate_file_name(filename)
         if callable(self.upload_to):
             filename = self.upload_to(instance, filename)
         else:
             dirname = datetime.datetime.now().strftime(str(self.upload_to))
             filename = posixpath.join(dirname, filename)
+        filename = validate_file_name(filename, allow_relative_path=True)
         return self.storage.generate_filename(filename)
 
     def save_form_data(self, instance, data):

```

## Test Patch

```diff
diff --git a/tests/file_storage/test_generate_filename.py b/tests/file_storage/test_generate_filename.py
--- a/tests/file_storage/test_generate_filename.py
+++ b/tests/file_storage/test_generate_filename.py
@@ -1,6 +1,4 @@
 import os
-import sys
-from unittest import skipIf
 
 from django.core.exceptions import SuspiciousFileOperation
 from django.core.files.base import ContentFile
@@ -64,19 +62,37 @@ def test_storage_dangerous_paths_dir_name(self):
             s.generate_filename(file_name)
 
     def test_filefield_dangerous_filename(self):
-        candidates = ['..', '.', '', '???', '$.$.$']
+        candidates = [
+            ('..', 'some/folder/..'),
+            ('.', 'some/folder/.'),
+            ('', 'some/folder/'),
+            ('???', '???'),
+            ('$.$.$', '$.$.$'),
+        ]
         f = FileField(upload_to='some/folder/')
-        msg = "Could not derive file name from '%s'"
-        for file_name in candidates:
+        for file_name, msg_file_name in candidates:
+            msg = f"Could not derive file name from '{msg_file_name}'"
             with self.subTest(file_name=file_name):
-                with self.assertRaisesMessage(SuspiciousFileOperation, msg % file_name):
+                with self.assertRaisesMessage(SuspiciousFileOperation, msg):
                     f.generate_filename(None, file_name)
 
-    def test_filefield_dangerous_filename_dir(self):
+    def test_filefield_dangerous_filename_dot_segments(self):
         f = FileField(upload_to='some/folder/')
-        msg = "File name '/tmp/path' includes path elements"
+        msg = "Detected path traversal attempt in 'some/folder/../path'"
         with self.assertRaisesMessage(SuspiciousFileOperation, msg):
-            f.generate_filename(None, '/tmp/path')
+            f.generate_filename(None, '../path')
+
+    def test_filefield_generate_filename_absolute_path(self):
+        f = FileField(upload_to='some/folder/')
+        candidates = [
+            '/tmp/path',
+            '/tmp/../path',
+        ]
+        for file_name in candidates:
+            msg = f"Detected path traversal attempt in '{file_name}'"
+            with self.subTest(file_name=file_name):
+                with self.assertRaisesMessage(SuspiciousFileOperation, msg):
+                    f.generate_filename(None, file_name)
 
     def test_filefield_generate_filename(self):
         f = FileField(upload_to='some/folder/')
@@ -95,7 +111,57 @@ def upload_to(instance, filename):
             os.path.normpath('some/folder/test_with_space.txt')
         )
 
-    @skipIf(sys.platform == 'win32', 'Path components in filename are not supported after 0b79eb3.')
+    def test_filefield_generate_filename_upload_to_overrides_dangerous_filename(self):
+        def upload_to(instance, filename):
+            return 'test.txt'
+
+        f = FileField(upload_to=upload_to)
+        candidates = [
+            '/tmp/.',
+            '/tmp/..',
+            '/tmp/../path',
+            '/tmp/path',
+            'some/folder/',
+            'some/folder/.',
+            'some/folder/..',
+            'some/folder/???',
+            'some/folder/$.$.$',
+            'some/../test.txt',
+            '',
+        ]
+        for file_name in candidates:
+            with self.subTest(file_name=file_name):
+                self.assertEqual(f.generate_filename(None, file_name), 'test.txt')
+
+    def test_filefield_generate_filename_upload_to_absolute_path(self):
+        def upload_to(instance, filename):
+            return '/tmp/' + filename
+
+        f = FileField(upload_to=upload_to)
+        candidates = [
+            'path',
+            '../path',
+            '???',
+            '$.$.$',
+        ]
+        for file_name in candidates:
+            msg = f"Detected path traversal attempt in '/tmp/{file_name}'"
+            with self.subTest(file_name=file_name):
+                with self.assertRaisesMessage(SuspiciousFileOperation, msg):
+                    f.generate_filename(None, file_name)
+
+    def test_filefield_generate_filename_upload_to_dangerous_filename(self):
+        def upload_to(instance, filename):
+            return '/tmp/' + filename
+
+        f = FileField(upload_to=upload_to)
+        candidates = ['..', '.', '']
+        for file_name in candidates:
+            msg = f"Could not derive file name from '/tmp/{file_name}'"
+            with self.subTest(file_name=file_name):
+                with self.assertRaisesMessage(SuspiciousFileOperation, msg):
+                    f.generate_filename(None, file_name)
+
     def test_filefield_awss3_storage(self):
         """
         Simulate a FileField with an S3 storage which uses keys rather than
diff --git a/tests/model_fields/test_filefield.py b/tests/model_fields/test_filefield.py
--- a/tests/model_fields/test_filefield.py
+++ b/tests/model_fields/test_filefield.py
@@ -5,6 +5,7 @@
 import unittest
 from pathlib import Path
 
+from django.core.exceptions import SuspiciousFileOperation
 from django.core.files import File, temp
 from django.core.files.base import ContentFile
 from django.core.files.uploadedfile import TemporaryUploadedFile
@@ -63,6 +64,15 @@ def test_refresh_from_db(self):
         d.refresh_from_db()
         self.assertIs(d.myfile.instance, d)
 
+    @unittest.skipIf(sys.platform == 'win32', "Crashes with OSError on Windows.")
+    def test_save_without_name(self):
+        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
+            document = Document.objects.create(myfile='something.txt')
+            document.myfile = File(tmp)
+            msg = f"Detected path traversal attempt in '{tmp.name}'"
+            with self.assertRaisesMessage(SuspiciousFileOperation, msg):
+                document.save()
+
     def test_defer(self):
         Document.objects.create(myfile='something.txt')
         self.assertEqual(Document.objects.defer('myfile')[0].myfile, 'something.txt')

```


## Code snippets

### 1 - django/db/models/fields/files.py:

Start line: 1, End line: 142

```python
import datetime
import posixpath

from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.utils.translation import gettext_lazy as _


class FieldFile(File):
    def __init__(self, instance, field, name):
        super().__init__(None, name)
        self.instance = instance
        self.field = field
        self.storage = field.storage
        self._committed = True

    def __eq__(self, other):
        # Older code may be expecting FileField values to be simple strings.
        # By overriding the == operator, it can remain backwards compatibility.
        if hasattr(other, 'name'):
            return self.name == other.name
        return self.name == other

    def __hash__(self):
        return hash(self.name)

    # The standard File contains most of the necessary properties, but
    # FieldFiles can be instantiated without a name, so that needs to
    # be checked for here.

    def _require_file(self):
        if not self:
            raise ValueError("The '%s' attribute has no file associated with it." % self.field.name)

    def _get_file(self):
        self._require_file()
        if getattr(self, '_file', None) is None:
            self._file = self.storage.open(self.name, 'rb')
        return self._file

    def _set_file(self, file):
        self._file = file

    def _del_file(self):
        del self._file

    file = property(_get_file, _set_file, _del_file)

    @property
    def path(self):
        self._require_file()
        return self.storage.path(self.name)

    @property
    def url(self):
        self._require_file()
        return self.storage.url(self.name)

    @property
    def size(self):
        self._require_file()
        if not self._committed:
            return self.file.size
        return self.storage.size(self.name)

    def open(self, mode='rb'):
        self._require_file()
        if getattr(self, '_file', None) is None:
            self.file = self.storage.open(self.name, mode)
        else:
            self.file.open(mode)
        return self
    # open() doesn't alter the file's contents, but it does reset the pointer
    open.alters_data = True

    # In addition to the standard File API, FieldFiles have extra methods
    # to further manipulate the underlying file, as well as update the
    # associated model instance.

    def save(self, name, content, save=True):
        name = self.field.generate_filename(self.instance, name)
        self.name = self.storage.save(name, content, max_length=self.field.max_length)
        setattr(self.instance, self.field.attname, self.name)
        self._committed = True

        # Save the object because it has changed, unless save is False
        if save:
            self.instance.save()
    save.alters_data = True

    def delete(self, save=True):
        if not self:
            return
        # Only close the file if it's already open, which we know by the
        # presence of self._file
        if hasattr(self, '_file'):
            self.close()
            del self.file

        self.storage.delete(self.name)

        self.name = None
        setattr(self.instance, self.field.attname, self.name)
        self._committed = False

        if save:
            self.instance.save()
    delete.alters_data = True

    @property
    def closed(self):
        file = getattr(self, '_file', None)
        return file is None or file.closed

    def close(self):
        file = getattr(self, '_file', None)
        if file is not None:
            file.close()

    def __getstate__(self):
        # FieldFile needs access to its associated model field, an instance and
        # the file's name. Everything else will be restored later, by
        # FileDescriptor below.
        return {
            'name': self.name,
            'closed': False,
            '_committed': True,
            '_file': None,
            'instance': self.instance,
            'field': self.field,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.storage = self.field.storage
```
### 2 - django/db/models/fields/files.py:

Start line: 217, End line: 339

```python
class FileField(Field):

    # The class to wrap instance attributes in. Accessing the file object off
    # the instance will always return an instance of attr_class.
    attr_class = FieldFile

    # The descriptor to use for accessing the attribute off of the class.
    descriptor_class = FileDescriptor

    description = _("File")

    def __init__(self, verbose_name=None, name=None, upload_to='', storage=None, **kwargs):
        self._primary_key_set_explicitly = 'primary_key' in kwargs

        self.storage = storage or default_storage
        if callable(self.storage):
            # Hold a reference to the callable for deconstruct().
            self._storage_callable = self.storage
            self.storage = self.storage()
            if not isinstance(self.storage, Storage):
                raise TypeError(
                    "%s.storage must be a subclass/instance of %s.%s"
                    % (self.__class__.__qualname__, Storage.__module__, Storage.__qualname__)
                )
        self.upload_to = upload_to

        kwargs.setdefault('max_length', 100)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_primary_key(),
            *self._check_upload_to(),
        ]

    def _check_primary_key(self):
        if self._primary_key_set_explicitly:
            return [
                checks.Error(
                    "'primary_key' is not a valid argument for a %s." % self.__class__.__name__,
                    obj=self,
                    id='fields.E201',
                )
            ]
        else:
            return []

    def _check_upload_to(self):
        if isinstance(self.upload_to, str) and self.upload_to.startswith('/'):
            return [
                checks.Error(
                    "%s's 'upload_to' argument must be a relative path, not an "
                    "absolute path." % self.__class__.__name__,
                    obj=self,
                    id='fields.E202',
                    hint='Remove the leading slash.',
                )
            ]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if kwargs.get("max_length") == 100:
            del kwargs["max_length"]
        kwargs['upload_to'] = self.upload_to
        if self.storage is not default_storage:
            kwargs['storage'] = getattr(self, '_storage_callable', self.storage)
        return name, path, args, kwargs

    def get_internal_type(self):
        return "FileField"

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        # Need to convert File objects provided via a form to string for database insertion
        if value is None:
            return None
        return str(value)

    def pre_save(self, model_instance, add):
        file = super().pre_save(model_instance, add)
        if file and not file._committed:
            # Commit the file to storage prior to saving the model
            file.save(file.name, file.file, save=False)
        return file

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        setattr(cls, self.attname, self.descriptor_class(self))

    def generate_filename(self, instance, filename):
        """
        Apply (if callable) or prepend (if a string) upload_to to the filename,
        then delegate further processing of the name to the storage backend.
        Until the storage layer, all file paths are expected to be Unix style
        (with forward slashes).
        """
        filename = validate_file_name(filename)
        if callable(self.upload_to):
            filename = self.upload_to(instance, filename)
        else:
            dirname = datetime.datetime.now().strftime(str(self.upload_to))
            filename = posixpath.join(dirname, filename)
        return self.storage.generate_filename(filename)

    def save_form_data(self, instance, data):
        # Important: None means "no change", other false value means "clear"
        # This subtle distinction (rather than a more explicit marker) is
        # needed because we need to consume values that are also sane for a
        # regular (non Model-) Form to find in its cleaned_data dictionary.
        if data is not None:
            # This value will be converted to str and stored in the
            # database, so leaving False as-is is not acceptable.
            setattr(instance, self.name, data or '')

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.FileField,
            'max_length': self.max_length,
            **kwargs,
        })
```
### 3 - django/db/migrations/operations/fields.py:

Start line: 1, End line: 37

```python
from django.core.exceptions import FieldDoesNotExist
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property

from .base import Operation
from .utils import field_is_referenced, field_references, get_references


class FieldOperation(Operation):
    def __init__(self, model_name, name, field=None):
        self.model_name = model_name
        self.name = name
        self.field = field

    @cached_property
    def model_name_lower(self):
        return self.model_name.lower()

    @cached_property
    def name_lower(self):
        return self.name.lower()

    def is_same_model_operation(self, operation):
        return self.model_name_lower == operation.model_name_lower

    def is_same_field_operation(self, operation):
        return self.is_same_model_operation(operation) and self.name_lower == operation.name_lower

    def references_model(self, name, app_label):
        name_lower = name.lower()
        if name_lower == self.model_name_lower:
            return True
        if self.field:
            return bool(field_references(
                (app_label, self.model_name_lower), self.field, (app_label, name_lower)
            ))
        return False
```
### 4 - django/core/files/uploadedfile.py:

Start line: 39, End line: 55

```python
class UploadedFile(File):

    def _set_name(self, name):
        # Sanitize the file name so that it can't be dangerous.
        if name is not None:
            # Just use the basename of the file -- anything else is dangerous.
            name = os.path.basename(name)

            # File names longer than 255 characters can cause problems on older OSes.
            if len(name) > 255:
                name, ext = os.path.splitext(name)
                ext = ext[:255]
                name = name[:255 - len(ext)] + ext

            name = validate_file_name(name)

        self._name = name

    name = property(_get_name, _set_name)
```
### 5 - django/core/files/storage.py:

Start line: 240, End line: 301

```python
@deconstructible
class FileSystemStorage(Storage):

    def _save(self, name, content):
        full_path = self.path(name)

        # Create any intermediate directories that do not exist.
        directory = os.path.dirname(full_path)
        try:
            if self.directory_permissions_mode is not None:
                # Set the umask because os.makedirs() doesn't apply the "mode"
                # argument to intermediate-level directories.
                old_umask = os.umask(0o777 & ~self.directory_permissions_mode)
                try:
                    os.makedirs(directory, self.directory_permissions_mode, exist_ok=True)
                finally:
                    os.umask(old_umask)
            else:
                os.makedirs(directory, exist_ok=True)
        except FileExistsError:
            raise FileExistsError('%s exists and is not a directory.' % directory)

        # There's a potential race condition between get_available_name and
        # saving the file; it's possible that two threads might return the
        # same name, at which point all sorts of fun happens. So we need to
        # try to create the file, but if it already exists we have to go back
        # to get_available_name() and try again.

        while True:
            try:
                # This file has a file path that we can move.
                if hasattr(content, 'temporary_file_path'):
                    file_move_safe(content.temporary_file_path(), full_path)

                # This is a normal uploadedfile that we can stream.
                else:
                    # The current umask value is masked out by os.open!
                    fd = os.open(full_path, self.OS_OPEN_FLAGS, 0o666)
                    _file = None
                    try:
                        locks.lock(fd, locks.LOCK_EX)
                        for chunk in content.chunks():
                            if _file is None:
                                mode = 'wb' if isinstance(chunk, bytes) else 'wt'
                                _file = os.fdopen(fd, mode)
                            _file.write(chunk)
                    finally:
                        locks.unlock(fd)
                        if _file is not None:
                            _file.close()
                        else:
                            os.close(fd)
            except FileExistsError:
                # A new name is needed if the file exists.
                name = self.get_available_name(name)
                full_path = self.path(name)
            else:
                # OK, the file save worked. Break out of the loop.
                break

        if self.file_permissions_mode is not None:
            os.chmod(full_path, self.file_permissions_mode)

        # Store filenames with forward slashes, even on Windows.
        return str(name).replace('\\', '/')
```
### 6 - django/db/models/base.py:

Start line: 915, End line: 947

```python
class Model(metaclass=ModelBase):

    def _prepare_related_fields_for_save(self, operation_name):
        # Ensure that a model instance without a PK hasn't been assigned to
        # a ForeignKey or OneToOneField on this model. If the field is
        # nullable, allowing the save would result in silent data loss.
        for field in self._meta.concrete_fields:
            # If the related field isn't cached, then an instance hasn't been
            # assigned and there's no need to worry about this check.
            if field.is_relation and field.is_cached(self):
                obj = getattr(self, field.name, None)
                if not obj:
                    continue
                # A pk may have been assigned manually to a model instance not
                # saved to the database (or auto-generated in a case like
                # UUIDField), but we allow the save to proceed and rely on the
                # database to raise an IntegrityError if applicable. If
                # constraints aren't supported by the database, there's the
                # unavoidable risk of data corruption.
                if obj.pk is None:
                    # Remove the object from a related instance cache.
                    if not field.remote_field.multiple:
                        field.remote_field.delete_cached_value(obj)
                    raise ValueError(
                        "%s() prohibited to prevent data loss due to unsaved "
                        "related object '%s'." % (operation_name, field.name)
                    )
                elif getattr(self, field.attname) in field.empty_values:
                    # Use pk from related object if it has been saved after
                    # an assignment.
                    setattr(self, field.attname, obj.pk)
                # If the relationship's pk/to_field was changed, clear the
                # cached relationship.
                if getattr(obj, field.target_field.attname) != getattr(self, field.attname):
                    field.delete_cached_value(self)
```
### 7 - django/core/files/base.py:

Start line: 1, End line: 29

```python
import os
from io import BytesIO, StringIO, UnsupportedOperation

from django.core.files.utils import FileProxyMixin
from django.utils.functional import cached_property


class File(FileProxyMixin):
    DEFAULT_CHUNK_SIZE = 64 * 2 ** 10

    def __init__(self, file, name=None):
        self.file = file
        if name is None:
            name = getattr(file, 'name', None)
        self.name = name
        if hasattr(file, 'mode'):
            self.mode = file.mode

    def __str__(self):
        return self.name or ''

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self or "None")

    def __bool__(self):
        return bool(self.name)

    def __len__(self):
        return self.size
```
### 8 - django/db/models/fields/__init__.py:

Start line: 1660, End line: 1685

```python
class FilePathField(Field):
    description = _("File path")

    def __init__(self, verbose_name=None, name=None, path='', match=None,
                 recursive=False, allow_files=True, allow_folders=False, **kwargs):
        self.path, self.match, self.recursive = path, match, recursive
        self.allow_files, self.allow_folders = allow_files, allow_folders
        kwargs.setdefault('max_length', 100)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_allowing_files_or_folders(**kwargs),
        ]

    def _check_allowing_files_or_folders(self, **kwargs):
        if not self.allow_files and not self.allow_folders:
            return [
                checks.Error(
                    "FilePathFields must have either 'allow_files' or 'allow_folders' set to True.",
                    obj=self,
                    id='fields.E140',
                )
            ]
        return []
```
### 9 - django/db/models/fields/files.py:

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
### 10 - django/forms/fields.py:

Start line: 552, End line: 571

```python
class FileField(Field):

    def to_python(self, data):
        if data in self.empty_values:
            return None

        # UploadedFile objects should have name and size attributes.
        try:
            file_name = data.name
            file_size = data.size
        except AttributeError:
            raise ValidationError(self.error_messages['invalid'], code='invalid')

        if self.max_length is not None and len(file_name) > self.max_length:
            params = {'max': self.max_length, 'length': len(file_name)}
            raise ValidationError(self.error_messages['max_length'], code='max_length', params=params)
        if not file_name:
            raise ValidationError(self.error_messages['invalid'], code='invalid')
        if not self.allow_empty_file and not file_size:
            raise ValidationError(self.error_messages['empty'], code='empty')

        return data
```
### 19 - django/db/models/fields/files.py:

Start line: 145, End line: 214

```python
class FileDescriptor(DeferredAttribute):
    """
    The descriptor for the file attribute on the model instance. Return a
    FieldFile when accessed so you can write code like::

        >>> from myapp.models import MyModel
        >>> instance = MyModel.objects.get(pk=1)
        >>> instance.file.size

    Assign a file object on assignment so you can do::

        >>> with open('/path/to/hello.world') as f:
        ...     instance.file = File(f)
    """
    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        # This is slightly complicated, so worth an explanation.
        # instance.file`needs to ultimately return some instance of `File`,
        # probably a subclass. Additionally, this returned object needs to have
        # the FieldFile API so that users can easily do things like
        # instance.file.path and have that delegated to the file storage engine.
        # Easy enough if we're strict about assignment in __set__, but if you
        # peek below you can see that we're not. So depending on the current
        # value of the field we have to dynamically construct some sort of
        # "thing" to return.

        # The instance dict contains whatever was originally assigned
        # in __set__.
        file = super().__get__(instance, cls)

        # If this value is a string (instance.file = "path/to/file") or None
        # then we simply wrap it with the appropriate attribute class according
        # to the file field. [This is FieldFile for FileFields and
        # ImageFieldFile for ImageFields; it's also conceivable that user
        # subclasses might also want to subclass the attribute class]. This
        # object understands how to convert a path to a file, and also how to
        # handle None.
        if isinstance(file, str) or file is None:
            attr = self.field.attr_class(instance, self.field, file)
            instance.__dict__[self.field.attname] = attr

        # Other types of files may be assigned as well, but they need to have
        # the FieldFile interface added to them. Thus, we wrap any other type of
        # File inside a FieldFile (well, the field's attr_class, which is
        # usually FieldFile).
        elif isinstance(file, File) and not isinstance(file, FieldFile):
            file_copy = self.field.attr_class(instance, self.field, file.name)
            file_copy.file = file
            file_copy._committed = False
            instance.__dict__[self.field.attname] = file_copy

        # Finally, because of the (some would say boneheaded) way pickle works,
        # the underlying FieldFile might not actually itself have an associated
        # file. So we need to reset the details of the FieldFile in those cases.
        elif isinstance(file, FieldFile) and not hasattr(file, 'field'):
            file.instance = instance
            file.field = self.field
            file.storage = self.field.storage

        # Make sure that the instance is correct.
        elif isinstance(file, FieldFile) and instance is not file.instance:
            file.instance = instance

        # That was fun, wasn't it?
        return instance.__dict__[self.field.attname]

    def __set__(self, instance, value):
        instance.__dict__[self.field.attname] = value
```
### 58 - django/db/models/fields/files.py:

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
### 61 - django/db/models/fields/files.py:

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
### 123 - django/core/files/utils.py:

Start line: 1, End line: 69

```python
import os

from django.core.exceptions import SuspiciousFileOperation


def validate_file_name(name):
    if name != os.path.basename(name):
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

    # Remove potentially dangerous names
    if name in {'', '.', '..'}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)

    return name


class FileProxyMixin:
    """
    A mixin class used to forward file methods to an underlaying file
    object.  The internal file object has to be called "file"::

        class FileProxy(FileProxyMixin):
            def __init__(self, file):
                self.file = file
    """

    encoding = property(lambda self: self.file.encoding)
    fileno = property(lambda self: self.file.fileno)
    flush = property(lambda self: self.file.flush)
    isatty = property(lambda self: self.file.isatty)
    newlines = property(lambda self: self.file.newlines)
    read = property(lambda self: self.file.read)
    readinto = property(lambda self: self.file.readinto)
    readline = property(lambda self: self.file.readline)
    readlines = property(lambda self: self.file.readlines)
    seek = property(lambda self: self.file.seek)
    tell = property(lambda self: self.file.tell)
    truncate = property(lambda self: self.file.truncate)
    write = property(lambda self: self.file.write)
    writelines = property(lambda self: self.file.writelines)

    @property
    def closed(self):
        return not self.file or self.file.closed

    def readable(self):
        if self.closed:
            return False
        if hasattr(self.file, 'readable'):
            return self.file.readable()
        return True

    def writable(self):
        if self.closed:
            return False
        if hasattr(self.file, 'writable'):
            return self.file.writable()
        return 'w' in getattr(self.file, 'mode', '')

    def seekable(self):
        if self.closed:
            return False
        if hasattr(self.file, 'seekable'):
            return self.file.seekable()
        return True

    def __iter__(self):
        return iter(self.file)
```
