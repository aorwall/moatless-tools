# django__django-12148

| **django/django** | `c9bf1910e2c1a72244dbd1e3dd9a3ff7215b8b4a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 310 |
| **Any found context length** | 310 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/flatpages/models.py b/django/contrib/flatpages/models.py
--- a/django/contrib/flatpages/models.py
+++ b/django/contrib/flatpages/models.py
@@ -1,6 +1,6 @@
 from django.contrib.sites.models import Site
 from django.db import models
-from django.urls import get_script_prefix
+from django.urls import NoReverseMatch, get_script_prefix, reverse
 from django.utils.encoding import iri_to_uri
 from django.utils.translation import gettext_lazy as _
 
@@ -36,5 +36,12 @@ def __str__(self):
         return "%s -- %s" % (self.url, self.title)
 
     def get_absolute_url(self):
+        from .views import flatpage
+
+        for url in (self.url.lstrip('/'), self.url):
+            try:
+                return reverse(flatpage, kwargs={'url': url})
+            except NoReverseMatch:
+                pass
         # Handle script prefix manually because we bypass reverse()
         return iri_to_uri(get_script_prefix().rstrip('/') + self.url)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/flatpages/models.py | 3 | 3 | 1 | 1 | 310
| django/contrib/flatpages/models.py | 39 | 39 | 1 | 1 | 310


## Problem Statement

```
reverse() and get_absolute_url() may return different values for same FlatPage
Description
	 
		(last modified by Tim Graham)
	 
The FlatPage model implements get_absolute_url() without using reverse(). The comment suggests, that this handles SCRIPT_NAME issues, but the link in the admin interface does not work, if you are using a prefix for the flatpages urls. The templatetag for resolving a flatpage works just fine.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/flatpages/models.py** | 1 | 41| 310 | 310 | 310 | 
| 2 | 2 django/contrib/flatpages/urls.py | 1 | 7| 0 | 310 | 348 | 
| 3 | 3 django/contrib/flatpages/views.py | 1 | 45| 399 | 709 | 938 | 
| 4 | 4 django/contrib/flatpages/forms.py | 1 | 50| 346 | 1055 | 1421 | 
| 5 | 5 django/urls/base.py | 28 | 87| 443 | 1498 | 2607 | 
| 6 | 6 django/urls/resolvers.py | 613 | 686| 682 | 2180 | 8138 | 
| 7 | 6 django/contrib/flatpages/views.py | 48 | 70| 191 | 2371 | 8138 | 
| 8 | 6 django/contrib/flatpages/forms.py | 52 | 70| 143 | 2514 | 8138 | 
| 9 | 7 django/contrib/flatpages/templatetags/flatpages.py | 1 | 42| 300 | 2814 | 8912 | 
| 10 | 8 django/contrib/flatpages/__init__.py | 1 | 2| 0 | 2814 | 8926 | 
| 11 | 9 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 2961 | 9074 | 
| 12 | 10 django/shortcuts.py | 102 | 142| 281 | 3242 | 10172 | 
| 13 | 11 django/contrib/flatpages/admin.py | 1 | 20| 144 | 3386 | 10316 | 
| 14 | 12 django/contrib/flatpages/apps.py | 1 | 8| 0 | 3386 | 10356 | 
| 15 | 12 django/contrib/flatpages/templatetags/flatpages.py | 45 | 100| 473 | 3859 | 10356 | 
| 16 | 13 django/contrib/admin/utils.py | 437 | 465| 225 | 4084 | 14473 | 
| 17 | 13 django/urls/resolvers.py | 601 | 611| 121 | 4205 | 14473 | 
| 18 | 14 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 4610 | 14878 | 
| 19 | 15 django/views/generic/base.py | 188 | 219| 247 | 4857 | 16477 | 
| 20 | 16 django/contrib/admindocs/urls.py | 1 | 51| 307 | 5164 | 16784 | 
| 21 | 17 django/contrib/contenttypes/fields.py | 414 | 429| 124 | 5288 | 22218 | 
| 22 | 18 django/template/defaulttags.py | 1315 | 1379| 504 | 5792 | 33259 | 
| 23 | 19 django/contrib/flatpages/sitemaps.py | 1 | 13| 112 | 5904 | 33371 | 
| 24 | 20 django/contrib/sitemaps/views.py | 1 | 19| 131 | 6035 | 34145 | 
| 25 | 21 django/contrib/sitemaps/__init__.py | 30 | 51| 232 | 6267 | 35397 | 
| 26 | 22 django/contrib/admin/widgets.py | 166 | 197| 243 | 6510 | 39263 | 
| 27 | 22 django/template/defaulttags.py | 421 | 455| 270 | 6780 | 39263 | 
| 28 | 23 django/contrib/admin/options.py | 602 | 624| 280 | 7060 | 57764 | 
| 29 | 24 django/contrib/redirects/admin.py | 1 | 11| 0 | 7060 | 57832 | 
| 30 | 25 django/contrib/gis/views.py | 1 | 21| 155 | 7215 | 57987 | 
| 31 | 26 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 7522 | 58294 | 
| 32 | 27 django/utils/html.py | 306 | 349| 438 | 7960 | 61396 | 
| 33 | 28 django/contrib/admin/templatetags/admin_list.py | 47 | 102| 473 | 8433 | 65225 | 
| 34 | 29 django/db/models/fields/reverse_related.py | 156 | 181| 269 | 8702 | 67368 | 
| 35 | 30 django/views/defaults.py | 1 | 24| 149 | 8851 | 68410 | 
| 36 | 30 django/shortcuts.py | 23 | 54| 243 | 9094 | 68410 | 
| 37 | 30 django/urls/base.py | 160 | 181| 181 | 9275 | 68410 | 
| 38 | 31 django/contrib/redirects/models.py | 1 | 30| 218 | 9493 | 68628 | 
| 39 | 31 django/contrib/admin/templatetags/admin_list.py | 29 | 44| 127 | 9620 | 68628 | 
| 40 | 32 django/contrib/staticfiles/storage.py | 113 | 148| 307 | 9927 | 72158 | 
| 41 | 33 django/http/response.py | 481 | 499| 186 | 10113 | 76677 | 
| 42 | 33 django/contrib/admin/options.py | 1310 | 1335| 232 | 10345 | 76677 | 
| 43 | 34 django/db/models/functions/text.py | 227 | 243| 149 | 10494 | 79133 | 
| 44 | 35 django/urls/conf.py | 57 | 78| 162 | 10656 | 79742 | 
| 45 | 35 django/urls/base.py | 1 | 25| 177 | 10833 | 79742 | 
| 46 | 35 django/urls/resolvers.py | 504 | 537| 229 | 11062 | 79742 | 
| 47 | 36 django/contrib/admin/sites.py | 219 | 238| 221 | 11283 | 83933 | 
| 48 | 36 django/urls/resolvers.py | 539 | 577| 355 | 11638 | 83933 | 
| 49 | 36 django/urls/resolvers.py | 395 | 411| 164 | 11802 | 83933 | 
| 50 | 37 django/contrib/contenttypes/views.py | 1 | 89| 711 | 12513 | 84644 | 
| 51 | 37 django/contrib/sitemaps/views.py | 22 | 45| 250 | 12763 | 84644 | 
| 52 | 38 django/contrib/redirects/__init__.py | 1 | 2| 0 | 12763 | 84658 | 
| 53 | 38 django/contrib/admin/templatetags/admin_list.py | 197 | 211| 136 | 12899 | 84658 | 
| 54 | 38 django/urls/base.py | 90 | 157| 383 | 13282 | 84658 | 
| 55 | 39 django/contrib/admindocs/views.py | 318 | 347| 201 | 13483 | 87966 | 
| 56 | 40 django/contrib/redirects/apps.py | 1 | 8| 0 | 13483 | 88006 | 
| 57 | 41 django/middleware/common.py | 76 | 97| 227 | 13710 | 89517 | 
| 58 | 42 django/contrib/auth/urls.py | 1 | 21| 225 | 13935 | 89742 | 
| 59 | 42 django/contrib/sitemaps/views.py | 48 | 93| 391 | 14326 | 89742 | 
| 60 | 43 django/core/handlers/wsgi.py | 155 | 183| 326 | 14652 | 91439 | 
| 61 | 43 django/urls/resolvers.py | 112 | 142| 232 | 14884 | 91439 | 
| 62 | 43 django/urls/resolvers.py | 443 | 502| 548 | 15432 | 91439 | 
| 63 | 43 django/urls/resolvers.py | 1 | 29| 209 | 15641 | 91439 | 
| 64 | 43 django/db/models/fields/reverse_related.py | 1 | 16| 110 | 15751 | 91439 | 
| 65 | 43 django/contrib/admin/sites.py | 240 | 289| 472 | 16223 | 91439 | 
| 66 | 44 django/contrib/redirects/middleware.py | 1 | 51| 355 | 16578 | 91795 | 
| 67 | 45 django/contrib/auth/views.py | 1 | 37| 278 | 16856 | 94459 | 
| 68 | 45 django/http/response.py | 502 | 533| 153 | 17009 | 94459 | 
| 69 | 45 django/contrib/admindocs/views.py | 136 | 154| 187 | 17196 | 94459 | 
| 70 | 46 django/utils/http.py | 418 | 480| 318 | 17514 | 98637 | 
| 71 | 47 django/views/csrf.py | 15 | 100| 835 | 18349 | 100181 | 
| 72 | 48 django/contrib/syndication/views.py | 50 | 75| 202 | 18551 | 101907 | 
| 73 | 49 django/contrib/admindocs/utils.py | 1 | 25| 151 | 18702 | 103813 | 
| 74 | 50 django/contrib/admin/models.py | 136 | 151| 131 | 18833 | 104936 | 
| 75 | 51 django/conf/urls/__init__.py | 1 | 14| 0 | 18833 | 105031 | 
| 76 | 52 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 18989 | 115389 | 
| 77 | 52 django/contrib/admindocs/views.py | 183 | 249| 584 | 19573 | 115389 | 
| 78 | 53 django/views/static.py | 57 | 80| 211 | 19784 | 116441 | 
| 79 | 53 django/urls/resolvers.py | 67 | 86| 168 | 19952 | 116441 | 
| 80 | 53 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 170 | 20122 | 116441 | 
| 81 | 54 django/template/defaultfilters.py | 325 | 409| 499 | 20621 | 122515 | 
| 82 | 54 django/urls/resolvers.py | 375 | 393| 174 | 20795 | 122515 | 
| 83 | 55 django/core/paginator.py | 58 | 111| 406 | 21201 | 123802 | 
| 84 | 56 django/core/management/templates.py | 324 | 340| 136 | 21337 | 126483 | 
| 85 | 57 django/contrib/gis/db/backends/oracle/operations.py | 38 | 49| 202 | 21539 | 128560 | 
| 86 | 57 django/contrib/admin/options.py | 1894 | 1931| 330 | 21869 | 128560 | 
| 87 | 57 django/contrib/admindocs/views.py | 156 | 180| 234 | 22103 | 128560 | 
| 88 | 58 django/urls/__init__.py | 1 | 24| 239 | 22342 | 128799 | 
| 89 | 58 django/contrib/admindocs/views.py | 56 | 84| 285 | 22627 | 128799 | 
| 90 | 59 django/http/request.py | 153 | 160| 111 | 22738 | 134050 | 
| 91 | 59 django/core/paginator.py | 132 | 191| 428 | 23166 | 134050 | 
| 92 | 60 django/utils/regex_helper.py | 1 | 38| 250 | 23416 | 136691 | 
| 93 | 60 django/contrib/admin/sites.py | 512 | 546| 293 | 23709 | 136691 | 
| 94 | 60 django/shortcuts.py | 1 | 20| 155 | 23864 | 136691 | 
| 95 | 60 django/views/defaults.py | 27 | 76| 401 | 24265 | 136691 | 
| 96 | 61 django/db/models/fields/related.py | 883 | 902| 145 | 24410 | 150265 | 
| 97 | 61 django/middleware/common.py | 99 | 115| 166 | 24576 | 150265 | 
| 98 | 61 django/contrib/contenttypes/fields.py | 274 | 331| 434 | 25010 | 150265 | 
| 99 | 61 django/contrib/admindocs/views.py | 250 | 315| 585 | 25595 | 150265 | 
| 100 | 61 django/core/management/templates.py | 243 | 295| 405 | 26000 | 150265 | 
| 101 | 61 django/views/generic/base.py | 154 | 186| 228 | 26228 | 150265 | 
| 102 | 61 django/urls/resolvers.py | 247 | 269| 173 | 26401 | 150265 | 
| 103 | 61 django/contrib/admin/utils.py | 287 | 305| 175 | 26576 | 150265 | 
| 104 | 62 django/contrib/admin/views/main.py | 257 | 287| 270 | 26846 | 154519 | 
| 105 | 63 django/contrib/staticfiles/urls.py | 1 | 20| 0 | 26846 | 154616 | 
| 106 | 63 django/contrib/admin/options.py | 1611 | 1635| 279 | 27125 | 154616 | 
| 107 | 63 django/middleware/common.py | 63 | 74| 117 | 27242 | 154616 | 
| 108 | 63 django/views/csrf.py | 1 | 13| 132 | 27374 | 154616 | 
| 109 | 63 django/contrib/syndication/views.py | 1 | 26| 220 | 27594 | 154616 | 
| 110 | 64 django/utils/feedgenerator.py | 46 | 56| 114 | 27708 | 157949 | 
| 111 | 64 django/utils/html.py | 235 | 257| 254 | 27962 | 157949 | 
| 112 | 64 django/contrib/auth/views.py | 129 | 163| 269 | 28231 | 157949 | 
| 113 | 64 django/urls/resolvers.py | 324 | 372| 367 | 28598 | 157949 | 
| 114 | 64 django/contrib/admin/options.py | 1 | 96| 769 | 29367 | 157949 | 
| 115 | 65 django/template/loader_tags.py | 220 | 247| 239 | 29606 | 160500 | 
| 116 | 65 django/contrib/admin/options.py | 1235 | 1308| 659 | 30265 | 160500 | 
| 117 | 65 django/db/models/fields/related_descriptors.py | 326 | 342| 113 | 30378 | 160500 | 
| 118 | 65 django/urls/resolvers.py | 271 | 287| 160 | 30538 | 160500 | 
| 119 | 65 django/contrib/staticfiles/storage.py | 150 | 202| 425 | 30963 | 160500 | 
| 120 | 65 django/urls/resolvers.py | 579 | 599| 177 | 31140 | 160500 | 
| 121 | 65 django/contrib/admindocs/utils.py | 86 | 112| 175 | 31315 | 160500 | 
| 122 | 65 django/contrib/admin/widgets.py | 331 | 349| 168 | 31483 | 160500 | 
| 123 | 65 django/middleware/common.py | 34 | 61| 257 | 31740 | 160500 | 
| 124 | 65 django/contrib/admin/utils.py | 121 | 156| 303 | 32043 | 160500 | 
| 125 | 65 django/contrib/admin/views/main.py | 206 | 255| 423 | 32466 | 160500 | 
| 126 | 66 django/views/debug.py | 322 | 351| 267 | 32733 | 164845 | 
| 127 | 66 django/contrib/auth/views.py | 247 | 284| 348 | 33081 | 164845 | 
| 128 | 66 django/db/models/fields/related.py | 701 | 739| 335 | 33416 | 164845 | 
| 129 | 67 django/views/generic/edit.py | 103 | 126| 194 | 33610 | 166561 | 
| 130 | 67 django/contrib/admindocs/views.py | 118 | 133| 154 | 33764 | 166561 | 
| 131 | 68 django/db/models/options.py | 556 | 579| 228 | 33992 | 173607 | 
| 132 | 68 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 34176 | 173607 | 
| 133 | 68 django/contrib/auth/views.py | 330 | 362| 239 | 34415 | 173607 | 
| 134 | 68 django/db/models/fields/related_descriptors.py | 383 | 428| 355 | 34770 | 173607 | 
| 135 | 69 django/views/i18n.py | 88 | 191| 711 | 35481 | 176154 | 
| 136 | 69 django/contrib/syndication/views.py | 29 | 48| 195 | 35676 | 176154 | 
| 137 | 69 django/shortcuts.py | 57 | 78| 216 | 35892 | 176154 | 
| 138 | 69 django/urls/resolvers.py | 290 | 321| 190 | 36082 | 176154 | 
| 139 | 69 django/contrib/auth/views.py | 286 | 327| 314 | 36396 | 176154 | 
| 140 | 69 django/contrib/admindocs/views.py | 1 | 30| 223 | 36619 | 176154 | 
| 141 | 70 django/db/models/fields/__init__.py | 2199 | 2219| 163 | 36782 | 193722 | 
| 142 | 70 django/contrib/sitemaps/__init__.py | 54 | 139| 631 | 37413 | 193722 | 
| 143 | 70 django/views/csrf.py | 101 | 155| 577 | 37990 | 193722 | 
| 144 | 70 django/contrib/admindocs/views.py | 33 | 53| 159 | 38149 | 193722 | 
| 145 | 71 django/contrib/sites/admin.py | 1 | 9| 0 | 38149 | 193768 | 
| 146 | 71 django/utils/html.py | 259 | 289| 321 | 38470 | 193768 | 
| 147 | 72 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 38744 | 194042 | 
| 148 | 72 django/contrib/admin/utils.py | 405 | 434| 203 | 38947 | 194042 | 
| 149 | 72 django/views/debug.py | 458 | 524| 544 | 39491 | 194042 | 
| 150 | 73 django/template/backends/utils.py | 1 | 15| 0 | 39491 | 194131 | 
| 151 | 74 django/contrib/humanize/templatetags/humanize.py | 182 | 216| 280 | 39771 | 197291 | 
| 152 | 74 django/views/static.py | 83 | 105| 155 | 39926 | 197291 | 
| 153 | 74 django/contrib/admin/options.py | 1404 | 1443| 309 | 40235 | 197291 | 
| 154 | 74 django/db/models/fields/related.py | 190 | 254| 673 | 40908 | 197291 | 


### Hint

```
The implementation is intended to match the catchall middleware. We could try reversing and only if that fails fall back to the current logic.
I just sent a pull request for a fix for this: ​https://github.com/django/django/pull/2554 This solves all the cases I can think of for including flatpages and running get_absolute_url() on them. By the way, not sure if this is relevant, but: When I was writing the tests for this, I noticed that all the flatpages tests include the flatpages url without a slash, e.g.: url(r'flatpage_root', include('django.contrib.flatpages.urls')), (Note lack of '/' after flatpage_root). This is different than how the documentation recommends including the flatpages, and doesn't make much sense. Not sure what the purpose of this is.
I left comments for improvement on the PR. Please uncheck "Patch needs improvement" when you update it, thanks.
#28118 is a duplicate.
I created another ​PR based on the ​first PR.
```

## Patch

```diff
diff --git a/django/contrib/flatpages/models.py b/django/contrib/flatpages/models.py
--- a/django/contrib/flatpages/models.py
+++ b/django/contrib/flatpages/models.py
@@ -1,6 +1,6 @@
 from django.contrib.sites.models import Site
 from django.db import models
-from django.urls import get_script_prefix
+from django.urls import NoReverseMatch, get_script_prefix, reverse
 from django.utils.encoding import iri_to_uri
 from django.utils.translation import gettext_lazy as _
 
@@ -36,5 +36,12 @@ def __str__(self):
         return "%s -- %s" % (self.url, self.title)
 
     def get_absolute_url(self):
+        from .views import flatpage
+
+        for url in (self.url.lstrip('/'), self.url):
+            try:
+                return reverse(flatpage, kwargs={'url': url})
+            except NoReverseMatch:
+                pass
         # Handle script prefix manually because we bypass reverse()
         return iri_to_uri(get_script_prefix().rstrip('/') + self.url)

```

## Test Patch

```diff
diff --git a/tests/flatpages_tests/absolute_urls.py b/tests/flatpages_tests/absolute_urls.py
new file mode 100644
--- /dev/null
+++ b/tests/flatpages_tests/absolute_urls.py
@@ -0,0 +1,6 @@
+from django.contrib.flatpages import views
+from django.urls import path
+
+urlpatterns = [
+    path('flatpage/', views.flatpage, {'url': '/hardcoded/'}),
+]
diff --git a/tests/flatpages_tests/no_slash_urls.py b/tests/flatpages_tests/no_slash_urls.py
new file mode 100644
--- /dev/null
+++ b/tests/flatpages_tests/no_slash_urls.py
@@ -0,0 +1,5 @@
+from django.urls import include, path
+
+urlpatterns = [
+    path('flatpage', include('django.contrib.flatpages.urls')),
+]
diff --git a/tests/flatpages_tests/test_models.py b/tests/flatpages_tests/test_models.py
--- a/tests/flatpages_tests/test_models.py
+++ b/tests/flatpages_tests/test_models.py
@@ -1,5 +1,5 @@
 from django.contrib.flatpages.models import FlatPage
-from django.test import SimpleTestCase
+from django.test import SimpleTestCase, override_settings
 from django.test.utils import override_script_prefix
 
 
@@ -17,3 +17,16 @@ def test_get_absolute_url_honors_script_prefix(self):
 
     def test_str(self):
         self.assertEqual(str(self.page), '/café/ -- Café!')
+
+    @override_settings(ROOT_URLCONF='flatpages_tests.urls')
+    def test_get_absolute_url_include(self):
+        self.assertEqual(self.page.get_absolute_url(), '/flatpage_root/caf%C3%A9/')
+
+    @override_settings(ROOT_URLCONF='flatpages_tests.no_slash_urls')
+    def test_get_absolute_url_include_no_slash(self):
+        self.assertEqual(self.page.get_absolute_url(), '/flatpagecaf%C3%A9/')
+
+    @override_settings(ROOT_URLCONF='flatpages_tests.absolute_urls')
+    def test_get_absolute_url_with_hardcoded_url(self):
+        fp = FlatPage(title='Test', url='/hardcoded/')
+        self.assertEqual(fp.get_absolute_url(), '/flatpage/')
diff --git a/tests/flatpages_tests/test_sitemaps.py b/tests/flatpages_tests/test_sitemaps.py
--- a/tests/flatpages_tests/test_sitemaps.py
+++ b/tests/flatpages_tests/test_sitemaps.py
@@ -31,5 +31,8 @@ def setUpTestData(cls):
 
     def test_flatpage_sitemap(self):
         response = self.client.get('/flatpages/sitemap.xml')
-        self.assertIn(b'<url><loc>http://example.com/foo/</loc></url>', response.getvalue())
-        self.assertNotIn(b'<url><loc>http://example.com/private-foo/</loc></url>', response.getvalue())
+        self.assertIn(b'<url><loc>http://example.com/flatpage_root/foo/</loc></url>', response.getvalue())
+        self.assertNotIn(
+            b'<url><loc>http://example.com/flatpage_root/private-foo/</loc></url>',
+            response.getvalue(),
+        )
diff --git a/tests/flatpages_tests/urls.py b/tests/flatpages_tests/urls.py
--- a/tests/flatpages_tests/urls.py
+++ b/tests/flatpages_tests/urls.py
@@ -8,6 +8,6 @@
         {'sitemaps': {'flatpages': FlatPageSitemap}},
         name='django.contrib.sitemaps.views.sitemap'),
 
-    path('flatpage_root', include('django.contrib.flatpages.urls')),
+    path('flatpage_root/', include('django.contrib.flatpages.urls')),
     path('accounts/', include('django.contrib.auth.urls')),
 ]

```


## Code snippets

### 1 - django/contrib/flatpages/models.py:

Start line: 1, End line: 41

```python
from django.contrib.sites.models import Site
from django.db import models
from django.urls import get_script_prefix
from django.utils.encoding import iri_to_uri
from django.utils.translation import gettext_lazy as _


class FlatPage(models.Model):
    url = models.CharField(_('URL'), max_length=100, db_index=True)
    title = models.CharField(_('title'), max_length=200)
    content = models.TextField(_('content'), blank=True)
    enable_comments = models.BooleanField(_('enable comments'), default=False)
    template_name = models.CharField(
        _('template name'),
        max_length=70,
        blank=True,
        help_text=_(
            'Example: “flatpages/contact_page.html”. If this isn’t provided, '
            'the system will use “flatpages/default.html”.'
        ),
    )
    registration_required = models.BooleanField(
        _('registration required'),
        help_text=_("If this is checked, only logged-in users will be able to view the page."),
        default=False,
    )
    sites = models.ManyToManyField(Site, verbose_name=_('sites'))

    class Meta:
        db_table = 'django_flatpage'
        verbose_name = _('flat page')
        verbose_name_plural = _('flat pages')
        ordering = ['url']

    def __str__(self):
        return "%s -- %s" % (self.url, self.title)

    def get_absolute_url(self):
        # Handle script prefix manually because we bypass reverse()
        return iri_to_uri(get_script_prefix().rstrip('/') + self.url)
```
### 2 - django/contrib/flatpages/urls.py:

Start line: 1, End line: 7

```python

```
### 3 - django/contrib/flatpages/views.py:

Start line: 1, End line: 45

```python
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.contrib.sites.shortcuts import get_current_site
from django.http import Http404, HttpResponse, HttpResponsePermanentRedirect
from django.shortcuts import get_object_or_404
from django.template import loader
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_protect

DEFAULT_TEMPLATE = 'flatpages/default.html'

# This view is called from FlatpageFallbackMiddleware.process_response
# when a 404 is raised, which often means CsrfViewMiddleware.process_view
# has not been called even if CsrfViewMiddleware is installed. So we need
# to use @csrf_protect, in case the template needs {% csrf_token %}.
# However, we can't just wrap this view; if no matching flatpage exists,
# or a redirect is required for authentication, the 404 needs to be returned
# without any CSRF checks. Therefore, we only
# CSRF protect the internal implementation.


def flatpage(request, url):
    """
    Public interface to the flat page view.

    Models: `flatpages.flatpages`
    Templates: Uses the template defined by the ``template_name`` field,
        or :template:`flatpages/default.html` if template_name is not defined.
    Context:
        flatpage
            `flatpages.flatpages` object
    """
    if not url.startswith('/'):
        url = '/' + url
    site_id = get_current_site(request).id
    try:
        f = get_object_or_404(FlatPage, url=url, sites=site_id)
    except Http404:
        if not url.endswith('/') and settings.APPEND_SLASH:
            url += '/'
            f = get_object_or_404(FlatPage, url=url, sites=site_id)
            return HttpResponsePermanentRedirect('%s/' % request.path)
        else:
            raise
    return render_flatpage(request, f)
```
### 4 - django/contrib/flatpages/forms.py:

Start line: 1, End line: 50

```python
from django import forms
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.utils.translation import gettext, gettext_lazy as _


class FlatpageForm(forms.ModelForm):
    url = forms.RegexField(
        label=_("URL"),
        max_length=100,
        regex=r'^[-\w/\.~]+$',
        help_text=_('Example: “/about/contact/”. Make sure to have leading and trailing slashes.'),
        error_messages={
            "invalid": _(
                "This value must contain only letters, numbers, dots, "
                "underscores, dashes, slashes or tildes."
            ),
        },
    )

    class Meta:
        model = FlatPage
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._trailing_slash_required():
            self.fields['url'].help_text = _(
                'Example: “/about/contact”. Make sure to have a leading slash.'
            )

    def _trailing_slash_required(self):
        return (
            settings.APPEND_SLASH and
            'django.middleware.common.CommonMiddleware' in settings.MIDDLEWARE
        )

    def clean_url(self):
        url = self.cleaned_data['url']
        if not url.startswith('/'):
            raise forms.ValidationError(
                gettext("URL is missing a leading slash."),
                code='missing_leading_slash',
            )
        if self._trailing_slash_required() and not url.endswith('/'):
            raise forms.ValidationError(
                gettext("URL is missing a trailing slash."),
                code='missing_trailing_slash',
            )
        return url
```
### 5 - django/urls/base.py:

Start line: 28, End line: 87

```python
def reverse(viewname, urlconf=None, args=None, kwargs=None, current_app=None):
    if urlconf is None:
        urlconf = get_urlconf()
    resolver = get_resolver(urlconf)
    args = args or []
    kwargs = kwargs or {}

    prefix = get_script_prefix()

    if not isinstance(viewname, str):
        view = viewname
    else:
        *path, view = viewname.split(':')

        if current_app:
            current_path = current_app.split(':')
            current_path.reverse()
        else:
            current_path = None

        resolved_path = []
        ns_pattern = ''
        ns_converters = {}
        for ns in path:
            current_ns = current_path.pop() if current_path else None
            # Lookup the name to see if it could be an app identifier.
            try:
                app_list = resolver.app_dict[ns]
                # Yes! Path part matches an app in the current Resolver.
                if current_ns and current_ns in app_list:
                    # If we are reversing for a particular app, use that
                    # namespace.
                    ns = current_ns
                elif ns not in app_list:
                    # The name isn't shared by one of the instances (i.e.,
                    # the default) so pick the first instance as the default.
                    ns = app_list[0]
            except KeyError:
                pass

            if ns != current_ns:
                current_path = None

            try:
                extra, resolver = resolver.namespace_dict[ns]
                resolved_path.append(ns)
                ns_pattern = ns_pattern + extra
                ns_converters.update(resolver.pattern.converters)
            except KeyError as key:
                if resolved_path:
                    raise NoReverseMatch(
                        "%s is not a registered namespace inside '%s'" %
                        (key, ':'.join(resolved_path))
                    )
                else:
                    raise NoReverseMatch("%s is not a registered namespace" % key)
        if ns_pattern:
            resolver = get_ns_resolver(ns_pattern, resolver, tuple(ns_converters.items()))

    return iri_to_uri(resolver._reverse_with_prefix(view, prefix, *args, **kwargs))
```
### 6 - django/urls/resolvers.py:

Start line: 613, End line: 686

```python
class URLResolver:

    def _reverse_with_prefix(self, lookup_view, _prefix, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Don't mix *args and **kwargs in call to reverse()!")

        if not self._populated:
            self._populate()

        possibilities = self.reverse_dict.getlist(lookup_view)

        for possibility, pattern, defaults, converters in possibilities:
            for result, params in possibility:
                if args:
                    if len(args) != len(params):
                        continue
                    candidate_subs = dict(zip(params, args))
                else:
                    if set(kwargs).symmetric_difference(params).difference(defaults):
                        continue
                    if any(kwargs.get(k, v) != v for k, v in defaults.items()):
                        continue
                    candidate_subs = kwargs
                # Convert the candidate subs to text using Converter.to_url().
                text_candidate_subs = {}
                match = True
                for k, v in candidate_subs.items():
                    if k in converters:
                        try:
                            text_candidate_subs[k] = converters[k].to_url(v)
                        except ValueError:
                            match = False
                            break
                    else:
                        text_candidate_subs[k] = str(v)
                if not match:
                    continue
                # WSGI provides decoded URLs, without %xx escapes, and the URL
                # resolver operates on such URLs. First substitute arguments
                # without quoting to build a decoded URL and look for a match.
                # Then, if we have a match, redo the substitution with quoted
                # arguments in order to return a properly encoded URL.
                candidate_pat = _prefix.replace('%', '%%') + result
                if re.search('^%s%s' % (re.escape(_prefix), pattern), candidate_pat % text_candidate_subs):
                    # safe characters from `pchar` definition of RFC 3986
                    url = quote(candidate_pat % text_candidate_subs, safe=RFC3986_SUBDELIMS + '/~:@')
                    # Don't allow construction of scheme relative urls.
                    return escape_leading_slashes(url)
        # lookup_view can be URL name or callable, but callables are not
        # friendly in error messages.
        m = getattr(lookup_view, '__module__', None)
        n = getattr(lookup_view, '__name__', None)
        if m is not None and n is not None:
            lookup_view_s = "%s.%s" % (m, n)
        else:
            lookup_view_s = lookup_view

        patterns = [pattern for (_, pattern, _, _) in possibilities]
        if patterns:
            if args:
                arg_msg = "arguments '%s'" % (args,)
            elif kwargs:
                arg_msg = "keyword arguments '%s'" % (kwargs,)
            else:
                arg_msg = "no arguments"
            msg = (
                "Reverse for '%s' with %s not found. %d pattern(s) tried: %s" %
                (lookup_view_s, arg_msg, len(patterns), patterns)
            )
        else:
            msg = (
                "Reverse for '%(view)s' not found. '%(view)s' is not "
                "a valid view function or pattern name." % {'view': lookup_view_s}
            )
        raise NoReverseMatch(msg)
```
### 7 - django/contrib/flatpages/views.py:

Start line: 48, End line: 70

```python
@csrf_protect
def render_flatpage(request, f):
    """
    Internal interface to the flat page view.
    """
    # If registration is required for accessing this page, and the user isn't
    # logged in, redirect to the login page.
    if f.registration_required and not request.user.is_authenticated:
        from django.contrib.auth.views import redirect_to_login
        return redirect_to_login(request.path)
    if f.template_name:
        template = loader.select_template((f.template_name, DEFAULT_TEMPLATE))
    else:
        template = loader.get_template(DEFAULT_TEMPLATE)

    # To avoid having to always use the "|safe" filter in flatpage templates,
    # mark the title and content as already safe (since they are raw HTML
    # content in the first place).
    f.title = mark_safe(f.title)
    f.content = mark_safe(f.content)

    return HttpResponse(template.render({'flatpage': f}, request))
```
### 8 - django/contrib/flatpages/forms.py:

Start line: 52, End line: 70

```python
class FlatpageForm(forms.ModelForm):

    def clean(self):
        url = self.cleaned_data.get('url')
        sites = self.cleaned_data.get('sites')

        same_url = FlatPage.objects.filter(url=url)
        if self.instance.pk:
            same_url = same_url.exclude(pk=self.instance.pk)

        if sites and same_url.filter(sites__in=sites).exists():
            for site in sites:
                if same_url.filter(sites=site).exists():
                    raise forms.ValidationError(
                        _('Flatpage with url %(url)s already exists for site %(site)s'),
                        code='duplicate_url',
                        params={'url': url, 'site': site},
                    )

        return super().clean()
```
### 9 - django/contrib/flatpages/templatetags/flatpages.py:

Start line: 1, End line: 42

```python
from django import template
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.contrib.sites.shortcuts import get_current_site

register = template.Library()


class FlatpageNode(template.Node):
    def __init__(self, context_name, starts_with=None, user=None):
        self.context_name = context_name
        if starts_with:
            self.starts_with = template.Variable(starts_with)
        else:
            self.starts_with = None
        if user:
            self.user = template.Variable(user)
        else:
            self.user = None

    def render(self, context):
        if 'request' in context:
            site_pk = get_current_site(context['request']).pk
        else:
            site_pk = settings.SITE_ID
        flatpages = FlatPage.objects.filter(sites__id=site_pk)
        # If a prefix was specified, add a filter
        if self.starts_with:
            flatpages = flatpages.filter(
                url__startswith=self.starts_with.resolve(context))

        # If the provided user is not authenticated, or no user
        # was provided, filter the list to only public flatpages.
        if self.user:
            user = self.user.resolve(context)
            if not user.is_authenticated:
                flatpages = flatpages.filter(registration_required=False)
        else:
            flatpages = flatpages.filter(registration_required=False)

        context[self.context_name] = flatpages
        return ''
```
### 10 - django/contrib/flatpages/__init__.py:

Start line: 1, End line: 2

```python

```
