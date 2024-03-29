# django__django-15648

| **django/django** | `7e4656e4b2189390a433a149091442d53a777e2b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11666 |
| **Any found context length** | 141 |
| **Avg pos** | 44.0 |
| **Min pos** | 1 |
| **Max pos** | 43 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/syndication/views.py b/django/contrib/syndication/views.py
--- a/django/contrib/syndication/views.py
+++ b/django/contrib/syndication/views.py
@@ -1,3 +1,5 @@
+from inspect import getattr_static, unwrap
+
 from django.contrib.sites.shortcuts import get_current_site
 from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
 from django.http import Http404, HttpResponse
@@ -82,10 +84,21 @@ def _get_dynamic_attr(self, attname, obj, default=None):
             # Check co_argcount rather than try/excepting the function and
             # catching the TypeError, because something inside the function
             # may raise the TypeError. This technique is more accurate.
+            func = unwrap(attr)
             try:
-                code = attr.__code__
+                code = func.__code__
             except AttributeError:
-                code = attr.__call__.__code__
+                func = unwrap(attr.__call__)
+                code = func.__code__
+            # If function doesn't have arguments and it is not a static method,
+            # it was decorated without using @functools.wraps.
+            if not code.co_argcount and not isinstance(
+                getattr_static(self, func.__name__, None), staticmethod
+            ):
+                raise ImproperlyConfigured(
+                    f"Feed method {attname!r} decorated by {func.__name__!r} needs to "
+                    f"use @functools.wraps."
+                )
             if code.co_argcount == 2:  # one argument is 'self'
                 return attr(obj)
             else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/syndication/views.py | 1 | 1 | 43 | 1 | 11666
| django/contrib/syndication/views.py | 85 | 87 | 1 | 1 | 141


## Problem Statement

```
views.Feed methods cannot be decorated
Description
	
If one applies a decorator on a method which is called by __get_dynamic_attr a TypeError like this occurs:
Exception Type: TypeError at /blog/feed/
Exception Value: item_link() takes exactly 2 arguments (1 given)
I think this is because __get_dynamic_attr tries to count the function's arguments, but decorators usally get defined with the *args, **kwargs syntax, so this trick does not work here.
			if code.co_argcount == 2:	 # one argument is 'self'
				return attr(obj)
			else:
				return attr()
I think the best approach would be to remove one of the two methods. IMHO We should have either attr(item) or attr() not both, as "there should be one, and preferably only one, obvious way to do it".

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/syndication/views.py** | 76 | 93| 141 | 141 | 1737 | 
| 2 | **1 django/contrib/syndication/views.py** | 49 | 74| 202 | 343 | 1737 | 
| 3 | **1 django/contrib/syndication/views.py** | 167 | 222| 480 | 823 | 1737 | 
| 4 | 2 django/utils/decorators.py | 25 | 53| 300 | 1123 | 3149 | 
| 5 | 3 django/utils/feedgenerator.py | 341 | 448| 946 | 2069 | 6530 | 
| 6 | **3 django/contrib/syndication/views.py** | 27 | 47| 193 | 2262 | 6530 | 
| 7 | 3 django/utils/decorators.py | 56 | 89| 357 | 2619 | 6530 | 
| 8 | 3 django/utils/feedgenerator.py | 281 | 338| 487 | 3106 | 6530 | 
| 9 | 3 django/utils/feedgenerator.py | 101 | 149| 331 | 3437 | 6530 | 
| 10 | 3 django/utils/decorators.py | 1 | 22| 142 | 3579 | 6530 | 
| 11 | 4 django/views/decorators/http.py | 85 | 133| 367 | 3946 | 7510 | 
| 12 | 4 django/utils/feedgenerator.py | 151 | 195| 259 | 4205 | 7510 | 
| 13 | 5 django/contrib/gis/feeds.py | 136 | 152| 128 | 4333 | 8828 | 
| 14 | **5 django/contrib/syndication/views.py** | 95 | 120| 180 | 4513 | 8828 | 
| 15 | 5 django/utils/feedgenerator.py | 59 | 99| 265 | 4778 | 8828 | 
| 16 | 5 django/utils/feedgenerator.py | 224 | 247| 185 | 4963 | 8828 | 
| 17 | 5 django/contrib/gis/feeds.py | 121 | 133| 136 | 5099 | 8828 | 
| 18 | 5 django/views/decorators/http.py | 62 | 83| 272 | 5371 | 8828 | 
| 19 | 5 django/utils/feedgenerator.py | 249 | 278| 309 | 5680 | 8828 | 
| 20 | 6 django/utils/functional.py | 215 | 271| 323 | 6003 | 12111 | 
| 21 | 6 django/contrib/gis/feeds.py | 106 | 118| 115 | 6118 | 12111 | 
| 22 | 6 django/contrib/gis/feeds.py | 90 | 103| 123 | 6241 | 12111 | 
| 23 | 7 django/template/base.py | 749 | 773| 210 | 6451 | 20384 | 
| 24 | 8 django/contrib/admin/decorators.py | 34 | 77| 271 | 6722 | 21038 | 
| 25 | 9 django/views/decorators/clickjacking.py | 1 | 22| 140 | 6862 | 21423 | 
| 26 | 10 django/views/decorators/debug.py | 79 | 97| 135 | 6997 | 22014 | 
| 27 | **10 django/contrib/syndication/views.py** | 122 | 165| 357 | 7354 | 22014 | 
| 28 | 11 django/template/defaultfilters.py | 505 | 545| 248 | 7602 | 28494 | 
| 29 | 12 django/dispatch/dispatcher.py | 283 | 306| 139 | 7741 | 30599 | 
| 30 | 12 django/views/decorators/clickjacking.py | 25 | 63| 243 | 7984 | 30599 | 
| 31 | 12 django/contrib/gis/feeds.py | 32 | 87| 550 | 8534 | 30599 | 
| 32 | 13 django/contrib/gis/views.py | 1 | 23| 160 | 8694 | 30759 | 
| 33 | 13 django/template/defaultfilters.py | 697 | 729| 217 | 8911 | 30759 | 
| 34 | 13 django/views/decorators/http.py | 1 | 59| 358 | 9269 | 30759 | 
| 35 | 13 django/utils/functional.py | 1 | 58| 370 | 9639 | 30759 | 
| 36 | 13 django/template/defaultfilters.py | 1 | 90| 559 | 10198 | 30759 | 
| 37 | 13 django/utils/feedgenerator.py | 197 | 221| 184 | 10382 | 30759 | 
| 38 | 14 django/utils/deconstruct.py | 1 | 60| 403 | 10785 | 31162 | 
| 39 | 14 django/utils/functional.py | 61 | 84| 123 | 10908 | 31162 | 
| 40 | 14 django/contrib/admin/decorators.py | 1 | 31| 181 | 11089 | 31162 | 
| 41 | 14 django/contrib/gis/feeds.py | 1 | 17| 141 | 11230 | 31162 | 
| 42 | 15 django/utils/asyncio.py | 1 | 40| 221 | 11451 | 31384 | 
| **-> 43 <-** | **15 django/contrib/syndication/views.py** | 1 | 24| 215 | 11666 | 31384 | 
| 44 | 16 django/forms/widgets.py | 232 | 315| 624 | 12290 | 39641 | 
| 45 | 17 django/views/decorators/vary.py | 1 | 47| 232 | 12522 | 39874 | 
| 46 | 17 django/utils/functional.py | 143 | 212| 520 | 13042 | 39874 | 
| 47 | 18 django/contrib/auth/decorators.py | 43 | 83| 276 | 13318 | 40466 | 
| 48 | 18 django/contrib/auth/decorators.py | 1 | 40| 315 | 13633 | 40466 | 
| 49 | 19 django/views/decorators/csrf.py | 1 | 59| 462 | 14095 | 40929 | 
| 50 | 19 django/utils/functional.py | 373 | 395| 201 | 14296 | 40929 | 
| 51 | 20 django/core/checks/registry.py | 28 | 69| 254 | 14550 | 41607 | 
| 52 | 21 django/utils/html.py | 403 | 422| 168 | 14718 | 44859 | 
| 53 | 22 django/contrib/admin/views/decorators.py | 1 | 20| 137 | 14855 | 44997 | 
| 54 | 22 django/utils/decorators.py | 117 | 162| 321 | 15176 | 44997 | 
| 55 | 22 django/template/defaultfilters.py | 548 | 641| 525 | 15701 | 44997 | 
| 56 | 23 django/views/decorators/cache.py | 29 | 46| 129 | 15830 | 45472 | 
| 57 | 24 django/contrib/sitemaps/views.py | 1 | 38| 249 | 16079 | 46631 | 
| 58 | 25 django/views/decorators/common.py | 1 | 17| 112 | 16191 | 46744 | 
| 59 | 26 django/contrib/admin/actions.py | 1 | 97| 647 | 16838 | 47391 | 
| 60 | 26 django/views/decorators/cache.py | 49 | 67| 133 | 16971 | 47391 | 
| 61 | 26 django/views/decorators/debug.py | 1 | 46| 273 | 17244 | 47391 | 
| 62 | 27 django/contrib/admindocs/views.py | 102 | 138| 301 | 17545 | 50873 | 
| 63 | 28 django/contrib/admin/sites.py | 205 | 227| 167 | 17712 | 55325 | 
| 64 | 29 django/db/migrations/autodetector.py | 52 | 88| 319 | 18031 | 68688 | 
| 65 | 30 django/contrib/admin/options.py | 1 | 114| 776 | 18807 | 87955 | 
| 66 | 31 django/utils/deprecation.py | 42 | 84| 339 | 19146 | 89026 | 
| 67 | 31 django/template/defaultfilters.py | 644 | 695| 322 | 19468 | 89026 | 
| 68 | 31 django/utils/functional.py | 121 | 141| 211 | 19679 | 89026 | 
| 69 | 31 django/template/defaultfilters.py | 365 | 452| 504 | 20183 | 89026 | 
| 70 | 31 django/contrib/admindocs/views.py | 430 | 451| 132 | 20315 | 89026 | 
| 71 | 31 django/utils/feedgenerator.py | 1 | 42| 321 | 20636 | 89026 | 
| 72 | 31 django/template/defaultfilters.py | 183 | 204| 212 | 20848 | 89026 | 
| 73 | 31 django/utils/decorators.py | 92 | 114| 152 | 21000 | 89026 | 
| 74 | 32 django/contrib/admin/checks.py | 955 | 979| 197 | 21197 | 98558 | 
| 75 | 32 django/contrib/admindocs/views.py | 164 | 181| 175 | 21372 | 98558 | 
| 76 | 33 django/template/defaulttags.py | 768 | 866| 774 | 22146 | 109331 | 
| 77 | 33 django/contrib/admin/checks.py | 1091 | 1143| 430 | 22576 | 109331 | 
| 78 | 34 django/contrib/admin/utils.py | 294 | 320| 189 | 22765 | 113537 | 
| 79 | 35 django/core/checks/templates.py | 50 | 76| 166 | 22931 | 114016 | 
| 80 | 35 django/template/defaulttags.py | 183 | 245| 532 | 23463 | 114016 | 
| 81 | 35 django/template/defaulttags.py | 545 | 569| 191 | 23654 | 114016 | 
| 82 | 36 docs/_ext/djangodocs.py | 111 | 175| 567 | 24221 | 117240 | 
| 83 | 36 django/contrib/admin/checks.py | 894 | 928| 222 | 24443 | 117240 | 
| 84 | 36 django/contrib/admin/decorators.py | 80 | 112| 142 | 24585 | 117240 | 
| 85 | 36 django/utils/functional.py | 87 | 119| 235 | 24820 | 117240 | 
| 86 | 36 django/template/defaulttags.py | 117 | 156| 236 | 25056 | 117240 | 
| 87 | 37 django/views/debug.py | 105 | 140| 262 | 25318 | 121986 | 
| 88 | 37 django/contrib/admin/checks.py | 981 | 1038| 457 | 25775 | 121986 | 
| 89 | 38 django/views/generic/__init__.py | 1 | 40| 204 | 25979 | 122191 | 
| 90 | 38 django/utils/deprecation.py | 1 | 39| 228 | 26207 | 122191 | 
| 91 | 38 django/template/defaulttags.py | 527 | 542| 142 | 26349 | 122191 | 
| 92 | 38 django/contrib/gis/feeds.py | 19 | 30| 135 | 26484 | 122191 | 
| 93 | 38 django/contrib/admin/checks.py | 879 | 892| 121 | 26605 | 122191 | 
| 94 | 38 django/views/decorators/debug.py | 49 | 77| 199 | 26804 | 122191 | 
| 95 | 39 django/forms/utils.py | 23 | 45| 195 | 26999 | 123910 | 
| 96 | 40 django/db/models/fields/related.py | 1830 | 1849| 189 | 27188 | 138522 | 
| 97 | 41 django/contrib/admin/widgets.py | 297 | 343| 439 | 27627 | 142706 | 
| 98 | 41 django/forms/widgets.py | 882 | 915| 282 | 27909 | 142706 | 
| 99 | 42 django/contrib/contenttypes/fields.py | 160 | 172| 125 | 28034 | 148333 | 
| 100 | 42 django/db/models/fields/related.py | 1851 | 1879| 275 | 28309 | 148333 | 
| 101 | 42 django/utils/deprecation.py | 87 | 140| 382 | 28691 | 148333 | 
| 102 | 43 django/core/checks/model_checks.py | 187 | 228| 345 | 29036 | 150143 | 
| 103 | 43 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 29294 | 150143 | 
| 104 | 43 django/contrib/admindocs/views.py | 65 | 99| 297 | 29591 | 150143 | 
| 105 | 43 django/views/decorators/cache.py | 1 | 26| 211 | 29802 | 150143 | 
| 106 | 43 django/contrib/admin/checks.py | 930 | 953| 195 | 29997 | 150143 | 
| 107 | 44 django/db/models/options.py | 169 | 232| 596 | 30593 | 157645 | 
| 108 | 44 django/forms/widgets.py | 745 | 760| 122 | 30715 | 157645 | 
| 109 | 44 django/contrib/admin/options.py | 1752 | 1855| 790 | 31505 | 157645 | 
| 110 | 44 django/views/debug.py | 225 | 277| 471 | 31976 | 157645 | 
| 111 | 44 django/utils/html.py | 139 | 165| 149 | 32125 | 157645 | 
| 112 | 45 django/contrib/flatpages/templatetags/flatpages.py | 1 | 43| 302 | 32427 | 158424 | 
| 113 | 45 django/template/defaulttags.py | 1164 | 1229| 647 | 33074 | 158424 | 
| 114 | 45 django/template/defaulttags.py | 327 | 345| 151 | 33225 | 158424 | 
| 115 | 45 django/template/defaulttags.py | 572 | 662| 812 | 34037 | 158424 | 
| 116 | 46 django/contrib/admin/views/main.py | 153 | 254| 863 | 34900 | 162955 | 
| 117 | 47 django/db/models/query.py | 2325 | 2381| 483 | 35383 | 183128 | 


### Hint

```
Yes, that code is fragile.
I reproduced it. I'm working now on fix.
sample project, which show how patch works now
I sended pull request. ​https://github.com/django/django/pull/2584 The problem is that methods without the 'item' argument (only with self) should be static now: @staticmethod def item_description(self): return force_text(item) But the decorators work now (sample project is in attachments)
```

## Patch

```diff
diff --git a/django/contrib/syndication/views.py b/django/contrib/syndication/views.py
--- a/django/contrib/syndication/views.py
+++ b/django/contrib/syndication/views.py
@@ -1,3 +1,5 @@
+from inspect import getattr_static, unwrap
+
 from django.contrib.sites.shortcuts import get_current_site
 from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
 from django.http import Http404, HttpResponse
@@ -82,10 +84,21 @@ def _get_dynamic_attr(self, attname, obj, default=None):
             # Check co_argcount rather than try/excepting the function and
             # catching the TypeError, because something inside the function
             # may raise the TypeError. This technique is more accurate.
+            func = unwrap(attr)
             try:
-                code = attr.__code__
+                code = func.__code__
             except AttributeError:
-                code = attr.__call__.__code__
+                func = unwrap(attr.__call__)
+                code = func.__code__
+            # If function doesn't have arguments and it is not a static method,
+            # it was decorated without using @functools.wraps.
+            if not code.co_argcount and not isinstance(
+                getattr_static(self, func.__name__, None), staticmethod
+            ):
+                raise ImproperlyConfigured(
+                    f"Feed method {attname!r} decorated by {func.__name__!r} needs to "
+                    f"use @functools.wraps."
+                )
             if code.co_argcount == 2:  # one argument is 'self'
                 return attr(obj)
             else:

```

## Test Patch

```diff
diff --git a/tests/syndication_tests/feeds.py b/tests/syndication_tests/feeds.py
--- a/tests/syndication_tests/feeds.py
+++ b/tests/syndication_tests/feeds.py
@@ -1,3 +1,5 @@
+from functools import wraps
+
 from django.contrib.syndication import views
 from django.utils import feedgenerator
 from django.utils.timezone import get_fixed_timezone
@@ -5,6 +7,23 @@
 from .models import Article, Entry
 
 
+def wraps_decorator(f):
+    @wraps(f)
+    def wrapper(*args, **kwargs):
+        value = f(*args, **kwargs)
+        return f"{value} -- decorated by @wraps."
+
+    return wrapper
+
+
+def common_decorator(f):
+    def wrapper(*args, **kwargs):
+        value = f(*args, **kwargs)
+        return f"{value} -- common decorated."
+
+    return wrapper
+
+
 class TestRss2Feed(views.Feed):
     title = "My blog"
     description = "A more thorough description of my blog."
@@ -47,11 +66,45 @@ def __call__(self):
     ttl = TimeToLive()
 
 
-class TestRss2FeedWithStaticMethod(TestRss2Feed):
+class TestRss2FeedWithDecoratedMethod(TestRss2Feed):
+    class TimeToLive:
+        @wraps_decorator
+        def __call__(self):
+            return 800
+
+    @staticmethod
+    @wraps_decorator
+    def feed_copyright():
+        return "Copyright (c) 2022, John Doe"
+
+    ttl = TimeToLive()
+
     @staticmethod
     def categories():
         return ("javascript", "vue")
 
+    @wraps_decorator
+    def title(self):
+        return "Overridden title"
+
+    @wraps_decorator
+    def item_title(self, item):
+        return f"Overridden item title: {item.title}"
+
+    @wraps_decorator
+    def description(self, obj):
+        return "Overridden description"
+
+    @wraps_decorator
+    def item_description(self):
+        return "Overridden item description"
+
+
+class TestRss2FeedWithWrongDecoratedMethod(TestRss2Feed):
+    @common_decorator
+    def item_description(self, item):
+        return f"Overridden item description: {item.title}"
+
 
 class TestRss2FeedWithGuidIsPermaLinkTrue(TestRss2Feed):
     def item_guid_is_permalink(self, item):
diff --git a/tests/syndication_tests/tests.py b/tests/syndication_tests/tests.py
--- a/tests/syndication_tests/tests.py
+++ b/tests/syndication_tests/tests.py
@@ -202,11 +202,38 @@ def test_rss2_feed_with_callable_object(self):
         chan = doc.getElementsByTagName("rss")[0].getElementsByTagName("channel")[0]
         self.assertChildNodeContent(chan, {"ttl": "700"})
 
-    def test_rss2_feed_with_static_methods(self):
-        response = self.client.get("/syndication/rss2/with-static-methods/")
+    def test_rss2_feed_with_decorated_methods(self):
+        response = self.client.get("/syndication/rss2/with-decorated-methods/")
         doc = minidom.parseString(response.content)
         chan = doc.getElementsByTagName("rss")[0].getElementsByTagName("channel")[0]
         self.assertCategories(chan, ["javascript", "vue"])
+        self.assertChildNodeContent(
+            chan,
+            {
+                "title": "Overridden title -- decorated by @wraps.",
+                "description": "Overridden description -- decorated by @wraps.",
+                "ttl": "800 -- decorated by @wraps.",
+                "copyright": "Copyright (c) 2022, John Doe -- decorated by @wraps.",
+            },
+        )
+        items = chan.getElementsByTagName("item")
+        self.assertChildNodeContent(
+            items[0],
+            {
+                "title": (
+                    f"Overridden item title: {self.e1.title} -- decorated by @wraps."
+                ),
+                "description": "Overridden item description -- decorated by @wraps.",
+            },
+        )
+
+    def test_rss2_feed_with_wrong_decorated_methods(self):
+        msg = (
+            "Feed method 'item_description' decorated by 'wrapper' needs to use "
+            "@functools.wraps."
+        )
+        with self.assertRaisesMessage(ImproperlyConfigured, msg):
+            self.client.get("/syndication/rss2/with-wrong-decorated-methods/")
 
     def test_rss2_feed_guid_permalink_false(self):
         """
diff --git a/tests/syndication_tests/urls.py b/tests/syndication_tests/urls.py
--- a/tests/syndication_tests/urls.py
+++ b/tests/syndication_tests/urls.py
@@ -7,7 +7,14 @@
     path(
         "syndication/rss2/with-callable-object/", feeds.TestRss2FeedWithCallableObject()
     ),
-    path("syndication/rss2/with-static-methods/", feeds.TestRss2FeedWithStaticMethod()),
+    path(
+        "syndication/rss2/with-decorated-methods/",
+        feeds.TestRss2FeedWithDecoratedMethod(),
+    ),
+    path(
+        "syndication/rss2/with-wrong-decorated-methods/",
+        feeds.TestRss2FeedWithWrongDecoratedMethod(),
+    ),
     path("syndication/rss2/articles/<int:entry_id>/", feeds.TestGetObjectFeed()),
     path(
         "syndication/rss2/guid_ispermalink_true/",

```


## Code snippets

### 1 - django/contrib/syndication/views.py:

Start line: 76, End line: 93

```python
class Feed:

    def _get_dynamic_attr(self, attname, obj, default=None):
        try:
            attr = getattr(self, attname)
        except AttributeError:
            return default
        if callable(attr):
            # Check co_argcount rather than try/excepting the function and
            # catching the TypeError, because something inside the function
            # may raise the TypeError. This technique is more accurate.
            try:
                code = attr.__code__
            except AttributeError:
                code = attr.__call__.__code__
            if code.co_argcount == 2:  # one argument is 'self'
                return attr(obj)
            else:
                return attr()
        return attr
```
### 2 - django/contrib/syndication/views.py:

Start line: 49, End line: 74

```python
class Feed:

    def item_title(self, item):
        # Titles should be double escaped by default (see #6533)
        return escape(str(item))

    def item_description(self, item):
        return str(item)

    def item_link(self, item):
        try:
            return item.get_absolute_url()
        except AttributeError:
            raise ImproperlyConfigured(
                "Give your %s class a get_absolute_url() method, or define an "
                "item_link() method in your Feed class." % item.__class__.__name__
            )

    def item_enclosures(self, item):
        enc_url = self._get_dynamic_attr("item_enclosure_url", item)
        if enc_url:
            enc = feedgenerator.Enclosure(
                url=str(enc_url),
                length=str(self._get_dynamic_attr("item_enclosure_length", item)),
                mime_type=str(self._get_dynamic_attr("item_enclosure_mime_type", item)),
            )
            return [enc]
        return []
```
### 3 - django/contrib/syndication/views.py:

Start line: 167, End line: 222

```python
class Feed:

    def get_feed(self, obj, request):
        # ... other code

        for item in self._get_dynamic_attr("items", obj):
            context = self.get_context_data(
                item=item, site=current_site, obj=obj, request=request
            )
            if title_tmp is not None:
                title = title_tmp.render(context, request)
            else:
                title = self._get_dynamic_attr("item_title", item)
            if description_tmp is not None:
                description = description_tmp.render(context, request)
            else:
                description = self._get_dynamic_attr("item_description", item)
            link = add_domain(
                current_site.domain,
                self._get_dynamic_attr("item_link", item),
                request.is_secure(),
            )
            enclosures = self._get_dynamic_attr("item_enclosures", item)
            author_name = self._get_dynamic_attr("item_author_name", item)
            if author_name is not None:
                author_email = self._get_dynamic_attr("item_author_email", item)
                author_link = self._get_dynamic_attr("item_author_link", item)
            else:
                author_email = author_link = None

            tz = get_default_timezone()

            pubdate = self._get_dynamic_attr("item_pubdate", item)
            if pubdate and is_naive(pubdate):
                pubdate = make_aware(pubdate, tz)

            updateddate = self._get_dynamic_attr("item_updateddate", item)
            if updateddate and is_naive(updateddate):
                updateddate = make_aware(updateddate, tz)

            feed.add_item(
                title=title,
                link=link,
                description=description,
                unique_id=self._get_dynamic_attr("item_guid", item, link),
                unique_id_is_permalink=self._get_dynamic_attr(
                    "item_guid_is_permalink", item
                ),
                enclosures=enclosures,
                pubdate=pubdate,
                updateddate=updateddate,
                author_name=author_name,
                author_email=author_email,
                author_link=author_link,
                comments=self._get_dynamic_attr("item_comments", item),
                categories=self._get_dynamic_attr("item_categories", item),
                item_copyright=self._get_dynamic_attr("item_copyright", item),
                **self.item_extra_kwargs(item),
            )
        return feed
```
### 4 - django/utils/decorators.py:

Start line: 25, End line: 53

```python
def _multi_decorate(decorators, method):
    """
    Decorate `method` with one or more function decorators. `decorators` can be
    a single decorator or an iterable of decorators.
    """
    if hasattr(decorators, "__iter__"):
        # Apply a list/tuple of decorators if 'decorators' is one. Decorator
        # functions are applied so that the call order is the same as the
        # order in which they appear in the iterable.
        decorators = decorators[::-1]
    else:
        decorators = [decorators]

    def _wrapper(self, *args, **kwargs):
        # bound_method has the signature that 'decorator' expects i.e. no
        # 'self' argument, but it's a closure over self so it can call
        # 'func'. Also, wrap method.__get__() in a function because new
        # attributes can't be set on bound method objects, only on functions.
        bound_method = wraps(method)(partial(method.__get__(self, type(self))))
        for dec in decorators:
            bound_method = dec(bound_method)
        return bound_method(*args, **kwargs)

    # Copy any attributes that a decorator adds to the function it decorates.
    for dec in decorators:
        _update_method_wrapper(_wrapper, dec)
    # Preserve any existing attributes of 'method', including the name.
    update_wrapper(_wrapper, method)
    return _wrapper
```
### 5 - django/utils/feedgenerator.py:

Start line: 341, End line: 448

```python
class Atom1Feed(SyndicationFeed):
    # Spec: https://tools.ietf.org/html/rfc4287
    content_type = "application/atom+xml; charset=utf-8"
    ns = "http://www.w3.org/2005/Atom"

    def write(self, outfile, encoding):
        handler = SimplerXMLGenerator(outfile, encoding, short_empty_elements=True)
        handler.startDocument()
        handler.startElement("feed", self.root_attributes())
        self.add_root_elements(handler)
        self.write_items(handler)
        handler.endElement("feed")

    def root_attributes(self):
        if self.feed["language"] is not None:
            return {"xmlns": self.ns, "xml:lang": self.feed["language"]}
        else:
            return {"xmlns": self.ns}

    def add_root_elements(self, handler):
        handler.addQuickElement("title", self.feed["title"])
        handler.addQuickElement(
            "link", "", {"rel": "alternate", "href": self.feed["link"]}
        )
        if self.feed["feed_url"] is not None:
            handler.addQuickElement(
                "link", "", {"rel": "self", "href": self.feed["feed_url"]}
            )
        handler.addQuickElement("id", self.feed["id"])
        handler.addQuickElement("updated", rfc3339_date(self.latest_post_date()))
        if self.feed["author_name"] is not None:
            handler.startElement("author", {})
            handler.addQuickElement("name", self.feed["author_name"])
            if self.feed["author_email"] is not None:
                handler.addQuickElement("email", self.feed["author_email"])
            if self.feed["author_link"] is not None:
                handler.addQuickElement("uri", self.feed["author_link"])
            handler.endElement("author")
        if self.feed["subtitle"] is not None:
            handler.addQuickElement("subtitle", self.feed["subtitle"])
        for cat in self.feed["categories"]:
            handler.addQuickElement("category", "", {"term": cat})
        if self.feed["feed_copyright"] is not None:
            handler.addQuickElement("rights", self.feed["feed_copyright"])

    def write_items(self, handler):
        for item in self.items:
            handler.startElement("entry", self.item_attributes(item))
            self.add_item_elements(handler, item)
            handler.endElement("entry")

    def add_item_elements(self, handler, item):
        handler.addQuickElement("title", item["title"])
        handler.addQuickElement("link", "", {"href": item["link"], "rel": "alternate"})

        if item["pubdate"] is not None:
            handler.addQuickElement("published", rfc3339_date(item["pubdate"]))

        if item["updateddate"] is not None:
            handler.addQuickElement("updated", rfc3339_date(item["updateddate"]))

        # Author information.
        if item["author_name"] is not None:
            handler.startElement("author", {})
            handler.addQuickElement("name", item["author_name"])
            if item["author_email"] is not None:
                handler.addQuickElement("email", item["author_email"])
            if item["author_link"] is not None:
                handler.addQuickElement("uri", item["author_link"])
            handler.endElement("author")

        # Unique ID.
        if item["unique_id"] is not None:
            unique_id = item["unique_id"]
        else:
            unique_id = get_tag_uri(item["link"], item["pubdate"])
        handler.addQuickElement("id", unique_id)

        # Summary.
        if item["description"] is not None:
            handler.addQuickElement("summary", item["description"], {"type": "html"})

        # Enclosures.
        for enclosure in item["enclosures"]:
            handler.addQuickElement(
                "link",
                "",
                {
                    "rel": "enclosure",
                    "href": enclosure.url,
                    "length": enclosure.length,
                    "type": enclosure.mime_type,
                },
            )

        # Categories.
        for cat in item["categories"]:
            handler.addQuickElement("category", "", {"term": cat})

        # Rights.
        if item["item_copyright"] is not None:
            handler.addQuickElement("rights", item["item_copyright"])


# This isolates the decision of what the system default is, so calling code can
# do "feedgenerator.DefaultFeed" instead of "feedgenerator.Rss201rev2Feed".
DefaultFeed = Rss201rev2Feed
```
### 6 - django/contrib/syndication/views.py:

Start line: 27, End line: 47

```python
class Feed:
    feed_type = feedgenerator.DefaultFeed
    title_template = None
    description_template = None
    language = None

    def __call__(self, request, *args, **kwargs):
        try:
            obj = self.get_object(request, *args, **kwargs)
        except ObjectDoesNotExist:
            raise Http404("Feed object does not exist.")
        feedgen = self.get_feed(obj, request)
        response = HttpResponse(content_type=feedgen.content_type)
        if hasattr(self, "item_pubdate") or hasattr(self, "item_updateddate"):
            # if item_pubdate or item_updateddate is defined for the feed, set
            # header so as ConditionalGetMiddleware is able to send 304 NOT MODIFIED
            response.headers["Last-Modified"] = http_date(
                feedgen.latest_post_date().timestamp()
            )
        feedgen.write(response, "utf-8")
        return response
```
### 7 - django/utils/decorators.py:

Start line: 56, End line: 89

```python
def method_decorator(decorator, name=""):
    """
    Convert a function decorator into a method decorator
    """
    # 'obj' can be a class or a function. If 'obj' is a function at the time it
    # is passed to _dec,  it will eventually be a method of the class it is
    # defined on. If 'obj' is a class, the 'name' is required to be the name
    # of the method that will be decorated.
    def _dec(obj):
        if not isinstance(obj, type):
            return _multi_decorate(decorator, obj)
        if not (name and hasattr(obj, name)):
            raise ValueError(
                "The keyword argument `name` must be the name of a method "
                "of the decorated class: %s. Got '%s' instead." % (obj, name)
            )
        method = getattr(obj, name)
        if not callable(method):
            raise TypeError(
                "Cannot decorate '%s' as it isn't a callable attribute of "
                "%s (%s)." % (name, obj, method)
            )
        _wrapper = _multi_decorate(decorator, method)
        setattr(obj, name, _wrapper)
        return obj

    # Don't worry about making _dec look similar to a list/tuple as it's rather
    # meaningless.
    if not hasattr(decorator, "__iter__"):
        update_wrapper(_dec, decorator)
    # Change the name to aid debugging.
    obj = decorator if hasattr(decorator, "__name__") else decorator.__class__
    _dec.__name__ = "method_decorator(%s)" % obj.__name__
    return _dec
```
### 8 - django/utils/feedgenerator.py:

Start line: 281, End line: 338

```python
class Rss201rev2Feed(RssFeed):
    # Spec: https://cyber.harvard.edu/rss/rss.html
    _version = "2.0"

    def add_item_elements(self, handler, item):
        handler.addQuickElement("title", item["title"])
        handler.addQuickElement("link", item["link"])
        if item["description"] is not None:
            handler.addQuickElement("description", item["description"])

        # Author information.
        if item["author_name"] and item["author_email"]:
            handler.addQuickElement(
                "author", "%s (%s)" % (item["author_email"], item["author_name"])
            )
        elif item["author_email"]:
            handler.addQuickElement("author", item["author_email"])
        elif item["author_name"]:
            handler.addQuickElement(
                "dc:creator",
                item["author_name"],
                {"xmlns:dc": "http://purl.org/dc/elements/1.1/"},
            )

        if item["pubdate"] is not None:
            handler.addQuickElement("pubDate", rfc2822_date(item["pubdate"]))
        if item["comments"] is not None:
            handler.addQuickElement("comments", item["comments"])
        if item["unique_id"] is not None:
            guid_attrs = {}
            if isinstance(item.get("unique_id_is_permalink"), bool):
                guid_attrs["isPermaLink"] = str(item["unique_id_is_permalink"]).lower()
            handler.addQuickElement("guid", item["unique_id"], guid_attrs)
        if item["ttl"] is not None:
            handler.addQuickElement("ttl", item["ttl"])

        # Enclosure.
        if item["enclosures"]:
            enclosures = list(item["enclosures"])
            if len(enclosures) > 1:
                raise ValueError(
                    "RSS feed items may only have one enclosure, see "
                    "http://www.rssboard.org/rss-profile#element-channel-item-enclosure"
                )
            enclosure = enclosures[0]
            handler.addQuickElement(
                "enclosure",
                "",
                {
                    "url": enclosure.url,
                    "length": enclosure.length,
                    "type": enclosure.mime_type,
                },
            )

        # Categories.
        for cat in item["categories"]:
            handler.addQuickElement("category", cat)
```
### 9 - django/utils/feedgenerator.py:

Start line: 101, End line: 149

```python
class SyndicationFeed:

    def add_item(
        self,
        title,
        link,
        description,
        author_email=None,
        author_name=None,
        author_link=None,
        pubdate=None,
        comments=None,
        unique_id=None,
        unique_id_is_permalink=None,
        categories=(),
        item_copyright=None,
        ttl=None,
        updateddate=None,
        enclosures=None,
        **kwargs,
    ):
        """
        Add an item to the feed. All args are expected to be strings except
        pubdate and updateddate, which are datetime.datetime objects, and
        enclosures, which is an iterable of instances of the Enclosure class.
        """

        def to_str(s):
            return str(s) if s is not None else s

        categories = categories and [to_str(c) for c in categories]
        self.items.append(
            {
                "title": to_str(title),
                "link": iri_to_uri(link),
                "description": to_str(description),
                "author_email": to_str(author_email),
                "author_name": to_str(author_name),
                "author_link": iri_to_uri(author_link),
                "pubdate": pubdate,
                "updateddate": updateddate,
                "comments": to_str(comments),
                "unique_id": to_str(unique_id),
                "unique_id_is_permalink": unique_id_is_permalink,
                "enclosures": enclosures or (),
                "categories": categories or (),
                "item_copyright": to_str(item_copyright),
                "ttl": to_str(ttl),
                **kwargs,
            }
        )
```
### 10 - django/utils/decorators.py:

Start line: 1, End line: 22

```python
"Functions that help with dynamically creating decorators for views."

from functools import partial, update_wrapper, wraps


class classonlymethod(classmethod):
    def __get__(self, instance, cls=None):
        if instance is not None:
            raise AttributeError(
                "This method is available only on the class, not on instances."
            )
        return super().__get__(instance, cls)


def _update_method_wrapper(_wrapper, decorator):
    # _multi_decorate()'s bound_method isn't available in this scope. Cheat by
    # using it on a dummy function.
    @decorator
    def dummy(*args, **kwargs):
        pass

    update_wrapper(_wrapper, dummy)
```
### 14 - django/contrib/syndication/views.py:

Start line: 95, End line: 120

```python
class Feed:

    def feed_extra_kwargs(self, obj):
        """
        Return an extra keyword arguments dictionary that is used when
        initializing the feed generator.
        """
        return {}

    def item_extra_kwargs(self, item):
        """
        Return an extra keyword arguments dictionary that is used with
        the `add_item` call of the feed generator.
        """
        return {}

    def get_object(self, request, *args, **kwargs):
        return None

    def get_context_data(self, **kwargs):
        """
        Return a dictionary to use as extra context if either
        ``self.description_template`` or ``self.item_template`` are used.

        Default implementation preserves the old behavior
        of using {'obj': item, 'site': current_site} as the context.
        """
        return {"obj": kwargs.get("item"), "site": kwargs.get("site")}
```
### 27 - django/contrib/syndication/views.py:

Start line: 122, End line: 165

```python
class Feed:

    def get_feed(self, obj, request):
        """
        Return a feedgenerator.DefaultFeed object, fully populated, for
        this feed. Raise FeedDoesNotExist for invalid parameters.
        """
        current_site = get_current_site(request)

        link = self._get_dynamic_attr("link", obj)
        link = add_domain(current_site.domain, link, request.is_secure())

        feed = self.feed_type(
            title=self._get_dynamic_attr("title", obj),
            subtitle=self._get_dynamic_attr("subtitle", obj),
            link=link,
            description=self._get_dynamic_attr("description", obj),
            language=self.language or get_language(),
            feed_url=add_domain(
                current_site.domain,
                self._get_dynamic_attr("feed_url", obj) or request.path,
                request.is_secure(),
            ),
            author_name=self._get_dynamic_attr("author_name", obj),
            author_link=self._get_dynamic_attr("author_link", obj),
            author_email=self._get_dynamic_attr("author_email", obj),
            categories=self._get_dynamic_attr("categories", obj),
            feed_copyright=self._get_dynamic_attr("feed_copyright", obj),
            feed_guid=self._get_dynamic_attr("feed_guid", obj),
            ttl=self._get_dynamic_attr("ttl", obj),
            **self.feed_extra_kwargs(obj),
        )

        title_tmp = None
        if self.title_template is not None:
            try:
                title_tmp = loader.get_template(self.title_template)
            except TemplateDoesNotExist:
                pass

        description_tmp = None
        if self.description_template is not None:
            try:
                description_tmp = loader.get_template(self.description_template)
            except TemplateDoesNotExist:
                pass
        # ... other code
```
### 43 - django/contrib/syndication/views.py:

Start line: 1, End line: 24

```python
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
from django.http import Http404, HttpResponse
from django.template import TemplateDoesNotExist, loader
from django.utils import feedgenerator
from django.utils.encoding import iri_to_uri
from django.utils.html import escape
from django.utils.http import http_date
from django.utils.timezone import get_default_timezone, is_naive, make_aware
from django.utils.translation import get_language


def add_domain(domain, url, secure=False):
    protocol = "https" if secure else "http"
    if url.startswith("//"):
        # Support network-path reference (see #16753) - RSS requires a protocol
        url = "%s:%s" % (protocol, url)
    elif not url.startswith(("http://", "https://", "mailto:")):
        url = iri_to_uri("%s://%s%s" % (protocol, domain, url))
    return url


class FeedDoesNotExist(ObjectDoesNotExist):
    pass
```
