# django__django-13230

| **django/django** | `184a6eebb0ef56d5f1b1315a8e666830e37f3f81` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 5246 |
| **Any found context length** | 5246 |
| **Avg pos** | 17.0 |
| **Min pos** | 17 |
| **Max pos** | 17 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/syndication/views.py b/django/contrib/syndication/views.py
--- a/django/contrib/syndication/views.py
+++ b/django/contrib/syndication/views.py
@@ -212,6 +212,7 @@ def get_feed(self, obj, request):
                 author_name=author_name,
                 author_email=author_email,
                 author_link=author_link,
+                comments=self._get_dynamic_attr('item_comments', item),
                 categories=self._get_dynamic_attr('item_categories', item),
                 item_copyright=self._get_dynamic_attr('item_copyright', item),
                 **self.item_extra_kwargs(item)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/syndication/views.py | 215 | 215 | 17 | 2 | 5246


## Problem Statement

```
Add support for item_comments to syndication framework
Description
	
Add comments argument to feed.add_item() in syndication.views so that item_comments can be defined directly without having to take the detour via item_extra_kwargs .
Additionally, comments is already explicitly mentioned in the feedparser, but not implemented in the view.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/utils/feedgenerator.py | 85 | 114| 310 | 310 | 3333 | 
| 2 | **2 django/contrib/syndication/views.py** | 50 | 75| 202 | 512 | 5059 | 
| 3 | **2 django/contrib/syndication/views.py** | 96 | 121| 180 | 692 | 5059 | 
| 4 | 2 django/utils/feedgenerator.py | 116 | 158| 255 | 947 | 5059 | 
| 5 | 2 django/utils/feedgenerator.py | 59 | 83| 249 | 1196 | 5059 | 
| 6 | 2 django/utils/feedgenerator.py | 294 | 393| 926 | 2122 | 5059 | 
| 7 | 2 django/utils/feedgenerator.py | 212 | 239| 306 | 2428 | 5059 | 
| 8 | 2 django/utils/feedgenerator.py | 242 | 291| 475 | 2903 | 5059 | 
| 9 | **2 django/contrib/syndication/views.py** | 29 | 48| 195 | 3098 | 5059 | 
| 10 | 3 django/contrib/gis/feeds.py | 80 | 93| 123 | 3221 | 6357 | 
| 11 | 3 django/utils/feedgenerator.py | 187 | 210| 180 | 3401 | 6357 | 
| 12 | 3 django/contrib/gis/feeds.py | 126 | 141| 128 | 3529 | 6357 | 
| 13 | 3 django/contrib/gis/feeds.py | 32 | 77| 530 | 4059 | 6357 | 
| 14 | **3 django/contrib/syndication/views.py** | 77 | 94| 141 | 4200 | 6357 | 
| 15 | 3 django/contrib/gis/feeds.py | 111 | 123| 136 | 4336 | 6357 | 
| 16 | 3 django/contrib/gis/feeds.py | 96 | 108| 115 | 4451 | 6357 | 
| **-> 17 <-** | **3 django/contrib/syndication/views.py** | 123 | 220| 795 | 5246 | 6357 | 
| 18 | 3 django/utils/feedgenerator.py | 1 | 43| 335 | 5581 | 6357 | 
| 19 | 3 django/utils/feedgenerator.py | 160 | 184| 206 | 5787 | 6357 | 
| 20 | **3 django/contrib/syndication/views.py** | 1 | 26| 220 | 6007 | 6357 | 
| 21 | 3 django/contrib/gis/feeds.py | 1 | 17| 141 | 6148 | 6357 | 
| 22 | 4 django/contrib/gis/views.py | 1 | 21| 155 | 6303 | 6512 | 
| 23 | 4 django/contrib/gis/feeds.py | 19 | 30| 135 | 6438 | 6512 | 
| 24 | 5 django/template/defaultfilters.py | 640 | 668| 212 | 6650 | 12598 | 
| 25 | 6 django/contrib/syndication/apps.py | 1 | 8| 0 | 6650 | 12640 | 
| 26 | 7 django/contrib/sitemaps/__init__.py | 54 | 196| 1026 | 7676 | 14284 | 
| 27 | 8 docs/_ext/djangodocs.py | 173 | 206| 286 | 7962 | 17440 | 
| 28 | 8 docs/_ext/djangodocs.py | 74 | 106| 255 | 8217 | 17440 | 
| 29 | 9 django/contrib/admindocs/views.py | 87 | 115| 285 | 8502 | 20736 | 
| 30 | 9 django/contrib/admindocs/views.py | 33 | 53| 159 | 8661 | 20736 | 
| 31 | 10 django/views/generic/dates.py | 375 | 394| 140 | 8801 | 26177 | 
| 32 | 11 django/db/models/sql/query.py | 1922 | 1966| 364 | 9165 | 48523 | 
| 33 | 11 django/db/models/sql/query.py | 1375 | 1395| 247 | 9412 | 48523 | 
| 34 | 11 django/db/models/sql/query.py | 2368 | 2394| 176 | 9588 | 48523 | 
| 35 | 12 django/utils/translation/__init__.py | 1 | 37| 297 | 9885 | 50859 | 
| 36 | 12 django/contrib/admindocs/views.py | 250 | 315| 573 | 10458 | 50859 | 
| 37 | 12 docs/_ext/djangodocs.py | 26 | 71| 398 | 10856 | 50859 | 
| 38 | 12 django/contrib/sitemaps/__init__.py | 199 | 218| 155 | 11011 | 50859 | 
| 39 | 13 django/contrib/admin/options.py | 1540 | 1626| 760 | 11771 | 69428 | 
| 40 | 13 django/contrib/admindocs/views.py | 183 | 249| 584 | 12355 | 69428 | 
| 41 | 14 django/contrib/sitemaps/views.py | 22 | 45| 250 | 12605 | 70202 | 
| 42 | 15 django/contrib/admin/checks.py | 897 | 945| 416 | 13021 | 79339 | 
| 43 | 15 django/contrib/admin/checks.py | 786 | 807| 190 | 13211 | 79339 | 
| 44 | 16 django/contrib/admindocs/middleware.py | 1 | 29| 234 | 13445 | 79574 | 
| 45 | 17 django/db/backends/mysql/schema.py | 88 | 98| 138 | 13583 | 81070 | 
| 46 | 17 django/contrib/admin/options.py | 2166 | 2201| 315 | 13898 | 81070 | 
| 47 | 18 django/views/generic/edit.py | 1 | 67| 479 | 14377 | 82786 | 
| 48 | 18 django/contrib/admindocs/views.py | 118 | 133| 154 | 14531 | 82786 | 
| 49 | 18 django/views/generic/dates.py | 557 | 572| 123 | 14654 | 82786 | 
| 50 | 18 django/contrib/admindocs/views.py | 136 | 154| 187 | 14841 | 82786 | 
| 51 | 19 django/template/defaulttags.py | 519 | 542| 188 | 15029 | 93929 | 
| 52 | 20 django/contrib/auth/admin.py | 101 | 126| 286 | 15315 | 95655 | 
| 53 | 20 django/db/models/sql/query.py | 1018 | 1048| 325 | 15640 | 95655 | 
| 54 | 20 django/contrib/sitemaps/views.py | 48 | 93| 391 | 16031 | 95655 | 
| 55 | 21 django/db/backends/sqlite3/features.py | 1 | 76| 702 | 16733 | 96357 | 
| 56 | 21 django/contrib/admin/options.py | 1 | 97| 767 | 17500 | 96357 | 
| 57 | 21 django/contrib/admindocs/views.py | 156 | 180| 234 | 17734 | 96357 | 
| 58 | 22 django/contrib/contenttypes/fields.py | 430 | 451| 248 | 17982 | 101790 | 
| 59 | 22 django/contrib/admin/checks.py | 736 | 767| 219 | 18201 | 101790 | 
| 60 | 22 django/db/models/sql/query.py | 1352 | 1373| 250 | 18451 | 101790 | 
| 61 | 22 django/db/models/sql/query.py | 1884 | 1920| 318 | 18769 | 101790 | 
| 62 | 22 django/contrib/admin/checks.py | 809 | 859| 443 | 19212 | 101790 | 
| 63 | 22 django/contrib/admindocs/views.py | 56 | 84| 285 | 19497 | 101790 | 
| 64 | 23 django/utils/cache.py | 38 | 103| 557 | 20054 | 105520 | 
| 65 | 23 django/contrib/admin/options.py | 805 | 835| 216 | 20270 | 105520 | 
| 66 | 23 django/template/defaulttags.py | 1455 | 1488| 246 | 20516 | 105520 | 
| 67 | 23 django/contrib/admin/options.py | 1757 | 1838| 744 | 21260 | 105520 | 
| 68 | 23 django/template/defaulttags.py | 1 | 49| 327 | 21587 | 105520 | 
| 69 | 24 django/contrib/gis/geos/mutable_list.py | 100 | 227| 876 | 22463 | 107774 | 
| 70 | 25 django/contrib/admindocs/urls.py | 1 | 51| 307 | 22770 | 108081 | 
| 71 | 26 django/contrib/postgres/aggregates/general.py | 1 | 68| 411 | 23181 | 108493 | 
| 72 | 26 django/contrib/admindocs/views.py | 318 | 347| 201 | 23382 | 108493 | 
| 73 | 27 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 23792 | 109518 | 
| 74 | 27 django/contrib/admin/checks.py | 342 | 368| 221 | 24013 | 109518 | 
| 75 | 27 django/views/generic/edit.py | 152 | 199| 340 | 24353 | 109518 | 
| 76 | 27 django/contrib/admin/options.py | 100 | 130| 223 | 24576 | 109518 | 
| 77 | 27 docs/_ext/djangodocs.py | 1 | 23| 170 | 24746 | 109518 | 
| 78 | 27 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 25000 | 109518 | 
| 79 | 28 django/db/backends/base/features.py | 313 | 331| 161 | 25161 | 112184 | 
| 80 | 28 django/views/generic/edit.py | 70 | 101| 269 | 25430 | 112184 | 
| 81 | 29 django/contrib/gis/db/backends/spatialite/schema.py | 103 | 124| 212 | 25642 | 113536 | 
| 82 | 29 django/contrib/admin/options.py | 1685 | 1756| 653 | 26295 | 113536 | 
| 83 | 30 django/db/migrations/operations/models.py | 737 | 784| 379 | 26674 | 120431 | 
| 84 | 30 django/contrib/gis/db/backends/spatialite/schema.py | 1 | 35| 316 | 26990 | 120431 | 
| 85 | 30 django/contrib/contenttypes/fields.py | 598 | 629| 278 | 27268 | 120431 | 
| 86 | 30 django/views/generic/dates.py | 521 | 554| 278 | 27546 | 120431 | 
| 87 | 30 django/contrib/admin/checks.py | 416 | 428| 137 | 27683 | 120431 | 
| 88 | 31 django/db/migrations/questioner.py | 207 | 224| 171 | 27854 | 122504 | 
| 89 | 32 django/contrib/admin/helpers.py | 383 | 406| 195 | 28049 | 125674 | 
| 90 | 32 docs/_ext/djangodocs.py | 109 | 170| 526 | 28575 | 125674 | 
| 91 | 33 django/db/backends/sqlite3/schema.py | 307 | 328| 218 | 28793 | 129790 | 
| 92 | 33 django/contrib/admin/options.py | 1995 | 2048| 454 | 29247 | 129790 | 
| 93 | 34 django/db/migrations/serializer.py | 165 | 181| 151 | 29398 | 132461 | 
| 94 | 34 django/views/generic/dates.py | 478 | 518| 358 | 29756 | 132461 | 
| 95 | 35 django/http/multipartparser.py | 154 | 289| 1122 | 30878 | 137501 | 
| 96 | 36 django/db/backends/base/schema.py | 350 | 368| 143 | 31021 | 149343 | 
| 97 | 36 django/contrib/admin/checks.py | 563 | 598| 304 | 31325 | 149343 | 
| 98 | 36 django/views/generic/dates.py | 120 | 163| 285 | 31610 | 149343 | 
| 99 | 36 django/contrib/admin/checks.py | 232 | 245| 161 | 31771 | 149343 | 
| 100 | 36 django/views/generic/edit.py | 202 | 242| 263 | 32034 | 149343 | 
| 101 | 37 django/contrib/admin/templatetags/admin_list.py | 300 | 352| 350 | 32384 | 153285 | 
| 102 | 38 django/views/decorators/clickjacking.py | 1 | 19| 138 | 32522 | 153661 | 
| 103 | 39 django/utils/xmlutils.py | 1 | 34| 247 | 32769 | 153908 | 
| 104 | 40 django/db/backends/mysql/features.py | 110 | 171| 561 | 33330 | 155279 | 
| 105 | 40 django/contrib/admin/templatetags/admin_list.py | 442 | 500| 343 | 33673 | 155279 | 
| 106 | 41 django/contrib/admin/models.py | 1 | 20| 118 | 33791 | 156402 | 
| 107 | 42 django/db/models/options.py | 147 | 206| 587 | 34378 | 163508 | 
| 108 | 43 django/contrib/admindocs/utils.py | 86 | 112| 175 | 34553 | 165413 | 
| 109 | 44 django/views/decorators/csrf.py | 1 | 57| 460 | 35013 | 165873 | 
| 110 | 44 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 35136 | 165873 | 
| 111 | 45 django/contrib/admin/views/main.py | 123 | 212| 861 | 35997 | 170272 | 
| 112 | 45 django/contrib/admin/templatetags/admin_list.py | 1 | 27| 178 | 36175 | 170272 | 
| 113 | 45 django/contrib/admin/views/main.py | 1 | 45| 324 | 36499 | 170272 | 
| 114 | 46 django/db/models/__init__.py | 1 | 53| 619 | 37118 | 170891 | 
| 115 | 47 django/contrib/admin/filters.py | 305 | 368| 627 | 37745 | 174984 | 
| 116 | 47 django/db/models/sql/query.py | 1397 | 1417| 212 | 37957 | 174984 | 
| 117 | 47 django/db/models/options.py | 256 | 288| 331 | 38288 | 174984 | 
| 118 | 47 django/contrib/admin/checks.py | 294 | 327| 381 | 38669 | 174984 | 
| 119 | 47 django/views/generic/dates.py | 293 | 316| 203 | 38872 | 174984 | 
| 120 | 47 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 39358 | 174984 | 
| 121 | 48 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 39575 | 175201 | 
| 122 | 48 django/contrib/admin/options.py | 286 | 375| 641 | 40216 | 175201 | 
| 123 | 48 django/contrib/admin/checks.py | 329 | 340| 138 | 40354 | 175201 | 
| 124 | 49 django/contrib/auth/models.py | 232 | 285| 352 | 40706 | 178473 | 
| 125 | 49 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 40887 | 178473 | 
| 126 | 50 django/contrib/postgres/indexes.py | 159 | 185| 239 | 41126 | 180269 | 
| 127 | 50 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 41340 | 180269 | 
| 128 | 50 django/contrib/admin/options.py | 1627 | 1651| 279 | 41619 | 180269 | 
| 129 | 51 django/contrib/admin/widgets.py | 273 | 308| 382 | 42001 | 184063 | 
| 130 | 52 django/db/migrations/operations/fields.py | 97 | 109| 130 | 42131 | 187161 | 
| 131 | 52 django/contrib/admin/checks.py | 177 | 218| 325 | 42456 | 187161 | 
| 132 | 52 django/contrib/admindocs/utils.py | 1 | 25| 151 | 42607 | 187161 | 
| 133 | 52 django/views/decorators/clickjacking.py | 22 | 54| 238 | 42845 | 187161 | 
| 134 | 52 django/contrib/gis/geos/mutable_list.py | 80 | 98| 160 | 43005 | 187161 | 
| 135 | 52 django/contrib/admin/templatetags/admin_list.py | 215 | 297| 791 | 43796 | 187161 | 
| 136 | 52 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 44411 | 187161 | 
| 137 | 52 django/views/generic/dates.py | 229 | 271| 334 | 44745 | 187161 | 
| 138 | 53 django/db/migrations/autodetector.py | 857 | 891| 339 | 45084 | 198780 | 
| 140 | 54 django/contrib/admindocs/views.py | 1 | 30| 223 | 45579 | 199732 | 


## Patch

```diff
diff --git a/django/contrib/syndication/views.py b/django/contrib/syndication/views.py
--- a/django/contrib/syndication/views.py
+++ b/django/contrib/syndication/views.py
@@ -212,6 +212,7 @@ def get_feed(self, obj, request):
                 author_name=author_name,
                 author_email=author_email,
                 author_link=author_link,
+                comments=self._get_dynamic_attr('item_comments', item),
                 categories=self._get_dynamic_attr('item_categories', item),
                 item_copyright=self._get_dynamic_attr('item_copyright', item),
                 **self.item_extra_kwargs(item)

```

## Test Patch

```diff
diff --git a/tests/syndication_tests/feeds.py b/tests/syndication_tests/feeds.py
--- a/tests/syndication_tests/feeds.py
+++ b/tests/syndication_tests/feeds.py
@@ -29,6 +29,9 @@ def item_pubdate(self, item):
     def item_updateddate(self, item):
         return item.updated
 
+    def item_comments(self, item):
+        return "%scomments" % item.get_absolute_url()
+
     item_author_name = 'Sally Smith'
     item_author_email = 'test@example.com'
     item_author_link = 'http://www.example.com/'
diff --git a/tests/syndication_tests/tests.py b/tests/syndication_tests/tests.py
--- a/tests/syndication_tests/tests.py
+++ b/tests/syndication_tests/tests.py
@@ -136,10 +136,20 @@ def test_rss2_feed(self):
             'guid': 'http://example.com/blog/1/',
             'pubDate': pub_date,
             'author': 'test@example.com (Sally Smith)',
+            'comments': '/blog/1/comments',
         })
         self.assertCategories(items[0], ['python', 'testing'])
         for item in items:
-            self.assertChildNodes(item, ['title', 'link', 'description', 'guid', 'category', 'pubDate', 'author'])
+            self.assertChildNodes(item, [
+                'title',
+                'link',
+                'description',
+                'guid',
+                'category',
+                'pubDate',
+                'author',
+                'comments',
+            ])
             # Assert that <guid> does not have any 'isPermaLink' attribute
             self.assertIsNone(item.getElementsByTagName(
                 'guid')[0].attributes.get('isPermaLink'))

```


## Code snippets

### 1 - django/utils/feedgenerator.py:

Start line: 85, End line: 114

```python
class SyndicationFeed:

    def add_item(self, title, link, description, author_email=None,
                 author_name=None, author_link=None, pubdate=None, comments=None,
                 unique_id=None, unique_id_is_permalink=None, categories=(),
                 item_copyright=None, ttl=None, updateddate=None, enclosures=None, **kwargs):
        """
        Add an item to the feed. All args are expected to be strings except
        pubdate and updateddate, which are datetime.datetime objects, and
        enclosures, which is an iterable of instances of the Enclosure class.
        """
        def to_str(s):
            return str(s) if s is not None else s
        categories = categories and [to_str(c) for c in categories]
        self.items.append({
            'title': to_str(title),
            'link': iri_to_uri(link),
            'description': to_str(description),
            'author_email': to_str(author_email),
            'author_name': to_str(author_name),
            'author_link': iri_to_uri(author_link),
            'pubdate': pubdate,
            'updateddate': updateddate,
            'comments': to_str(comments),
            'unique_id': to_str(unique_id),
            'unique_id_is_permalink': unique_id_is_permalink,
            'enclosures': enclosures or (),
            'categories': categories or (),
            'item_copyright': to_str(item_copyright),
            'ttl': to_str(ttl),
            **kwargs,
        })
```
### 2 - django/contrib/syndication/views.py:

Start line: 50, End line: 75

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
                'Give your %s class a get_absolute_url() method, or define an '
                'item_link() method in your Feed class.' % item.__class__.__name__
            )

    def item_enclosures(self, item):
        enc_url = self._get_dynamic_attr('item_enclosure_url', item)
        if enc_url:
            enc = feedgenerator.Enclosure(
                url=str(enc_url),
                length=str(self._get_dynamic_attr('item_enclosure_length', item)),
                mime_type=str(self._get_dynamic_attr('item_enclosure_mime_type', item)),
            )
            return [enc]
        return []
```
### 3 - django/contrib/syndication/views.py:

Start line: 96, End line: 121

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
        return {'obj': kwargs.get('item'), 'site': kwargs.get('site')}
```
### 4 - django/utils/feedgenerator.py:

Start line: 116, End line: 158

```python
class SyndicationFeed:

    def num_items(self):
        return len(self.items)

    def root_attributes(self):
        """
        Return extra attributes to place on the root (i.e. feed/channel) element.
        Called from write().
        """
        return {}

    def add_root_elements(self, handler):
        """
        Add elements in the root (i.e. feed/channel) element. Called
        from write().
        """
        pass

    def item_attributes(self, item):
        """
        Return extra attributes to place on each item (i.e. item/entry) element.
        """
        return {}

    def add_item_elements(self, handler, item):
        """
        Add elements on each item (i.e. item/entry) element.
        """
        pass

    def write(self, outfile, encoding):
        """
        Output the feed in the given encoding to outfile, which is a file-like
        object. Subclasses should override this.
        """
        raise NotImplementedError('subclasses of SyndicationFeed must provide a write() method')

    def writeString(self, encoding):
        """
        Return the feed in the given encoding as a string.
        """
        s = StringIO()
        self.write(s, encoding)
        return s.getvalue()
```
### 5 - django/utils/feedgenerator.py:

Start line: 59, End line: 83

```python
class SyndicationFeed:
    "Base class for all syndication feeds. Subclasses should provide write()"
    def __init__(self, title, link, description, language=None, author_email=None,
                 author_name=None, author_link=None, subtitle=None, categories=None,
                 feed_url=None, feed_copyright=None, feed_guid=None, ttl=None, **kwargs):
        def to_str(s):
            return str(s) if s is not None else s
        categories = categories and [str(c) for c in categories]
        self.feed = {
            'title': to_str(title),
            'link': iri_to_uri(link),
            'description': to_str(description),
            'language': to_str(language),
            'author_email': to_str(author_email),
            'author_name': to_str(author_name),
            'author_link': iri_to_uri(author_link),
            'subtitle': to_str(subtitle),
            'categories': categories or (),
            'feed_url': iri_to_uri(feed_url),
            'feed_copyright': to_str(feed_copyright),
            'id': feed_guid or link,
            'ttl': to_str(ttl),
            **kwargs,
        }
        self.items = []
```
### 6 - django/utils/feedgenerator.py:

Start line: 294, End line: 393

```python
class Atom1Feed(SyndicationFeed):
    # Spec: https://tools.ietf.org/html/rfc4287
    content_type = 'application/atom+xml; charset=utf-8'
    ns = "http://www.w3.org/2005/Atom"

    def write(self, outfile, encoding):
        handler = SimplerXMLGenerator(outfile, encoding)
        handler.startDocument()
        handler.startElement('feed', self.root_attributes())
        self.add_root_elements(handler)
        self.write_items(handler)
        handler.endElement("feed")

    def root_attributes(self):
        if self.feed['language'] is not None:
            return {"xmlns": self.ns, "xml:lang": self.feed['language']}
        else:
            return {"xmlns": self.ns}

    def add_root_elements(self, handler):
        handler.addQuickElement("title", self.feed['title'])
        handler.addQuickElement("link", "", {"rel": "alternate", "href": self.feed['link']})
        if self.feed['feed_url'] is not None:
            handler.addQuickElement("link", "", {"rel": "self", "href": self.feed['feed_url']})
        handler.addQuickElement("id", self.feed['id'])
        handler.addQuickElement("updated", rfc3339_date(self.latest_post_date()))
        if self.feed['author_name'] is not None:
            handler.startElement("author", {})
            handler.addQuickElement("name", self.feed['author_name'])
            if self.feed['author_email'] is not None:
                handler.addQuickElement("email", self.feed['author_email'])
            if self.feed['author_link'] is not None:
                handler.addQuickElement("uri", self.feed['author_link'])
            handler.endElement("author")
        if self.feed['subtitle'] is not None:
            handler.addQuickElement("subtitle", self.feed['subtitle'])
        for cat in self.feed['categories']:
            handler.addQuickElement("category", "", {"term": cat})
        if self.feed['feed_copyright'] is not None:
            handler.addQuickElement("rights", self.feed['feed_copyright'])

    def write_items(self, handler):
        for item in self.items:
            handler.startElement("entry", self.item_attributes(item))
            self.add_item_elements(handler, item)
            handler.endElement("entry")

    def add_item_elements(self, handler, item):
        handler.addQuickElement("title", item['title'])
        handler.addQuickElement("link", "", {"href": item['link'], "rel": "alternate"})

        if item['pubdate'] is not None:
            handler.addQuickElement('published', rfc3339_date(item['pubdate']))

        if item['updateddate'] is not None:
            handler.addQuickElement('updated', rfc3339_date(item['updateddate']))

        # Author information.
        if item['author_name'] is not None:
            handler.startElement("author", {})
            handler.addQuickElement("name", item['author_name'])
            if item['author_email'] is not None:
                handler.addQuickElement("email", item['author_email'])
            if item['author_link'] is not None:
                handler.addQuickElement("uri", item['author_link'])
            handler.endElement("author")

        # Unique ID.
        if item['unique_id'] is not None:
            unique_id = item['unique_id']
        else:
            unique_id = get_tag_uri(item['link'], item['pubdate'])
        handler.addQuickElement("id", unique_id)

        # Summary.
        if item['description'] is not None:
            handler.addQuickElement("summary", item['description'], {"type": "html"})

        # Enclosures.
        for enclosure in item['enclosures']:
            handler.addQuickElement('link', '', {
                'rel': 'enclosure',
                'href': enclosure.url,
                'length': enclosure.length,
                'type': enclosure.mime_type,
            })

        # Categories.
        for cat in item['categories']:
            handler.addQuickElement("category", "", {"term": cat})

        # Rights.
        if item['item_copyright'] is not None:
            handler.addQuickElement("rights", item['item_copyright'])


# This isolates the decision of what the system default is, so calling code can
# do "feedgenerator.DefaultFeed" instead of "feedgenerator.Rss201rev2Feed".
DefaultFeed = Rss201rev2Feed
```
### 7 - django/utils/feedgenerator.py:

Start line: 212, End line: 239

```python
class RssFeed(SyndicationFeed):

    def add_root_elements(self, handler):
        handler.addQuickElement("title", self.feed['title'])
        handler.addQuickElement("link", self.feed['link'])
        handler.addQuickElement("description", self.feed['description'])
        if self.feed['feed_url'] is not None:
            handler.addQuickElement("atom:link", None, {"rel": "self", "href": self.feed['feed_url']})
        if self.feed['language'] is not None:
            handler.addQuickElement("language", self.feed['language'])
        for cat in self.feed['categories']:
            handler.addQuickElement("category", cat)
        if self.feed['feed_copyright'] is not None:
            handler.addQuickElement("copyright", self.feed['feed_copyright'])
        handler.addQuickElement("lastBuildDate", rfc2822_date(self.latest_post_date()))
        if self.feed['ttl'] is not None:
            handler.addQuickElement("ttl", self.feed['ttl'])

    def endChannelElement(self, handler):
        handler.endElement("channel")


class RssUserland091Feed(RssFeed):
    _version = "0.91"

    def add_item_elements(self, handler, item):
        handler.addQuickElement("title", item['title'])
        handler.addQuickElement("link", item['link'])
        if item['description'] is not None:
            handler.addQuickElement("description", item['description'])
```
### 8 - django/utils/feedgenerator.py:

Start line: 242, End line: 291

```python
class Rss201rev2Feed(RssFeed):
    # Spec: https://cyber.harvard.edu/rss/rss.html
    _version = "2.0"

    def add_item_elements(self, handler, item):
        handler.addQuickElement("title", item['title'])
        handler.addQuickElement("link", item['link'])
        if item['description'] is not None:
            handler.addQuickElement("description", item['description'])

        # Author information.
        if item["author_name"] and item["author_email"]:
            handler.addQuickElement("author", "%s (%s)" % (item['author_email'], item['author_name']))
        elif item["author_email"]:
            handler.addQuickElement("author", item["author_email"])
        elif item["author_name"]:
            handler.addQuickElement(
                "dc:creator", item["author_name"], {"xmlns:dc": "http://purl.org/dc/elements/1.1/"}
            )

        if item['pubdate'] is not None:
            handler.addQuickElement("pubDate", rfc2822_date(item['pubdate']))
        if item['comments'] is not None:
            handler.addQuickElement("comments", item['comments'])
        if item['unique_id'] is not None:
            guid_attrs = {}
            if isinstance(item.get('unique_id_is_permalink'), bool):
                guid_attrs['isPermaLink'] = str(item['unique_id_is_permalink']).lower()
            handler.addQuickElement("guid", item['unique_id'], guid_attrs)
        if item['ttl'] is not None:
            handler.addQuickElement("ttl", item['ttl'])

        # Enclosure.
        if item['enclosures']:
            enclosures = list(item['enclosures'])
            if len(enclosures) > 1:
                raise ValueError(
                    "RSS feed items may only have one enclosure, see "
                    "http://www.rssboard.org/rss-profile#element-channel-item-enclosure"
                )
            enclosure = enclosures[0]
            handler.addQuickElement('enclosure', '', {
                'url': enclosure.url,
                'length': enclosure.length,
                'type': enclosure.mime_type,
            })

        # Categories.
        for cat in item['categories']:
            handler.addQuickElement("category", cat)
```
### 9 - django/contrib/syndication/views.py:

Start line: 29, End line: 48

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
            raise Http404('Feed object does not exist.')
        feedgen = self.get_feed(obj, request)
        response = HttpResponse(content_type=feedgen.content_type)
        if hasattr(self, 'item_pubdate') or hasattr(self, 'item_updateddate'):
            # if item_pubdate or item_updateddate is defined for the feed, set
            # header so as ConditionalGetMiddleware is able to send 304 NOT MODIFIED
            response['Last-Modified'] = http_date(
                timegm(feedgen.latest_post_date().utctimetuple()))
        feedgen.write(response, 'utf-8')
        return response
```
### 10 - django/contrib/gis/feeds.py:

Start line: 80, End line: 93

```python
# ### SyndicationFeed subclasses ###
class GeoRSSFeed(Rss201rev2Feed, GeoFeedMixin):
    def rss_attributes(self):
        attrs = super().rss_attributes()
        attrs['xmlns:georss'] = 'http://www.georss.org/georss'
        return attrs

    def add_item_elements(self, handler, item):
        super().add_item_elements(handler, item)
        self.add_georss_element(handler, item)

    def add_root_elements(self, handler):
        super().add_root_elements(handler)
        self.add_georss_element(handler, self.feed)
```
### 14 - django/contrib/syndication/views.py:

Start line: 77, End line: 94

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
            if code.co_argcount == 2:       # one argument is 'self'
                return attr(obj)
            else:
                return attr()
        return attr
```
### 17 - django/contrib/syndication/views.py:

Start line: 123, End line: 220

```python
class Feed:

    def get_feed(self, obj, request):
        """
        Return a feedgenerator.DefaultFeed object, fully populated, for
        this feed. Raise FeedDoesNotExist for invalid parameters.
        """
        current_site = get_current_site(request)

        link = self._get_dynamic_attr('link', obj)
        link = add_domain(current_site.domain, link, request.is_secure())

        feed = self.feed_type(
            title=self._get_dynamic_attr('title', obj),
            subtitle=self._get_dynamic_attr('subtitle', obj),
            link=link,
            description=self._get_dynamic_attr('description', obj),
            language=self.language or get_language(),
            feed_url=add_domain(
                current_site.domain,
                self._get_dynamic_attr('feed_url', obj) or request.path,
                request.is_secure(),
            ),
            author_name=self._get_dynamic_attr('author_name', obj),
            author_link=self._get_dynamic_attr('author_link', obj),
            author_email=self._get_dynamic_attr('author_email', obj),
            categories=self._get_dynamic_attr('categories', obj),
            feed_copyright=self._get_dynamic_attr('feed_copyright', obj),
            feed_guid=self._get_dynamic_attr('feed_guid', obj),
            ttl=self._get_dynamic_attr('ttl', obj),
            **self.feed_extra_kwargs(obj)
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

        for item in self._get_dynamic_attr('items', obj):
            context = self.get_context_data(item=item, site=current_site,
                                            obj=obj, request=request)
            if title_tmp is not None:
                title = title_tmp.render(context, request)
            else:
                title = self._get_dynamic_attr('item_title', item)
            if description_tmp is not None:
                description = description_tmp.render(context, request)
            else:
                description = self._get_dynamic_attr('item_description', item)
            link = add_domain(
                current_site.domain,
                self._get_dynamic_attr('item_link', item),
                request.is_secure(),
            )
            enclosures = self._get_dynamic_attr('item_enclosures', item)
            author_name = self._get_dynamic_attr('item_author_name', item)
            if author_name is not None:
                author_email = self._get_dynamic_attr('item_author_email', item)
                author_link = self._get_dynamic_attr('item_author_link', item)
            else:
                author_email = author_link = None

            tz = get_default_timezone()

            pubdate = self._get_dynamic_attr('item_pubdate', item)
            if pubdate and is_naive(pubdate):
                pubdate = make_aware(pubdate, tz)

            updateddate = self._get_dynamic_attr('item_updateddate', item)
            if updateddate and is_naive(updateddate):
                updateddate = make_aware(updateddate, tz)

            feed.add_item(
                title=title,
                link=link,
                description=description,
                unique_id=self._get_dynamic_attr('item_guid', item, link),
                unique_id_is_permalink=self._get_dynamic_attr(
                    'item_guid_is_permalink', item),
                enclosures=enclosures,
                pubdate=pubdate,
                updateddate=updateddate,
                author_name=author_name,
                author_email=author_email,
                author_link=author_link,
                categories=self._get_dynamic_attr('item_categories', item),
                item_copyright=self._get_dynamic_attr('item_copyright', item),
                **self.item_extra_kwargs(item)
            )
        return feed
```
### 20 - django/contrib/syndication/views.py:

Start line: 1, End line: 26

```python
from calendar import timegm

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
    protocol = 'https' if secure else 'http'
    if url.startswith('//'):
        # Support network-path reference (see #16753) - RSS requires a protocol
        url = '%s:%s' % (protocol, url)
    elif not url.startswith(('http://', 'https://', 'mailto:')):
        url = iri_to_uri('%s://%s%s' % (protocol, domain, url))
    return url


class FeedDoesNotExist(ObjectDoesNotExist):
    pass
```
