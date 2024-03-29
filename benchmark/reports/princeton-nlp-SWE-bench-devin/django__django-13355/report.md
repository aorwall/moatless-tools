# django__django-13355

| **django/django** | `e26a7a8ef41f0d69951affb21655cdc2cf94a209` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 319 |
| **Any found context length** | 319 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -146,8 +146,14 @@ def merge(*lists):
 
     def __add__(self, other):
         combined = Media()
-        combined._css_lists = self._css_lists + other._css_lists
-        combined._js_lists = self._js_lists + other._js_lists
+        combined._css_lists = self._css_lists[:]
+        combined._js_lists = self._js_lists[:]
+        for item in other._css_lists:
+            if item and item not in self._css_lists:
+                combined._css_lists.append(item)
+        for item in other._js_lists:
+            if item and item not in self._js_lists:
+                combined._js_lists.append(item)
         return combined
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/widgets.py | 149 | 150 | 1 | 1 | 319


## Problem Statement

```
Optimize django.forms.widgets.Media.__add__.
Description
	
While working with another project that make extensive use of django.forms.widgets.Media I discovered that the fix for ticket #30153 has unintended consequences on the performance of Media.__add__. If the number of Media objects added grows beyond a certain point (not sure it may be machine specific) then the performance of all subsequent Media.__add__ calls becomes terrible.
This was causing page load times of several minutes on my development machine. I agree that you need to delay as long as possible as #30153 intends but it seems that there probably should be an upper bound on this so that performance does not suddenly decrease.
Here is some sample code that can reproduce the issue:
from django.forms import Media
import datetime
def create_media(MediaClass):
	'''Creates a simple Media object with only one or two items.'''
	return MediaClass(css={'all': ['main.css']}, js=['main.js'])
start = datetime.datetime.now()
media = create_media(Media)
for i in range(100000):
	media = media + create_media(Media)
	
print('100000 additions took: %s' % (datetime.datetime.now() - start))
On my machine several runs of this code result in times between 1:35 - 1:44 (eg. 1 minute 35 seconds to 1 minute 44 seconds). However, taking away one zero from the number of media objects runs in under a second. Near as I can tell this has to do with the memory used to store these arrays and when it gets past a certain point the performance is awful. Since I am not sure if it is machine specific, for reference my machine is a i7-8700 with 64 GB RAM.
Here is a sample that has a modified Media class that does not have theses issues:
from django.forms import Media
import datetime
def create_media(MediaClass):
	'''Creates a simple Media object with only one or two items.'''
	return MediaClass(css={'all': ['main.css']}, js=['main.js'])
class CustomMedia(Media):
	def __add__(self, other):
		combined = CustomMedia()
		if len(self._css_lists) + len(other._css_lists) > 1000:
			combined._css_lists = [self._css, other._css]
		else:
			combined._css_lists = self._css_lists + other._css_lists
		
		if len(self._js_lists) + len(other._js_lists) > 1000:
			combined._js_lists = [self._js, other._js]
		else:
			combined._js_lists = self._js_lists + other._js_lists
		
		return combined
start = datetime.datetime.now()
media = create_media(CustomMedia)
for i in range(100000):
	media = media + create_media(CustomMedia)
	
print('100000 additions took: %s' % (datetime.datetime.now() - start))
With this change it again runs in under a second. If I increase the number of loops the performance seems to change at a much more expected level.
I set an upper limit on the length allowed before a merge occurs. I'm not sure if the number of additions allowed should be a setting that can be adjusted to meet specific needs or if something that is just "reasonably" high is sufficient. It does appear that limiting the total number of items in the list to about 1000 works best on my machine. I'm also not sure that this is the best solution.
Thanks for your consideration.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/forms/widgets.py** | 114 | 151| 319 | 319 | 8060 | 
| 2 | 2 django/forms/forms.py | 453 | 500| 382 | 701 | 12074 | 
| 3 | **2 django/forms/widgets.py** | 154 | 190| 228 | 929 | 12074 | 
| 4 | **2 django/forms/widgets.py** | 44 | 112| 531 | 1460 | 12074 | 
| 5 | **2 django/forms/widgets.py** | 1 | 41| 293 | 1753 | 12074 | 
| 6 | 3 django/contrib/admin/widgets.py | 444 | 472| 203 | 1956 | 15868 | 
| 7 | 3 django/contrib/admin/widgets.py | 49 | 70| 168 | 2124 | 15868 | 
| 8 | **3 django/forms/widgets.py** | 846 | 889| 315 | 2439 | 15868 | 
| 9 | 4 django/contrib/admin/options.py | 1540 | 1626| 760 | 3199 | 34455 | 
| 10 | 5 django/forms/formsets.py | 28 | 42| 195 | 3394 | 38423 | 
| 11 | 5 django/forms/formsets.py | 394 | 432| 359 | 3753 | 38423 | 
| 12 | 5 django/contrib/admin/options.py | 1758 | 1840| 750 | 4503 | 38423 | 
| 13 | 5 django/contrib/admin/options.py | 1998 | 2051| 454 | 4957 | 38423 | 
| 14 | 6 django/http/request.py | 600 | 631| 241 | 5198 | 43895 | 
| 15 | 6 django/contrib/admin/options.py | 1 | 97| 767 | 5965 | 43895 | 
| 16 | 7 django/db/models/options.py | 256 | 288| 331 | 6296 | 51001 | 
| 17 | 8 django/http/multipartparser.py | 408 | 427| 205 | 6501 | 56041 | 
| 18 | 9 django/forms/models.py | 310 | 349| 387 | 6888 | 67815 | 
| 19 | 10 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 29 | 95| 517 | 7405 | 68535 | 
| 20 | 11 django/conf/global_settings.py | 267 | 349| 800 | 8205 | 74295 | 
| 21 | 11 django/contrib/admin/options.py | 1627 | 1652| 291 | 8496 | 74295 | 
| 22 | 12 django/db/models/fields/files.py | 216 | 337| 960 | 9456 | 78103 | 
| 23 | 13 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 9710 | 83536 | 
| 24 | 14 django/contrib/admin/helpers.py | 69 | 87| 152 | 9862 | 86706 | 
| 25 | 14 django/forms/forms.py | 22 | 50| 219 | 10081 | 86706 | 
| 26 | **14 django/forms/widgets.py** | 740 | 759| 144 | 10225 | 86706 | 
| 27 | 14 django/contrib/admin/options.py | 377 | 429| 504 | 10729 | 86706 | 
| 28 | 14 django/contrib/admin/options.py | 1482 | 1505| 319 | 11048 | 86706 | 
| 29 | 15 django/db/models/fields/__init__.py | 1111 | 1140| 218 | 11266 | 105077 | 
| 30 | 16 django/contrib/staticfiles/storage.py | 251 | 321| 575 | 11841 | 108604 | 
| 31 | 16 django/contrib/contenttypes/fields.py | 598 | 629| 278 | 12119 | 108604 | 
| 32 | 17 django/db/migrations/operations/models.py | 124 | 243| 823 | 12942 | 115499 | 
| 33 | 18 django/forms/fields.py | 535 | 551| 180 | 13122 | 124846 | 
| 34 | 18 django/contrib/contenttypes/fields.py | 656 | 676| 188 | 13310 | 124846 | 
| 35 | 18 django/contrib/admin/options.py | 2169 | 2204| 315 | 13625 | 124846 | 
| 36 | 18 django/contrib/admin/options.py | 100 | 130| 223 | 13848 | 124846 | 
| 37 | 18 django/contrib/admin/options.py | 635 | 652| 128 | 13976 | 124846 | 
| 38 | 18 django/contrib/admin/options.py | 1952 | 1995| 403 | 14379 | 124846 | 
| 39 | 18 django/forms/models.py | 680 | 751| 732 | 15111 | 124846 | 
| 40 | 18 django/contrib/admin/options.py | 476 | 498| 241 | 15352 | 124846 | 
| 41 | 18 django/forms/fields.py | 475 | 500| 174 | 15526 | 124846 | 
| 42 | 18 django/contrib/admin/helpers.py | 32 | 66| 230 | 15756 | 124846 | 
| 43 | 19 django/db/models/base.py | 759 | 808| 456 | 16212 | 141490 | 
| 44 | 19 django/contrib/contenttypes/fields.py | 564 | 596| 330 | 16542 | 141490 | 
| 45 | 20 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 16807 | 143084 | 
| 46 | 20 django/contrib/admin/widgets.py | 1 | 46| 330 | 17137 | 143084 | 
| 47 | 21 django/db/migrations/optimizer.py | 40 | 70| 248 | 17385 | 143674 | 
| 48 | 22 django/db/backends/sqlite3/operations.py | 1 | 38| 258 | 17643 | 146705 | 
| 49 | 22 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 17857 | 146705 | 
| 50 | 23 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 18182 | 147382 | 
| 51 | 24 django/db/models/query.py | 1160 | 1175| 149 | 18331 | 164452 | 
| 52 | 24 django/forms/forms.py | 1 | 19| 119 | 18450 | 164452 | 
| 53 | 24 django/http/multipartparser.py | 154 | 289| 1122 | 19572 | 164452 | 
| 54 | 24 django/db/models/query.py | 1638 | 1744| 1063 | 20635 | 164452 | 
| 55 | 25 django/db/models/deletion.py | 123 | 163| 339 | 20974 | 168278 | 
| 56 | 26 django/db/migrations/operations/fields.py | 85 | 95| 124 | 21098 | 171376 | 
| 57 | 26 django/contrib/admin/options.py | 286 | 375| 641 | 21739 | 171376 | 
| 58 | 26 django/db/models/fields/__init__.py | 1245 | 1263| 180 | 21919 | 171376 | 
| 59 | 26 django/contrib/admin/options.py | 431 | 474| 350 | 22269 | 171376 | 
| 60 | 26 django/http/multipartparser.py | 367 | 406| 258 | 22527 | 171376 | 
| 61 | 27 django/contrib/gis/geos/mutable_list.py | 100 | 227| 876 | 23403 | 173630 | 
| 62 | 27 django/forms/models.py | 776 | 814| 314 | 23717 | 173630 | 
| 63 | **27 django/forms/widgets.py** | 463 | 503| 275 | 23992 | 173630 | 
| 64 | 28 django/forms/boundfield.py | 36 | 51| 149 | 24141 | 175786 | 
| 65 | 29 django/db/migrations/autodetector.py | 356 | 370| 138 | 24279 | 187405 | 
| 66 | **29 django/forms/widgets.py** | 762 | 785| 203 | 24482 | 187405 | 
| 67 | **29 django/forms/widgets.py** | 442 | 460| 195 | 24677 | 187405 | 
| 68 | 29 django/forms/fields.py | 1128 | 1161| 293 | 24970 | 187405 | 
| 69 | 30 django/db/models/functions/mixins.py | 23 | 53| 257 | 25227 | 187823 | 
| 70 | **30 django/forms/widgets.py** | 193 | 275| 617 | 25844 | 187823 | 
| 71 | **30 django/forms/widgets.py** | 373 | 393| 138 | 25982 | 187823 | 
| 72 | 30 django/contrib/admin/options.py | 1127 | 1172| 482 | 26464 | 187823 | 
| 73 | 30 django/db/models/query.py | 1253 | 1273| 253 | 26717 | 187823 | 
| 74 | 31 django/contrib/admin/views/main.py | 1 | 45| 324 | 27041 | 192219 | 
| 75 | 31 django/contrib/gis/geos/mutable_list.py | 261 | 281| 188 | 27229 | 192219 | 
| 76 | 31 django/db/models/options.py | 147 | 206| 587 | 27816 | 192219 | 
| 77 | 31 django/contrib/admin/options.py | 2088 | 2140| 451 | 28267 | 192219 | 


### Hint

```
I confirmed that 959d0c078a1c903cd1e4850932be77c4f0d2294d caused a performance issue. We should be able to optimize this by calling merge more often.
I wonder how many distinct lists of assets you have? In the example above you're merging the same list 100'000 times. When calling merge earlier than at the end it will always be (relatively) easy to construct a failing test case. Maybe deduplicating the list of assets to merge would help? Something like (completely untested) ~/Projects/django$ git diff diff --git a/django/forms/widgets.py b/django/forms/widgets.py index c8ec3c35d5..248c29b4c1 100644 --- a/django/forms/widgets.py +++ b/django/forms/widgets.py @@ -124,7 +124,7 @@ class Media: """ dependency_graph = defaultdict(set) all_items = OrderedSet() - for list_ in filter(None, lists): + for list_ in OrderedSet(filter(None, lists)): head = list_[0] # The first items depend on nothing but have to be part of the # dependency graph to be included in the result. ~/Projects/django$ It might be better for the memory usage to deduplicate earlier (inside Media.__add__ maybe).
j
Replying to Matthias Kestenholz: I wonder how many distinct lists of assets you have? In the example above you're merging the same list 100'000 times. I encountered this issue when using the StreamField from Wagtail (​http://wagtail.io) where we have a use case that dictates creating significant nesting of common forms to be able to edit the stream of output. To be fair, our use case of Wagtail is beyond the expected use of Wagtail as the performance is poor. I have created a customization that helps prevent the performance issue in Wagtail by declaring all stream objects as classes that only generate their forms once per object. This workaround resolves the wagtail performance issue (for our needs) but not the Media.__add__ performance issue in Django. I may be able to share my code if that will help. I do not personally own that code though so I'd have to get permission. We have nested classes of up to four or five levels deep and about 70 unique classes. But because of the way it works the number of forms generated grows exponentially. Since these are all the same classes though the number of distinct lists is limited that I think your de-duplication idea would probably work. I'm not sure whether your code works or not because I actually short circuited in a subclass of a wagtail class to overcome the issue. I might be able to replace the Django Media class via monkey patching with a different version to test but I'm not sure. If I get some time to try this out I'll post a followup comment and try to include some sample code or possibly a patch with some matching tests. Not sure how much time I will be able to dedicate to this. Thanks for looking into this issue.
I apologize for my delay in responding to this. I had to switch tasks for a little while to a different project that took a bit longer than intended. I have done some further research into this issue and have some information based on the actual use. I have 194016 Media objects that are being added together. For the JavaScript references when the lists are de-duplicated in the way you describe above (in __add__ not in merge) the list is trimmed to 13 records. CSS is handled differently and I'm not sure it is correct (see code later in this comment). This provides a performance on my machine of around 3-5 seconds. This is not as fast as my original optimization but is definitely close enough. As far as sharing code, I cannot give you access to my repository. However, I can show you the code where the adding of Media objects is occurring at. First, here is my attempt at improving the Media.__add__ method: def combine_css(destination, css_list): for x in filter(None, css_list): for key in x.keys(): if key not in destination: destination[key] = OrderedSet() for item in x[key]: if item: destination[key].add(item) class ImprovedMedia(forms.Media): def __add__(self, other): combined = ImprovedMedia() combined._css_lists = list(filter(None, self._css_lists + other._css_lists)) css_lists = {} combine_css(css_lists, self._css_lists) combine_css(css_lists, other._css_lists) combined._css_lists = [css_lists] combined._js_lists = list(OrderedSet([tuple(x) for x in filter(None, self._js_lists + other._js_lists)])) return combined I am concerned that my combine_css() function may not be the best implementation because it does keep order but it always results in exactly one new self._css_lists which may not be the correct solution. I don't think I have enough context to know for sure if this is right. Also, I have some reservations about how I use OrderedSet for Media._css_lists because in my initial attempts on the Media._js_lists I had to convert the OrderedSet back into a list to make things work. Second, here is the code where the __add__ method is called: class StructuredStreamBlock(StreamBlock): js_classes = {} # ... def all_media(self): media = ImprovedMedia() all_blocks = self.all_blocks() for block in all_blocks: media += block. return media # ... The original code can be found in wagtail's source code here: ​https://github.com/wagtail/wagtail/blob/e1d3390a1f62ae4b30c0ef38e7cfa5070d8565dc/wagtail/core/blocks/base.py#L74 Unfortunately I could not think of a way to monkey patch the Media class in Django without actually changing the Django code in my virtual environment. Should I attempt at this point to create a fork and a branch (and hopefully eventually a pull request) and see if the change I make break any existing tests or should some others with more familiarity weigh in on this first?
```

## Patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -146,8 +146,14 @@ def merge(*lists):
 
     def __add__(self, other):
         combined = Media()
-        combined._css_lists = self._css_lists + other._css_lists
-        combined._js_lists = self._js_lists + other._js_lists
+        combined._css_lists = self._css_lists[:]
+        combined._js_lists = self._js_lists[:]
+        for item in other._css_lists:
+            if item and item not in self._css_lists:
+                combined._css_lists.append(item)
+        for item in other._js_lists:
+            if item and item not in self._js_lists:
+                combined._js_lists.append(item)
         return combined
 
 

```

## Test Patch

```diff
diff --git a/tests/forms_tests/tests/test_media.py b/tests/forms_tests/tests/test_media.py
--- a/tests/forms_tests/tests/test_media.py
+++ b/tests/forms_tests/tests/test_media.py
@@ -581,6 +581,7 @@ def test_merge_css_three_way(self):
         widget1 = Media(css={'screen': ['c.css'], 'all': ['d.css', 'e.css']})
         widget2 = Media(css={'screen': ['a.css']})
         widget3 = Media(css={'screen': ['a.css', 'b.css', 'c.css'], 'all': ['e.css']})
+        widget4 = Media(css={'all': ['d.css', 'e.css'], 'screen': ['c.css']})
         merged = widget1 + widget2
         # c.css comes before a.css because widget1 + widget2 establishes this
         # order.
@@ -588,3 +589,60 @@ def test_merge_css_three_way(self):
         merged = merged + widget3
         # widget3 contains an explicit ordering of c.css and a.css.
         self.assertEqual(merged._css, {'screen': ['a.css', 'b.css', 'c.css'], 'all': ['d.css', 'e.css']})
+        # Media ordering does not matter.
+        merged = widget1 + widget4
+        self.assertEqual(merged._css, {'screen': ['c.css'], 'all': ['d.css', 'e.css']})
+
+    def test_add_js_deduplication(self):
+        widget1 = Media(js=['a', 'b', 'c'])
+        widget2 = Media(js=['a', 'b'])
+        widget3 = Media(js=['a', 'c', 'b'])
+        merged = widget1 + widget1
+        self.assertEqual(merged._js_lists, [['a', 'b', 'c']])
+        self.assertEqual(merged._js, ['a', 'b', 'c'])
+        merged = widget1 + widget2
+        self.assertEqual(merged._js_lists, [['a', 'b', 'c'], ['a', 'b']])
+        self.assertEqual(merged._js, ['a', 'b', 'c'])
+        # Lists with items in a different order are preserved when added.
+        merged = widget1 + widget3
+        self.assertEqual(merged._js_lists, [['a', 'b', 'c'], ['a', 'c', 'b']])
+        msg = (
+            "Detected duplicate Media files in an opposite order: "
+            "['a', 'b', 'c'], ['a', 'c', 'b']"
+        )
+        with self.assertWarnsMessage(RuntimeWarning, msg):
+            merged._js
+
+    def test_add_css_deduplication(self):
+        widget1 = Media(css={'screen': ['a.css'], 'all': ['b.css']})
+        widget2 = Media(css={'screen': ['c.css']})
+        widget3 = Media(css={'screen': ['a.css'], 'all': ['b.css', 'c.css']})
+        widget4 = Media(css={'screen': ['a.css'], 'all': ['c.css', 'b.css']})
+        merged = widget1 + widget1
+        self.assertEqual(merged._css_lists, [{'screen': ['a.css'], 'all': ['b.css']}])
+        self.assertEqual(merged._css, {'screen': ['a.css'], 'all': ['b.css']})
+        merged = widget1 + widget2
+        self.assertEqual(merged._css_lists, [
+            {'screen': ['a.css'], 'all': ['b.css']},
+            {'screen': ['c.css']},
+        ])
+        self.assertEqual(merged._css, {'screen': ['a.css', 'c.css'], 'all': ['b.css']})
+        merged = widget3 + widget4
+        # Ordering within lists is preserved.
+        self.assertEqual(merged._css_lists, [
+            {'screen': ['a.css'], 'all': ['b.css', 'c.css']},
+            {'screen': ['a.css'], 'all': ['c.css', 'b.css']}
+        ])
+        msg = (
+            "Detected duplicate Media files in an opposite order: "
+            "['b.css', 'c.css'], ['c.css', 'b.css']"
+        )
+        with self.assertWarnsMessage(RuntimeWarning, msg):
+            merged._css
+
+    def test_add_empty(self):
+        media = Media(css={'screen': ['a.css']}, js=['a'])
+        empty_media = Media()
+        merged = media + empty_media
+        self.assertEqual(merged._css_lists, [{'screen': ['a.css']}])
+        self.assertEqual(merged._js_lists, [['a']])

```


## Code snippets

### 1 - django/forms/widgets.py:

Start line: 114, End line: 151

```python
@html_safe
class Media:

    @staticmethod
    def merge(*lists):
        """
        Merge lists while trying to keep the relative order of the elements.
        Warn if the lists have the same elements in a different relative order.

        For static assets it can be important to have them included in the DOM
        in a certain order. In JavaScript you may not be able to reference a
        global or in CSS you might want to override a style.
        """
        dependency_graph = defaultdict(set)
        all_items = OrderedSet()
        for list_ in filter(None, lists):
            head = list_[0]
            # The first items depend on nothing but have to be part of the
            # dependency graph to be included in the result.
            dependency_graph.setdefault(head, set())
            for item in list_:
                all_items.add(item)
                # No self dependencies
                if head != item:
                    dependency_graph[item].add(head)
                head = item
        try:
            return stable_topological_sort(all_items, dependency_graph)
        except CyclicDependencyError:
            warnings.warn(
                'Detected duplicate Media files in an opposite order: {}'.format(
                    ', '.join(repr(list_) for list_ in lists)
                ), MediaOrderConflictWarning,
            )
            return list(all_items)

    def __add__(self, other):
        combined = Media()
        combined._css_lists = self._css_lists + other._css_lists
        combined._js_lists = self._js_lists + other._js_lists
        return combined
```
### 2 - django/forms/forms.py:

Start line: 453, End line: 500

```python
@html_safe
class BaseForm:

    @property
    def media(self):
        """Return all media required to render the widgets on this form."""
        media = Media()
        for field in self.fields.values():
            media = media + field.widget.media
        return media

    def is_multipart(self):
        """
        Return True if the form needs to be multipart-encoded, i.e. it has
        FileInput, or False otherwise.
        """
        return any(field.widget.needs_multipart_form for field in self.fields.values())

    def hidden_fields(self):
        """
        Return a list of all the BoundField objects that are hidden fields.
        Useful for manual form layout in templates.
        """
        return [field for field in self if field.is_hidden]

    def visible_fields(self):
        """
        Return a list of BoundField objects that aren't hidden fields.
        The opposite of the hidden_fields() method.
        """
        return [field for field in self if not field.is_hidden]

    def get_initial_for_field(self, field, field_name):
        """
        Return initial data for field on form. Use initial data from the form
        or the field, in that order. Evaluate callable values.
        """
        value = self.initial.get(field_name, field.initial)
        if callable(value):
            value = value()
        return value


class Form(BaseForm, metaclass=DeclarativeFieldsMetaclass):
    "A collection of Fields, plus their associated data."
    # This is a separate class from BaseForm in order to abstract the way
    # self.fields is specified. This class (Form) is the one that does the
    # fancy metaclass stuff purely for the semantic sugar -- it allows one
    # to define a form using declarative syntax.
    # BaseForm itself has no way of designating self.fields.
```
### 3 - django/forms/widgets.py:

Start line: 154, End line: 190

```python
def media_property(cls):
    def _media(self):
        # Get the media property of the superclass, if it exists
        sup_cls = super(cls, self)
        try:
            base = sup_cls.media
        except AttributeError:
            base = Media()

        # Get the media definition for this class
        definition = getattr(cls, 'Media', None)
        if definition:
            extend = getattr(definition, 'extend', True)
            if extend:
                if extend is True:
                    m = base
                else:
                    m = Media()
                    for medium in extend:
                        m = m + base[medium]
                return m + Media(definition)
            return Media(definition)
        return base
    return property(_media)


class MediaDefiningClass(type):
    """
    Metaclass for classes that can have media definitions.
    """
    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)

        if 'media' not in attrs:
            new_class.media = media_property(new_class)

        return new_class
```
### 4 - django/forms/widgets.py:

Start line: 44, End line: 112

```python
@html_safe
class Media:
    def __init__(self, media=None, css=None, js=None):
        if media is not None:
            css = getattr(media, 'css', {})
            js = getattr(media, 'js', [])
        else:
            if css is None:
                css = {}
            if js is None:
                js = []
        self._css_lists = [css]
        self._js_lists = [js]

    def __repr__(self):
        return 'Media(css=%r, js=%r)' % (self._css, self._js)

    def __str__(self):
        return self.render()

    @property
    def _css(self):
        css = defaultdict(list)
        for css_list in self._css_lists:
            for medium, sublist in css_list.items():
                css[medium].append(sublist)
        return {medium: self.merge(*lists) for medium, lists in css.items()}

    @property
    def _js(self):
        return self.merge(*self._js_lists)

    def render(self):
        return mark_safe('\n'.join(chain.from_iterable(getattr(self, 'render_' + name)() for name in MEDIA_TYPES)))

    def render_js(self):
        return [
            format_html(
                '<script src="{}"></script>',
                self.absolute_path(path)
            ) for path in self._js
        ]

    def render_css(self):
        # To keep rendering order consistent, we can't just iterate over items().
        # We need to sort the keys, and iterate over the sorted list.
        media = sorted(self._css)
        return chain.from_iterable([
            format_html(
                '<link href="{}" type="text/css" media="{}" rel="stylesheet">',
                self.absolute_path(path), medium
            ) for path in self._css[medium]
        ] for medium in media)

    def absolute_path(self, path):
        """
        Given a relative or absolute path to a static asset, return an absolute
        path. An absolute path will be returned unchanged while a relative path
        will be passed to django.templatetags.static.static().
        """
        if path.startswith(('http://', 'https://', '/')):
            return path
        return static(path)

    def __getitem__(self, name):
        """Return a Media object that only contains media of the given type."""
        if name in MEDIA_TYPES:
            return Media(**{str(name): getattr(self, '_' + name)})
        raise KeyError('Unknown media type "%s"' % name)
```
### 5 - django/forms/widgets.py:

Start line: 1, End line: 41

```python
"""
HTML Widget classes
"""

import copy
import datetime
import warnings
from collections import defaultdict
from itertools import chain

from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import datetime_safe, formats
from django.utils.datastructures import OrderedSet
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.topological_sort import (
    CyclicDependencyError, stable_topological_sort,
)
from django.utils.translation import gettext_lazy as _

from .renderers import get_default_renderer

__all__ = (
    'Media', 'MediaDefiningClass', 'Widget', 'TextInput', 'NumberInput',
    'EmailInput', 'URLInput', 'PasswordInput', 'HiddenInput',
    'MultipleHiddenInput', 'FileInput', 'ClearableFileInput', 'Textarea',
    'DateInput', 'DateTimeInput', 'TimeInput', 'CheckboxInput', 'Select',
    'NullBooleanSelect', 'SelectMultiple', 'RadioSelect',
    'CheckboxSelectMultiple', 'MultiWidget', 'SplitDateTimeWidget',
    'SplitHiddenDateTimeWidget', 'SelectDateWidget',
)

MEDIA_TYPES = ('css', 'js')


class MediaOrderConflictWarning(RuntimeWarning):
    pass
```
### 6 - django/contrib/admin/widgets.py:

Start line: 444, End line: 472

```python
class AutocompleteMixin:

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        i18n_name = SELECT2_TRANSLATIONS.get(get_language())
        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else ()
        return forms.Media(
            js=(
                'admin/js/vendor/jquery/jquery%s.js' % extra,
                'admin/js/vendor/select2/select2.full%s.js' % extra,
            ) + i18n_file + (
                'admin/js/jquery.init.js',
                'admin/js/autocomplete.js',
            ),
            css={
                'screen': (
                    'admin/css/vendor/select2/select2%s.css' % extra,
                    'admin/css/autocomplete.css',
                ),
            },
        )


class AutocompleteSelect(AutocompleteMixin, forms.Select):
    pass


class AutocompleteSelectMultiple(AutocompleteMixin, forms.SelectMultiple):
    pass
```
### 7 - django/contrib/admin/widgets.py:

Start line: 49, End line: 70

```python
class AdminDateWidget(forms.DateInput):
    class Media:
        js = [
            'admin/js/calendar.js',
            'admin/js/admin/DateTimeShortcuts.js',
        ]

    def __init__(self, attrs=None, format=None):
        attrs = {'class': 'vDateField', 'size': '10', **(attrs or {})}
        super().__init__(attrs=attrs, format=format)


class AdminTimeWidget(forms.TimeInput):
    class Media:
        js = [
            'admin/js/calendar.js',
            'admin/js/admin/DateTimeShortcuts.js',
        ]

    def __init__(self, attrs=None, format=None):
        attrs = {'class': 'vTimeField', 'size': '8', **(attrs or {})}
        super().__init__(attrs=attrs, format=format)
```
### 8 - django/forms/widgets.py:

Start line: 846, End line: 889

```python
class MultiWidget(Widget):

    def id_for_label(self, id_):
        if id_:
            id_ += '_0'
        return id_

    def value_from_datadict(self, data, files, name):
        return [
            widget.value_from_datadict(data, files, name + widget_name)
            for widget_name, widget in zip(self.widgets_names, self.widgets)
        ]

    def value_omitted_from_data(self, data, files, name):
        return all(
            widget.value_omitted_from_data(data, files, name + widget_name)
            for widget_name, widget in zip(self.widgets_names, self.widgets)
        )

    def decompress(self, value):
        """
        Return a list of decompressed values for the given compressed value.
        The given value can be assumed to be valid, but not necessarily
        non-empty.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def _get_media(self):
        """
        Media for a multiwidget is the combination of all media of the
        subwidgets.
        """
        media = Media()
        for w in self.widgets:
            media = media + w.media
        return media
    media = property(_get_media)

    def __deepcopy__(self, memo):
        obj = super().__deepcopy__(memo)
        obj.widgets = copy.deepcopy(self.widgets)
        return obj

    @property
    def needs_multipart_form(self):
        return any(w.needs_multipart_form for w in self.widgets)
```
### 9 - django/contrib/admin/options.py:

Start line: 1540, End line: 1626

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField("The field %s cannot be referenced." % to_field)

        model = self.model
        opts = model._meta

        if request.method == 'POST' and '_saveasnew' in request.POST:
            object_id = None

        add = object_id is None

        if add:
            if not self.has_add_permission(request):
                raise PermissionDenied
            obj = None

        else:
            obj = self.get_object(request, unquote(object_id), to_field)

            if request.method == 'POST':
                if not self.has_change_permission(request, obj):
                    raise PermissionDenied
            else:
                if not self.has_view_or_change_permission(request, obj):
                    raise PermissionDenied

            if obj is None:
                return self._get_obj_does_not_exist_redirect(request, opts, object_id)

        fieldsets = self.get_fieldsets(request, obj)
        ModelForm = self.get_form(
            request, obj, change=not add, fields=flatten_fieldsets(fieldsets)
        )
        if request.method == 'POST':
            form = ModelForm(request.POST, request.FILES, instance=obj)
            form_validated = form.is_valid()
            if form_validated:
                new_object = self.save_form(request, form, change=not add)
            else:
                new_object = form.instance
            formsets, inline_instances = self._create_formsets(request, new_object, change=not add)
            if all_valid(formsets) and form_validated:
                self.save_model(request, new_object, form, not add)
                self.save_related(request, form, formsets, not add)
                change_message = self.construct_change_message(request, form, formsets, add)
                if add:
                    self.log_addition(request, new_object, change_message)
                    return self.response_add(request, new_object)
                else:
                    self.log_change(request, new_object, change_message)
                    return self.response_change(request, new_object)
            else:
                form_validated = False
        else:
            if add:
                initial = self.get_changeform_initial_data(request)
                form = ModelForm(initial=initial)
                formsets, inline_instances = self._create_formsets(request, form.instance, change=False)
            else:
                form = ModelForm(instance=obj)
                formsets, inline_instances = self._create_formsets(request, obj, change=True)

        if not add and not self.has_change_permission(request, obj):
            readonly_fields = flatten_fieldsets(fieldsets)
        else:
            readonly_fields = self.get_readonly_fields(request, obj)
        adminForm = helpers.AdminForm(
            form,
            list(fieldsets),
            # Clear prepopulated fields on a view-only form to avoid a crash.
            self.get_prepopulated_fields(request, obj) if add or self.has_change_permission(request, obj) else {},
            readonly_fields,
            model_admin=self)
        media = self.media + adminForm.media

        inline_formsets = self.get_inline_formsets(request, formsets, inline_instances, obj)
        for inline_formset in inline_formsets:
            media = media + inline_formset.media

        if add:
            title = _('Add %s')
        elif self.has_change_permission(request, obj):
            title = _('Change %s')
        else:
            title = _('View %s')
        # ... other code
```
### 10 - django/forms/formsets.py:

Start line: 28, End line: 42

```python
class ManagementForm(Form):
    """
    Keep track of how many form instances are displayed on the page. If adding
    new forms via JavaScript, you should increment the count field of this form
    as well.
    """
    def __init__(self, *args, **kwargs):
        self.base_fields[TOTAL_FORM_COUNT] = IntegerField(widget=HiddenInput)
        self.base_fields[INITIAL_FORM_COUNT] = IntegerField(widget=HiddenInput)
        # MIN_NUM_FORM_COUNT and MAX_NUM_FORM_COUNT are output with the rest of
        # the management form, but only for the convenience of client-side
        # code. The POST value of them returned from the client is not checked.
        self.base_fields[MIN_NUM_FORM_COUNT] = IntegerField(required=False, widget=HiddenInput)
        self.base_fields[MAX_NUM_FORM_COUNT] = IntegerField(required=False, widget=HiddenInput)
        super().__init__(*args, **kwargs)
```
### 26 - django/forms/widgets.py:

Start line: 740, End line: 759

```python
class SelectMultiple(Select):
    allow_multiple_selected = True

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def value_omitted_from_data(self, data, files, name):
        # An unselected <select multiple> doesn't appear in POST data, so it's
        # never known if the value is actually omitted.
        return False


class RadioSelect(ChoiceWidget):
    input_type = 'radio'
    template_name = 'django/forms/widgets/radio.html'
    option_template_name = 'django/forms/widgets/radio_option.html'
```
### 63 - django/forms/widgets.py:

Start line: 463, End line: 503

```python
class Textarea(Widget):
    template_name = 'django/forms/widgets/textarea.html'

    def __init__(self, attrs=None):
        # Use slightly better defaults than HTML's 20x2 box
        default_attrs = {'cols': '40', 'rows': '10'}
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)


class DateTimeBaseInput(TextInput):
    format_key = ''
    supports_microseconds = False

    def __init__(self, attrs=None, format=None):
        super().__init__(attrs)
        self.format = format or None

    def format_value(self, value):
        return formats.localize_input(value, self.format or formats.get_format(self.format_key)[0])


class DateInput(DateTimeBaseInput):
    format_key = 'DATE_INPUT_FORMATS'
    template_name = 'django/forms/widgets/date.html'


class DateTimeInput(DateTimeBaseInput):
    format_key = 'DATETIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/datetime.html'


class TimeInput(DateTimeBaseInput):
    format_key = 'TIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/time.html'


# Defined at module level so that CheckboxInput is picklable (#17976)
def boolean_check(v):
    return not (v is False or v is None or v == '')
```
### 66 - django/forms/widgets.py:

Start line: 762, End line: 785

```python
class CheckboxSelectMultiple(ChoiceWidget):
    allow_multiple_selected = True
    input_type = 'checkbox'
    template_name = 'django/forms/widgets/checkbox_select.html'
    option_template_name = 'django/forms/widgets/checkbox_option.html'

    def use_required_attribute(self, initial):
        # Don't use the 'required' attribute because browser validation would
        # require all checkboxes to be checked instead of at least one.
        return False

    def value_omitted_from_data(self, data, files, name):
        # HTML checkboxes don't appear in POST data if not checked, so it's
        # never known if the value is actually omitted.
        return False

    def id_for_label(self, id_, index=None):
        """
        Don't include for="field_0" in <label> because clicking such a label
        would toggle the first checkbox.
        """
        if index is None:
            return ''
        return super().id_for_label(id_, index)
```
### 67 - django/forms/widgets.py:

Start line: 442, End line: 460

```python
class ClearableFileInput(FileInput):

    def value_from_datadict(self, data, files, name):
        upload = super().value_from_datadict(data, files, name)
        if not self.is_required and CheckboxInput().value_from_datadict(
                data, files, self.clear_checkbox_name(name)):

            if upload:
                # If the user contradicts themselves (uploads a new file AND
                # checks the "clear" checkbox), we return a unique marker
                # object that FileField will turn into a ValidationError.
                return FILE_INPUT_CONTRADICTION
            # False signals to clear any existing value, as opposed to just None
            return False
        return upload

    def value_omitted_from_data(self, data, files, name):
        return (
            super().value_omitted_from_data(data, files, name) and
            self.clear_checkbox_name(name) not in data
        )
```
### 70 - django/forms/widgets.py:

Start line: 193, End line: 275

```python
class Widget(metaclass=MediaDefiningClass):
    needs_multipart_form = False  # Determines does this widget need multipart form
    is_localized = False
    is_required = False
    supports_microseconds = True

    def __init__(self, attrs=None):
        self.attrs = {} if attrs is None else attrs.copy()

    def __deepcopy__(self, memo):
        obj = copy.copy(self)
        obj.attrs = self.attrs.copy()
        memo[id(self)] = obj
        return obj

    @property
    def is_hidden(self):
        return self.input_type == 'hidden' if hasattr(self, 'input_type') else False

    def subwidgets(self, name, value, attrs=None):
        context = self.get_context(name, value, attrs)
        yield context['widget']

    def format_value(self, value):
        """
        Return a value as it should appear when rendered in a template.
        """
        if value == '' or value is None:
            return None
        if self.is_localized:
            return formats.localize_input(value)
        return str(value)

    def get_context(self, name, value, attrs):
        return {
            'widget': {
                'name': name,
                'is_hidden': self.is_hidden,
                'required': self.is_required,
                'value': self.format_value(value),
                'attrs': self.build_attrs(self.attrs, attrs),
                'template_name': self.template_name,
            },
        }

    def render(self, name, value, attrs=None, renderer=None):
        """Render the widget as an HTML string."""
        context = self.get_context(name, value, attrs)
        return self._render(self.template_name, context, renderer)

    def _render(self, template_name, context, renderer=None):
        if renderer is None:
            renderer = get_default_renderer()
        return mark_safe(renderer.render(template_name, context))

    def build_attrs(self, base_attrs, extra_attrs=None):
        """Build an attribute dictionary."""
        return {**base_attrs, **(extra_attrs or {})}

    def value_from_datadict(self, data, files, name):
        """
        Given a dictionary of data and this widget's name, return the value
        of this widget or None if it's not provided.
        """
        return data.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in data

    def id_for_label(self, id_):
        """
        Return the HTML ID attribute of this Widget for use by a <label>,
        given the ID of the field. Return None if no ID is available.

        This hook is necessary because some widgets have multiple HTML
        elements and, thus, multiple IDs. In that case, this method should
        return an ID value that corresponds to the first ID in the widget's
        tags.
        """
        return id_

    def use_required_attribute(self, initial):
        return not self.is_hidden
```
### 71 - django/forms/widgets.py:

Start line: 373, End line: 393

```python
class FileInput(Input):
    input_type = 'file'
    needs_multipart_form = True
    template_name = 'django/forms/widgets/file.html'

    def format_value(self, value):
        """File input never renders a value."""
        return

    def value_from_datadict(self, data, files, name):
        "File widgets take data from FILES, not POST"
        return files.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in files

    def use_required_attribute(self, initial):
        return super().use_required_attribute(initial) and not initial


FILE_INPUT_CONTRADICTION = object()
```
