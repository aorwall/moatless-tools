# django__django-13794

| **django/django** | `fe886eee36be8022f34cfe59aa61ff1c21fe01d9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 8829 |
| **Any found context length** | 8829 |
| **Avg pos** | 28.0 |
| **Min pos** | 28 |
| **Max pos** | 28 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -176,6 +176,12 @@ def __mod__(self, rhs):
                 return str(self) % rhs
             return self.__cast() % rhs
 
+        def __add__(self, other):
+            return self.__cast() + other
+
+        def __radd__(self, other):
+            return other + self.__cast()
+
         def __deepcopy__(self, memo):
             # Instances of this class are effectively immutable. It's just a
             # collection of functions. So we don't need to do anything

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/functional.py | 179 | 179 | 28 | 3 | 8829


## Problem Statement

```
add filter is unable to concatenate strings with lazy string
Description
	
If you try to concatenate a string with a lazy string with the add template filter, the result is always the empty string because the add filter generates an exception (TypeError: can only concatenate str (not "__proxy__") to str).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/template/defaultfilters.py | 31 | 53| 191 | 191 | 6230 | 
| 2 | 2 django/template/base.py | 668 | 703| 272 | 463 | 14108 | 
| 3 | 2 django/template/defaultfilters.py | 655 | 683| 212 | 675 | 14108 | 
| 4 | 2 django/template/base.py | 572 | 607| 361 | 1036 | 14108 | 
| 5 | 2 django/template/defaultfilters.py | 56 | 91| 203 | 1239 | 14108 | 
| 6 | **3 django/utils/functional.py** | 194 | 242| 299 | 1538 | 17164 | 
| 7 | 3 django/template/base.py | 705 | 724| 179 | 1717 | 17164 | 
| 8 | 4 django/template/backends/dummy.py | 1 | 53| 325 | 2042 | 17489 | 
| 9 | 5 django/db/models/sql/query.py | 1375 | 1396| 250 | 2292 | 40035 | 
| 10 | 5 django/template/defaultfilters.py | 340 | 424| 499 | 2791 | 40035 | 
| 11 | 5 django/template/base.py | 537 | 569| 207 | 2998 | 40035 | 
| 12 | 6 django/db/models/functions/text.py | 64 | 92| 231 | 3229 | 42371 | 
| 13 | 6 django/template/base.py | 979 | 998| 143 | 3372 | 42371 | 
| 14 | 7 django/template/library.py | 54 | 94| 338 | 3710 | 44908 | 
| 15 | **7 django/utils/functional.py** | 109 | 127| 208 | 3918 | 44908 | 
| 16 | 7 django/template/base.py | 1 | 94| 779 | 4697 | 44908 | 
| 17 | 8 django/contrib/admin/filters.py | 434 | 446| 142 | 4839 | 49031 | 
| 18 | 9 django/utils/translation/template.py | 1 | 32| 376 | 5215 | 50980 | 
| 19 | 9 django/template/base.py | 610 | 666| 439 | 5654 | 50980 | 
| 20 | 9 django/template/defaultfilters.py | 255 | 320| 425 | 6079 | 50980 | 
| 21 | 9 django/template/defaultfilters.py | 206 | 233| 155 | 6234 | 50980 | 
| 22 | 9 django/template/library.py | 1 | 52| 344 | 6578 | 50980 | 
| 23 | 10 django/db/models/query_utils.py | 57 | 108| 396 | 6974 | 53686 | 
| 24 | 10 django/db/models/sql/query.py | 1228 | 1306| 729 | 7703 | 53686 | 
| 25 | 11 django/contrib/admindocs/views.py | 87 | 115| 285 | 7988 | 56982 | 
| 26 | 12 django/template/defaulttags.py | 1455 | 1488| 246 | 8234 | 68125 | 
| 27 | 12 django/template/defaultfilters.py | 323 | 337| 111 | 8345 | 68125 | 
| **-> 28 <-** | **12 django/utils/functional.py** | 129 | 191| 484 | 8829 | 68125 | 
| 29 | 12 django/contrib/admin/filters.py | 62 | 115| 411 | 9240 | 68125 | 
| 30 | 12 django/db/models/functions/text.py | 95 | 116| 202 | 9442 | 68125 | 
| 31 | 12 django/template/defaulttags.py | 97 | 132| 224 | 9666 | 68125 | 
| 32 | 12 django/contrib/admin/filters.py | 448 | 477| 226 | 9892 | 68125 | 
| 33 | **12 django/utils/functional.py** | 76 | 107| 235 | 10127 | 68125 | 
| 34 | 13 django/template/smartif.py | 150 | 209| 423 | 10550 | 69651 | 
| 35 | 13 django/template/defaulttags.py | 633 | 680| 331 | 10881 | 69651 | 
| 36 | 14 django/utils/encoding.py | 48 | 67| 156 | 11037 | 72013 | 
| 37 | 14 django/contrib/admin/filters.py | 20 | 59| 295 | 11332 | 72013 | 
| 38 | 14 django/template/defaultfilters.py | 439 | 474| 233 | 11565 | 72013 | 
| 39 | 14 django/db/models/sql/query.py | 1308 | 1373| 772 | 12337 | 72013 | 
| 40 | 15 django/utils/safestring.py | 21 | 37| 112 | 12449 | 72399 | 
| 41 | 15 django/template/base.py | 381 | 404| 230 | 12679 | 72399 | 
| 42 | 16 django/template/loader_tags.py | 254 | 273| 239 | 12918 | 74978 | 
| 43 | 16 django/template/defaultfilters.py | 1 | 28| 207 | 13125 | 74978 | 
| 44 | 16 django/contrib/admin/filters.py | 399 | 421| 211 | 13336 | 74978 | 
| 45 | 17 django/utils/regex_helper.py | 286 | 352| 484 | 13820 | 77619 | 
| 46 | 17 django/contrib/admin/filters.py | 280 | 304| 217 | 14037 | 77619 | 
| 47 | 17 django/template/defaulttags.py | 824 | 839| 163 | 14200 | 77619 | 
| 48 | 17 django/db/models/sql/query.py | 1398 | 1418| 247 | 14447 | 77619 | 
| 49 | 17 django/template/loader_tags.py | 276 | 322| 392 | 14839 | 77619 | 
| 50 | 17 django/template/library.py | 96 | 134| 324 | 15163 | 77619 | 
| 51 | 17 django/template/base.py | 174 | 197| 152 | 15315 | 77619 | 
| 52 | 18 django/contrib/admin/templatetags/admin_list.py | 1 | 25| 175 | 15490 | 81279 | 
| 53 | 18 django/contrib/admin/filters.py | 209 | 226| 190 | 15680 | 81279 | 
| 54 | 18 django/template/smartif.py | 114 | 147| 188 | 15868 | 81279 | 
| 55 | 18 django/contrib/admin/filters.py | 373 | 397| 294 | 16162 | 81279 | 
| 56 | 18 django/template/defaultfilters.py | 603 | 653| 322 | 16484 | 81279 | 
| 57 | 19 django/views/debug.py | 64 | 77| 115 | 16599 | 85870 | 
| 58 | 20 django/contrib/admin/views/main.py | 123 | 212| 861 | 17460 | 90266 | 
| 59 | 21 django/utils/translation/__init__.py | 152 | 196| 335 | 17795 | 92606 | 
| 60 | 22 django/template/utils.py | 1 | 62| 401 | 18196 | 93314 | 
| 61 | 23 django/template/loaders/cached.py | 67 | 98| 225 | 18421 | 94032 | 
| 62 | 24 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 18826 | 94437 | 
| 63 | 24 django/template/defaultfilters.py | 236 | 252| 116 | 18942 | 94437 | 
| 64 | 25 django/db/models/query.py | 968 | 983| 124 | 19066 | 111747 | 
| 65 | 25 django/views/debug.py | 181 | 193| 143 | 19209 | 111747 | 
| 66 | 26 django/template/response.py | 45 | 58| 120 | 19329 | 112838 | 
| 67 | 26 django/contrib/admin/filters.py | 229 | 244| 196 | 19525 | 112838 | 
| 68 | 27 django/template/loader.py | 52 | 67| 117 | 19642 | 113254 | 
| 69 | 28 django/db/models/aggregates.py | 70 | 96| 266 | 19908 | 114555 | 
| 70 | 28 django/template/defaultfilters.py | 507 | 600| 527 | 20435 | 114555 | 
| 71 | 28 django/template/defaulttags.py | 1065 | 1093| 225 | 20660 | 114555 | 
| 72 | 29 django/templatetags/i18n.py | 70 | 97| 229 | 20889 | 118581 | 
| 73 | 29 django/template/base.py | 332 | 378| 455 | 21344 | 118581 | 
| 74 | 29 django/contrib/admin/filters.py | 307 | 370| 627 | 21971 | 118581 | 
| 75 | 29 django/template/defaulttags.py | 866 | 900| 218 | 22189 | 118581 | 
| 76 | 29 django/template/defaultfilters.py | 427 | 436| 111 | 22300 | 118581 | 
| 77 | 30 django/core/management/templates.py | 40 | 56| 181 | 22481 | 121256 | 
| 78 | **30 django/utils/functional.py** | 245 | 345| 905 | 23386 | 121256 | 
| 79 | 31 django/contrib/admin/templatetags/log.py | 26 | 60| 317 | 23703 | 121735 | 
| 80 | 32 django/http/multipartparser.py | 348 | 372| 169 | 23872 | 126820 | 
| 81 | 33 django/template/context_processors.py | 1 | 32| 218 | 24090 | 127309 | 
| 82 | 33 django/template/defaulttags.py | 1 | 49| 327 | 24417 | 127309 | 
| 83 | 33 django/contrib/admin/filters.py | 118 | 159| 365 | 24782 | 127309 | 
| 84 | 33 django/contrib/admin/filters.py | 266 | 278| 149 | 24931 | 127309 | 
| 85 | 33 django/template/smartif.py | 1 | 40| 299 | 25230 | 127309 | 
| 86 | 34 django/forms/widgets.py | 114 | 157| 367 | 25597 | 135417 | 
| 87 | 35 django/template/backends/jinja2.py | 1 | 51| 341 | 25938 | 136239 | 
| 88 | 36 django/template/backends/django.py | 48 | 76| 210 | 26148 | 137095 | 
| 89 | 36 django/template/response.py | 1 | 43| 389 | 26537 | 137095 | 
| 90 | 36 django/template/loader_tags.py | 224 | 251| 239 | 26776 | 137095 | 
| 91 | 36 django/contrib/admin/filters.py | 246 | 263| 184 | 26960 | 137095 | 
| 92 | 37 django/template/engine.py | 1 | 53| 388 | 27348 | 138405 | 
| 93 | 37 django/template/defaulttags.py | 501 | 516| 142 | 27490 | 138405 | 
| 94 | 37 django/template/base.py | 507 | 535| 244 | 27734 | 138405 | 
| 95 | 37 django/db/models/sql/query.py | 1420 | 1440| 212 | 27946 | 138405 | 
| 96 | 37 django/template/defaultfilters.py | 182 | 203| 206 | 28152 | 138405 | 
| 97 | 38 django/template/exceptions.py | 1 | 43| 262 | 28414 | 138668 | 
| 98 | 38 django/template/loader_tags.py | 109 | 124| 155 | 28569 | 138668 | 
| 99 | 38 django/template/defaultfilters.py | 707 | 784| 443 | 29012 | 138668 | 
| 100 | 38 django/contrib/admin/filters.py | 1 | 17| 127 | 29139 | 138668 | 
| 101 | 39 django/shortcuts.py | 1 | 20| 155 | 29294 | 139765 | 
| 102 | 39 django/template/defaultfilters.py | 868 | 923| 539 | 29833 | 139765 | 
| 103 | 39 django/db/models/sql/query.py | 2402 | 2428| 176 | 30009 | 139765 | 
| 104 | 39 django/utils/encoding.py | 102 | 115| 130 | 30139 | 139765 | 
| 105 | 40 django/db/models/expressions.py | 33 | 147| 836 | 30975 | 150686 | 
| 106 | 40 django/contrib/admin/filters.py | 424 | 431| 107 | 31082 | 150686 | 
| 107 | 41 django/contrib/admin/widgets.py | 1 | 46| 330 | 31412 | 154480 | 
| 108 | 41 django/contrib/admin/filters.py | 162 | 207| 427 | 31839 | 154480 | 
| 109 | 41 django/template/defaultfilters.py | 477 | 504| 164 | 32003 | 154480 | 
| 110 | 41 django/template/defaulttags.py | 519 | 542| 188 | 32191 | 154480 | 
| 111 | 41 django/template/base.py | 140 | 172| 255 | 32446 | 154480 | 
| 112 | 41 django/template/smartif.py | 68 | 111| 463 | 32909 | 154480 | 
| 113 | 41 django/template/defaulttags.py | 264 | 281| 196 | 33105 | 154480 | 
| 114 | 42 django/utils/text.py | 277 | 318| 289 | 33394 | 157931 | 
| 115 | 42 django/views/debug.py | 195 | 243| 467 | 33861 | 157931 | 
| 116 | 42 django/template/defaulttags.py | 842 | 863| 122 | 33983 | 157931 | 
| 117 | 42 django/template/defaulttags.py | 157 | 218| 532 | 34515 | 157931 | 
| 118 | 42 django/db/models/query.py | 1320 | 1338| 186 | 34701 | 157931 | 
| 119 | 42 django/template/backends/django.py | 79 | 111| 225 | 34926 | 157931 | 
| 120 | 43 django/contrib/admin/options.py | 1038 | 1060| 198 | 35124 | 176524 | 
| 121 | 43 django/template/smartif.py | 43 | 65| 150 | 35274 | 176524 | 
| 122 | 44 django/contrib/gis/gdal/layer.py | 169 | 182| 177 | 35451 | 178443 | 
| 123 | 45 django/contrib/postgres/utils.py | 1 | 30| 218 | 35669 | 178661 | 
| 124 | 46 django/utils/feedgenerator.py | 85 | 114| 310 | 35979 | 181994 | 
| 125 | 46 django/template/loaders/cached.py | 1 | 65| 497 | 36476 | 181994 | 
| 126 | 46 django/http/multipartparser.py | 322 | 346| 179 | 36655 | 181994 | 
| 127 | 46 django/utils/text.py | 410 | 426| 104 | 36759 | 181994 | 
| 128 | 46 django/template/base.py | 407 | 424| 127 | 36886 | 181994 | 
| 129 | 46 django/template/utils.py | 64 | 90| 195 | 37081 | 181994 | 
| 130 | 46 django/template/defaultfilters.py | 787 | 818| 287 | 37368 | 181994 | 
| 131 | 46 django/utils/translation/template.py | 61 | 228| 1428 | 38796 | 181994 | 
| 132 | 46 django/db/models/sql/query.py | 953 | 999| 439 | 39235 | 181994 | 
| 133 | 46 django/contrib/admin/options.py | 377 | 429| 504 | 39739 | 181994 | 
| 134 | 46 django/template/defaulttags.py | 545 | 630| 803 | 40542 | 181994 | 
| 135 | 46 django/template/library.py | 237 | 309| 626 | 41168 | 181994 | 
| 136 | 46 django/db/models/query_utils.py | 312 | 352| 286 | 41454 | 181994 | 
| 137 | 46 django/template/defaulttags.py | 1096 | 1140| 370 | 41824 | 181994 | 
| 138 | 46 django/views/debug.py | 80 | 112| 258 | 42082 | 181994 | 
| 139 | 47 django/db/models/fields/related.py | 343 | 360| 163 | 42245 | 195870 | 
| 140 | 47 django/template/defaulttags.py | 1327 | 1391| 504 | 42749 | 195870 | 
| 141 | 48 django/utils/html.py | 150 | 176| 149 | 42898 | 198972 | 
| 142 | 48 django/template/defaulttags.py | 1292 | 1324| 287 | 43185 | 198972 | 
| 143 | 48 django/template/base.py | 486 | 505| 187 | 43372 | 198972 | 
| 144 | 48 django/db/models/expressions.py | 417 | 439| 204 | 43576 | 198972 | 
| 145 | 48 django/http/multipartparser.py | 374 | 413| 258 | 43834 | 198972 | 
| 146 | 48 django/utils/translation/__init__.py | 70 | 149| 489 | 44323 | 198972 | 


### Hint

```
Tests.
Thanks. I attached a test.
```

## Patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -176,6 +176,12 @@ def __mod__(self, rhs):
                 return str(self) % rhs
             return self.__cast() % rhs
 
+        def __add__(self, other):
+            return self.__cast() + other
+
+        def __radd__(self, other):
+            return other + self.__cast()
+
         def __deepcopy__(self, memo):
             # Instances of this class are effectively immutable. It's just a
             # collection of functions. So we don't need to do anything

```

## Test Patch

```diff
diff --git a/tests/template_tests/filter_tests/test_add.py b/tests/template_tests/filter_tests/test_add.py
--- a/tests/template_tests/filter_tests/test_add.py
+++ b/tests/template_tests/filter_tests/test_add.py
@@ -2,6 +2,7 @@
 
 from django.template.defaultfilters import add
 from django.test import SimpleTestCase
+from django.utils.translation import gettext_lazy
 
 from ..utils import setup
 
@@ -46,6 +47,22 @@ def test_add07(self):
         output = self.engine.render_to_string('add07', {'d': date(2000, 1, 1), 't': timedelta(10)})
         self.assertEqual(output, 'Jan. 11, 2000')
 
+    @setup({'add08': '{{ s1|add:lazy_s2 }}'})
+    def test_add08(self):
+        output = self.engine.render_to_string(
+            'add08',
+            {'s1': 'string', 'lazy_s2': gettext_lazy('lazy')},
+        )
+        self.assertEqual(output, 'stringlazy')
+
+    @setup({'add09': '{{ lazy_s1|add:lazy_s2 }}'})
+    def test_add09(self):
+        output = self.engine.render_to_string(
+            'add09',
+            {'lazy_s1': gettext_lazy('string'), 'lazy_s2': gettext_lazy('lazy')},
+        )
+        self.assertEqual(output, 'stringlazy')
+
 
 class FunctionTests(SimpleTestCase):
 
diff --git a/tests/utils_tests/test_functional.py b/tests/utils_tests/test_functional.py
--- a/tests/utils_tests/test_functional.py
+++ b/tests/utils_tests/test_functional.py
@@ -184,6 +184,11 @@ class Foo:
         with self.assertRaisesMessage(TypeError, msg):
             Foo().cp
 
+    def test_lazy_add(self):
+        lazy_4 = lazy(lambda: 4, int)
+        lazy_5 = lazy(lambda: 5, int)
+        self.assertEqual(lazy_4() + lazy_5(), 9)
+
     def test_lazy_equality(self):
         """
         == and != work correctly for Promises.

```


## Code snippets

### 1 - django/template/defaultfilters.py:

Start line: 31, End line: 53

```python
#######################
# STRING DECORATOR    #
#######################

def stringfilter(func):
    """
    Decorator for filters which should only receive strings. The object
    passed as the first positional argument will be converted to a string.
    """
    def _dec(*args, **kwargs):
        args = list(args)
        args[0] = str(args[0])
        if (isinstance(args[0], SafeData) and
                getattr(_dec._decorated_function, 'is_safe', False)):
            return mark_safe(func(*args, **kwargs))
        return func(*args, **kwargs)

    # Include a reference to the real function (used to check original
    # arguments by the template parser, and to bear the 'is_safe' attribute
    # when multiple decorators are applied).
    _dec._decorated_function = getattr(func, '_decorated_function', func)

    return wraps(func)(_dec)
```
### 2 - django/template/base.py:

Start line: 668, End line: 703

```python
class FilterExpression:

    def resolve(self, context, ignore_failures=False):
        if isinstance(self.var, Variable):
            try:
                obj = self.var.resolve(context)
            except VariableDoesNotExist:
                if ignore_failures:
                    obj = None
                else:
                    string_if_invalid = context.template.engine.string_if_invalid
                    if string_if_invalid:
                        if '%s' in string_if_invalid:
                            return string_if_invalid % self.var
                        else:
                            return string_if_invalid
                    else:
                        obj = string_if_invalid
        else:
            obj = self.var
        for func, args in self.filters:
            arg_vals = []
            for lookup, arg in args:
                if not lookup:
                    arg_vals.append(mark_safe(arg))
                else:
                    arg_vals.append(arg.resolve(context))
            if getattr(func, 'expects_localtime', False):
                obj = template_localtime(obj, context.use_tz)
            if getattr(func, 'needs_autoescape', False):
                new_obj = func(obj, autoescape=context.autoescape, *arg_vals)
            else:
                new_obj = func(obj, *arg_vals)
            if getattr(func, 'is_safe', False) and isinstance(obj, SafeData):
                obj = mark_safe(new_obj)
            else:
                obj = new_obj
        return obj
```
### 3 - django/template/defaultfilters.py:

Start line: 655, End line: 683

```python
@register.filter(is_safe=True, needs_autoescape=True)
def unordered_list(value, autoescape=True):
    # ... other code

    def list_formatter(item_list, tabs=1):
        indent = '\t' * tabs
        output = []
        for item, children in walk_items(item_list):
            sublist = ''
            if children:
                sublist = '\n%s<ul>\n%s\n%s</ul>\n%s' % (
                    indent, list_formatter(children, tabs + 1), indent, indent)
            output.append('%s<li>%s%s</li>' % (
                indent, escaper(item), sublist))
        return '\n'.join(output)

    return mark_safe(list_formatter(value))


###################
# INTEGERS        #
###################

@register.filter(is_safe=False)
def add(value, arg):
    """Add the arg to the value."""
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        try:
            return value + arg
        except Exception:
            return ''
```
### 4 - django/template/base.py:

Start line: 572, End line: 607

```python
# This only matches constant *strings* (things in quotes or marked for
# translation). Numbers are treated as variables for implementation reasons
# (so that they retain their type when passed to filters).
constant_string = r"""
(?:%(i18n_open)s%(strdq)s%(i18n_close)s|
%(i18n_open)s%(strsq)s%(i18n_close)s|
%(strdq)s|
%(strsq)s)
""" % {
    'strdq': r'"[^"\\]*(?:\\.[^"\\]*)*"',  # double-quoted string
    'strsq': r"'[^'\\]*(?:\\.[^'\\]*)*'",  # single-quoted string
    'i18n_open': re.escape("_("),
    'i18n_close': re.escape(")"),
}
constant_string = constant_string.replace("\n", "")

filter_raw_string = r"""
^(?P<constant>%(constant)s)|
^(?P<var>[%(var_chars)s]+|%(num)s)|
 (?:\s*%(filter_sep)s\s*
     (?P<filter_name>\w+)
         (?:%(arg_sep)s
             (?:
              (?P<constant_arg>%(constant)s)|
              (?P<var_arg>[%(var_chars)s]+|%(num)s)
             )
         )?
 )""" % {
    'constant': constant_string,
    'num': r'[-+\.]?\d[\d\.e]*',
    'var_chars': r'\w\.',
    'filter_sep': re.escape(FILTER_SEPARATOR),
    'arg_sep': re.escape(FILTER_ARGUMENT_SEPARATOR),
}

filter_re = _lazy_re_compile(filter_raw_string, re.VERBOSE)
```
### 5 - django/template/defaultfilters.py:

Start line: 56, End line: 91

```python
###################
# STRINGS         #
###################

@register.filter(is_safe=True)
@stringfilter
def addslashes(value):
    """
    Add slashes before quotes. Useful for escaping strings in CSV, for
    example. Less useful for escaping JavaScript; use the ``escapejs``
    filter instead.
    """
    return value.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")


@register.filter(is_safe=True)
@stringfilter
def capfirst(value):
    """Capitalize the first character of the value."""
    return value and value[0].upper() + value[1:]


@register.filter("escapejs")
@stringfilter
def escapejs_filter(value):
    """Hex encode characters for use in JavaScript strings."""
    return escapejs(value)


@register.filter(is_safe=True)
def json_script(value, element_id):
    """
    Output value JSON-encoded, wrapped in a <script type="application/json">
    tag.
    """
    return _json_script(value, element_id)
```
### 6 - django/utils/functional.py:

Start line: 194, End line: 242

```python
def _lazy_proxy_unpickle(func, args, kwargs, *resultclasses):
    return lazy(func, *resultclasses)(*args, **kwargs)


def lazystr(text):
    """
    Shortcut for the common case of a lazy callable that returns str.
    """
    return lazy(str, str)(text)


def keep_lazy(*resultclasses):
    """
    A decorator that allows a function to be called with one or more lazy
    arguments. If none of the args are lazy, the function is evaluated
    immediately, otherwise a __proxy__ is returned that will evaluate the
    function when needed.
    """
    if not resultclasses:
        raise TypeError("You must pass at least one argument to keep_lazy().")

    def decorator(func):
        lazy_func = lazy(func, *resultclasses)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if any(isinstance(arg, Promise) for arg in itertools.chain(args, kwargs.values())):
                return lazy_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def keep_lazy_text(func):
    """
    A decorator for functions that accept lazy arguments and return text.
    """
    return keep_lazy(str)(func)


empty = object()


def new_method_proxy(func):
    def inner(self, *args):
        if self._wrapped is empty:
            self._setup()
        return func(self._wrapped, *args)
    return inner
```
### 7 - django/template/base.py:

Start line: 705, End line: 724

```python
class FilterExpression:

    def args_check(name, func, provided):
        provided = list(provided)
        # First argument, filter input, is implied.
        plen = len(provided) + 1
        # Check to see if a decorator is providing the real function.
        func = inspect.unwrap(func)

        args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
        alen = len(args)
        dlen = len(defaults or [])
        # Not enough OR Too many
        if plen < (alen - dlen) or plen > alen:
            raise TemplateSyntaxError("%s requires %d arguments, %d provided" %
                                      (name, alen - dlen, plen))

        return True
    args_check = staticmethod(args_check)

    def __str__(self):
        return self.token
```
### 8 - django/template/backends/dummy.py:

Start line: 1, End line: 53

```python
import string

from django.core.exceptions import ImproperlyConfigured
from django.template import Origin, TemplateDoesNotExist
from django.utils.html import conditional_escape

from .base import BaseEngine
from .utils import csrf_input_lazy, csrf_token_lazy


class TemplateStrings(BaseEngine):

    app_dirname = 'template_strings'

    def __init__(self, params):
        params = params.copy()
        options = params.pop('OPTIONS').copy()
        if options:
            raise ImproperlyConfigured(
                "Unknown options: {}".format(", ".join(options)))
        super().__init__(params)

    def from_string(self, template_code):
        return Template(template_code)

    def get_template(self, template_name):
        tried = []
        for template_file in self.iter_template_filenames(template_name):
            try:
                with open(template_file, encoding='utf-8') as fp:
                    template_code = fp.read()
            except FileNotFoundError:
                tried.append((
                    Origin(template_file, template_name, self),
                    'Source does not exist',
                ))
            else:
                return Template(template_code)
        raise TemplateDoesNotExist(template_name, tried=tried, backend=self)


class Template(string.Template):

    def render(self, context=None, request=None):
        if context is None:
            context = {}
        else:
            context = {k: conditional_escape(v) for k, v in context.items()}
        if request is not None:
            context['csrf_input'] = csrf_input_lazy(request)
            context['csrf_token'] = csrf_token_lazy(request)
        return self.safe_substitute(context)
```
### 9 - django/db/models/sql/query.py:

Start line: 1375, End line: 1396

```python
class Query(BaseExpression):

    def add_filter(self, filter_clause):
        self.add_q(Q(**{filter_clause[0]: filter_clause[1]}))

    def add_q(self, q_object):
        """
        A preprocessor for the internal _add_q(). Responsible for doing final
        join promotion.
        """
        # For join promotion this case is doing an AND for the added q_object
        # and existing conditions. So, any existing inner join forces the join
        # type to remain inner. Existing outer joins can however be demoted.
        # (Consider case where rel_a is LOUTER and rel_a__col=1 is added - if
        # rel_a doesn't produce any rows, then the whole condition must fail.
        # So, demotion is OK.
        existing_inner = {a for a in self.alias_map if self.alias_map[a].join_type == INNER}
        clause, _ = self._add_q(q_object, self.used_aliases)
        if clause:
            self.where.add(clause, AND)
        self.demote_joins(existing_inner)

    def build_where(self, filter_expr):
        return self.build_filter(filter_expr, allow_joins=False)[0]
```
### 10 - django/template/defaultfilters.py:

Start line: 340, End line: 424

```python
@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def urlize(value, autoescape=True):
    """Convert URLs in plain text into clickable links."""
    return mark_safe(_urlize(value, nofollow=True, autoescape=autoescape))


@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def urlizetrunc(value, limit, autoescape=True):
    """
    Convert URLs into clickable links, truncating URLs to the given character
    limit, and adding 'rel=nofollow' attribute to discourage spamming.

    Argument: Length to truncate URLs to.
    """
    return mark_safe(_urlize(value, trim_url_limit=int(limit), nofollow=True, autoescape=autoescape))


@register.filter(is_safe=False)
@stringfilter
def wordcount(value):
    """Return the number of words."""
    return len(value.split())


@register.filter(is_safe=True)
@stringfilter
def wordwrap(value, arg):
    """Wrap words at `arg` line length."""
    return wrap(value, int(arg))


@register.filter(is_safe=True)
@stringfilter
def ljust(value, arg):
    """Left-align the value in a field of a given width."""
    return value.ljust(int(arg))


@register.filter(is_safe=True)
@stringfilter
def rjust(value, arg):
    """Right-align the value in a field of a given width."""
    return value.rjust(int(arg))


@register.filter(is_safe=True)
@stringfilter
def center(value, arg):
    """Center the value in a field of a given width."""
    return value.center(int(arg))


@register.filter
@stringfilter
def cut(value, arg):
    """Remove all values of arg from the given string."""
    safe = isinstance(value, SafeData)
    value = value.replace(arg, '')
    if safe and arg != ';':
        return mark_safe(value)
    return value


###################
# HTML STRINGS    #
###################

@register.filter("escape", is_safe=True)
@stringfilter
def escape_filter(value):
    """Mark the value as a string that should be auto-escaped."""
    return conditional_escape(value)


@register.filter(is_safe=True)
@stringfilter
def force_escape(value):
    """
    Escape a string's HTML. Return a new string containing the escaped
    characters (as opposed to "escape", which marks the content for later
    possible escaping).
    """
    return escape(value)
```
### 15 - django/utils/functional.py:

Start line: 109, End line: 127

```python
def lazy(func, *resultclasses):

    @total_ordering
    class __proxy__(Promise):

        @classmethod
        def __prepare_class__(cls):
            for resultclass in resultclasses:
                for type_ in resultclass.mro():
                    for method_name in type_.__dict__:
                        # All __promise__ return the same wrapper method, they
                        # look up the correct implementation when called.
                        if hasattr(cls, method_name):
                            continue
                        meth = cls.__promise__(method_name)
                        setattr(cls, method_name, meth)
            cls._delegate_bytes = bytes in resultclasses
            cls._delegate_text = str in resultclasses
            assert not (cls._delegate_bytes and cls._delegate_text), (
                "Cannot call lazy() with both bytes and text return types.")
            if cls._delegate_text:
                cls.__str__ = cls.__text_cast
            elif cls._delegate_bytes:
                cls.__bytes__ = cls.__bytes_cast
    # ... other code
```
### 28 - django/utils/functional.py:

Start line: 129, End line: 191

```python
def lazy(func, *resultclasses):

    @total_ordering
    class __proxy__(Promise):

        @classmethod
        def __promise__(cls, method_name):
            # Builds a wrapper around some magic method
            def __wrapper__(self, *args, **kw):
                # Automatically triggers the evaluation of a lazy value and
                # applies the given magic method of the result type.
                res = func(*self.__args, **self.__kw)
                return getattr(res, method_name)(*args, **kw)
            return __wrapper__

        def __text_cast(self):
            return func(*self.__args, **self.__kw)

        def __bytes_cast(self):
            return bytes(func(*self.__args, **self.__kw))

        def __bytes_cast_encoded(self):
            return func(*self.__args, **self.__kw).encode()

        def __cast(self):
            if self._delegate_bytes:
                return self.__bytes_cast()
            elif self._delegate_text:
                return self.__text_cast()
            else:
                return func(*self.__args, **self.__kw)

        def __str__(self):
            # object defines __str__(), so __prepare_class__() won't overload
            # a __str__() method from the proxied class.
            return str(self.__cast())

        def __eq__(self, other):
            if isinstance(other, Promise):
                other = other.__cast()
            return self.__cast() == other

        def __lt__(self, other):
            if isinstance(other, Promise):
                other = other.__cast()
            return self.__cast() < other

        def __hash__(self):
            return hash(self.__cast())

        def __mod__(self, rhs):
            if self._delegate_text:
                return str(self) % rhs
            return self.__cast() % rhs

        def __deepcopy__(self, memo):
            # Instances of this class are effectively immutable. It's just a
            # collection of functions. So we don't need to do anything
            # complicated for copying.
            memo[id(self)] = self
            return self

    @wraps(func)
    def __wrapper__(*args, **kw):
        # Creates the proxy object, instead of the actual value.
        return __proxy__(args, kw)

    return __wrapper__
```
### 33 - django/utils/functional.py:

Start line: 76, End line: 107

```python
def lazy(func, *resultclasses):
    """
    Turn any callable into a lazy evaluated callable. result classes or types
    is required -- at least one is needed so that the automatic forcing of
    the lazy evaluation code is triggered. Results are not memoized; the
    function is evaluated on every access.
    """

    @total_ordering
    class __proxy__(Promise):
        """
        Encapsulate a function call and act as a proxy for methods that are
        called on the result of that function. The function is not evaluated
        until one of the methods on the result is called.
        """
        __prepared = False

        def __init__(self, args, kw):
            self.__args = args
            self.__kw = kw
            if not self.__prepared:
                self.__prepare_class__()
            self.__class__.__prepared = True

        def __reduce__(self):
            return (
                _lazy_proxy_unpickle,
                (func, self.__args, self.__kw) + resultclasses
            )

        def __repr__(self):
            return repr(self.__cast())
    # ... other code
```
### 78 - django/utils/functional.py:

Start line: 245, End line: 345

```python
class LazyObject:
    """
    A wrapper for another class that can be used to delay instantiation of the
    wrapped class.

    By subclassing, you have the opportunity to intercept and alter the
    instantiation. If you don't need to do that, use SimpleLazyObject.
    """

    # Avoid infinite recursion when tracing __init__ (#19456).
    _wrapped = None

    def __init__(self):
        # Note: if a subclass overrides __init__(), it will likely need to
        # override __copy__() and __deepcopy__() as well.
        self._wrapped = empty

    __getattr__ = new_method_proxy(getattr)

    def __setattr__(self, name, value):
        if name == "_wrapped":
            # Assign to __dict__ to avoid infinite __setattr__ loops.
            self.__dict__["_wrapped"] = value
        else:
            if self._wrapped is empty:
                self._setup()
            setattr(self._wrapped, name, value)

    def __delattr__(self, name):
        if name == "_wrapped":
            raise TypeError("can't delete _wrapped.")
        if self._wrapped is empty:
            self._setup()
        delattr(self._wrapped, name)

    def _setup(self):
        """
        Must be implemented by subclasses to initialize the wrapped object.
        """
        raise NotImplementedError('subclasses of LazyObject must provide a _setup() method')

    # Because we have messed with __class__ below, we confuse pickle as to what
    # class we are pickling. We're going to have to initialize the wrapped
    # object to successfully pickle it, so we might as well just pickle the
    # wrapped object since they're supposed to act the same way.
    #
    # Unfortunately, if we try to simply act like the wrapped object, the ruse
    # will break down when pickle gets our id(). Thus we end up with pickle
    # thinking, in effect, that we are a distinct object from the wrapped
    # object, but with the same __dict__. This can cause problems (see #25389).
    #
    # So instead, we define our own __reduce__ method and custom unpickler. We
    # pickle the wrapped object as the unpickler's argument, so that pickle
    # will pickle it normally, and then the unpickler simply returns its
    # argument.
    def __reduce__(self):
        if self._wrapped is empty:
            self._setup()
        return (unpickle_lazyobject, (self._wrapped,))

    def __copy__(self):
        if self._wrapped is empty:
            # If uninitialized, copy the wrapper. Use type(self), not
            # self.__class__, because the latter is proxied.
            return type(self)()
        else:
            # If initialized, return a copy of the wrapped object.
            return copy.copy(self._wrapped)

    def __deepcopy__(self, memo):
        if self._wrapped is empty:
            # We have to use type(self), not self.__class__, because the
            # latter is proxied.
            result = type(self)()
            memo[id(self)] = result
            return result
        return copy.deepcopy(self._wrapped, memo)

    __bytes__ = new_method_proxy(bytes)
    __str__ = new_method_proxy(str)
    __bool__ = new_method_proxy(bool)

    # Introspection support
    __dir__ = new_method_proxy(dir)

    # Need to pretend to be the wrapped class, for the sake of objects that
    # care about this (especially in equality tests)
    __class__ = property(new_method_proxy(operator.attrgetter("__class__")))
    __eq__ = new_method_proxy(operator.eq)
    __lt__ = new_method_proxy(operator.lt)
    __gt__ = new_method_proxy(operator.gt)
    __ne__ = new_method_proxy(operator.ne)
    __hash__ = new_method_proxy(hash)

    # List/Tuple/Dictionary methods support
    __getitem__ = new_method_proxy(operator.getitem)
    __setitem__ = new_method_proxy(operator.setitem)
    __delitem__ = new_method_proxy(operator.delitem)
    __iter__ = new_method_proxy(iter)
    __len__ = new_method_proxy(len)
    __contains__ = new_method_proxy(operator.contains)
```
