# django__django-11399

| **django/django** | `b711eafd2aabdf22e1d529bfb76dd8d3356d7000` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6794 |
| **Any found context length** | 6794 |
| **Avg pos** | 15.0 |
| **Min pos** | 15 |
| **Max pos** | 15 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -79,7 +79,7 @@ def __init__(self, args, kw):
             self.__kw = kw
             if not self.__prepared:
                 self.__prepare_class__()
-            self.__prepared = True
+            self.__class__.__prepared = True
 
         def __reduce__(self):
             return (

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/functional.py | 82 | 82 | 15 | 1 | 6794


## Problem Statement

```
lazy() class preparation is not being cached correctly.
Description
	
Doing self.__prepared = True changes the instance, but the intention is to change the class variable: ​https://github.com/django/django/blob/888fdf182e164fa4b24aa82fa833c90a2b9bee7a/django/utils/functional.py#L82
This makes functions like gettext_lazy, format_lazy and reverse_lazy a lot slower than they ought to be.
Regressed in Django 1.8 (b4e76f30d12bfa8a53cc297c60055c6f4629cc4c).
Using this micro-benchmark on Python 3.7:
import cProfile
from django.utils.functional import lazy
def identity(x): return x
lazy_identity = lazy(identity, int)
cProfile.run("for i in range(10000): str(lazy_identity(1))")
Before:
		 910049 function calls in 0.208 seconds
	Ordered by: standard name
	ncalls tottime percall cumtime percall filename:lineno(function)
		 1	0.010	0.010	0.208	0.208 <string>:1(<module>)
	 10000	0.001	0.000	0.001	0.000 bench.py:4(identity)
	 10000	0.005	0.000	0.010	0.000 functional.py:105(__str__)
	 10000	0.004	0.000	0.188	0.000 functional.py:159(__wrapper__)
	 10000	0.007	0.000	0.185	0.000 functional.py:76(__init__)
	 10000	0.089	0.000	0.178	0.000 functional.py:83(__prepare_class__)
	 10000	0.004	0.000	0.005	0.000 functional.py:99(__cast)
		 1	0.000	0.000	0.208	0.208 {built-in method builtins.exec}
	840000	0.087	0.000	0.087	0.000 {built-in method builtins.hasattr}
		46	0.000	0.000	0.000	0.000 {built-in method builtins.setattr}
		 1	0.000	0.000	0.000	0.000 {method 'disable' of '_lsprof.Profiler' objects}
	 10000	0.002	0.000	0.002	0.000 {method 'mro' of 'type' objects}
After:
		 50135 function calls in 0.025 seconds
	Ordered by: standard name
	ncalls tottime percall cumtime percall filename:lineno(function)
		 1	0.008	0.008	0.025	0.025 <string>:1(<module>)
	 10000	0.001	0.000	0.001	0.000 bench.py:4(identity)
	 10000	0.005	0.000	0.009	0.000 functional.py:105(__str__)
	 10000	0.003	0.000	0.008	0.000 functional.py:159(__wrapper__)
	 10000	0.005	0.000	0.005	0.000 functional.py:76(__init__)
		 1	0.000	0.000	0.000	0.000 functional.py:83(__prepare_class__)
	 10000	0.004	0.000	0.005	0.000 functional.py:99(__cast)
		 1	0.000	0.000	0.025	0.025 {built-in method builtins.exec}
		84	0.000	0.000	0.000	0.000 {built-in method builtins.hasattr}
		46	0.000	0.000	0.000	0.000 {built-in method builtins.setattr}
		 1	0.000	0.000	0.000	0.000 {method 'disable' of '_lsprof.Profiler' objects}
		 1	0.000	0.000	0.000	0.000 {method 'mro' of 'type' objects}

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/functional.py** | 93 | 111| 208 | 208 | 2966 | 
| 2 | **1 django/utils/functional.py** | 229 | 329| 905 | 1113 | 2966 | 
| 3 | **1 django/utils/functional.py** | 113 | 175| 484 | 1597 | 2966 | 
| 4 | **1 django/utils/functional.py** | 178 | 226| 299 | 1896 | 2966 | 
| 5 | 2 django/utils/translation/__init__.py | 149 | 193| 335 | 2231 | 5250 | 
| 6 | **2 django/utils/functional.py** | 332 | 402| 513 | 2744 | 5250 | 
| 7 | 3 django/conf/__init__.py | 65 | 95| 260 | 3004 | 7036 | 
| 8 | 3 django/utils/translation/__init__.py | 67 | 146| 489 | 3493 | 7036 | 
| 9 | 3 django/utils/translation/__init__.py | 196 | 268| 440 | 3933 | 7036 | 
| 10 | 4 django/db/models/query.py | 1556 | 1662| 1063 | 4996 | 23507 | 
| 11 | 4 django/conf/__init__.py | 112 | 129| 150 | 5146 | 23507 | 
| 12 | 5 django/views/i18n.py | 77 | 180| 711 | 5857 | 26014 | 
| 13 | **5 django/utils/functional.py** | 1 | 49| 336 | 6193 | 26014 | 
| 14 | 6 django/core/checks/model_checks.py | 166 | 199| 332 | 6525 | 27695 | 
| **-> 15 <-** | **6 django/utils/functional.py** | 52 | 91| 269 | 6794 | 27695 | 
| 16 | 7 django/db/migrations/recorder.py | 24 | 45| 145 | 6939 | 28365 | 
| 17 | 8 django/template/loaders/cached.py | 1 | 60| 467 | 7406 | 29053 | 
| 18 | 9 django/db/models/functions/__init__.py | 1 | 45| 668 | 8074 | 29721 | 
| 19 | 9 django/conf/__init__.py | 42 | 63| 199 | 8273 | 29721 | 
| 20 | 10 django/db/models/base.py | 399 | 503| 903 | 9176 | 44679 | 
| 21 | 11 django/db/models/fields/related.py | 62 | 80| 223 | 9399 | 58250 | 
| 22 | 12 django/utils/translation/trans_null.py | 1 | 68| 269 | 9668 | 58519 | 
| 23 | 13 django/core/cache/backends/db.py | 112 | 197| 794 | 10462 | 60605 | 
| 24 | 14 django/utils/encoding.py | 48 | 67| 156 | 10618 | 62901 | 
| 25 | 14 django/db/models/query.py | 1712 | 1775| 658 | 11276 | 62901 | 
| 26 | 14 django/core/checks/model_checks.py | 117 | 141| 268 | 11544 | 62901 | 
| 27 | 14 django/db/models/query.py | 188 | 231| 352 | 11896 | 62901 | 
| 28 | 14 django/core/checks/model_checks.py | 77 | 98| 168 | 12064 | 62901 | 
| 29 | 14 django/db/models/base.py | 1814 | 1865| 351 | 12415 | 62901 | 
| 30 | 15 django/utils/autoreload.py | 323 | 362| 266 | 12681 | 67495 | 
| 31 | 15 django/db/models/query.py | 1527 | 1555| 246 | 12927 | 67495 | 
| 32 | 15 django/core/checks/model_checks.py | 100 | 115| 176 | 13103 | 67495 | 
| 33 | 16 django/core/cache/backends/dummy.py | 1 | 39| 251 | 13354 | 67747 | 
| 34 | 17 django/utils/deprecation.py | 1 | 30| 195 | 13549 | 68439 | 
| 35 | 17 django/utils/translation/__init__.py | 1 | 36| 281 | 13830 | 68439 | 
| 36 | 17 django/db/models/query.py | 1777 | 1809| 314 | 14144 | 68439 | 
| 37 | 17 django/utils/deprecation.py | 76 | 98| 158 | 14302 | 68439 | 
| 38 | 18 django/core/cache/backends/locmem.py | 1 | 66| 505 | 14807 | 69340 | 
| 39 | 18 django/utils/translation/__init__.py | 54 | 64| 127 | 14934 | 69340 | 
| 40 | 18 django/db/models/query.py | 1478 | 1524| 433 | 15367 | 69340 | 
| 41 | 19 django/db/models/options.py | 1 | 36| 304 | 15671 | 76206 | 
| 42 | 20 django/apps/registry.py | 378 | 428| 465 | 16136 | 79613 | 
| 43 | 20 django/utils/translation/__init__.py | 37 | 52| 154 | 16290 | 79613 | 
| 44 | 21 django/db/models/__init__.py | 1 | 49| 548 | 16838 | 80161 | 
| 45 | 21 django/conf/__init__.py | 97 | 110| 131 | 16969 | 80161 | 
| 46 | 21 django/db/models/base.py | 319 | 377| 525 | 17494 | 80161 | 
| 47 | 22 django/utils/formats.py | 1 | 57| 377 | 17871 | 82253 | 
| 48 | 23 django/contrib/gis/geos/prototypes/prepared.py | 1 | 29| 282 | 18153 | 82535 | 
| 49 | 24 django/template/engine.py | 1 | 53| 388 | 18541 | 83845 | 
| 50 | 25 django/utils/translation/trans_real.py | 1 | 56| 482 | 19023 | 87670 | 
| 51 | 26 django/db/backends/mysql/features.py | 1 | 101| 843 | 19866 | 88650 | 
| 52 | 26 django/core/cache/backends/db.py | 253 | 278| 308 | 20174 | 88650 | 
| 53 | 27 django/db/models/query_utils.py | 154 | 218| 492 | 20666 | 91281 | 
| 54 | 28 django/contrib/gis/geos/libgeos.py | 134 | 175| 293 | 20959 | 92565 | 
| 55 | 29 django/http/multipartparser.py | 365 | 404| 258 | 21217 | 97592 | 
| 56 | 30 django/db/models/fields/related_lookups.py | 102 | 117| 212 | 21429 | 99038 | 
| 57 | 31 django/core/cache/backends/base.py | 1 | 47| 245 | 21674 | 101188 | 
| 58 | 32 django/db/models/functions/window.py | 28 | 49| 154 | 21828 | 101831 | 
| 59 | 33 django/contrib/sessions/backends/base.py | 127 | 210| 563 | 22391 | 104362 | 
| 60 | 33 django/utils/deprecation.py | 33 | 73| 336 | 22727 | 104362 | 
| 61 | 34 django/db/models/lookups.py | 171 | 187| 170 | 22897 | 108516 | 
| 62 | 34 django/core/cache/backends/db.py | 228 | 251| 259 | 23156 | 108516 | 
| 63 | 34 django/db/models/base.py | 1 | 45| 289 | 23445 | 108516 | 
| 64 | 34 django/db/models/options.py | 149 | 202| 518 | 23963 | 108516 | 
| 65 | 34 django/db/models/functions/window.py | 52 | 79| 182 | 24145 | 108516 | 
| 66 | 34 django/utils/autoreload.py | 285 | 320| 259 | 24404 | 108516 | 
| 67 | 34 django/core/cache/backends/db.py | 97 | 110| 234 | 24638 | 108516 | 
| 68 | 34 django/db/models/lookups.py | 190 | 225| 308 | 24946 | 108516 | 
| 69 | 35 django/template/backends/dummy.py | 1 | 54| 330 | 25276 | 108846 | 
| 70 | 36 django/core/management/commands/loaddata.py | 81 | 148| 593 | 25869 | 111714 | 
| 71 | 37 django/db/migrations/optimizer.py | 41 | 71| 249 | 26118 | 112310 | 
| 72 | 37 django/core/cache/backends/db.py | 40 | 95| 431 | 26549 | 112310 | 
| 73 | 37 django/core/checks/model_checks.py | 143 | 164| 263 | 26812 | 112310 | 
| 74 | 38 django/core/cache/backends/memcached.py | 92 | 108| 167 | 26979 | 114028 | 
| 75 | 39 django/contrib/staticfiles/storage.py | 256 | 325| 569 | 27548 | 117923 | 
| 76 | 39 django/core/cache/backends/locmem.py | 83 | 123| 270 | 27818 | 117923 | 
| 77 | 39 django/utils/autoreload.py | 269 | 283| 146 | 27964 | 117923 | 
| 78 | 39 django/core/cache/backends/memcached.py | 110 | 126| 168 | 28132 | 117923 | 
| 79 | 39 django/db/models/options.py | 204 | 242| 362 | 28494 | 117923 | 
| 80 | 39 django/db/models/lookups.py | 55 | 80| 218 | 28712 | 117923 | 
| 81 | 40 django/contrib/humanize/templatetags/humanize.py | 218 | 261| 731 | 29443 | 121064 | 
| 82 | 41 django/core/files/storage.py | 289 | 361| 483 | 29926 | 123900 | 
| 83 | 41 django/conf/__init__.py | 132 | 185| 472 | 30398 | 123900 | 
| 84 | 41 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 30622 | 123900 | 
| 85 | 41 django/core/cache/backends/locmem.py | 68 | 81| 138 | 30760 | 123900 | 
| 86 | 42 django/utils/timesince.py | 1 | 24| 220 | 30980 | 124754 | 
| 87 | 43 django/db/backends/base/features.py | 1 | 115| 900 | 31880 | 127207 | 
| 88 | 43 django/http/multipartparser.py | 406 | 425| 205 | 32085 | 127207 | 
| 89 | 43 django/contrib/humanize/templatetags/humanize.py | 263 | 301| 370 | 32455 | 127207 | 
| 90 | 43 django/db/models/options.py | 703 | 718| 144 | 32599 | 127207 | 
| 91 | 43 django/db/models/query.py | 1665 | 1709| 439 | 33038 | 127207 | 
| 92 | 43 django/db/models/lookups.py | 128 | 147| 134 | 33172 | 127207 | 
| 93 | 44 django/db/migrations/autodetector.py | 358 | 372| 141 | 33313 | 138878 | 
| 94 | 45 django/core/files/base.py | 1 | 29| 174 | 33487 | 139930 | 
| 95 | 46 django/urls/base.py | 93 | 160| 381 | 33868 | 141124 | 
| 96 | 47 django/conf/global_settings.py | 145 | 263| 876 | 34744 | 146728 | 
| 97 | 47 django/db/models/base.py | 207 | 317| 866 | 35610 | 146728 | 
| 98 | 48 django/core/checks/messages.py | 53 | 76| 161 | 35771 | 147301 | 
| 99 | 48 django/db/models/lookups.py | 96 | 125| 243 | 36014 | 147301 | 
| 100 | 48 django/conf/global_settings.py | 347 | 397| 826 | 36840 | 147301 | 
| 101 | 48 django/core/cache/backends/memcached.py | 1 | 36| 248 | 37088 | 147301 | 
| 102 | 48 django/db/models/lookups.py | 1 | 36| 283 | 37371 | 147301 | 
| 103 | 49 django/utils/cache.py | 116 | 131| 188 | 37559 | 150850 | 
| 104 | 49 django/utils/encoding.py | 102 | 115| 130 | 37689 | 150850 | 
| 105 | 50 django/contrib/admin/sites.py | 1 | 29| 175 | 37864 | 155041 | 
| 106 | 51 django/contrib/sessions/backends/cached_db.py | 1 | 41| 253 | 38117 | 155462 | 
| 107 | 51 django/template/loaders/cached.py | 62 | 93| 225 | 38342 | 155462 | 
| 108 | 52 django/db/models/sql/compiler.py | 45 | 57| 139 | 38481 | 168981 | 
| 109 | 52 django/template/engine.py | 81 | 147| 457 | 38938 | 168981 | 
| 110 | 53 django/db/models/fields/related_descriptors.py | 356 | 372| 184 | 39122 | 179266 | 
| 111 | 54 django/utils/dates.py | 1 | 50| 679 | 39801 | 179945 | 
| 112 | 54 django/db/models/lookups.py | 82 | 94| 126 | 39927 | 179945 | 
| 113 | 54 django/db/models/options.py | 345 | 367| 164 | 40091 | 179945 | 
| 114 | 54 django/contrib/staticfiles/storage.py | 1 | 18| 129 | 40220 | 179945 | 
| 115 | 55 django/template/base.py | 571 | 606| 359 | 40579 | 187806 | 
| 116 | 56 django/views/debug.py | 98 | 122| 155 | 40734 | 192024 | 
| 117 | 56 django/template/base.py | 117 | 138| 127 | 40861 | 192024 | 
| 118 | 56 django/http/multipartparser.py | 339 | 363| 169 | 41030 | 192024 | 
| 119 | 56 django/db/migrations/autodetector.py | 264 | 335| 748 | 41778 | 192024 | 
| 120 | 57 django/utils/itercompat.py | 1 | 9| 0 | 41778 | 192064 | 
| 121 | 57 django/db/models/query_utils.py | 114 | 151| 320 | 42098 | 192064 | 
| 122 | 58 django/core/cache/utils.py | 1 | 13| 0 | 42098 | 192151 | 
| 123 | 58 django/utils/encoding.py | 70 | 99| 250 | 42348 | 192151 | 
| 124 | 58 django/db/models/fields/related.py | 1202 | 1325| 1010 | 43358 | 192151 | 
| 125 | 59 django/conf/locale/id/formats.py | 5 | 50| 708 | 44066 | 192904 | 
| 126 | 59 django/db/models/base.py | 1235 | 1264| 242 | 44308 | 192904 | 
| 127 | 60 django/core/validators.py | 1 | 25| 190 | 44498 | 197224 | 


## Patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -79,7 +79,7 @@ def __init__(self, args, kw):
             self.__kw = kw
             if not self.__prepared:
                 self.__prepare_class__()
-            self.__prepared = True
+            self.__class__.__prepared = True
 
         def __reduce__(self):
             return (

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_functional.py b/tests/utils_tests/test_functional.py
--- a/tests/utils_tests/test_functional.py
+++ b/tests/utils_tests/test_functional.py
@@ -1,3 +1,5 @@
+from unittest import mock
+
 from django.test import SimpleTestCase
 from django.utils.functional import cached_property, lazy
 
@@ -207,3 +209,12 @@ def test_lazy_repr_bytes(self):
         original_object = b'J\xc3\xbcst a str\xc3\xadng'
         lazy_obj = lazy(lambda: original_object, bytes)
         self.assertEqual(repr(original_object), repr(lazy_obj()))
+
+    def test_lazy_class_preparation_caching(self):
+        # lazy() should prepare the proxy class only once i.e. the first time
+        # it's used.
+        lazified = lazy(lambda: 0, int)
+        __proxy__ = lazified().__class__
+        with mock.patch.object(__proxy__, '__prepare_class__') as mocked:
+            lazified()
+            mocked.assert_not_called()

```


## Code snippets

### 1 - django/utils/functional.py:

Start line: 93, End line: 111

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
### 2 - django/utils/functional.py:

Start line: 229, End line: 329

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
### 3 - django/utils/functional.py:

Start line: 113, End line: 175

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
### 4 - django/utils/functional.py:

Start line: 178, End line: 226

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
### 5 - django/utils/translation/__init__.py:

Start line: 149, End line: 193

```python
def lazy_number(func, resultclass, number=None, **kwargs):
    if isinstance(number, int):
        kwargs['number'] = number
        proxy = lazy(func, resultclass)(**kwargs)
    else:
        original_kwargs = kwargs.copy()

        class NumberAwareString(resultclass):
            def __bool__(self):
                return bool(kwargs['singular'])

            def _get_number_value(self, values):
                try:
                    return values[number]
                except KeyError:
                    raise KeyError(
                        "Your dictionary lacks key '%s\'. Please provide "
                        "it, because it is required to determine whether "
                        "string is singular or plural." % number
                    )

            def _translate(self, number_value):
                kwargs['number'] = number_value
                return func(**kwargs)

            def format(self, *args, **kwargs):
                number_value = self._get_number_value(kwargs) if kwargs and number else args[0]
                return self._translate(number_value).format(*args, **kwargs)

            def __mod__(self, rhs):
                if isinstance(rhs, dict) and number:
                    number_value = self._get_number_value(rhs)
                else:
                    number_value = rhs
                translated = self._translate(number_value)
                try:
                    translated = translated % rhs
                except TypeError:
                    # String doesn't contain a placeholder for the number.
                    pass
                return translated

        proxy = lazy(lambda **kwargs: NumberAwareString(), NumberAwareString)(**kwargs)
        proxy.__reduce__ = lambda: (_lazy_number_unpickle, (func, resultclass, number, original_kwargs))
    return proxy
```
### 6 - django/utils/functional.py:

Start line: 332, End line: 402

```python
def unpickle_lazyobject(wrapped):
    """
    Used to unpickle lazy objects. Just return its argument, which will be the
    wrapped object.
    """
    return wrapped


class SimpleLazyObject(LazyObject):
    """
    A lazy object initialized from any function.

    Designed for compound objects of unknown type. For builtins or objects of
    known type, use django.utils.functional.lazy.
    """
    def __init__(self, func):
        """
        Pass in a callable that returns the object to be wrapped.

        If copies are made of the resulting SimpleLazyObject, which can happen
        in various circumstances within Django, then you must ensure that the
        callable can be safely run more than once and will return the same
        value.
        """
        self.__dict__['_setupfunc'] = func
        super().__init__()

    def _setup(self):
        self._wrapped = self._setupfunc()

    # Return a meaningful representation of the lazy object for debugging
    # without evaluating the wrapped object.
    def __repr__(self):
        if self._wrapped is empty:
            repr_attr = self._setupfunc
        else:
            repr_attr = self._wrapped
        return '<%s: %r>' % (type(self).__name__, repr_attr)

    def __copy__(self):
        if self._wrapped is empty:
            # If uninitialized, copy the wrapper. Use SimpleLazyObject, not
            # self.__class__, because the latter is proxied.
            return SimpleLazyObject(self._setupfunc)
        else:
            # If initialized, return a copy of the wrapped object.
            return copy.copy(self._wrapped)

    def __deepcopy__(self, memo):
        if self._wrapped is empty:
            # We have to use SimpleLazyObject, not self.__class__, because the
            # latter is proxied.
            result = SimpleLazyObject(self._setupfunc)
            memo[id(self)] = result
            return result
        return copy.deepcopy(self._wrapped, memo)


def partition(predicate, values):
    """
    Split the values into two sets, based on the return value of the function
    (True/False). e.g.:

        >>> partition(lambda x: x > 3, range(5))
        [0, 1, 2, 3], [4]
    """
    results = ([], [])
    for item in values:
        results[predicate(item)].append(item)
    return results
```
### 7 - django/conf/__init__.py:

Start line: 65, End line: 95

```python
class LazySettings(LazyObject):

    def __repr__(self):
        # Hardcode the class name as otherwise it yields 'Settings'.
        if self._wrapped is empty:
            return '<LazySettings [Unevaluated]>'
        return '<LazySettings "%(settings_module)s">' % {
            'settings_module': self._wrapped.SETTINGS_MODULE,
        }

    def __getattr__(self, name):
        """Return the value of a setting and cache it in self.__dict__."""
        if self._wrapped is empty:
            self._setup(name)
        val = getattr(self._wrapped, name)
        self.__dict__[name] = val
        return val

    def __setattr__(self, name, value):
        """
        Set the value of setting. Clear all cached values if _wrapped changes
        (@override_settings does this) or clear single values when set.
        """
        if name == '_wrapped':
            self.__dict__.clear()
        else:
            self.__dict__.pop(name, None)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        """Delete a setting and clear it from cache if needed."""
        super().__delattr__(name)
        self.__dict__.pop(name, None)
```
### 8 - django/utils/translation/__init__.py:

Start line: 67, End line: 146

```python
_trans = Trans()

# The Trans class is no more needed, so remove it from the namespace.
del Trans


def gettext_noop(message):
    return _trans.gettext_noop(message)


def ugettext_noop(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of gettext_noop() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext_noop() is deprecated in favor of '
        'django.utils.translation.gettext_noop().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext_noop(message)


def gettext(message):
    return _trans.gettext(message)


def ugettext(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of gettext() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext() is deprecated in favor of '
        'django.utils.translation.gettext().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext(message)


def ngettext(singular, plural, number):
    return _trans.ngettext(singular, plural, number)


def ungettext(singular, plural, number):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    Alias of ngettext() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ungettext() is deprecated in favor of '
        'django.utils.translation.ngettext().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return ngettext(singular, plural, number)


def pgettext(context, message):
    return _trans.pgettext(context, message)


def npgettext(context, singular, plural, number):
    return _trans.npgettext(context, singular, plural, number)


gettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)


def ugettext_lazy(message):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2. Has been
    Alias of gettext_lazy since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ugettext_lazy() is deprecated in favor of '
        'django.utils.translation.gettext_lazy().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return gettext_lazy(message)
```
### 9 - django/utils/translation/__init__.py:

Start line: 196, End line: 268

```python
def _lazy_number_unpickle(func, resultclass, number, kwargs):
    return lazy_number(func, resultclass, number=number, **kwargs)


def ngettext_lazy(singular, plural, number=None):
    return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)


def ungettext_lazy(singular, plural, number=None):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    An alias of ungettext_lazy() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ungettext_lazy() is deprecated in favor of '
        'django.utils.translation.ngettext_lazy().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return ngettext_lazy(singular, plural, number)


def npgettext_lazy(context, singular, plural, number=None):
    return lazy_number(npgettext, str, context=context, singular=singular, plural=plural, number=number)


def activate(language):
    return _trans.activate(language)


def deactivate():
    return _trans.deactivate()


class override(ContextDecorator):
    def __init__(self, language, deactivate=False):
        self.language = language
        self.deactivate = deactivate

    def __enter__(self):
        self.old_language = get_language()
        if self.language is not None:
            activate(self.language)
        else:
            deactivate_all()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_language is None:
            deactivate_all()
        elif self.deactivate:
            deactivate()
        else:
            activate(self.old_language)


def get_language():
    return _trans.get_language()


def get_language_bidi():
    return _trans.get_language_bidi()


def check_for_language(lang_code):
    return _trans.check_for_language(lang_code)


def to_language(locale):
    """Turn a locale name (en_US) into a language name (en-us)."""
    p = locale.find('_')
    if p >= 0:
        return locale[:p].lower() + '-' + locale[p + 1:].lower()
    else:
        return locale.lower()
```
### 10 - django/db/models/query.py:

Start line: 1556, End line: 1662

```python
def prefetch_related_objects(model_instances, *related_lookups):
    # ... other code
    while all_lookups:
        lookup = all_lookups.pop()
        if lookup.prefetch_to in done_queries:
            if lookup.queryset is not None:
                raise ValueError("'%s' lookup was already seen with a different queryset. "
                                 "You may need to adjust the ordering of your lookups." % lookup.prefetch_to)

            continue

        # Top level, the list of objects to decorate is the result cache
        # from the primary QuerySet. It won't be for deeper levels.
        obj_list = model_instances

        through_attrs = lookup.prefetch_through.split(LOOKUP_SEP)
        for level, through_attr in enumerate(through_attrs):
            # Prepare main instances
            if not obj_list:
                break

            prefetch_to = lookup.get_current_prefetch_to(level)
            if prefetch_to in done_queries:
                # Skip any prefetching, and any object preparation
                obj_list = done_queries[prefetch_to]
                continue

            # Prepare objects:
            good_objects = True
            for obj in obj_list:
                # Since prefetching can re-use instances, it is possible to have
                # the same instance multiple times in obj_list, so obj might
                # already be prepared.
                if not hasattr(obj, '_prefetched_objects_cache'):
                    try:
                        obj._prefetched_objects_cache = {}
                    except (AttributeError, TypeError):
                        # Must be an immutable object from
                        # values_list(flat=True), for example (TypeError) or
                        # a QuerySet subclass that isn't returning Model
                        # instances (AttributeError), either in Django or a 3rd
                        # party. prefetch_related() doesn't make sense, so quit.
                        good_objects = False
                        break
            if not good_objects:
                break

            # Descend down tree

            # We assume that objects retrieved are homogeneous (which is the premise
            # of prefetch_related), so what applies to first object applies to all.
            first_obj = obj_list[0]
            to_attr = lookup.get_current_to_attr(level)[0]
            prefetcher, descriptor, attr_found, is_fetched = get_prefetcher(first_obj, through_attr, to_attr)

            if not attr_found:
                raise AttributeError("Cannot find '%s' on %s object, '%s' is an invalid "
                                     "parameter to prefetch_related()" %
                                     (through_attr, first_obj.__class__.__name__, lookup.prefetch_through))

            if level == len(through_attrs) - 1 and prefetcher is None:
                # Last one, this *must* resolve to something that supports
                # prefetching, otherwise there is no point adding it and the
                # developer asking for it has made a mistake.
                raise ValueError("'%s' does not resolve to an item that supports "
                                 "prefetching - this is an invalid parameter to "
                                 "prefetch_related()." % lookup.prefetch_through)

            if prefetcher is not None and not is_fetched:
                obj_list, additional_lookups = prefetch_one_level(obj_list, prefetcher, lookup, level)
                # We need to ensure we don't keep adding lookups from the
                # same relationships to stop infinite recursion. So, if we
                # are already on an automatically added lookup, don't add
                # the new lookups from relationships we've seen already.
                if not (prefetch_to in done_queries and lookup in auto_lookups and descriptor in followed_descriptors):
                    done_queries[prefetch_to] = obj_list
                    new_lookups = normalize_prefetch_lookups(reversed(additional_lookups), prefetch_to)
                    auto_lookups.update(new_lookups)
                    all_lookups.extend(new_lookups)
                followed_descriptors.add(descriptor)
            else:
                # Either a singly related object that has already been fetched
                # (e.g. via select_related), or hopefully some other property
                # that doesn't support prefetching but needs to be traversed.

                # We replace the current list of parent objects with the list
                # of related objects, filtering out empty or missing values so
                # that we can continue with nullable or reverse relations.
                new_obj_list = []
                for obj in obj_list:
                    if through_attr in getattr(obj, '_prefetched_objects_cache', ()):
                        # If related objects have been prefetched, use the
                        # cache rather than the object's through_attr.
                        new_obj = list(obj._prefetched_objects_cache.get(through_attr))
                    else:
                        try:
                            new_obj = getattr(obj, through_attr)
                        except exceptions.ObjectDoesNotExist:
                            continue
                    if new_obj is None:
                        continue
                    # We special-case `list` rather than something more generic
                    # like `Iterable` because we don't want to accidentally match
                    # user models that define __iter__.
                    if isinstance(new_obj, list):
                        new_obj_list.extend(new_obj)
                    else:
                        new_obj_list.append(new_obj)
                obj_list = new_obj_list
```
### 13 - django/utils/functional.py:

Start line: 1, End line: 49

```python
import copy
import itertools
import operator
from functools import total_ordering, wraps


class cached_property:
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    A cached property can be made out of an existing method:
    (e.g. ``url = cached_property(get_absolute_url)``).
    The optional ``name`` argument is obsolete as of Python 3.6 and will be
    deprecated in Django 4.0 (#30127).
    """
    name = None

    @staticmethod
    def func(instance):
        raise TypeError(
            'Cannot use cached_property instance without calling '
            '__set_name__() on it.'
        )

    def __init__(self, func, name=None):
        self.real_func = func
        self.__doc__ = getattr(func, '__doc__')

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name
            self.func = self.real_func
        elif name != self.name:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                "(%r and %r)." % (self.name, name)
            )

    def __get__(self, instance, cls=None):
        """
        Call the function and put the return value in instance.__dict__ so that
        subsequent attribute access on the instance returns the cached value
        instead of calling cached_property.__get__().
        """
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res
```
### 15 - django/utils/functional.py:

Start line: 52, End line: 91

```python
class Promise:
    """
    Base class for the proxy class created in the closure of the lazy function.
    It's used to recognize promises in code.
    """
    pass


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
            self.__prepared = True

        def __reduce__(self):
            return (
                _lazy_proxy_unpickle,
                (func, self.__args, self.__kw) + resultclasses
            )

        def __repr__(self):
            return repr(self.__cast())
    # ... other code
```
