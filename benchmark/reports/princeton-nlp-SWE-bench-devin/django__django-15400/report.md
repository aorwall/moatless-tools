# django__django-15400

| **django/django** | `4c76ffc2d6c77c850b4bef8d9acc197d11c47937` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2691 |
| **Any found context length** | 2691 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -432,6 +432,12 @@ def __deepcopy__(self, memo):
             return result
         return copy.deepcopy(self._wrapped, memo)
 
+    __add__ = new_method_proxy(operator.add)
+
+    @new_method_proxy
+    def __radd__(self, other):
+        return other + self
+
 
 def partition(predicate, values):
     """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/functional.py | 435 | 435 | 6 | 1 | 2691


## Problem Statement

```
SimpleLazyObject doesn't implement __radd__
Description
	
Technically, there's a whole bunch of magic methods it doesn't implement, compared to a complete proxy implementation, like that of wrapt.ObjectProxy, but __radd__ being missing is the one that's biting me at the moment.
As far as I can tell, the implementation can't just be
__radd__ = new_method_proxy(operator.radd)
because that doesn't exist, which is rubbish.
__radd__ = new_method_proxy(operator.attrgetter("__radd__"))
also won't work because types may not have that attr, and attrgetter doesn't supress the exception (correctly)
The minimal implementation I've found that works for me is:
	def __radd__(self, other):
		if self._wrapped is empty:
			self._setup()
		return other + self._wrapped

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/utils/functional.py** | 272 | 374| 909 | 909 | 3137 | 
| 2 | **1 django/utils/functional.py** | 143 | 212| 520 | 1429 | 3137 | 
| 3 | **1 django/utils/functional.py** | 121 | 141| 211 | 1640 | 3137 | 
| 4 | **1 django/utils/functional.py** | 215 | 269| 303 | 1943 | 3137 | 
| 5 | **1 django/utils/functional.py** | 87 | 119| 235 | 2178 | 3137 | 
| **-> 6 <-** | **1 django/utils/functional.py** | 377 | 448| 513 | 2691 | 3137 | 
| 7 | 2 django/contrib/gis/db/models/proxy.py | 1 | 47| 363 | 3054 | 3813 | 
| 8 | 3 django/conf/__init__.py | 56 | 79| 201 | 3255 | 6192 | 
| 9 | **3 django/utils/functional.py** | 1 | 58| 370 | 3625 | 6192 | 
| 10 | 4 django/core/checks/model_checks.py | 187 | 228| 345 | 3970 | 8002 | 
| 11 | 5 django/db/models/base.py | 194 | 246| 458 | 4428 | 25949 | 
| 12 | 6 django/utils/translation/__init__.py | 114 | 163| 343 | 4771 | 27827 | 
| 13 | 6 django/conf/__init__.py | 156 | 175| 172 | 4943 | 27827 | 
| 14 | 6 django/conf/__init__.py | 121 | 134| 131 | 5074 | 27827 | 
| 15 | 7 django/utils/connection.py | 1 | 31| 192 | 5266 | 28291 | 
| 16 | 7 django/db/models/base.py | 366 | 433| 541 | 5807 | 28291 | 
| 17 | 8 django/db/models/fields/proxy.py | 1 | 19| 117 | 5924 | 28408 | 
| 18 | 8 django/core/checks/model_checks.py | 118 | 133| 176 | 6100 | 28408 | 
| 19 | **8 django/utils/functional.py** | 61 | 84| 123 | 6223 | 28408 | 
| 20 | 9 django/core/files/utils.py | 27 | 79| 378 | 6601 | 28979 | 
| 21 | 10 django/utils/decorators.py | 1 | 22| 142 | 6743 | 30391 | 
| 22 | 10 django/db/models/base.py | 476 | 589| 953 | 7696 | 30391 | 
| 23 | 10 django/db/models/base.py | 247 | 364| 874 | 8570 | 30391 | 
| 24 | 11 django/db/models/query.py | 214 | 275| 469 | 9039 | 49211 | 
| 25 | 12 django/contrib/admin/sites.py | 595 | 609| 116 | 9155 | 53693 | 
| 26 | 12 django/core/checks/model_checks.py | 135 | 159| 268 | 9423 | 53693 | 
| 27 | 12 django/conf/__init__.py | 105 | 119| 129 | 9552 | 53693 | 
| 28 | 13 django/db/migrations/autodetector.py | 896 | 978| 707 | 10259 | 66156 | 
| 29 | 13 django/conf/__init__.py | 81 | 103| 221 | 10480 | 66156 | 
| 30 | 14 django/contrib/admin/utils.py | 164 | 190| 239 | 10719 | 70362 | 
| 31 | 15 django/db/models/fields/related.py | 870 | 897| 237 | 10956 | 84868 | 
| 32 | 16 django/apps/registry.py | 386 | 437| 465 | 11421 | 88296 | 
| 33 | 16 django/db/models/fields/related.py | 68 | 86| 223 | 11644 | 88296 | 
| 34 | 17 django/template/smartif.py | 70 | 114| 462 | 12106 | 89824 | 
| 35 | 17 django/db/migrations/autodetector.py | 768 | 799| 278 | 12384 | 89824 | 
| 36 | 18 django/db/models/sql/query.py | 1477 | 1503| 263 | 12647 | 112788 | 
| 37 | 19 django/contrib/contenttypes/fields.py | 564 | 621| 439 | 13086 | 118415 | 
| 38 | 19 django/core/checks/model_checks.py | 161 | 185| 267 | 13353 | 118415 | 
| 39 | 20 django/db/backends/dummy/base.py | 51 | 75| 173 | 13526 | 118856 | 
| 40 | 21 django/http/multipartparser.py | 374 | 399| 179 | 13705 | 124175 | 
| 41 | 21 django/conf/__init__.py | 136 | 154| 146 | 13851 | 124175 | 
| 42 | 21 django/contrib/contenttypes/fields.py | 722 | 742| 188 | 14039 | 124175 | 
| 43 | 22 django/db/models/expressions.py | 34 | 148| 834 | 14873 | 135865 | 
| 44 | 22 django/template/smartif.py | 44 | 67| 150 | 15023 | 135865 | 
| 45 | 23 django/db/models/options.py | 359 | 377| 124 | 15147 | 143367 | 
| 46 | 23 django/db/models/query.py | 1999 | 2133| 1129 | 16276 | 143367 | 
| 47 | 24 django/db/models/fields/related_descriptors.py | 752 | 822| 557 | 16833 | 154220 | 
| 48 | 25 django/utils/deprecation.py | 138 | 156| 122 | 16955 | 155264 | 
| 49 | 25 django/db/models/base.py | 2364 | 2416| 341 | 17296 | 155264 | 
| 50 | 25 django/http/multipartparser.py | 427 | 466| 258 | 17554 | 155264 | 
| 51 | 25 django/core/checks/model_checks.py | 93 | 116| 170 | 17724 | 155264 | 
| 52 | 25 django/db/models/fields/related_descriptors.py | 677 | 714| 341 | 18065 | 155264 | 
| 53 | 25 django/utils/deprecation.py | 38 | 80| 339 | 18404 | 155264 | 
| 54 | 25 django/contrib/contenttypes/fields.py | 657 | 691| 284 | 18688 | 155264 | 
| 55 | 25 django/db/models/options.py | 83 | 167| 657 | 19345 | 155264 | 
| 56 | 26 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 19465 | 157796 | 
| 57 | 27 django/views/debug.py | 75 | 102| 178 | 19643 | 162560 | 
| 58 | 27 django/utils/deprecation.py | 83 | 136| 382 | 20025 | 162560 | 
| 59 | 27 django/template/smartif.py | 117 | 151| 188 | 20213 | 162560 | 
| 60 | 27 django/db/models/fields/related_descriptors.py | 824 | 851| 222 | 20435 | 162560 | 
| 61 | 27 django/contrib/contenttypes/fields.py | 693 | 720| 214 | 20649 | 162560 | 
| 62 | 27 django/db/models/fields/related_descriptors.py | 612 | 675| 495 | 21144 | 162560 | 
| 63 | 27 django/db/models/options.py | 1 | 54| 319 | 21463 | 162560 | 
| 64 | 27 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 21721 | 162560 | 
| 65 | 27 django/db/models/sql/query.py | 1505 | 1537| 262 | 21983 | 162560 | 
| 66 | 27 django/db/models/fields/reverse_related.py | 185 | 203| 167 | 22150 | 162560 | 
| 67 | 27 django/db/models/base.py | 89 | 193| 817 | 22967 | 162560 | 
| 68 | 28 django/db/migrations/recorder.py | 24 | 46| 145 | 23112 | 163247 | 
| 69 | 29 django/db/models/__init__.py | 1 | 116| 682 | 23794 | 163929 | 
| 70 | 29 django/db/models/fields/related_descriptors.py | 716 | 750| 257 | 24051 | 163929 | 
| 71 | 29 django/db/migrations/autodetector.py | 592 | 766| 1213 | 25264 | 163929 | 
| 72 | 29 django/http/multipartparser.py | 401 | 425| 169 | 25433 | 163929 | 
| 73 | 29 django/db/migrations/autodetector.py | 980 | 1035| 415 | 25848 | 163929 | 
| 74 | 29 django/contrib/contenttypes/fields.py | 744 | 772| 254 | 26102 | 163929 | 
| 75 | 30 django/contrib/admin/options.py | 2469 | 2504| 315 | 26417 | 183182 | 
| 76 | 30 django/conf/__init__.py | 280 | 307| 177 | 26594 | 183182 | 
| 77 | 30 django/db/models/query.py | 1658 | 1714| 453 | 27047 | 183182 | 
| 78 | 30 django/db/models/fields/related_descriptors.py | 912 | 978| 588 | 27635 | 183182 | 
| 79 | 31 django/db/migrations/operations/models.py | 136 | 293| 906 | 28541 | 189823 | 
| 80 | 31 django/db/models/fields/related_descriptors.py | 1065 | 1091| 220 | 28761 | 189823 | 
| 81 | 31 django/db/migrations/autodetector.py | 1060 | 1177| 982 | 29743 | 189823 | 
| 82 | 32 django/contrib/admin/decorators.py | 34 | 77| 271 | 30014 | 190477 | 
| 83 | 33 django/db/models/deletion.py | 243 | 309| 536 | 30550 | 194409 | 
| 84 | 33 django/db/models/fields/related_descriptors.py | 1273 | 1342| 522 | 31072 | 194409 | 
| 85 | 33 django/db/models/query.py | 2301 | 2369| 785 | 31857 | 194409 | 
| 86 | 34 django/db/models/signals.py | 1 | 55| 337 | 32194 | 194746 | 
| 87 | 34 django/db/models/options.py | 321 | 357| 338 | 32532 | 194746 | 
| 88 | 34 django/db/models/fields/related.py | 776 | 802| 222 | 32754 | 194746 | 
| 89 | 35 django/template/loaders/cached.py | 1 | 66| 501 | 33255 | 195471 | 
| 90 | 35 django/contrib/contenttypes/fields.py | 623 | 655| 333 | 33588 | 195471 | 
| 91 | 35 django/db/models/fields/related_descriptors.py | 375 | 396| 158 | 33746 | 195471 | 
| 92 | 35 django/db/models/fields/related_descriptors.py | 1153 | 1187| 341 | 34087 | 195471 | 
| 93 | 35 django/db/models/options.py | 169 | 232| 596 | 34683 | 195471 | 
| 94 | 35 django/db/models/query.py | 2195 | 2264| 665 | 35348 | 195471 | 
| 95 | 36 django/db/models/manager.py | 1 | 173| 1252 | 36600 | 196930 | 
| 96 | 37 django/core/handlers/base.py | 136 | 172| 257 | 36857 | 199583 | 
| 97 | 37 django/db/models/fields/related.py | 732 | 754| 172 | 37029 | 199583 | 


### Hint

```
Could you please give some sample code with your use case?
In a boiled-down nutshell: def lazy_consumer(): # something more complex, obviously. return [1, 3, 5] consumer = SimpleLazyObject(lazy_consumer) # inside third party code ... def some_func(param): third_party_code = [...] # then, through parameter passing, my value is provided to be used. # param is at this point, `consumer` third_party_code_plus_mine = third_party_code + param which ultimately yields: TypeError: unsupported operand type(s) for +: 'list' and 'SimpleLazyObject'
Seems okay, although I'm not an expert on the SimpleLazyObject class.
Replying to kezabelle: def lazy_consumer(): # something more complex, obviously. return [1, 3, 5] consumer = SimpleLazyObject(lazy_consumer) If you know what is the resulting type or possible resulting types of your expression, I think you better use django.utils.functional.lazy which will provide all the necessary methods.
Replying to kezabelle: As far as I can tell, the implementation can't just be __radd__ = new_method_proxy(operator.radd) because that doesn't exist, which is rubbish. __radd__ = new_method_proxy(operator.attrgetter("__radd__")) also won't work because types may not have that attr, and attrgetter doesn't supress the exception (correctly) Wouldn't the following code work? __add__ = new_method_proxy(operator.add) __radd__ = new_method_proxy(lambda a, b: operator.add(b, a)) I have tested this and it seems to work as excepted.
```

## Patch

```diff
diff --git a/django/utils/functional.py b/django/utils/functional.py
--- a/django/utils/functional.py
+++ b/django/utils/functional.py
@@ -432,6 +432,12 @@ def __deepcopy__(self, memo):
             return result
         return copy.deepcopy(self._wrapped, memo)
 
+    __add__ = new_method_proxy(operator.add)
+
+    @new_method_proxy
+    def __radd__(self, other):
+        return other + self
+
 
 def partition(predicate, values):
     """

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_lazyobject.py b/tests/utils_tests/test_lazyobject.py
--- a/tests/utils_tests/test_lazyobject.py
+++ b/tests/utils_tests/test_lazyobject.py
@@ -317,6 +317,17 @@ def test_repr(self):
         self.assertIsInstance(obj._wrapped, int)
         self.assertEqual(repr(obj), "<SimpleLazyObject: 42>")
 
+    def test_add(self):
+        obj1 = self.lazy_wrap(1)
+        self.assertEqual(obj1 + 1, 2)
+        obj2 = self.lazy_wrap(2)
+        self.assertEqual(obj2 + obj1, 3)
+        self.assertEqual(obj1 + obj2, 3)
+
+    def test_radd(self):
+        obj1 = self.lazy_wrap(1)
+        self.assertEqual(1 + obj1, 2)
+
     def test_trace(self):
         # See ticket #19456
         old_trace_func = sys.gettrace()

```


## Code snippets

### 1 - django/utils/functional.py:

Start line: 272, End line: 374

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
        raise NotImplementedError(
            "subclasses of LazyObject must provide a _setup() method"
        )

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
### 2 - django/utils/functional.py:

Start line: 143, End line: 212

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

        def __add__(self, other):
            return self.__cast() + other

        def __radd__(self, other):
            return other + self.__cast()

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
### 3 - django/utils/functional.py:

Start line: 121, End line: 141

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
            if cls._delegate_bytes and cls._delegate_text:
                raise ValueError(
                    "Cannot call lazy() with both bytes and text return types."
                )
            if cls._delegate_text:
                cls.__str__ = cls.__text_cast
            elif cls._delegate_bytes:
                cls.__bytes__ = cls.__bytes_cast
    # ... other code
```
### 4 - django/utils/functional.py:

Start line: 215, End line: 269

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
            if any(
                isinstance(arg, Promise)
                for arg in itertools.chain(args, kwargs.values())
            ):
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
### 5 - django/utils/functional.py:

Start line: 87, End line: 119

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
                (func, self.__args, self.__kw) + resultclasses,
            )

        def __repr__(self):
            return repr(self.__cast())
    # ... other code
```
### 6 - django/utils/functional.py:

Start line: 377, End line: 448

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
        self.__dict__["_setupfunc"] = func
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
        return "<%s: %r>" % (type(self).__name__, repr_attr)

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
### 7 - django/contrib/gis/db/models/proxy.py:

Start line: 1, End line: 47

```python
"""
The SpatialProxy object allows for lazy-geometries and lazy-rasters. The proxy
uses Python descriptors for instantiating and setting Geometry or Raster
objects corresponding to geographic model fields.

Thanks to Robert Coup for providing this functionality (see #4322).
"""
from django.db.models.query_utils import DeferredAttribute


class SpatialProxy(DeferredAttribute):
    def __init__(self, klass, field, load_func=None):
        """
        Initialize on the given Geometry or Raster class (not an instance)
        and the corresponding field.
        """
        self._klass = klass
        self._load_func = load_func or klass
        super().__init__(field)

    def __get__(self, instance, cls=None):
        """
        Retrieve the geometry or raster, initializing it using the
        corresponding class specified during initialization and the value of
        the field. Currently, GEOS or OGR geometries as well as GDALRasters are
        supported.
        """
        if instance is None:
            # Accessed on a class, not an instance
            return self

        # Getting the value of the field.
        try:
            geo_value = instance.__dict__[self.field.attname]
        except KeyError:
            geo_value = super().__get__(instance, cls)

        if isinstance(geo_value, self._klass):
            geo_obj = geo_value
        elif (geo_value is None) or (geo_value == ""):
            geo_obj = None
        else:
            # Otherwise, a geometry or raster object is built using the field's
            # contents, and the model's corresponding attribute is set.
            geo_obj = self._load_func(geo_value)
            setattr(instance, self.field.attname, geo_obj)
        return geo_obj
```
### 8 - django/conf/__init__.py:

Start line: 56, End line: 79

```python
class LazySettings(LazyObject):
    """
    A lazy proxy for either global Django settings or a custom settings object.
    The user can manually configure settings prior to using them. Otherwise,
    Django uses the settings module pointed to by DJANGO_SETTINGS_MODULE.
    """

    def _setup(self, name=None):
        """
        Load the settings module pointed to by the environment variable. This
        is used the first time settings are needed, if the user hasn't
        configured settings manually.
        """
        settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
        if not settings_module:
            desc = ("setting %s" % name) if name else "settings"
            raise ImproperlyConfigured(
                "Requested %s, but settings are not configured. "
                "You must either define the environment variable %s "
                "or call settings.configure() before accessing settings."
                % (desc, ENVIRONMENT_VARIABLE)
            )

        self._wrapped = Settings(settings_module)
```
### 9 - django/utils/functional.py:

Start line: 1, End line: 58

```python
import copy
import itertools
import operator
import warnings
from functools import total_ordering, wraps


class cached_property:
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    A cached property can be made out of an existing method:
    (e.g. ``url = cached_property(get_absolute_url)``).
    """

    name = None

    @staticmethod
    def func(instance):
        raise TypeError(
            "Cannot use cached_property instance without calling "
            "__set_name__() on it."
        )

    def __init__(self, func, name=None):
        from django.utils.deprecation import RemovedInDjango50Warning

        if name is not None:
            warnings.warn(
                "The name argument is deprecated as it's unnecessary as of "
                "Python 3.6.",
                RemovedInDjango50Warning,
                stacklevel=2,
            )
        self.real_func = func
        self.__doc__ = getattr(func, "__doc__")

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
### 10 - django/core/checks/model_checks.py:

Start line: 187, End line: 228

```python
def _check_lazy_references(apps, ignore=None):
    # ... other code

    def default_error(model_key, func, args, keywords):
        error_msg = (
            "%(op)s contains a lazy reference to %(model)s, but %(model_error)s."
        )
        params = {
            "op": func,
            "model": ".".join(model_key),
            "model_error": app_model_error(model_key),
        }
        return Error(error_msg % params, obj=func, id="models.E022")

    # Maps common uses of lazy operations to corresponding error functions
    # defined above. If a key maps to None, no error will be produced.
    # default_error() will be used for usages that don't appear in this dict.
    known_lazy = {
        ("django.db.models.fields.related", "resolve_related_class"): field_error,
        ("django.db.models.fields.related", "set_managed"): None,
        ("django.dispatch.dispatcher", "connect"): signal_connect_error,
    }

    def build_error(model_key, func, args, keywords):
        key = (func.__module__, func.__name__)
        error_fn = known_lazy.get(key, default_error)
        return error_fn(model_key, func, args, keywords) if error_fn else None

    return sorted(
        filter(
            None,
            (
                build_error(model_key, *extract_operation(func))
                for model_key in pending_models
                for func in apps._pending_operations[model_key]
            ),
        ),
        key=lambda error: error.msg,
    )


@register(Tags.models)
def check_lazy_references(app_configs=None, **kwargs):
    return _check_lazy_references(apps)
```
### 19 - django/utils/functional.py:

Start line: 61, End line: 84

```python
class classproperty:
    """
    Decorator that converts a method with a single cls argument into a property
    that can be accessed directly from the class.
    """

    def __init__(self, method=None):
        self.fget = method

    def __get__(self, instance, cls=None):
        return self.fget(cls)

    def getter(self, method):
        self.fget = method
        return self


class Promise:
    """
    Base class for the proxy class created in the closure of the lazy function.
    It's used to recognize promises in code.
    """

    pass
```
