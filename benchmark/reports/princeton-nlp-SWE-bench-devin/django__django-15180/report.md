# django__django-15180

| **django/django** | `7e4a9a9f696574a18f5c98f34d5a88e254b2d394` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 222 |
| **Any found context length** | 222 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/urls/conf.py b/django/urls/conf.py
--- a/django/urls/conf.py
+++ b/django/urls/conf.py
@@ -57,6 +57,10 @@ def include(arg, namespace=None):
 def _path(route, view, kwargs=None, name=None, Pattern=None):
     from django.views import View
 
+    if kwargs is not None and not isinstance(kwargs, dict):
+        raise TypeError(
+            f'kwargs argument must be a dict, but got {kwargs.__class__.__name__}.'
+        )
     if isinstance(view, (list, tuple)):
         # For include(...) processing.
         pattern = Pattern(route, is_endpoint=False)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/urls/conf.py | 60 | 60 | 1 | 1 | 222


## Problem Statement

```
path()/re_path() should raise a TypeError when kwargs is not a dict.
Description
	
Apparently, however many years into using Django, I'm still capable of making a "newbie" mistake and getting confused. So perhaps other actual new users encounter similar, especially given the lack of typing specifiers.
I defined a URL like so:
urlpatterns = [
	path("path/to/thing", MyView.as_view(), "my_view"),
]
which ... well, you either spot the issue immediately or you don't, and end up with the following. If you try and resolve() the path (eg: by making a request in your browser), you'll get something like:
In [3]: resolve("/path/to/thing")
~/Code/django/django/urls/base.py in resolve(path, urlconf)
	 22	 if urlconf is None:
	 23		 urlconf = get_urlconf()
---> 24	 return get_resolver(urlconf).resolve(path)
	 25
	 26
~/Code/django/django/urls/resolvers.py in resolve(self, path)
	586			 for pattern in self.url_patterns:
	587				 try:
--> 588					 sub_match = pattern.resolve(new_path)
	589				 except Resolver404 as e:
	590					 self._extend_tried(tried, pattern, e.args[0].get('tried'))
~/Code/django/django/urls/resolvers.py in resolve(self, path)
	388			 new_path, args, kwargs = match
	389			 # Pass any extra_kwargs as **kwargs.
--> 390			 kwargs.update(self.default_args)
	391			 return ResolverMatch(self.callback, args, kwargs, self.pattern.name, route=str(self.pattern))
	392
ValueError: dictionary update sequence element #0 has length 1; 2 is required
The crux of the issue being that I meant to give the URL a name, and it's a super unfortunate history that kwargs comes before the name argument (because nearly everyone gives a URL a name, but passing static kwargs is comparatively infrequent). So what's actually happened is that kwargs = "my_view" and eventually self.default_args = "my_view".
If I update to path("path/to/thing", MyView.as_view(), "my_view", name="my_view"), leaving the type incorrect, I can get the following error via reverse, too:
In [4]: reverse("my_view")
~/Code/django/django/urls/base.py in reverse(viewname, urlconf, args, kwargs, current_app)
	 84			 resolver = get_ns_resolver(ns_pattern, resolver, tuple(ns_converters.items()))
	 85
---> 86	 return resolver._reverse_with_prefix(view, prefix, *args, **kwargs)
	 87
	 88
~/Code/django/django/urls/resolvers.py in _reverse_with_prefix(self, lookup_view, _prefix, *args, **kwargs)
	669					 if set(kwargs).symmetric_difference(params).difference(defaults):
	670						 continue
--> 671					 if any(kwargs.get(k, v) != v for k, v in defaults.items()):
	672						 continue
	673					 candidate_subs = kwargs
AttributeError: 'str' object has no attribute 'items'
Both of these suggest that either there should be a type-guard in _path to assert it's dict-ish (if not None), or a system check on URLPattern to raise a friendly message. Well, they actually continue to suggest to me that everything after the view argument should be keyword-only, or that kwargs should come later, but I suspect those to be a harder sell ;)
This is specifically around the kwargs, but it doesn't look like there's any guarding on the name either, and I feel like a name of {'test': 'test'} (i.e. accidentally swapped both positionals) is likely to bite & cause an issue somewhere.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/urls/conf.py** | 57 | 86| 222 | 222 | 669 | 
| 2 | 2 django/urls/resolvers.py | 657 | 730| 681 | 903 | 6488 | 
| 3 | 2 django/urls/resolvers.py | 584 | 620| 347 | 1250 | 6488 | 
| 4 | 3 django/urls/base.py | 27 | 86| 438 | 1688 | 7671 | 
| 5 | 3 django/urls/resolvers.py | 290 | 306| 160 | 1848 | 7671 | 
| 6 | 3 django/urls/resolvers.py | 645 | 655| 120 | 1968 | 7671 | 
| 7 | 3 django/urls/resolvers.py | 343 | 395| 370 | 2338 | 7671 | 
| 8 | 3 django/urls/resolvers.py | 266 | 288| 173 | 2511 | 7671 | 
| 9 | 4 django/shortcuts.py | 102 | 141| 280 | 2791 | 8768 | 
| 10 | 5 django/views/debug.py | 1 | 55| 351 | 3142 | 13521 | 
| 11 | 5 django/urls/resolvers.py | 413 | 431| 174 | 3316 | 13521 | 
| 12 | 6 django/contrib/admindocs/urls.py | 1 | 51| 307 | 3623 | 13828 | 
| 13 | 6 django/urls/resolvers.py | 124 | 154| 232 | 3855 | 13828 | 
| 14 | 6 django/urls/resolvers.py | 397 | 410| 121 | 3976 | 13828 | 
| 15 | 7 django/core/checks/urls.py | 71 | 111| 264 | 4240 | 14529 | 
| 16 | 7 django/urls/base.py | 1 | 24| 170 | 4410 | 14529 | 
| 17 | 7 django/urls/resolvers.py | 33 | 76| 410 | 4820 | 14529 | 
| 18 | 7 django/urls/resolvers.py | 433 | 449| 164 | 4984 | 14529 | 
| 19 | 7 django/urls/resolvers.py | 542 | 582| 282 | 5266 | 14529 | 
| 20 | 7 django/urls/resolvers.py | 183 | 217| 265 | 5531 | 14529 | 
| 21 | 7 django/urls/resolvers.py | 1 | 30| 216 | 5747 | 14529 | 
| 22 | 7 django/urls/resolvers.py | 481 | 540| 548 | 6295 | 14529 | 
| 23 | 7 django/urls/resolvers.py | 157 | 181| 203 | 6498 | 14529 | 
| 24 | 8 django/views/generic/__init__.py | 1 | 23| 189 | 6687 | 14719 | 
| 25 | 8 django/urls/resolvers.py | 451 | 479| 288 | 6975 | 14719 | 
| 26 | 9 django/contrib/admindocs/views.py | 1 | 31| 240 | 7215 | 18048 | 
| 27 | 10 django/urls/__init__.py | 1 | 24| 239 | 7454 | 18287 | 
| 28 | 10 django/urls/resolvers.py | 622 | 643| 187 | 7641 | 18287 | 
| 29 | 11 django/views/defaults.py | 1 | 24| 149 | 7790 | 19315 | 
| 30 | 12 django/db/models/base.py | 424 | 529| 913 | 8703 | 36933 | 
| 31 | 13 django/views/decorators/debug.py | 77 | 93| 138 | 8841 | 37528 | 
| 32 | **13 django/urls/conf.py** | 1 | 54| 447 | 9288 | 37528 | 
| 33 | 14 django/views/csrf.py | 15 | 100| 835 | 10123 | 39072 | 
| 34 | 15 django/views/generic/base.py | 191 | 222| 247 | 10370 | 40722 | 
| 35 | 16 django/utils/regex_helper.py | 78 | 190| 962 | 11332 | 43363 | 
| 36 | 17 django/contrib/auth/urls.py | 1 | 21| 224 | 11556 | 43587 | 
| 37 | 18 django/template/defaulttags.py | 1267 | 1331| 504 | 12060 | 54239 | 
| 38 | 18 django/views/generic/base.py | 29 | 45| 136 | 12196 | 54239 | 
| 39 | 18 django/contrib/admindocs/views.py | 407 | 419| 111 | 12307 | 54239 | 
| 40 | 19 django/utils/html.py | 269 | 313| 390 | 12697 | 57486 | 
| 41 | 20 django/core/validators.py | 62 | 98| 575 | 13272 | 61997 | 
| 42 | 20 django/urls/resolvers.py | 309 | 340| 190 | 13462 | 61997 | 
| 43 | 21 django/views/decorators/cache.py | 28 | 42| 119 | 13581 | 62456 | 
| 44 | 22 django/contrib/admin/options.py | 1551 | 1641| 770 | 14351 | 81140 | 
| 45 | 23 django/core/checks/model_checks.py | 178 | 211| 332 | 14683 | 82925 | 
| 46 | 24 django/utils/deprecation.py | 33 | 73| 336 | 15019 | 83950 | 
| 47 | 25 django/core/checks/security/csrf.py | 1 | 42| 304 | 15323 | 84412 | 
| 48 | 25 django/views/generic/base.py | 1 | 26| 142 | 15465 | 84412 | 
| 49 | 25 django/shortcuts.py | 23 | 54| 243 | 15708 | 84412 | 
| 50 | 26 django/core/checks/security/base.py | 1 | 72| 660 | 16368 | 86453 | 
| 51 | 27 django/contrib/gis/views.py | 1 | 21| 155 | 16523 | 86608 | 
| 52 | 28 django/core/checks/templates.py | 1 | 41| 301 | 16824 | 87073 | 
| 53 | 29 django/contrib/auth/views.py | 1 | 37| 284 | 17108 | 89803 | 
| 54 | 30 docs/conf.py | 128 | 231| 914 | 18022 | 93276 | 
| 55 | 30 django/views/generic/base.py | 47 | 83| 339 | 18361 | 93276 | 
| 56 | 31 django/conf/urls/__init__.py | 1 | 10| 0 | 18361 | 93341 | 
| 57 | 31 django/utils/regex_helper.py | 1 | 38| 250 | 18611 | 93341 | 
| 58 | 32 django/utils/http.py | 317 | 355| 402 | 19013 | 96583 | 
| 59 | 32 django/core/checks/urls.py | 1 | 27| 142 | 19155 | 96583 | 
| 60 | 33 django/urls/exceptions.py | 1 | 10| 0 | 19155 | 96608 | 
| 61 | 33 django/core/checks/security/csrf.py | 45 | 68| 157 | 19312 | 96608 | 
| 62 | 33 django/urls/resolvers.py | 79 | 98| 168 | 19480 | 96608 | 
| 63 | 34 django/urls/utils.py | 1 | 63| 460 | 19940 | 97068 | 
| 64 | 34 django/utils/regex_helper.py | 41 | 76| 375 | 20315 | 97068 | 
| 65 | 35 django/core/checks/messages.py | 54 | 77| 161 | 20476 | 97645 | 
| 66 | 36 django/contrib/admindocs/utils.py | 192 | 241| 548 | 21024 | 99669 | 
| 67 | 36 django/urls/base.py | 89 | 155| 383 | 21407 | 99669 | 
| 68 | 37 django/views/generic/edit.py | 1 | 69| 484 | 21891 | 101664 | 
| 69 | 37 django/core/checks/templates.py | 44 | 68| 162 | 22053 | 101664 | 
| 70 | 37 django/core/validators.py | 100 | 143| 468 | 22521 | 101664 | 
| 71 | 38 docs/_ext/djangodocs.py | 26 | 71| 398 | 22919 | 104873 | 
| 72 | 38 django/contrib/admindocs/utils.py | 1 | 28| 184 | 23103 | 104873 | 
| 73 | 38 docs/_ext/djangodocs.py | 109 | 176| 572 | 23675 | 104873 | 
| 74 | 39 django/utils/inspect.py | 51 | 80| 198 | 23873 | 105386 | 
| 75 | 39 django/urls/resolvers.py | 220 | 263| 379 | 24252 | 105386 | 
| 76 | 40 django/views/__init__.py | 1 | 4| 0 | 24252 | 105401 | 
| 77 | 41 django/http/request.py | 151 | 158| 111 | 24363 | 110607 | 
| 78 | 41 docs/_ext/djangodocs.py | 74 | 106| 255 | 24618 | 110607 | 
| 79 | 42 django/middleware/csrf.py | 411 | 464| 572 | 25190 | 114760 | 
| 80 | 43 django/utils/autoreload.py | 1 | 56| 287 | 25477 | 119908 | 
| 81 | 43 django/core/checks/urls.py | 30 | 50| 165 | 25642 | 119908 | 
| 82 | 43 django/core/checks/security/base.py | 234 | 258| 210 | 25852 | 119908 | 
| 83 | 43 django/views/defaults.py | 27 | 78| 403 | 26255 | 119908 | 
| 84 | 43 django/core/validators.py | 1 | 14| 111 | 26366 | 119908 | 
| 85 | 44 django/views/i18n.py | 79 | 182| 702 | 27068 | 122369 | 
| 86 | 45 django/conf/global_settings.py | 157 | 272| 859 | 27927 | 128176 | 
| 87 | 45 django/middleware/csrf.py | 294 | 344| 450 | 28377 | 128176 | 
| 88 | 46 django/conf/urls/static.py | 1 | 29| 197 | 28574 | 128373 | 
| 89 | 46 django/core/checks/security/base.py | 74 | 168| 746 | 29320 | 128373 | 
| 90 | 47 django/template/base.py | 602 | 637| 361 | 29681 | 136559 | 
| 91 | 47 django/views/csrf.py | 101 | 155| 577 | 30258 | 136559 | 
| 92 | 48 django/http/response.py | 505 | 523| 186 | 30444 | 141273 | 
| 93 | 49 django/contrib/auth/decorators.py | 1 | 35| 313 | 30757 | 141860 | 
| 94 | 49 django/views/defaults.py | 102 | 121| 149 | 30906 | 141860 | 
| 95 | 50 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 31044 | 141999 | 
| 96 | 51 django/db/models/options.py | 1 | 35| 300 | 31344 | 149361 | 
| 97 | 52 django/db/models/fields/related.py | 1281 | 1398| 963 | 32307 | 163520 | 
| 98 | 52 django/views/debug.py | 466 | 510| 429 | 32736 | 163520 | 
| 99 | 52 django/contrib/admin/options.py | 1 | 96| 756 | 33492 | 163520 | 
| 100 | 52 django/template/base.py | 1 | 93| 758 | 34250 | 163520 | 
| 101 | 53 django/db/migrations/exceptions.py | 1 | 55| 249 | 34499 | 163770 | 
| 102 | 54 django/core/handlers/base.py | 160 | 210| 379 | 34878 | 166377 | 
| 103 | 55 django/db/models/sql/query.py | 1493 | 1581| 816 | 35694 | 189047 | 
| 104 | 55 django/template/base.py | 735 | 757| 207 | 35901 | 189047 | 
| 105 | 56 django/contrib/admin/sites.py | 1 | 35| 225 | 36126 | 193476 | 
| 106 | 57 django/contrib/auth/checks.py | 1 | 102| 720 | 36846 | 194974 | 
| 107 | 58 django/http/__init__.py | 1 | 22| 197 | 37043 | 195171 | 
| 108 | 58 django/contrib/admindocs/views.py | 379 | 404| 186 | 37229 | 195171 | 
| 109 | 58 django/views/debug.py | 427 | 464| 298 | 37527 | 195171 | 
| 110 | 59 django/conf/__init__.py | 174 | 239| 605 | 38132 | 197530 | 


### Hint

```
I agree that this behavior should be edited, but I think the ticket type should be Cleanup/optimization.
Well, they actually continue to suggest to me that everything after the view argument should be keyword-only, or that kwargs should come later, but I suspect those to be a harder sell ;) Keyword-only arguments would be great, but it will affect too many users. We reject such tickets in most of cases, however here it's justified because kwargs as a positional argument can be confusing, so let's raise a TypeError when kwargs is not a dict.
```

## Patch

```diff
diff --git a/django/urls/conf.py b/django/urls/conf.py
--- a/django/urls/conf.py
+++ b/django/urls/conf.py
@@ -57,6 +57,10 @@ def include(arg, namespace=None):
 def _path(route, view, kwargs=None, name=None, Pattern=None):
     from django.views import View
 
+    if kwargs is not None and not isinstance(kwargs, dict):
+        raise TypeError(
+            f'kwargs argument must be a dict, but got {kwargs.__class__.__name__}.'
+        )
     if isinstance(view, (list, tuple)):
         # For include(...) processing.
         pattern = Pattern(route, is_endpoint=False)

```

## Test Patch

```diff
diff --git a/tests/urlpatterns/tests.py b/tests/urlpatterns/tests.py
--- a/tests/urlpatterns/tests.py
+++ b/tests/urlpatterns/tests.py
@@ -4,7 +4,9 @@
 from django.core.exceptions import ImproperlyConfigured
 from django.test import SimpleTestCase
 from django.test.utils import override_settings
-from django.urls import NoReverseMatch, Resolver404, path, resolve, reverse
+from django.urls import (
+    NoReverseMatch, Resolver404, path, re_path, resolve, reverse,
+)
 from django.views import View
 
 from .converters import DynamicConverter
@@ -137,6 +139,13 @@ def test_path_inclusion_is_reversible(self):
         url = reverse('inner-extra', kwargs={'extra': 'something'})
         self.assertEqual(url, '/included_urls/extra/something/')
 
+    def test_invalid_kwargs(self):
+        msg = 'kwargs argument must be a dict, but got str.'
+        with self.assertRaisesMessage(TypeError, msg):
+            path('hello/', empty_view, 'name')
+        with self.assertRaisesMessage(TypeError, msg):
+            re_path('^hello/$', empty_view, 'name')
+
     def test_invalid_converter(self):
         msg = "URL route 'foo/<nonexistent:var>/' uses invalid converter 'nonexistent'."
         with self.assertRaisesMessage(ImproperlyConfigured, msg):

```


## Code snippets

### 1 - django/urls/conf.py:

Start line: 57, End line: 86

```python
def _path(route, view, kwargs=None, name=None, Pattern=None):
    from django.views import View

    if isinstance(view, (list, tuple)):
        # For include(...) processing.
        pattern = Pattern(route, is_endpoint=False)
        urlconf_module, app_name, namespace = view
        return URLResolver(
            pattern,
            urlconf_module,
            kwargs,
            app_name=app_name,
            namespace=namespace,
        )
    elif callable(view):
        pattern = Pattern(route, name=name, is_endpoint=True)
        return URLPattern(pattern, view, kwargs, name)
    elif isinstance(view, View):
        view_cls_name = view.__class__.__name__
        raise TypeError(
            f'view must be a callable, pass {view_cls_name}.as_view(), not '
            f'{view_cls_name}().'
        )
    else:
        raise TypeError('view must be a callable or a list/tuple in the case of include().')


path = partial(_path, Pattern=RoutePattern)
re_path = partial(_path, Pattern=RegexPattern)
```
### 2 - django/urls/resolvers.py:

Start line: 657, End line: 730

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
                arg_msg = "keyword arguments '%s'" % kwargs
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
### 3 - django/urls/resolvers.py:

Start line: 584, End line: 620

```python
class URLResolver:

    def resolve(self, path):
        path = str(path)  # path may be a reverse_lazy object
        tried = []
        match = self.pattern.match(path)
        if match:
            new_path, args, kwargs = match
            for pattern in self.url_patterns:
                try:
                    sub_match = pattern.resolve(new_path)
                except Resolver404 as e:
                    self._extend_tried(tried, pattern, e.args[0].get('tried'))
                else:
                    if sub_match:
                        # Merge captured arguments in match with submatch
                        sub_match_dict = {**kwargs, **self.default_kwargs}
                        # Update the sub_match_dict with the kwargs from the sub_match.
                        sub_match_dict.update(sub_match.kwargs)
                        # If there are *any* named groups, ignore all non-named groups.
                        # Otherwise, pass all non-named arguments as positional arguments.
                        sub_match_args = sub_match.args
                        if not sub_match_dict:
                            sub_match_args = args + sub_match.args
                        current_route = '' if isinstance(pattern, URLPattern) else str(pattern.pattern)
                        self._extend_tried(tried, pattern, sub_match.tried)
                        return ResolverMatch(
                            sub_match.func,
                            sub_match_args,
                            sub_match_dict,
                            sub_match.url_name,
                            [self.app_name] + sub_match.app_names,
                            [self.namespace] + sub_match.namespaces,
                            self._join_route(current_route, sub_match.route),
                            tried,
                        )
                    tried.append([pattern])
            raise Resolver404({'tried': tried, 'path': new_path})
        raise Resolver404({'path': path})
```
### 4 - django/urls/base.py:

Start line: 27, End line: 86

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

    return resolver._reverse_with_prefix(view, prefix, *args, **kwargs)
```
### 5 - django/urls/resolvers.py:

Start line: 290, End line: 306

```python
class RoutePattern(CheckURLMixin):

    def check(self):
        warnings = self._check_pattern_startswith_slash()
        route = self._route
        if '(?P<' in route or route.startswith('^') or route.endswith('$'):
            warnings.append(Warning(
                "Your URL pattern {} has a route that contains '(?P<', begins "
                "with a '^', or ends with a '$'. This was likely an oversight "
                "when migrating to django.urls.path().".format(self.describe()),
                id='2_0.W001',
            ))
        return warnings

    def _compile(self, route):
        return re.compile(_route_to_regex(route, self._is_endpoint)[0])

    def __str__(self):
        return str(self._route)
```
### 6 - django/urls/resolvers.py:

Start line: 645, End line: 655

```python
class URLResolver:

    def resolve_error_handler(self, view_type):
        callback = getattr(self.urlconf_module, 'handler%s' % view_type, None)
        if not callback:
            # No handler specified in file; use lazy import, since
            # django.conf.urls imports this file.
            from django.conf import urls
            callback = getattr(urls, 'handler%s' % view_type)
        return get_callable(callback)

    def reverse(self, lookup_view, *args, **kwargs):
        return self._reverse_with_prefix(lookup_view, '', *args, **kwargs)
```
### 7 - django/urls/resolvers.py:

Start line: 343, End line: 395

```python
class URLPattern:
    def __init__(self, pattern, callback, default_args=None, name=None):
        self.pattern = pattern
        self.callback = callback  # the view
        self.default_args = default_args or {}
        self.name = name

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.pattern.describe())

    def check(self):
        warnings = self._check_pattern_name()
        warnings.extend(self.pattern.check())
        warnings.extend(self._check_callback())
        return warnings

    def _check_pattern_name(self):
        """
        Check that the pattern name does not contain a colon.
        """
        if self.pattern.name is not None and ":" in self.pattern.name:
            warning = Warning(
                "Your URL pattern {} has a name including a ':'. Remove the colon, to "
                "avoid ambiguous namespace references.".format(self.pattern.describe()),
                id="urls.W003",
            )
            return [warning]
        else:
            return []

    def _check_callback(self):
        from django.views import View

        view = self.callback
        if inspect.isclass(view) and issubclass(view, View):
            return [Error(
                'Your URL pattern %s has an invalid view, pass %s.as_view() '
                'instead of %s.' % (
                    self.pattern.describe(),
                    view.__name__,
                    view.__name__,
                ),
                id='urls.E009',
            )]
        return []

    def resolve(self, path):
        match = self.pattern.match(path)
        if match:
            new_path, args, kwargs = match
            # Pass any extra_kwargs as **kwargs.
            kwargs.update(self.default_args)
            return ResolverMatch(self.callback, args, kwargs, self.pattern.name, route=str(self.pattern))
```
### 8 - django/urls/resolvers.py:

Start line: 266, End line: 288

```python
class RoutePattern(CheckURLMixin):
    regex = LocaleRegexDescriptor('_route')

    def __init__(self, route, name=None, is_endpoint=False):
        self._route = route
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = _route_to_regex(str(route), is_endpoint)[1]

    def match(self, path):
        match = self.regex.search(path)
        if match:
            # RoutePattern doesn't allow non-named groups so args are ignored.
            kwargs = match.groupdict()
            for key, value in kwargs.items():
                converter = self.converters[key]
                try:
                    kwargs[key] = converter.to_python(value)
                except ValueError:
                    return None
            return path[match.end():], (), kwargs
        return None
```
### 9 - django/shortcuts.py:

Start line: 102, End line: 141

```python
def resolve_url(to, *args, **kwargs):
    """
    Return a URL appropriate for the arguments passed.

    The arguments could be:

        * A model: the model's `get_absolute_url()` function will be called.

        * A view name, possibly with arguments: `urls.reverse()` will be used
          to reverse-resolve the name.

        * A URL, which will be returned as-is.
    """
    # If it's a model, use get_absolute_url()
    if hasattr(to, 'get_absolute_url'):
        return to.get_absolute_url()

    if isinstance(to, Promise):
        # Expand the lazy instance, as it can cause issues when it is passed
        # further to some Python functions like urlparse.
        to = str(to)

    # Handle relative URLs
    if isinstance(to, str) and to.startswith(('./', '../')):
        return to

    # Next try a reverse URL resolution.
    try:
        return reverse(to, args=args, kwargs=kwargs)
    except NoReverseMatch:
        # If this is a callable, re-raise.
        if callable(to):
            raise
        # If this doesn't "feel" like a URL, re-raise.
        if '/' not in to and '.' not in to:
            raise

    # Finally, fall back and assume it's a URL
    return to
```
### 10 - django/views/debug.py:

Start line: 1, End line: 55

```python
import functools
import re
import sys
import types
import warnings
from pathlib import Path

from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import get_docs_version

# Minimal Django templates engine to render the error templates
# regardless of the project's TEMPLATES setting. Templates are
# read directly from the filesystem so that the error handler
# works even if the template loader is broken.
DEBUG_ENGINE = Engine(
    debug=True,
    libraries={'i18n': 'django.templatetags.i18n'},
)


def builtin_template_path(name):
    """
    Return a path to a builtin template.

    Avoid calling this function at the module level or in a class-definition
    because __file__ may not exist, e.g. in frozen environments.
    """
    return Path(__file__).parent / 'templates' / name


class ExceptionCycleWarning(UserWarning):
    pass


class CallableSettingWrapper:
    """
    Object to wrap callable appearing in settings.
    * Not to call in the debug page (#21345).
    * Not to break the debug page if the callable forbidding to set attributes
      (#23070).
    """
    def __init__(self, callable_setting):
        self._wrapped = callable_setting

    def __repr__(self):
        return repr(self._wrapped)
```
### 32 - django/urls/conf.py:

Start line: 1, End line: 54

```python
"""Functions for use in URLsconfs."""
from functools import partial
from importlib import import_module

from django.core.exceptions import ImproperlyConfigured

from .resolvers import (
    LocalePrefixPattern, RegexPattern, RoutePattern, URLPattern, URLResolver,
)


def include(arg, namespace=None):
    app_name = None
    if isinstance(arg, tuple):
        # Callable returning a namespace hint.
        try:
            urlconf_module, app_name = arg
        except ValueError:
            if namespace:
                raise ImproperlyConfigured(
                    'Cannot override the namespace for a dynamic module that '
                    'provides a namespace.'
                )
            raise ImproperlyConfigured(
                'Passing a %d-tuple to include() is not supported. Pass a '
                '2-tuple containing the list of patterns and app_name, and '
                'provide the namespace argument to include() instead.' % len(arg)
            )
    else:
        # No namespace hint - use manually provided namespace.
        urlconf_module = arg

    if isinstance(urlconf_module, str):
        urlconf_module = import_module(urlconf_module)
    patterns = getattr(urlconf_module, 'urlpatterns', urlconf_module)
    app_name = getattr(urlconf_module, 'app_name', app_name)
    if namespace and not app_name:
        raise ImproperlyConfigured(
            'Specifying a namespace in include() without providing an app_name '
            'is not supported. Set the app_name attribute in the included '
            'module, or pass a 2-tuple containing the list of patterns and '
            'app_name instead.',
        )
    namespace = namespace or app_name
    # Make sure the patterns can be iterated through (without this, some
    # testcases will break).
    if isinstance(patterns, (list, tuple)):
        for url_pattern in patterns:
            pattern = getattr(url_pattern, 'pattern', None)
            if isinstance(pattern, LocalePrefixPattern):
                raise ImproperlyConfigured(
                    'Using i18n_patterns in an included URLconf is not allowed.'
                )
    return (urlconf_module, app_name, namespace)
```
