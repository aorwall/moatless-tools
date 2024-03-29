# django__django-15789

| **django/django** | `d4d5427571b4bf3a21c902276c2a00215c2a37cc` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 156 |
| **Any found context length** | 156 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/html.py b/django/utils/html.py
--- a/django/utils/html.py
+++ b/django/utils/html.py
@@ -59,7 +59,7 @@ def escapejs(value):
 }
 
 
-def json_script(value, element_id=None):
+def json_script(value, element_id=None, encoder=None):
     """
     Escape all the HTML/XML special characters with their unicode escapes, so
     value is safe to be output anywhere except for inside a tag attribute. Wrap
@@ -67,7 +67,9 @@ def json_script(value, element_id=None):
     """
     from django.core.serializers.json import DjangoJSONEncoder
 
-    json_str = json.dumps(value, cls=DjangoJSONEncoder).translate(_json_script_escapes)
+    json_str = json.dumps(value, cls=encoder or DjangoJSONEncoder).translate(
+        _json_script_escapes
+    )
     if element_id:
         template = '<script id="{}" type="application/json">{}</script>'
         args = (element_id, mark_safe(json_str))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/html.py | 62 | 62 | 1 | 1 | 156
| django/utils/html.py | 70 | 70 | 1 | 1 | 156


## Problem Statement

```
Add an encoder parameter to django.utils.html.json_script().
Description
	
I have a use case where I want to customize the JSON encoding of some values to output to the template layer. It looks like django.utils.html.json_script is a good utility for that, however the JSON encoder is hardcoded to DjangoJSONEncoder. I think it would be nice to be able to pass a custom encoder class.
By the way, django.utils.html.json_script is not documented (only its template filter counterpart is), would it be a good thing to add to the docs?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/utils/html.py** | 62 | 77| 156 | 156 | 3252 | 
| 2 | 2 django/forms/fields.py | 1337 | 1391| 364 | 520 | 12888 | 
| 3 | 3 docs/_ext/djangodocs.py | 178 | 213| 290 | 810 | 16112 | 
| 4 | **3 django/utils/html.py** | 1 | 59| 440 | 1250 | 16112 | 
| 5 | 4 django/core/serializers/json.py | 62 | 107| 336 | 1586 | 16816 | 
| 6 | 5 django/db/models/fields/json.py | 1 | 44| 278 | 1864 | 21094 | 
| 7 | 6 django/contrib/admin/templatetags/admin_modify.py | 1 | 58| 393 | 2257 | 22138 | 
| 8 | 6 django/db/models/fields/json.py | 70 | 126| 366 | 2623 | 22138 | 
| 9 | 7 django/contrib/messages/storage/cookie.py | 1 | 26| 183 | 2806 | 23438 | 
| 10 | 8 django/template/defaultfilters.py | 697 | 729| 217 | 3023 | 29918 | 
| 11 | 9 django/core/serializers/__init__.py | 1 | 52| 287 | 3310 | 31688 | 
| 12 | 9 docs/_ext/djangodocs.py | 111 | 175| 567 | 3877 | 31688 | 
| 13 | 10 django/contrib/admin/models.py | 1 | 21| 123 | 4000 | 32881 | 
| 14 | 11 django/views/i18n.py | 88 | 191| 702 | 4702 | 35384 | 
| 15 | 11 django/template/defaultfilters.py | 1 | 90| 559 | 5261 | 35384 | 
| 16 | 12 django/template/backends/jinja2.py | 1 | 52| 344 | 5605 | 36200 | 
| 17 | 13 django/http/response.py | 678 | 710| 271 | 5876 | 41342 | 
| 18 | 14 django/contrib/admin/utils.py | 513 | 579| 479 | 6355 | 45548 | 
| 19 | 14 docs/_ext/djangodocs.py | 26 | 71| 398 | 6753 | 45548 | 
| 20 | 14 django/template/defaultfilters.py | 348 | 362| 111 | 6864 | 45548 | 
| 21 | 15 django/views/defaults.py | 1 | 26| 151 | 7015 | 46531 | 
| 22 | 16 django/template/engine.py | 1 | 63| 397 | 7412 | 48029 | 
| 23 | **16 django/utils/html.py** | 80 | 103| 200 | 7612 | 48029 | 
| 24 | **16 django/utils/html.py** | 403 | 422| 168 | 7780 | 48029 | 
| 25 | 16 django/forms/fields.py | 1297 | 1334| 199 | 7979 | 48029 | 
| 26 | 17 django/contrib/gis/serializers/geojson.py | 47 | 82| 308 | 8287 | 48663 | 
| 27 | 18 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 8706 | 49082 | 
| 28 | 18 django/core/serializers/json.py | 1 | 59| 368 | 9074 | 49082 | 
| 29 | 19 django/db/migrations/serializer.py | 1 | 78| 408 | 9482 | 51769 | 
| 30 | 20 django/contrib/admin/views/autocomplete.py | 44 | 64| 190 | 9672 | 52610 | 
| 31 | 21 django/forms/renderers.py | 32 | 103| 443 | 10115 | 53259 | 
| 32 | 22 django/core/serializers/jsonl.py | 1 | 39| 258 | 10373 | 53642 | 
| 33 | 22 django/template/defaultfilters.py | 365 | 452| 504 | 10877 | 53642 | 
| 34 | 23 django/utils/feedgenerator.py | 101 | 149| 331 | 11208 | 57023 | 
| 35 | 24 django/template/__init__.py | 1 | 76| 394 | 11602 | 57417 | 
| 36 | 25 django/contrib/admin/helpers.py | 1 | 39| 227 | 11829 | 61083 | 
| 37 | 26 django/contrib/postgres/forms/hstore.py | 1 | 60| 340 | 12169 | 61423 | 
| 38 | 27 django/http/request.py | 265 | 317| 351 | 12520 | 66722 | 
| 39 | 28 django/utils/http.py | 50 | 88| 301 | 12821 | 70361 | 
| 40 | 28 django/views/i18n.py | 300 | 323| 167 | 12988 | 70361 | 
| 41 | 29 django/shortcuts.py | 1 | 25| 161 | 13149 | 71485 | 
| 42 | 30 django/contrib/syndication/views.py | 108 | 133| 180 | 13329 | 73343 | 
| 43 | 31 django/conf/__init__.py | 137 | 155| 146 | 13475 | 75745 | 
| 44 | 32 django/template/backends/dummy.py | 1 | 53| 327 | 13802 | 76072 | 
| 45 | 33 django/utils/encoding.py | 56 | 75| 156 | 13958 | 78297 | 
| 46 | 34 django/contrib/admin/options.py | 1 | 114| 776 | 14734 | 97518 | 
| 47 | 34 django/db/migrations/serializer.py | 174 | 193| 156 | 14890 | 97518 | 
| 48 | 35 docs/conf.py | 129 | 232| 931 | 15821 | 101030 | 
| 49 | 36 django/contrib/auth/decorators.py | 1 | 40| 315 | 16136 | 101622 | 
| 50 | 37 django/core/handlers/asgi.py | 27 | 133| 888 | 17024 | 104006 | 
| 51 | 38 django/forms/widgets.py | 61 | 108| 311 | 17335 | 112266 | 
| 52 | 38 django/db/models/fields/json.py | 46 | 68| 154 | 17489 | 112266 | 
| 53 | 38 django/http/request.py | 586 | 616| 208 | 17697 | 112266 | 
| 54 | 38 django/db/migrations/serializer.py | 111 | 121| 114 | 17811 | 112266 | 
| 55 | 38 django/db/models/fields/json.py | 417 | 443| 220 | 18031 | 112266 | 
| 56 | 39 django/contrib/admindocs/views.py | 394 | 427| 211 | 18242 | 115748 | 
| 57 | 40 django/template/utils.py | 1 | 65| 407 | 18649 | 116464 | 
| 58 | 40 django/forms/widgets.py | 338 | 374| 204 | 18853 | 116464 | 
| 59 | 41 django/core/signing.py | 98 | 128| 218 | 19071 | 118651 | 
| 60 | 42 django/db/models/sql/query.py | 2281 | 2312| 266 | 19337 | 141816 | 
| 61 | 43 django/template/base.py | 1 | 91| 754 | 20091 | 150089 | 
| 62 | 44 django/db/models/options.py | 169 | 232| 596 | 20687 | 157591 | 
| 63 | 45 django/contrib/staticfiles/storage.py | 175 | 238| 471 | 21158 | 161483 | 
| 64 | 45 django/template/engine.py | 112 | 179| 459 | 21617 | 161483 | 
| 65 | 46 django/core/management/__init__.py | 201 | 248| 350 | 21967 | 165078 | 
| 66 | 47 django/core/management/base.py | 105 | 131| 173 | 22140 | 169864 | 
| 67 | 48 django/forms/utils.py | 48 | 55| 113 | 22253 | 171583 | 
| 68 | 48 django/template/defaultfilters.py | 280 | 345| 425 | 22678 | 171583 | 
| 69 | 49 django/utils/safestring.py | 47 | 73| 166 | 22844 | 171996 | 
| 70 | 49 django/core/serializers/__init__.py | 91 | 146| 369 | 23213 | 171996 | 
| 71 | 49 django/db/models/fields/json.py | 262 | 280| 170 | 23383 | 171996 | 
| 72 | 50 django/contrib/admin/widgets.py | 555 | 588| 201 | 23584 | 176180 | 
| 73 | 50 docs/_ext/djangodocs.py | 383 | 402| 204 | 23788 | 176180 | 
| 74 | 50 django/contrib/admin/models.py | 109 | 171| 417 | 24205 | 176180 | 
| 75 | 51 django/db/migrations/questioner.py | 249 | 267| 177 | 24382 | 178876 | 
| 76 | 51 django/contrib/admin/widgets.py | 469 | 488| 141 | 24523 | 178876 | 
| 77 | 51 docs/_ext/djangodocs.py | 243 | 285| 435 | 24958 | 178876 | 
| 78 | 52 django/conf/global_settings.py | 157 | 272| 859 | 25817 | 184732 | 
| 79 | 52 django/db/models/fields/json.py | 283 | 307| 185 | 26002 | 184732 | 
| 80 | 53 django/utils/functional.py | 215 | 271| 323 | 26325 | 188015 | 
| 81 | 53 docs/_ext/djangodocs.py | 1 | 23| 178 | 26503 | 188015 | 
| 82 | 53 django/forms/widgets.py | 508 | 550| 279 | 26782 | 188015 | 
| 83 | 53 django/contrib/admin/widgets.py | 51 | 72| 168 | 26950 | 188015 | 
| 84 | 53 django/contrib/messages/storage/cookie.py | 29 | 59| 210 | 27160 | 188015 | 
| 85 | 53 django/conf/__init__.py | 272 | 280| 121 | 27281 | 188015 | 
| 86 | 54 django/db/backends/mysql/schema.py | 104 | 118| 144 | 27425 | 189638 | 
| 87 | 54 django/conf/__init__.py | 157 | 177| 184 | 27609 | 189638 | 
| 88 | 54 django/contrib/admin/options.py | 283 | 331| 397 | 28006 | 189638 | 
| 89 | 54 django/conf/__init__.py | 56 | 79| 201 | 28207 | 189638 | 
| 90 | 55 django/contrib/admin/decorators.py | 1 | 31| 181 | 28388 | 190292 | 
| 91 | 55 django/db/models/fields/json.py | 310 | 341| 310 | 28698 | 190292 | 
| 92 | 55 django/contrib/messages/storage/cookie.py | 94 | 113| 145 | 28843 | 190292 | 
| 93 | 56 django/views/debug.py | 184 | 209| 181 | 29024 | 195038 | 
| 94 | 57 django/contrib/admindocs/utils.py | 1 | 32| 206 | 29230 | 197323 | 
| 95 | 58 django/template/context_processors.py | 58 | 90| 143 | 29373 | 197818 | 
| 96 | 58 django/core/serializers/__init__.py | 149 | 164| 136 | 29509 | 197818 | 
| 97 | 58 django/template/utils.py | 67 | 94| 197 | 29706 | 197818 | 
| 98 | 58 django/db/models/fields/json.py | 486 | 576| 515 | 30221 | 197818 | 
| 99 | 58 django/db/migrations/serializer.py | 314 | 353| 297 | 30518 | 197818 | 
| 100 | 59 django/contrib/postgres/fields/jsonb.py | 1 | 15| 0 | 30518 | 197907 | 
| 101 | 59 django/db/models/fields/json.py | 371 | 390| 160 | 30678 | 197907 | 


### Hint

```
Sounds good, and yes, we should document django.utils.html.json_script().
​PR I'll also add docs for json_script() soon
​PR
```

## Patch

```diff
diff --git a/django/utils/html.py b/django/utils/html.py
--- a/django/utils/html.py
+++ b/django/utils/html.py
@@ -59,7 +59,7 @@ def escapejs(value):
 }
 
 
-def json_script(value, element_id=None):
+def json_script(value, element_id=None, encoder=None):
     """
     Escape all the HTML/XML special characters with their unicode escapes, so
     value is safe to be output anywhere except for inside a tag attribute. Wrap
@@ -67,7 +67,9 @@ def json_script(value, element_id=None):
     """
     from django.core.serializers.json import DjangoJSONEncoder
 
-    json_str = json.dumps(value, cls=DjangoJSONEncoder).translate(_json_script_escapes)
+    json_str = json.dumps(value, cls=encoder or DjangoJSONEncoder).translate(
+        _json_script_escapes
+    )
     if element_id:
         template = '<script id="{}" type="application/json">{}</script>'
         args = (element_id, mark_safe(json_str))

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_html.py b/tests/utils_tests/test_html.py
--- a/tests/utils_tests/test_html.py
+++ b/tests/utils_tests/test_html.py
@@ -1,6 +1,7 @@
 import os
 from datetime import datetime
 
+from django.core.serializers.json import DjangoJSONEncoder
 from django.test import SimpleTestCase
 from django.utils.functional import lazystr
 from django.utils.html import (
@@ -211,6 +212,16 @@ def test_json_script(self):
             with self.subTest(arg=arg):
                 self.assertEqual(json_script(arg, "test_id"), expected)
 
+    def test_json_script_custom_encoder(self):
+        class CustomDjangoJSONEncoder(DjangoJSONEncoder):
+            def encode(self, o):
+                return '{"hello": "world"}'
+
+        self.assertHTMLEqual(
+            json_script({}, encoder=CustomDjangoJSONEncoder),
+            '<script type="application/json">{"hello": "world"}</script>',
+        )
+
     def test_json_script_without_id(self):
         self.assertHTMLEqual(
             json_script({"key": "value"}),

```


## Code snippets

### 1 - django/utils/html.py:

Start line: 62, End line: 77

```python
def json_script(value, element_id=None):
    """
    Escape all the HTML/XML special characters with their unicode escapes, so
    value is safe to be output anywhere except for inside a tag attribute. Wrap
    the escaped JSON in a script tag.
    """
    from django.core.serializers.json import DjangoJSONEncoder

    json_str = json.dumps(value, cls=DjangoJSONEncoder).translate(_json_script_escapes)
    if element_id:
        template = '<script id="{}" type="application/json">{}</script>'
        args = (element_id, mark_safe(json_str))
    else:
        template = '<script type="application/json">{}</script>'
        args = (mark_safe(json_str),)
    return format_html(template, *args)
```
### 2 - django/forms/fields.py:

Start line: 1337, End line: 1391

```python
class JSONField(CharField):
    default_error_messages = {
        "invalid": _("Enter a valid JSON."),
    }
    widget = Textarea

    def __init__(self, encoder=None, decoder=None, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(**kwargs)

    def to_python(self, value):
        if self.disabled:
            return value
        if value in self.empty_values:
            return None
        elif isinstance(value, (list, dict, int, float, JSONString)):
            return value
        try:
            converted = json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            raise ValidationError(
                self.error_messages["invalid"],
                code="invalid",
                params={"value": value},
            )
        if isinstance(converted, str):
            return JSONString(converted)
        else:
            return converted

    def bound_data(self, data, initial):
        if self.disabled:
            return initial
        if data is None:
            return None
        try:
            return json.loads(data, cls=self.decoder)
        except json.JSONDecodeError:
            return InvalidJSONInput(data)

    def prepare_value(self, value):
        if isinstance(value, InvalidJSONInput):
            return value
        return json.dumps(value, ensure_ascii=False, cls=self.encoder)

    def has_changed(self, initial, data):
        if super().has_changed(initial, data):
            return True
        # For purposes of seeing whether something has changed, True isn't the
        # same as 1 and the order of keys doesn't matter.
        return json.dumps(initial, sort_keys=True, cls=self.encoder) != json.dumps(
            self.to_python(data), sort_keys=True, cls=self.encoder
        )
```
### 3 - docs/_ext/djangodocs.py:

Start line: 178, End line: 213

```python
def parse_django_admin_node(env, sig, signode):
    command = sig.split(" ")[0]
    env.ref_context["std:program"] = command
    title = "django-admin %s" % sig
    signode += addnodes.desc_name(title, title)
    return command


class DjangoStandaloneHTMLBuilder(StandaloneHTMLBuilder):
    """
    Subclass to add some extra things we need.
    """

    name = "djangohtml"

    def finish(self):
        super().finish()
        logger.info(bold("writing templatebuiltins.js..."))
        xrefs = self.env.domaindata["std"]["objects"]
        templatebuiltins = {
            "ttags": [
                n
                for ((t, n), (k, a)) in xrefs.items()
                if t == "templatetag" and k == "ref/templates/builtins"
            ],
            "tfilters": [
                n
                for ((t, n), (k, a)) in xrefs.items()
                if t == "templatefilter" and k == "ref/templates/builtins"
            ],
        }
        outfilename = os.path.join(self.outdir, "templatebuiltins.js")
        with open(outfilename, "w") as fp:
            fp.write("var django_template_builtins = ")
            json.dump(templatebuiltins, fp)
            fp.write(";\n")
```
### 4 - django/utils/html.py:

Start line: 1, End line: 59

```python
"""HTML utilities suitable for global use."""

import html
import json
import re
from html.parser import HTMLParser
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from django.utils.encoding import punycode
from django.utils.functional import Promise, keep_lazy, keep_lazy_text
from django.utils.http import RFC3986_GENDELIMS, RFC3986_SUBDELIMS
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import normalize_newlines


@keep_lazy(SafeString)
def escape(text):
    """
    Return the given text with ampersands, quotes and angle brackets encoded
    for use in HTML.

    Always escape input, even if it's already escaped and marked as such.
    This may result in double-escaping. If this is a concern, use
    conditional_escape() instead.
    """
    return SafeString(html.escape(str(text)))


_js_escapes = {
    ord("\\"): "\\u005C",
    ord("'"): "\\u0027",
    ord('"'): "\\u0022",
    ord(">"): "\\u003E",
    ord("<"): "\\u003C",
    ord("&"): "\\u0026",
    ord("="): "\\u003D",
    ord("-"): "\\u002D",
    ord(";"): "\\u003B",
    ord("`"): "\\u0060",
    ord("\u2028"): "\\u2028",
    ord("\u2029"): "\\u2029",
}

# Escape every ASCII character with a value less than 32.
_js_escapes.update((ord("%c" % z), "\\u%04X" % z) for z in range(32))


@keep_lazy(SafeString)
def escapejs(value):
    """Hex encode characters for use in JavaScript strings."""
    return mark_safe(str(value).translate(_js_escapes))


_json_script_escapes = {
    ord(">"): "\\u003E",
    ord("<"): "\\u003C",
    ord("&"): "\\u0026",
}
```
### 5 - django/core/serializers/json.py:

Start line: 62, End line: 107

```python
def Deserializer(stream_or_string, **options):
    """Deserialize a stream or string of JSON data."""
    if not isinstance(stream_or_string, (bytes, str)):
        stream_or_string = stream_or_string.read()
    if isinstance(stream_or_string, bytes):
        stream_or_string = stream_or_string.decode()
    try:
        objects = json.loads(stream_or_string)
        yield from PythonDeserializer(objects, **options)
    except (GeneratorExit, DeserializationError):
        raise
    except Exception as exc:
        raise DeserializationError() from exc


class DjangoJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that knows how to encode date/time, decimal types, and
    UUIDs.
    """

    def default(self, o):
        # See "Date Time String Format" in the ECMA-262 specification.
        if isinstance(o, datetime.datetime):
            r = o.isoformat()
            if o.microsecond:
                r = r[:23] + r[26:]
            if r.endswith("+00:00"):
                r = r[:-6] + "Z"
            return r
        elif isinstance(o, datetime.date):
            return o.isoformat()
        elif isinstance(o, datetime.time):
            if is_aware(o):
                raise ValueError("JSON can't represent timezone-aware times.")
            r = o.isoformat()
            if o.microsecond:
                r = r[:12]
            return r
        elif isinstance(o, datetime.timedelta):
            return duration_iso_string(o)
        elif isinstance(o, (decimal.Decimal, uuid.UUID, Promise)):
            return str(o)
        else:
            return super().default(o)
```
### 6 - django/db/models/fields/json.py:

Start line: 1, End line: 44

```python
import json

from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import lookups
from django.db.models.lookups import PostgresOperatorLookup, Transform
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import CheckFieldDefaultMixin

__all__ = ["JSONField"]


class JSONField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    description = _("A JSON object")
    default_error_messages = {
        "invalid": _("Value must be valid JSON."),
    }
    _default_hint = ("dict", "{}")

    def __init__(
        self,
        verbose_name=None,
        name=None,
        encoder=None,
        decoder=None,
        **kwargs,
    ):
        if encoder and not callable(encoder):
            raise ValueError("The encoder parameter must be a callable object.")
        if decoder and not callable(decoder):
            raise ValueError("The decoder parameter must be a callable object.")
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        databases = kwargs.get("databases") or []
        errors.extend(self._check_supported(databases))
        return errors
```
### 7 - django/contrib/admin/templatetags/admin_modify.py:

Start line: 1, End line: 58

```python
import json

from django import template
from django.template.context import Context

from .base import InclusionAdminNode

register = template.Library()


def prepopulated_fields_js(context):
    """
    Create a list of prepopulated_fields that should render JavaScript for
    the prepopulated fields for both the admin form and inlines.
    """
    prepopulated_fields = []
    if "adminform" in context:
        prepopulated_fields.extend(context["adminform"].prepopulated_fields)
    if "inline_admin_formsets" in context:
        for inline_admin_formset in context["inline_admin_formsets"]:
            for inline_admin_form in inline_admin_formset:
                if inline_admin_form.original is None:
                    prepopulated_fields.extend(inline_admin_form.prepopulated_fields)

    prepopulated_fields_json = []
    for field in prepopulated_fields:
        prepopulated_fields_json.append(
            {
                "id": "#%s" % field["field"].auto_id,
                "name": field["field"].name,
                "dependency_ids": [
                    "#%s" % dependency.auto_id for dependency in field["dependencies"]
                ],
                "dependency_list": [
                    dependency.name for dependency in field["dependencies"]
                ],
                "maxLength": field["field"].field.max_length or 50,
                "allowUnicode": getattr(field["field"].field, "allow_unicode", False),
            }
        )

    context.update(
        {
            "prepopulated_fields": prepopulated_fields,
            "prepopulated_fields_json": json.dumps(prepopulated_fields_json),
        }
    )
    return context


@register.tag(name="prepopulated_fields_js")
def prepopulated_fields_js_tag(parser, token):
    return InclusionAdminNode(
        parser,
        token,
        func=prepopulated_fields_js,
        template_name="prepopulated_fields_js.html",
    )
```
### 8 - django/db/models/fields/json.py:

Start line: 70, End line: 126

```python
class JSONField(CheckFieldDefaultMixin, Field):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.encoder is not None:
            kwargs["encoder"] = self.encoder
        if self.decoder is not None:
            kwargs["decoder"] = self.decoder
        return name, path, args, kwargs

    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        # Some backends (SQLite at least) extract non-string values in their
        # SQL datatypes.
        if isinstance(expression, KeyTransform) and not isinstance(value, str):
            return value
        try:
            return json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            return value

    def get_internal_type(self):
        return "JSONField"

    def get_prep_value(self, value):
        if value is None:
            return value
        return json.dumps(value, cls=self.encoder)

    def get_transform(self, name):
        transform = super().get_transform(name)
        if transform:
            return transform
        return KeyTransformFactory(name)

    def validate(self, value, model_instance):
        super().validate(value, model_instance)
        try:
            json.dumps(value, cls=self.encoder)
        except TypeError:
            raise exceptions.ValidationError(
                self.error_messages["invalid"],
                code="invalid",
                params={"value": value},
            )

    def value_to_string(self, obj):
        return self.value_from_object(obj)

    def formfield(self, **kwargs):
        return super().formfield(
            **{
                "form_class": forms.JSONField,
                "encoder": self.encoder,
                "decoder": self.decoder,
                **kwargs,
            }
        )
```
### 9 - django/contrib/messages/storage/cookie.py:

Start line: 1, End line: 26

```python
import binascii
import json

from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe


class MessageEncoder(json.JSONEncoder):
    """
    Compactly serialize instances of the ``Message`` class as JSON.
    """

    message_key = "__json_message"

    def default(self, obj):
        if isinstance(obj, Message):
            # Using 0/1 here instead of False/True to produce more compact json
            is_safedata = 1 if isinstance(obj.message, SafeData) else 0
            message = [self.message_key, is_safedata, obj.level, obj.message]
            if obj.extra_tags is not None:
                message.append(obj.extra_tags)
            return message
        return super().default(obj)
```
### 10 - django/template/defaultfilters.py:

Start line: 697, End line: 729

```python
@register.filter(is_safe=True, needs_autoescape=True)
def unordered_list(value, autoescape=True):
    # ... other code

    def list_formatter(item_list, tabs=1):
        indent = "\t" * tabs
        output = []
        for item, children in walk_items(item_list):
            sublist = ""
            if children:
                sublist = "\n%s<ul>\n%s\n%s</ul>\n%s" % (
                    indent,
                    list_formatter(children, tabs + 1),
                    indent,
                    indent,
                )
            output.append("%s<li>%s%s</li>" % (indent, escaper(item), sublist))
        return "\n".join(output)

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
            return ""
```
### 23 - django/utils/html.py:

Start line: 80, End line: 103

```python
def conditional_escape(text):
    """
    Similar to escape(), except that it doesn't operate on pre-escaped strings.

    This function relies on the __html__ convention used both by Django's
    SafeData class and by third-party libraries like markupsafe.
    """
    if isinstance(text, Promise):
        text = str(text)
    if hasattr(text, "__html__"):
        return text.__html__()
    else:
        return escape(text)


def format_html(format_string, *args, **kwargs):
    """
    Similar to str.format, but pass all arguments through conditional_escape(),
    and call mark_safe() on the result. This function should be used instead
    of str.format or % interpolation to build up small HTML fragments.
    """
    args_safe = map(conditional_escape, args)
    kwargs_safe = {k: conditional_escape(v) for (k, v) in kwargs.items()}
    return mark_safe(format_string.format(*args_safe, **kwargs_safe))
```
### 24 - django/utils/html.py:

Start line: 403, End line: 422

```python
def html_safe(klass):
    """
    A decorator that defines the __html__ method. This helps non-Django
    templates to detect classes whose __str__ methods return SafeString.
    """
    if "__html__" in klass.__dict__:
        raise ValueError(
            "can't apply @html_safe to %s because it defines "
            "__html__()." % klass.__name__
        )
    if "__str__" not in klass.__dict__:
        raise ValueError(
            "can't apply @html_safe to %s because it doesn't "
            "define __str__()." % klass.__name__
        )
    klass_str = klass.__str__
    klass.__str__ = lambda self: mark_safe(klass_str(self))
    klass.__html__ = lambda self: str(self)
    return klass
```
