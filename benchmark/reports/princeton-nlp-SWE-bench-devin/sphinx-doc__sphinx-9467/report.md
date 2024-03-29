# sphinx-doc__sphinx-9467

| **sphinx-doc/sphinx** | `9a2c3c4a1559e37e95fdee88c128bb116642c897` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3448 |
| **Any found context length** | 3448 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -714,7 +714,10 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_event('linkcheck-process-uri')
 
     app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=800)
-    app.connect('linkcheck-process-uri', rewrite_github_anchor)
+
+    # FIXME: Disable URL rewrite handler for github.com temporarily.
+    # ref: https://github.com/sphinx-doc/sphinx/issues/9435
+    # app.connect('linkcheck-process-uri', rewrite_github_anchor)
 
     return {
         'version': 'builtin',

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/linkcheck.py | 717 | 717 | 11 | 1 | 3448


## Problem Statement

```
github linkcheck anchor change in 4.1.0 break some usage
### Describe the bug

Given a link like:

\`\`\`rst
.. _`OpenSSL's test vectors`: https://github.com/openssl/openssl/blob/97cf1f6c2854a3a955fd7dd3a1f113deba00c9ef/crypto/evp/evptests.txt#L232 
\`\`\`

in a github doc, with release 4.1.0 this will fail with linkcheck, while previously it worked.

### How to Reproduce

\`\`\`
$ git clone https://github.com/pyca/cryptography
$ cd cryptography
$ tox -e docs-linkcheck
\`\`\`


### Expected behavior

It passes.

### Your project

https://github.com/pyca/cryptography

### Screenshots

_No response_

### OS

Linux

### Python version

3.9.5

### Sphinx version

4.1.0

### Sphinx extensions

_No response_

### Extra tools

_No response_

### Additional context

The relevant change happened in https://github.com/sphinx-doc/sphinx/commit/92335bd6e67dec9d8cadfdfb6d441a440e8dc87e

Failing test logs: https://github.com/pyca/cryptography/runs/3046691393

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/builders/linkcheck.py** | 417 | 511| 763 | 763 | 5917 | 
| 2 | **1 sphinx/builders/linkcheck.py** | 513 | 555| 413 | 1176 | 5917 | 
| 3 | **1 sphinx/builders/linkcheck.py** | 341 | 361| 175 | 1351 | 5917 | 
| 4 | **1 sphinx/builders/linkcheck.py** | 114 | 132| 170 | 1521 | 5917 | 
| 5 | **1 sphinx/builders/linkcheck.py** | 307 | 339| 278 | 1799 | 5917 | 
| 6 | **1 sphinx/builders/linkcheck.py** | 286 | 304| 203 | 2002 | 5917 | 
| 7 | **1 sphinx/builders/linkcheck.py** | 557 | 592| 331 | 2333 | 5917 | 
| 8 | **1 sphinx/builders/linkcheck.py** | 231 | 284| 577 | 2910 | 5917 | 
| 9 | **1 sphinx/builders/linkcheck.py** | 669 | 681| 117 | 3027 | 5917 | 
| 10 | **1 sphinx/builders/linkcheck.py** | 397 | 415| 142 | 3169 | 5917 | 
| **-> 11 <-** | **1 sphinx/builders/linkcheck.py** | 697 | 724| 279 | 3448 | 5917 | 
| 12 | **1 sphinx/builders/linkcheck.py** | 134 | 229| 727 | 4175 | 5917 | 
| 13 | 2 sphinx/ext/linkcode.py | 11 | 80| 467 | 4642 | 6440 | 
| 14 | **2 sphinx/builders/linkcheck.py** | 633 | 666| 275 | 4917 | 6440 | 
| 15 | **2 sphinx/builders/linkcheck.py** | 96 | 111| 156 | 5073 | 6440 | 
| 16 | **2 sphinx/builders/linkcheck.py** | 594 | 630| 308 | 5381 | 6440 | 
| 17 | **2 sphinx/builders/linkcheck.py** | 11 | 93| 662 | 6043 | 6440 | 
| 18 | **2 sphinx/builders/linkcheck.py** | 364 | 395| 286 | 6329 | 6440 | 
| 19 | 3 sphinx/__init__.py | 14 | 60| 476 | 6805 | 6990 | 
| 20 | 4 utils/checks.py | 33 | 109| 545 | 7350 | 7896 | 
| 21 | **4 sphinx/builders/linkcheck.py** | 684 | 694| 126 | 7476 | 7896 | 
| 22 | 4 utils/checks.py | 11 | 30| 175 | 7651 | 7896 | 
| 23 | 5 sphinx/ext/doctest.py | 12 | 43| 227 | 7878 | 12914 | 
| 24 | 6 sphinx/util/requests.py | 11 | 67| 415 | 8293 | 13862 | 
| 25 | 7 sphinx/highlighting.py | 11 | 68| 620 | 8913 | 15408 | 
| 26 | 8 sphinx/ext/graphviz.py | 12 | 44| 243 | 9156 | 19144 | 
| 27 | 9 doc/conf.py | 1 | 82| 731 | 9887 | 20608 | 
| 28 | 10 sphinx/errors.py | 77 | 134| 297 | 10184 | 21404 | 
| 29 | 11 sphinx/testing/util.py | 10 | 45| 270 | 10454 | 23136 | 
| 30 | 11 sphinx/ext/doctest.py | 246 | 273| 277 | 10731 | 23136 | 
| 31 | 12 sphinx/ext/extlinks.py | 28 | 89| 615 | 11346 | 24005 | 
| 32 | 13 sphinx/io.py | 10 | 39| 234 | 11580 | 25410 | 
| 33 | 14 sphinx/builders/changes.py | 125 | 168| 438 | 12018 | 26923 | 
| 34 | 15 sphinx/environment/__init__.py | 619 | 639| 178 | 12196 | 32420 | 
| 35 | 16 sphinx/util/pycompat.py | 11 | 46| 328 | 12524 | 32913 | 
| 36 | 17 sphinx/domains/changeset.py | 49 | 107| 516 | 13040 | 34167 | 
| 37 | 18 utils/bump_version.py | 67 | 102| 201 | 13241 | 35529 | 
| 38 | 19 setup.py | 176 | 252| 638 | 13879 | 37289 | 
| 39 | 20 sphinx/domains/citation.py | 11 | 29| 125 | 14004 | 38570 | 
| 40 | 21 sphinx/cmd/build.py | 11 | 30| 132 | 14136 | 41231 | 
| 41 | 22 sphinx/setup_command.py | 142 | 190| 415 | 14551 | 42776 | 
| 42 | 23 sphinx/domains/python.py | 287 | 321| 382 | 14933 | 54862 | 
| 43 | 23 doc/conf.py | 83 | 138| 476 | 15409 | 54862 | 
| 44 | 23 sphinx/builders/changes.py | 11 | 26| 124 | 15533 | 54862 | 
| 45 | 23 setup.py | 1 | 78| 478 | 16011 | 54862 | 
| 46 | 24 sphinx/directives/patches.py | 9 | 34| 192 | 16203 | 56706 | 
| 47 | 24 sphinx/ext/doctest.py | 315 | 330| 130 | 16333 | 56706 | 
| 48 | 24 sphinx/ext/doctest.py | 403 | 479| 749 | 17082 | 56706 | 
| 49 | 25 sphinx/ext/viewcode.py | 11 | 45| 228 | 17310 | 59801 | 
| 50 | 26 sphinx/transforms/post_transforms/code.py | 114 | 143| 208 | 17518 | 60818 | 
| 51 | 26 sphinx/domains/python.py | 996 | 1016| 248 | 17766 | 60818 | 
| 52 | 27 sphinx/testing/__init__.py | 1 | 15| 0 | 17766 | 60900 | 
| 53 | 27 sphinx/environment/__init__.py | 11 | 82| 508 | 18274 | 60900 | 
| 54 | 27 sphinx/setup_command.py | 91 | 118| 229 | 18503 | 60900 | 
| 55 | 27 sphinx/domains/python.py | 11 | 80| 518 | 19021 | 60900 | 
| 56 | 27 sphinx/ext/doctest.py | 481 | 504| 264 | 19285 | 60900 | 
| 57 | 27 sphinx/ext/viewcode.py | 77 | 109| 277 | 19562 | 60900 | 
| 58 | 27 sphinx/domains/changeset.py | 11 | 46| 209 | 19771 | 60900 | 
| 59 | 27 sphinx/domains/python.py | 975 | 993| 140 | 19911 | 60900 | 
| 60 | 28 sphinx/registry.py | 11 | 51| 314 | 20225 | 65539 | 
| 61 | 28 sphinx/ext/doctest.py | 232 | 243| 121 | 20346 | 65539 | 
| 62 | 29 sphinx/builders/html/__init__.py | 11 | 62| 432 | 20778 | 77875 | 
| 63 | 30 sphinx/cmd/make_mode.py | 17 | 54| 532 | 21310 | 79577 | 
| 64 | 30 sphinx/highlighting.py | 71 | 180| 873 | 22183 | 79577 | 
| 65 | 31 sphinx/builders/latex/__init__.py | 11 | 42| 331 | 22514 | 85289 | 
| 66 | 32 sphinx/util/osutil.py | 11 | 45| 214 | 22728 | 86923 | 
| 67 | 33 sphinx/testing/fixtures.py | 205 | 229| 167 | 22895 | 88708 | 
| 68 | 33 sphinx/ext/doctest.py | 46 | 61| 138 | 23033 | 88708 | 
| 69 | 33 sphinx/builders/changes.py | 51 | 124| 805 | 23838 | 88708 | 
| 70 | 33 sphinx/util/requests.py | 70 | 89| 150 | 23988 | 88708 | 
| 71 | 34 sphinx/util/inspect.py | 11 | 47| 290 | 24278 | 95278 | 
| 72 | 35 sphinx/transforms/post_transforms/__init__.py | 171 | 218| 489 | 24767 | 97730 | 
| 73 | 36 sphinx/cmd/quickstart.py | 11 | 119| 756 | 25523 | 103299 | 
| 74 | 37 sphinx/util/__init__.py | 11 | 64| 446 | 25969 | 108180 | 
| 75 | 38 sphinx/domains/std.py | 1113 | 1138| 209 | 26178 | 118477 | 
| 76 | 39 sphinx/application.py | 13 | 58| 361 | 26539 | 130091 | 
| 77 | 39 sphinx/ext/doctest.py | 506 | 553| 509 | 27048 | 130091 | 
| 78 | 40 sphinx/domains/rst.py | 11 | 32| 170 | 27218 | 132570 | 
| 79 | 40 sphinx/ext/doctest.py | 353 | 372| 211 | 27429 | 132570 | 
| 80 | 41 sphinx/util/smartypants.py | 380 | 392| 137 | 27566 | 136715 | 
| 81 | 41 sphinx/ext/viewcode.py | 111 | 138| 286 | 27852 | 136715 | 
| 82 | 42 sphinx/util/cfamily.py | 63 | 87| 230 | 28082 | 140172 | 
| 83 | 43 sphinx/transforms/__init__.py | 11 | 44| 231 | 28313 | 143343 | 
| 84 | 44 sphinx/cmd/__init__.py | 1 | 10| 0 | 28313 | 143392 | 
| 85 | 44 sphinx/builders/html/__init__.py | 1253 | 1277| 198 | 28511 | 143392 | 
| 86 | 44 sphinx/setup_command.py | 14 | 89| 415 | 28926 | 143392 | 
| 87 | 45 sphinx/domains/javascript.py | 11 | 33| 199 | 29125 | 147456 | 
| 88 | 46 sphinx/builders/singlehtml.py | 11 | 25| 112 | 29237 | 149252 | 
| 89 | 47 sphinx/config.py | 462 | 481| 223 | 29460 | 153708 | 
| 90 | 48 sphinx/directives/other.py | 9 | 38| 229 | 29689 | 156835 | 
| 91 | 48 sphinx/util/cfamily.py | 11 | 62| 749 | 30438 | 156835 | 
| 92 | 48 sphinx/ext/doctest.py | 153 | 197| 292 | 30730 | 156835 | 
| 93 | 48 sphinx/builders/singlehtml.py | 54 | 66| 141 | 30871 | 156835 | 
| 94 | 49 sphinx/extension.py | 42 | 69| 225 | 31096 | 157341 | 
| 95 | 50 sphinx/ext/mathjax.py | 13 | 38| 222 | 31318 | 158538 | 
| 96 | 50 sphinx/ext/viewcode.py | 160 | 195| 287 | 31605 | 158538 | 
| 97 | 50 utils/bump_version.py | 119 | 146| 255 | 31860 | 158538 | 
| 98 | 51 sphinx/builders/_epub_base.py | 298 | 316| 205 | 32065 | 164847 | 
| 99 | 51 utils/bump_version.py | 104 | 117| 123 | 32188 | 164847 | 
| 100 | 52 sphinx/directives/code.py | 9 | 30| 148 | 32336 | 168698 | 
| 101 | 52 sphinx/transforms/post_transforms/__init__.py | 11 | 28| 132 | 32468 | 168698 | 
| 102 | 53 sphinx/builders/__init__.py | 11 | 48| 302 | 32770 | 174048 | 
| 103 | 54 sphinx/ext/inheritance_diagram.py | 38 | 67| 243 | 33013 | 177914 | 
| 104 | 54 sphinx/util/__init__.py | 145 | 172| 228 | 33241 | 177914 | 
| 105 | 54 utils/bump_version.py | 149 | 180| 224 | 33465 | 177914 | 
| 106 | 55 sphinx/builders/latex/constants.py | 74 | 124| 537 | 34002 | 180161 | 
| 107 | 55 sphinx/builders/html/__init__.py | 1215 | 1237| 242 | 34244 | 180161 | 
| 108 | 56 sphinx/ext/intersphinx.py | 306 | 359| 590 | 34834 | 184007 | 
| 109 | 57 sphinx/util/docutils.py | 146 | 170| 194 | 35028 | 188149 | 
| 110 | 58 sphinx/transforms/references.py | 11 | 54| 266 | 35294 | 188467 | 
| 111 | 59 sphinx/__main__.py | 1 | 16| 0 | 35294 | 188538 | 
| 112 | 60 sphinx/ext/todo.py | 14 | 42| 193 | 35487 | 190378 | 
| 113 | 60 sphinx/ext/todo.py | 197 | 222| 211 | 35698 | 190378 | 
| 114 | 60 sphinx/ext/doctest.py | 332 | 351| 164 | 35862 | 190378 | 
| 115 | 60 sphinx/domains/javascript.py | 300 | 317| 196 | 36058 | 190378 | 
| 116 | 60 sphinx/directives/patches.py | 63 | 77| 143 | 36201 | 190378 | 
| 117 | 61 sphinx/ext/__init__.py | 1 | 10| 0 | 36201 | 190428 | 
| 118 | 62 sphinx/ext/coverage.py | 81 | 112| 261 | 36462 | 193044 | 
| 119 | 62 sphinx/domains/std.py | 11 | 46| 311 | 36773 | 193044 | 
| 120 | 63 sphinx/ext/imgmath.py | 11 | 82| 506 | 37279 | 196229 | 
| 121 | 63 sphinx/builders/_epub_base.py | 339 | 364| 301 | 37580 | 196229 | 
| 122 | 64 sphinx/ext/githubpages.py | 11 | 37| 227 | 37807 | 196516 | 


### Hint

```
A simple fix here would be for the linkcheck logic to simply exclude links of the form `L\d+`.

But of course really the current logic is broken for _any_ link that's generated by github itself and thus doesn't have this prefix.
> But of course really the current logic is broken for any link that's generated by github itself and thus doesn't have this prefix.

Agreed. Here's a subset of native GitHub anchor prefixes that are currently broken in Sphinx 4.1.0: `#L`, `#issuecomment-`, `#pullrequestreview-`, `#commits-pushed-`, `#event-`, `#ref-commit-`, `#ref-pullrequest`. My feeling is that it's going to be hard to maintain an exhaustive list of valid prefixes especially since there does not seem to be any reference authoritative GitHub page listing all of these.

Yes, it's a pretty unfortunate circumstance. It's not at all surprising github doesn't make a list of these, I haven't seen any website with a list of their anchor refs!

ATM I'm interested in whatever solution will get this back to a working state as quickly as possible.
Completely untested but my understanding of [the original PR](https://github.com/sphinx-doc/sphinx/pull/9260) is that it attempts to fix the linkchecking of anchors in the scenario of a rendered rST/Markdown file. As an immediate patch, would it help to add an additional check to the `rewrite_github_anchor` method of the like of the `parsed.path.endswith('.rst') or parsed.path.endswith('.md')`? Happy to open a PR to that effect.
OMG! I must admit our approach in #9016 is wrong. So we have to reconsider the way to check the anchors in reST/MD files. Until then, we have to fix the broken linkcheck-builder. I think it's better to disable `rewrite_github_anchor` handler for a while.
That'd be fine with me :-) 
In #9260 a new `linkcheck-process-uri` event was added, which I think it was a good idea. The problem here is that the current `rewrite_github_anchor` is insufficient - perhaps it should be moved to an extension that folks can install separately?
```

## Patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -714,7 +714,10 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_event('linkcheck-process-uri')
 
     app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=800)
-    app.connect('linkcheck-process-uri', rewrite_github_anchor)
+
+    # FIXME: Disable URL rewrite handler for github.com temporarily.
+    # ref: https://github.com/sphinx-doc/sphinx/issues/9435
+    # app.connect('linkcheck-process-uri', rewrite_github_anchor)
 
     return {
         'version': 'builtin',

```

## Test Patch

```diff
diff --git a/tests/roots/test-linkcheck/links.txt b/tests/roots/test-linkcheck/links.txt
--- a/tests/roots/test-linkcheck/links.txt
+++ b/tests/roots/test-linkcheck/links.txt
@@ -13,8 +13,7 @@ Some additional anchors to exercise ignore code
 * `Complete nonsense <https://localhost:7777/doesnotexist>`_
 * `Example valid local file <conf.py>`_
 * `Example invalid local file <path/to/notfound>`_
-* https://github.com/sphinx-doc/sphinx#documentation
-* https://github.com/sphinx-doc/sphinx#user-content-testing
+* https://github.com/sphinx-doc/sphinx/blob/4.x/sphinx/__init__.py#L2
 
 .. image:: https://www.google.com/image.png
 .. figure:: https://www.google.com/image2.png
diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -66,8 +66,8 @@ def test_defaults_json(app):
                  "info"]:
         assert attr in row
 
-    assert len(content.splitlines()) == 12
-    assert len(rows) == 12
+    assert len(content.splitlines()) == 11
+    assert len(rows) == 11
     # the output order of the rows is not stable
     # due to possible variance in network latency
     rowsby = {row["uri"]: row for row in rows}
@@ -88,7 +88,7 @@ def test_defaults_json(app):
     assert dnerow['uri'] == 'https://localhost:7777/doesnotexist'
     assert rowsby['https://www.google.com/image2.png'] == {
         'filename': 'links.txt',
-        'lineno': 20,
+        'lineno': 19,
         'status': 'broken',
         'code': 0,
         'uri': 'https://www.google.com/image2.png',
@@ -102,10 +102,6 @@ def test_defaults_json(app):
     # images should fail
     assert "Not Found for url: https://www.google.com/image.png" in \
         rowsby["https://www.google.com/image.png"]["info"]
-    # The anchor of the URI for github.com is automatically modified
-    assert 'https://github.com/sphinx-doc/sphinx#documentation' not in rowsby
-    assert 'https://github.com/sphinx-doc/sphinx#user-content-documentation' in rowsby
-    assert 'https://github.com/sphinx-doc/sphinx#user-content-testing' in rowsby
 
 
 @pytest.mark.sphinx(

```


## Code snippets

### 1 - sphinx/builders/linkcheck.py:

Start line: 417, End line: 511

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def run(self) -> None:
        # ... other code

        def check_uri() -> Tuple[str, str, int]:
            # split off anchor
            if '#' in uri:
                req_url, anchor = uri.split('#', 1)
                for rex in self.anchors_ignore:
                    if rex.match(anchor):
                        anchor = None
                        break
            else:
                req_url = uri
                anchor = None

            # handle non-ASCII URIs
            try:
                req_url.encode('ascii')
            except UnicodeError:
                req_url = encode_uri(req_url)

            # Get auth info, if any
            for pattern, auth_info in self.auth:
                if pattern.match(uri):
                    break
            else:
                auth_info = None

            # update request headers for the URL
            kwargs['headers'] = get_request_headers()

            try:
                if anchor and self.config.linkcheck_anchors:
                    # Read the whole document and see if #anchor exists
                    response = requests.get(req_url, stream=True, config=self.config,
                                            auth=auth_info, **kwargs)
                    response.raise_for_status()
                    found = check_anchor(response, unquote(anchor))

                    if not found:
                        raise Exception(__("Anchor '%s' not found") % anchor)
                else:
                    try:
                        # try a HEAD request first, which should be easier on
                        # the server and the network
                        response = requests.head(req_url, allow_redirects=True,
                                                 config=self.config, auth=auth_info,
                                                 **kwargs)
                        response.raise_for_status()
                    # Servers drop the connection on HEAD requests, causing
                    # ConnectionError.
                    except (ConnectionError, HTTPError, TooManyRedirects) as err:
                        if isinstance(err, HTTPError) and err.response.status_code == 429:
                            raise
                        # retry with GET request if that fails, some servers
                        # don't like HEAD requests.
                        response = requests.get(req_url, stream=True,
                                                config=self.config,
                                                auth=auth_info, **kwargs)
                        response.raise_for_status()
            except HTTPError as err:
                if err.response.status_code == 401:
                    # We'll take "Unauthorized" as working.
                    return 'working', ' - unauthorized', 0
                elif err.response.status_code == 429:
                    next_check = self.limit_rate(err.response)
                    if next_check is not None:
                        self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                        return 'rate-limited', '', 0
                    return 'broken', str(err), 0
                elif err.response.status_code == 503:
                    # We'll take "Service Unavailable" as ignored.
                    return 'ignored', str(err), 0
                else:
                    return 'broken', str(err), 0
            except Exception as err:
                return 'broken', str(err), 0
            else:
                netloc = urlparse(req_url).netloc
                try:
                    del self.rate_limits[netloc]
                except KeyError:
                    pass
            if response.url.rstrip('/') == req_url.rstrip('/'):
                return 'working', '', 0
            else:
                new_url = response.url
                if anchor:
                    new_url += '#' + anchor

                if allowed_redirect(req_url, new_url):
                    return 'working', '', 0
                elif response.history:
                    # history contains any redirects, get last
                    code = response.history[-1].status_code
                    return 'redirected', new_url, code
                else:
                    return 'redirected', new_url, 0
        # ... other code
```
### 2 - sphinx/builders/linkcheck.py:

Start line: 513, End line: 555

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def run(self) -> None:
        # ... other code

        def allowed_redirect(url: str, new_url: str) -> bool:
            for from_url, to_url in self.config.linkcheck_allowed_redirects.items():
                if from_url.match(url) and to_url.match(new_url):
                    return True

            return False

        def check(docname: str) -> Tuple[str, str, int]:
            # check for various conditions without bothering the network
            if len(uri) == 0 or uri.startswith(('#', 'mailto:', 'tel:')):
                return 'unchecked', '', 0
            elif not uri.startswith(('http:', 'https:')):
                if uri_re.match(uri):
                    # non supported URI schemes (ex. ftp)
                    return 'unchecked', '', 0
                else:
                    srcdir = path.dirname(self.env.doc2path(docname))
                    if path.exists(path.join(srcdir, uri)):
                        return 'working', '', 0
                    else:
                        self._broken[uri] = ''
                        return 'broken', '', 0
            elif uri in self._good:
                return 'working', 'old', 0
            elif uri in self._broken:
                return 'broken', self._broken[uri], 0
            elif uri in self._redirected:
                return 'redirected', self._redirected[uri][0], self._redirected[uri][1]

            # need to actually check the URI
            for _ in range(self.config.linkcheck_retries):
                status, info, code = check_uri()
                if status != "broken":
                    break

            if status == "working":
                self._good.add(uri)
            elif status == "broken":
                self._broken[uri] = info
            elif status == "redirected":
                self._redirected[uri] = (info, code)

            return (status, info, code)
        # ... other code
```
### 3 - sphinx/builders/linkcheck.py:

Start line: 341, End line: 361

```python
class HyperlinkAvailabilityChecker:

    def check(self, hyperlinks: Dict[str, Hyperlink]) -> Generator[CheckResult, None, None]:
        self.invoke_threads()

        total_links = 0
        for hyperlink in hyperlinks.values():
            if self.is_ignored_uri(hyperlink.uri):
                yield CheckResult(hyperlink.uri, hyperlink.docname, hyperlink.lineno,
                                  'ignored', '', 0)
            else:
                self.wqueue.put(CheckRequest(CHECK_IMMEDIATELY, hyperlink), False)
                total_links += 1

        done = 0
        while done < total_links:
            yield self.rqueue.get()
            done += 1

        self.shutdown_threads()

    def is_ignored_uri(self, uri: str) -> bool:
        return any(pat.match(uri) for pat in self.to_ignore)
```
### 4 - sphinx/builders/linkcheck.py:

Start line: 114, End line: 132

```python
class CheckExternalLinksBuilder(DummyBuilder):
    """
    Checks for broken external links.
    """
    name = 'linkcheck'
    epilog = __('Look for any errors in the above output or in '
                '%(outdir)s/output.txt')

    def init(self) -> None:
        self.hyperlinks: Dict[str, Hyperlink] = {}
        self._good: Set[str] = set()
        self._broken: Dict[str, str] = {}
        self._redirected: Dict[str, Tuple[str, int]] = {}
        # set a timeout for non-responding servers
        socket.setdefaulttimeout(5.0)

        # create queues and worker threads
        self._wqueue: PriorityQueue[CheckRequestType] = PriorityQueue()
        self._rqueue: Queue[CheckResult] = Queue()
```
### 5 - sphinx/builders/linkcheck.py:

Start line: 307, End line: 339

```python
class HyperlinkAvailabilityChecker:
    def __init__(self, env: BuildEnvironment, config: Config,
                 builder: CheckExternalLinksBuilder = None) -> None:
        # Warning: builder argument will be removed in the sphinx-5.0.
        # Don't use it from extensions.
        # tag: RemovedInSphinx50Warning
        self.builder = builder
        self.config = config
        self.env = env
        self.rate_limits: Dict[str, RateLimit] = {}
        self.workers: List[Thread] = []

        self.to_ignore = [re.compile(x) for x in self.config.linkcheck_ignore]

        if builder:
            self.rqueue = builder._rqueue
            self.wqueue = builder._wqueue
        else:
            self.rqueue = Queue()
            self.wqueue = PriorityQueue()

    def invoke_threads(self) -> None:
        for i in range(self.config.linkcheck_workers):
            thread = HyperlinkAvailabilityCheckWorker(self.env, self.config,
                                                      self.rqueue, self.wqueue,
                                                      self.rate_limits, self.builder)
            thread.start()
            self.workers.append(thread)

    def shutdown_threads(self) -> None:
        self.wqueue.join()
        for worker in self.workers:
            self.wqueue.put(CheckRequest(CHECK_IMMEDIATELY, None), False)
```
### 6 - sphinx/builders/linkcheck.py:

Start line: 286, End line: 304

```python
class CheckExternalLinksBuilder(DummyBuilder):

    def write_entry(self, what: str, docname: str, filename: str, line: int,
                    uri: str) -> None:
        self.txt_outfile.write("%s:%s: [%s] %s\n" % (filename, line, what, uri))

    def write_linkstat(self, data: dict) -> None:
        self.json_outfile.write(json.dumps(data))
        self.json_outfile.write('\n')

    def finish(self) -> None:
        checker = HyperlinkAvailabilityChecker(self.env, self.config, self)
        logger.info('')

        with open(path.join(self.outdir, 'output.txt'), 'w') as self.txt_outfile,\
             open(path.join(self.outdir, 'output.json'), 'w') as self.json_outfile:
            for result in checker.check(self.hyperlinks):
                self.process_result(result)

        if self._broken:
            self.app.statuscode = 1
```
### 7 - sphinx/builders/linkcheck.py:

Start line: 557, End line: 592

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def run(self) -> None:
        # ... other code

        while True:
            check_request = self.wqueue.get()
            try:
                next_check, hyperlink = check_request
                if hyperlink is None:
                    break

                uri, docname, lineno = hyperlink
            except ValueError:
                # old styled check_request (will be deprecated in Sphinx-5.0)
                next_check, uri, docname, lineno = check_request

            if uri is None:
                break
            netloc = urlparse(uri).netloc
            try:
                # Refresh rate limit.
                # When there are many links in the queue, workers are all stuck waiting
                # for responses, but the builder keeps queuing. Links in the queue may
                # have been queued before rate limits were discovered.
                next_check = self.rate_limits[netloc].next_check
            except KeyError:
                pass
            if next_check > time.time():
                # Sleep before putting message back in the queue to avoid
                # waking up other threads.
                time.sleep(QUEUE_POLL_SECS)
                self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                self.wqueue.task_done()
                continue
            status, info, code = check(docname)
            if status == 'rate-limited':
                logger.info(darkgray('-rate limited-   ') + uri + darkgray(' | sleeping...'))
            else:
                self.rqueue.put(CheckResult(uri, docname, lineno, status, info, code))
            self.wqueue.task_done()
```
### 8 - sphinx/builders/linkcheck.py:

Start line: 231, End line: 284

```python
class CheckExternalLinksBuilder(DummyBuilder):

    def process_result(self, result: CheckResult) -> None:
        filename = self.env.doc2path(result.docname, None)

        linkstat = dict(filename=filename, lineno=result.lineno,
                        status=result.status, code=result.code, uri=result.uri,
                        info=result.message)
        self.write_linkstat(linkstat)

        if result.status == 'unchecked':
            return
        if result.status == 'working' and result.message == 'old':
            return
        if result.lineno:
            logger.info('(%16s: line %4d) ', result.docname, result.lineno, nonl=True)
        if result.status == 'ignored':
            if result.message:
                logger.info(darkgray('-ignored- ') + result.uri + ': ' + result.message)
            else:
                logger.info(darkgray('-ignored- ') + result.uri)
        elif result.status == 'local':
            logger.info(darkgray('-local-   ') + result.uri)
            self.write_entry('local', result.docname, filename, result.lineno, result.uri)
        elif result.status == 'working':
            logger.info(darkgreen('ok        ') + result.uri + result.message)
        elif result.status == 'broken':
            if self.app.quiet or self.app.warningiserror:
                logger.warning(__('broken link: %s (%s)'), result.uri, result.message,
                               location=(filename, result.lineno))
            else:
                logger.info(red('broken    ') + result.uri + red(' - ' + result.message))
            self.write_entry('broken', result.docname, filename, result.lineno,
                             result.uri + ': ' + result.message)
        elif result.status == 'redirected':
            try:
                text, color = {
                    301: ('permanently', purple),
                    302: ('with Found', purple),
                    303: ('with See Other', purple),
                    307: ('temporarily', turquoise),
                    308: ('permanently', purple),
                }[result.code]
            except KeyError:
                text, color = ('with unknown code', purple)
            linkstat['text'] = text
            if self.config.linkcheck_allowed_redirects:
                logger.warning('redirect  ' + result.uri + ' - ' + text + ' to ' +
                               result.message, location=(filename, result.lineno))
            else:
                logger.info(color('redirect  ') + result.uri +
                            color(' - ' + text + ' to ' + result.message))
            self.write_entry('redirected ' + text, result.docname, filename,
                             result.lineno, result.uri + ' to ' + result.message)
        else:
            raise ValueError("Unknown status %s." % result.status)
```
### 9 - sphinx/builders/linkcheck.py:

Start line: 669, End line: 681

```python
def rewrite_github_anchor(app: Sphinx, uri: str) -> Optional[str]:
    """Rewrite anchor name of the hyperlink to github.com

    The hyperlink anchors in github.com are dynamically generated.  This rewrites
    them before checking and makes them comparable.
    """
    parsed = urlparse(uri)
    if parsed.hostname == "github.com" and parsed.fragment:
        prefixed = parsed.fragment.startswith('user-content-')
        if not prefixed:
            fragment = f'user-content-{parsed.fragment}'
            return urlunparse(parsed._replace(fragment=fragment))
    return None
```
### 10 - sphinx/builders/linkcheck.py:

Start line: 397, End line: 415

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def run(self) -> None:
        kwargs = {}
        if self.config.linkcheck_timeout:
            kwargs['timeout'] = self.config.linkcheck_timeout

        def get_request_headers() -> Dict:
            url = urlparse(uri)
            candidates = ["%s://%s" % (url.scheme, url.netloc),
                          "%s://%s/" % (url.scheme, url.netloc),
                          uri,
                          "*"]

            for u in candidates:
                if u in self.config.linkcheck_request_headers:
                    headers = dict(DEFAULT_REQUEST_HEADERS)
                    headers.update(self.config.linkcheck_request_headers[u])
                    return headers

            return {}
        # ... other code
```
### 11 - sphinx/builders/linkcheck.py:

Start line: 697, End line: 724

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(CheckExternalLinksBuilder)
    app.add_post_transform(HyperlinkCollector)

    app.add_config_value('linkcheck_ignore', [], None)
    app.add_config_value('linkcheck_allowed_redirects', {}, None)
    app.add_config_value('linkcheck_auth', [], None)
    app.add_config_value('linkcheck_request_headers', {}, None)
    app.add_config_value('linkcheck_retries', 1, None)
    app.add_config_value('linkcheck_timeout', None, None, [int])
    app.add_config_value('linkcheck_workers', 5, None)
    app.add_config_value('linkcheck_anchors', True, None)
    # Anchors starting with ! are ignored since they are
    # commonly used for dynamic pages
    app.add_config_value('linkcheck_anchors_ignore', ["^!"], None)
    app.add_config_value('linkcheck_rate_limit_timeout', 300.0, None)

    app.add_event('linkcheck-process-uri')

    app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=800)
    app.connect('linkcheck-process-uri', rewrite_github_anchor)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 12 - sphinx/builders/linkcheck.py:

Start line: 134, End line: 229

```python
class CheckExternalLinksBuilder(DummyBuilder):

    @property
    def anchors_ignore(self) -> List[Pattern]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "anchors_ignore"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return [re.compile(x) for x in self.config.linkcheck_anchors_ignore]

    @property
    def auth(self) -> List[Tuple[Pattern, Any]]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "auth"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return [(re.compile(pattern), auth_info) for pattern, auth_info
                in self.config.linkcheck_auth]

    @property
    def to_ignore(self) -> List[Pattern]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "to_ignore"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return [re.compile(x) for x in self.config.linkcheck_ignore]

    @property
    def good(self) -> Set[str]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "good"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return self._good

    @property
    def broken(self) -> Dict[str, str]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "broken"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return self._broken

    @property
    def redirected(self) -> Dict[str, Tuple[str, int]]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "redirected"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return self._redirected

    def check_thread(self) -> None:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "check_thread"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        # do nothing.

    def limit_rate(self, response: Response) -> Optional[float]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "limit_rate"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        worker = HyperlinkAvailabilityCheckWorker(self.env, self.config,
                                                  None, None, {})
        return worker.limit_rate(response)

    def rqueue(self, response: Response) -> Queue:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "rqueue"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return self._rqueue

    def workers(self, response: Response) -> List[Thread]:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "workers"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return []

    def wqueue(self, response: Response) -> Queue:
        warnings.warn(
            "%s.%s is deprecated." % (self.__class__.__name__, "wqueue"),
            RemovedInSphinx50Warning,
            stacklevel=2,
        )
        return self._wqueue
```
### 14 - sphinx/builders/linkcheck.py:

Start line: 633, End line: 666

```python
class HyperlinkCollector(SphinxPostTransform):
    builders = ('linkcheck',)
    default_priority = 800

    def run(self, **kwargs: Any) -> None:
        builder = cast(CheckExternalLinksBuilder, self.app.builder)
        hyperlinks = builder.hyperlinks

        # reference nodes
        for refnode in self.document.traverse(nodes.reference):
            if 'refuri' not in refnode:
                continue
            uri = refnode['refuri']
            newuri = self.app.emit_firstresult('linkcheck-process-uri', uri)
            if newuri:
                uri = newuri

            lineno = get_node_line(refnode)
            uri_info = Hyperlink(uri, self.env.docname, lineno)
            if uri not in hyperlinks:
                hyperlinks[uri] = uri_info

        # image nodes
        for imgnode in self.document.traverse(nodes.image):
            uri = imgnode['candidates'].get('?')
            if uri and '://' in uri:
                newuri = self.app.emit_firstresult('linkcheck-process-uri', uri)
                if newuri:
                    uri = newuri

                lineno = get_node_line(imgnode)
                uri_info = Hyperlink(uri, self.env.docname, lineno)
                if uri not in hyperlinks:
                    hyperlinks[uri] = uri_info
```
### 15 - sphinx/builders/linkcheck.py:

Start line: 96, End line: 111

```python
def check_anchor(response: requests.requests.Response, anchor: str) -> bool:
    """Reads HTML data from a response object `response` searching for `anchor`.
    Returns True if anchor was found, False otherwise.
    """
    parser = AnchorCheckParser(anchor)
    # Read file in chunks. If we find a matching anchor, we break
    # the loop early in hopes not to have to download the whole thing.
    for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
        if isinstance(chunk, bytes):    # requests failed to decode
            chunk = chunk.decode()      # manually try to decode it

        parser.feed(chunk)
        if parser.found:
            break
    parser.close()
    return parser.found
```
### 16 - sphinx/builders/linkcheck.py:

Start line: 594, End line: 630

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def limit_rate(self, response: Response) -> Optional[float]:
        next_check = None
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                # Integer: time to wait before next attempt.
                delay = float(retry_after)
            except ValueError:
                try:
                    # An HTTP-date: time of next attempt.
                    until = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    # TypeError: Invalid date format.
                    # ValueError: Invalid date, e.g. Oct 52th.
                    pass
                else:
                    next_check = datetime.timestamp(until)
                    delay = (until - datetime.now(timezone.utc)).total_seconds()
            else:
                next_check = time.time() + delay
        netloc = urlparse(response.url).netloc
        if next_check is None:
            max_delay = self.config.linkcheck_rate_limit_timeout
            try:
                rate_limit = self.rate_limits[netloc]
            except KeyError:
                delay = DEFAULT_DELAY
            else:
                last_wait_time = rate_limit.delay
                delay = 2.0 * last_wait_time
                if delay > max_delay and last_wait_time < max_delay:
                    delay = max_delay
            if delay > max_delay:
                return None
            next_check = time.time() + delay
        self.rate_limits[netloc] = RateLimit(delay, next_check)
        return next_check
```
### 17 - sphinx/builders/linkcheck.py:

Start line: 11, End line: 93

```python
import json
import re
import socket
import time
import warnings
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import (Any, Dict, Generator, List, NamedTuple, Optional, Pattern, Set, Tuple,
                    Union, cast)
from urllib.parse import unquote, urlparse, urlunparse

from docutils import nodes
from docutils.nodes import Element
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, TooManyRedirects

from sphinx.application import Sphinx
from sphinx.builders.dummy import DummyBuilder
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line

logger = logging.getLogger(__name__)

uri_re = re.compile('([a-z]+:)?//')  # matches to foo:// and // (a protocol relative URL)

Hyperlink = NamedTuple('Hyperlink', (('uri', str),
                                     ('docname', str),
                                     ('lineno', Optional[int])))
CheckRequest = NamedTuple('CheckRequest', (('next_check', float),
                                           ('hyperlink', Optional[Hyperlink])))
CheckResult = NamedTuple('CheckResult', (('uri', str),
                                         ('docname', str),
                                         ('lineno', int),
                                         ('status', str),
                                         ('message', str),
                                         ('code', int)))
RateLimit = NamedTuple('RateLimit', (('delay', float), ('next_check', float)))

# Tuple is old styled CheckRequest
CheckRequestType = Union[CheckRequest, Tuple[float, str, str, int]]

DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
}
CHECK_IMMEDIATELY = 0
QUEUE_POLL_SECS = 1
DEFAULT_DELAY = 60.0


def node_line_or_0(node: Element) -> int:
    """
    PriorityQueue items must be comparable. The line number is part of the
    tuple used by the PriorityQueue, keep an homogeneous type for comparison.
    """
    warnings.warn('node_line_or_0() is deprecated.',
                  RemovedInSphinx50Warning, stacklevel=2)
    return get_node_line(node) or 0


class AnchorCheckParser(HTMLParser):
    """Specialized HTML parser that looks for a specific anchor."""

    def __init__(self, search_anchor: str) -> None:
        super().__init__()

        self.search_anchor = search_anchor
        self.found = False

    def handle_starttag(self, tag: Any, attrs: Any) -> None:
        for key, value in attrs:
            if key in ('id', 'name') and value == self.search_anchor:
                self.found = True
                break
```
### 18 - sphinx/builders/linkcheck.py:

Start line: 364, End line: 395

```python
class HyperlinkAvailabilityCheckWorker(Thread):
    """A worker class for checking the availability of hyperlinks."""

    def __init__(self, env: BuildEnvironment, config: Config, rqueue: Queue,
                 wqueue: Queue, rate_limits: Dict[str, RateLimit],
                 builder: CheckExternalLinksBuilder = None) -> None:
        # Warning: builder argument will be removed in the sphinx-5.0.
        # Don't use it from extensions.
        # tag: RemovedInSphinx50Warning
        self.config = config
        self.env = env
        self.rate_limits = rate_limits
        self.rqueue = rqueue
        self.wqueue = wqueue

        self.anchors_ignore = [re.compile(x)
                               for x in self.config.linkcheck_anchors_ignore]
        self.auth = [(re.compile(pattern), auth_info) for pattern, auth_info
                     in self.config.linkcheck_auth]

        if builder:
            # if given, fill the result of checks as cache
            self._good = builder._good
            self._broken = builder._broken
            self._redirected = builder._redirected
        else:
            # only for compatibility. Will be removed in Sphinx-5.0
            self._good = set()
            self._broken = {}
            self._redirected = {}

        super().__init__(daemon=True)
```
### 21 - sphinx/builders/linkcheck.py:

Start line: 684, End line: 694

```python
def compile_linkcheck_allowed_redirects(app: Sphinx, config: Config) -> None:
    """Compile patterns in linkcheck_allowed_redirects to the regexp objects."""
    for url, pattern in list(app.config.linkcheck_allowed_redirects.items()):
        try:
            app.config.linkcheck_allowed_redirects[re.compile(url)] = re.compile(pattern)
        except re.error as exc:
            logger.warning(__('Failed to compile regex in linkcheck_allowed_redirects: %r %s'),
                           exc.pattern, exc.msg)
        finally:
            # Remove the original regexp-string
            app.config.linkcheck_allowed_redirects.pop(url)
```
