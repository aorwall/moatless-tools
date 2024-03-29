# sphinx-doc__sphinx-8579

| **sphinx-doc/sphinx** | `955d6558ec155dffaef999d890c2cdb224cbfbb9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6208 |
| **Any found context length** | 2209 |
| **Avg pos** | 46.0 |
| **Min pos** | 6 |
| **Max pos** | 17 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -22,7 +22,7 @@
 from urllib.parse import unquote, urlparse
 
 from docutils import nodes
-from docutils.nodes import Node
+from docutils.nodes import Element, Node
 from requests import Response
 from requests.exceptions import HTTPError, TooManyRedirects
 
@@ -47,6 +47,14 @@
 DEFAULT_DELAY = 60.0
 
 
+def node_line_or_0(node: Element) -> int:
+    """
+    PriorityQueue items must be comparable. The line number is part of the
+    tuple used by the PriorityQueue, keep an homogeneous type for comparison.
+    """
+    return get_node_line(node) or 0
+
+
 class AnchorCheckParser(HTMLParser):
     """Specialized HTML parser that looks for a specific anchor."""
 
@@ -406,7 +414,7 @@ def write_doc(self, docname: str, doctree: Node) -> None:
             if 'refuri' not in refnode:
                 continue
             uri = refnode['refuri']
-            lineno = get_node_line(refnode)
+            lineno = node_line_or_0(refnode)
             uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
             self.wqueue.put(uri_info, False)
             n += 1
@@ -415,7 +423,7 @@ def write_doc(self, docname: str, doctree: Node) -> None:
         for imgnode in doctree.traverse(nodes.image):
             uri = imgnode['candidates'].get('?')
             if uri and '://' in uri:
-                lineno = get_node_line(imgnode)
+                lineno = node_line_or_0(imgnode)
                 uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
                 self.wqueue.put(uri_info, False)
                 n += 1

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/linkcheck.py | 25 | 25 | 17 | 1 | 6208
| sphinx/builders/linkcheck.py | 50 | 50 | 17 | 1 | 6208
| sphinx/builders/linkcheck.py | 409 | 409 | 6 | 1 | 2209
| sphinx/builders/linkcheck.py | 418 | 418 | 6 | 1 | 2209


## Problem Statement

```
Linkcheck crashes in 3.4.0
**Describe the bug**

When running linkcheck in Weblate docs, it crashes with:

\`\`\`
 Exception in thread Thread-2:
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.8.6/x64/lib/python3.8/threading.py", line 932, in _bootstrap_inner

Exception occurred:
    self.run()
  File "/opt/hostedtoolcache/Python/3.8.6/x64/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/hostedtoolcache/Python/3.8.6/x64/lib/python3.8/site-packages/sphinx/builders/linkcheck.py", line 298, in check_thread
    self.wqueue.task_done()
  File "/opt/hostedtoolcache/Python/3.8.6/x64/lib/python3.8/queue.py", line 74, in task_done
Error:     raise ValueError('task_done() called too many times')
ValueError: task_done() called too many times
  File "/opt/hostedtoolcache/Python/3.8.6/x64/lib/python3.8/queue.py", line 233, in _put
    heappush(self.queue, item)
TypeError: '<' not supported between instances of 'int' and 'NoneType'
\`\`\`

**To Reproduce**
Steps to reproduce the behavior:
\`\`\`
<Paste your command-line here which cause the problem>

$ git clone https://github.com/WeblateOrg/weblate.git
$ cd weblate
$ pip install -r docs/requirements.txt
$ cd docs
$ make linkcheck
\`\`\`

**Expected behavior**
No crash :-)

**Your project**
https://github.com/WeblateOrg/weblate/tree/master/docs

**Screenshots**
CI failure: https://github.com/WeblateOrg/weblate/runs/1585580811?check_suite_focus=true

**Environment info**
- OS: Linux
- Python version: 3.8.6
- Sphinx version: 3.4.0
- Sphinx extensions:  several, but should not be relevant here
- Extra tools: none involved

**Additional context**
Add any other context about the problem here.

- [e.g. URL or Ticket]



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/builders/linkcheck.py** | 273 | 298| 274 | 274 | 3911 | 
| 2 | **1 sphinx/builders/linkcheck.py** | 84 | 116| 350 | 624 | 3911 | 
| 3 | **1 sphinx/builders/linkcheck.py** | 229 | 271| 403 | 1027 | 3911 | 
| 4 | **1 sphinx/builders/linkcheck.py** | 138 | 227| 732 | 1759 | 3911 | 
| 5 | **1 sphinx/builders/linkcheck.py** | 118 | 136| 145 | 1904 | 3911 | 
| **-> 6 <-** | **1 sphinx/builders/linkcheck.py** | 391 | 429| 305 | 2209 | 3911 | 
| 7 | **1 sphinx/builders/linkcheck.py** | 431 | 445| 175 | 2384 | 3911 | 
| 8 | **1 sphinx/builders/linkcheck.py** | 338 | 389| 533 | 2917 | 3911 | 
| 9 | **1 sphinx/builders/linkcheck.py** | 300 | 336| 309 | 3226 | 3911 | 
| 10 | **1 sphinx/builders/linkcheck.py** | 448 | 468| 211 | 3437 | 3911 | 
| 11 | 2 sphinx/ext/linkcode.py | 11 | 80| 470 | 3907 | 4437 | 
| 12 | 3 utils/checks.py | 33 | 109| 545 | 4452 | 5343 | 
| 13 | 4 sphinx/errors.py | 70 | 127| 297 | 4749 | 6085 | 
| 14 | 5 sphinx/ext/extlinks.py | 26 | 71| 400 | 5149 | 6704 | 
| 15 | 6 sphinx/__init__.py | 14 | 65| 495 | 5644 | 7273 | 
| 16 | 7 sphinx/environment/__init__.py | 624 | 643| 178 | 5822 | 13128 | 
| **-> 17 <-** | **7 sphinx/builders/linkcheck.py** | 11 | 63| 386 | 6208 | 13128 | 
| 18 | 8 sphinx/setup_command.py | 147 | 196| 423 | 6631 | 14711 | 
| 19 | 9 sphinx/util/pycompat.py | 11 | 53| 357 | 6988 | 15579 | 
| 20 | 10 sphinx/util/requests.py | 11 | 67| 415 | 7403 | 16527 | 
| 21 | 10 utils/checks.py | 11 | 30| 175 | 7578 | 16527 | 
| 22 | 11 doc/conf.py | 59 | 127| 688 | 8266 | 18085 | 
| 23 | 12 sphinx/builders/changes.py | 125 | 168| 438 | 8704 | 19607 | 
| 24 | 13 sphinx/domains/std.py | 1114 | 1138| 205 | 8909 | 30111 | 
| 25 | 14 sphinx/transforms/post_transforms/__init__.py | 152 | 178| 299 | 9208 | 32128 | 
| 26 | **14 sphinx/builders/linkcheck.py** | 66 | 81| 156 | 9364 | 32128 | 
| 27 | 15 sphinx/domains/python.py | 1371 | 1409| 304 | 9668 | 44065 | 
| 28 | 16 sphinx/ext/todo.py | 275 | 301| 271 | 9939 | 46779 | 
| 29 | 17 sphinx/ext/doctest.py | 251 | 278| 277 | 10216 | 51867 | 
| 30 | 18 sphinx/cmd/build.py | 11 | 30| 132 | 10348 | 54528 | 
| 31 | 19 sphinx/transforms/i18n.py | 229 | 384| 1673 | 12021 | 59128 | 
| 32 | 20 setup.py | 173 | 250| 650 | 12671 | 60863 | 
| 33 | 21 sphinx/writers/texinfo.py | 865 | 969| 788 | 13459 | 73197 | 
| 34 | 22 sphinx/ext/graphviz.py | 12 | 43| 233 | 13692 | 76847 | 
| 35 | 23 utils/bump_version.py | 67 | 102| 201 | 13893 | 78210 | 
| 36 | 23 sphinx/builders/changes.py | 51 | 124| 814 | 14707 | 78210 | 
| 37 | 24 sphinx/directives/code.py | 59 | 83| 189 | 14896 | 82115 | 
| 38 | 24 sphinx/environment/__init__.py | 11 | 80| 481 | 15377 | 82115 | 
| 39 | 24 sphinx/ext/todo.py | 14 | 43| 204 | 15581 | 82115 | 
| 40 | 25 sphinx/cmd/quickstart.py | 11 | 118| 771 | 16352 | 87710 | 
| 41 | 25 doc/conf.py | 1 | 58| 511 | 16863 | 87710 | 
| 42 | 26 sphinx/builders/htmlhelp.py | 12 | 48| 297 | 17160 | 88072 | 
| 43 | 27 sphinx/writers/latex.py | 2125 | 2164| 467 | 17627 | 107955 | 
| 44 | 28 sphinx/ext/viewcode.py | 120 | 138| 196 | 17823 | 110193 | 
| 45 | 29 sphinx/builders/html/__init__.py | 1196 | 1272| 909 | 18732 | 121576 | 
| 46 | 30 sphinx/config.py | 449 | 462| 161 | 18893 | 126032 | 
| 47 | 30 sphinx/config.py | 394 | 446| 474 | 19367 | 126032 | 
| 48 | 31 sphinx/util/smartypants.py | 375 | 387| 137 | 19504 | 130143 | 
| 49 | 31 sphinx/ext/doctest.py | 12 | 52| 286 | 19790 | 130143 | 
| 50 | 32 sphinx/extension.py | 43 | 70| 225 | 20015 | 130649 | 
| 51 | 32 setup.py | 1 | 75| 441 | 20456 | 130649 | 
| 52 | 32 sphinx/ext/viewcode.py | 11 | 44| 293 | 20749 | 130649 | 
| 53 | 32 sphinx/builders/html/__init__.py | 147 | 168| 223 | 20972 | 130649 | 
| 54 | 33 sphinx/util/parallel.py | 77 | 91| 124 | 21096 | 131806 | 
| 55 | 33 sphinx/directives/code.py | 9 | 30| 141 | 21237 | 131806 | 
| 56 | 33 sphinx/setup_command.py | 94 | 122| 240 | 21477 | 131806 | 
| 57 | 33 sphinx/domains/python.py | 1032 | 1052| 248 | 21725 | 131806 | 
| 58 | 33 sphinx/cmd/quickstart.py | 552 | 619| 491 | 22216 | 131806 | 
| 59 | 33 sphinx/writers/texinfo.py | 755 | 863| 813 | 23029 | 131806 | 
| 60 | 34 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 23204 | 132714 | 
| 61 | 35 sphinx/writers/html.py | 11 | 52| 312 | 23516 | 140110 | 
| 62 | 36 sphinx/builders/applehelp.py | 11 | 46| 266 | 23782 | 140426 | 
| 63 | 37 sphinx/cmd/make_mode.py | 17 | 53| 515 | 24297 | 142111 | 
| 64 | 38 sphinx/builders/__init__.py | 432 | 466| 326 | 24623 | 147502 | 
| 65 | 39 sphinx/writers/manpage.py | 309 | 407| 757 | 25380 | 151014 | 
| 66 | 40 sphinx/writers/html5.py | 231 | 252| 208 | 25588 | 157937 | 
| 67 | 40 sphinx/builders/__init__.py | 546 | 575| 318 | 25906 | 157937 | 
| 68 | 40 sphinx/cmd/make_mode.py | 95 | 139| 375 | 26281 | 157937 | 
| 69 | 40 sphinx/ext/doctest.py | 320 | 335| 130 | 26411 | 157937 | 
| 70 | 41 sphinx/application.py | 341 | 390| 410 | 26821 | 169227 | 
| 71 | 42 sphinx/util/__init__.py | 632 | 660| 265 | 27086 | 175529 | 
| 72 | 43 sphinx/ext/autosummary/generate.py | 20 | 57| 275 | 27361 | 180778 | 
| 73 | 44 sphinx/util/docutils.py | 382 | 401| 181 | 27542 | 184901 | 
| 74 | 45 sphinx/writers/text.py | 929 | 1042| 873 | 28415 | 193874 | 


### Hint

```
Hi,
Thanks for the report! I can reproduce the issue.
I’ll be looking into fixing it later today.
I understand what is wrong:

Linkcheck organizes the urls to checks in a `PriorityQueue`. The items are tuples `(priority, url, docname, lineno)`. For some links, the `get_node_line()` returns `None`, I’m guessing the line information is not available on that node.
A tuple where the `lineno` is `None` is not comparable with `tuples` that have an integer `lineno` (`None` and `int` aren’t comparable), and `PriorityQueue` items must be comparable (see https://bugs.python.org/issue31145).

That issue only manifests when a link has no `lineno` and a document contains two links to the same URL. In [Weblate README.rst](https://raw.githubusercontent.com/WeblateOrg/weblate/master/README.rst):

\`\`\`sphinx
.. image:: https://s.weblate.org/cdn/Logo-Darktext-borders.png
   :alt: Weblate
   :target: https://weblate.org/
\`\`\`
And:
\`\`\`sphinx
Install it, or use the Hosted Weblate service at `weblate.org`_.

.. _weblate.org: https://weblate.org/
\`\`\`

I have a minimal regression test and will investigate how the line number is acquired tomorrow. If that’s reasonable, I think it would be more helpful to have the original line number where the URL appeared. If it is too big of a change for a fix release, I’ll probably end-up wrapping the data in a class that handles the comparison between no line number information and a line number information (and its variants).
```

## Patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -22,7 +22,7 @@
 from urllib.parse import unquote, urlparse
 
 from docutils import nodes
-from docutils.nodes import Node
+from docutils.nodes import Element, Node
 from requests import Response
 from requests.exceptions import HTTPError, TooManyRedirects
 
@@ -47,6 +47,14 @@
 DEFAULT_DELAY = 60.0
 
 
+def node_line_or_0(node: Element) -> int:
+    """
+    PriorityQueue items must be comparable. The line number is part of the
+    tuple used by the PriorityQueue, keep an homogeneous type for comparison.
+    """
+    return get_node_line(node) or 0
+
+
 class AnchorCheckParser(HTMLParser):
     """Specialized HTML parser that looks for a specific anchor."""
 
@@ -406,7 +414,7 @@ def write_doc(self, docname: str, doctree: Node) -> None:
             if 'refuri' not in refnode:
                 continue
             uri = refnode['refuri']
-            lineno = get_node_line(refnode)
+            lineno = node_line_or_0(refnode)
             uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
             self.wqueue.put(uri_info, False)
             n += 1
@@ -415,7 +423,7 @@ def write_doc(self, docname: str, doctree: Node) -> None:
         for imgnode in doctree.traverse(nodes.image):
             uri = imgnode['candidates'].get('?')
             if uri and '://' in uri:
-                lineno = get_node_line(imgnode)
+                lineno = node_line_or_0(imgnode)
                 uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
                 self.wqueue.put(uri_info, False)
                 n += 1

```

## Test Patch

```diff
diff --git a/tests/roots/test-linkcheck-localserver-two-links/conf.py b/tests/roots/test-linkcheck-localserver-two-links/conf.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver-two-links/conf.py
@@ -0,0 +1 @@
+exclude_patterns = ['_build']
diff --git a/tests/roots/test-linkcheck-localserver-two-links/index.rst b/tests/roots/test-linkcheck-localserver-two-links/index.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver-two-links/index.rst
@@ -0,0 +1,6 @@
+.. image:: http://localhost:7777/
+   :target: http://localhost:7777/
+
+`weblate.org`_
+
+.. _weblate.org: http://localhost:7777/
diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -573,3 +573,40 @@ def test_limit_rate_bails_out_after_waiting_max_time(app):
     checker.rate_limits = {"localhost": RateLimit(90.0, 0.0)}
     next_check = checker.limit_rate(FakeResponse())
     assert next_check is None
+
+
+@pytest.mark.sphinx(
+    'linkcheck', testroot='linkcheck-localserver-two-links', freshenv=True,
+)
+def test_priorityqueue_items_are_comparable(app):
+    with http_server(OKHandler):
+        app.builder.build_all()
+    content = (app.outdir / 'output.json').read_text()
+    rows = [json.loads(x) for x in sorted(content.splitlines())]
+    assert rows == [
+        {
+            'filename': 'index.rst',
+            # Should not be None.
+            'lineno': 0,
+            'status': 'working',
+            'code': 0,
+            'uri': 'http://localhost:7777/',
+            'info': '',
+        },
+        {
+            'filename': 'index.rst',
+            'lineno': 0,
+            'status': 'working',
+            'code': 0,
+            'uri': 'http://localhost:7777/',
+            'info': '',
+        },
+        {
+            'filename': 'index.rst',
+            'lineno': 4,
+            'status': 'working',
+            'code': 0,
+            'uri': 'http://localhost:7777/',
+            'info': '',
+        }
+    ]

```


## Code snippets

### 1 - sphinx/builders/linkcheck.py:

Start line: 273, End line: 298

```python
class CheckExternalLinksBuilder(Builder):

    def check_thread(self) -> None:
        # ... other code

        while True:
            next_check, uri, docname, lineno = self.wqueue.get()
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
                self.wqueue.put((next_check, uri, docname, lineno), False)
                self.wqueue.task_done()
                continue
            status, info, code = check(docname)
            if status == 'rate-limited':
                logger.info(darkgray('-rate limited-   ') + uri + darkgray(' | sleeping...'))
            else:
                self.rqueue.put((uri, docname, lineno, status, info, code))
            self.wqueue.task_done()
```
### 2 - sphinx/builders/linkcheck.py:

Start line: 84, End line: 116

```python
class CheckExternalLinksBuilder(Builder):
    """
    Checks for broken external links.
    """
    name = 'linkcheck'
    epilog = __('Look for any errors in the above output or in '
                '%(outdir)s/output.txt')

    def init(self) -> None:
        self.to_ignore = [re.compile(x) for x in self.app.config.linkcheck_ignore]
        self.anchors_ignore = [re.compile(x)
                               for x in self.app.config.linkcheck_anchors_ignore]
        self.auth = [(re.compile(pattern), auth_info) for pattern, auth_info
                     in self.app.config.linkcheck_auth]
        self.good = set()       # type: Set[str]
        self.broken = {}        # type: Dict[str, str]
        self.redirected = {}    # type: Dict[str, Tuple[str, int]]
        # set a timeout for non-responding servers
        socket.setdefaulttimeout(5.0)
        # create output file
        open(path.join(self.outdir, 'output.txt'), 'w').close()
        # create JSON output file
        open(path.join(self.outdir, 'output.json'), 'w').close()

        # create queues and worker threads
        self.rate_limits = {}  # type: Dict[str, RateLimit]
        self.wqueue = queue.PriorityQueue()  # type: queue.PriorityQueue
        self.rqueue = queue.Queue()  # type: queue.Queue
        self.workers = []  # type: List[threading.Thread]
        for i in range(self.app.config.linkcheck_workers):
            thread = threading.Thread(target=self.check_thread, daemon=True)
            thread.start()
            self.workers.append(thread)
```
### 3 - sphinx/builders/linkcheck.py:

Start line: 229, End line: 271

```python
class CheckExternalLinksBuilder(Builder):

    def check_thread(self) -> None:
        # ... other code

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
                        for rex in self.to_ignore:
                            if rex.match(uri):
                                return 'ignored', '', 0
                        else:
                            self.broken[uri] = ''
                            return 'broken', '', 0
            elif uri in self.good:
                return 'working', 'old', 0
            elif uri in self.broken:
                return 'broken', self.broken[uri], 0
            elif uri in self.redirected:
                return 'redirected', self.redirected[uri][0], self.redirected[uri][1]
            for rex in self.to_ignore:
                if rex.match(uri):
                    return 'ignored', '', 0

            # need to actually check the URI
            for _ in range(self.app.config.linkcheck_retries):
                status, info, code = check_uri()
                if status != "broken":
                    break

            if status == "working":
                self.good.add(uri)
            elif status == "broken":
                self.broken[uri] = info
            elif status == "redirected":
                self.redirected[uri] = (info, code)

            return (status, info, code)
        # ... other code
```
### 4 - sphinx/builders/linkcheck.py:

Start line: 138, End line: 227

```python
class CheckExternalLinksBuilder(Builder):

    def check_thread(self) -> None:
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
                if anchor and self.app.config.linkcheck_anchors:
                    # Read the whole document and see if #anchor exists
                    response = requests.get(req_url, stream=True, config=self.app.config,
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
                                                 config=self.app.config, auth=auth_info,
                                                 **kwargs)
                        response.raise_for_status()
                    except (HTTPError, TooManyRedirects) as err:
                        if isinstance(err, HTTPError) and err.response.status_code == 429:
                            raise
                        # retry with GET request if that fails, some servers
                        # don't like HEAD requests.
                        response = requests.get(req_url, stream=True,
                                                config=self.app.config,
                                                auth=auth_info, **kwargs)
                        response.raise_for_status()
            except HTTPError as err:
                if err.response.status_code == 401:
                    # We'll take "Unauthorized" as working.
                    return 'working', ' - unauthorized', 0
                elif err.response.status_code == 429:
                    next_check = self.limit_rate(err.response)
                    if next_check is not None:
                        self.wqueue.put((next_check, uri, docname, lineno), False)
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
                # history contains any redirects, get last
                if response.history:
                    code = response.history[-1].status_code
                    return 'redirected', new_url, code
                else:
                    return 'redirected', new_url, 0
        # ... other code
```
### 5 - sphinx/builders/linkcheck.py:

Start line: 118, End line: 136

```python
class CheckExternalLinksBuilder(Builder):

    def check_thread(self) -> None:
        kwargs = {}
        if self.app.config.linkcheck_timeout:
            kwargs['timeout'] = self.app.config.linkcheck_timeout

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
### 6 - sphinx/builders/linkcheck.py:

Start line: 391, End line: 429

```python
class CheckExternalLinksBuilder(Builder):

    def get_target_uri(self, docname: str, typ: str = None) -> str:
        return ''

    def get_outdated_docs(self) -> Set[str]:
        return self.env.found_docs

    def prepare_writing(self, docnames: Set[str]) -> None:
        return

    def write_doc(self, docname: str, doctree: Node) -> None:
        logger.info('')
        n = 0

        # reference nodes
        for refnode in doctree.traverse(nodes.reference):
            if 'refuri' not in refnode:
                continue
            uri = refnode['refuri']
            lineno = get_node_line(refnode)
            uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
            self.wqueue.put(uri_info, False)
            n += 1

        # image nodes
        for imgnode in doctree.traverse(nodes.image):
            uri = imgnode['candidates'].get('?')
            if uri and '://' in uri:
                lineno = get_node_line(imgnode)
                uri_info = (CHECK_IMMEDIATELY, uri, docname, lineno)
                self.wqueue.put(uri_info, False)
                n += 1

        done = 0
        while done < n:
            self.process_result(self.rqueue.get())
            done += 1

        if self.broken:
            self.app.statuscode = 1
```
### 7 - sphinx/builders/linkcheck.py:

Start line: 431, End line: 445

```python
class CheckExternalLinksBuilder(Builder):

    def write_entry(self, what: str, docname: str, filename: str, line: int,
                    uri: str) -> None:
        with open(path.join(self.outdir, 'output.txt'), 'a') as output:
            output.write("%s:%s: [%s] %s\n" % (filename, line, what, uri))

    def write_linkstat(self, data: dict) -> None:
        with open(path.join(self.outdir, 'output.json'), 'a') as output:
            output.write(json.dumps(data))
            output.write('\n')

    def finish(self) -> None:
        self.wqueue.join()
        # Shutdown threads.
        for worker in self.workers:
            self.wqueue.put((CHECK_IMMEDIATELY, None, None, None), False)
```
### 8 - sphinx/builders/linkcheck.py:

Start line: 338, End line: 389

```python
class CheckExternalLinksBuilder(Builder):

    def process_result(self, result: Tuple[str, str, int, str, str, int]) -> None:
        uri, docname, lineno, status, info, code = result

        filename = self.env.doc2path(docname, None)
        linkstat = dict(filename=filename, lineno=lineno,
                        status=status, code=code, uri=uri,
                        info=info)
        if status == 'unchecked':
            self.write_linkstat(linkstat)
            return
        if status == 'working' and info == 'old':
            self.write_linkstat(linkstat)
            return
        if lineno:
            logger.info('(line %4d) ', lineno, nonl=True)
        if status == 'ignored':
            if info:
                logger.info(darkgray('-ignored- ') + uri + ': ' + info)
            else:
                logger.info(darkgray('-ignored- ') + uri)
            self.write_linkstat(linkstat)
        elif status == 'local':
            logger.info(darkgray('-local-   ') + uri)
            self.write_entry('local', docname, filename, lineno, uri)
            self.write_linkstat(linkstat)
        elif status == 'working':
            logger.info(darkgreen('ok        ') + uri + info)
            self.write_linkstat(linkstat)
        elif status == 'broken':
            if self.app.quiet or self.app.warningiserror:
                logger.warning(__('broken link: %s (%s)'), uri, info,
                               location=(filename, lineno))
            else:
                logger.info(red('broken    ') + uri + red(' - ' + info))
            self.write_entry('broken', docname, filename, lineno, uri + ': ' + info)
            self.write_linkstat(linkstat)
        elif status == 'redirected':
            try:
                text, color = {
                    301: ('permanently', purple),
                    302: ('with Found', purple),
                    303: ('with See Other', purple),
                    307: ('temporarily', turquoise),
                    308: ('permanently', purple),
                }[code]
            except KeyError:
                text, color = ('with unknown code', purple)
            linkstat['text'] = text
            logger.info(color('redirect  ') + uri + color(' - ' + text + ' to ' + info))
            self.write_entry('redirected ' + text, docname, filename,
                             lineno, uri + ' to ' + info)
            self.write_linkstat(linkstat)
```
### 9 - sphinx/builders/linkcheck.py:

Start line: 300, End line: 336

```python
class CheckExternalLinksBuilder(Builder):

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
            max_delay = self.app.config.linkcheck_rate_limit_timeout
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
### 10 - sphinx/builders/linkcheck.py:

Start line: 448, End line: 468

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(CheckExternalLinksBuilder)

    app.add_config_value('linkcheck_ignore', [], None)
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

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 17 - sphinx/builders/linkcheck.py:

Start line: 11, End line: 63

```python
import json
import queue
import re
import socket
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from docutils import nodes
from docutils.nodes import Node
from requests import Response
from requests.exceptions import HTTPError, TooManyRedirects

from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line

logger = logging.getLogger(__name__)

uri_re = re.compile('([a-z]+:)?//')  # matches to foo:// and // (a protocol relative URL)

RateLimit = NamedTuple('RateLimit', (('delay', float), ('next_check', float)))

DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
}
CHECK_IMMEDIATELY = 0
QUEUE_POLL_SECS = 1
DEFAULT_DELAY = 60.0


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
### 26 - sphinx/builders/linkcheck.py:

Start line: 66, End line: 81

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
