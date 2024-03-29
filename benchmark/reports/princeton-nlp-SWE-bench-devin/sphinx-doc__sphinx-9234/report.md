# sphinx-doc__sphinx-9234

| **sphinx-doc/sphinx** | `f0fef96906d80d89e290a780767a92ba85977733` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4954 |
| **Any found context length** | 744 |
| **Avg pos** | 33.0 |
| **Min pos** | 1 |
| **Max pos** | 14 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -272,8 +272,12 @@ def process_result(self, result: CheckResult) -> None:
             except KeyError:
                 text, color = ('with unknown code', purple)
             linkstat['text'] = text
-            logger.info(color('redirect  ') + result.uri +
-                        color(' - ' + text + ' to ' + result.message))
+            if self.config.linkcheck_allowed_redirects:
+                logger.warning('redirect  ' + result.uri + ' - ' + text + ' to ' +
+                               result.message, location=(filename, result.lineno))
+            else:
+                logger.info(color('redirect  ') + result.uri +
+                            color(' - ' + text + ' to ' + result.message))
             self.write_entry('redirected ' + text, result.docname, filename,
                              result.lineno, result.uri + ' to ' + result.message)
         else:
@@ -496,13 +500,23 @@ def check_uri() -> Tuple[str, str, int]:
                 new_url = response.url
                 if anchor:
                     new_url += '#' + anchor
-                # history contains any redirects, get last
-                if response.history:
+
+                if allowed_redirect(req_url, new_url):
+                    return 'working', '', 0
+                elif response.history:
+                    # history contains any redirects, get last
                     code = response.history[-1].status_code
                     return 'redirected', new_url, code
                 else:
                     return 'redirected', new_url, 0
 
+        def allowed_redirect(url: str, new_url: str) -> bool:
+            for from_url, to_url in self.config.linkcheck_allowed_redirects.items():
+                if from_url.match(url) and to_url.match(new_url):
+                    return True
+
+            return False
+
         def check(docname: str) -> Tuple[str, str, int]:
             # check for various conditions without bothering the network
             if len(uri) == 0 or uri.startswith(('#', 'mailto:', 'tel:')):
@@ -667,11 +681,25 @@ def rewrite_github_anchor(app: Sphinx, uri: str) -> Optional[str]:
     return None
 
 
+def compile_linkcheck_allowed_redirects(app: Sphinx, config: Config) -> None:
+    """Compile patterns in linkcheck_allowed_redirects to the regexp objects."""
+    for url, pattern in list(app.config.linkcheck_allowed_redirects.items()):
+        try:
+            app.config.linkcheck_allowed_redirects[re.compile(url)] = re.compile(pattern)
+        except re.error as exc:
+            logger.warning(__('Failed to compile regex in linkcheck_allowed_redirects: %r %s'),
+                           exc.pattern, exc.msg)
+        finally:
+            # Remove the original regexp-string
+            app.config.linkcheck_allowed_redirects.pop(url)
+
+
 def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_builder(CheckExternalLinksBuilder)
     app.add_post_transform(HyperlinkCollector)
 
     app.add_config_value('linkcheck_ignore', [], None)
+    app.add_config_value('linkcheck_allowed_redirects', {}, None)
     app.add_config_value('linkcheck_auth', [], None)
     app.add_config_value('linkcheck_request_headers', {}, None)
     app.add_config_value('linkcheck_retries', 1, None)
@@ -684,6 +712,8 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('linkcheck_rate_limit_timeout', 300.0, None)
 
     app.add_event('linkcheck-process-uri')
+
+    app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=800)
     app.connect('linkcheck-process-uri', rewrite_github_anchor)
 
     return {

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/linkcheck.py | 275 | 276 | 4 | 1 | 1808
| sphinx/builders/linkcheck.py | 499 | 500 | 1 | 1 | 744
| sphinx/builders/linkcheck.py | 670 | 670 | 14 | 1 | 4954
| sphinx/builders/linkcheck.py | 687 | 687 | 14 | 1 | 4954


## Problem Statement

```
Link checker should be able to prohibit unknown redirects
**Is your feature request related to a problem? Please describe.**
A lot of links become stale or move. Good websites will provide redirects to the correct new location or return an HTTP error code. Bad websites will redirect to an unrelated page or the root of the website.

Preventing all redirects does not allow links to URLs like https://www.sphinx-doc.org/ which redirects to https://www.sphinx-doc.org/en/master/. It needs to be possible to allow these redirects but disallow others.

**Describe the solution you'd like**
It should be possible to prohibit unknown redirects by listing all of the allowed redirects as pairs of URLs.

**Describe alternatives you've considered**
Post-process `linkcheck/output.txt` by removing filenames and line numbers then sorting it and comparing it with known good output.

**Additional context**
A link to https://blogs.windows.com/buildingapps/2016/12/02/symlinks-windows-10/ (which used to work) now redirects to https://blogs.windows.com/windowsdeveloper/. Linkcheck allows this but the original link is not valid and needs to be updated to the article's new URL of https://blogs.windows.com/windowsdeveloper/2016/12/02/symlinks-windows-10/.

Linkcheck should be able to report an error for this redirect.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/builders/linkcheck.py** | 413 | 504| 744 | 744 | 5637 | 
| 2 | **1 sphinx/builders/linkcheck.py** | 506 | 541| 358 | 1102 | 5637 | 
| 3 | **1 sphinx/builders/linkcheck.py** | 337 | 357| 175 | 1277 | 5637 | 
| **-> 4 <-** | **1 sphinx/builders/linkcheck.py** | 231 | 280| 531 | 1808 | 5637 | 
| 5 | **1 sphinx/builders/linkcheck.py** | 619 | 652| 275 | 2083 | 5637 | 
| 6 | **1 sphinx/builders/linkcheck.py** | 114 | 132| 170 | 2253 | 5637 | 
| 7 | **1 sphinx/builders/linkcheck.py** | 543 | 578| 331 | 2584 | 5637 | 
| 8 | **1 sphinx/builders/linkcheck.py** | 303 | 335| 278 | 2862 | 5637 | 
| 9 | **1 sphinx/builders/linkcheck.py** | 580 | 616| 308 | 3170 | 5637 | 
| 10 | **1 sphinx/builders/linkcheck.py** | 134 | 229| 727 | 3897 | 5637 | 
| 11 | **1 sphinx/builders/linkcheck.py** | 282 | 300| 203 | 4100 | 5637 | 
| 12 | **1 sphinx/builders/linkcheck.py** | 393 | 411| 142 | 4242 | 5637 | 
| 13 | 2 sphinx/ext/linkcode.py | 11 | 80| 467 | 4709 | 6160 | 
| **-> 14 <-** | **2 sphinx/builders/linkcheck.py** | 670 | 694| 245 | 4954 | 6160 | 
| 15 | **2 sphinx/builders/linkcheck.py** | 360 | 391| 286 | 5240 | 6160 | 
| 16 | **2 sphinx/builders/linkcheck.py** | 655 | 667| 117 | 5357 | 6160 | 
| 17 | **2 sphinx/builders/linkcheck.py** | 11 | 93| 662 | 6019 | 6160 | 
| 18 | 3 sphinx/util/requests.py | 11 | 67| 415 | 6434 | 7108 | 
| 19 | 4 sphinx/ext/extlinks.py | 28 | 89| 615 | 7049 | 7977 | 
| 20 | 5 sphinx/domains/std.py | 1113 | 1138| 209 | 7258 | 18274 | 
| 21 | **5 sphinx/builders/linkcheck.py** | 96 | 111| 156 | 7414 | 18274 | 
| 22 | 6 sphinx/transforms/post_transforms/__init__.py | 171 | 218| 489 | 7903 | 20726 | 
| 23 | 7 sphinx/util/__init__.py | 458 | 494| 306 | 8209 | 25607 | 
| 24 | 8 sphinx/builders/html/__init__.py | 1270 | 1291| 198 | 8407 | 37868 | 
| 25 | 9 sphinx/transforms/references.py | 11 | 54| 266 | 8673 | 38186 | 
| 26 | 10 utils/checks.py | 33 | 109| 545 | 9218 | 39092 | 
| 27 | 10 sphinx/util/__init__.py | 11 | 64| 446 | 9664 | 39092 | 
| 28 | 11 sphinx/writers/texinfo.py | 318 | 344| 259 | 9923 | 51406 | 
| 29 | 12 sphinx/ext/todo.py | 157 | 194| 340 | 10263 | 53246 | 
| 30 | 13 sphinx/writers/html.py | 342 | 360| 233 | 10496 | 60794 | 
| 31 | 13 sphinx/domains/std.py | 11 | 46| 311 | 10807 | 60794 | 
| 32 | 14 sphinx/builders/_epub_base.py | 265 | 296| 281 | 11088 | 67103 | 
| 33 | 15 sphinx/builders/latex/transforms.py | 79 | 100| 214 | 11302 | 71422 | 
| 34 | 16 sphinx/directives/other.py | 9 | 38| 229 | 11531 | 74549 | 
| 35 | 17 sphinx/ext/intersphinx.py | 261 | 340| 864 | 12395 | 78229 | 
| 36 | 17 sphinx/domains/std.py | 630 | 660| 324 | 12719 | 78229 | 
| 37 | 18 sphinx/domains/python.py | 971 | 989| 140 | 12859 | 90274 | 
| 38 | 18 sphinx/domains/python.py | 992 | 1012| 248 | 13107 | 90274 | 
| 39 | 19 sphinx/util/osutil.py | 11 | 45| 214 | 13321 | 91908 | 
| 40 | 19 sphinx/util/__init__.py | 272 | 289| 207 | 13528 | 91908 | 
| 41 | 20 sphinx/domains/javascript.py | 300 | 317| 196 | 13724 | 95972 | 
| 42 | 20 sphinx/domains/std.py | 279 | 289| 130 | 13854 | 95972 | 
| 43 | 21 utils/doclinter.py | 11 | 86| 519 | 14373 | 96543 | 
| 44 | 21 sphinx/builders/_epub_base.py | 298 | 316| 205 | 14578 | 96543 | 
| 45 | 22 sphinx/directives/code.py | 9 | 30| 148 | 14726 | 100394 | 
| 46 | 23 sphinx/directives/patches.py | 9 | 34| 192 | 14918 | 102238 | 
| 47 | 24 sphinx/util/cfamily.py | 63 | 87| 230 | 15148 | 105695 | 
| 48 | 24 sphinx/builders/html/__init__.py | 848 | 875| 246 | 15394 | 105695 | 
| 49 | 24 sphinx/transforms/post_transforms/__init__.py | 62 | 120| 489 | 15883 | 105695 | 
| 50 | 24 sphinx/builders/latex/transforms.py | 102 | 127| 277 | 16160 | 105695 | 
| 51 | 24 sphinx/directives/other.py | 191 | 212| 138 | 16298 | 105695 | 
| 52 | 25 sphinx/util/nodes.py | 179 | 236| 491 | 16789 | 111180 | 
| 53 | 26 sphinx/testing/comparer.py | 10 | 48| 267 | 17056 | 111893 | 
| 54 | 27 sphinx/transforms/__init__.py | 251 | 271| 192 | 17248 | 115064 | 
| 55 | 28 sphinx/transforms/i18n.py | 227 | 386| 1673 | 18921 | 119662 | 
| 56 | 29 sphinx/config.py | 407 | 459| 474 | 19395 | 124118 | 
| 57 | 29 sphinx/builders/html/__init__.py | 1230 | 1240| 147 | 19542 | 124118 | 
| 58 | 30 sphinx/transforms/compact_bullet_list.py | 11 | 52| 287 | 19829 | 124733 | 
| 59 | 31 sphinx/testing/util.py | 10 | 45| 270 | 20099 | 126465 | 
| 60 | 31 sphinx/util/osutil.py | 214 | 235| 149 | 20248 | 126465 | 
| 61 | 31 sphinx/domains/std.py | 125 | 176| 457 | 20705 | 126465 | 
| 62 | 31 sphinx/testing/comparer.py | 69 | 104| 282 | 20987 | 126465 | 
| 63 | 31 sphinx/domains/std.py | 852 | 913| 553 | 21540 | 126465 | 
| 64 | 31 sphinx/domains/std.py | 915 | 923| 122 | 21662 | 126465 | 
| 65 | 31 sphinx/config.py | 462 | 481| 223 | 21885 | 126465 | 
| 66 | 32 sphinx/util/smartypants.py | 28 | 127| 1450 | 23335 | 130612 | 
| 67 | 33 sphinx/io.py | 10 | 39| 234 | 23569 | 132017 | 
| 68 | 34 sphinx/application.py | 1218 | 1251| 315 | 23884 | 143631 | 
| 69 | 34 sphinx/directives/other.py | 86 | 154| 646 | 24530 | 143631 | 
| 70 | 35 sphinx/cmd/quickstart.py | 11 | 119| 756 | 25286 | 149200 | 
| 71 | 36 sphinx/ext/viewcode.py | 160 | 195| 287 | 25573 | 152295 | 
| 72 | 37 sphinx/writers/html5.py | 238 | 259| 208 | 25781 | 159375 | 
| 73 | 37 sphinx/transforms/post_transforms/__init__.py | 122 | 169| 573 | 26354 | 159375 | 
| 74 | 37 sphinx/domains/python.py | 11 | 80| 518 | 26872 | 159375 | 
| 75 | 38 sphinx/util/pycompat.py | 11 | 46| 328 | 27200 | 159868 | 
| 76 | 38 sphinx/util/requests.py | 70 | 89| 150 | 27350 | 159868 | 
| 77 | 39 sphinx/ext/graphviz.py | 12 | 44| 243 | 27593 | 163604 | 
| 78 | 39 sphinx/builders/_epub_base.py | 339 | 364| 301 | 27894 | 163604 | 
| 79 | 39 sphinx/directives/code.py | 407 | 470| 642 | 28536 | 163604 | 
| 80 | 39 sphinx/builders/html/__init__.py | 11 | 62| 432 | 28968 | 163604 | 
| 81 | 39 sphinx/transforms/__init__.py | 11 | 44| 231 | 29199 | 163604 | 
| 82 | 39 sphinx/directives/code.py | 372 | 405| 288 | 29487 | 163604 | 
| 83 | 39 sphinx/builders/html/__init__.py | 1205 | 1227| 242 | 29729 | 163604 | 
| 84 | 39 sphinx/ext/intersphinx.py | 127 | 149| 174 | 29903 | 163604 | 
| 85 | 39 sphinx/builders/html/__init__.py | 1243 | 1267| 198 | 30101 | 163604 | 
| 86 | 39 sphinx/writers/html5.py | 313 | 366| 484 | 30585 | 163604 | 
| 87 | 40 sphinx/domains/rst.py | 140 | 171| 356 | 30941 | 166083 | 
| 88 | 41 sphinx/registry.py | 11 | 51| 314 | 31255 | 170723 | 
| 89 | 42 sphinx/environment/__init__.py | 619 | 639| 178 | 31433 | 176220 | 
| 90 | 43 sphinx/parsers.py | 11 | 26| 113 | 31546 | 177066 | 
| 91 | 43 sphinx/util/requests.py | 119 | 133| 127 | 31673 | 177066 | 
| 92 | 43 sphinx/writers/html5.py | 653 | 739| 686 | 32359 | 177066 | 
| 93 | 43 sphinx/writers/html5.py | 517 | 546| 253 | 32612 | 177066 | 
| 94 | 43 sphinx/builders/latex/transforms.py | 52 | 77| 165 | 32777 | 177066 | 
| 95 | 43 sphinx/util/osutil.py | 71 | 171| 718 | 33495 | 177066 | 
| 96 | 43 sphinx/builders/latex/transforms.py | 129 | 153| 211 | 33706 | 177066 | 
| 97 | 43 sphinx/util/cfamily.py | 11 | 62| 749 | 34455 | 177066 | 
| 98 | 43 sphinx/domains/std.py | 834 | 850| 189 | 34644 | 177066 | 
| 99 | 43 sphinx/transforms/compact_bullet_list.py | 55 | 95| 270 | 34914 | 177066 | 
| 100 | 43 sphinx/domains/python.py | 283 | 317| 382 | 35296 | 177066 | 
| 101 | 43 sphinx/writers/html5.py | 210 | 236| 319 | 35615 | 177066 | 
| 102 | 44 doc/conf.py | 83 | 138| 476 | 36091 | 178530 | 
| 103 | 44 sphinx/builders/html/__init__.py | 362 | 406| 347 | 36438 | 178530 | 
| 104 | 45 sphinx/directives/__init__.py | 11 | 47| 253 | 36691 | 180782 | 
| 105 | 45 sphinx/writers/texinfo.py | 685 | 746| 606 | 37297 | 180782 | 
| 106 | 45 sphinx/writers/html.py | 267 | 289| 214 | 37511 | 180782 | 
| 107 | 46 sphinx/domains/changeset.py | 49 | 107| 516 | 38027 | 182036 | 
| 108 | 46 sphinx/transforms/__init__.py | 182 | 212| 229 | 38256 | 182036 | 
| 109 | 47 sphinx/writers/text.py | 946 | 1059| 873 | 39129 | 191037 | 
| 110 | 48 sphinx/ext/apidoc.py | 172 | 200| 249 | 39378 | 195267 | 
| 111 | 49 sphinx/ext/githubpages.py | 11 | 37| 227 | 39605 | 195554 | 
| 112 | 50 sphinx/builders/singlehtml.py | 54 | 66| 141 | 39746 | 197350 | 
| 113 | 50 sphinx/domains/python.py | 1374 | 1396| 211 | 39957 | 197350 | 


### Hint

```
If one can tell in advance where they are redirected, might as well use the direct link in the docs and skip the redirect.
Perhaps a step forward would be a new setting to treat redirects as errors?
I provided a reason why I want to be able to link to a redirect, unless you think the base URL of sphinx itself should not be linkable?
I misread the issue originally, I was hoping all redirects could be replaced by the final version of the URL, but that’s not true.
In the provided example, sphinx-doc.org could redirect to a different language based on your language preferences. Replacing the link with the final version would force users to visit the English version of the page.

What do you think of a mapping in the config: `{"original_URL": final_url}`, perhaps named `linkcheck_validate_redirects`?
The behavior upon redirect would be:
- original URL present in the mapping, verify the final URL matches the value from `linkcheck_validate_redirects`,
- original URL not present, mark link as broken.
For the sphinx-doc.org case I would not expect to specify the exact final URL because I don't care where it redirects to when I link to `/`. (It may decide that CI in another country should get a different language by default.)

If `final_url` could be `None` to allow any final URL, that would appear to work but I'd really want it to redirect within the same domain. If https://www.sphinx-doc.org/ redirects to https://this-domain-is-for-sale.example.com/sphinx-doc.org then the link is broken.

So `final_url` could be `None`, a string or a regex.

\`\`\`
{"https://www.sphinx-doc.org/": None}
\`\`\`

\`\`\`
{"https://www.sphinx-doc.org/": "https://www\.sphinx-doc\.org/en/master/"}
\`\`\`

\`\`\`
import re
{"https://www.sphinx-doc.org/": re.compile(r"^https://www\.sphinx-doc\.org/.*$")}
\`\`\`

Of course, when you start allowing regex in the `final_url`you might want to allow regex in the `original_url` and group references:

\`\`\`
import re
{re.compile("^https://sphinx-doc.org/(.*)$"): re.compile(r"^https://(www\.)?sphinx-doc\.org/\1$")}
\`\`\`

There may be multiple conflicting mappings, if any one of them matches then the link is ok.
This is something I have just come across myself, and such a setting would be helpful to ignore the fact that a redirect happened - in other words, set the state as "working" instead of "redirected" as long as the target page is available.

Another example of a case where this would be helpful is wanting to ignore redirects in the case of documentation versions, e.g. `.../en/stable/` → `.../en/3.2/`. In this case it is preferable to always link to the latest/stable version via a URL rewrite.

I could see a configuration along the following lines (very much what @nomis has specified above):

\`\`\`python
# Check that the link is "working" but don't flag as "redirected" unless the target doesn't match.
linkcheck_redirects_ignore = {
    r'^https://([^/?#]+)/$': r'^https://\1/(?:home|index)\.html?$',
    r'^https://(nodejs\.org)/$', r'^https://\1/[-a-z]+/$',
    r'^https://(pip\.pypa\.io)/$', r'^https://\1/[-a-z]+/stable/$',
    r'^https://(www\.sphinx-doc\.org)/$', r'^https://\1/[-a-z]+/master/$',
    r'^https://(pytest\.org)/$', r'^https://docs\.\1/[-a-z]+/\d+\.\d+\.x/$',
    r'^https://github.com/([^/?#]+)/([^/?#])+/blob/(.*)$': r'https://github.com/\1/\2/tree/\3$',
    r'^https://([^/?#\.]+)\.readthedocs\.io/$': r'^https://\1\.readthedocs\.io/[-a-z]+/(?:master|latest|stable)/$',
    r'^https://dev\.mysql\.com/doc/refman/': r'^https://dev\.mysql\.com/doc/refman/\d+\.\d+/',
    r'^https://docs\.djangoproject\.com/': r'^https://docs\.djangoproject\.com/[-a-z]+/\d+\.\d+/',
    r'^https://docs\.djangoproject\.com/([-a-z]+)/stable/': r'^https://docs\.djangoproject\.com/\1/\d+\.\d+/',
}
\`\`\`
```

## Patch

```diff
diff --git a/sphinx/builders/linkcheck.py b/sphinx/builders/linkcheck.py
--- a/sphinx/builders/linkcheck.py
+++ b/sphinx/builders/linkcheck.py
@@ -272,8 +272,12 @@ def process_result(self, result: CheckResult) -> None:
             except KeyError:
                 text, color = ('with unknown code', purple)
             linkstat['text'] = text
-            logger.info(color('redirect  ') + result.uri +
-                        color(' - ' + text + ' to ' + result.message))
+            if self.config.linkcheck_allowed_redirects:
+                logger.warning('redirect  ' + result.uri + ' - ' + text + ' to ' +
+                               result.message, location=(filename, result.lineno))
+            else:
+                logger.info(color('redirect  ') + result.uri +
+                            color(' - ' + text + ' to ' + result.message))
             self.write_entry('redirected ' + text, result.docname, filename,
                              result.lineno, result.uri + ' to ' + result.message)
         else:
@@ -496,13 +500,23 @@ def check_uri() -> Tuple[str, str, int]:
                 new_url = response.url
                 if anchor:
                     new_url += '#' + anchor
-                # history contains any redirects, get last
-                if response.history:
+
+                if allowed_redirect(req_url, new_url):
+                    return 'working', '', 0
+                elif response.history:
+                    # history contains any redirects, get last
                     code = response.history[-1].status_code
                     return 'redirected', new_url, code
                 else:
                     return 'redirected', new_url, 0
 
+        def allowed_redirect(url: str, new_url: str) -> bool:
+            for from_url, to_url in self.config.linkcheck_allowed_redirects.items():
+                if from_url.match(url) and to_url.match(new_url):
+                    return True
+
+            return False
+
         def check(docname: str) -> Tuple[str, str, int]:
             # check for various conditions without bothering the network
             if len(uri) == 0 or uri.startswith(('#', 'mailto:', 'tel:')):
@@ -667,11 +681,25 @@ def rewrite_github_anchor(app: Sphinx, uri: str) -> Optional[str]:
     return None
 
 
+def compile_linkcheck_allowed_redirects(app: Sphinx, config: Config) -> None:
+    """Compile patterns in linkcheck_allowed_redirects to the regexp objects."""
+    for url, pattern in list(app.config.linkcheck_allowed_redirects.items()):
+        try:
+            app.config.linkcheck_allowed_redirects[re.compile(url)] = re.compile(pattern)
+        except re.error as exc:
+            logger.warning(__('Failed to compile regex in linkcheck_allowed_redirects: %r %s'),
+                           exc.pattern, exc.msg)
+        finally:
+            # Remove the original regexp-string
+            app.config.linkcheck_allowed_redirects.pop(url)
+
+
 def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_builder(CheckExternalLinksBuilder)
     app.add_post_transform(HyperlinkCollector)
 
     app.add_config_value('linkcheck_ignore', [], None)
+    app.add_config_value('linkcheck_allowed_redirects', {}, None)
     app.add_config_value('linkcheck_auth', [], None)
     app.add_config_value('linkcheck_request_headers', {}, None)
     app.add_config_value('linkcheck_retries', 1, None)
@@ -684,6 +712,8 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('linkcheck_rate_limit_timeout', 300.0, None)
 
     app.add_event('linkcheck-process-uri')
+
+    app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=800)
     app.connect('linkcheck-process-uri', rewrite_github_anchor)
 
     return {

```

## Test Patch

```diff
diff --git a/tests/roots/test-linkcheck-localserver-warn-redirects/conf.py b/tests/roots/test-linkcheck-localserver-warn-redirects/conf.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver-warn-redirects/conf.py
@@ -0,0 +1 @@
+exclude_patterns = ['_build']
diff --git a/tests/roots/test-linkcheck-localserver-warn-redirects/index.rst b/tests/roots/test-linkcheck-localserver-warn-redirects/index.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver-warn-redirects/index.rst
@@ -0,0 +1,2 @@
+`local server1 <http://localhost:7777/path1>`_
+`local server2 <http://localhost:7777/path2>`_
diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -23,6 +23,7 @@
 import requests
 
 from sphinx.builders.linkcheck import HyperlinkAvailabilityCheckWorker, RateLimit
+from sphinx.testing.util import strip_escseq
 from sphinx.util.console import strip_colors
 
 from .utils import CERT_FILE, http_server, https_server
@@ -254,7 +255,7 @@ def log_date_time_string(self):
 
 
 @pytest.mark.sphinx('linkcheck', testroot='linkcheck-localserver', freshenv=True)
-def test_follows_redirects_on_HEAD(app, capsys):
+def test_follows_redirects_on_HEAD(app, capsys, warning):
     with http_server(make_redirect_handler(support_head=True)):
         app.build()
     stdout, stderr = capsys.readouterr()
@@ -269,10 +270,11 @@ def test_follows_redirects_on_HEAD(app, capsys):
         127.0.0.1 - - [] "HEAD /?redirected=1 HTTP/1.1" 204 -
         """
     )
+    assert warning.getvalue() == ''
 
 
 @pytest.mark.sphinx('linkcheck', testroot='linkcheck-localserver', freshenv=True)
-def test_follows_redirects_on_GET(app, capsys):
+def test_follows_redirects_on_GET(app, capsys, warning):
     with http_server(make_redirect_handler(support_head=False)):
         app.build()
     stdout, stderr = capsys.readouterr()
@@ -288,6 +290,28 @@ def test_follows_redirects_on_GET(app, capsys):
         127.0.0.1 - - [] "GET /?redirected=1 HTTP/1.1" 204 -
         """
     )
+    assert warning.getvalue() == ''
+
+
+@pytest.mark.sphinx('linkcheck', testroot='linkcheck-localserver-warn-redirects',
+                    freshenv=True, confoverrides={
+                        'linkcheck_allowed_redirects': {'http://localhost:7777/.*1': '.*'}
+                    })
+def test_linkcheck_allowed_redirects(app, warning):
+    with http_server(make_redirect_handler(support_head=False)):
+        app.build()
+
+    with open(app.outdir / 'output.json') as fp:
+        records = [json.loads(l) for l in fp.readlines()]
+
+    assert len(records) == 2
+    result = {r["uri"]: r["status"] for r in records}
+    assert result["http://localhost:7777/path1"] == "working"
+    assert result["http://localhost:7777/path2"] == "redirected"
+
+    assert ("index.rst.rst:1: WARNING: redirect  http://localhost:7777/path2 - with Found to "
+            "http://localhost:7777/?redirected=1\n" in strip_escseq(warning.getvalue()))
+    assert len(warning.getvalue().splitlines()) == 1
 
 
 class OKHandler(http.server.BaseHTTPRequestHandler):

```


## Code snippets

### 1 - sphinx/builders/linkcheck.py:

Start line: 413, End line: 504

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
                # history contains any redirects, get last
                if response.history:
                    code = response.history[-1].status_code
                    return 'redirected', new_url, code
                else:
                    return 'redirected', new_url, 0
        # ... other code
```
### 2 - sphinx/builders/linkcheck.py:

Start line: 506, End line: 541

```python
class HyperlinkAvailabilityCheckWorker(Thread):

    def run(self) -> None:
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

Start line: 337, End line: 357

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

Start line: 231, End line: 280

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
            logger.info(color('redirect  ') + result.uri +
                        color(' - ' + text + ' to ' + result.message))
            self.write_entry('redirected ' + text, result.docname, filename,
                             result.lineno, result.uri + ' to ' + result.message)
        else:
            raise ValueError("Unknown status %s." % result.status)
```
### 5 - sphinx/builders/linkcheck.py:

Start line: 619, End line: 652

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
### 6 - sphinx/builders/linkcheck.py:

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
### 7 - sphinx/builders/linkcheck.py:

Start line: 543, End line: 578

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

Start line: 303, End line: 335

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
### 9 - sphinx/builders/linkcheck.py:

Start line: 580, End line: 616

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
### 10 - sphinx/builders/linkcheck.py:

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
### 11 - sphinx/builders/linkcheck.py:

Start line: 282, End line: 300

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
### 12 - sphinx/builders/linkcheck.py:

Start line: 393, End line: 411

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
### 14 - sphinx/builders/linkcheck.py:

Start line: 670, End line: 694

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(CheckExternalLinksBuilder)
    app.add_post_transform(HyperlinkCollector)

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

    app.add_event('linkcheck-process-uri')
    app.connect('linkcheck-process-uri', rewrite_github_anchor)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 15 - sphinx/builders/linkcheck.py:

Start line: 360, End line: 391

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
### 16 - sphinx/builders/linkcheck.py:

Start line: 655, End line: 667

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
### 21 - sphinx/builders/linkcheck.py:

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
