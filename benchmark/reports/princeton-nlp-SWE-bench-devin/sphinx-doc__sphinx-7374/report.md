# sphinx-doc__sphinx-7374

| **sphinx-doc/sphinx** | `70c61e44c34b4dadf1a7552be7c5feabd74b98bc` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 4612 |
| **Any found context length** | 4612 |
| **Avg pos** | 15.0 |
| **Min pos** | 15 |
| **Max pos** | 15 |
| **Top file pos** | 6 |
| **Missing snippets** | 5 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/builders/_epub_base.py b/sphinx/builders/_epub_base.py
--- a/sphinx/builders/_epub_base.py
+++ b/sphinx/builders/_epub_base.py
@@ -259,6 +259,15 @@ def fix_ids(self, tree: nodes.document) -> None:
         Some readers crash because they interpret the part as a
         transport protocol specification.
         """
+        def update_node_id(node: Element) -> None:
+            """Update IDs of given *node*."""
+            new_ids = []
+            for node_id in node['ids']:
+                new_id = self.fix_fragment('', node_id)
+                if new_id not in new_ids:
+                    new_ids.append(new_id)
+            node['ids'] = new_ids
+
         for reference in tree.traverse(nodes.reference):
             if 'refuri' in reference:
                 m = self.refuri_re.match(reference['refuri'])
@@ -268,22 +277,14 @@ def fix_ids(self, tree: nodes.document) -> None:
                 reference['refid'] = self.fix_fragment('', reference['refid'])
 
         for target in tree.traverse(nodes.target):
-            for i, node_id in enumerate(target['ids']):
-                if ':' in node_id:
-                    target['ids'][i] = self.fix_fragment('', node_id)
+            update_node_id(target)
 
             next_node = target.next_node(ascend=True)  # type: Node
             if isinstance(next_node, nodes.Element):
-                for i, node_id in enumerate(next_node['ids']):
-                    if ':' in node_id:
-                        next_node['ids'][i] = self.fix_fragment('', node_id)
+                update_node_id(next_node)
 
         for desc_signature in tree.traverse(addnodes.desc_signature):
-            ids = desc_signature.attributes['ids']
-            newids = []
-            for id in ids:
-                newids.append(self.fix_fragment('', id))
-            desc_signature.attributes['ids'] = newids
+            update_node_id(desc_signature)
 
     def add_visible_links(self, tree: nodes.document, show_urls: str = 'inline') -> None:
         """Add visible link targets for external links"""
diff --git a/sphinx/util/nodes.py b/sphinx/util/nodes.py
--- a/sphinx/util/nodes.py
+++ b/sphinx/util/nodes.py
@@ -445,6 +445,7 @@ def _make_id(string: str) -> str:
 
     Changes:
 
+    * Allow to use capital alphabet characters
     * Allow to use dots (".") and underscores ("_") for an identifier
       without a leading character.
 
@@ -452,8 +453,7 @@ def _make_id(string: str) -> str:
     # Maintainer: docutils-develop@lists.sourceforge.net
     # Copyright: This module has been placed in the public domain.
     """
-    id = string.lower()
-    id = id.translate(_non_id_translate_digraphs)
+    id = string.translate(_non_id_translate_digraphs)
     id = id.translate(_non_id_translate)
     # get rid of non-ascii characters.
     # 'ascii' lowercase to prevent problems with turkish locale.
@@ -464,7 +464,7 @@ def _make_id(string: str) -> str:
     return str(id)
 
 
-_non_id_chars = re.compile('[^a-z0-9._]+')
+_non_id_chars = re.compile('[^a-zA-Z0-9._]+')
 _non_id_at_ends = re.compile('^[-0-9._]+|-+$')
 _non_id_translate = {
     0x00f8: u'o',       # o with stroke

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/_epub_base.py | 262 | 262 | 15 | 6 | 4612
| sphinx/builders/_epub_base.py | 271 | 286 | 15 | 6 | 4612
| sphinx/util/nodes.py | 448 | 448 | - | - | -
| sphinx/util/nodes.py | 455 | 456 | - | - | -
| sphinx/util/nodes.py | 467 | 467 | - | - | -


## Problem Statement

```
Breaking change to Python domain IDs
**Describe the bug**

Previously, anchors for Python functions were using underscores, #7236 changed this to dashes.

**To Reproduce**

Document some Python function whose name contains underscores:

\`\`\`rst
.. py:function:: example_python_function(foo)

    Some function.
\`\`\`

**Expected behavior**

This used to create a fragment identifier `#example_python_function` , but since #7236 this creates `#example-python-function`.

**Your project**

This breaks links to python functions when used with `nbsphinx`: https://nbsphinx.readthedocs.io/en/0.5.1/markdown-cells.html#Links-to-Domain-Objects

Apart from that all links (containing underscores) from external sites to Python API docs created by Sphinx (which I guess are a lot) will break!

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/napoleon/__init__.py | 312 | 328| 158 | 158 | 3763 | 
| 2 | 2 sphinx/domains/python.py | 221 | 239| 214 | 372 | 15209 | 
| 3 | 2 sphinx/domains/python.py | 11 | 68| 446 | 818 | 15209 | 
| 4 | 2 sphinx/domains/python.py | 1273 | 1284| 127 | 945 | 15209 | 
| 5 | 3 sphinx/directives/__init__.py | 259 | 297| 273 | 1218 | 17632 | 
| 6 | 3 sphinx/domains/python.py | 966 | 986| 248 | 1466 | 17632 | 
| 7 | 3 sphinx/domains/python.py | 882 | 942| 521 | 1987 | 17632 | 
| 8 | 3 sphinx/domains/python.py | 1227 | 1251| 283 | 2270 | 17632 | 
| 9 | 3 sphinx/domains/python.py | 945 | 963| 143 | 2413 | 17632 | 
| 10 | 4 sphinx/domains/std.py | 11 | 48| 323 | 2736 | 27584 | 
| 11 | 5 sphinx/util/__init__.py | 11 | 75| 505 | 3241 | 33361 | 
| 12 | 5 sphinx/domains/python.py | 1142 | 1169| 326 | 3567 | 33361 | 
| 13 | 5 sphinx/domains/python.py | 1077 | 1128| 526 | 4093 | 33361 | 
| 14 | 5 sphinx/domains/python.py | 1253 | 1271| 220 | 4313 | 33361 | 
| **-> 15 <-** | **6 sphinx/builders/_epub_base.py** | 256 | 286| 299 | 4612 | 40096 | 
| 16 | 6 sphinx/domains/python.py | 1286 | 1300| 192 | 4804 | 40096 | 
| 17 | 6 sphinx/domains/std.py | 1045 | 1081| 323 | 5127 | 40096 | 
| 18 | 7 sphinx/domains/javascript.py | 11 | 34| 190 | 5317 | 44111 | 
| 19 | 8 sphinx/util/docutils.py | 174 | 198| 187 | 5504 | 48231 | 
| 20 | 9 sphinx/transforms/references.py | 11 | 68| 362 | 5866 | 48645 | 
| 21 | 9 sphinx/domains/python.py | 537 | 570| 293 | 6159 | 48645 | 
| 22 | 9 sphinx/domains/python.py | 524 | 534| 145 | 6304 | 48645 | 
| 23 | 9 sphinx/domains/python.py | 1130 | 1140| 132 | 6436 | 48645 | 
| 24 | 9 sphinx/domains/std.py | 877 | 885| 120 | 6556 | 48645 | 
| 25 | 10 sphinx/ext/linkcode.py | 11 | 80| 470 | 7026 | 49171 | 
| 26 | 10 sphinx/domains/python.py | 845 | 857| 132 | 7158 | 49171 | 
| 27 | 11 sphinx/ext/napoleon/docstring.py | 13 | 38| 311 | 7469 | 57965 | 
| 28 | 11 sphinx/domains/std.py | 243 | 260| 125 | 7594 | 57965 | 
| 29 | 11 sphinx/domains/std.py | 775 | 795| 244 | 7838 | 57965 | 
| 30 | 11 sphinx/domains/python.py | 264 | 276| 140 | 7978 | 57965 | 
| 31 | 12 sphinx/transforms/__init__.py | 11 | 44| 226 | 8204 | 61112 | 
| 32 | 12 sphinx/domains/python.py | 668 | 722| 530 | 8734 | 61112 | 
| 33 | 12 sphinx/domains/std.py | 1022 | 1043| 232 | 8966 | 61112 | 
| 34 | 13 sphinx/domains/rst.py | 11 | 33| 161 | 9127 | 63582 | 
| 35 | 13 sphinx/domains/javascript.py | 256 | 292| 391 | 9518 | 63582 | 
| 36 | 14 sphinx/util/cfamily.py | 11 | 81| 732 | 10250 | 65592 | 
| 37 | 14 sphinx/domains/python.py | 241 | 261| 219 | 10469 | 65592 | 
| 38 | 14 sphinx/domains/python.py | 761 | 780| 227 | 10696 | 65592 | 
| 39 | 14 sphinx/domains/std.py | 535 | 605| 726 | 11422 | 65592 | 
| 40 | 15 sphinx/builders/changes.py | 11 | 28| 127 | 11549 | 67117 | 
| 41 | 16 sphinx/directives/other.py | 9 | 40| 238 | 11787 | 70274 | 
| 42 | 17 sphinx/builders/linkcheck.py | 107 | 189| 652 | 12439 | 73171 | 
| 43 | 17 sphinx/domains/std.py | 815 | 875| 546 | 12985 | 73171 | 
| 44 | 18 sphinx/application.py | 13 | 60| 378 | 13363 | 83793 | 
| 45 | 18 sphinx/domains/std.py | 68 | 99| 354 | 13717 | 83793 | 
| 46 | 19 sphinx/highlighting.py | 11 | 54| 379 | 14096 | 85101 | 
| 47 | 20 sphinx/domains/__init__.py | 12 | 30| 140 | 14236 | 88612 | 
| 48 | 21 sphinx/registry.py | 11 | 50| 307 | 14543 | 93102 | 
| 49 | 22 doc/conf.py | 81 | 140| 513 | 15056 | 94638 | 
| 50 | 23 sphinx/domains/math.py | 143 | 175| 308 | 15364 | 96138 | 
| 51 | 23 sphinx/domains/rst.py | 249 | 258| 134 | 15498 | 96138 | 
| 52 | 24 sphinx/domains/index.py | 11 | 30| 127 | 15625 | 97072 | 
| 53 | 25 sphinx/util/pycompat.py | 11 | 58| 364 | 15989 | 97794 | 
| 54 | 25 sphinx/domains/math.py | 11 | 40| 207 | 16196 | 97794 | 
| 55 | 26 sphinx/domains/c.py | 2667 | 3411| 6430 | 22626 | 125530 | 
| 56 | 26 sphinx/domains/rst.py | 230 | 236| 113 | 22739 | 125530 | 
| 57 | 27 sphinx/util/smartypants.py | 376 | 388| 137 | 22876 | 129641 | 
| 58 | 28 sphinx/builders/html/__init__.py | 11 | 67| 457 | 23333 | 140815 | 
| 59 | 29 sphinx/builders/latex/transforms.py | 91 | 116| 277 | 23610 | 144994 | 
| 60 | 30 sphinx/environment/__init__.py | 11 | 82| 489 | 24099 | 150817 | 
| 61 | 31 sphinx/writers/latex.py | 2124 | 2145| 256 | 24355 | 170479 | 
| 62 | 32 sphinx/io.py | 10 | 46| 274 | 24629 | 172170 | 
| 63 | 33 sphinx/transforms/post_transforms/code.py | 87 | 112| 235 | 24864 | 173198 | 
| 64 | 34 sphinx/domains/changeset.py | 11 | 45| 214 | 25078 | 174383 | 
| 65 | 35 sphinx/ext/extlinks.py | 26 | 71| 400 | 25478 | 175002 | 
| 66 | 35 sphinx/domains/rst.py | 260 | 286| 274 | 25752 | 175002 | 
| 67 | 35 sphinx/domains/javascript.py | 295 | 312| 196 | 25948 | 175002 | 
| 68 | 35 sphinx/domains/std.py | 678 | 691| 157 | 26105 | 175002 | 
| 69 | 35 sphinx/highlighting.py | 57 | 166| 876 | 26981 | 175002 | 
| 70 | 35 sphinx/domains/python.py | 336 | 418| 700 | 27681 | 175002 | 
| 71 | 36 sphinx/transforms/compact_bullet_list.py | 56 | 96| 270 | 27951 | 175623 | 


## Missing Patch Files

 * 1: sphinx/builders/_epub_base.py
 * 2: sphinx/util/nodes.py

### Hint

```
Are you sure the old links are broken? While the permalink is indeed using a new ID generation scheme, the old ID should still be attached to the declaration (using ``<span id="theOldID"></span>``).
Yes, I changed the style of node_ids in #7236. Therefore, the main hyperlink anchor will be changed in the next release. But old-styled node_ids are still available. So old hyperlinks are still working.

There are some reasons why I changed them. First is that their naming is very simple and getting conflicted often (refs: #6903). Second is the rule of naming is against docutils specification. Last is that it allows sharing one node_ids to multiple names. For example, it helps to represent `io.StringIO` and `_io.StringIO` are the same (or they have the same ID).

To improve both python domain and autodoc, we have to change the structure of the domain and the rule of naming IDs. I don't know the change is harmful. But it is needed to improve Sphinx, I believe.
Thanks for the quick responses!

> Are you sure the old links are broken?

I thought so, but links from external sites seem to be *not yet* broken.

However, it looks like they will be broken at some point in the future, according to this comment:

https://github.com/sphinx-doc/sphinx/blob/f85b870ad59f39c8637160a4cd4d865ce1e1628e/sphinx/domains/python.py#L367-L370

Whether that happens sooner or later, it will be quite bad.

But that's not actually the situation where I found the problem. The ID change is breaking my Sphinx extension `nbsphinx` because the code assumed that underscores are not changed to dashes:

https://github.com/spatialaudio/nbsphinx/blob/5be6da7b212e0cfed34ebd7da0ede5501549571d/src/nbsphinx.py#L1446-L1467

This causes a build warning (and a missing link) when running Sphinx.
You can reproduce this by building the `nbsphinx` docs, see https://nbsphinx.readthedocs.io/en/0.5.1/contributing.html.

I could of course change the implementation of `nbsphinx` (depending on the Sphinx version), but that would still break all the API links my users have made in their notebooks!

And it would break them depending on which Sphinx version they are using, wouldn't that be horrible?

> First is that their naming is very simple and getting conflicted often (refs: #6903).

How does changing `#example_python_function` to `#example-python-function` help in this case?

> Second is the rule of naming is against docutils specification.

Because underscores are not allowed?

I think it would be worth violating the specification for that.
I don't see any negative consequences to this.
And it has worked fine for many years.

Also, having dots and underscores in link to API docs just looks so much more sensible!

Here's an example: https://sfs-python.readthedocs.io/en/0.5.0/sfs.fd.source.html#sfs.fd.source.point_velocity

The link contains the correct Python function name: `#sfs.fd.source.point_velocity`.

When the ID is changed, this becomes: `#sfs-fd-source-point-velocity`, which doesn't really make any sense anymore.

> Last is that it allows sharing one node_ids to multiple names. For example, it helps to represent `io.StringIO` and `_io.StringIO` are the same (or they have the same ID).

I don't understand. Are `io` and `_io` not different names in Python?

And what does that have to do with changing underscores to dashes?
>>First is that their naming is very simple and getting conflicted often (refs: #6903).
>
>How does changing #example_python_function to #example-python-function help in this case?

No, this change is not only replacing underscores by hyphens. New ID Generator tries to generate node_id by following steps; 1) Generate node_id by the given string. But generated one is already used in the document,  2) Generate node_id by sequence number like `id0`.

It means the node_id is not guess-able by its name. Indeed, it would be almost working fine if we use hyphens and dots for the ID generation. But it will be sometimes broken.

>>Second is the rule of naming is against docutils specification.
>
>Because underscores are not allowed?
>
>I think it would be worth violating the specification for that.
>I don't see any negative consequences to this.
>And it has worked fine for many years.

Yes, docutils' spec does not allow to use hyphens and dots in node_id.

I know the current rule for node_id generation is not so wrong. But it surely contains problems. Have you ever try to use invalid characters to the signature? How about multibyte characters?

For example, this is an attacking code for the ID generator:
\`\`\`
.. py:function:: "><script>alert('hello sphinx')</script>
\`\`\`

I know this is a very mean example and not related to hyphens' problem directly. But our code and docutils do not expect to pass malicious characters as a node_id. I suppose dots and hyphens may not harm our code. But we need to investigate all of our code to prove the safety.

>>Last is that it allows sharing one node_ids to multiple names. For example, it helps to represent io.StringIO and _io.StringIO are the same (or they have the same ID).
>
>I don't understand. Are io and _io not different names in Python?
>
>And what does that have to do with changing underscores to dashes?

Indeed, `io` and `_io` are different names in python interpreter. But please read the python-doc. The latter one is not documented in it. We have some issues to document a python object as "public name" instead of "canonical name". see #4065. It is one of them. This feature is not implemented yet. But I'll do that in the (nearly) future. It tells us the real name of the living object does not match the documentation of it.

As you know, it is not related to hyphens problem. It also conflicts with the hyperlinks which human builds manually. It's no longer guess-able. If we'll keep using dots and hyphens for node_id, the cross-reference feature is needed to create references for nbsphinx, I think.
> > How does changing #example_python_function to #example-python-function help in this case?
> 
> No, this change is not only replacing underscores by hyphens. New ID Generator tries to generate node_id by following steps; 1) Generate node_id by the given string. But generated one is already used in the document, 2) Generate node_id by sequence number like `id0`.

OK, that sounds great. So what about doing that, but also allow underscores (`_`) and dots (`.`)?

> I know the current rule for node_id generation is not so wrong. But it surely contains problems. Have you ever try to use invalid characters to the signature? How about multibyte characters?

OK, I understand that it might be problematic to allow arbitrary characters/code points.

But what about just adding `_` and `.` to the allowed characters?

> For example, this is an attacking code for the ID generator:
> 
> \`\`\`
> .. py:function:: "><script>alert('hello sphinx')</script>
> \`\`\`

I'm not sure if that's really a problematic case, because the attack would have to come from the document content itself. I'm not a security specialist, so I'm probably wrong.

Anyway, I'm not suggesting to allow arbitrary characters.

`_` and `.` should be safe from a security standpoint, right?

> > > Last is that it allows sharing one node_ids to multiple names. For example, it helps to represent io.StringIO and _io.StringIO are the same (or they have the same ID).
> > 
> > I don't understand. Are io and _io not different names in Python?
> > And what does that have to do with changing underscores to dashes?
> 
> Indeed, `io` and `_io` are different names in python interpreter. But please read the python-doc.

I can see that those are not the same name.

What IDs are those supposed to get?

IMHO it would make perfect sense to give them the IDs `#io` and `#_io`, respectively, wouldn't it?

> The latter one is not documented in it. We have some issues to document a python object as "public name" instead of "canonical name". see #4065. It is one of them. This feature is not implemented yet. But I'll do that in the (nearly) future. It tells us the real name of the living object does not match the documentation of it.

I don't really understand any of this, but would it make a difference if underscores (`_`) were allowed in IDs?

> As you know, it is not related to hyphens problem. It also conflicts with the hyperlinks which human builds manually. It's no longer guess-able. If we'll keep using dots and hyphens for node_id, the cross-reference feature is needed to create references for nbsphinx, I think.

I don't understand.
Do you mean I should create my own custom IDs in `nbsphinx` and overwrite the ones generated by Sphinx?
I guess I will have to do something like that if you are not modifying the way IDs are generated by Sphinx.
I could probably do something similar to https://github.com/spatialaudio/nbsphinx/blob/559fc4e82bc9e2123e546e67b8032643c87cfaf6/src/nbsphinx.py#L1384-L1407.

I *do* understand that IDs should be unique per HTML page, and I don't mind if the second (and third etc.) duplicate is re-written to `#id0` etc., but I would really like to have readable and understandable IDs (for Python API links) for the case that there are no duplicate IDs (and for the first one even if there are duplicates). And (probably more importantly?) I would like to avoid too much breakage in the projects of `nbsphinx` users.
>OK, I understand that it might be problematic to allow arbitrary characters/code points.
>But what about just adding _ and . to the allowed characters?

Surely, I don't think `_` and `.` will not cause the problem, as you said. But are you satisfied if I agree to support `_` and `.`? How about capital characters too. How about the latin-1 characters? If you need to change the charset, we need to define the allowed charset for node IDs (and confirm they are safe as possible). Is it okay to support only `_` and `.`?

>I don't understand.
>Do you mean I should create my own custom IDs in nbsphinx and overwrite the ones generated by Sphinx?
>I guess I will have to do something like that if you are not modifying the way IDs are generated by Sphinx.

So far, a node_id of a python object had been the same as its name. Since Sphinx-3.0, it will be changed. The new implementation is almost the same as other domains do for the cross-references.

To realize the new cross-reference feature, we use a "reference name" and location info. A reference name equals the name of the object. For example, `io`, `io.StringIO`, `int`, `MyClass`, `MyClass.meth` and so on. A location info is a pair of a docname and a node ID.

On building a document, the python domain goes the following steps:

1. Collects definitions of python objects written by their directives (ex. `py:function::`, `py:class::`, etc.) and creates a mapping from the reference-names to the location info.
2. Every time find cross-reference roles, look up the location info from the mapping using the reference name of the role. For example, when a role \`\`\`:py:mod:`io` \`\`\` found, the python domain look up the location info by the reference name; `io`. If succeeded, the role is converted to a reference to the specified node in the document.
3. Renders the references to the output in the arbitrary format.

This means generating URL manually is not recommended. The node_id is not guess-able because it is sometimes auto-generated (ex. `id0`). I still don't think about how to implement `io` and `_io` case. But it will obey the structure above.

Note: docutils spec says node_id should be starts with a letter `[a-z]`. So `_io` will be converted into `io` (if not already registered).
It's a tricky issue, but I think it would be good to be a bit more permissive on the IDs, and ignore the docutils spec a bit as it is not enforced anyway.

My reasoning behind the ID generation for, primarily the C++ domain, but also the C domain:
- IDs should be as stable as possible, as they may be used in external links.
- That means, if one permutes declarations, then it should have no effect on their generated IDs. This rules out having counters.
- The extreme cases are for example singlehtml and Latex where IDs need to be unique throughout the whole document, and not just per page.
- Domains are independent and users may implement their own, so all IDs of a domain should have a unique prefix (as best as possible*). E.g., currently ``c.`` for C and ``_CPPv4`` for C++.
- Changes are sometimes needed to the ID generation scheme, so nodes should also be assigned their old IDs, though on a best-effort  basis. If this gets out of hand we can deprecate the oldest IDs.

So
- For most languages/domains a unique ID can be made up by identifiers (i.e., something like [a-zA-Z_][a-zA-Z_0-9]*), and then a scope separator (e.g., ``.`` in Python and C).
- The docutils restrictions/guidelines (https://docutils.sourceforge.io/docs/ref/rst/directives.html#identifier-normalization) seems to be based on "old" technologies, HTML 4.1 and CSS 1. Though it doesn't look like this normalisation is applied to IDs coming from Sphinx. Anyway, HTML 5 loosened the restrictions: https://mathiasbynens.be/notes/html5-id-class
- Therefore, if we use restrict IDs to non-empty strings matching ``[A-Za-z0-9_.]*`` I believe we can guarantee uniqueness almost always. Optionally, we can map ``.`` into ``-`` to make CSS selection easier (i.e., no escaping), though has this been a problem so far?
- Regarding *, domain ID prefixes: if we get really strict on this, we can require domains to register their primary ID prefix they use and enforce uniqueness.
I can accept adding `.` and `_` to the allowed character list. I suppose it will harm nobody.

>The docutils restrictions/guidelines (https://docutils.sourceforge.io/docs/ref/rst/directives.html#identifier-normalization) seems to be based on "old" technologies, HTML 4.1 and CSS 1. Though it doesn't look like this normalisation is applied to IDs coming from Sphinx. Anyway, HTML 5 loosened the restrictions:

Unfortunately, we still support HTML4 technology... Especially HTMLHelp builder depends on it. I don't know how many users use it. But bug reports are sometimes filed. So it seems used.

So I hesitate to use `.` and `_` to the first character. 

>Optionally, we can map . into - to make CSS selection easier (i.e., no escaping), though has this been a problem so far?

Unfortunately, I don't know. To be exact, I've never seen the CSS class converted from node ID.
I made a PR #7356 to allow `.` and `_` in node id generation. It seems nobody objects to support them for node ids. So I'll merge it soon.
Now I merged #7356 for beta release. But reopened to continue this discussion. Please check it and give me comments. I'd like to refine it before 3.0 final (if needed).
Thanks! As a last thing I believe it is ok to allow capital letters as well, which would make it much easier to guarantee uniqueness. Maybe I just missed the docutils rationale for converting to lowercase, but I haven't seen any, and I don't know of any output format where IDs are not case sensitive.

Thanks @tk0miya, you are the best!

I agree with @jakobandersen that keeping capital letters would  be great.

For example, right now both these links work:

* https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.templatebridge
* https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.TemplateBridge

... but only the first one creates the yellow highlighting.
It's not necessary to have both, I think the lowercase target should be removed.

The corresponding `[source]` link respects uppercase, which is good: https://www.sphinx-doc.org/en/master/_modules/sphinx/application.html#TemplateBridge

And the `[docs]` link back from there also respects uppercase, which is also good: https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.TemplateBridge

Now the last missing piece would be to remove the lower-case link target and replace it with the correct case.

> How about capital characters too.

Yes, please!

> How about the latin-1 characters?

I guess we'll have to draw the line somewhere. Python allows many Unicode characters in identifiers, but AFAICT, in practice most people still use only ASCII letters (lower and upper case) and numbers. And, very importantly, underscores. And the dot (`.`) is used for namespacing, also very important.

I don't care about any other latin-1 or Unicode characters.
```

## Patch

```diff
diff --git a/sphinx/builders/_epub_base.py b/sphinx/builders/_epub_base.py
--- a/sphinx/builders/_epub_base.py
+++ b/sphinx/builders/_epub_base.py
@@ -259,6 +259,15 @@ def fix_ids(self, tree: nodes.document) -> None:
         Some readers crash because they interpret the part as a
         transport protocol specification.
         """
+        def update_node_id(node: Element) -> None:
+            """Update IDs of given *node*."""
+            new_ids = []
+            for node_id in node['ids']:
+                new_id = self.fix_fragment('', node_id)
+                if new_id not in new_ids:
+                    new_ids.append(new_id)
+            node['ids'] = new_ids
+
         for reference in tree.traverse(nodes.reference):
             if 'refuri' in reference:
                 m = self.refuri_re.match(reference['refuri'])
@@ -268,22 +277,14 @@ def fix_ids(self, tree: nodes.document) -> None:
                 reference['refid'] = self.fix_fragment('', reference['refid'])
 
         for target in tree.traverse(nodes.target):
-            for i, node_id in enumerate(target['ids']):
-                if ':' in node_id:
-                    target['ids'][i] = self.fix_fragment('', node_id)
+            update_node_id(target)
 
             next_node = target.next_node(ascend=True)  # type: Node
             if isinstance(next_node, nodes.Element):
-                for i, node_id in enumerate(next_node['ids']):
-                    if ':' in node_id:
-                        next_node['ids'][i] = self.fix_fragment('', node_id)
+                update_node_id(next_node)
 
         for desc_signature in tree.traverse(addnodes.desc_signature):
-            ids = desc_signature.attributes['ids']
-            newids = []
-            for id in ids:
-                newids.append(self.fix_fragment('', id))
-            desc_signature.attributes['ids'] = newids
+            update_node_id(desc_signature)
 
     def add_visible_links(self, tree: nodes.document, show_urls: str = 'inline') -> None:
         """Add visible link targets for external links"""
diff --git a/sphinx/util/nodes.py b/sphinx/util/nodes.py
--- a/sphinx/util/nodes.py
+++ b/sphinx/util/nodes.py
@@ -445,6 +445,7 @@ def _make_id(string: str) -> str:
 
     Changes:
 
+    * Allow to use capital alphabet characters
     * Allow to use dots (".") and underscores ("_") for an identifier
       without a leading character.
 
@@ -452,8 +453,7 @@ def _make_id(string: str) -> str:
     # Maintainer: docutils-develop@lists.sourceforge.net
     # Copyright: This module has been placed in the public domain.
     """
-    id = string.lower()
-    id = id.translate(_non_id_translate_digraphs)
+    id = string.translate(_non_id_translate_digraphs)
     id = id.translate(_non_id_translate)
     # get rid of non-ascii characters.
     # 'ascii' lowercase to prevent problems with turkish locale.
@@ -464,7 +464,7 @@ def _make_id(string: str) -> str:
     return str(id)
 
 
-_non_id_chars = re.compile('[^a-z0-9._]+')
+_non_id_chars = re.compile('[^a-zA-Z0-9._]+')
 _non_id_at_ends = re.compile('^[-0-9._]+|-+$')
 _non_id_translate = {
     0x00f8: u'o',       # o with stroke

```

## Test Patch

```diff
diff --git a/tests/test_build_epub.py b/tests/test_build_epub.py
--- a/tests/test_build_epub.py
+++ b/tests/test_build_epub.py
@@ -320,13 +320,11 @@ def test_epub_anchor_id(app):
     app.build()
 
     html = (app.outdir / 'index.xhtml').read_text()
-    assert ('<p id="std-setting-staticfiles_finders">'
-            '<span id="std-setting-STATICFILES_FINDERS"></span>'
+    assert ('<p id="std-setting-STATICFILES_FINDERS">'
             'blah blah blah</p>' in html)
-    assert ('<span id="std-setting-staticfiles_section"></span>'
-            '<span id="std-setting-STATICFILES_SECTION"></span>'
+    assert ('<span id="std-setting-STATICFILES_SECTION"></span>'
             '<h1>blah blah blah</h1>' in html)
-    assert 'see <a class="reference internal" href="#std-setting-staticfiles_finders">' in html
+    assert 'see <a class="reference internal" href="#std-setting-STATICFILES_FINDERS">' in html
 
 
 @pytest.mark.sphinx('epub', testroot='html_assets')
diff --git a/tests/test_build_html.py b/tests/test_build_html.py
--- a/tests/test_build_html.py
+++ b/tests/test_build_html.py
@@ -176,7 +176,7 @@ def test_html4_output(app, status, warning):
          r'-|      |-'),
     ],
     'autodoc.html': [
-        (".//dl[@class='py class']/dt[@id='autodoc_target.class']", ''),
+        (".//dl[@class='py class']/dt[@id='autodoc_target.Class']", ''),
         (".//dl[@class='py function']/dt[@id='autodoc_target.function']/em/span", r'\*\*'),
         (".//dl[@class='py function']/dt[@id='autodoc_target.function']/em/span", r'kwds'),
         (".//dd/p", r'Return spam\.'),
@@ -219,7 +219,7 @@ def test_html4_output(app, status, warning):
          "[@class='rfc reference external']/strong", 'RFC 1'),
         (".//a[@href='https://tools.ietf.org/html/rfc1.html']"
          "[@class='rfc reference external']/strong", 'Request for Comments #1'),
-        (".//a[@href='objects.html#envvar-home']"
+        (".//a[@href='objects.html#envvar-HOME']"
          "[@class='reference internal']/code/span[@class='pre']", 'HOME'),
         (".//a[@href='#with']"
          "[@class='reference internal']/code/span[@class='pre']", '^with$'),
@@ -275,18 +275,18 @@ def test_html4_output(app, status, warning):
         (".//p", 'Il dit : « C’est “super” ! »'),
     ],
     'objects.html': [
-        (".//dt[@id='mod.cls.meth1']", ''),
-        (".//dt[@id='errmod.error']", ''),
+        (".//dt[@id='mod.Cls.meth1']", ''),
+        (".//dt[@id='errmod.Error']", ''),
         (".//dt/code", r'long\(parameter,\s* list\)'),
         (".//dt/code", 'another one'),
-        (".//a[@href='#mod.cls'][@class='reference internal']", ''),
+        (".//a[@href='#mod.Cls'][@class='reference internal']", ''),
         (".//dl[@class='std userdesc']", ''),
         (".//dt[@id='userdesc-myobj']", ''),
         (".//a[@href='#userdesc-myobj'][@class='reference internal']", ''),
         # docfields
-        (".//a[@class='reference internal'][@href='#timeint']/em", 'TimeInt'),
-        (".//a[@class='reference internal'][@href='#time']", 'Time'),
-        (".//a[@class='reference internal'][@href='#errmod.error']/strong", 'Error'),
+        (".//a[@class='reference internal'][@href='#TimeInt']/em", 'TimeInt'),
+        (".//a[@class='reference internal'][@href='#Time']", 'Time'),
+        (".//a[@class='reference internal'][@href='#errmod.Error']/strong", 'Error'),
         # C references
         (".//span[@class='pre']", 'CFunction()'),
         (".//a[@href='#c.Sphinx_DoSomething']", ''),
@@ -323,7 +323,7 @@ def test_html4_output(app, status, warning):
          'perl'),
         (".//a[@class='reference internal'][@href='#cmdoption-perl-arg-p']/code/span",
          '\\+p'),
-        (".//a[@class='reference internal'][@href='#cmdoption-perl-objc']/code/span",
+        (".//a[@class='reference internal'][@href='#cmdoption-perl-ObjC']/code/span",
          '--ObjC\\+\\+'),
         (".//a[@class='reference internal'][@href='#cmdoption-perl-plugin.option']/code/span",
          '--plugin.option'),
diff --git a/tests/test_domain_js.py b/tests/test_domain_js.py
--- a/tests/test_domain_js.py
+++ b/tests/test_domain_js.py
@@ -120,25 +120,25 @@ def find_obj(mod_name, prefix, obj_name, obj_type, searchmode=0):
 
     assert (find_obj(None, None, 'NONEXISTANT', 'class') == (None, None))
     assert (find_obj(None, None, 'NestedParentA', 'class') ==
-            ('NestedParentA', ('roles', 'nestedparenta', 'class')))
+            ('NestedParentA', ('roles', 'NestedParentA', 'class')))
     assert (find_obj(None, None, 'NestedParentA.NestedChildA', 'class') ==
             ('NestedParentA.NestedChildA',
-             ('roles', 'nestedparenta.nestedchilda', 'class')))
+             ('roles', 'NestedParentA.NestedChildA', 'class')))
     assert (find_obj(None, 'NestedParentA', 'NestedChildA', 'class') ==
             ('NestedParentA.NestedChildA',
-             ('roles', 'nestedparenta.nestedchilda', 'class')))
+             ('roles', 'NestedParentA.NestedChildA', 'class')))
     assert (find_obj(None, None, 'NestedParentA.NestedChildA.subchild_1', 'func') ==
             ('NestedParentA.NestedChildA.subchild_1',
-             ('roles', 'nestedparenta.nestedchilda.subchild_1', 'function')))
+             ('roles', 'NestedParentA.NestedChildA.subchild_1', 'function')))
     assert (find_obj(None, 'NestedParentA', 'NestedChildA.subchild_1', 'func') ==
             ('NestedParentA.NestedChildA.subchild_1',
-             ('roles', 'nestedparenta.nestedchilda.subchild_1', 'function')))
+             ('roles', 'NestedParentA.NestedChildA.subchild_1', 'function')))
     assert (find_obj(None, 'NestedParentA.NestedChildA', 'subchild_1', 'func') ==
             ('NestedParentA.NestedChildA.subchild_1',
-             ('roles', 'nestedparenta.nestedchilda.subchild_1', 'function')))
+             ('roles', 'NestedParentA.NestedChildA.subchild_1', 'function')))
     assert (find_obj('module_a.submodule', 'ModTopLevel', 'mod_child_2', 'meth') ==
             ('module_a.submodule.ModTopLevel.mod_child_2',
-             ('module', 'module_a.submodule.modtoplevel.mod_child_2', 'method')))
+             ('module', 'module_a.submodule.ModTopLevel.mod_child_2', 'method')))
     assert (find_obj('module_b.submodule', 'ModTopLevel', 'module_a.submodule', 'mod') ==
             ('module_a.submodule',
              ('module', 'module-module_a.submodule', 'module')))
@@ -205,7 +205,7 @@ def test_js_class(app):
                                                     [desc_parameterlist, ()])],
                                   [desc_content, ()])]))
     assert_node(doctree[0], addnodes.index,
-                entries=[("single", "Application() (class)", "application", "", None)])
+                entries=[("single", "Application() (class)", "Application", "", None)])
     assert_node(doctree[1], addnodes.desc, domain="js", objtype="class", noindex=False)
 
 
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -171,11 +171,11 @@ def test_resolve_xref_for_properties(app, status, warning):
     app.builder.build_all()
 
     content = (app.outdir / 'module.html').read_text()
-    assert ('Link to <a class="reference internal" href="#module_a.submodule.modtoplevel.prop"'
+    assert ('Link to <a class="reference internal" href="#module_a.submodule.ModTopLevel.prop"'
             ' title="module_a.submodule.ModTopLevel.prop">'
             '<code class="xref py py-attr docutils literal notranslate"><span class="pre">'
             'prop</span> <span class="pre">attribute</span></code></a>' in content)
-    assert ('Link to <a class="reference internal" href="#module_a.submodule.modtoplevel.prop"'
+    assert ('Link to <a class="reference internal" href="#module_a.submodule.ModTopLevel.prop"'
             ' title="module_a.submodule.ModTopLevel.prop">'
             '<code class="xref py py-meth docutils literal notranslate"><span class="pre">'
             'prop</span> <span class="pre">method</span></code></a>' in content)
@@ -192,20 +192,20 @@ def find_obj(modname, prefix, obj_name, obj_type, searchmode=0):
 
     assert (find_obj(None, None, 'NONEXISTANT', 'class') == [])
     assert (find_obj(None, None, 'NestedParentA', 'class') ==
-            [('NestedParentA', ('roles', 'nestedparenta', 'class'))])
+            [('NestedParentA', ('roles', 'NestedParentA', 'class'))])
     assert (find_obj(None, None, 'NestedParentA.NestedChildA', 'class') ==
-            [('NestedParentA.NestedChildA', ('roles', 'nestedparenta.nestedchilda', 'class'))])
+            [('NestedParentA.NestedChildA', ('roles', 'NestedParentA.NestedChildA', 'class'))])
     assert (find_obj(None, 'NestedParentA', 'NestedChildA', 'class') ==
-            [('NestedParentA.NestedChildA', ('roles', 'nestedparenta.nestedchilda', 'class'))])
+            [('NestedParentA.NestedChildA', ('roles', 'NestedParentA.NestedChildA', 'class'))])
     assert (find_obj(None, None, 'NestedParentA.NestedChildA.subchild_1', 'meth') ==
             [('NestedParentA.NestedChildA.subchild_1',
-              ('roles', 'nestedparenta.nestedchilda.subchild_1', 'method'))])
+              ('roles', 'NestedParentA.NestedChildA.subchild_1', 'method'))])
     assert (find_obj(None, 'NestedParentA', 'NestedChildA.subchild_1', 'meth') ==
             [('NestedParentA.NestedChildA.subchild_1',
-              ('roles', 'nestedparenta.nestedchilda.subchild_1', 'method'))])
+              ('roles', 'NestedParentA.NestedChildA.subchild_1', 'method'))])
     assert (find_obj(None, 'NestedParentA.NestedChildA', 'subchild_1', 'meth') ==
             [('NestedParentA.NestedChildA.subchild_1',
-              ('roles', 'nestedparenta.nestedchilda.subchild_1', 'method'))])
+              ('roles', 'NestedParentA.NestedChildA.subchild_1', 'method'))])
 
 
 def test_get_full_qualified_name():
@@ -525,61 +525,61 @@ def test_pymethod_options(app):
 
     # method
     assert_node(doctree[1][1][0], addnodes.index,
-                entries=[('single', 'meth1() (Class method)', 'class.meth1', '', None)])
+                entries=[('single', 'meth1() (Class method)', 'Class.meth1', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_name, "meth1"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth1' in domain.objects
-    assert domain.objects['Class.meth1'] == ('index', 'class.meth1', 'method')
+    assert domain.objects['Class.meth1'] == ('index', 'Class.meth1', 'method')
 
     # :classmethod:
     assert_node(doctree[1][1][2], addnodes.index,
-                entries=[('single', 'meth2() (Class class method)', 'class.meth2', '', None)])
+                entries=[('single', 'meth2() (Class class method)', 'Class.meth2', '', None)])
     assert_node(doctree[1][1][3], ([desc_signature, ([desc_annotation, "classmethod "],
                                                      [desc_name, "meth2"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth2' in domain.objects
-    assert domain.objects['Class.meth2'] == ('index', 'class.meth2', 'method')
+    assert domain.objects['Class.meth2'] == ('index', 'Class.meth2', 'method')
 
     # :staticmethod:
     assert_node(doctree[1][1][4], addnodes.index,
-                entries=[('single', 'meth3() (Class static method)', 'class.meth3', '', None)])
+                entries=[('single', 'meth3() (Class static method)', 'Class.meth3', '', None)])
     assert_node(doctree[1][1][5], ([desc_signature, ([desc_annotation, "static "],
                                                      [desc_name, "meth3"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth3' in domain.objects
-    assert domain.objects['Class.meth3'] == ('index', 'class.meth3', 'method')
+    assert domain.objects['Class.meth3'] == ('index', 'Class.meth3', 'method')
 
     # :async:
     assert_node(doctree[1][1][6], addnodes.index,
-                entries=[('single', 'meth4() (Class method)', 'class.meth4', '', None)])
+                entries=[('single', 'meth4() (Class method)', 'Class.meth4', '', None)])
     assert_node(doctree[1][1][7], ([desc_signature, ([desc_annotation, "async "],
                                                      [desc_name, "meth4"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth4' in domain.objects
-    assert domain.objects['Class.meth4'] == ('index', 'class.meth4', 'method')
+    assert domain.objects['Class.meth4'] == ('index', 'Class.meth4', 'method')
 
     # :property:
     assert_node(doctree[1][1][8], addnodes.index,
-                entries=[('single', 'meth5() (Class property)', 'class.meth5', '', None)])
+                entries=[('single', 'meth5() (Class property)', 'Class.meth5', '', None)])
     assert_node(doctree[1][1][9], ([desc_signature, ([desc_annotation, "property "],
                                                      [desc_name, "meth5"])],
                                    [desc_content, ()]))
     assert 'Class.meth5' in domain.objects
-    assert domain.objects['Class.meth5'] == ('index', 'class.meth5', 'method')
+    assert domain.objects['Class.meth5'] == ('index', 'Class.meth5', 'method')
 
     # :abstractmethod:
     assert_node(doctree[1][1][10], addnodes.index,
-                entries=[('single', 'meth6() (Class method)', 'class.meth6', '', None)])
+                entries=[('single', 'meth6() (Class method)', 'Class.meth6', '', None)])
     assert_node(doctree[1][1][11], ([desc_signature, ([desc_annotation, "abstract "],
                                                       [desc_name, "meth6"],
                                                       [desc_parameterlist, ()])],
                                     [desc_content, ()]))
     assert 'Class.meth6' in domain.objects
-    assert domain.objects['Class.meth6'] == ('index', 'class.meth6', 'method')
+    assert domain.objects['Class.meth6'] == ('index', 'Class.meth6', 'method')
 
 
 def test_pyclassmethod(app):
@@ -594,13 +594,13 @@ def test_pyclassmethod(app):
                                   [desc_content, (addnodes.index,
                                                   desc)])]))
     assert_node(doctree[1][1][0], addnodes.index,
-                entries=[('single', 'meth() (Class class method)', 'class.meth', '', None)])
+                entries=[('single', 'meth() (Class class method)', 'Class.meth', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_annotation, "classmethod "],
                                                      [desc_name, "meth"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth' in domain.objects
-    assert domain.objects['Class.meth'] == ('index', 'class.meth', 'method')
+    assert domain.objects['Class.meth'] == ('index', 'Class.meth', 'method')
 
 
 def test_pystaticmethod(app):
@@ -615,13 +615,13 @@ def test_pystaticmethod(app):
                                   [desc_content, (addnodes.index,
                                                   desc)])]))
     assert_node(doctree[1][1][0], addnodes.index,
-                entries=[('single', 'meth() (Class static method)', 'class.meth', '', None)])
+                entries=[('single', 'meth() (Class static method)', 'Class.meth', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_annotation, "static "],
                                                      [desc_name, "meth"],
                                                      [desc_parameterlist, ()])],
                                    [desc_content, ()]))
     assert 'Class.meth' in domain.objects
-    assert domain.objects['Class.meth'] == ('index', 'class.meth', 'method')
+    assert domain.objects['Class.meth'] == ('index', 'Class.meth', 'method')
 
 
 def test_pyattribute(app):
@@ -638,13 +638,13 @@ def test_pyattribute(app):
                                   [desc_content, (addnodes.index,
                                                   desc)])]))
     assert_node(doctree[1][1][0], addnodes.index,
-                entries=[('single', 'attr (Class attribute)', 'class.attr', '', None)])
+                entries=[('single', 'attr (Class attribute)', 'Class.attr', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_name, "attr"],
                                                      [desc_annotation, ": str"],
                                                      [desc_annotation, " = ''"])],
                                    [desc_content, ()]))
     assert 'Class.attr' in domain.objects
-    assert domain.objects['Class.attr'] == ('index', 'class.attr', 'attribute')
+    assert domain.objects['Class.attr'] == ('index', 'Class.attr', 'attribute')
 
 
 def test_pydecorator_signature(app):
diff --git a/tests/test_domain_std.py b/tests/test_domain_std.py
--- a/tests/test_domain_std.py
+++ b/tests/test_domain_std.py
@@ -353,23 +353,23 @@ def test_productionlist(app, status, warning):
         linkText = span.text.strip()
         cases.append((text, link, linkText))
     assert cases == [
-        ('A', 'Bare.html#grammar-token-a', 'A'),
-        ('B', 'Bare.html#grammar-token-b', 'B'),
-        ('P1:A', 'P1.html#grammar-token-p1-a', 'P1:A'),
-        ('P1:B', 'P1.html#grammar-token-p1-b', 'P1:B'),
-        ('P2:A', 'P1.html#grammar-token-p1-a', 'P1:A'),
-        ('P2:B', 'P2.html#grammar-token-p2-b', 'P2:B'),
-        ('Explicit title A, plain', 'Bare.html#grammar-token-a', 'MyTitle'),
-        ('Explicit title A, colon', 'Bare.html#grammar-token-a', 'My:Title'),
-        ('Explicit title P1:A, plain', 'P1.html#grammar-token-p1-a', 'MyTitle'),
-        ('Explicit title P1:A, colon', 'P1.html#grammar-token-p1-a', 'My:Title'),
-        ('Tilde A', 'Bare.html#grammar-token-a', 'A'),
-        ('Tilde P1:A', 'P1.html#grammar-token-p1-a', 'A'),
-        ('Tilde explicit title P1:A', 'P1.html#grammar-token-p1-a', '~MyTitle'),
-        ('Tilde, explicit title P1:A', 'P1.html#grammar-token-p1-a', 'MyTitle'),
-        ('Dup', 'Dup2.html#grammar-token-dup', 'Dup'),
-        ('FirstLine', 'firstLineRule.html#grammar-token-firstline', 'FirstLine'),
-        ('SecondLine', 'firstLineRule.html#grammar-token-secondline', 'SecondLine'),
+        ('A', 'Bare.html#grammar-token-A', 'A'),
+        ('B', 'Bare.html#grammar-token-B', 'B'),
+        ('P1:A', 'P1.html#grammar-token-P1-A', 'P1:A'),
+        ('P1:B', 'P1.html#grammar-token-P1-B', 'P1:B'),
+        ('P2:A', 'P1.html#grammar-token-P1-A', 'P1:A'),
+        ('P2:B', 'P2.html#grammar-token-P2-B', 'P2:B'),
+        ('Explicit title A, plain', 'Bare.html#grammar-token-A', 'MyTitle'),
+        ('Explicit title A, colon', 'Bare.html#grammar-token-A', 'My:Title'),
+        ('Explicit title P1:A, plain', 'P1.html#grammar-token-P1-A', 'MyTitle'),
+        ('Explicit title P1:A, colon', 'P1.html#grammar-token-P1-A', 'My:Title'),
+        ('Tilde A', 'Bare.html#grammar-token-A', 'A'),
+        ('Tilde P1:A', 'P1.html#grammar-token-P1-A', 'A'),
+        ('Tilde explicit title P1:A', 'P1.html#grammar-token-P1-A', '~MyTitle'),
+        ('Tilde, explicit title P1:A', 'P1.html#grammar-token-P1-A', 'MyTitle'),
+        ('Dup', 'Dup2.html#grammar-token-Dup', 'Dup'),
+        ('FirstLine', 'firstLineRule.html#grammar-token-FirstLine', 'FirstLine'),
+        ('SecondLine', 'firstLineRule.html#grammar-token-SecondLine', 'SecondLine'),
     ]
 
     text = (app.outdir / 'LineContinuation.html').read_text()
diff --git a/tests/test_environment_indexentries.py b/tests/test_environment_indexentries.py
--- a/tests/test_environment_indexentries.py
+++ b/tests/test_environment_indexentries.py
@@ -161,5 +161,5 @@ def test_create_index_by_key(app):
     index = IndexEntries(app.env).create_index(app.builder)
     assert len(index) == 3
     assert index[0] == ('D', [('docutils', [[('main', '#term-docutils')], [], None])])
-    assert index[1] == ('P', [('Python', [[('main', '#term-python')], [], None])])
+    assert index[1] == ('P', [('Python', [[('main', '#term-Python')], [], None])])
     assert index[2] == ('ス', [('スフィンクス', [[('main', '#term-0')], [], 'ス'])])
diff --git a/tests/test_intl.py b/tests/test_intl.py
--- a/tests/test_intl.py
+++ b/tests/test_intl.py
@@ -946,14 +946,14 @@ def test_xml_role_xref(app):
         ['LINK TO', "I18N ROCK'N ROLE XREF", ',', 'CONTENTS', ',',
          'SOME NEW TERM', '.'],
         ['i18n-role-xref', 'index',
-         'glossary_terms#term-some-term'])
+         'glossary_terms#term-Some-term'])
 
     para2 = sec2.findall('paragraph')
     assert_elem(
         para2[0],
         ['LINK TO', 'SOME OTHER NEW TERM', 'AND', 'SOME NEW TERM', '.'],
-        ['glossary_terms#term-some-other-term',
-         'glossary_terms#term-some-term'])
+        ['glossary_terms#term-Some-other-term',
+         'glossary_terms#term-Some-term'])
     assert_elem(
         para2[1],
         ['LINK TO', 'SAME TYPE LINKS', 'AND',
diff --git a/tests/test_util_nodes.py b/tests/test_util_nodes.py
--- a/tests/test_util_nodes.py
+++ b/tests/test_util_nodes.py
@@ -188,13 +188,13 @@ def test_clean_astext():
     [
         ('', '', 'id0'),
         ('term', '', 'term-0'),
-        ('term', 'Sphinx', 'term-sphinx'),
-        ('', 'io.StringIO', 'io.stringio'),   # contains a dot
+        ('term', 'Sphinx', 'term-Sphinx'),
+        ('', 'io.StringIO', 'io.StringIO'),   # contains a dot
         ('', 'sphinx.setup_command', 'sphinx.setup_command'),  # contains a dot & underscore
-        ('', '_io.StringIO', 'io.stringio'),  # starts with underscore
+        ('', '_io.StringIO', 'io.StringIO'),  # starts with underscore
         ('', 'ｓｐｈｉｎｘ', 'sphinx'),  # alphabets in unicode fullwidth characters
         ('', '悠好', 'id0'),  # multibytes text (in Chinese)
-        ('', 'Hello=悠好=こんにちは', 'hello'),  # alphabets and multibytes text
+        ('', 'Hello=悠好=こんにちは', 'Hello'),  # alphabets and multibytes text
         ('', 'fünf', 'funf'),  # latin1 (umlaut)
         ('', '0sphinx', 'sphinx'),  # starts with number
         ('', 'sphinx-', 'sphinx'),  # ends with hyphen
@@ -206,7 +206,7 @@ def test_make_id(app, prefix, term, expected):
 
 def test_make_id_already_registered(app):
     document = create_new_document()
-    document.ids['term-sphinx'] = True  # register "term-sphinx" manually
+    document.ids['term-Sphinx'] = True  # register "term-Sphinx" manually
     assert make_id(app.env, document, 'term', 'Sphinx') == 'term-0'
 
 

```


## Code snippets

### 1 - sphinx/ext/napoleon/__init__.py:

Start line: 312, End line: 328

```python
def _patch_python_domain() -> None:
    try:
        from sphinx.domains.python import PyTypedField
    except ImportError:
        pass
    else:
        import sphinx.domains.python
        from sphinx.locale import _
        for doc_field in sphinx.domains.python.PyObject.doc_field_types:
            if doc_field.name == 'parameter':
                doc_field.names = ('param', 'parameter', 'arg', 'argument')
                break
        sphinx.domains.python.PyObject.doc_field_types.append(
            PyTypedField('keyword', label=_('Keyword Arguments'),
                         names=('keyword', 'kwarg', 'kwparam'),
                         typerolename='obj', typenames=('paramtype', 'kwtype'),
                         can_collapse=True))
```
### 2 - sphinx/domains/python.py:

Start line: 221, End line: 239

```python
# This override allows our inline type specifiers to behave like :class: link
# when it comes to handling "." and "~" prefixes.
class PyXrefMixin:
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        result = super().make_xref(rolename, domain, target,  # type: ignore
                                   innernode, contnode, env)
        result['refspecific'] = True
        if target.startswith(('.', '~')):
            prefix, result['reftarget'] = target[0], target[1:]
            if prefix == '.':
                text = target[1:]
            elif prefix == '~':
                text = target.split('.')[-1]
            for node in result.traverse(nodes.Text):
                node.parent[node.parent.index(node)] = nodes.Text(text)
                break
        return result
```
### 3 - sphinx/domains/python.py:

Start line: 11, End line: 68

```python
import builtins
import inspect
import re
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, Tuple
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import pending_xref, desc_signature
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast, parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import TextlikeNode

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)


# REs for Python signatures
py_sig_re = re.compile(
    r'''^ ([\w.]*\.)?            # class name(s)
          (\w+)  \s*             # thing name
          (?: \(\s*(.*)\s*\)     # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)


pairindextypes = {
    'module':    _('module'),
    'keyword':   _('keyword'),
    'operator':  _('operator'),
    'object':    _('object'),
    'exception': _('exception'),
    'statement': _('statement'),
    'builtin':   _('built-in function'),
}
```
### 4 - sphinx/domains/python.py:

Start line: 1273, End line: 1284

```python
class PythonDomain(Domain):

    def _make_module_refnode(self, builder: Builder, fromdocname: str, name: str,
                             contnode: Node) -> Element:
        # get additional info for modules
        docname, node_id, synopsis, platform, deprecated = self.modules[name]
        title = name
        if synopsis:
            title += ': ' + synopsis
        if deprecated:
            title += _(' (deprecated)')
        if platform:
            title += ' (' + platform + ')'
        return make_refnode(builder, fromdocname, docname, node_id, contnode, title)
```
### 5 - sphinx/directives/__init__.py:

Start line: 259, End line: 297

```python
from sphinx.directives.code import (  # noqa
    Highlight, CodeBlock, LiteralInclude
)
from sphinx.directives.other import (  # noqa
    TocTree, Author, VersionChange, SeeAlso,
    TabularColumns, Centered, Acks, HList, Only, Include, Class
)
from sphinx.directives.patches import (  # noqa
    Figure, Meta
)
from sphinx.domains.index import IndexDirective  # noqa

deprecated_alias('sphinx.directives',
                 {
                     'Highlight': Highlight,
                     'CodeBlock': CodeBlock,
                     'LiteralInclude': LiteralInclude,
                     'TocTree': TocTree,
                     'Author': Author,
                     'Index': IndexDirective,
                     'VersionChange': VersionChange,
                     'SeeAlso': SeeAlso,
                     'TabularColumns': TabularColumns,
                     'Centered': Centered,
                     'Acks': Acks,
                     'HList': HList,
                     'Only': Only,
                     'Include': Include,
                     'Class': Class,
                     'Figure': Figure,
                     'Meta': Meta,
                 },
                 RemovedInSphinx40Warning)

deprecated_alias('sphinx.directives',
                 {
                     'DescDirective': ObjectDescription,
                 },
                 RemovedInSphinx50Warning)
```
### 6 - sphinx/domains/python.py:

Start line: 966, End line: 986

```python
class PyXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode['py:module'] = env.ref_context.get('py:module')
        refnode['py:class'] = env.ref_context.get('py:class')
        if not has_explicit_title:
            title = title.lstrip('.')    # only has a meaning for the target
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        # if the first character is a dot, search more specific namespaces first
        # else search builtins first
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        return title, target
```
### 7 - sphinx/domains/python.py:

Start line: 882, End line: 942

```python
class PyModule(SphinxDirective):
    """
    Directive to mark description of a new module.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'platform': lambda x: x,
        'synopsis': lambda x: x,
        'noindex': directives.flag,
        'deprecated': directives.flag,
    }

    def run(self) -> List[Node]:
        domain = cast(PythonDomain, self.env.get_domain('py'))

        modname = self.arguments[0].strip()
        noindex = 'noindex' in self.options
        self.env.ref_context['py:module'] = modname
        ret = []  # type: List[Node]
        if not noindex:
            # note module to the domain
            node_id = make_id(self.env, self.state.document, 'module', modname)
            target = nodes.target('', '', ids=[node_id], ismod=True)
            self.set_source_info(target)

            # Assign old styled node_id not to break old hyperlinks (if possible)
            # Note: Will removed in Sphinx-5.0  (RemovedInSphinx50Warning)
            old_node_id = self.make_old_id(modname)
            if node_id != old_node_id and old_node_id not in self.state.document.ids:
                target['ids'].append(old_node_id)

            self.state.document.note_explicit_target(target)

            domain.note_module(modname,
                               node_id,
                               self.options.get('synopsis', ''),
                               self.options.get('platform', ''),
                               'deprecated' in self.options)
            domain.note_object(modname, 'module', node_id, location=target)

            # the platform and synopsis aren't printed; in fact, they are only
            # used in the modindex currently
            ret.append(target)
            indextext = '%s; %s' % (pairindextypes['module'], modname)
            inode = addnodes.index(entries=[('pair', indextext, node_id, '', None)])
            ret.append(inode)
        return ret

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id.

        Old styled node_id is incompatible with docutils' node_id.
        It can contain dots and hyphens.

        .. note:: Old styled node_id was mainly used until Sphinx-3.0.
        """
        return 'module-%s' % name
```
### 8 - sphinx/domains/python.py:

Start line: 1227, End line: 1251

```python
class PythonDomain(Domain):

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                     type: str, target: str, node: pending_xref, contnode: Element
                     ) -> Element:
        modname = node.get('py:module')
        clsname = node.get('py:class')
        searchmode = 1 if node.hasattr('refspecific') else 0
        matches = self.find_obj(env, modname, clsname, target,
                                type, searchmode)

        if not matches and type == 'attr':
            # fallback to meth (for property)
            matches = self.find_obj(env, modname, clsname, target, 'meth', searchmode)

        if not matches:
            return None
        elif len(matches) > 1:
            logger.warning(__('more than one target found for cross-reference %r: %s'),
                           target, ', '.join(match[0] for match in matches),
                           type='ref', subtype='python', location=node)
        name, obj = matches[0]

        if obj[2] == 'module':
            return self._make_module_refnode(builder, fromdocname, name, contnode)
        else:
            return make_refnode(builder, fromdocname, obj[0], obj[1], contnode, name)
```
### 9 - sphinx/domains/python.py:

Start line: 945, End line: 963

```python
class PyCurrentModule(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting
    stuff in module foo, but links to module foo won't lead here.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == 'None':
            self.env.ref_context.pop('py:module', None)
        else:
            self.env.ref_context['py:module'] = modname
        return []
```
### 10 - sphinx/domains/std.py:

Start line: 11, End line: 48

```python
import re
import unicodedata
import warnings
from copy import copy
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import ws_re, logging, docname_join
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import RoleFunction

if False:
    # For type annotation
    from typing import Type  # for python3.5.1
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)


# RE for option descriptions
option_desc_re = re.compile(r'((?:/|--|-|\+)?[^\s=]+)(=?\s*.*)')
# RE for grammar tokens
token_re = re.compile(r'`(\w+)`', re.U)
```
### 15 - sphinx/builders/_epub_base.py:

Start line: 256, End line: 286

```python
class EpubBuilder(StandaloneHTMLBuilder):

    def fix_ids(self, tree: nodes.document) -> None:
        """Replace colons with hyphens in href and id attributes.

        Some readers crash because they interpret the part as a
        transport protocol specification.
        """
        for reference in tree.traverse(nodes.reference):
            if 'refuri' in reference:
                m = self.refuri_re.match(reference['refuri'])
                if m:
                    reference['refuri'] = self.fix_fragment(m.group(1), m.group(2))
            if 'refid' in reference:
                reference['refid'] = self.fix_fragment('', reference['refid'])

        for target in tree.traverse(nodes.target):
            for i, node_id in enumerate(target['ids']):
                if ':' in node_id:
                    target['ids'][i] = self.fix_fragment('', node_id)

            next_node = target.next_node(ascend=True)  # type: Node
            if isinstance(next_node, nodes.Element):
                for i, node_id in enumerate(next_node['ids']):
                    if ':' in node_id:
                        next_node['ids'][i] = self.fix_fragment('', node_id)

        for desc_signature in tree.traverse(addnodes.desc_signature):
            ids = desc_signature.attributes['ids']
            newids = []
            for id in ids:
                newids.append(self.fix_fragment('', id))
            desc_signature.attributes['ids'] = newids
```
