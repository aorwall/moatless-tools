# sphinx-doc__sphinx-7906

| **sphinx-doc/sphinx** | `fdd1aaf77058a579a5b5c2e3e6aff935265a7e49` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | - |
| **Any found context length** | 29136 |
| **Avg pos** | 33.5 |
| **Min pos** | 67 |
| **Max pos** | 67 |
| **Top file pos** | 6 |
| **Missing snippets** | 10 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/domains/c.py b/sphinx/domains/c.py
--- a/sphinx/domains/c.py
+++ b/sphinx/domains/c.py
@@ -16,6 +16,7 @@
 
 from docutils import nodes
 from docutils.nodes import Element, Node, TextElement, system_message
+from docutils.parsers.rst import directives
 
 from sphinx import addnodes
 from sphinx.addnodes import pending_xref
@@ -3023,10 +3024,7 @@ class CObject(ObjectDescription):
     ]
 
     option_spec = {
-        # have a dummy option to ensure proper errors on options,
-        # otherwise the option is taken as a continuation of the
-        # argument
-        'dummy': None
+        'noindexentry': directives.flag,
     }
 
     def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
@@ -3098,8 +3096,9 @@ def add_target_and_index(self, ast: ASTDeclaration, sig: str,
             if name not in domain.objects:
                 domain.objects[name] = (domain.env.docname, newestId, self.objtype)
 
-        indexText = self.get_index_text(name)
-        self.indexnode['entries'].append(('single', indexText, newestId, '', None))
+        if 'noindexentry' not in self.options:
+            indexText = self.get_index_text(name)
+            self.indexnode['entries'].append(('single', indexText, newestId, '', None))
 
     @property
     def object_type(self) -> str:
diff --git a/sphinx/domains/cpp.py b/sphinx/domains/cpp.py
--- a/sphinx/domains/cpp.py
+++ b/sphinx/domains/cpp.py
@@ -6625,6 +6625,7 @@ class CPPObject(ObjectDescription):
     ]
 
     option_spec = {
+        'noindexentry': directives.flag,
         'tparam-line-spec': directives.flag,
     }
 
@@ -6701,7 +6702,7 @@ def add_target_and_index(self, ast: ASTDeclaration, sig: str,
             if decl.objectType == 'concept':
                 isInConcept = True
                 break
-        if not isInConcept:
+        if not isInConcept and 'noindexentry' not in self.options:
             strippedName = name
             for prefix in self.env.config.cpp_index_common_prefix:
                 if name.startswith(prefix):
diff --git a/sphinx/domains/javascript.py b/sphinx/domains/javascript.py
--- a/sphinx/domains/javascript.py
+++ b/sphinx/domains/javascript.py
@@ -49,6 +49,11 @@ class JSObject(ObjectDescription):
     #: based on directive nesting
     allow_nesting = False
 
+    option_spec = {
+        'noindex': directives.flag,
+        'noindexentry': directives.flag,
+    }
+
     def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
         """Breaks down construct signatures
 
@@ -120,9 +125,10 @@ def add_target_and_index(self, name_obj: Tuple[str, str], sig: str,
         domain = cast(JavaScriptDomain, self.env.get_domain('js'))
         domain.note_object(fullname, self.objtype, node_id, location=signode)
 
-        indextext = self.get_index_text(mod_name, name_obj)
-        if indextext:
-            self.indexnode['entries'].append(('single', indextext, node_id, '', None))
+        if 'noindexentry' not in self.options:
+            indextext = self.get_index_text(mod_name, name_obj)
+            if indextext:
+                self.indexnode['entries'].append(('single', indextext, node_id, '', None))
 
     def get_index_text(self, objectname: str, name_obj: Tuple[str, str]) -> str:
         name, obj = name_obj
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -317,6 +317,7 @@ class PyObject(ObjectDescription):
     """
     option_spec = {
         'noindex': directives.flag,
+        'noindexentry': directives.flag,
         'module': directives.unchanged,
         'annotation': directives.unchanged,
     }
@@ -459,9 +460,10 @@ def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
         domain = cast(PythonDomain, self.env.get_domain('py'))
         domain.note_object(fullname, self.objtype, node_id, location=signode)
 
-        indextext = self.get_index_text(modname, name_cls)
-        if indextext:
-            self.indexnode['entries'].append(('single', indextext, node_id, '', None))
+        if 'noindexentry' not in self.options:
+            indextext = self.get_index_text(modname, name_cls)
+            if indextext:
+                self.indexnode['entries'].append(('single', indextext, node_id, '', None))
 
     def before_content(self) -> None:
         """Handle object nesting before content
@@ -576,16 +578,17 @@ def needs_arglist(self) -> bool:
     def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                              signode: desc_signature) -> None:
         super().add_target_and_index(name_cls, sig, signode)
-        modname = self.options.get('module', self.env.ref_context.get('py:module'))
-        node_id = signode['ids'][0]
+        if 'noindexentry' not in self.options:
+            modname = self.options.get('module', self.env.ref_context.get('py:module'))
+            node_id = signode['ids'][0]
 
-        name, cls = name_cls
-        if modname:
-            text = _('%s() (in module %s)') % (name, modname)
-            self.indexnode['entries'].append(('single', text, node_id, '', None))
-        else:
-            text = '%s; %s()' % (pairindextypes['builtin'], name)
-            self.indexnode['entries'].append(('pair', text, node_id, '', None))
+            name, cls = name_cls
+            if modname:
+                text = _('%s() (in module %s)') % (name, modname)
+                self.indexnode['entries'].append(('single', text, node_id, '', None))
+            else:
+                text = '%s; %s()' % (pairindextypes['builtin'], name)
+                self.indexnode['entries'].append(('pair', text, node_id, '', None))
 
     def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
         # add index in own add_target_and_index() instead.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/c.py | 19 | 19 | - | 25 | -
| sphinx/domains/c.py | 3026 | 3029 | 67 | 25 | 29136
| sphinx/domains/c.py | 3101 | 3102 | 67 | 25 | 29136
| sphinx/domains/cpp.py | 6628 | 6628 | - | - | -
| sphinx/domains/cpp.py | 6704 | 6704 | - | - | -
| sphinx/domains/javascript.py | 52 | 52 | - | 24 | -
| sphinx/domains/javascript.py | 123 | 125 | - | 24 | -
| sphinx/domains/python.py | 320 | 320 | - | 6 | -
| sphinx/domains/python.py | 462 | 464 | - | 6 | -
| sphinx/domains/python.py | 579 | 588 | - | 6 | -


## Problem Statement

```
:noindex: prevents cross-referencing
If a `:noindex:` flag is added to a directive, it can't be cross-referenced, and no permalink to it is generated.

The following ReST:
\`\`\`
.. class:: Indexed

.. class:: Unindexed
   :noindex:

\`\`\`
generates the following HTML:
\`\`\`
<dl class="class">
<dt id="Indexed">
<em class="property">class </em><code class="descname">Indexed</code><a class="headerlink" href="#Indexed" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="class">
<dt>
<em class="property">class </em><code class="descname">Unindexed</code></dt>
<dd></dd></dl>
\`\`\`

I would expect `:noindex:` only to result in no index entry, not to prevent cross-referencing or permalinking. The HTML generated for the two class directives should be the same, i.e. the HTML for the Unindexed class should be
\`\`\`
<dl class="class">
<dt id="Unindexed">
<em class="property">class </em><code class="descname">Unindexed</code><a class="headerlink" href="#Unindexed" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>
\`\`\`
- OS: Linux Mint 19.1 (based on Ubuntu 18.04)
- Python version: 3.8.1
- Sphinx version: 3.0.0 (HEAD) but also occurs with Sphinx 2.x, 1.x


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/domains/rst.py | 141 | 172| 356 | 356 | 2470 | 
| 2 | 1 sphinx/domains/rst.py | 92 | 115| 202 | 558 | 2470 | 
| 3 | 2 sphinx/directives/__init__.py | 269 | 307| 273 | 831 | 4965 | 
| 4 | 2 sphinx/domains/rst.py | 36 | 69| 331 | 1162 | 4965 | 
| 5 | 3 sphinx/domains/index.py | 64 | 93| 230 | 1392 | 5899 | 
| 6 | 3 sphinx/directives/__init__.py | 114 | 143| 216 | 1608 | 5899 | 
| 7 | 4 sphinx/domains/__init__.py | 68 | 99| 338 | 1946 | 9410 | 
| 8 | 4 sphinx/domains/index.py | 96 | 130| 281 | 2227 | 9410 | 
| 9 | 4 sphinx/domains/rst.py | 174 | 200| 223 | 2450 | 9410 | 
| 10 | 5 sphinx/directives/other.py | 9 | 40| 238 | 2688 | 12567 | 
| 11 | **6 sphinx/domains/python.py** | 1041 | 1111| 583 | 3271 | 24214 | 
| 12 | **6 sphinx/domains/python.py** | 242 | 260| 214 | 3485 | 24214 | 
| 13 | 6 sphinx/domains/rst.py | 118 | 139| 186 | 3671 | 24214 | 
| 14 | 7 doc/conf.py | 59 | 126| 693 | 4364 | 25768 | 
| 15 | 8 sphinx/roles.py | 576 | 597| 223 | 4587 | 31423 | 
| 16 | 8 sphinx/domains/rst.py | 249 | 258| 134 | 4721 | 31423 | 
| 17 | 9 sphinx/builders/html/__init__.py | 866 | 884| 230 | 4951 | 42648 | 
| 18 | 10 sphinx/addnodes.py | 254 | 353| 580 | 5531 | 45167 | 
| 19 | 10 sphinx/directives/other.py | 289 | 344| 454 | 5985 | 45167 | 
| 20 | 11 sphinx/environment/adapters/indexentries.py | 57 | 102| 583 | 6568 | 46827 | 
| 21 | 12 sphinx/writers/latex.py | 14 | 73| 453 | 7021 | 66531 | 
| 22 | 12 sphinx/domains/rst.py | 260 | 286| 274 | 7295 | 66531 | 
| 23 | 13 sphinx/search/__init__.py | 322 | 357| 344 | 7639 | 70573 | 
| 24 | 13 sphinx/directives/other.py | 88 | 150| 602 | 8241 | 70573 | 
| 25 | 14 sphinx/domains/std.py | 127 | 178| 463 | 8704 | 80787 | 
| 26 | **14 sphinx/domains/python.py** | 982 | 1000| 143 | 8847 | 80787 | 
| 27 | 14 sphinx/domains/std.py | 817 | 877| 546 | 9393 | 80787 | 
| 28 | **14 sphinx/domains/python.py** | 1003 | 1023| 248 | 9641 | 80787 | 
| 29 | 15 sphinx/writers/html.py | 694 | 793| 803 | 10444 | 88153 | 
| 30 | 15 sphinx/domains/rst.py | 230 | 236| 113 | 10557 | 88153 | 
| 31 | 15 sphinx/directives/__init__.py | 52 | 74| 206 | 10763 | 88153 | 
| 32 | 15 sphinx/builders/html/__init__.py | 657 | 684| 270 | 11033 | 88153 | 
| 33 | 15 sphinx/roles.py | 198 | 264| 701 | 11734 | 88153 | 
| 34 | 16 sphinx/directives/code.py | 418 | 482| 658 | 12392 | 92061 | 
| 35 | 16 sphinx/directives/other.py | 347 | 371| 186 | 12578 | 92061 | 
| 36 | **16 sphinx/domains/python.py** | 919 | 979| 521 | 13099 | 92061 | 
| 37 | 16 sphinx/domains/std.py | 535 | 605| 723 | 13822 | 92061 | 
| 38 | 16 sphinx/writers/html.py | 335 | 352| 227 | 14049 | 92061 | 
| 39 | 16 sphinx/directives/code.py | 9 | 31| 146 | 14195 | 92061 | 
| 40 | 16 sphinx/domains/std.py | 68 | 99| 354 | 14549 | 92061 | 
| 41 | 16 sphinx/roles.py | 544 | 573| 328 | 14877 | 92061 | 
| 42 | 17 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 15156 | 94061 | 
| 43 | 17 sphinx/directives/other.py | 187 | 208| 141 | 15297 | 94061 | 
| 44 | 17 sphinx/domains/std.py | 879 | 887| 120 | 15417 | 94061 | 
| 45 | 18 sphinx/ext/autosummary/__init__.py | 137 | 181| 350 | 15767 | 100615 | 
| 46 | 18 sphinx/search/__init__.py | 160 | 189| 194 | 15961 | 100615 | 
| 47 | 18 sphinx/search/__init__.py | 385 | 401| 172 | 16133 | 100615 | 
| 48 | 18 sphinx/writers/latex.py | 507 | 545| 384 | 16517 | 100615 | 
| 49 | 18 sphinx/domains/std.py | 51 | 66| 140 | 16657 | 100615 | 
| 50 | **18 sphinx/domains/python.py** | 11 | 88| 598 | 17255 | 100615 | 
| 51 | 18 sphinx/addnodes.py | 237 | 251| 146 | 17401 | 100615 | 
| 52 | 18 sphinx/domains/std.py | 11 | 48| 323 | 17724 | 100615 | 
| 53 | 19 sphinx/writers/html5.py | 307 | 361| 507 | 18231 | 107518 | 
| 54 | 20 sphinx/builders/latex/transforms.py | 91 | 116| 277 | 18508 | 111697 | 
| 55 | 20 sphinx/writers/html.py | 171 | 231| 562 | 19070 | 111697 | 
| 56 | 20 sphinx/builders/html/__init__.py | 848 | 864| 189 | 19259 | 111697 | 
| 57 | 20 sphinx/domains/rst.py | 203 | 228| 219 | 19478 | 111697 | 
| 58 | 20 sphinx/domains/rst.py | 11 | 33| 161 | 19639 | 111697 | 
| 59 | 21 sphinx/directives/patches.py | 54 | 68| 143 | 19782 | 113310 | 
| 60 | 22 sphinx/util/nodes.py | 11 | 41| 227 | 20009 | 118731 | 
| 61 | 22 sphinx/domains/std.py | 775 | 797| 266 | 20275 | 118731 | 
| 62 | 23 sphinx/writers/texinfo.py | 689 | 750| 607 | 20882 | 130910 | 
| 63 | 23 sphinx/writers/latex.py | 1522 | 1586| 621 | 21503 | 130910 | 
| 64 | **24 sphinx/domains/javascript.py** | 295 | 312| 196 | 21699 | 134925 | 
| 65 | 24 sphinx/writers/html5.py | 631 | 717| 686 | 22385 | 134925 | 
| 66 | 24 sphinx/writers/html.py | 260 | 282| 214 | 22599 | 134925 | 
| **-> 67 <-** | **25 sphinx/domains/c.py** | 2704 | 3482| 6537 | 29136 | 164443 | 
| 68 | 25 sphinx/domains/std.py | 1069 | 1105| 338 | 29474 | 164443 | 
| 69 | 26 sphinx/writers/manpage.py | 313 | 411| 757 | 30231 | 167954 | 
| 70 | 26 sphinx/environment/adapters/indexentries.py | 116 | 157| 417 | 30648 | 167954 | 
| 71 | 27 sphinx/registry.py | 235 | 253| 240 | 30888 | 172526 | 
| 72 | 28 sphinx/builders/_epub_base.py | 330 | 355| 301 | 31189 | 179247 | 
| 73 | 28 sphinx/directives/code.py | 383 | 416| 284 | 31473 | 179247 | 
| 74 | 28 sphinx/domains/std.py | 988 | 1007| 296 | 31769 | 179247 | 
| 75 | 29 sphinx/transforms/references.py | 11 | 68| 362 | 32131 | 179661 | 
| 76 | 29 sphinx/writers/html.py | 354 | 409| 472 | 32603 | 179661 | 
| 77 | 29 sphinx/domains/std.py | 214 | 240| 274 | 32877 | 179661 | 
| 78 | 29 sphinx/writers/html.py | 232 | 258| 318 | 33195 | 179661 | 
| 79 | 29 sphinx/domains/std.py | 906 | 926| 199 | 33394 | 179661 | 
| 80 | 30 doc/development/tutorials/examples/todo.py | 30 | 61| 230 | 33624 | 180499 | 
| 81 | 30 sphinx/writers/texinfo.py | 862 | 966| 788 | 34412 | 180499 | 
| 82 | 30 sphinx/domains/rst.py | 238 | 247| 126 | 34538 | 180499 | 
| 83 | 31 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 34836 | 181760 | 
| 84 | 31 sphinx/domains/std.py | 459 | 532| 630 | 35466 | 181760 | 
| 85 | 31 sphinx/environment/adapters/indexentries.py | 103 | 115| 233 | 35699 | 181760 | 
| 86 | 31 sphinx/writers/texinfo.py | 752 | 860| 813 | 36512 | 181760 | 
| 87 | 31 sphinx/search/__init__.py | 436 | 455| 178 | 36690 | 181760 | 
| 88 | 32 sphinx/transforms/__init__.py | 252 | 272| 192 | 36882 | 184966 | 
| 89 | 33 sphinx/ext/todo.py | 14 | 44| 207 | 37089 | 187683 | 
| 90 | 33 sphinx/writers/html5.py | 232 | 253| 208 | 37297 | 187683 | 


## Missing Patch Files

 * 1: sphinx/domains/c.py
 * 2: sphinx/domains/cpp.py
 * 3: sphinx/domains/javascript.py
 * 4: sphinx/domains/python.py

### Hint

```
The `:noindex:` option is usually used to disable to create a cross-reference target. For example, `.. py:module::` directive goes to create a target and to switch current module (current namespace). And we give `:noindex:` option to one of `py:module:` definition to avoid conflicts.
\`\`\`
# in getcwd.rst
.. py:module:: os
.. py:function:: getcwd()
\`\`\`
\`\`\`
# in system.rst
# use `:noindex:` not to create a cross-reference target to avoid conflicts
.. py:module:: os
   :noindex:

.. py:function:: system()
\`\`\`

Unfortunately, it is strongly coupled with cross-reference system. So this is very difficult problem.
Ideally there would be `:noxref:` for this and `:noindex:` for the index, but I realise that would break backward compatibility :disappointed: 

Perhaps, add new flags `:noxref:` and `:skipindex:` wherever `:noindex:` is allowed. Ignore these flags unless a specific setting is enabled in the environment via `conf.py`, say `enable_noxref`. It defaults to disabled, so current behaviour is maintained exactly. If it _is_ enabled, then:
* If an ambiguity arises during cross-referencing, user is warned to use `:noxref:` for the entry which will not be cross-referenced. Since an entry so marked isn't to be cross-referenced, it won't show up in the index and won't be cross-referencable/permalinkable.
* If a node is marked with `:skipindex:`, no entry for it will be added in the index, but it will still be able to be cross-referenced/permalinked, as long as it is not marked `:noxref:`.

What do you think?
@vsajip Have you found a workaround in the meantime?
> Have you found a workaround in the meantime?

No, unfortunately not. I think what I suggested is the appropriate fix, but waiting for feedback on this from the maintainers.
I believe some of the problem may be related to documentation, for what ``noindex`` actually means.
One way to view it is that we have 3 things in play:
1. Recording a declaration in the internal index in a domain (and show a perma-link).
2. Cross-referencing declarations (requires (1) to have happened).
3. Showing links to declarations in a generated index  (requires (1) to have happened).

For the domains declarations that support ``noindex`` I believe in all cases it suppresses (1), making suppression of  (2) and (3) incidental. From this (somewhat technical) point of view, the name makes sense, and basically makes a declaration not exist (except for a visual side-effect). Though, in hindsight the name ``noindex`` is probably too ambiguous.
We should improve the documentation to make this clear (after checking that this is how all domains handle it (and maybe implement it in the directives that don't support it yet)).

At this point I can not imagine a use case where suppression of (2) but not (1) makes sense.

Do I understand it correctly that you would like to suppress (3) but keep (1) and (2)?
That is not unreasonable, and we should then come up with a good name for such an option.

(We could also invent a new name for what ``noindex`` does and reuse the name, but this is a really heavy breaking change)
> Do I understand it correctly that you would like to suppress (3) but keep (1) and (2)

Well, the ability to include/exclude index items is orthogonal to cross-referencing within the documentation. As you've identified, (1) needs to happen always, so that (2) and (3) are not "incidentally" made impossible. (2) is the most useful IMO (follow an intra-document link for more information about something) and (3) is useful but not as much (mainly used when you know the name of what you're looking for). One might want to suppress (3) because it makes the index too voluminous or confusing to include _everything_ that's indexable, for example. The documenter can control (2) by choosing to use a `:ref:` tag or not, as they see fit, but currently can't exercise fine control over (3).
new to sphinx and all that, so I can't comment on all these cases. I came across this issue because in our documentation, we have automodules in our RST docs and at the same time have a autodoc generated files that are accessible through the search bar. One of the two needs the :noindex: as told by sphinx error message, but this also removes/prevents anchor links. 
Right, (2) and (3) are of course orthogonal. There are definitely use-cases for suppressing all (e.g., writing example declarations), but I guess not necessarily used that often.
What I meant by (2) is a bit different: the user can always simply not use one of the xref roles, as you suggest, but by (2) I meant the ability for an xref to resolve to a given declaration, but still that declaration in the index. I believe this is what @tk0miya meant by the comment that ``:noindex`` is strongly coupled with cross-referencing.

Anyway, I guess we are back to looking for a name for the option the suppressed (3) but not (1)?
> new to sphinx and all that, so I can't comment on all these cases. I came across this issue because in our documentation, we have automodules in our RST docs and at the same time have a autodoc generated files that are accessible through the search bar. One of the two needs the :noindex: as told by sphinx error message, but this also removes/prevents anchor links.

I'm not too familiar with autodoc but it doesn't sound like ``:noindex:`` is the right solution. Rather, it sounds like you are effectively documenting things twice?
> I believe this is what @tk0miya meant by the comment that :noindex is strongly coupled with cross-referencing.

I understood it to mean that in the current code base, you can't separate out (2) and (3) - the only thing you have to prevent something appearing in (3) is `:noindex:`, which disables (1) and hence (3), but also (2).
Thank you for your explanation. Yes, both (2) and (3) depend on (1)'s database (the internal index). And the `:noindex:` option disables (1).

But I understand the idea to disable only (3). So +0 if there is good naming for the option. But we need to keep the name and its behavior of `:noindex:` to keep compatibility.

Note: I still don't understand disabling (2) is really needed. So I don't think to add an option to do that at present.
I agree that there is no strong need to disable (2), except to flag a particular target in the case of ambiguities - which should be a rare case. What about the naming I suggested in [the above comment](https://github.com/sphinx-doc/sphinx/issues/7052#issuecomment-578422370), specifically `:skipindex:`?
I don't find ``skipindex`` sufficiently different from ``noindex``. How about ``hideindex``, ``hideindexentry``, or ``noindexentry``? Hiding something is different from not having it, and skipping the addition of an index entry is literally what we are talking about implementing.  I prefer ``noindexentry`` at the moment.
I think we should use a different word for the new option. We already use the "index" at `:noindex:` option. I thought `:skipindex:` and `:hideindex:` are using the same wording. I guess users suppose them to control the same thing that `:noindex:` controls. But they are different behavior. So I feel "indexentry" is better. +1 for `:noindexentry:`.

Note: Ideally, it would be best if we can rename `:noindex:` to new name. But the option has been used for a long time. So we can't do that. So we should choose better name without using "index" keyword.
I'm fine with `:noindexentry:`, too. We can't completely avoid the word `index`, since we're talking about excluding an entry from the index.
> I thought `:skipindex:` and `:hideindex:` are using the same wording. I guess users suppose them to control the same thing that `:noindex:` controls.

I agree with @tk0miya.

+1 for `:noindexentry:`.

```

## Patch

```diff
diff --git a/sphinx/domains/c.py b/sphinx/domains/c.py
--- a/sphinx/domains/c.py
+++ b/sphinx/domains/c.py
@@ -16,6 +16,7 @@
 
 from docutils import nodes
 from docutils.nodes import Element, Node, TextElement, system_message
+from docutils.parsers.rst import directives
 
 from sphinx import addnodes
 from sphinx.addnodes import pending_xref
@@ -3023,10 +3024,7 @@ class CObject(ObjectDescription):
     ]
 
     option_spec = {
-        # have a dummy option to ensure proper errors on options,
-        # otherwise the option is taken as a continuation of the
-        # argument
-        'dummy': None
+        'noindexentry': directives.flag,
     }
 
     def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
@@ -3098,8 +3096,9 @@ def add_target_and_index(self, ast: ASTDeclaration, sig: str,
             if name not in domain.objects:
                 domain.objects[name] = (domain.env.docname, newestId, self.objtype)
 
-        indexText = self.get_index_text(name)
-        self.indexnode['entries'].append(('single', indexText, newestId, '', None))
+        if 'noindexentry' not in self.options:
+            indexText = self.get_index_text(name)
+            self.indexnode['entries'].append(('single', indexText, newestId, '', None))
 
     @property
     def object_type(self) -> str:
diff --git a/sphinx/domains/cpp.py b/sphinx/domains/cpp.py
--- a/sphinx/domains/cpp.py
+++ b/sphinx/domains/cpp.py
@@ -6625,6 +6625,7 @@ class CPPObject(ObjectDescription):
     ]
 
     option_spec = {
+        'noindexentry': directives.flag,
         'tparam-line-spec': directives.flag,
     }
 
@@ -6701,7 +6702,7 @@ def add_target_and_index(self, ast: ASTDeclaration, sig: str,
             if decl.objectType == 'concept':
                 isInConcept = True
                 break
-        if not isInConcept:
+        if not isInConcept and 'noindexentry' not in self.options:
             strippedName = name
             for prefix in self.env.config.cpp_index_common_prefix:
                 if name.startswith(prefix):
diff --git a/sphinx/domains/javascript.py b/sphinx/domains/javascript.py
--- a/sphinx/domains/javascript.py
+++ b/sphinx/domains/javascript.py
@@ -49,6 +49,11 @@ class JSObject(ObjectDescription):
     #: based on directive nesting
     allow_nesting = False
 
+    option_spec = {
+        'noindex': directives.flag,
+        'noindexentry': directives.flag,
+    }
+
     def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
         """Breaks down construct signatures
 
@@ -120,9 +125,10 @@ def add_target_and_index(self, name_obj: Tuple[str, str], sig: str,
         domain = cast(JavaScriptDomain, self.env.get_domain('js'))
         domain.note_object(fullname, self.objtype, node_id, location=signode)
 
-        indextext = self.get_index_text(mod_name, name_obj)
-        if indextext:
-            self.indexnode['entries'].append(('single', indextext, node_id, '', None))
+        if 'noindexentry' not in self.options:
+            indextext = self.get_index_text(mod_name, name_obj)
+            if indextext:
+                self.indexnode['entries'].append(('single', indextext, node_id, '', None))
 
     def get_index_text(self, objectname: str, name_obj: Tuple[str, str]) -> str:
         name, obj = name_obj
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -317,6 +317,7 @@ class PyObject(ObjectDescription):
     """
     option_spec = {
         'noindex': directives.flag,
+        'noindexentry': directives.flag,
         'module': directives.unchanged,
         'annotation': directives.unchanged,
     }
@@ -459,9 +460,10 @@ def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
         domain = cast(PythonDomain, self.env.get_domain('py'))
         domain.note_object(fullname, self.objtype, node_id, location=signode)
 
-        indextext = self.get_index_text(modname, name_cls)
-        if indextext:
-            self.indexnode['entries'].append(('single', indextext, node_id, '', None))
+        if 'noindexentry' not in self.options:
+            indextext = self.get_index_text(modname, name_cls)
+            if indextext:
+                self.indexnode['entries'].append(('single', indextext, node_id, '', None))
 
     def before_content(self) -> None:
         """Handle object nesting before content
@@ -576,16 +578,17 @@ def needs_arglist(self) -> bool:
     def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                              signode: desc_signature) -> None:
         super().add_target_and_index(name_cls, sig, signode)
-        modname = self.options.get('module', self.env.ref_context.get('py:module'))
-        node_id = signode['ids'][0]
+        if 'noindexentry' not in self.options:
+            modname = self.options.get('module', self.env.ref_context.get('py:module'))
+            node_id = signode['ids'][0]
 
-        name, cls = name_cls
-        if modname:
-            text = _('%s() (in module %s)') % (name, modname)
-            self.indexnode['entries'].append(('single', text, node_id, '', None))
-        else:
-            text = '%s; %s()' % (pairindextypes['builtin'], name)
-            self.indexnode['entries'].append(('pair', text, node_id, '', None))
+            name, cls = name_cls
+            if modname:
+                text = _('%s() (in module %s)') % (name, modname)
+                self.indexnode['entries'].append(('single', text, node_id, '', None))
+            else:
+                text = '%s; %s()' % (pairindextypes['builtin'], name)
+                self.indexnode['entries'].append(('pair', text, node_id, '', None))
 
     def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
         # add index in own add_target_and_index() instead.

```

## Test Patch

```diff
diff --git a/tests/test_domain_c.py b/tests/test_domain_c.py
--- a/tests/test_domain_c.py
+++ b/tests/test_domain_c.py
@@ -10,6 +10,7 @@
 import pytest
 
 from sphinx import addnodes
+from sphinx.addnodes import desc
 from sphinx.domains.c import DefinitionParser, DefinitionError
 from sphinx.domains.c import _max_id, _id_prefix, Symbol
 from sphinx.testing import restructuredtext
@@ -590,3 +591,13 @@ def test_cvar(app):
     domain = app.env.get_domain('c')
     entry = domain.objects.get('PyClass_Type')
     assert entry == ('index', 'c.PyClass_Type', 'var')
+
+
+def test_noindexentry(app):
+    text = (".. c:function:: void f()\n"
+            ".. c:function:: void g()\n"
+            "   :noindexentry:\n")
+    doctree = restructuredtext.parse(app, text)
+    assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
+    assert_node(doctree[0], addnodes.index, entries=[('single', 'f (C function)', 'c.f', '', None)])
+    assert_node(doctree[2], addnodes.index, entries=[])
diff --git a/tests/test_domain_cpp.py b/tests/test_domain_cpp.py
--- a/tests/test_domain_cpp.py
+++ b/tests/test_domain_cpp.py
@@ -14,8 +14,11 @@
 
 import sphinx.domains.cpp as cppDomain
 from sphinx import addnodes
+from sphinx.addnodes import desc
 from sphinx.domains.cpp import DefinitionParser, DefinitionError, NoOldIdError
 from sphinx.domains.cpp import Symbol, _max_id, _id_prefix
+from sphinx.testing import restructuredtext
+from sphinx.testing.util import assert_node
 from sphinx.util import docutils
 
 
@@ -1211,3 +1214,13 @@ def __init__(self, role, root, contents):
     assert any_role.classes == cpp_any_role.classes, expect
     assert any_role.classes == expr_role.content_classes['a'], expect
     assert any_role.classes == texpr_role.content_classes['a'], expect
+
+
+def test_noindexentry(app):
+    text = (".. cpp:function:: void f()\n"
+            ".. cpp:function:: void g()\n"
+            "   :noindexentry:\n")
+    doctree = restructuredtext.parse(app, text)
+    assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
+    assert_node(doctree[0], addnodes.index, entries=[('single', 'f (C++ function)', '_CPPv41fv', '', None)])
+    assert_node(doctree[2], addnodes.index, entries=[])
diff --git a/tests/test_domain_js.py b/tests/test_domain_js.py
--- a/tests/test_domain_js.py
+++ b/tests/test_domain_js.py
@@ -218,3 +218,13 @@ def test_js_data(app):
     assert_node(doctree[0], addnodes.index,
                 entries=[("single", "name (global variable or constant)", "name", "", None)])
     assert_node(doctree[1], addnodes.desc, domain="js", objtype="data", noindex=False)
+
+
+def test_noindexentry(app):
+    text = (".. js:function:: f()\n"
+            ".. js:function:: g()\n"
+            "   :noindexentry:\n")
+    doctree = restructuredtext.parse(app, text)
+    assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
+    assert_node(doctree[0], addnodes.index, entries=[('single', 'f() (built-in function)', 'f', '', None)])
+    assert_node(doctree[2], addnodes.index, entries=[])
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -799,3 +799,19 @@ def test_modindex_common_prefix(app):
     )
 
 
+def test_noindexentry(app):
+    text = (".. py:function:: f()\n"
+            ".. py:function:: g()\n"
+            "   :noindexentry:\n")
+    doctree = restructuredtext.parse(app, text)
+    assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
+    assert_node(doctree[0], addnodes.index, entries=[('pair', 'built-in function; f()', 'f', '', None)])
+    assert_node(doctree[2], addnodes.index, entries=[])
+
+    text = (".. py:class:: f\n"
+            ".. py:class:: g\n"
+            "   :noindexentry:\n")
+    doctree = restructuredtext.parse(app, text)
+    assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
+    assert_node(doctree[0], addnodes.index, entries=[('single', 'f (built-in class)', 'f', '', None)])
+    assert_node(doctree[2], addnodes.index, entries=[])

```


## Code snippets

### 1 - sphinx/domains/rst.py:

Start line: 141, End line: 172

```python
class ReSTDirectiveOption(ReSTMarkup):

    def add_target_and_index(self, name: str, sig: str, signode: desc_signature) -> None:
        domain = cast(ReSTDomain, self.env.get_domain('rst'))

        directive_name = self.current_directive
        if directive_name:
            prefix = '-'.join([self.objtype, directive_name])
            objname = ':'.join([directive_name, name])
        else:
            prefix = self.objtype
            objname = name

        node_id = make_id(self.env, self.state.document, prefix, name)
        signode['ids'].append(node_id)

        # Assign old styled node_id not to break old hyperlinks (if possible)
        # Note: Will be removed in Sphinx-5.0 (RemovedInSphinx50Warning)
        old_node_id = self.make_old_id(name)
        if old_node_id not in self.state.document.ids and old_node_id not in signode['ids']:
            signode['ids'].append(old_node_id)

        self.state.document.note_explicit_target(signode)
        domain.note_object(self.objtype, objname, node_id, location=signode)

        if directive_name:
            key = name[0].upper()
            pair = [_('%s (directive)') % directive_name,
                    _(':%s: (directive option)') % name]
            self.indexnode['entries'].append(('pair', '; '.join(pair), node_id, '', key))
        else:
            key = name[0].upper()
            text = _(':%s: (directive option)') % name
            self.indexnode['entries'].append(('single', text, node_id, '', key))
```
### 2 - sphinx/domains/rst.py:

Start line: 92, End line: 115

```python
class ReSTDirective(ReSTMarkup):
    """
    Description of a reST directive.
    """
    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        name, args = parse_directive(sig)
        desc_name = '.. %s::' % name
        signode += addnodes.desc_name(desc_name, desc_name)
        if len(args) > 0:
            signode += addnodes.desc_addname(args, args)
        return name

    def get_index_text(self, objectname: str, name: str) -> str:
        return _('%s (directive)') % name

    def before_content(self) -> None:
        if self.names:
            directives = self.env.ref_context.setdefault('rst:directives', [])
            directives.append(self.names[0])

    def after_content(self) -> None:
        directives = self.env.ref_context.setdefault('rst:directives', [])
        if directives:
            directives.pop()
```
### 3 - sphinx/directives/__init__.py:

Start line: 269, End line: 307

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
### 4 - sphinx/domains/rst.py:

Start line: 36, End line: 69

```python
class ReSTMarkup(ObjectDescription):
    """
    Description of generic reST markup.
    """

    def add_target_and_index(self, name: str, sig: str, signode: desc_signature) -> None:
        node_id = make_id(self.env, self.state.document, self.objtype, name)
        signode['ids'].append(node_id)

        # Assign old styled node_id not to break old hyperlinks (if possible)
        # Note: Will be removed in Sphinx-5.0 (RemovedInSphinx50Warning)
        old_node_id = self.make_old_id(name)
        if old_node_id not in self.state.document.ids and old_node_id not in signode['ids']:
            signode['ids'].append(old_node_id)

        self.state.document.note_explicit_target(signode)

        domain = cast(ReSTDomain, self.env.get_domain('rst'))
        domain.note_object(self.objtype, name, node_id, location=signode)

        indextext = self.get_index_text(self.objtype, name)
        if indextext:
            self.indexnode['entries'].append(('single', indextext, node_id, '', None))

    def get_index_text(self, objectname: str, name: str) -> str:
        return ''

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for reST markups.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return self.objtype + '-' + name
```
### 5 - sphinx/domains/index.py:

Start line: 64, End line: 93

```python
class IndexDirective(SphinxDirective):
    """
    Directive to add entries to the index.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'name': directives.unchanged,
    }

    def run(self) -> List[Node]:
        arguments = self.arguments[0].split('\n')

        if 'name' in self.options:
            targetname = self.options['name']
            targetnode = nodes.target('', '', names=[targetname])
        else:
            targetid = 'index-%s' % self.env.new_serialno('index')
            targetnode = nodes.target('', '', ids=[targetid])

        self.state.document.note_explicit_target(targetnode)
        indexnode = addnodes.index()
        indexnode['entries'] = []
        indexnode['inline'] = False
        self.set_source_info(indexnode)
        for entry in arguments:
            indexnode['entries'].extend(process_index_entry(entry, targetnode['ids'][0]))
        return [indexnode, targetnode]
```
### 6 - sphinx/directives/__init__.py:

Start line: 114, End line: 143

```python
class ObjectDescription(SphinxDirective):

    def add_target_and_index(self, name: Any, sig: str, signode: desc_signature) -> None:
        """
        Add cross-reference IDs and entries to self.indexnode, if applicable.

        *name* is whatever :meth:`handle_signature()` returned.
        """
        return  # do nothing by default

    def before_content(self) -> None:
        """
        Called before parsing content. Used to set information about the current
        directive context on the build environment.
        """
        pass

    def transform_content(self, contentnode: addnodes.desc_content) -> None:
        """
        Called after creating the content through nested parsing,
        but before the ``object-description-transform`` event is emitted,
        and before the info-fields are transformed.
        Can be used to manipulate the content.
        """
        pass

    def after_content(self) -> None:
        """
        Called after parsing content. Used to reset information about the
        current directive context on the build environment.
        """
        pass
```
### 7 - sphinx/domains/__init__.py:

Start line: 68, End line: 99

```python
class Index:
    """
    An Index is the description for a domain-specific index.  To add an index to
    a domain, subclass Index, overriding the three name attributes:

    * `name` is an identifier used for generating file names.
      It is also used for a hyperlink target for the index. Therefore, users can
      refer the index page using ``ref`` role and a string which is combined
      domain name and ``name`` attribute (ex. ``:ref:`py-modindex```).
    * `localname` is the section title for the index.
    * `shortname` is a short name for the index, for use in the relation bar in
      HTML output.  Can be empty to disable entries in the relation bar.

    and providing a :meth:`generate()` method.  Then, add the index class to
    your domain's `indices` list.  Extensions can add indices to existing
    domains using :meth:`~sphinx.application.Sphinx.add_index_to_domain()`.

    .. versionchanged:: 3.0

       Index pages can be referred by domain name and index name via
       :rst:role:`ref` role.
    """

    name = None  # type: str
    localname = None  # type: str
    shortname = None  # type: str

    def __init__(self, domain: "Domain") -> None:
        if self.name is None or self.localname is None:
            raise SphinxError('Index subclass %s has no valid name or localname'
                              % self.__class__.__name__)
        self.domain = domain
```
### 8 - sphinx/domains/index.py:

Start line: 96, End line: 130

```python
class IndexRole(ReferenceRole):
    def run(self) -> Tuple[List[Node], List[system_message]]:
        target_id = 'index-%s' % self.env.new_serialno('index')
        if self.has_explicit_title:
            # if an explicit target is given, process it as a full entry
            title = self.title
            entries = process_index_entry(self.target, target_id)
        else:
            # otherwise we just create a single entry
            if self.target.startswith('!'):
                title = self.title[1:]
                entries = [('single', self.target[1:], target_id, 'main', None)]
            else:
                title = self.title
                entries = [('single', self.target, target_id, '', None)]

        index = addnodes.index(entries=entries)
        target = nodes.target('', '', ids=[target_id])
        text = nodes.Text(title, title)
        self.set_source_info(index)
        return [index, target, text], []


def setup(app: "Sphinx") -> Dict[str, Any]:
    app.add_domain(IndexDomain)
    app.add_directive('index', IndexDirective)
    app.add_role('index', IndexRole())

    return {
        'version': 'builtin',
        'env_version': 1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 9 - sphinx/domains/rst.py:

Start line: 174, End line: 200

```python
class ReSTDirectiveOption(ReSTMarkup):

    @property
    def current_directive(self) -> str:
        directives = self.env.ref_context.get('rst:directives')
        if directives:
            return directives[-1]
        else:
            return ''

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for directive options.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return '-'.join([self.objtype, self.current_directive, name])


class ReSTRole(ReSTMarkup):
    """
    Description of a reST role.
    """
    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        signode += addnodes.desc_name(':%s:' % sig, ':%s:' % sig)
        return sig

    def get_index_text(self, objectname: str, name: str) -> str:
        return _('%s (role)') % name
```
### 10 - sphinx/directives/other.py:

Start line: 9, End line: 40

```python
import re
from typing import Any, Dict, List
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.directives.misc import Class
from docutils.parsers.rst.directives.misc import Include as BaseInclude

from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
from sphinx.domains.changeset import VersionChange  # NOQA  # for compatibility
from sphinx.locale import _
from sphinx.util import url_re, docname_join
from sphinx.util.docutils import SphinxDirective
from sphinx.util.matching import Matcher, patfilter
from sphinx.util.nodes import explicit_title_re

if False:
    # For type annotation
    from sphinx.application import Sphinx


glob_re = re.compile(r'.*[*?\[].*')


def int_or_nothing(argument: str) -> int:
    if not argument:
        return 999
    return int(argument)
```
### 11 - sphinx/domains/python.py:

Start line: 1041, End line: 1111

```python
class PythonModuleIndex(Index):
    """
    Index subclass to provide the Python module index.
    """

    name = 'modindex'
    localname = _('Python Module Index')
    shortname = _('modules')

    def generate(self, docnames: Iterable[str] = None
                 ) -> Tuple[List[Tuple[str, List[IndexEntry]]], bool]:
        content = {}  # type: Dict[str, List[IndexEntry]]
        # list of prefixes to ignore
        ignores = None  # type: List[str]
        ignores = self.domain.env.config['modindex_common_prefix']  # type: ignore
        ignores = sorted(ignores, key=len, reverse=True)
        # list of all modules, sorted by module name
        modules = sorted(self.domain.data['modules'].items(),
                         key=lambda x: x[0].lower())
        # sort out collapsable modules
        prev_modname = ''
        num_toplevels = 0
        for modname, (docname, node_id, synopsis, platforms, deprecated) in modules:
            if docnames and docname not in docnames:
                continue

            for ignore in ignores:
                if modname.startswith(ignore):
                    modname = modname[len(ignore):]
                    stripped = ignore
                    break
            else:
                stripped = ''

            # we stripped the whole module name?
            if not modname:
                modname, stripped = stripped, ''

            entries = content.setdefault(modname[0].lower(), [])

            package = modname.split('.')[0]
            if package != modname:
                # it's a submodule
                if prev_modname == package:
                    # first submodule - make parent a group head
                    if entries:
                        last = entries[-1]
                        entries[-1] = IndexEntry(last[0], 1, last[2], last[3],
                                                 last[4], last[5], last[6])
                elif not prev_modname.startswith(package):
                    # submodule without parent in list, add dummy entry
                    entries.append(IndexEntry(stripped + package, 1, '', '', '', '', ''))
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0

            qualifier = _('Deprecated') if deprecated else ''
            entries.append(IndexEntry(stripped + modname, subtype, docname,
                                      node_id, platforms, qualifier, synopsis))
            prev_modname = modname

        # apply heuristics when to collapse modindex at page load:
        # only collapse if number of toplevel modules is larger than
        # number of submodules
        collapse = len(modules) - num_toplevels < num_toplevels

        # sort by first letter
        sorted_content = sorted(content.items())

        return sorted_content, collapse
```
### 12 - sphinx/domains/python.py:

Start line: 242, End line: 260

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
### 26 - sphinx/domains/python.py:

Start line: 982, End line: 1000

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
### 28 - sphinx/domains/python.py:

Start line: 1003, End line: 1023

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
### 36 - sphinx/domains/python.py:

Start line: 919, End line: 979

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
### 50 - sphinx/domains/python.py:

Start line: 11, End line: 88

```python
import builtins
import inspect
import re
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple
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

ObjectEntry = NamedTuple('ObjectEntry', [('docname', str),
                                         ('node_id', str),
                                         ('objtype', str)])
ModuleEntry = NamedTuple('ModuleEntry', [('docname', str),
                                         ('node_id', str),
                                         ('synopsis', str),
                                         ('platform', str),
                                         ('deprecated', bool)])


def type_to_xref(text: str) -> addnodes.pending_xref:
    """Convert a type string to a cross reference node."""
    if text == 'None':
        reftype = 'obj'
    else:
        reftype = 'class'

    return pending_xref('', nodes.Text(text),
                        refdomain='py', reftype=reftype, reftarget=text)
```
### 64 - sphinx/domains/javascript.py:

Start line: 295, End line: 312

```python
class JSXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        # basically what sphinx.domains.python.PyXRefRole does
        refnode['js:object'] = env.ref_context.get('js:object')
        refnode['js:module'] = env.ref_context.get('js:module')
        if not has_explicit_title:
            title = title.lstrip('.')
            target = target.lstrip('~')
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        return title, target
```
### 67 - sphinx/domains/c.py:

Start line: 2704, End line: 3482

```python
class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    _simple_fundamental_types =
    # ... other code

    def _parse_declarator(self, named: Union[bool, str], paramMode: str,
                          typed: bool = True) -> ASTDeclarator:
        # 'typed' here means 'parse return type stuff'
        if paramMode not in ('type', 'function'):
            raise Exception(
                "Internal error, unknown paramMode '%s'." % paramMode)
        prevErrors = []
        self.skip_ws()
        if typed and self.skip_string('*'):
            self.skip_ws()
            restrict = False
            volatile = False
            const = False
            attrs = []
            while 1:
                if not restrict:
                    restrict = self.skip_word_and_ws('restrict')
                    if restrict:
                        continue
                if not volatile:
                    volatile = self.skip_word_and_ws('volatile')
                    if volatile:
                        continue
                if not const:
                    const = self.skip_word_and_ws('const')
                    if const:
                        continue
                attr = self._parse_attribute()
                if attr is not None:
                    attrs.append(attr)
                    continue
                break
            next = self._parse_declarator(named, paramMode, typed)
            return ASTDeclaratorPtr(next=next,
                                    restrict=restrict, volatile=volatile, const=const,
                                    attrs=attrs)
        if typed and self.current_char == '(':  # note: peeking, not skipping
            # maybe this is the beginning of params, try that first,
            # otherwise assume it's noptr->declarator > ( ptr-declarator )
            pos = self.pos
            try:
                # assume this is params
                res = self._parse_declarator_name_suffix(named, paramMode,
                                                         typed)
                return res
            except DefinitionError as exParamQual:
                msg = "If declarator-id with parameters"
                if paramMode == 'function':
                    msg += " (e.g., 'void f(int arg)')"
                prevErrors.append((exParamQual, msg))
                self.pos = pos
                try:
                    assert self.current_char == '('
                    self.skip_string('(')
                    # TODO: hmm, if there is a name, it must be in inner, right?
                    # TODO: hmm, if there must be parameters, they must b
                    # inside, right?
                    inner = self._parse_declarator(named, paramMode, typed)
                    if not self.skip_string(')'):
                        self.fail("Expected ')' in \"( ptr-declarator )\"")
                    next = self._parse_declarator(named=False,
                                                  paramMode="type",
                                                  typed=typed)
                    return ASTDeclaratorParen(inner=inner, next=next)
                except DefinitionError as exNoPtrParen:
                    self.pos = pos
                    msg = "If parenthesis in noptr-declarator"
                    if paramMode == 'function':
                        msg += " (e.g., 'void (*f(int arg))(double)')"
                    prevErrors.append((exNoPtrParen, msg))
                    header = "Error in declarator"
                    raise self._make_multi_error(prevErrors, header) from exNoPtrParen
        pos = self.pos
        try:
            return self._parse_declarator_name_suffix(named, paramMode, typed)
        except DefinitionError as e:
            self.pos = pos
            prevErrors.append((e, "If declarator-id"))
            header = "Error in declarator or parameters"
            raise self._make_multi_error(prevErrors, header) from e

    def _parse_initializer(self, outer: str = None, allowFallback: bool = True
                           ) -> ASTInitializer:
        self.skip_ws()
        if outer == 'member' and False:  # TODO
            bracedInit = self._parse_braced_init_list()
            if bracedInit is not None:
                return ASTInitializer(bracedInit, hasAssign=False)

        if not self.skip_string('='):
            return None

        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit)

        if outer == 'member':
            fallbackEnd = []  # type: List[str]
        elif outer is None:  # function parameter
            fallbackEnd = [',', ')']
        else:
            self.fail("Internal error, initializer for outer '%s' not "
                      "implemented." % outer)

        def parser():
            return self._parse_assignment_expression()

        value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
        return ASTInitializer(value)

    def _parse_type(self, named: Union[bool, str], outer: str = None) -> ASTType:
        """
        named=False|'maybe'|True: 'maybe' is e.g., for function objects which
        doesn't need to name the arguments
        """
        if outer:  # always named
            if outer not in ('type', 'member', 'function'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
            assert named

        if outer == 'type':
            # We allow type objects to just be a name.
            prevErrors = []
            startPos = self.pos
            # first try without the type
            try:
                declSpecs = self._parse_decl_specs(outer=outer, typed=False)
                decl = self._parse_declarator(named=True, paramMode=outer,
                                              typed=False)
                self.assert_end(allowSemicolon=True)
            except DefinitionError as exUntyped:
                desc = "If just a name"
                prevErrors.append((exUntyped, desc))
                self.pos = startPos
                try:
                    declSpecs = self._parse_decl_specs(outer=outer)
                    decl = self._parse_declarator(named=True, paramMode=outer)
                except DefinitionError as exTyped:
                    self.pos = startPos
                    desc = "If typedef-like declaration"
                    prevErrors.append((exTyped, desc))
                    # Retain the else branch for easier debugging.
                    # TODO: it would be nice to save the previous stacktrace
                    #       and output it here.
                    if True:
                        header = "Type must be either just a name or a "
                        header += "typedef-like declaration."
                        raise self._make_multi_error(prevErrors, header) from exTyped
                    else:
                        # For testing purposes.
                        # do it again to get the proper traceback (how do you
                        # reliably save a traceback when an exception is
                        # constructed?)
                        self.pos = startPos
                        typed = True
                        declSpecs = self._parse_decl_specs(outer=outer, typed=typed)
                        decl = self._parse_declarator(named=True, paramMode=outer,
                                                      typed=typed)
        elif outer == 'function':
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=True, paramMode=outer)
        else:
            paramMode = 'type'
            if outer == 'member':  # i.e., member
                named = True
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=named, paramMode=paramMode)
        return ASTType(declSpecs, decl)

    def _parse_type_with_init(self, named: Union[bool, str], outer: str) -> ASTTypeWithInit:
        if outer:
            assert outer in ('type', 'member', 'function')
        type = self._parse_type(outer=outer, named=named)
        init = self._parse_initializer(outer=outer)
        return ASTTypeWithInit(type, init)

    def _parse_macro(self) -> ASTMacro:
        self.skip_ws()
        ident = self._parse_nested_name()
        if ident is None:
            self.fail("Expected identifier in macro definition.")
        self.skip_ws()
        if not self.skip_string_and_ws('('):
            return ASTMacro(ident, None)
        if self.skip_string(')'):
            return ASTMacro(ident, [])
        args = []
        while 1:
            self.skip_ws()
            if self.skip_string('...'):
                args.append(ASTMacroParameter(None, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            if not self.match(identifier_re):
                self.fail("Expected identifier in macro parameters.")
            nn = ASTNestedName([ASTIdentifier(self.matched_text)], rooted=False)
            arg = ASTMacroParameter(nn)
            args.append(arg)
            self.skip_ws()
            if self.skip_string_and_ws(','):
                continue
            elif self.skip_string_and_ws(')'):
                break
            else:
                self.fail("Expected identifier, ')', or ',' in macro parameter list.")
        return ASTMacro(ident, args)

    def _parse_struct(self) -> ASTStruct:
        name = self._parse_nested_name()
        return ASTStruct(name)

    def _parse_union(self) -> ASTUnion:
        name = self._parse_nested_name()
        return ASTUnion(name)

    def _parse_enum(self) -> ASTEnum:
        name = self._parse_nested_name()
        return ASTEnum(name)

    def _parse_enumerator(self) -> ASTEnumerator:
        name = self._parse_nested_name()
        self.skip_ws()
        init = None
        if self.skip_string('='):
            self.skip_ws()

            def parser() -> ASTExpression:
                return self._parse_constant_expression()

            initVal = self._parse_expression_fallback([], parser)
            init = ASTInitializer(initVal)
        return ASTEnumerator(name, init)

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('function', 'member',
                              'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('function', 'member', 'var',
                                 'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)

        declaration = None  # type: Any
        if objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'function':
            declaration = self._parse_type(named=True, outer='function')
        elif objectType == 'macro':
            declaration = self._parse_macro()
        elif objectType == 'struct':
            declaration = self._parse_struct()
        elif objectType == 'union':
            declaration = self._parse_union()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        elif objectType == 'type':
            declaration = self._parse_type(named=True, outer='type')
        else:
            assert False
        if objectType != 'macro':
            self.skip_ws()
            semicolon = self.skip_string(';')
        else:
            semicolon = False
        return ASTDeclaration(objectType, directiveType, declaration, semicolon)

    def parse_namespace_object(self) -> ASTNestedName:
        return self._parse_nested_name()

    def parse_xref_object(self) -> ASTNestedName:
        name = self._parse_nested_name()
        # if there are '()' left, just skip them
        self.skip_ws()
        self.skip_string('()')
        self.assert_end()
        return name

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        res = None  # type: Union[ASTExpression, ASTType]
        try:
            res = self._parse_expression()
            self.skip_ws()
            self.assert_end()
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                res = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
            except DefinitionError as exType:
                header = "Error when parsing (type) expression."
                errs = []
                errs.append((exExpr, "If expression"))
                errs.append((exType, "If type"))
                raise self._make_multi_error(errs, header) from exType
        return res


def _make_phony_error_name() -> ASTNestedName:
    return ASTNestedName([ASTIdentifier("PhonyNameDueToError")], rooted=False)


class CObject(ObjectDescription):
    """
    Description of a C language object.
    """

    doc_field_types = [
        TypedField('parameter', label=_('Parameters'),
                   names=('param', 'parameter', 'arg', 'argument'),
                   typerolename='type', typenames=('type',)),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
        Field('returntype', label=_('Return type'), has_arg=False,
              names=('rtype',)),
    ]

    option_spec = {
        # have a dummy option to ensure proper errors on options,
        # otherwise the option is taken as a continuation of the
        # argument
        'dummy': None
    }

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        # find the parent, if it exists && is an enum
        #                  then add the name to the parent scope
        symbol = ast.symbol
        assert symbol
        assert symbol.ident is not None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            # TODO: we could warn, but it is somewhat equivalent to
            # enumeratorss, without the enum
            return  # no parent
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            # the parent is not explicitly declared
            # TODO: we could warn, but?
            return
        if parentDecl.objectType != 'enum':
            # TODO: maybe issue a warning, enumerators in non-enums is weird,
            # but it is somewhat equivalent to enumeratorss, without the enum
            return
        if parentDecl.directiveType != 'enum':
            return

        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.ident, matchSelf=False, recurseInAnon=True,
                                         searchInSiblings=False)
        if s is not None:
            # something is already declared with that name
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, ident=symbol.ident,
               declaration=declClone,
               docname=self.env.docname)

    def add_target_and_index(self, ast: ASTDeclaration, sig: str,
                             signode: TextElement) -> None:
        ids = []
        for i in range(1, _max_id + 1):
            try:
                id = ast.get_id(version=i)
                ids.append(id)
            except NoOldIdError:
                assert i < _max_id
        # let's keep the newest first
        ids = list(reversed(ids))
        newestId = ids[0]
        assert newestId  # shouldn't be None

        name = ast.symbol.get_full_nested_name().get_display_string().lstrip('.')
        if newestId not in self.state.document.ids:
            # always add the newest id
            assert newestId
            signode['ids'].append(newestId)
            # only add compatibility ids when there are no conflicts
            for id in ids[1:]:
                if not id:  # is None when the element didn't exist in that version
                    continue
                if id not in self.state.document.ids:
                    signode['ids'].append(id)

            self.state.document.note_explicit_target(signode)

            domain = cast(CDomain, self.env.get_domain('c'))
            if name not in domain.objects:
                domain.objects[name] = (domain.env.docname, newestId, self.objtype)

        indexText = self.get_index_text(name)
        self.indexnode['entries'].append(('single', indexText, newestId, '', None))

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def describe_signature(self, signode: TextElement, ast: Any, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env  # from ObjectDescription.run
        if 'c:parent_symbol' not in env.temp_data:
            root = env.domaindata['c']['root_symbol']
            env.temp_data['c:parent_symbol'] = root
            env.ref_context['c:parent_key'] = root.get_lookup_key()

        # When multiple declarations are made in the same directive
        # they need to know about each other to provide symbol lookup for function parameters.
        # We use last_symbol to store the latest added declaration in a directive.
        env.temp_data['c:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: TextElement) -> ASTDeclaration:
        parentSymbol = self.env.temp_data['c:parent_symbol']  # type: Symbol

        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            ast = self.parse_definition(parser)
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=signode)
            # It is easier to assume some phony name than handling the error in
            # the possibly inner declarations.
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['c:last_symbol'] = symbol
            raise ValueError from e

        try:
            symbol = parentSymbol.add_declaration(ast, docname=self.env.docname)
            # append the new declaration to the sibling list
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['c:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['c:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            # Assume we are actually in the old symbol,
            # instead of the newly created duplicate.
            self.env.temp_data['c:last_symbol'] = e.symbol
            msg = __("Duplicate C declaration, also defined in '%s'.\n"
                     "Declaration is '%s'.")
            msg = msg % (e.symbol.docname, sig)
            logger.warning(msg, location=signode)

        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)

        # note: handle_signature may be called multiple time per directive,
        # if it has multiple signatures, so don't mess with the original options.
        options = dict(self.options)
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol = self.env.temp_data['c:last_symbol']  # type: Symbol
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['c:parent_symbol']
        self.oldParentKey = self.env.ref_context['c:parent_key']  # type: LookupKey
        self.env.temp_data['c:parent_symbol'] = lastSymbol
        self.env.ref_context['c:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['c:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['c:parent_key'] = self.oldParentKey

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for C objects.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return 'c.' + name


class CMemberObject(CObject):
    object_type = 'member'

    @property
    def display_object_type(self) -> str:
        # the distinction between var and member is only cosmetic
        assert self.objtype in ('member', 'var')
        return self.objtype


class CFunctionObject(CObject):
    object_type = 'function'


class CMacroObject(CObject):
    object_type = 'macro'


class CStructObject(CObject):
    object_type = 'struct'


class CUnionObject(CObject):
    object_type = 'union'


class CEnumObject(CObject):
    object_type = 'enum'


class CEnumeratorObject(CObject):
    object_type = 'enumerator'


class CTypeObject(CObject):
    object_type = 'type'


class CNamespaceObject(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting stuff in
    namespace foo.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        rootSymbol = self.env.domaindata['c']['root_symbol']
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            symbol = rootSymbol
            stack = []  # type: List[Symbol]
        else:
            parser = DefinitionParser(self.arguments[0],
                                      location=self.get_source_info(),
                                      config=self.env.config)
            try:
                name = parser.parse_namespace_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=self.get_source_info())
                name = _make_phony_error_name()
            symbol = rootSymbol.add_name(name)
            stack = [symbol]
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePushObject(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            return []
        parser = DefinitionParser(self.arguments[0],
                                  location=self.get_source_info(),
                                  config=self.env.config)
        try:
            name = parser.parse_namespace_object()
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=self.get_source_info())
            name = _make_phony_error_name()
        oldParent = self.env.temp_data.get('c:parent_symbol', None)
        if not oldParent:
            oldParent = self.env.domaindata['c']['root_symbol']
        symbol = oldParent.add_name(name)
        stack = self.env.temp_data.get('c:namespace_stack', [])
        stack.append(symbol)
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePopObject(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        stack = self.env.temp_data.get('c:namespace_stack', None)
        if not stack or len(stack) == 0:
            logger.warning("C namespace pop on empty stack. Defaulting to gobal scope.",
                           location=self.get_source_info())
            stack = []
        else:
            stack.pop()
        if len(stack) > 0:
            symbol = stack[-1]
        else:
            symbol = self.env.domaindata['c']['root_symbol']
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['cp:parent_key'] = symbol.get_lookup_key()
        return []


class AliasNode(nodes.Element):
    def __init__(self, sig: str, env: "BuildEnvironment" = None,
                 parentKey: LookupKey = None) -> None:
        super().__init__()
        self.sig = sig
        if env is not None:
            if 'c:parent_symbol' not in env.temp_data:
                root = env.domaindata['c']['root_symbol']
                env.temp_data['c:parent_symbol'] = root
            self.parentKey = env.temp_data['c:parent_symbol'].get_lookup_key()
        else:
            assert parentKey is not None
            self.parentKey = parentKey

    def copy(self: T) -> T:
        return self.__class__(self.sig, env=None, parentKey=self.parentKey)  # type: ignore


class AliasTransform(SphinxTransform):
    default_priority = ReferencesResolver.default_priority - 1

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.traverse(AliasNode):
            sig = node.sig
            parentKey = node.parentKey
            try:
                parser = DefinitionParser(sig, location=node,
                                          config=self.env.config)
                name = parser.parse_xref_object()
            except DefinitionError as e:
                logger.warning(e, location=node)
                name = None

            if name is None:
                # could not be parsed, so stop here
                signode = addnodes.desc_signature(sig, '')
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                node.replace_self(signode)
                continue

            rootSymbol = self.env.domains['c'].data['root_symbol']  # type: Symbol
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", sig)
                print("ParentKey: ", parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there

            s = parentSymbol.find_declaration(
                name, 'any',
                matchSelf=True, recurseInAnon=True)
            if s is None:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)

                logger.warning("Could not find C declaration for alias '%s'." % name,
                               location=node)
                node.replace_self(signode)
            else:
                nodes = []
                options = dict()  # type: ignore
                signode = addnodes.desc_signature(sig, '')
                nodes.append(signode)
                s.declaration.describe_signature(signode, 'markName', self.env, options)
                node.replace_self(nodes)


class CAliasObject(ObjectDescription):
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = '', self.name

        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        # 'desctype' is a backwards compatible attribute
        node['objtype'] = node['desctype'] = self.objtype
        node['noindex'] = True

        self.names = []  # type: List[str]
        signatures = self.get_signatures()
        for i, sig in enumerate(signatures):
            node.append(AliasNode(sig, env=self.env))

        contentnode = addnodes.desc_content()
        node.append(contentnode)
        self.before_content()
        self.state.nested_parse(self.content, self.content_offset, contentnode)
        self.env.temp_data['object'] = None
        self.after_content()
        return [node]


class CXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)

        if not has_explicit_title:
            # major hax: replace anon names via simple string manipulation.
            # Can this actually fail?
            title = anon_identifier_re.sub("[anonymous]", str(title))

        if not has_explicit_title:
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        return title, target


class CExprRole(SphinxRole):
    def __init__(self, asCode: bool) -> None:
        super().__init__()
        if asCode:
            # render the expression as inline code
            self.class_type = 'c-expr'
            self.node_type = nodes.literal  # type: Type[TextElement]
        else:
            # render the expression as inline text
            self.class_type = 'c-texpr'
            self.node_type = nodes.inline

    def run(self) -> Tuple[List[Node], List[system_message]]:
        text = self.text.replace('\n', ' ')
        parser = DefinitionParser(text, location=self.get_source_info(),
                                  config=self.env.config)
        # attempt to mimic XRefRole classes, except that...
        classes = ['xref', 'c', self.class_type]
        try:
            ast = parser.parse_expression()
        except DefinitionError as ex:
            logger.warning('Unparseable C expression: %r\n%s', text, ex,
                           location=self.get_source_info())
            # see below
            return [self.node_type(text, text, classes=classes)], []
        parentSymbol = self.env.temp_data.get('cpp:parent_symbol', None)
        if parentSymbol is None:
            parentSymbol = self.env.domaindata['c']['root_symbol']
        # ...most if not all of these classes should really apply to the individual references,
        # not the container node
        signode = self.node_type(classes=classes)
        ast.describe_signature(signode, 'markType', self.env, parentSymbol)
        return [signode], []
```
