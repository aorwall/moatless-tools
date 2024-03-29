# sphinx-doc__sphinx-8658

| **sphinx-doc/sphinx** | `40f2c832ff3ce2d908b0d8bace3e1f6698eed712` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 359 |
| **Any found context length** | 164 |
| **Avg pos** | 24.5 |
| **Min pos** | 1 |
| **Max pos** | 43 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/napoleon/__init__.py b/sphinx/ext/napoleon/__init__.py
--- a/sphinx/ext/napoleon/__init__.py
+++ b/sphinx/ext/napoleon/__init__.py
@@ -253,10 +253,15 @@ def __unicode__(self):
           * To create a custom "generic" section, just pass a string.
           * To create an alias for an existing section, pass a tuple containing the
             alias name and the original, in that order.
+          * To create a custom section that displays like the parameters or returns
+            section, pass a tuple containing the custom section name and a string
+            value, "params_style" or "returns_style".
 
         If an entry is just a string, it is interpreted as a header for a generic
         section. If the entry is a tuple/list/indexed container, the first entry
-        is the name of the section, the second is the section key to emulate.
+        is the name of the section, the second is the section key to emulate. If the
+        second entry value is "params_style" or "returns_style", the custom section
+        will be displayed like the parameters section or returns section.
 
     napoleon_attr_annotations : :obj:`bool` (Defaults to True)
         Use the type annotations of class attributes that are documented in the docstring
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -549,11 +549,18 @@ def _load_custom_sections(self) -> None:
                     self._sections[entry.lower()] = self._parse_custom_generic_section
                 else:
                     # otherwise, assume entry is container;
-                    # [0] is new section, [1] is the section to alias.
-                    # in the case of key mismatch, just handle as generic section.
-                    self._sections[entry[0].lower()] = \
-                        self._sections.get(entry[1].lower(),
-                                           self._parse_custom_generic_section)
+                    if entry[1] == "params_style":
+                        self._sections[entry[0].lower()] = \
+                            self._parse_custom_params_style_section
+                    elif entry[1] == "returns_style":
+                        self._sections[entry[0].lower()] = \
+                            self._parse_custom_returns_style_section
+                    else:
+                        # [0] is new section, [1] is the section to alias.
+                        # in the case of key mismatch, just handle as generic section.
+                        self._sections[entry[0].lower()] = \
+                            self._sections.get(entry[1].lower(),
+                                               self._parse_custom_generic_section)
 
     def _parse(self) -> None:
         self._parsed_lines = self._consume_empty()
@@ -641,6 +648,13 @@ def _parse_custom_generic_section(self, section: str) -> List[str]:
         # for now, no admonition for simple custom sections
         return self._parse_generic_section(section, False)
 
+    def _parse_custom_params_style_section(self, section: str) -> List[str]:
+        return self._format_fields(section, self._consume_fields())
+
+    def _parse_custom_returns_style_section(self, section: str) -> List[str]:
+        fields = self._consume_returns_section()
+        return self._format_fields(section, fields)
+
     def _parse_usage_section(self, section: str) -> List[str]:
         header = ['.. rubric:: Usage:', '']
         block = ['.. code-block:: python', '']

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/napoleon/__init__.py | 256 | 256 | 43 | 2 | 13283
| sphinx/ext/napoleon/docstring.py | 552 | 556 | 1 | 1 | 164
| sphinx/ext/napoleon/docstring.py | 644 | 644 | 2 | 1 | 359


## Problem Statement

```
Napoleon: more custom docstring section styles
Although the `napoleon_custom_sections` option help renders custom docstring section, the style is inconsistent with the rest of the doc.

For example, I have a custom docstring section `Side Effect`. I would like it to be displayed as `returns` or `parameters` docstring section. However, `napoleon_custom_sections` option rendesr `Side Effect` in a different style shown in the following picture.

![微信截图_20201221155650](https://user-images.githubusercontent.com/24267981/102821833-c9d86900-43a5-11eb-9102-777c7ff3e478.png)


It will be really helpful if we can customize the custom sections a bit more. The following setting has a similar effect, but it renders the Parameters name instead of the custom name.
\`\`\`
napoleon_use_param = False
napoleon_custom_sections = [('Custom name', 'Parameters')]
\`\`\`
I would like to do something like the following so that my Custom section has the same style as the Parameter section, and it still keeps my custom name:

\`\`\`

napoleon_custom_sections = [("Side Effects", "display_like_parameters"), ...]

\`\`\`

or

\`\`\`
napoleon_custom_sections = [("Side Effects", "Parameters") ]
napoleon_custom_section_rename = False # True is default for backwards compatibility.
\`\`\`
The following link includes more details about the solutions:
[Format custom "Side Effects" docstring section in-toto/in-toto#401](https://github.com/in-toto/in-toto/issues/401)

Others people have expressed a similar desire (see sphinx-contrib/napoleon#2)

If you are interested, I would like to provide a PR for this. Thanks!




```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/ext/napoleon/docstring.py** | 543 | 556| 164 | 164 | 10667 | 
| **-> 2 <-** | **1 sphinx/ext/napoleon/docstring.py** | 631 | 649| 195 | 359 | 10667 | 
| **-> 3 <-** | **2 sphinx/ext/napoleon/__init__.py** | 20 | 265| 1937 | 2296 | 14625 | 
| 4 | **2 sphinx/ext/napoleon/docstring.py** | 685 | 699| 175 | 2471 | 14625 | 
| 5 | **2 sphinx/ext/napoleon/docstring.py** | 719 | 730| 147 | 2618 | 14625 | 
| 6 | **2 sphinx/ext/napoleon/docstring.py** | 651 | 662| 124 | 2742 | 14625 | 
| 7 | **2 sphinx/ext/napoleon/docstring.py** | 1159 | 1195| 340 | 3082 | 14625 | 
| 8 | **2 sphinx/ext/napoleon/docstring.py** | 315 | 355| 329 | 3411 | 14625 | 
| 9 | **2 sphinx/ext/napoleon/docstring.py** | 664 | 683| 192 | 3603 | 14625 | 
| 10 | **2 sphinx/ext/napoleon/docstring.py** | 1197 | 1301| 734 | 4337 | 14625 | 
| 11 | **2 sphinx/ext/napoleon/docstring.py** | 732 | 758| 226 | 4563 | 14625 | 
| 12 | **2 sphinx/ext/napoleon/docstring.py** | 760 | 790| 279 | 4842 | 14625 | 
| 13 | **2 sphinx/ext/napoleon/docstring.py** | 604 | 629| 255 | 5097 | 14625 | 
| 14 | **2 sphinx/ext/napoleon/docstring.py** | 396 | 411| 180 | 5277 | 14625 | 
| 15 | **2 sphinx/ext/napoleon/docstring.py** | 13 | 59| 515 | 5792 | 14625 | 
| 16 | **2 sphinx/ext/napoleon/__init__.py** | 266 | 289| 298 | 6090 | 14625 | 
| 17 | **2 sphinx/ext/napoleon/docstring.py** | 701 | 717| 198 | 6288 | 14625 | 
| 18 | **2 sphinx/ext/napoleon/docstring.py** | 296 | 313| 148 | 6436 | 14625 | 
| 19 | **2 sphinx/ext/napoleon/docstring.py** | 413 | 440| 258 | 6694 | 14625 | 
| 20 | **2 sphinx/ext/napoleon/docstring.py** | 461 | 502| 345 | 7039 | 14625 | 
| 21 | **2 sphinx/ext/napoleon/docstring.py** | 521 | 541| 195 | 7234 | 14625 | 
| 22 | **2 sphinx/ext/napoleon/docstring.py** | 442 | 459| 181 | 7415 | 14625 | 
| 23 | **2 sphinx/ext/napoleon/docstring.py** | 357 | 368| 119 | 7534 | 14625 | 
| 24 | **2 sphinx/ext/napoleon/docstring.py** | 370 | 394| 235 | 7769 | 14625 | 
| 25 | **2 sphinx/ext/napoleon/docstring.py** | 822 | 835| 153 | 7922 | 14625 | 
| 26 | **2 sphinx/ext/napoleon/docstring.py** | 1010 | 1107| 745 | 8667 | 14625 | 
| 27 | **2 sphinx/ext/napoleon/docstring.py** | 792 | 820| 213 | 8880 | 14625 | 
| 28 | **2 sphinx/ext/napoleon/docstring.py** | 1109 | 1129| 157 | 9037 | 14625 | 
| 29 | **2 sphinx/ext/napoleon/docstring.py** | 1131 | 1157| 259 | 9296 | 14625 | 
| 30 | **2 sphinx/ext/napoleon/docstring.py** | 558 | 590| 268 | 9564 | 14625 | 
| 31 | **2 sphinx/ext/napoleon/docstring.py** | 251 | 271| 217 | 9781 | 14625 | 
| 32 | 3 sphinx/pygments_styles.py | 38 | 96| 506 | 10287 | 15317 | 
| 33 | **3 sphinx/ext/napoleon/docstring.py** | 273 | 284| 139 | 10426 | 15317 | 
| 34 | **3 sphinx/ext/napoleon/docstring.py** | 504 | 519| 130 | 10556 | 15317 | 
| 35 | **3 sphinx/ext/napoleon/docstring.py** | 592 | 602| 124 | 10680 | 15317 | 
| 36 | **3 sphinx/ext/napoleon/docstring.py** | 286 | 294| 121 | 10801 | 15317 | 
| 37 | **3 sphinx/ext/napoleon/docstring.py** | 62 | 126| 599 | 11400 | 15317 | 
| 38 | **3 sphinx/ext/napoleon/docstring.py** | 128 | 213| 798 | 12198 | 15317 | 
| 39 | **3 sphinx/ext/napoleon/__init__.py** | 329 | 345| 158 | 12356 | 15317 | 
| 40 | **3 sphinx/ext/napoleon/__init__.py** | 348 | 395| 453 | 12809 | 15317 | 
| 41 | **3 sphinx/ext/napoleon/docstring.py** | 974 | 1007| 302 | 13111 | 15317 | 
| 42 | **3 sphinx/ext/napoleon/docstring.py** | 894 | 912| 115 | 13226 | 15317 | 
| **-> 43 <-** | **3 sphinx/ext/napoleon/__init__.py** | 11 | 289| 57 | 13283 | 15317 | 
| 44 | 4 sphinx/writers/html.py | 171 | 231| 562 | 13845 | 22685 | 
| 45 | 5 sphinx/writers/html5.py | 142 | 202| 567 | 14412 | 29580 | 
| 46 | 6 sphinx/config.py | 370 | 404| 294 | 14706 | 34181 | 
| 47 | 7 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 14815 | 36289 | 
| 48 | 8 sphinx/writers/manpage.py | 73 | 177| 837 | 15652 | 39798 | 
| 49 | 9 doc/usage/extensions/example_google.py | 261 | 275| 109 | 15761 | 41908 | 
| 50 | 9 sphinx/writers/manpage.py | 179 | 247| 538 | 16299 | 41908 | 
| 51 | **9 sphinx/ext/napoleon/docstring.py** | 215 | 249| 255 | 16554 | 41908 | 
| 52 | **9 sphinx/ext/napoleon/docstring.py** | 838 | 891| 262 | 16816 | 41908 | 
| 53 | 9 doc/usage/extensions/example_numpy.py | 1 | 47| 301 | 17117 | 41908 | 
| 54 | 9 sphinx/writers/manpage.py | 308 | 406| 757 | 17874 | 41908 | 
| 55 | **9 sphinx/ext/napoleon/__init__.py** | 398 | 481| 740 | 18614 | 41908 | 
| 56 | 10 sphinx/ext/autosectionlabel.py | 34 | 69| 339 | 18953 | 42427 | 
| 57 | 11 sphinx/writers/latex.py | 793 | 856| 573 | 19526 | 62267 | 
| 58 | 11 sphinx/writers/latex.py | 689 | 791| 825 | 20351 | 62267 | 
| 59 | 11 sphinx/writers/latex.py | 286 | 440| 1717 | 22068 | 62267 | 
| 60 | 12 sphinx/util/inspect.py | 829 | 902| 562 | 22630 | 69382 | 
| 61 | 13 sphinx/ext/autodoc/__init__.py | 1289 | 1402| 958 | 23588 | 91919 | 
| 62 | 13 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 23708 | 91919 | 
| 63 | 14 sphinx/transforms/compact_bullet_list.py | 55 | 95| 270 | 23978 | 92537 | 
| 64 | 15 sphinx/writers/texinfo.py | 865 | 969| 788 | 24766 | 104850 | 
| 65 | 15 sphinx/writers/html5.py | 53 | 140| 796 | 25562 | 104850 | 
| 66 | 15 sphinx/pygments_styles.py | 11 | 35| 135 | 25697 | 104850 | 
| 67 | 15 doc/usage/extensions/example_google.py | 298 | 314| 125 | 25822 | 104850 | 
| 68 | 15 sphinx/writers/html.py | 82 | 169| 789 | 26611 | 104850 | 
| 69 | 16 sphinx/transforms/i18n.py | 229 | 384| 1673 | 28284 | 109450 | 
| 70 | 17 sphinx/writers/text.py | 647 | 764| 839 | 29123 | 118423 | 
| 71 | 17 sphinx/writers/latex.py | 1327 | 1395| 667 | 29790 | 118423 | 
| 72 | 17 doc/usage/extensions/example_google.py | 277 | 296| 120 | 29910 | 118423 | 
| 73 | 17 sphinx/writers/html.py | 693 | 792| 803 | 30713 | 118423 | 
| 74 | 17 sphinx/ext/autodoc/__init__.py | 2022 | 2185| 1448 | 32161 | 118423 | 
| 75 | 17 sphinx/writers/latex.py | 576 | 639| 521 | 32682 | 118423 | 
| 76 | 18 sphinx/transforms/post_transforms/code.py | 86 | 110| 231 | 32913 | 119448 | 
| 77 | 19 sphinx/transforms/__init__.py | 274 | 315| 309 | 33222 | 122700 | 
| 78 | 19 sphinx/writers/texinfo.py | 1303 | 1386| 678 | 33900 | 122700 | 
| 79 | 19 sphinx/writers/texinfo.py | 270 | 298| 267 | 34167 | 122700 | 
| 80 | 19 sphinx/ext/autodoc/__init__.py | 1405 | 1731| 189 | 34356 | 122700 | 
| 81 | 19 sphinx/writers/texinfo.py | 11 | 114| 612 | 34968 | 122700 | 
| 82 | 19 doc/usage/extensions/example_numpy.py | 225 | 318| 589 | 35557 | 122700 | 
| 83 | 20 sphinx/ext/apidoc.py | 112 | 129| 203 | 35760 | 127421 | 
| 84 | 20 doc/usage/extensions/example_google.py | 1 | 37| 254 | 36014 | 127421 | 
| 85 | 20 sphinx/writers/texinfo.py | 971 | 1088| 846 | 36860 | 127421 | 
| 86 | 20 sphinx/writers/html5.py | 629 | 715| 686 | 37546 | 127421 | 
| 87 | 21 sphinx/util/rst.py | 11 | 76| 541 | 38087 | 128274 | 
| 88 | 22 sphinx/util/console.py | 101 | 142| 296 | 38383 | 129267 | 
| 89 | 22 sphinx/writers/text.py | 929 | 1042| 873 | 39256 | 129267 | 
| 90 | 22 sphinx/writers/text.py | 450 | 510| 455 | 39711 | 129267 | 
| 91 | 22 sphinx/writers/texinfo.py | 1090 | 1189| 839 | 40550 | 129267 | 
| 92 | 23 sphinx/builders/manpage.py | 109 | 128| 166 | 40716 | 130231 | 
| 93 | 24 sphinx/ext/autodoc/typehints.py | 82 | 138| 460 | 41176 | 131284 | 
| 94 | 25 sphinx/builders/latex/transforms.py | 182 | 358| 800 | 41976 | 135564 | 
| 95 | 26 sphinx/util/docfields.py | 212 | 228| 161 | 42137 | 138934 | 
| 96 | 26 sphinx/builders/latex/transforms.py | 11 | 49| 327 | 42464 | 138934 | 
| 97 | 26 sphinx/util/console.py | 85 | 98| 155 | 42619 | 138934 | 
| 98 | 26 sphinx/ext/autodoc/__init__.py | 1436 | 1731| 2767 | 45386 | 138934 | 
| 99 | 27 sphinx/directives/patches.py | 27 | 50| 187 | 45573 | 140544 | 
| 100 | 27 sphinx/writers/latex.py | 1827 | 1899| 564 | 46137 | 140544 | 
| 101 | 27 sphinx/ext/autodoc/__init__.py | 484 | 517| 286 | 46423 | 140544 | 
| 102 | 27 sphinx/writers/texinfo.py | 1407 | 1464| 402 | 46825 | 140544 | 
| 103 | 27 sphinx/ext/autodoc/__init__.py | 13 | 115| 790 | 47615 | 140544 | 
| 104 | 27 sphinx/writers/latex.py | 1609 | 1668| 496 | 48111 | 140544 | 
| 105 | 28 sphinx/addnodes.py | 150 | 261| 710 | 48821 | 143290 | 
| 106 | 28 sphinx/writers/html5.py | 306 | 360| 507 | 49328 | 143290 | 
| 107 | 28 sphinx/transforms/__init__.py | 318 | 358| 286 | 49614 | 143290 | 
| 108 | 28 sphinx/writers/latex.py | 442 | 502| 569 | 50183 | 143290 | 
| 109 | 28 sphinx/config.py | 345 | 367| 257 | 50440 | 143290 | 
| 110 | 29 sphinx/util/typing.py | 185 | 285| 1250 | 51690 | 148005 | 
| 111 | 30 sphinx/builders/latex/theming.py | 11 | 48| 256 | 51946 | 149032 | 
| 112 | 30 sphinx/addnodes.py | 281 | 380| 580 | 52526 | 149032 | 
| 113 | 30 sphinx/writers/texinfo.py | 755 | 863| 813 | 53339 | 149032 | 
| 114 | 31 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 53514 | 149940 | 
| 115 | 31 sphinx/ext/autodoc/__init__.py | 1257 | 1268| 134 | 53648 | 149940 | 
| 116 | 32 sphinx/directives/__init__.py | 269 | 323| 517 | 54165 | 152690 | 
| 117 | 32 sphinx/transforms/__init__.py | 182 | 212| 229 | 54394 | 152690 | 
| 118 | 32 sphinx/writers/texinfo.py | 1477 | 1552| 549 | 54943 | 152690 | 
| 119 | 32 doc/usage/extensions/example_google.py | 180 | 259| 563 | 55506 | 152690 | 
| 120 | 33 doc/conf.py | 59 | 127| 688 | 56194 | 154248 | 
| 121 | 34 sphinx/domains/c.py | 11 | 816| 6564 | 62758 | 185564 | 
| 122 | 34 sphinx/transforms/__init__.py | 142 | 163| 214 | 62972 | 185564 | 
| 123 | 35 sphinx/highlighting.py | 55 | 164| 876 | 63848 | 186866 | 
| 124 | 35 sphinx/domains/c.py | 2578 | 3409| 6792 | 70640 | 186866 | 
| 125 | 36 sphinx/builders/latex/constants.py | 120 | 202| 925 | 71565 | 188946 | 
| 126 | **36 sphinx/ext/napoleon/docstring.py** | 915 | 971| 384 | 71949 | 188946 | 
| 127 | 37 sphinx/util/nodes.py | 328 | 344| 162 | 72111 | 194352 | 
| 128 | 37 sphinx/builders/latex/transforms.py | 361 | 439| 666 | 72777 | 194352 | 
| 129 | 37 sphinx/builders/latex/transforms.py | 102 | 127| 277 | 73054 | 194352 | 
| 130 | 37 sphinx/writers/latex.py | 1453 | 1517| 863 | 73917 | 194352 | 
| 131 | 37 sphinx/writers/text.py | 1044 | 1156| 813 | 74730 | 194352 | 
| 132 | 37 sphinx/writers/text.py | 766 | 873| 830 | 75560 | 194352 | 
| 133 | 37 sphinx/writers/text.py | 528 | 627| 640 | 76200 | 194352 | 
| 134 | 38 sphinx/directives/other.py | 152 | 183| 213 | 76413 | 197523 | 
| 135 | 38 sphinx/writers/texinfo.py | 1217 | 1282| 457 | 76870 | 197523 | 


### Hint

```
Boy, I could really use this right away.  If it were up to me:

* `napoleon_custom_sections` would be called `napoleon_custom_aliases`, and only accept a list of `(new alias, existing section)` tuples.
* A hypothetical new `napoleon_custom_sections` would only accept a list of `(new section, existing section)` or `(new section, callback function)` tuples, and the output would always use `new section` as the title, in either case.

That would be a backwards-incompatible change, but you could argue that the current behavior isn’t actually documented in the Sphinx docs, and thus is fair game.

Cheers,
Tim
I also don't know napoleon module has such option. It was added at #4387. It has not been documented, but it was introduced in CHANGES. So I consider it's a secret feature. So -1 for incompatible change.

But I'm interested in the enhancement itself. Could you submit a PR please? I'll take a look.

```

## Patch

```diff
diff --git a/sphinx/ext/napoleon/__init__.py b/sphinx/ext/napoleon/__init__.py
--- a/sphinx/ext/napoleon/__init__.py
+++ b/sphinx/ext/napoleon/__init__.py
@@ -253,10 +253,15 @@ def __unicode__(self):
           * To create a custom "generic" section, just pass a string.
           * To create an alias for an existing section, pass a tuple containing the
             alias name and the original, in that order.
+          * To create a custom section that displays like the parameters or returns
+            section, pass a tuple containing the custom section name and a string
+            value, "params_style" or "returns_style".
 
         If an entry is just a string, it is interpreted as a header for a generic
         section. If the entry is a tuple/list/indexed container, the first entry
-        is the name of the section, the second is the section key to emulate.
+        is the name of the section, the second is the section key to emulate. If the
+        second entry value is "params_style" or "returns_style", the custom section
+        will be displayed like the parameters section or returns section.
 
     napoleon_attr_annotations : :obj:`bool` (Defaults to True)
         Use the type annotations of class attributes that are documented in the docstring
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -549,11 +549,18 @@ def _load_custom_sections(self) -> None:
                     self._sections[entry.lower()] = self._parse_custom_generic_section
                 else:
                     # otherwise, assume entry is container;
-                    # [0] is new section, [1] is the section to alias.
-                    # in the case of key mismatch, just handle as generic section.
-                    self._sections[entry[0].lower()] = \
-                        self._sections.get(entry[1].lower(),
-                                           self._parse_custom_generic_section)
+                    if entry[1] == "params_style":
+                        self._sections[entry[0].lower()] = \
+                            self._parse_custom_params_style_section
+                    elif entry[1] == "returns_style":
+                        self._sections[entry[0].lower()] = \
+                            self._parse_custom_returns_style_section
+                    else:
+                        # [0] is new section, [1] is the section to alias.
+                        # in the case of key mismatch, just handle as generic section.
+                        self._sections[entry[0].lower()] = \
+                            self._sections.get(entry[1].lower(),
+                                               self._parse_custom_generic_section)
 
     def _parse(self) -> None:
         self._parsed_lines = self._consume_empty()
@@ -641,6 +648,13 @@ def _parse_custom_generic_section(self, section: str) -> List[str]:
         # for now, no admonition for simple custom sections
         return self._parse_generic_section(section, False)
 
+    def _parse_custom_params_style_section(self, section: str) -> List[str]:
+        return self._format_fields(section, self._consume_fields())
+
+    def _parse_custom_returns_style_section(self, section: str) -> List[str]:
+        fields = self._consume_returns_section()
+        return self._format_fields(section, fields)
+
     def _parse_usage_section(self, section: str) -> List[str]:
         header = ['.. rubric:: Usage:', '']
         block = ['.. code-block:: python', '']

```

## Test Patch

```diff
diff --git a/tests/test_ext_napoleon_docstring.py b/tests/test_ext_napoleon_docstring.py
--- a/tests/test_ext_napoleon_docstring.py
+++ b/tests/test_ext_napoleon_docstring.py
@@ -1072,10 +1072,27 @@ def test_custom_generic_sections(self):
 Sooper Warning:
     Stop hitting yourself!
 """, """:Warns: **Stop hitting yourself!**
+"""),
+                      ("""\
+Params Style:
+    arg1 (int): Description of arg1
+    arg2 (str): Description of arg2
+
+""", """\
+:Params Style: * **arg1** (*int*) -- Description of arg1
+               * **arg2** (*str*) -- Description of arg2
+"""),
+                      ("""\
+Returns Style:
+    description of custom section
+
+""", """:Returns Style: description of custom section
 """))
 
         testConfig = Config(napoleon_custom_sections=['Really Important Details',
-                                                      ('Sooper Warning', 'warns')])
+                                                      ('Sooper Warning', 'warns'),
+                                                      ('Params Style', 'params_style'),
+                                                      ('Returns Style', 'returns_style')])
 
         for docstring, expected in docstrings:
             actual = str(GoogleDocstring(docstring, testConfig))

```


## Code snippets

### 1 - sphinx/ext/napoleon/docstring.py:

Start line: 543, End line: 556

```python
class GoogleDocstring:

    def _load_custom_sections(self) -> None:
        if self._config.napoleon_custom_sections is not None:
            for entry in self._config.napoleon_custom_sections:
                if isinstance(entry, str):
                    # if entry is just a label, add to sections list,
                    # using generic section logic.
                    self._sections[entry.lower()] = self._parse_custom_generic_section
                else:
                    # otherwise, assume entry is container;
                    # [0] is new section, [1] is the section to alias.
                    # in the case of key mismatch, just handle as generic section.
                    self._sections[entry[0].lower()] = \
                        self._sections.get(entry[1].lower(),
                                           self._parse_custom_generic_section)
```
### 2 - sphinx/ext/napoleon/docstring.py:

Start line: 631, End line: 649

```python
class GoogleDocstring:

    def _parse_examples_section(self, section: str) -> List[str]:
        labels = {
            'example': _('Example'),
            'examples': _('Examples'),
        }
        use_admonition = self._config.napoleon_use_admonition_for_examples
        label = labels.get(section.lower(), section)
        return self._parse_generic_section(label, use_admonition)

    def _parse_custom_generic_section(self, section: str) -> List[str]:
        # for now, no admonition for simple custom sections
        return self._parse_generic_section(section, False)

    def _parse_usage_section(self, section: str) -> List[str]:
        header = ['.. rubric:: Usage:', '']
        block = ['.. code-block:: python', '']
        lines = self._consume_usage_section()
        lines = self._indent(lines, 3)
        return header + block + lines + ['']
```
### 3 - sphinx/ext/napoleon/__init__.py:

Start line: 20, End line: 265

```python
class Config:
    """Sphinx napoleon extension settings in `conf.py`.

    Listed below are all the settings used by napoleon and their default
    values. These settings can be changed in the Sphinx `conf.py` file. Make
    sure that "sphinx.ext.napoleon" is enabled in `conf.py`::

        # conf.py

        # Add any Sphinx extension module names here, as strings
        extensions = ['sphinx.ext.napoleon']

        # Napoleon settings
        napoleon_google_docstring = True
        napoleon_numpy_docstring = True
        napoleon_include_init_with_doc = False
        napoleon_include_private_with_doc = False
        napoleon_include_special_with_doc = False
        napoleon_use_admonition_for_examples = False
        napoleon_use_admonition_for_notes = False
        napoleon_use_admonition_for_references = False
        napoleon_use_ivar = False
        napoleon_use_param = True
        napoleon_use_rtype = True
        napoleon_use_keyword = True
        napoleon_preprocess_types = False
        napoleon_type_aliases = None
        napoleon_custom_sections = None
        napoleon_attr_annotations = True

    .. _Google style:
       https://google.github.io/styleguide/pyguide.html
    .. _NumPy style:
       https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

    Attributes
    ----------
    napoleon_google_docstring : :obj:`bool` (Defaults to True)
        True to parse `Google style`_ docstrings. False to disable support
        for Google style docstrings.
    napoleon_numpy_docstring : :obj:`bool` (Defaults to True)
        True to parse `NumPy style`_ docstrings. False to disable support
        for NumPy style docstrings.
    napoleon_include_init_with_doc : :obj:`bool` (Defaults to False)
        True to list ``__init___`` docstrings separately from the class
        docstring. False to fall back to Sphinx's default behavior, which
        considers the ``__init___`` docstring as part of the class
        documentation.

        **If True**::

            def __init__(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"

            def __init__(self):
                # This will NOT be included in the docs

    napoleon_include_private_with_doc : :obj:`bool` (Defaults to False)
        True to include private members (like ``_membername``) with docstrings
        in the documentation. False to fall back to Sphinx's default behavior.

        **If True**::

            def _included(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"
                pass

            def _skipped(self):
                # This will NOT be included in the docs
                pass

    napoleon_include_special_with_doc : :obj:`bool` (Defaults to False)
        True to include special members (like ``__membername__``) with
        docstrings in the documentation. False to fall back to Sphinx's
        default behavior.

        **If True**::

            def __str__(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"
                return unicode(self).encode('utf-8')

            def __unicode__(self):
                # This will NOT be included in the docs
                return unicode(self.__class__.__name__)

    napoleon_use_admonition_for_examples : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for the **Example** and
        **Examples** sections. False to use the ``.. rubric::`` directive
        instead. One may look better than the other depending on what HTML
        theme is used.

        This `NumPy style`_ snippet will be converted as follows::

            Example
            -------
            This is just a quick example

        **If True**::

            .. admonition:: Example

               This is just a quick example

        **If False**::

            .. rubric:: Example

            This is just a quick example

    napoleon_use_admonition_for_notes : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for **Notes** sections.
        False to use the ``.. rubric::`` directive instead.

        Note
        ----
        The singular **Note** section will always be converted to a
        ``.. note::`` directive.

        See Also
        --------
        :attr:`napoleon_use_admonition_for_examples`

    napoleon_use_admonition_for_references : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for **References**
        sections. False to use the ``.. rubric::`` directive instead.

        See Also
        --------
        :attr:`napoleon_use_admonition_for_examples`

    napoleon_use_ivar : :obj:`bool` (Defaults to False)
        True to use the ``:ivar:`` role for instance variables. False to use
        the ``.. attribute::`` directive instead.

        This `NumPy style`_ snippet will be converted as follows::

            Attributes
            ----------
            attr1 : int
                Description of `attr1`

        **If True**::

            :ivar attr1: Description of `attr1`
            :vartype attr1: int

        **If False**::

            .. attribute:: attr1

               Description of `attr1`

               :type: int

    napoleon_use_param : :obj:`bool` (Defaults to True)
        True to use a ``:param:`` role for each function parameter. False to
        use a single ``:parameters:`` role for all the parameters.

        This `NumPy style`_ snippet will be converted as follows::

            Parameters
            ----------
            arg1 : str
                Description of `arg1`
            arg2 : int, optional
                Description of `arg2`, defaults to 0

        **If True**::

            :param arg1: Description of `arg1`
            :type arg1: str
            :param arg2: Description of `arg2`, defaults to 0
            :type arg2: int, optional

        **If False**::

            :parameters: * **arg1** (*str*) --
                           Description of `arg1`
                         * **arg2** (*int, optional*) --
                           Description of `arg2`, defaults to 0

    napoleon_use_keyword : :obj:`bool` (Defaults to True)
        True to use a ``:keyword:`` role for each function keyword argument.
        False to use a single ``:keyword arguments:`` role for all the
        keywords.

        This behaves similarly to  :attr:`napoleon_use_param`. Note unlike
        docutils, ``:keyword:`` and ``:param:`` will not be treated the same
        way - there will be a separate "Keyword Arguments" section, rendered
        in the same fashion as "Parameters" section (type links created if
        possible)

        See Also
        --------
        :attr:`napoleon_use_param`

    napoleon_use_rtype : :obj:`bool` (Defaults to True)
        True to use the ``:rtype:`` role for the return type. False to output
        the return type inline with the description.

        This `NumPy style`_ snippet will be converted as follows::

            Returns
            -------
            bool
                True if successful, False otherwise

        **If True**::

            :returns: True if successful, False otherwise
            :rtype: bool

        **If False**::

            :returns: *bool* -- True if successful, False otherwise

    napoleon_preprocess_types : :obj:`bool` (Defaults to False)
        Enable the type preprocessor for numpy style docstrings.

    napoleon_type_aliases : :obj:`dict` (Defaults to None)
        Add a mapping of strings to string, translating types in numpy
        style docstrings. Only works if ``napoleon_preprocess_types = True``.

    napoleon_custom_sections : :obj:`list` (Defaults to None)
        Add a list of custom sections to include, expanding the list of parsed sections.

        The entries can either be strings or tuples, depending on the intention:
          * To create a custom "generic" section, just pass a string.
          * To create an alias for an existing section, pass a tuple containing the
            alias name and the original, in that order.

        If an entry is just a string, it is interpreted as a header for a generic
        section. If the entry is a tuple/list/indexed container, the first entry
        is the name of the section, the second is the section key to emulate.

    napoleon_attr_annotations : :obj:`bool` (Defaults to True)
        Use the type annotations of class attributes that are documented in the docstring
        but do not have a type in the docstring.

    """
```
### 4 - sphinx/ext/napoleon/docstring.py:

Start line: 685, End line: 699

```python
class GoogleDocstring:

    def _parse_notes_section(self, section: str) -> List[str]:
        use_admonition = self._config.napoleon_use_admonition_for_notes
        return self._parse_generic_section(_('Notes'), use_admonition)

    def _parse_other_parameters_section(self, section: str) -> List[str]:
        return self._format_fields(_('Other Parameters'), self._consume_fields())

    def _parse_parameters_section(self, section: str) -> List[str]:
        if self._config.napoleon_use_param:
            # Allow to declare multiple parameters at once (ex: x, y: int)
            fields = self._consume_fields(multiple=True)
            return self._format_docutils_params(fields)
        else:
            fields = self._consume_fields()
            return self._format_fields(_('Parameters'), fields)
```
### 5 - sphinx/ext/napoleon/docstring.py:

Start line: 719, End line: 730

```python
class GoogleDocstring:

    def _parse_receives_section(self, section: str) -> List[str]:
        if self._config.napoleon_use_param:
            # Allow to declare multiple parameters at once (ex: x, y: int)
            fields = self._consume_fields(multiple=True)
            return self._format_docutils_params(fields)
        else:
            fields = self._consume_fields()
            return self._format_fields(_('Receives'), fields)

    def _parse_references_section(self, section: str) -> List[str]:
        use_admonition = self._config.napoleon_use_admonition_for_references
        return self._parse_generic_section(_('References'), use_admonition)
```
### 6 - sphinx/ext/napoleon/docstring.py:

Start line: 651, End line: 662

```python
class GoogleDocstring:

    def _parse_generic_section(self, section: str, use_admonition: bool) -> List[str]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        if use_admonition:
            header = '.. admonition:: %s' % section
            lines = self._indent(lines, 3)
        else:
            header = '.. rubric:: %s' % section
        if lines:
            return [header, ''] + lines + ['']
        else:
            return [header, '']
```
### 7 - sphinx/ext/napoleon/docstring.py:

Start line: 1159, End line: 1195

```python
class NumpyDocstring(GoogleDocstring):

    def _consume_returns_section(self) -> List[Tuple[str, str, List[str]]]:
        return self._consume_fields(prefer_type=True)

    def _consume_section_header(self) -> str:
        section = next(self._line_iter)
        if not _directive_regex.match(section):
            # Consume the header underline
            next(self._line_iter)
        return section

    def _is_section_break(self) -> bool:
        line1, line2 = self._line_iter.peek(2)
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                ['', ''] == [line1, line2] or
                (self._is_in_section and
                    line1 and
                    not self._is_indented(line1, self._section_indent)))

    def _is_section_header(self) -> bool:
        section, underline = self._line_iter.peek(2)
        section = section.lower()
        if section in self._sections and isinstance(underline, str):
            return bool(_numpy_section_regex.match(underline))
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _parse_see_also_section(self, section: str) -> List[str]:
        lines = self._consume_to_next_section()
        try:
            return self._parse_numpydoc_see_also_section(lines)
        except ValueError:
            return self._format_admonition('seealso', lines)
```
### 8 - sphinx/ext/napoleon/docstring.py:

Start line: 315, End line: 355

```python
class GoogleDocstring:

    def _consume_usage_section(self) -> List[str]:
        lines = self._dedent(self._consume_to_next_section())
        return lines

    def _consume_section_header(self) -> str:
        section = next(self._line_iter)
        stripped_section = section.strip(':')
        if stripped_section.lower() in self._sections:
            section = stripped_section
        return section

    def _consume_to_end(self) -> List[str]:
        lines = []
        while self._line_iter.has_next():
            lines.append(next(self._line_iter))
        return lines

    def _consume_to_next_section(self) -> List[str]:
        self._consume_empty()
        lines = []
        while not self._is_section_break():
            lines.append(next(self._line_iter))
        return lines + self._consume_empty()

    def _dedent(self, lines: List[str], full: bool = False) -> List[str]:
        if full:
            return [line.lstrip() for line in lines]
        else:
            min_indent = self._get_min_indent(lines)
            return [line[min_indent:] for line in lines]

    def _escape_args_and_kwargs(self, name: str) -> str:
        if name.endswith('_') and getattr(self._config, 'strip_signature_backslash', False):
            name = name[:-1] + r'\_'

        if name[:2] == '**':
            return r'\*\*' + name[2:]
        elif name[:1] == '*':
            return r'\*' + name[1:]
        else:
            return name
```
### 9 - sphinx/ext/napoleon/docstring.py:

Start line: 664, End line: 683

```python
class GoogleDocstring:

    def _parse_keyword_arguments_section(self, section: str) -> List[str]:
        fields = self._consume_fields()
        if self._config.napoleon_use_keyword:
            return self._format_docutils_params(
                fields,
                field_role="keyword",
                type_role="kwtype")
        else:
            return self._format_fields(_('Keyword Arguments'), fields)

    def _parse_methods_section(self, section: str) -> List[str]:
        lines = []  # type: List[str]
        for _name, _type, _desc in self._consume_fields(parse_type=False):
            lines.append('.. method:: %s' % _name)
            if self._opt and 'noindex' in self._opt:
                lines.append('   :noindex:')
            if _desc:
                lines.extend([''] + self._indent(_desc, 3))
            lines.append('')
        return lines
```
### 10 - sphinx/ext/napoleon/docstring.py:

Start line: 1197, End line: 1301

```python
class NumpyDocstring(GoogleDocstring):

    def _parse_numpydoc_see_also_section(self, content: List[str]) -> List[str]:
        """
        Derived from the NumpyDoc implementation of _parse_see_also.

        See Also
        --------
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
        items = []

        def parse_item_name(text: str) -> Tuple[str, str]:
            """Match ':role:`name`' or 'name'"""
            m = self._name_rgx.match(text)
            if m:
                g = m.groups()
                if g[1] is None:
                    return g[3], None
                else:
                    return g[2], g[1]
            raise ValueError("%s is not a item name" % text)

        def push_item(name: str, rest: List[str]) -> None:
            if not name:
                return
            name, role = parse_item_name(name)
            items.append((name, list(rest), role))
            del rest[:]

        def translate(func, description, role):
            translations = self._config.napoleon_type_aliases
            if role is not None or not translations:
                return func, description, role

            translated = translations.get(func, func)
            match = self._name_rgx.match(translated)
            if not match:
                return translated, description, role

            groups = match.groupdict()
            role = groups["role"]
            new_func = groups["name"] or groups["name2"]

            return new_func, description, role

        current_func = None
        rest = []  # type: List[str]

        for line in content:
            if not line.strip():
                continue

            m = self._name_rgx.match(line)
            if m and line[m.end():].strip().startswith(':'):
                push_item(current_func, rest)
                current_func, line = line[:m.end()], line[m.end():]
                rest = [line.split(':', 1)[1].strip()]
                if not rest[0]:
                    rest = []
            elif not line.startswith(' '):
                push_item(current_func, rest)
                current_func = None
                if ',' in line:
                    for func in line.split(','):
                        if func.strip():
                            push_item(func, [])
                elif line.strip():
                    current_func = line
            elif current_func is not None:
                rest.append(line.strip())
        push_item(current_func, rest)

        if not items:
            return []

        # apply type aliases
        items = [
            translate(func, description, role)
            for func, description, role in items
        ]

        lines = []  # type: List[str]
        last_had_desc = True
        for name, desc, role in items:
            if role:
                link = ':%s:`%s`' % (role, name)
            else:
                link = ':obj:`%s`' % name
            if desc or last_had_desc:
                lines += ['']
                lines += [link]
            else:
                lines[-1] += ", %s" % link
            if desc:
                lines += self._indent([' '.join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
        lines += ['']

        return self._format_admonition('seealso', lines)
```
### 11 - sphinx/ext/napoleon/docstring.py:

Start line: 732, End line: 758

```python
class GoogleDocstring:

    def _parse_returns_section(self, section: str) -> List[str]:
        fields = self._consume_returns_section()
        multi = len(fields) > 1
        if multi:
            use_rtype = False
        else:
            use_rtype = self._config.napoleon_use_rtype

        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            if use_rtype:
                field = self._format_field(_name, '', _desc)
            else:
                field = self._format_field(_name, _type, _desc)

            if multi:
                if lines:
                    lines.extend(self._format_block('          * ', field))
                else:
                    lines.extend(self._format_block(':returns: * ', field))
            else:
                lines.extend(self._format_block(':returns: ', field))
                if _type and use_rtype:
                    lines.extend([':rtype: %s' % _type, ''])
        if lines and lines[-1]:
            lines.append('')
        return lines
```
### 12 - sphinx/ext/napoleon/docstring.py:

Start line: 760, End line: 790

```python
class GoogleDocstring:

    def _parse_see_also_section(self, section: str) -> List[str]:
        return self._parse_admonition('seealso', section)

    def _parse_warns_section(self, section: str) -> List[str]:
        return self._format_fields(_('Warns'), self._consume_fields())

    def _parse_yields_section(self, section: str) -> List[str]:
        fields = self._consume_returns_section()
        return self._format_fields(_('Yields'), fields)

    def _partition_field_on_colon(self, line: str) -> Tuple[str, str, str]:
        before_colon = []
        after_colon = []
        colon = ''
        found_colon = False
        for i, source in enumerate(_xref_or_code_regex.split(line)):
            if found_colon:
                after_colon.append(source)
            else:
                m = _single_colon_regex.search(source)
                if (i % 2) == 0 and m:
                    found_colon = True
                    colon = source[m.start(): m.end()]
                    before_colon.append(source[:m.start()])
                    after_colon.append(source[m.end():])
                else:
                    before_colon.append(source)

        return ("".join(before_colon).strip(),
                colon,
                "".join(after_colon).strip())
```
### 13 - sphinx/ext/napoleon/docstring.py:

Start line: 604, End line: 629

```python
class GoogleDocstring:

    def _parse_attributes_section(self, section: str) -> List[str]:
        lines = []
        for _name, _type, _desc in self._consume_fields():
            if not _type:
                _type = self._lookup_annotation(_name)
            if self._config.napoleon_use_ivar:
                _name = self._qualify_name(_name, self._obj)
                field = ':ivar %s: ' % _name
                lines.extend(self._format_block(field, _desc))
                if _type:
                    lines.append(':vartype %s: %s' % (_name, _type))
            else:
                lines.append('.. attribute:: ' + _name)
                if self._opt and 'noindex' in self._opt:
                    lines.append('   :noindex:')
                lines.append('')

                fields = self._format_field('', '', _desc)
                lines.extend(self._indent(fields, 3))
                if _type:
                    lines.append('')
                    lines.extend(self._indent([':type: %s' % _type], 3))
                lines.append('')
        if self._config.napoleon_use_ivar:
            lines.append('')
        return lines
```
### 14 - sphinx/ext/napoleon/docstring.py:

Start line: 396, End line: 411

```python
class GoogleDocstring:

    def _format_docutils_params(self, fields: List[Tuple[str, str, List[str]]],
                                field_role: str = 'param', type_role: str = 'type'
                                ) -> List[str]:
        lines = []
        for _name, _type, _desc in fields:
            _desc = self._strip_empty(_desc)
            if any(_desc):
                _desc = self._fix_field_desc(_desc)
                field = ':%s %s: ' % (field_role, _name)
                lines.extend(self._format_block(field, _desc))
            else:
                lines.append(':%s %s:' % (field_role, _name))

            if _type:
                lines.append(':%s %s: %s' % (type_role, _name, _type))
        return lines + ['']
```
### 15 - sphinx/ext/napoleon/docstring.py:

Start line: 13, End line: 59

```python
import collections
import inspect
import re
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.iterators import modify_iter
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)

_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'(.+?)\(\s*(.*[^\s]+)\s*\)')
_numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_or_code_regex = re.compile(
    r'((?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)|'
    r'(?:``.+?``))')
_xref_regex = re.compile(
    r'(?:(?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:)?`.+?`)'
)
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')
_token_regex = re.compile(
    r"(,\sor\s|\sor\s|\sof\s|:\s|\sto\s|,\sand\s|\sand\s|,\s"
    r"|[{]|[}]"
    r'|"(?:\\"|[^"])*"'
    r"|'(?:\\'|[^'])*')"
)
_default_regex = re.compile(
    r"^default[^_0-9A-Za-z].*$",
)
_SINGLETONS = ("None", "True", "False", "Ellipsis")
```
### 16 - sphinx/ext/napoleon/__init__.py:

Start line: 266, End line: 289

```python
class Config:
    _config_values = {
        'napoleon_google_docstring': (True, 'env'),
        'napoleon_numpy_docstring': (True, 'env'),
        'napoleon_include_init_with_doc': (False, 'env'),
        'napoleon_include_private_with_doc': (False, 'env'),
        'napoleon_include_special_with_doc': (False, 'env'),
        'napoleon_use_admonition_for_examples': (False, 'env'),
        'napoleon_use_admonition_for_notes': (False, 'env'),
        'napoleon_use_admonition_for_references': (False, 'env'),
        'napoleon_use_ivar': (False, 'env'),
        'napoleon_use_param': (True, 'env'),
        'napoleon_use_rtype': (True, 'env'),
        'napoleon_use_keyword': (True, 'env'),
        'napoleon_preprocess_types': (False, 'env'),
        'napoleon_type_aliases': (None, 'env'),
        'napoleon_custom_sections': (None, 'env'),
        'napoleon_attr_annotations': (True, 'env'),
    }

    def __init__(self, **settings: Any) -> None:
        for name, (default, rebuild) in self._config_values.items():
            setattr(self, name, default)
        for name, value in settings.items():
            setattr(self, name, value)
```
### 17 - sphinx/ext/napoleon/docstring.py:

Start line: 701, End line: 717

```python
class GoogleDocstring:

    def _parse_raises_section(self, section: str) -> List[str]:
        fields = self._consume_fields(parse_type=False, prefer_type=True)
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            m = self._name_rgx.match(_type)
            if m and m.group('name'):
                _type = m.group('name')
            elif _xref_regex.match(_type):
                pos = _type.find('`')
                _type = _type[pos + 1:-1]
            _type = ' ' + _type if _type else ''
            _desc = self._strip_empty(_desc)
            _descs = ' ' + '\n    '.join(_desc) if any(_desc) else ''
            lines.append(':raises%s:%s' % (_type, _descs))
        if lines:
            lines.append('')
        return lines
```
### 18 - sphinx/ext/napoleon/docstring.py:

Start line: 296, End line: 313

```python
class GoogleDocstring:

    def _consume_returns_section(self) -> List[Tuple[str, str, List[str]]]:
        lines = self._dedent(self._consume_to_next_section())
        if lines:
            before, colon, after = self._partition_field_on_colon(lines[0])
            _name, _type, _desc = '', '', lines

            if colon:
                if after:
                    _desc = [after] + lines[1:]
                else:
                    _desc = lines[1:]

                _type = before

            _desc = self.__class__(_desc, self._config).lines()
            return [(_name, _type, _desc,)]
        else:
            return []
```
### 19 - sphinx/ext/napoleon/docstring.py:

Start line: 413, End line: 440

```python
class GoogleDocstring:

    def _format_field(self, _name: str, _type: str, _desc: List[str]) -> List[str]:
        _desc = self._strip_empty(_desc)
        has_desc = any(_desc)
        separator = ' -- ' if has_desc else ''
        if _name:
            if _type:
                if '`' in _type:
                    field = '**%s** (%s)%s' % (_name, _type, separator)
                else:
                    field = '**%s** (*%s*)%s' % (_name, _type, separator)
            else:
                field = '**%s**%s' % (_name, separator)
        elif _type:
            if '`' in _type:
                field = '%s%s' % (_type, separator)
            else:
                field = '*%s*%s' % (_type, separator)
        else:
            field = ''

        if has_desc:
            _desc = self._fix_field_desc(_desc)
            if _desc[0]:
                return [field + _desc[0]] + _desc[1:]
            else:
                return [field] + _desc
        else:
            return [field]
```
### 20 - sphinx/ext/napoleon/docstring.py:

Start line: 461, End line: 502

```python
class GoogleDocstring:

    def _get_current_indent(self, peek_ahead: int = 0) -> int:
        line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        while line != self._line_iter.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        return 0

    def _get_indent(self, line: str) -> int:
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_initial_indent(self, lines: List[str]) -> int:
        for line in lines:
            if line:
                return self._get_indent(line)
        return 0

    def _get_min_indent(self, lines: List[str]) -> int:
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent
        return min_indent or 0

    def _indent(self, lines: List[str], n: int = 4) -> List[str]:
        return [(' ' * n) + line for line in lines]

    def _is_indented(self, line: str, indent: int = 1) -> bool:
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False
```
### 21 - sphinx/ext/napoleon/docstring.py:

Start line: 521, End line: 541

```python
class GoogleDocstring:

    def _is_section_header(self) -> bool:
        section = self._line_iter.peek().lower()
        match = _google_section_regex.match(section)
        if match and section.strip(':') in self._sections:
            header_indent = self._get_indent(section)
            section_indent = self._get_current_indent(peek_ahead=1)
            return section_indent > header_indent
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _is_section_break(self) -> bool:
        line = self._line_iter.peek()
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                (self._is_in_section and
                    line and
                    not self._is_indented(line, self._section_indent)))
```
### 22 - sphinx/ext/napoleon/docstring.py:

Start line: 442, End line: 459

```python
class GoogleDocstring:

    def _format_fields(self, field_type: str, fields: List[Tuple[str, str, List[str]]]
                       ) -> List[str]:
        field_type = ':%s:' % field_type.strip()
        padding = ' ' * len(field_type)
        multi = len(fields) > 1
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block(padding + ' * ', field))
                else:
                    lines.extend(self._format_block(field_type + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' ', field))
        if lines and lines[-1]:
            lines.append('')
        return lines
```
### 23 - sphinx/ext/napoleon/docstring.py:

Start line: 357, End line: 368

```python
class GoogleDocstring:

    def _fix_field_desc(self, desc: List[str]) -> List[str]:
        if self._is_list(desc):
            desc = [''] + desc
        elif desc[0].endswith('::'):
            desc_block = desc[1:]
            indent = self._get_indent(desc[0])
            block_indent = self._get_initial_indent(desc_block)
            if block_indent > indent:
                desc = [''] + desc
            else:
                desc = ['', desc[0]] + self._indent(desc_block, 4)
        return desc
```
### 24 - sphinx/ext/napoleon/docstring.py:

Start line: 370, End line: 394

```python
class GoogleDocstring:

    def _format_admonition(self, admonition: str, lines: List[str]) -> List[str]:
        lines = self._strip_empty(lines)
        if len(lines) == 1:
            return ['.. %s:: %s' % (admonition, lines[0].strip()), '']
        elif lines:
            lines = self._indent(self._dedent(lines), 3)
            return ['.. %s::' % admonition, ''] + lines + ['']
        else:
            return ['.. %s::' % admonition, '']

    def _format_block(self, prefix: str, lines: List[str], padding: str = None) -> List[str]:
        if lines:
            if padding is None:
                padding = ' ' * len(prefix)
            result_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    result_lines.append((prefix + line).rstrip())
                elif line:
                    result_lines.append(padding + line)
                else:
                    result_lines.append('')
            return result_lines
        else:
            return [prefix]
```
### 25 - sphinx/ext/napoleon/docstring.py:

Start line: 822, End line: 835

```python
class GoogleDocstring:

    def _lookup_annotation(self, _name: str) -> str:
        if self._config.napoleon_attr_annotations:
            if self._what in ("module", "class", "exception") and self._obj:
                # cache the class annotations
                if not hasattr(self, "_annotations"):
                    localns = getattr(self._config, "autodoc_type_aliases", {})
                    localns.update(getattr(
                                   self._config, "napoleon_type_aliases", {}
                                   ) or {})
                    self._annotations = get_type_hints(self._obj, None, localns)
                if _name in self._annotations:
                    return stringify_annotation(self._annotations[_name])
        # No annotation found
        return ""
```
### 26 - sphinx/ext/napoleon/docstring.py:

Start line: 1010, End line: 1107

```python
class NumpyDocstring(GoogleDocstring):
    """Convert NumPy style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`sphinx.ext.napoleon.Config` object.


    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Parameters
    ... ----------
    ... arg1 : int
    ...     Description of `arg1`
    ... arg2 : str
    ...     Description of `arg2`
    ... Returns
    ... -------
    ... str
    ...     Description of return value.
    ... '''
    >>> print(NumpyDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    Methods
    -------
    __str__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        str
            UTF-8 encoded version of the docstring.

    __unicode__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        unicode
            Unicode version of the docstring.

    lines()
        Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

    """
    def __init__(self, docstring: Union[str, List[str]], config: SphinxConfig = None,
                 app: Sphinx = None, what: str = '', name: str = '',
                 obj: Any = None, options: Any = None) -> None:
        self._directive_sections = ['.. index::']
        super().__init__(docstring, config, app, what, name, obj, options)
```
### 27 - sphinx/ext/napoleon/docstring.py:

Start line: 792, End line: 820

```python
class GoogleDocstring:

    def _qualify_name(self, attr_name: str, klass: "Type") -> str:
        if klass and '.' not in attr_name:
            if attr_name.startswith('~'):
                attr_name = attr_name[1:]
            try:
                q = klass.__qualname__
            except AttributeError:
                q = klass.__name__
            return '~%s.%s' % (q, attr_name)
        return attr_name

    def _strip_empty(self, lines: List[str]) -> List[str]:
        if lines:
            start = -1
            for i, line in enumerate(lines):
                if line:
                    start = i
                    break
            if start == -1:
                lines = []
            end = -1
            for i in reversed(range(len(lines))):
                line = lines[i]
                if line:
                    end = i
                    break
            if start > 0 or end + 1 < len(lines):
                lines = lines[start:end + 1]
        return lines
```
### 28 - sphinx/ext/napoleon/docstring.py:

Start line: 1109, End line: 1129

```python
class NumpyDocstring(GoogleDocstring):

    def _get_location(self) -> str:
        try:
            filepath = inspect.getfile(self._obj) if self._obj is not None else None
        except TypeError:
            filepath = None
        name = self._name

        if filepath is None and name is None:
            return None
        elif filepath is None:
            filepath = ""

        return ":".join([filepath, "docstring of %s" % name])

    def _escape_args_and_kwargs(self, name: str) -> str:
        func = super()._escape_args_and_kwargs

        if ", " in name:
            return ", ".join(func(param) for param in name.split(", "))
        else:
            return func(name)
```
### 29 - sphinx/ext/napoleon/docstring.py:

Start line: 1131, End line: 1157

```python
class NumpyDocstring(GoogleDocstring):

    def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
                       ) -> Tuple[str, str, List[str]]:
        line = next(self._line_iter)
        if parse_type:
            _name, _, _type = self._partition_field_on_colon(line)
        else:
            _name, _type = line, ''
        _name, _type = _name.strip(), _type.strip()
        _name = self._escape_args_and_kwargs(_name)

        if parse_type and not _type:
            _type = self._lookup_annotation(_name)

        if prefer_type and not _type:
            _type, _name = _name, _type

        if self._config.napoleon_preprocess_types:
            _type = _convert_numpy_type_spec(
                _type,
                location=self._get_location(),
                translations=self._config.napoleon_type_aliases or {},
            )

        indent = self._get_indent(line) + 1
        _desc = self._dedent(self._consume_indented_block(indent))
        _desc = self.__class__(_desc, self._config).lines()
        return _name, _type, _desc
```
### 30 - sphinx/ext/napoleon/docstring.py:

Start line: 558, End line: 590

```python
class GoogleDocstring:

    def _parse(self) -> None:
        self._parsed_lines = self._consume_empty()

        if self._name and self._what in ('attribute', 'data', 'property'):
            # Implicit stop using StopIteration no longer allowed in
            # Python 3.7; see PEP 479
            res = []  # type: List[str]
            try:
                res = self._parse_attribute_docstring()
            except StopIteration:
                pass
            self._parsed_lines.extend(res)
            return

        while self._line_iter.has_next():
            if self._is_section_header():
                try:
                    section = self._consume_section_header()
                    self._is_in_section = True
                    self._section_indent = self._get_current_indent()
                    if _directive_regex.match(section):
                        lines = [section] + self._consume_to_next_section()
                    else:
                        lines = self._sections[section.lower()](section)
                finally:
                    self._is_in_section = False
                    self._section_indent = 0
            else:
                if not self._parsed_lines:
                    lines = self._consume_contiguous() + self._consume_empty()
                else:
                    lines = self._consume_to_next_section()
            self._parsed_lines.extend(lines)
```
### 31 - sphinx/ext/napoleon/docstring.py:

Start line: 251, End line: 271

```python
class GoogleDocstring:

    def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
                       ) -> Tuple[str, str, List[str]]:
        line = next(self._line_iter)

        before, colon, after = self._partition_field_on_colon(line)
        _name, _type, _desc = before, '', after

        if parse_type:
            match = _google_typed_arg_regex.match(before)
            if match:
                _name = match.group(1).strip()
                _type = match.group(2)

        _name = self._escape_args_and_kwargs(_name)

        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line) + 1
        _descs = [_desc] + self._dedent(self._consume_indented_block(indent))
        _descs = self.__class__(_descs, self._config).lines()
        return _name, _type, _descs
```
### 33 - sphinx/ext/napoleon/docstring.py:

Start line: 273, End line: 284

```python
class GoogleDocstring:

    def _consume_fields(self, parse_type: bool = True, prefer_type: bool = False,
                        multiple: bool = False) -> List[Tuple[str, str, List[str]]]:
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if multiple and _name:
                for name in _name.split(","):
                    fields.append((name.strip(), _type, _desc))
            elif _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields
```
### 34 - sphinx/ext/napoleon/docstring.py:

Start line: 504, End line: 519

```python
class GoogleDocstring:

    def _is_list(self, lines: List[str]) -> bool:
        if not lines:
            return False
        if _bullet_list_regex.match(lines[0]):
            return True
        if _enumerated_list_regex.match(lines[0]):
            return True
        if len(lines) < 2 or lines[0].endswith('::'):
            return False
        indent = self._get_indent(lines[0])
        next_indent = indent
        for line in lines[1:]:
            if line:
                next_indent = self._get_indent(line)
                break
        return next_indent > indent
```
### 35 - sphinx/ext/napoleon/docstring.py:

Start line: 592, End line: 602

```python
class GoogleDocstring:

    def _parse_admonition(self, admonition: str, section: str) -> List[str]:
        # type (str, str) -> List[str]
        lines = self._consume_to_next_section()
        return self._format_admonition(admonition, lines)

    def _parse_attribute_docstring(self) -> List[str]:
        _type, _desc = self._consume_inline_attribute()
        lines = self._format_field('', '', _desc)
        if _type:
            lines.extend(['', ':type: %s' % _type])
        return lines
```
### 36 - sphinx/ext/napoleon/docstring.py:

Start line: 286, End line: 294

```python
class GoogleDocstring:

    def _consume_inline_attribute(self) -> Tuple[str, List[str]]:
        line = next(self._line_iter)
        _type, colon, _desc = self._partition_field_on_colon(line)
        if not colon or not _desc:
            _type, _desc = _desc, _type
            _desc += colon
        _descs = [_desc] + self._dedent(self._consume_to_end())
        _descs = self.__class__(_descs, self._config).lines()
        return _type, _descs
```
### 37 - sphinx/ext/napoleon/docstring.py:

Start line: 62, End line: 126

```python
class GoogleDocstring:
    """Convert Google style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`sphinx.ext.napoleon.Config` object.


    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Args:
    ...   arg1(int): Description of `arg1`
    ...   arg2(str): Description of `arg2`
    ... Returns:
    ...   str: Description of return value.
    ... '''
    >>> print(GoogleDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    """

    _name_rgx = re.compile(r"^\s*((?::(?P<role>\S+):)?`(?P<name>~?[a-zA-Z0-9_.-]+)`|"
                           r" (?P<name2>~?[a-zA-Z0-9_.-]+))\s*", re.X)
```
### 38 - sphinx/ext/napoleon/docstring.py:

Start line: 128, End line: 213

```python
class GoogleDocstring:

    def __init__(self, docstring: Union[str, List[str]], config: SphinxConfig = None,
                 app: Sphinx = None, what: str = '', name: str = '',
                 obj: Any = None, options: Any = None) -> None:
        self._config = config
        self._app = app

        if not self._config:
            from sphinx.ext.napoleon import Config
            self._config = self._app.config if self._app else Config()  # type: ignore

        if not what:
            if inspect.isclass(obj):
                what = 'class'
            elif inspect.ismodule(obj):
                what = 'module'
            elif callable(obj):
                what = 'function'
            else:
                what = 'object'

        self._what = what
        self._name = name
        self._obj = obj
        self._opt = options
        if isinstance(docstring, str):
            lines = docstring.splitlines()
        else:
            lines = docstring
        self._line_iter = modify_iter(lines, modifier=lambda s: s.rstrip())
        self._parsed_lines = []  # type: List[str]
        self._is_in_section = False
        self._section_indent = 0
        if not hasattr(self, '_directive_sections'):
            self._directive_sections = []  # type: List[str]
        if not hasattr(self, '_sections'):
            self._sections = {
                'args': self._parse_parameters_section,
                'arguments': self._parse_parameters_section,
                'attention': partial(self._parse_admonition, 'attention'),
                'attributes': self._parse_attributes_section,
                'caution': partial(self._parse_admonition, 'caution'),
                'danger': partial(self._parse_admonition, 'danger'),
                'error': partial(self._parse_admonition, 'error'),
                'example': self._parse_examples_section,
                'examples': self._parse_examples_section,
                'hint': partial(self._parse_admonition, 'hint'),
                'important': partial(self._parse_admonition, 'important'),
                'keyword args': self._parse_keyword_arguments_section,
                'keyword arguments': self._parse_keyword_arguments_section,
                'methods': self._parse_methods_section,
                'note': partial(self._parse_admonition, 'note'),
                'notes': self._parse_notes_section,
                'other parameters': self._parse_other_parameters_section,
                'parameters': self._parse_parameters_section,
                'receive': self._parse_receives_section,
                'receives': self._parse_receives_section,
                'return': self._parse_returns_section,
                'returns': self._parse_returns_section,
                'raise': self._parse_raises_section,
                'raises': self._parse_raises_section,
                'references': self._parse_references_section,
                'see also': self._parse_see_also_section,
                'tip': partial(self._parse_admonition, 'tip'),
                'todo': partial(self._parse_admonition, 'todo'),
                'warning': partial(self._parse_admonition, 'warning'),
                'warnings': partial(self._parse_admonition, 'warning'),
                'warn': self._parse_warns_section,
                'warns': self._parse_warns_section,
                'yield': self._parse_yields_section,
                'yields': self._parse_yields_section,
            }  # type: Dict[str, Callable]

        self._load_custom_sections()

        self._parse()

    def __str__(self) -> str:
        """Return the parsed docstring in reStructuredText format.

        Returns
        -------
        unicode
            Unicode version of the docstring.

        """
        return '\n'.join(self.lines())
```
### 39 - sphinx/ext/napoleon/__init__.py:

Start line: 329, End line: 345

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
### 40 - sphinx/ext/napoleon/__init__.py:

Start line: 348, End line: 395

```python
def _process_docstring(app: Sphinx, what: str, name: str, obj: Any,
                       options: Any, lines: List[str]) -> None:
    """Process the docstring for a given python object.

    Called when autodoc has read and processed a docstring. `lines` is a list
    of docstring lines that `_process_docstring` modifies in place to change
    what Sphinx outputs.

    The following settings in conf.py control what styles of docstrings will
    be parsed:

    * ``napoleon_google_docstring`` -- parse Google style docstrings
    * ``napoleon_numpy_docstring`` -- parse NumPy style docstrings

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process.
    what : str
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : str
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : sphinx.ext.autodoc.Options
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.
    lines : list of str
        The lines of the docstring, see above.

        .. note:: `lines` is modified *in place*

    """
    result_lines = lines
    docstring = None  # type: GoogleDocstring
    if app.config.napoleon_numpy_docstring:
        docstring = NumpyDocstring(result_lines, app.config, app, what, name,
                                   obj, options)
        result_lines = docstring.lines()
    if app.config.napoleon_google_docstring:
        docstring = GoogleDocstring(result_lines, app.config, app, what, name,
                                    obj, options)
        result_lines = docstring.lines()
    lines[:] = result_lines[:]
```
### 41 - sphinx/ext/napoleon/docstring.py:

Start line: 974, End line: 1007

```python
def _convert_numpy_type_spec(_type: str, location: str = None, translations: dict = {}) -> str:
    def convert_obj(obj, translations, default_translation):
        translation = translations.get(obj, obj)

        # use :class: (the default) only if obj is not a standard singleton
        if translation in _SINGLETONS and default_translation == ":class:`%s`":
            default_translation = ":obj:`%s`"
        elif translation == "..." and default_translation == ":class:`%s`":
            # allow referencing the builtin ...
            default_translation = ":obj:`%s <Ellipsis>`"

        if _xref_regex.match(translation) is None:
            translation = default_translation % translation

        return translation

    tokens = _tokenize_type_spec(_type)
    combined_tokens = _recombine_set_tokens(tokens)
    types = [
        (token, _token_type(token, location))
        for token in combined_tokens
    ]

    converters = {
        "literal": lambda x: "``%s``" % x,
        "obj": lambda x: convert_obj(x, translations, ":class:`%s`"),
        "control": lambda x: "*%s*" % x,
        "delimiter": lambda x: x,
        "reference": lambda x: x,
    }

    converted = "".join(converters.get(type_)(token) for token, type_ in types)

    return converted
```
### 42 - sphinx/ext/napoleon/docstring.py:

Start line: 894, End line: 912

```python
def _tokenize_type_spec(spec: str) -> List[str]:
    def postprocess(item):
        if _default_regex.match(item):
            default = item[:7]
            # can't be separated by anything other than a single space
            # for now
            other = item[8:]

            return [default, " ", other]
        else:
            return [item]

    tokens = list(
        item
        for raw_token in _token_regex.split(spec)
        for item in postprocess(raw_token)
        if item
    )
    return tokens
```
### 43 - sphinx/ext/napoleon/__init__.py:

Start line: 11, End line: 289

```python
from typing import Any, Dict, List

from sphinx import __display_version__ as __version__
from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
from sphinx.util import inspect


class Config:
```
### 51 - sphinx/ext/napoleon/docstring.py:

Start line: 215, End line: 249

```python
class GoogleDocstring:

    def lines(self) -> List[str]:
        """Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

        """
        return self._parsed_lines

    def _consume_indented_block(self, indent: int = 1) -> List[str]:
        lines = []
        line = self._line_iter.peek()
        while(not self._is_section_break() and
              (not line or self._is_indented(line, indent))):
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines

    def _consume_contiguous(self) -> List[str]:
        lines = []
        while (self._line_iter.has_next() and
               self._line_iter.peek() and
               not self._is_section_header()):
            lines.append(next(self._line_iter))
        return lines

    def _consume_empty(self) -> List[str]:
        lines = []
        line = self._line_iter.peek()
        while self._line_iter.has_next() and not line:
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines
```
### 52 - sphinx/ext/napoleon/docstring.py:

Start line: 838, End line: 891

```python
def _recombine_set_tokens(tokens: List[str]) -> List[str]:
    token_queue = collections.deque(tokens)
    keywords = ("optional", "default")

    def takewhile_set(tokens):
        open_braces = 0
        previous_token = None
        while True:
            try:
                token = tokens.popleft()
            except IndexError:
                break

            if token == ", ":
                previous_token = token
                continue

            if not token.strip():
                continue

            if token in keywords:
                tokens.appendleft(token)
                if previous_token is not None:
                    tokens.appendleft(previous_token)
                break

            if previous_token is not None:
                yield previous_token
                previous_token = None

            if token == "{":
                open_braces += 1
            elif token == "}":
                open_braces -= 1

            yield token

            if open_braces == 0:
                break

    def combine_set(tokens):
        while True:
            try:
                token = tokens.popleft()
            except IndexError:
                break

            if token == "{":
                tokens.appendleft("{")
                yield "".join(takewhile_set(tokens))
            else:
                yield token

    return list(combine_set(token_queue))
```
### 55 - sphinx/ext/napoleon/__init__.py:

Start line: 398, End line: 481

```python
def _skip_member(app: Sphinx, what: str, name: str, obj: Any,
                 skip: bool, options: Any) -> bool:
    """Determine if private and special class members are included in docs.

    The following settings in conf.py determine if private and special class
    members or init methods are included in the generated documentation:

    * ``napoleon_include_init_with_doc`` --
      include init methods if they have docstrings
    * ``napoleon_include_private_with_doc`` --
      include private members if they have docstrings
    * ``napoleon_include_special_with_doc`` --
      include special members if they have docstrings

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process
    what : str
        A string specifying the type of the object to which the member
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : str
        The name of the member.
    obj : module, class, exception, function, method, or attribute.
        For example, if the member is the __init__ method of class A, then
        `obj` will be `A.__init__`.
    skip : bool
        A boolean indicating if autodoc will skip this member if `_skip_member`
        does not override the decision
    options : sphinx.ext.autodoc.Options
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.

    Returns
    -------
    bool
        True if the member should be skipped during creation of the docs,
        False if it should be included in the docs.

    """
    has_doc = getattr(obj, '__doc__', False)
    is_member = (what == 'class' or what == 'exception' or what == 'module')
    if name != '__weakref__' and has_doc and is_member:
        cls_is_owner = False
        if what == 'class' or what == 'exception':
            qualname = getattr(obj, '__qualname__', '')
            cls_path, _, _ = qualname.rpartition('.')
            if cls_path:
                try:
                    if '.' in cls_path:
                        import functools
                        import importlib

                        mod = importlib.import_module(obj.__module__)
                        mod_path = cls_path.split('.')
                        cls = functools.reduce(getattr, mod_path, mod)
                    else:
                        cls = inspect.unwrap(obj).__globals__[cls_path]
                except Exception:
                    cls_is_owner = False
                else:
                    cls_is_owner = (cls and hasattr(cls, name) and  # type: ignore
                                    name in cls.__dict__)
            else:
                cls_is_owner = False

        if what == 'module' or cls_is_owner:
            is_init = (name == '__init__')
            is_special = (not is_init and name.startswith('__') and
                          name.endswith('__'))
            is_private = (not is_init and not is_special and
                          name.startswith('_'))
            inc_init = app.config.napoleon_include_init_with_doc
            inc_special = app.config.napoleon_include_special_with_doc
            inc_private = app.config.napoleon_include_private_with_doc
            if ((is_special and inc_special) or
                    (is_private and inc_private) or
                    (is_init and inc_init)):
                return False
    return None
```
### 126 - sphinx/ext/napoleon/docstring.py:

Start line: 915, End line: 971

```python
def _token_type(token: str, location: str = None) -> str:
    def is_numeric(token):
        try:
            # use complex to make sure every numeric value is detected as literal
            complex(token)
        except ValueError:
            return False
        else:
            return True

    if token.startswith(" ") or token.endswith(" "):
        type_ = "delimiter"
    elif (
            is_numeric(token) or
            (token.startswith("{") and token.endswith("}")) or
            (token.startswith('"') and token.endswith('"')) or
            (token.startswith("'") and token.endswith("'"))
    ):
        type_ = "literal"
    elif token.startswith("{"):
        logger.warning(
            __("invalid value set (missing closing brace): %s"),
            token,
            location=location,
        )
        type_ = "literal"
    elif token.endswith("}"):
        logger.warning(
            __("invalid value set (missing opening brace): %s"),
            token,
            location=location,
        )
        type_ = "literal"
    elif token.startswith("'") or token.startswith('"'):
        logger.warning(
            __("malformed string literal (missing closing quote): %s"),
            token,
            location=location,
        )
        type_ = "literal"
    elif token.endswith("'") or token.endswith('"'):
        logger.warning(
            __("malformed string literal (missing opening quote): %s"),
            token,
            location=location,
        )
        type_ = "literal"
    elif token in ("optional", "default"):
        # default is not a official keyword (yet) but supported by the
        # reference implementation (numpydoc) and widely used
        type_ = "control"
    elif _xref_regex.match(token):
        type_ = "reference"
    else:
        type_ = "obj"

    return type_
```
