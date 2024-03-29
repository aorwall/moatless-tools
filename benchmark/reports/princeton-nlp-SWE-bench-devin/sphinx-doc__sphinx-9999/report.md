# sphinx-doc__sphinx-9999

| **sphinx-doc/sphinx** | `4e8bca2f2ffd6e3f1a4de4403de9e4600497fc61` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3041 |
| **Any found context length** | 3041 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/writers/latex.py b/sphinx/writers/latex.py
--- a/sphinx/writers/latex.py
+++ b/sphinx/writers/latex.py
@@ -1092,8 +1092,8 @@ def visit_term(self, node: Element) -> None:
             ctx = r'\phantomsection'
             for node_id in node['ids']:
                 ctx += self.hypertarget(node_id, anchor=False)
-        ctx += r'}] \leavevmode'
-        self.body.append(r'\item[{')
+        ctx += r'}'
+        self.body.append(r'\sphinxlineitem{')
         self.context.append(ctx)
 
     def depart_term(self, node: Element) -> None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/writers/latex.py | 1095 | 1096 | 3 | 3 | 3041


## Problem Statement

```
Latex: terms are not separated by a newline
### Describe the bug

I use simple indentations for terms and their explanation:

Example:
\`\`\`rst
Listing:

:samp:`{file}.cc` :samp:`{file}.cp` :samp:`{file}.cxx` :samp:`{file}.cpp` :samp:`{file}.CPP` :samp:`{file}.c++` :samp:`{file}.C`
  C++ source code that must be preprocessed.  Note that in :samp:`.cxx`,
  the last two letters must both be literally :samp:`x`.  Likewise,
  :samp:`.C` refers to a literal capital C.

:samp:`{file}.mm` :samp:`{file}.M`
  Objective-C++ source code that must be preprocessed.

:samp:`{file}.mii`
  Objective-C++ source code that should not be preprocessed.

:samp:`{file}.hh` :samp:`{file}.H` :samp:`{file}.hp` :samp:`{file}.hxx` :samp:`{file}.hpp` :samp:`{file}.HPP` :samp:`{file}.h++` :samp:`{file}.tcc`
  C++ header file to be turned into a precompiled header or Ada spec
\`\`\`

Which results in the following HTML output:

Alabaster:

![Screenshot from 2021-12-17 12-49-36](https://user-images.githubusercontent.com/2658545/146541051-3d7144ca-e155-41a9-a6f3-3b7ee48efb8e.png)

RTD theme:

![Screenshot from 2021-12-17 12-49-50](https://user-images.githubusercontent.com/2658545/146541053-a8164c2b-5ed7-48b2-b5bb-2ec304eefea4.png)

While xelatex output does not contain a new line:

![Screenshot from 2021-12-17 12-50-53](https://user-images.githubusercontent.com/2658545/146541114-0d1ebd78-1ae3-4fed-b06d-7ed45fe1db33.png)

@jfbu

### How to Reproduce

Build the snippet.

### Expected behavior

_No response_

### Your project

Build the snippet

### Screenshots

_No response_

### OS

Linux

### Python version

3.8

### Sphinx version

4.3.0

### Sphinx extensions

_No response_

### Extra tools

_No response_

### Additional context

_No response_
Latex: terms are not separated by a newline
### Describe the bug

I use simple indentations for terms and their explanation:

Example:
\`\`\`rst
Listing:

:samp:`{file}.cc` :samp:`{file}.cp` :samp:`{file}.cxx` :samp:`{file}.cpp` :samp:`{file}.CPP` :samp:`{file}.c++` :samp:`{file}.C`
  C++ source code that must be preprocessed.  Note that in :samp:`.cxx`,
  the last two letters must both be literally :samp:`x`.  Likewise,
  :samp:`.C` refers to a literal capital C.

:samp:`{file}.mm` :samp:`{file}.M`
  Objective-C++ source code that must be preprocessed.

:samp:`{file}.mii`
  Objective-C++ source code that should not be preprocessed.

:samp:`{file}.hh` :samp:`{file}.H` :samp:`{file}.hp` :samp:`{file}.hxx` :samp:`{file}.hpp` :samp:`{file}.HPP` :samp:`{file}.h++` :samp:`{file}.tcc`
  C++ header file to be turned into a precompiled header or Ada spec
\`\`\`

Which results in the following HTML output:

Alabaster:

![Screenshot from 2021-12-17 12-49-36](https://user-images.githubusercontent.com/2658545/146541051-3d7144ca-e155-41a9-a6f3-3b7ee48efb8e.png)

RTD theme:

![Screenshot from 2021-12-17 12-49-50](https://user-images.githubusercontent.com/2658545/146541053-a8164c2b-5ed7-48b2-b5bb-2ec304eefea4.png)

While xelatex output does not contain a new line:

![Screenshot from 2021-12-17 12-50-53](https://user-images.githubusercontent.com/2658545/146541114-0d1ebd78-1ae3-4fed-b06d-7ed45fe1db33.png)

@jfbu

### How to Reproduce

Build the snippet.

### Expected behavior

_No response_

### Your project

Build the snippet

### Screenshots

_No response_

### OS

Linux

### Python version

3.8

### Sphinx version

4.3.0

### Sphinx extensions

_No response_

### Extra tools

_No response_

### Additional context

_No response_
LaTeX: glossary terms with common definition are rendered with too much vertical whitespace
### Describe the bug

as in title

### How to Reproduce

\`\`\`
.. glossary::
   :sorted:

   boson
      Particle with integer spin.

   *fermion*
      Particle with half-integer spin.

   tauon
   myon
   electron
      Examples for fermions.

   über
      Gewisse

\`\`\`

and `make latexpdf`

### Expected behavior

_No response_

### Your project

see code snippet

### Screenshots

![Capture d’écran 2021-12-20 à 12 09 48](https://user-images.githubusercontent.com/2589111/146820019-b58a287e-ec41-483a-8013-85e347b221db.png)


### OS

Mac

### Python version

3.8.0

### Sphinx version

4.3.2



Latex: terms are not separated by a newline
### Describe the bug

I use simple indentations for terms and their explanation:

Example:
\`\`\`rst
Listing:

:samp:`{file}.cc` :samp:`{file}.cp` :samp:`{file}.cxx` :samp:`{file}.cpp` :samp:`{file}.CPP` :samp:`{file}.c++` :samp:`{file}.C`
  C++ source code that must be preprocessed.  Note that in :samp:`.cxx`,
  the last two letters must both be literally :samp:`x`.  Likewise,
  :samp:`.C` refers to a literal capital C.

:samp:`{file}.mm` :samp:`{file}.M`
  Objective-C++ source code that must be preprocessed.

:samp:`{file}.mii`
  Objective-C++ source code that should not be preprocessed.

:samp:`{file}.hh` :samp:`{file}.H` :samp:`{file}.hp` :samp:`{file}.hxx` :samp:`{file}.hpp` :samp:`{file}.HPP` :samp:`{file}.h++` :samp:`{file}.tcc`
  C++ header file to be turned into a precompiled header or Ada spec
\`\`\`

Which results in the following HTML output:

Alabaster:

![Screenshot from 2021-12-17 12-49-36](https://user-images.githubusercontent.com/2658545/146541051-3d7144ca-e155-41a9-a6f3-3b7ee48efb8e.png)

RTD theme:

![Screenshot from 2021-12-17 12-49-50](https://user-images.githubusercontent.com/2658545/146541053-a8164c2b-5ed7-48b2-b5bb-2ec304eefea4.png)

While xelatex output does not contain a new line:

![Screenshot from 2021-12-17 12-50-53](https://user-images.githubusercontent.com/2658545/146541114-0d1ebd78-1ae3-4fed-b06d-7ed45fe1db33.png)

@jfbu

### How to Reproduce

Build the snippet.

### Expected behavior

_No response_

### Your project

Build the snippet

### Screenshots

_No response_

### OS

Linux

### Python version

3.8

### Sphinx version

4.3.0

### Sphinx extensions

_No response_

### Extra tools

_No response_

### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/builders/latex/constants.py | 126 | 217| 1040 | 1040 | 2245 | 
| 2 | 2 sphinx/util/smartypants.py | 28 | 127| 1450 | 2490 | 6388 | 
| **-> 3 <-** | **3 sphinx/writers/latex.py** | 1063 | 1133| 551 | 3041 | 25416 | 
| 4 | 3 sphinx/builders/latex/constants.py | 11 | 72| 612 | 3653 | 25416 | 
| 5 | **3 sphinx/writers/latex.py** | 1321 | 1389| 683 | 4336 | 25416 | 
| 6 | **3 sphinx/writers/latex.py** | 266 | 401| 1503 | 5839 | 25416 | 
| 7 | **3 sphinx/writers/latex.py** | 526 | 590| 541 | 6380 | 25416 | 
| 8 | **3 sphinx/writers/latex.py** | 640 | 741| 826 | 7206 | 25416 | 
| 9 | **3 sphinx/writers/latex.py** | 743 | 811| 566 | 7772 | 25416 | 
| 10 | **3 sphinx/writers/latex.py** | 403 | 453| 493 | 8265 | 25416 | 
| 11 | **3 sphinx/writers/latex.py** | 1603 | 1662| 496 | 8761 | 25416 | 
| 12 | 4 sphinx/highlighting.py | 11 | 68| 613 | 9374 | 26955 | 
| 13 | **4 sphinx/writers/latex.py** | 1513 | 1576| 620 | 9994 | 26955 | 
| 14 | 4 sphinx/builders/latex/constants.py | 74 | 124| 537 | 10531 | 26955 | 
| 15 | 5 sphinx/writers/texinfo.py | 970 | 1087| 822 | 11353 | 39318 | 
| 16 | **5 sphinx/writers/latex.py** | 1447 | 1511| 865 | 12218 | 39318 | 
| 17 | **5 sphinx/writers/latex.py** | 1675 | 1733| 563 | 12781 | 39318 | 
| 18 | **5 sphinx/writers/latex.py** | 1825 | 1897| 570 | 13351 | 39318 | 
| 19 | 6 sphinx/builders/latex/__init__.py | 42 | 97| 825 | 14176 | 44901 | 
| 20 | 6 sphinx/writers/texinfo.py | 1312 | 1400| 699 | 14875 | 44901 | 
| 21 | 7 doc/conf.py | 1 | 82| 732 | 15607 | 46392 | 
| 22 | 7 sphinx/writers/texinfo.py | 860 | 968| 839 | 16446 | 46392 | 
| 23 | 7 sphinx/builders/latex/__init__.py | 11 | 40| 315 | 16761 | 46392 | 
| 24 | 8 sphinx/writers/text.py | 1060 | 1172| 813 | 17574 | 55391 | 
| 25 | 9 sphinx/util/texescape.py | 11 | 63| 632 | 18206 | 57106 | 
| 26 | 10 sphinx/writers/html.py | 365 | 418| 443 | 18649 | 64821 | 
| 27 | 11 sphinx/domains/std.py | 320 | 438| 934 | 19583 | 75069 | 
| 28 | 11 sphinx/builders/latex/__init__.py | 212 | 249| 488 | 20071 | 75069 | 
| 29 | **11 sphinx/writers/latex.py** | 455 | 492| 396 | 20467 | 75069 | 
| 30 | 11 sphinx/highlighting.py | 71 | 180| 873 | 21340 | 75069 | 
| 31 | 11 sphinx/writers/text.py | 782 | 889| 831 | 22171 | 75069 | 
| 32 | **11 sphinx/writers/latex.py** | 1735 | 1784| 552 | 22723 | 75069 | 
| 33 | 11 sphinx/writers/texinfo.py | 748 | 858| 808 | 23531 | 75069 | 
| 34 | 11 sphinx/domains/std.py | 290 | 317| 264 | 23795 | 75069 | 
| 35 | 12 sphinx/cmd/quickstart.py | 11 | 119| 756 | 24551 | 80639 | 
| 36 | **12 sphinx/writers/latex.py** | 866 | 905| 309 | 24860 | 80639 | 
| 37 | **12 sphinx/writers/latex.py** | 1032 | 1061| 334 | 25194 | 80639 | 
| 38 | **12 sphinx/writers/latex.py** | 1012 | 1030| 204 | 25398 | 80639 | 
| 39 | 12 sphinx/writers/text.py | 945 | 1058| 873 | 26271 | 80639 | 
| 40 | 12 sphinx/writers/texinfo.py | 1089 | 1185| 830 | 27101 | 80639 | 
| 41 | 12 sphinx/builders/latex/__init__.py | 261 | 306| 442 | 27543 | 80639 | 
| 42 | 12 sphinx/builders/latex/__init__.py | 451 | 511| 521 | 28064 | 80639 | 
| 43 | **12 sphinx/writers/latex.py** | 1977 | 2002| 195 | 28259 | 80639 | 
| 44 | 12 sphinx/util/texescape.py | 64 | 126| 727 | 28986 | 80639 | 
| 45 | 12 sphinx/builders/latex/__init__.py | 326 | 367| 464 | 29450 | 80639 | 
| 46 | **12 sphinx/writers/latex.py** | 73 | 94| 169 | 29619 | 80639 | 
| 47 | **12 sphinx/writers/latex.py** | 1391 | 1445| 465 | 30084 | 80639 | 
| 48 | 13 sphinx/domains/cpp.py | 1513 | 2280| 6612 | 36696 | 148384 | 
| 49 | **13 sphinx/writers/latex.py** | 1578 | 1601| 297 | 36993 | 148384 | 
| 50 | 13 sphinx/builders/latex/__init__.py | 251 | 259| 136 | 37129 | 148384 | 
| 51 | 13 sphinx/writers/html.py | 721 | 819| 769 | 37898 | 148384 | 
| 52 | 13 sphinx/writers/text.py | 663 | 780| 839 | 38737 | 148384 | 
| 53 | 14 sphinx/writers/html5.py | 152 | 216| 605 | 39342 | 155686 | 
| 54 | **14 sphinx/writers/latex.py** | 1279 | 1319| 488 | 39830 | 155686 | 
| 55 | **14 sphinx/writers/latex.py** | 1916 | 1966| 366 | 40196 | 155686 | 
| 56 | 14 sphinx/writers/texinfo.py | 1506 | 1569| 463 | 40659 | 155686 | 
| 57 | **14 sphinx/writers/latex.py** | 943 | 978| 499 | 41158 | 155686 | 
| 58 | 15 sphinx/builders/latex/transforms.py | 359 | 437| 665 | 41823 | 160010 | 
| 59 | 16 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 41998 | 160918 | 
| 60 | **16 sphinx/writers/latex.py** | 1786 | 1808| 234 | 42232 | 160918 | 
| 61 | **16 sphinx/writers/latex.py** | 14 | 70| 441 | 42673 | 160918 | 
| 62 | 16 sphinx/writers/text.py | 526 | 643| 716 | 43389 | 160918 | 
| 63 | **16 sphinx/writers/latex.py** | 2020 | 2053| 314 | 43703 | 160918 | 
| 64 | 16 sphinx/builders/latex/__init__.py | 435 | 448| 180 | 43883 | 160918 | 
| 65 | 17 sphinx/domains/c.py | 11 | 785| 6550 | 50433 | 193401 | 
| 66 | **17 sphinx/writers/latex.py** | 592 | 638| 450 | 50883 | 193401 | 
| 67 | 17 sphinx/builders/latex/__init__.py | 99 | 145| 381 | 51264 | 193401 | 
| 68 | 17 sphinx/writers/html5.py | 661 | 747| 686 | 51950 | 193401 | 
| 70 | 18 sphinx/builders/latex/__init__.py | 166 | 195| 266 | 52903 | 199714 | 


### Hint

```
I don't think this is not a bug. It's not promised a term and description of the definition list are displayed as line-folded.

But I agree it's better to fold them as HTML does.
Great, thanks for the suggested pull request!
@marxin It may take some time before a solution is found because the natural #9988 approach breaks some numpy's LaTeX hack. As a work-around you may try to add to your project their hack
\`\`\`
latex_elements = {
    'preamble': r'''
% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\makeatletter
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
\makeatother
% 2) fix bug in expdlist's way of breaking the line after long item label
\makeatletter
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
\makeatother
''',
}
\`\`\`
I am not completely happy with the resulting looks (see https://github.com/sphinx-doc/sphinx/pull/9988#issuecomment-997227939) but it may serve temporarily.
I don't think this is not a bug. It's not promised a term and description of the definition list are displayed as line-folded.

But I agree it's better to fold them as HTML does.
Great, thanks for the suggested pull request!
@marxin It may take some time before a solution is found because the natural #9988 approach breaks some numpy's LaTeX hack. As a work-around you may try to add to your project their hack
\`\`\`
latex_elements = {
    'preamble': r'''
% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\makeatletter
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
\makeatother
% 2) fix bug in expdlist's way of breaking the line after long item label
\makeatletter
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
\makeatother
''',
}
\`\`\`
I am not completely happy with the resulting looks (see https://github.com/sphinx-doc/sphinx/pull/9988#issuecomment-997227939) but it may serve temporarily.

I don't think this is not a bug. It's not promised a term and description of the definition list are displayed as line-folded.

But I agree it's better to fold them as HTML does.
Great, thanks for the suggested pull request!
@marxin It may take some time before a solution is found because the natural #9988 approach breaks some numpy's LaTeX hack. As a work-around you may try to add to your project their hack
\`\`\`
latex_elements = {
    'preamble': r'''
% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\makeatletter
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
\makeatother
% 2) fix bug in expdlist's way of breaking the line after long item label
\makeatletter
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
\makeatother
''',
}
\`\`\`
I am not completely happy with the resulting looks (see https://github.com/sphinx-doc/sphinx/pull/9988#issuecomment-997227939) but it may serve temporarily.
```

## Patch

```diff
diff --git a/sphinx/writers/latex.py b/sphinx/writers/latex.py
--- a/sphinx/writers/latex.py
+++ b/sphinx/writers/latex.py
@@ -1092,8 +1092,8 @@ def visit_term(self, node: Element) -> None:
             ctx = r'\phantomsection'
             for node_id in node['ids']:
                 ctx += self.hypertarget(node_id, anchor=False)
-        ctx += r'}] \leavevmode'
-        self.body.append(r'\item[{')
+        ctx += r'}'
+        self.body.append(r'\sphinxlineitem{')
         self.context.append(ctx)
 
     def depart_term(self, node: Element) -> None:

```

## Test Patch

```diff
diff --git a/tests/test_build_latex.py b/tests/test_build_latex.py
--- a/tests/test_build_latex.py
+++ b/tests/test_build_latex.py
@@ -834,18 +834,18 @@ def test_latex_show_urls_is_inline(app, status, warning):
             'Footnote inside footnote\n%\n\\end{footnotetext}\\ignorespaces') in result
     assert ('\\sphinxhref{http://sphinx-doc.org/~test/}{URL including tilde} '
             '(http://sphinx\\sphinxhyphen{}doc.org/\\textasciitilde{}test/)') in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}{URL in term} '
-            '(http://sphinx\\sphinxhyphen{}doc.org/)}] '
-            '\\leavevmode\n\\sphinxAtStartPar\nDescription' in result)
-    assert ('\\item[{Footnote in term \\sphinxfootnotemark[6]}] '
-            '\\leavevmode%\n\\begin{footnotetext}[6]'
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}{URL in term} '
+            '(http://sphinx\\sphinxhyphen{}doc.org/)}'
+            '\n\\sphinxAtStartPar\nDescription' in result)
+    assert ('\\sphinxlineitem{Footnote in term \\sphinxfootnotemark[6]}'
+            '%\n\\begin{footnotetext}[6]'
             '\\phantomsection\\label{\\thesphinxscope.6}%\n'
             '\\sphinxAtStartFootnote\n'
             'Footnote in term\n%\n\\end{footnotetext}\\ignorespaces '
             '\n\\sphinxAtStartPar\nDescription') in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist} '
-            '(http://sphinx\\sphinxhyphen{}doc.org/)}] '
-            '\\leavevmode\n\\sphinxAtStartPar\nDescription') in result
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist} '
+            '(http://sphinx\\sphinxhyphen{}doc.org/)}'
+            '\n\\sphinxAtStartPar\nDescription') in result
     assert '\\sphinxurl{https://github.com/sphinx-doc/sphinx}\n' in result
     assert ('\\sphinxhref{mailto:sphinx-dev@googlegroups.com}'
             '{sphinx\\sphinxhyphen{}dev@googlegroups.com}') in result
@@ -893,22 +893,22 @@ def test_latex_show_urls_is_footnote(app, status, warning):
     assert ('\\sphinxhref{http://sphinx-doc.org/~test/}{URL including tilde}'
             '%\n\\begin{footnote}[5]\\sphinxAtStartFootnote\n'
             '\\sphinxnolinkurl{http://sphinx-doc.org/~test/}\n%\n\\end{footnote}') in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}'
-            '{URL in term}\\sphinxfootnotemark[9]}] '
-            '\\leavevmode%\n\\begin{footnotetext}[9]'
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}'
+            '{URL in term}\\sphinxfootnotemark[9]}'
+            '%\n\\begin{footnotetext}[9]'
             '\\phantomsection\\label{\\thesphinxscope.9}%\n'
             '\\sphinxAtStartFootnote\n'
             '\\sphinxnolinkurl{http://sphinx-doc.org/}\n%\n'
             '\\end{footnotetext}\\ignorespaces \n\\sphinxAtStartPar\nDescription') in result
-    assert ('\\item[{Footnote in term \\sphinxfootnotemark[11]}] '
-            '\\leavevmode%\n\\begin{footnotetext}[11]'
+    assert ('\\sphinxlineitem{Footnote in term \\sphinxfootnotemark[11]}'
+            '%\n\\begin{footnotetext}[11]'
             '\\phantomsection\\label{\\thesphinxscope.11}%\n'
             '\\sphinxAtStartFootnote\n'
             'Footnote in term\n%\n\\end{footnotetext}\\ignorespaces '
             '\n\\sphinxAtStartPar\nDescription') in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist}'
-            '\\sphinxfootnotemark[10]}] '
-            '\\leavevmode%\n\\begin{footnotetext}[10]'
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist}'
+            '\\sphinxfootnotemark[10]}'
+            '%\n\\begin{footnotetext}[10]'
             '\\phantomsection\\label{\\thesphinxscope.10}%\n'
             '\\sphinxAtStartFootnote\n'
             '\\sphinxnolinkurl{http://sphinx-doc.org/}\n%\n'
@@ -955,16 +955,16 @@ def test_latex_show_urls_is_no(app, status, warning):
             '\\sphinxAtStartFootnote\n'
             'Footnote inside footnote\n%\n\\end{footnotetext}\\ignorespaces') in result
     assert '\\sphinxhref{http://sphinx-doc.org/~test/}{URL including tilde}' in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}{URL in term}}] '
-            '\\leavevmode\n\\sphinxAtStartPar\nDescription') in result
-    assert ('\\item[{Footnote in term \\sphinxfootnotemark[6]}] '
-            '\\leavevmode%\n\\begin{footnotetext}[6]'
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}{URL in term}}'
+            '\n\\sphinxAtStartPar\nDescription') in result
+    assert ('\\sphinxlineitem{Footnote in term \\sphinxfootnotemark[6]}'
+            '%\n\\begin{footnotetext}[6]'
             '\\phantomsection\\label{\\thesphinxscope.6}%\n'
             '\\sphinxAtStartFootnote\n'
             'Footnote in term\n%\n\\end{footnotetext}\\ignorespaces '
             '\n\\sphinxAtStartPar\nDescription') in result
-    assert ('\\item[{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist}}] '
-            '\\leavevmode\n\\sphinxAtStartPar\nDescription') in result
+    assert ('\\sphinxlineitem{\\sphinxhref{http://sphinx-doc.org/}{Term in deflist}}'
+            '\n\\sphinxAtStartPar\nDescription') in result
     assert ('\\sphinxurl{https://github.com/sphinx-doc/sphinx}\n' in result)
     assert ('\\sphinxhref{mailto:sphinx-dev@googlegroups.com}'
             '{sphinx\\sphinxhyphen{}dev@googlegroups.com}\n') in result
@@ -1454,23 +1454,23 @@ def test_latex_glossary(app, status, warning):
     app.builder.build_all()
 
     result = (app.outdir / 'python.tex').read_text()
-    assert ('\\item[{ähnlich\\index{ähnlich@\\spxentry{ähnlich}|spxpagem}'
+    assert (r'\sphinxlineitem{ähnlich\index{ähnlich@\spxentry{ähnlich}|spxpagem}'
             r'\phantomsection'
-            r'\label{\detokenize{index:term-ahnlich}}}] \leavevmode' in result)
-    assert (r'\item[{boson\index{boson@\spxentry{boson}|spxpagem}\phantomsection'
-            r'\label{\detokenize{index:term-boson}}}] \leavevmode' in result)
-    assert (r'\item[{\sphinxstyleemphasis{fermion}'
+            r'\label{\detokenize{index:term-ahnlich}}}' in result)
+    assert (r'\sphinxlineitem{boson\index{boson@\spxentry{boson}|spxpagem}\phantomsection'
+            r'\label{\detokenize{index:term-boson}}}' in result)
+    assert (r'\sphinxlineitem{\sphinxstyleemphasis{fermion}'
             r'\index{fermion@\spxentry{fermion}|spxpagem}'
             r'\phantomsection'
-            r'\label{\detokenize{index:term-fermion}}}] \leavevmode' in result)
-    assert (r'\item[{tauon\index{tauon@\spxentry{tauon}|spxpagem}\phantomsection'
-            r'\label{\detokenize{index:term-tauon}}}] \leavevmode'
-            r'\item[{myon\index{myon@\spxentry{myon}|spxpagem}\phantomsection'
-            r'\label{\detokenize{index:term-myon}}}] \leavevmode'
-            r'\item[{electron\index{electron@\spxentry{electron}|spxpagem}\phantomsection'
-            r'\label{\detokenize{index:term-electron}}}] \leavevmode' in result)
-    assert ('\\item[{über\\index{über@\\spxentry{über}|spxpagem}\\phantomsection'
-            r'\label{\detokenize{index:term-uber}}}] \leavevmode' in result)
+            r'\label{\detokenize{index:term-fermion}}}' in result)
+    assert (r'\sphinxlineitem{tauon\index{tauon@\spxentry{tauon}|spxpagem}\phantomsection'
+            r'\label{\detokenize{index:term-tauon}}}'
+            r'\sphinxlineitem{myon\index{myon@\spxentry{myon}|spxpagem}\phantomsection'
+            r'\label{\detokenize{index:term-myon}}}'
+            r'\sphinxlineitem{electron\index{electron@\spxentry{electron}|spxpagem}\phantomsection'
+            r'\label{\detokenize{index:term-electron}}}' in result)
+    assert (r'\sphinxlineitem{über\index{über@\spxentry{über}|spxpagem}\phantomsection'
+            r'\label{\detokenize{index:term-uber}}}' in result)
 
 
 @pytest.mark.sphinx('latex', testroot='latex-labels')

```


## Code snippets

### 1 - sphinx/builders/latex/constants.py:

Start line: 126, End line: 217

```python
ADDITIONAL_SETTINGS: Dict[Any, Dict[str, Any]] = {
    'pdflatex': {
        'inputenc':     '\\usepackage[utf8]{inputenc}',
        'utf8extra':   ('\\ifdefined\\DeclareUnicodeCharacter\n'
                        '% support both utf8 and utf8x syntaxes\n'
                        '  \\ifdefined\\DeclareUnicodeCharacterAsOptional\n'
                        '    \\def\\sphinxDUC#1{\\DeclareUnicodeCharacter{"#1}}\n'
                        '  \\else\n'
                        '    \\let\\sphinxDUC\\DeclareUnicodeCharacter\n'
                        '  \\fi\n'
                        '  \\sphinxDUC{00A0}{\\nobreakspace}\n'
                        '  \\sphinxDUC{2500}{\\sphinxunichar{2500}}\n'
                        '  \\sphinxDUC{2502}{\\sphinxunichar{2502}}\n'
                        '  \\sphinxDUC{2514}{\\sphinxunichar{2514}}\n'
                        '  \\sphinxDUC{251C}{\\sphinxunichar{251C}}\n'
                        '  \\sphinxDUC{2572}{\\textbackslash}\n'
                        '\\fi'),
    },
    'xelatex': {
        'latex_engine': 'xelatex',
        'polyglossia':  '\\usepackage{polyglossia}',
        'babel':        '',
        'fontenc':     ('\\usepackage{fontspec}\n'
                        '\\defaultfontfeatures[\\rmfamily,\\sffamily,\\ttfamily]{}'),
        'fontpkg':      XELATEX_DEFAULT_FONTPKG,
        'fvset':        '\\fvset{fontsize=\\small}',
        'fontsubstitution': '',
        'textgreek':    '',
        'utf8extra':   ('\\catcode`^^^^00a0\\active\\protected\\def^^^^00a0'
                        '{\\leavevmode\\nobreak\\ }'),
    },
    'lualatex': {
        'latex_engine': 'lualatex',
        'polyglossia':  '\\usepackage{polyglossia}',
        'babel':        '',
        'fontenc':     ('\\usepackage{fontspec}\n'
                        '\\defaultfontfeatures[\\rmfamily,\\sffamily,\\ttfamily]{}'),
        'fontpkg':      LUALATEX_DEFAULT_FONTPKG,
        'fvset':        '\\fvset{fontsize=\\small}',
        'fontsubstitution': '',
        'textgreek':    '',
        'utf8extra':   ('\\catcode`^^^^00a0\\active\\protected\\def^^^^00a0'
                        '{\\leavevmode\\nobreak\\ }'),
    },
    'platex': {
        'latex_engine': 'platex',
        'babel':        '',
        'classoptions': ',dvipdfmx',
        'fontpkg':      PDFLATEX_DEFAULT_FONTPKG,
        'fontsubstitution': '',
        'textgreek':    '',
        'fncychap':     '',
        'geometry':     '\\usepackage[dvipdfm]{geometry}',
    },
    'uplatex': {
        'latex_engine': 'uplatex',
        'babel':        '',
        'classoptions': ',dvipdfmx',
        'fontpkg':      PDFLATEX_DEFAULT_FONTPKG,
        'fontsubstitution': '',
        'textgreek':    '',
        'fncychap':     '',
        'geometry':     '\\usepackage[dvipdfm]{geometry}',
    },

    # special settings for latex_engine + language_code
    ('xelatex', 'fr'): {
        # use babel instead of polyglossia by default
        'polyglossia':  '',
        'babel':        '\\usepackage{babel}',
    },
    ('xelatex', 'zh'): {
        'polyglossia':  '',
        'babel':        '\\usepackage{babel}',
        'fontenc':      '\\usepackage{xeCJK}',
        # set formatcom=\xeCJKVerbAddon to prevent xeCJK from adding extra spaces in
        # fancyvrb Verbatim environment.
        'fvset':        '\\fvset{fontsize=\\small,formatcom=\\xeCJKVerbAddon}',
    },
    ('xelatex', 'el'): {
        'fontpkg':      XELATEX_GREEK_DEFAULT_FONTPKG,
    },
}


SHORTHANDOFF = r'''
\ifdefined\shorthandoff
  \ifnum\catcode`\=\string=\active\shorthandoff{=}\fi
  \ifnum\catcode`\"=\active\shorthandoff{"}\fi
\fi
'''
```
### 2 - sphinx/util/smartypants.py:

Start line: 28, End line: 127

```python
import re
import warnings
from typing import Generator, Iterable, Tuple

from docutils.utils import smartquotes

from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.util.docutils import __version_info__ as docutils_version

warnings.warn('sphinx.util.smartypants is deprecated.',
              RemovedInSphinx60Warning)

langquotes = {'af':           '“”‘’',
              'af-x-altquot': '„”‚’',
              'bg':           '„“‚‘',  # Bulgarian, https://bg.wikipedia.org/wiki/Кавички
              'ca':           '«»“”',
              'ca-x-altquot': '“”‘’',
              'cs':           '„“‚‘',
              'cs-x-altquot': '»«›‹',
              'da':           '»«›‹',
              'da-x-altquot': '„“‚‘',
              # 'da-x-altquot2': '””’’',
              'de':           '„“‚‘',
              'de-x-altquot': '»«›‹',
              'de-ch':        '«»‹›',
              'el':           '«»“”',
              'en':           '“”‘’',
              'en-uk-x-altquot': '‘’“”',  # Attention: " → ‘ and ' → “ !
              'eo':           '“”‘’',
              'es':           '«»“”',
              'es-x-altquot': '“”‘’',
              'et':           '„“‚‘',  # no secondary quote listed in
              'et-x-altquot': '«»‹›',  # the sources above (wikipedia.org)
              'eu':           '«»‹›',
              'fi':           '””’’',
              'fi-x-altquot': '»»››',
              'fr':           ('« ', ' »', '“', '”'),  # full no-break space
              'fr-x-altquot': ('« ', ' »', '“', '”'),  # narrow no-break space
              'fr-ch':        '«»‹›',
              'fr-ch-x-altquot': ('« ',  ' »', '‹ ', ' ›'),  # narrow no-break space
              # http://typoguide.ch/
              'gl':           '«»“”',
              'he':           '”“»«',  # Hebrew is RTL, test position:
              'he-x-altquot': '„”‚’',  # low quotation marks are opening.
              # 'he-x-altquot': '“„‘‚',  # RTL: low quotation marks opening
              'hr':           '„”‘’',  # https://hrvatska-tipografija.com/polunavodnici/
              'hr-x-altquot': '»«›‹',
              'hsb':          '„“‚‘',
              'hsb-x-altquot': '»«›‹',
              'hu':           '„”«»',
              'is':           '„“‚‘',
              'it':           '«»“”',
              'it-ch':        '«»‹›',
              'it-x-altquot': '“”‘’',
              # 'it-x-altquot2': '“„‘‚',  # [7] in headlines
              'ja':           '「」『』',
              'lt':           '„“‚‘',
              'lv':           '„“‚‘',
              'mk':           '„“‚‘',  # Macedonian,
              # https://mk.wikipedia.org/wiki/Правопис_и_правоговор_на_македонскиот_јазик
              'nl':           '“”‘’',
              'nl-x-altquot': '„”‚’',
              # 'nl-x-altquot2': '””’’',
              'nb':           '«»’’',  # Norsk bokmål (canonical form 'no')
              'nn':           '«»’’',  # Nynorsk [10]
              'nn-x-altquot': '«»‘’',  # [8], [10]
              # 'nn-x-altquot2': '«»«»',  # [9], [10]
              # 'nn-x-altquot3': '„“‚‘',  # [10]
              'no':           '«»’’',  # Norsk bokmål [10]
              'no-x-altquot': '«»‘’',  # [8], [10]
              # 'no-x-altquot2': '«»«»',  # [9], [10]
              # 'no-x-altquot3': '„“‚‘',  # [10]
              'pl':           '„”«»',
              'pl-x-altquot': '«»‚’',
              # 'pl-x-altquot2': '„”‚’',
              # https://pl.wikipedia.org/wiki/Cudzys%C5%82%C3%B3w
              'pt':           '«»“”',
              'pt-br':        '“”‘’',
              'ro':           '„”«»',
              'ru':           '«»„“',
              'sh':           '„”‚’',  # Serbo-Croatian
              'sh-x-altquot': '»«›‹',
              'sk':           '„“‚‘',  # Slovak
              'sk-x-altquot': '»«›‹',
              'sl':           '„“‚‘',  # Slovenian
              'sl-x-altquot': '»«›‹',
              'sq':           '«»‹›',  # Albanian
              'sq-x-altquot': '“„‘‚',
              'sr':           '„”’’',
              'sr-x-altquot': '»«›‹',
              'sv':           '””’’',
              'sv-x-altquot': '»»››',
              'tr':           '“”‘’',
              'tr-x-altquot': '«»‹›',
              # 'tr-x-altquot2': '“„‘‚',  # [7] antiquated?
              'uk':           '«»„“',
              'uk-x-altquot': '„“‚‘',
              'zh-cn':        '“”‘’',
              'zh-tw':        '「」『』',
              }
```
### 3 - sphinx/writers/latex.py:

Start line: 1063, End line: 1133

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_enumerated_list(self, node: Element) -> None:
        self.body.append(r'\end{enumerate}' + CR)

    def visit_list_item(self, node: Element) -> None:
        # Append "{}" in case the next character is "[", which would break
        # LaTeX's list environment (no numbering and the "[" is not printed).
        self.body.append(r'\item {} ')

    def depart_list_item(self, node: Element) -> None:
        self.body.append(CR)

    def visit_definition_list(self, node: Element) -> None:
        self.body.append(r'\begin{description}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_definition_list(self, node: Element) -> None:
        self.body.append(r'\end{description}' + CR)

    def visit_definition_list_item(self, node: Element) -> None:
        pass

    def depart_definition_list_item(self, node: Element) -> None:
        pass

    def visit_term(self, node: Element) -> None:
        self.in_term += 1
        ctx = ''
        if node.get('ids'):
            ctx = r'\phantomsection'
            for node_id in node['ids']:
                ctx += self.hypertarget(node_id, anchor=False)
        ctx += r'}] \leavevmode'
        self.body.append(r'\item[{')
        self.context.append(ctx)

    def depart_term(self, node: Element) -> None:
        self.body.append(self.context.pop())
        self.in_term -= 1

    def visit_classifier(self, node: Element) -> None:
        self.body.append('{[}')

    def depart_classifier(self, node: Element) -> None:
        self.body.append('{]}')

    def visit_definition(self, node: Element) -> None:
        pass

    def depart_definition(self, node: Element) -> None:
        self.body.append(CR)

    def visit_field_list(self, node: Element) -> None:
        self.body.append(r'\begin{quote}\begin{description}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_field_list(self, node: Element) -> None:
        self.body.append(r'\end{description}\end{quote}' + CR)

    def visit_field(self, node: Element) -> None:
        pass

    def depart_field(self, node: Element) -> None:
        pass

    visit_field_name = visit_term
    depart_field_name = depart_term

    visit_field_body = visit_definition
    depart_field_body = depart_definition
```
### 4 - sphinx/builders/latex/constants.py:

Start line: 11, End line: 72

```python
from typing import Any, Dict

PDFLATEX_DEFAULT_FONTPKG = r'''
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand{\ttdefault}{txtt}
'''

PDFLATEX_DEFAULT_FONTSUBSTITUTION = r'''
\expandafter\ifx\csname T@LGR\endcsname\relax
\else
% LGR was declared as font encoding
  \substitutefont{LGR}{\rmdefault}{cmr}
  \substitutefont{LGR}{\sfdefault}{cmss}
  \substitutefont{LGR}{\ttdefault}{cmtt}
\fi
\expandafter\ifx\csname T@X2\endcsname\relax
  \expandafter\ifx\csname T@T2A\endcsname\relax
  \else
  % T2A was declared as font encoding
    \substitutefont{T2A}{\rmdefault}{cmr}
    \substitutefont{T2A}{\sfdefault}{cmss}
    \substitutefont{T2A}{\ttdefault}{cmtt}
  \fi
\else
% X2 was declared as font encoding
  \substitutefont{X2}{\rmdefault}{cmr}
  \substitutefont{X2}{\sfdefault}{cmss}
  \substitutefont{X2}{\ttdefault}{cmtt}
\fi
'''

XELATEX_DEFAULT_FONTPKG = r'''
\setmainfont{FreeSerif}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Italic,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldItalic
]
\setsansfont{FreeSans}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]
\setmonofont{FreeMono}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]
'''

XELATEX_GREEK_DEFAULT_FONTPKG = (XELATEX_DEFAULT_FONTPKG +
                                 '\n\\newfontfamily\\greekfont{FreeSerif}' +
                                 '\n\\newfontfamily\\greekfontsf{FreeSans}' +
                                 '\n\\newfontfamily\\greekfonttt{FreeMono}')

LUALATEX_DEFAULT_FONTPKG = XELATEX_DEFAULT_FONTPKG
```
### 5 - sphinx/writers/latex.py:

Start line: 1321, End line: 1389

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_figure(self, node: Element) -> None:
        self.body.append(self.context.pop())

    def visit_caption(self, node: Element) -> None:
        self.in_caption += 1
        if isinstance(node.parent, captioned_literal_block):
            self.body.append(r'\sphinxSetupCaptionForVerbatim{')
        elif self.in_minipage and isinstance(node.parent, nodes.figure):
            self.body.append(r'\captionof{figure}{')
        elif self.table and node.parent.tagname == 'figure':
            self.body.append(r'\sphinxfigcaption{')
        else:
            self.body.append(r'\caption{')

    def depart_caption(self, node: Element) -> None:
        self.body.append('}')
        if isinstance(node.parent, nodes.figure):
            labels = self.hypertarget_to(node.parent)
            self.body.append(labels)
        self.in_caption -= 1

    def visit_legend(self, node: Element) -> None:
        self.body.append(CR + r'\begin{sphinxlegend}')

    def depart_legend(self, node: Element) -> None:
        self.body.append(r'\end{sphinxlegend}' + CR)

    def visit_admonition(self, node: Element) -> None:
        self.body.append(CR + r'\begin{sphinxadmonition}{note}')
        self.no_latex_floats += 1

    def depart_admonition(self, node: Element) -> None:
        self.body.append(r'\end{sphinxadmonition}' + CR)
        self.no_latex_floats -= 1

    def _visit_named_admonition(self, node: Element) -> None:
        label = admonitionlabels[node.tagname]
        self.body.append(CR + r'\begin{sphinxadmonition}{%s}{%s:}' %
                         (node.tagname, label))
        self.no_latex_floats += 1

    def _depart_named_admonition(self, node: Element) -> None:
        self.body.append(r'\end{sphinxadmonition}' + CR)
        self.no_latex_floats -= 1

    visit_attention = _visit_named_admonition
    depart_attention = _depart_named_admonition
    visit_caution = _visit_named_admonition
    depart_caution = _depart_named_admonition
    visit_danger = _visit_named_admonition
    depart_danger = _depart_named_admonition
    visit_error = _visit_named_admonition
    depart_error = _depart_named_admonition
    visit_hint = _visit_named_admonition
    depart_hint = _depart_named_admonition
    visit_important = _visit_named_admonition
    depart_important = _depart_named_admonition
    visit_note = _visit_named_admonition
    depart_note = _depart_named_admonition
    visit_tip = _visit_named_admonition
    depart_tip = _depart_named_admonition
    visit_warning = _visit_named_admonition
    depart_warning = _depart_named_admonition

    def visit_versionmodified(self, node: Element) -> None:
        pass

    def depart_versionmodified(self, node: Element) -> None:
        pass
```
### 6 - sphinx/writers/latex.py:

Start line: 266, End line: 401

```python
class LaTeXTranslator(SphinxTranslator):
    builder: "LaTeXBuilder" = None

    secnumdepth = 2  # legacy sphinxhowto.cls uses this, whereas article.cls
    # default is originally 3. For book/report, 2 is already LaTeX default.
    ignore_missing_images = False

    def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
                 theme: "Theme") -> None:
        super().__init__(document, builder)
        self.body: List[str] = []
        self.theme = theme

        # flags
        self.in_title = 0
        self.in_production_list = 0
        self.in_footnote = 0
        self.in_caption = 0
        self.in_term = 0
        self.needs_linetrimming = 0
        self.in_minipage = 0
        self.no_latex_floats = 0
        self.first_document = 1
        self.this_is_the_title = 1
        self.literal_whitespace = 0
        self.in_parsed_literal = 0
        self.compact_list = 0
        self.first_param = 0

        sphinxpkgoptions = []

        # sort out some elements
        self.elements = self.builder.context.copy()

        # initial section names
        self.sectionnames = LATEXSECTIONNAMES[:]
        if self.theme.toplevel_sectioning == 'section':
            self.sectionnames.remove('chapter')

        # determine top section level
        self.top_sectionlevel = 1
        if self.config.latex_toplevel_sectioning:
            try:
                self.top_sectionlevel = \
                    self.sectionnames.index(self.config.latex_toplevel_sectioning)
            except ValueError:
                logger.warning(__('unknown %r toplevel_sectioning for class %r') %
                               (self.config.latex_toplevel_sectioning, self.theme.docclass))

        if self.config.numfig:
            self.numfig_secnum_depth = self.config.numfig_secnum_depth
            if self.numfig_secnum_depth > 0:  # default is 1
                # numfig_secnum_depth as passed to sphinx.sty indices same names as in
                # LATEXSECTIONNAMES but with -1 for part, 0 for chapter, 1 for section...
                if len(self.sectionnames) < len(LATEXSECTIONNAMES) and \
                   self.top_sectionlevel > 0:
                    self.numfig_secnum_depth += self.top_sectionlevel
                else:
                    self.numfig_secnum_depth += self.top_sectionlevel - 1
                # this (minus one) will serve as minimum to LaTeX's secnumdepth
                self.numfig_secnum_depth = min(self.numfig_secnum_depth,
                                               len(LATEXSECTIONNAMES) - 1)
                # if passed key value is < 1 LaTeX will act as if 0; see sphinx.sty
                sphinxpkgoptions.append('numfigreset=%s' % self.numfig_secnum_depth)
            else:
                sphinxpkgoptions.append('nonumfigreset')

        if self.config.numfig and self.config.math_numfig:
            sphinxpkgoptions.append('mathnumfig')

        if (self.config.language not in {None, 'en', 'ja'} and
                'fncychap' not in self.config.latex_elements):
            # use Sonny style if any language specified (except English)
            self.elements['fncychap'] = (r'\usepackage[Sonny]{fncychap}' + CR +
                                         r'\ChNameVar{\Large\normalfont\sffamily}' + CR +
                                         r'\ChTitleVar{\Large\normalfont\sffamily}')

        self.babel = self.builder.babel
        if self.config.language and not self.babel.is_supported_language():
            # emit warning if specified language is invalid
            # (only emitting, nothing changed to processing)
            logger.warning(__('no Babel option known for language %r'),
                           self.config.language)

        minsecnumdepth = self.secnumdepth  # 2 from legacy sphinx manual/howto
        if self.document.get('tocdepth'):
            # reduce tocdepth if `part` or `chapter` is used for top_sectionlevel
            #   tocdepth = -1: show only parts
            #   tocdepth =  0: show parts and chapters
            #   tocdepth =  1: show parts, chapters and sections
            #   tocdepth =  2: show parts, chapters, sections and subsections
            #   ...
            tocdepth = self.document.get('tocdepth', 999) + self.top_sectionlevel - 2
            if len(self.sectionnames) < len(LATEXSECTIONNAMES) and \
               self.top_sectionlevel > 0:
                tocdepth += 1  # because top_sectionlevel is shifted by -1
            if tocdepth > len(LATEXSECTIONNAMES) - 2:  # default is 5 <-> subparagraph
                logger.warning(__('too large :maxdepth:, ignored.'))
                tocdepth = len(LATEXSECTIONNAMES) - 2

            self.elements['tocdepth'] = r'\setcounter{tocdepth}{%d}' % tocdepth
            minsecnumdepth = max(minsecnumdepth, tocdepth)

        if self.config.numfig and (self.config.numfig_secnum_depth > 0):
            minsecnumdepth = max(minsecnumdepth, self.numfig_secnum_depth - 1)

        if minsecnumdepth > self.secnumdepth:
            self.elements['secnumdepth'] = r'\setcounter{secnumdepth}{%d}' %\
                                           minsecnumdepth

        contentsname = document.get('contentsname')
        if contentsname:
            self.elements['contentsname'] = self.babel_renewcommand(r'\contentsname',
                                                                    contentsname)

        if self.elements['maxlistdepth']:
            sphinxpkgoptions.append('maxlistdepth=%s' % self.elements['maxlistdepth'])
        if sphinxpkgoptions:
            self.elements['sphinxpkgoptions'] = '[,%s]' % ','.join(sphinxpkgoptions)
        if self.elements['sphinxsetup']:
            self.elements['sphinxsetup'] = (r'\sphinxsetup{%s}' % self.elements['sphinxsetup'])
        if self.elements['extraclassoptions']:
            self.elements['classoptions'] += ',' + \
                                             self.elements['extraclassoptions']

        self.highlighter = highlighting.PygmentsBridge('latex', self.config.pygments_style,
                                                       latex_engine=self.config.latex_engine)
        self.context: List[Any] = []
        self.descstack: List[str] = []
        self.tables: List[Table] = []
        self.next_table_colspec: str = None
        self.bodystack: List[List[str]] = []
        self.footnote_restricted: Element = None
        self.pending_footnotes: List[nodes.footnote_reference] = []
        self.curfilestack: List[str] = []
        self.handled_abbrs: Set[str] = set()
```
### 7 - sphinx/writers/latex.py:

Start line: 526, End line: 590

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_document(self, node: Element) -> None:
        pass

    def visit_start_of_file(self, node: Element) -> None:
        self.curfilestack.append(node['docname'])

    def depart_start_of_file(self, node: Element) -> None:
        self.curfilestack.pop()

    def visit_section(self, node: Element) -> None:
        if not self.this_is_the_title:
            self.sectionlevel += 1
        self.body.append(BLANKLINE)

    def depart_section(self, node: Element) -> None:
        self.sectionlevel = max(self.sectionlevel - 1,
                                self.top_sectionlevel - 1)

    def visit_problematic(self, node: Element) -> None:
        self.body.append(r'{\color{red}\bfseries{}')

    def depart_problematic(self, node: Element) -> None:
        self.body.append('}')

    def visit_topic(self, node: Element) -> None:
        self.in_minipage = 1
        self.body.append(CR + r'\begin{sphinxShadowBox}' + CR)

    def depart_topic(self, node: Element) -> None:
        self.in_minipage = 0
        self.body.append(r'\end{sphinxShadowBox}' + CR)
    visit_sidebar = visit_topic
    depart_sidebar = depart_topic

    def visit_glossary(self, node: Element) -> None:
        pass

    def depart_glossary(self, node: Element) -> None:
        pass

    def visit_productionlist(self, node: Element) -> None:
        self.body.append(BLANKLINE)
        self.body.append(r'\begin{productionlist}' + CR)
        self.in_production_list = 1

    def depart_productionlist(self, node: Element) -> None:
        self.body.append(r'\end{productionlist}' + BLANKLINE)
        self.in_production_list = 0

    def visit_production(self, node: Element) -> None:
        if node['tokenname']:
            tn = node['tokenname']
            self.body.append(self.hypertarget('grammar-token-' + tn))
            self.body.append(r'\production{%s}{' % self.encode(tn))
        else:
            self.body.append(r'\productioncont{')

    def depart_production(self, node: Element) -> None:
        self.body.append('}' + CR)

    def visit_transition(self, node: Element) -> None:
        self.body.append(self.elements['transition'])

    def depart_transition(self, node: Element) -> None:
        pass
```
### 8 - sphinx/writers/latex.py:

Start line: 640, End line: 741

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_title(self, node: Element) -> None:
        self.in_title = 0
        if isinstance(node.parent, nodes.table):
            self.table.caption = self.popbody()
        else:
            self.body.append(self.context.pop())

    def visit_subtitle(self, node: Element) -> None:
        if isinstance(node.parent, nodes.sidebar):
            self.body.append(r'\sphinxstylesidebarsubtitle{')
            self.context.append('}' + CR)
        else:
            self.context.append('')

    def depart_subtitle(self, node: Element) -> None:
        self.body.append(self.context.pop())

    #############################################################
    # Domain-specific object descriptions
    #############################################################

    # Top-level nodes for descriptions
    ##################################

    def visit_desc(self, node: Element) -> None:
        if self.config.latex_show_urls == 'footnote':
            self.body.append(BLANKLINE)
            self.body.append(r'\begin{savenotes}\begin{fulllineitems}' + CR)
        else:
            self.body.append(BLANKLINE)
            self.body.append(r'\begin{fulllineitems}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_desc(self, node: Element) -> None:
        if self.config.latex_show_urls == 'footnote':
            self.body.append(CR + r'\end{fulllineitems}\end{savenotes}' + BLANKLINE)
        else:
            self.body.append(CR + r'\end{fulllineitems}' + BLANKLINE)

    def _visit_signature_line(self, node: Element) -> None:
        for child in node:
            if isinstance(child, addnodes.desc_parameterlist):
                self.body.append(r'\pysiglinewithargsret{')
                break
        else:
            self.body.append(r'\pysigline{')

    def _depart_signature_line(self, node: Element) -> None:
        self.body.append('}')

    def visit_desc_signature(self, node: Element) -> None:
        if node.parent['objtype'] != 'describe' and node['ids']:
            hyper = self.hypertarget(node['ids'][0])
        else:
            hyper = ''
        self.body.append(hyper)
        if not node.get('is_multiline'):
            self._visit_signature_line(node)
        else:
            self.body.append('%' + CR)
            self.body.append(r'\pysigstartmultiline' + CR)

    def depart_desc_signature(self, node: Element) -> None:
        if not node.get('is_multiline'):
            self._depart_signature_line(node)
        else:
            self.body.append('%' + CR)
            self.body.append(r'\pysigstopmultiline')

    def visit_desc_signature_line(self, node: Element) -> None:
        self._visit_signature_line(node)

    def depart_desc_signature_line(self, node: Element) -> None:
        self._depart_signature_line(node)

    def visit_desc_content(self, node: Element) -> None:
        pass

    def depart_desc_content(self, node: Element) -> None:
        pass

    def visit_desc_inline(self, node: Element) -> None:
        self.body.append(r'\sphinxcode{\sphinxupquote{')

    def depart_desc_inline(self, node: Element) -> None:
        self.body.append('}}')

    # Nodes for high-level structure in signatures
    ##############################################

    def visit_desc_name(self, node: Element) -> None:
        self.body.append(r'\sphinxbfcode{\sphinxupquote{')
        self.literal_whitespace += 1

    def depart_desc_name(self, node: Element) -> None:
        self.body.append('}}')
        self.literal_whitespace -= 1

    def visit_desc_addname(self, node: Element) -> None:
        self.body.append(r'\sphinxcode{\sphinxupquote{')
        self.literal_whitespace += 1
```
### 9 - sphinx/writers/latex.py:

Start line: 743, End line: 811

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_desc_addname(self, node: Element) -> None:
        self.body.append('}}')
        self.literal_whitespace -= 1

    def visit_desc_type(self, node: Element) -> None:
        pass

    def depart_desc_type(self, node: Element) -> None:
        pass

    def visit_desc_returns(self, node: Element) -> None:
        self.body.append(r'{ $\rightarrow$ ')

    def depart_desc_returns(self, node: Element) -> None:
        self.body.append(r'}')

    def visit_desc_parameterlist(self, node: Element) -> None:
        # close name, open parameterlist
        self.body.append('}{')
        self.first_param = 1

    def depart_desc_parameterlist(self, node: Element) -> None:
        # close parameterlist, open return annotation
        self.body.append('}{')

    def visit_desc_parameter(self, node: Element) -> None:
        if not self.first_param:
            self.body.append(', ')
        else:
            self.first_param = 0
        if not node.hasattr('noemph'):
            self.body.append(r'\emph{')

    def depart_desc_parameter(self, node: Element) -> None:
        if not node.hasattr('noemph'):
            self.body.append('}')

    def visit_desc_optional(self, node: Element) -> None:
        self.body.append(r'\sphinxoptional{')

    def depart_desc_optional(self, node: Element) -> None:
        self.body.append('}')

    def visit_desc_annotation(self, node: Element) -> None:
        self.body.append(r'\sphinxbfcode{\sphinxupquote{')

    def depart_desc_annotation(self, node: Element) -> None:
        self.body.append('}}')

    ##############################################

    def visit_seealso(self, node: Element) -> None:
        self.body.append(BLANKLINE)
        self.body.append(r'\sphinxstrong{%s:}' % admonitionlabels['seealso'] + CR)
        self.body.append(r'\nopagebreak' + BLANKLINE)

    def depart_seealso(self, node: Element) -> None:
        self.body.append(BLANKLINE)

    def visit_rubric(self, node: Element) -> None:
        if len(node) == 1 and node.astext() in ('Footnotes', _('Footnotes')):
            raise nodes.SkipNode
        self.body.append(r'\subsubsection*{')
        self.context.append('}' + CR)
        self.in_title = 1

    def depart_rubric(self, node: Element) -> None:
        self.in_title = 0
        self.body.append(self.context.pop())
```
### 10 - sphinx/writers/latex.py:

Start line: 403, End line: 453

```python
class LaTeXTranslator(SphinxTranslator):

    def pushbody(self, newbody: List[str]) -> None:
        self.bodystack.append(self.body)
        self.body = newbody

    def popbody(self) -> List[str]:
        body = self.body
        self.body = self.bodystack.pop()
        return body

    def astext(self) -> str:
        self.elements.update({
            'body': ''.join(self.body),
            'indices': self.generate_indices()
        })
        return self.render('latex.tex_t', self.elements)

    def hypertarget(self, id: str, withdoc: bool = True, anchor: bool = True) -> str:
        if withdoc:
            id = self.curfilestack[-1] + ':' + id
        return (r'\phantomsection' if anchor else '') + r'\label{%s}' % self.idescape(id)

    def hypertarget_to(self, node: Element, anchor: bool = False) -> str:
        labels = ''.join(self.hypertarget(node_id, anchor=False) for node_id in node['ids'])
        if anchor:
            return r'\phantomsection' + labels
        else:
            return labels

    def hyperlink(self, id: str) -> str:
        return r'{\hyperref[%s]{' % self.idescape(id)

    def hyperpageref(self, id: str) -> str:
        return r'\autopageref*{%s}' % self.idescape(id)

    def escape(self, s: str) -> str:
        return texescape.escape(s, self.config.latex_engine)

    def idescape(self, id: str) -> str:
        return r'\detokenize{%s}' % str(id).translate(tex_replace_map).\
            encode('ascii', 'backslashreplace').decode('ascii').\
            replace('\\', '_')

    def babel_renewcommand(self, command: str, definition: str) -> str:
        if self.elements['multilingual']:
            prefix = r'\addto\captions%s{' % self.babel.get_language()
            suffix = '}'
        else:  # babel is disabled (mainly for Japanese environment)
            prefix = ''
            suffix = ''

        return r'%s\renewcommand{%s}{%s}%s' % (prefix, command, definition, suffix) + CR
```
### 11 - sphinx/writers/latex.py:

Start line: 1603, End line: 1662

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_download_reference(self, node: Element) -> None:
        pass

    def depart_download_reference(self, node: Element) -> None:
        pass

    def visit_pending_xref(self, node: Element) -> None:
        pass

    def depart_pending_xref(self, node: Element) -> None:
        pass

    def visit_emphasis(self, node: Element) -> None:
        self.body.append(r'\sphinxstyleemphasis{')

    def depart_emphasis(self, node: Element) -> None:
        self.body.append('}')

    def visit_literal_emphasis(self, node: Element) -> None:
        self.body.append(r'\sphinxstyleliteralemphasis{\sphinxupquote{')

    def depart_literal_emphasis(self, node: Element) -> None:
        self.body.append('}}')

    def visit_strong(self, node: Element) -> None:
        self.body.append(r'\sphinxstylestrong{')

    def depart_strong(self, node: Element) -> None:
        self.body.append('}')

    def visit_literal_strong(self, node: Element) -> None:
        self.body.append(r'\sphinxstyleliteralstrong{\sphinxupquote{')

    def depart_literal_strong(self, node: Element) -> None:
        self.body.append('}}')

    def visit_abbreviation(self, node: Element) -> None:
        abbr = node.astext()
        self.body.append(r'\sphinxstyleabbreviation{')
        # spell out the explanation once
        if node.hasattr('explanation') and abbr not in self.handled_abbrs:
            self.context.append('} (%s)' % self.encode(node['explanation']))
            self.handled_abbrs.add(abbr)
        else:
            self.context.append('}')

    def depart_abbreviation(self, node: Element) -> None:
        self.body.append(self.context.pop())

    def visit_manpage(self, node: Element) -> None:
        return self.visit_literal_emphasis(node)

    def depart_manpage(self, node: Element) -> None:
        return self.depart_literal_emphasis(node)

    def visit_title_reference(self, node: Element) -> None:
        self.body.append(r'\sphinxtitleref{')

    def depart_title_reference(self, node: Element) -> None:
        self.body.append('}')
```
### 13 - sphinx/writers/latex.py:

Start line: 1513, End line: 1576

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_raw(self, node: Element) -> None:
        if not self.is_inline(node):
            self.body.append(CR)
        if 'latex' in node.get('format', '').split():
            self.body.append(node.astext())
        if not self.is_inline(node):
            self.body.append(CR)
        raise nodes.SkipNode

    def visit_reference(self, node: Element) -> None:
        if not self.in_title:
            for id in node.get('ids'):
                anchor = not self.in_caption
                self.body += self.hypertarget(id, anchor=anchor)
        if not self.is_inline(node):
            self.body.append(CR)
        uri = node.get('refuri', '')
        if not uri and node.get('refid'):
            uri = '%' + self.curfilestack[-1] + '#' + node['refid']
        if self.in_title or not uri:
            self.context.append('')
        elif uri.startswith('#'):
            # references to labels in the same document
            id = self.curfilestack[-1] + ':' + uri[1:]
            self.body.append(self.hyperlink(id))
            self.body.append(r'\emph{')
            if self.config.latex_show_pagerefs and not \
                    self.in_production_list:
                self.context.append('}}} (%s)' % self.hyperpageref(id))
            else:
                self.context.append('}}}')
        elif uri.startswith('%'):
            # references to documents or labels inside documents
            hashindex = uri.find('#')
            if hashindex == -1:
                # reference to the document
                id = uri[1:] + '::doc'
            else:
                # reference to a label
                id = uri[1:].replace('#', ':')
            self.body.append(self.hyperlink(id))
            if (len(node) and
                    isinstance(node[0], nodes.Element) and
                    'std-term' in node[0].get('classes', [])):
                # don't add a pageref for glossary terms
                self.context.append('}}}')
                # mark up as termreference
                self.body.append(r'\sphinxtermref{')
            else:
                self.body.append(r'\sphinxcrossref{')
                if self.config.latex_show_pagerefs and not self.in_production_list:
                    self.context.append('}}} (%s)' % self.hyperpageref(id))
                else:
                    self.context.append('}}}')
        else:
            if len(node) == 1 and uri == node[0]:
                if node.get('nolinkurl'):
                    self.body.append(r'\sphinxnolinkurl{%s}' % self.encode_uri(uri))
                else:
                    self.body.append(r'\sphinxurl{%s}' % self.encode_uri(uri))
                raise nodes.SkipNode
            else:
                self.body.append(r'\sphinxhref{%s}{' % self.encode_uri(uri))
                self.context.append('}')
```
### 16 - sphinx/writers/latex.py:

Start line: 1447, End line: 1511

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_index(self, node: Element) -> None:
        def escape(value: str) -> str:
            value = self.encode(value)
            value = value.replace(r'\{', r'\sphinxleftcurlybrace{}')
            value = value.replace(r'\}', r'\sphinxrightcurlybrace{}')
            value = value.replace('"', '""')
            value = value.replace('@', '"@')
            value = value.replace('!', '"!')
            value = value.replace('|', r'\textbar{}')
            return value

        def style(string: str) -> str:
            match = EXTRA_RE.match(string)
            if match:
                return match.expand(r'\\spxentry{\1}\\spxextra{\2}')
            else:
                return r'\spxentry{%s}' % string

        if not node.get('inline', True):
            self.body.append(CR)
        entries = node['entries']
        for type, string, tid, ismain, key_ in entries:
            m = ''
            if ismain:
                m = '|spxpagem'
            try:
                if type == 'single':
                    try:
                        p1, p2 = [escape(x) for x in split_into(2, 'single', string)]
                        P1, P2 = style(p1), style(p2)
                        self.body.append(r'\index{%s@%s!%s@%s%s}' % (p1, P1, p2, P2, m))
                    except ValueError:
                        p = escape(split_into(1, 'single', string)[0])
                        P = style(p)
                        self.body.append(r'\index{%s@%s%s}' % (p, P, m))
                elif type == 'pair':
                    p1, p2 = [escape(x) for x in split_into(2, 'pair', string)]
                    P1, P2 = style(p1), style(p2)
                    self.body.append(r'\index{%s@%s!%s@%s%s}\index{%s@%s!%s@%s%s}' %
                                     (p1, P1, p2, P2, m, p2, P2, p1, P1, m))
                elif type == 'triple':
                    p1, p2, p3 = [escape(x) for x in split_into(3, 'triple', string)]
                    P1, P2, P3 = style(p1), style(p2), style(p3)
                    self.body.append(
                        r'\index{%s@%s!%s %s@%s %s%s}'
                        r'\index{%s@%s!%s, %s@%s, %s%s}'
                        r'\index{%s@%s!%s %s@%s %s%s}' %
                        (p1, P1, p2, p3, P2, P3, m,
                         p2, P2, p3, p1, P3, P1, m,
                         p3, P3, p1, p2, P1, P2, m))
                elif type == 'see':
                    p1, p2 = [escape(x) for x in split_into(2, 'see', string)]
                    P1 = style(p1)
                    self.body.append(r'\index{%s@%s|see{%s}}' % (p1, P1, p2))
                elif type == 'seealso':
                    p1, p2 = [escape(x) for x in split_into(2, 'seealso', string)]
                    P1 = style(p1)
                    self.body.append(r'\index{%s@%s|see{%s}}' % (p1, P1, p2))
                else:
                    logger.warning(__('unknown index entry type %s found'), type)
            except ValueError as err:
                logger.warning(str(err))
        if not node.get('inline', True):
            self.body.append(r'\ignorespaces ')
        raise nodes.SkipNode
```
### 17 - sphinx/writers/latex.py:

Start line: 1675, End line: 1733

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_thebibliography(self, node: Element) -> None:
        self.body.append(r'\end{sphinxthebibliography}' + CR)

    def visit_citation(self, node: Element) -> None:
        label = cast(nodes.label, node[0])
        self.body.append(r'\bibitem[%s]{%s:%s}' % (self.encode(label.astext()),
                                                   node['docname'], node['ids'][0]))

    def depart_citation(self, node: Element) -> None:
        pass

    def visit_citation_reference(self, node: Element) -> None:
        if self.in_title:
            pass
        else:
            self.body.append(r'\sphinxcite{%s:%s}' % (node['docname'], node['refname']))
            raise nodes.SkipNode

    def depart_citation_reference(self, node: Element) -> None:
        pass

    def visit_literal(self, node: Element) -> None:
        if self.in_title:
            self.body.append(r'\sphinxstyleliteralintitle{\sphinxupquote{')
        elif 'kbd' in node['classes']:
            self.body.append(r'\sphinxkeyboard{\sphinxupquote{')
        else:
            self.body.append(r'\sphinxcode{\sphinxupquote{')

    def depart_literal(self, node: Element) -> None:
        self.body.append('}}')

    def visit_footnote_reference(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_footnotemark(self, node: Element) -> None:
        self.body.append(r'\sphinxfootnotemark[')

    def depart_footnotemark(self, node: Element) -> None:
        self.body.append(']')

    def visit_footnotetext(self, node: Element) -> None:
        label = cast(nodes.label, node[0])
        self.body.append('%' + CR)
        self.body.append(r'\begin{footnotetext}[%s]'
                         r'\phantomsection\label{\thesphinxscope.%s}%%'
                         % (label.astext(), label.astext()) + CR)
        self.body.append(r'\sphinxAtStartFootnote' + CR)

    def depart_footnotetext(self, node: Element) -> None:
        # the \ignorespaces in particular for after table header use
        self.body.append('%' + CR)
        self.body.append(r'\end{footnotetext}\ignorespaces ')

    def visit_captioned_literal_block(self, node: Element) -> None:
        pass

    def depart_captioned_literal_block(self, node: Element) -> None:
        pass
```
### 18 - sphinx/writers/latex.py:

Start line: 1825, End line: 1897

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_block_quote(self, node: Element) -> None:
        done = 0
        if len(node.children) == 1:
            child = node.children[0]
            if isinstance(child, nodes.bullet_list) or \
                    isinstance(child, nodes.enumerated_list):
                done = 1
        if not done:
            self.body.append(r'\end{quote}' + CR)

    # option node handling copied from docutils' latex writer

    def visit_option(self, node: Element) -> None:
        if self.context[-1]:
            # this is not the first option
            self.body.append(', ')

    def depart_option(self, node: Element) -> None:
        # flag that the first option is done.
        self.context[-1] += 1

    def visit_option_argument(self, node: Element) -> None:
        """The delimiter between an option and its argument."""
        self.body.append(node.get('delimiter', ' '))

    def depart_option_argument(self, node: Element) -> None:
        pass

    def visit_option_group(self, node: Element) -> None:
        self.body.append(r'\item [')
        # flag for first option
        self.context.append(0)

    def depart_option_group(self, node: Element) -> None:
        self.context.pop()  # the flag
        self.body.append('] ')

    def visit_option_list(self, node: Element) -> None:
        self.body.append(r'\begin{optionlist}{3cm}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_option_list(self, node: Element) -> None:
        self.body.append(r'\end{optionlist}' + CR)

    def visit_option_list_item(self, node: Element) -> None:
        pass

    def depart_option_list_item(self, node: Element) -> None:
        pass

    def visit_option_string(self, node: Element) -> None:
        ostring = node.astext()
        self.body.append(self.encode(ostring))
        raise nodes.SkipNode

    def visit_description(self, node: Element) -> None:
        self.body.append(' ')

    def depart_description(self, node: Element) -> None:
        pass

    def visit_superscript(self, node: Element) -> None:
        self.body.append(r'$^{\text{')

    def depart_superscript(self, node: Element) -> None:
        self.body.append('}}$')

    def visit_subscript(self, node: Element) -> None:
        self.body.append(r'$_{\text{')

    def depart_subscript(self, node: Element) -> None:
        self.body.append('}}$')
```
### 29 - sphinx/writers/latex.py:

Start line: 455, End line: 492

```python
class LaTeXTranslator(SphinxTranslator):

    def generate_indices(self) -> str:
        def generate(content: List[Tuple[str, List[IndexEntry]]], collapsed: bool) -> None:
            ret.append(r'\begin{sphinxtheindex}' + CR)
            ret.append(r'\let\bigletter\sphinxstyleindexlettergroup' + CR)
            for i, (letter, entries) in enumerate(content):
                if i > 0:
                    ret.append(r'\indexspace' + CR)
                ret.append(r'\bigletter{%s}' % self.escape(letter) + CR)
                for entry in entries:
                    if not entry[3]:
                        continue
                    ret.append(r'\item\relax\sphinxstyleindexentry{%s}' %
                               self.encode(entry[0]))
                    if entry[4]:
                        # add "extra" info
                        ret.append(r'\sphinxstyleindexextra{%s}' % self.encode(entry[4]))
                    ret.append(r'\sphinxstyleindexpageref{%s:%s}' %
                               (entry[2], self.idescape(entry[3])) + CR)
            ret.append(r'\end{sphinxtheindex}' + CR)

        ret = []
        # latex_domain_indices can be False/True or a list of index names
        indices_config = self.config.latex_domain_indices
        if indices_config:
            for domain in self.builder.env.domains.values():
                for indexcls in domain.indices:
                    indexname = '%s-%s' % (domain.name, indexcls.name)
                    if isinstance(indices_config, list):
                        if indexname not in indices_config:
                            continue
                    content, collapsed = indexcls(domain).generate(
                        self.builder.docnames)
                    if not content:
                        continue
                    ret.append(r'\renewcommand{\indexname}{%s}' % indexcls.localname + CR)
                    generate(content, collapsed)

        return ''.join(ret)
```
### 32 - sphinx/writers/latex.py:

Start line: 1735, End line: 1784

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_literal_block(self, node: Element) -> None:
        if node.rawsource != node.astext():
            # most probably a parsed-literal block -- don't highlight
            self.in_parsed_literal += 1
            self.body.append(r'\begin{sphinxalltt}' + CR)
        else:
            labels = self.hypertarget_to(node)
            if isinstance(node.parent, captioned_literal_block):
                labels += self.hypertarget_to(node.parent)
            if labels and not self.in_footnote:
                self.body.append(CR + r'\def\sphinxLiteralBlockLabel{' + labels + '}')

            lang = node.get('language', 'default')
            linenos = node.get('linenos', False)
            highlight_args = node.get('highlight_args', {})
            highlight_args['force'] = node.get('force', False)
            opts = self.config.highlight_options.get(lang, {})

            hlcode = self.highlighter.highlight_block(
                node.rawsource, lang, opts=opts, linenos=linenos,
                location=node, **highlight_args
            )
            if self.in_footnote:
                self.body.append(CR + r'\sphinxSetupCodeBlockInFootnote')
                hlcode = hlcode.replace(r'\begin{Verbatim}',
                                        r'\begin{sphinxVerbatim}')
            # if in table raise verbatim flag to avoid "tabulary" environment
            # and opt for sphinxVerbatimintable to handle caption & long lines
            elif self.table:
                self.table.has_problematic = True
                self.table.has_verbatim = True
                hlcode = hlcode.replace(r'\begin{Verbatim}',
                                        r'\begin{sphinxVerbatimintable}')
            else:
                hlcode = hlcode.replace(r'\begin{Verbatim}',
                                        r'\begin{sphinxVerbatim}')
            # get consistent trailer
            hlcode = hlcode.rstrip()[:-14]  # strip \end{Verbatim}
            if self.table and not self.in_footnote:
                hlcode += r'\end{sphinxVerbatimintable}'
            else:
                hlcode += r'\end{sphinxVerbatim}'

            hllines = str(highlight_args.get('hl_lines', []))[1:-1]
            if hllines:
                self.body.append(CR + r'\fvset{hllines={, %s,}}%%' % hllines)
            self.body.append(CR + hlcode + CR)
            if hllines:
                self.body.append(r'\sphinxresetverbatimhllines' + CR)
            raise nodes.SkipNode
```
### 36 - sphinx/writers/latex.py:

Start line: 866, End line: 905

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_table(self, node: Element) -> None:
        labels = self.hypertarget_to(node)
        table_type = self.table.get_table_type()
        table = self.render(table_type + '.tex_t',
                            dict(table=self.table, labels=labels))
        self.body.append(BLANKLINE)
        self.body.append(table)
        self.body.append(CR)

        self.tables.pop()

    def visit_colspec(self, node: Element) -> None:
        self.table.colcount += 1
        if 'colwidth' in node:
            self.table.colwidths.append(node['colwidth'])
        if 'stub' in node:
            self.table.stubs.append(self.table.colcount - 1)

    def depart_colspec(self, node: Element) -> None:
        pass

    def visit_tgroup(self, node: Element) -> None:
        pass

    def depart_tgroup(self, node: Element) -> None:
        pass

    def visit_thead(self, node: Element) -> None:
        # Redirect head output until header is finished.
        self.pushbody(self.table.header)

    def depart_thead(self, node: Element) -> None:
        self.popbody()

    def visit_tbody(self, node: Element) -> None:
        # Redirect body output until table is finished.
        self.pushbody(self.table.body)

    def depart_tbody(self, node: Element) -> None:
        self.popbody()
```
### 37 - sphinx/writers/latex.py:

Start line: 1032, End line: 1061

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_enumerated_list(self, node: Element) -> None:
        def get_enumtype(node: Element) -> str:
            enumtype = node.get('enumtype', 'arabic')
            if 'alpha' in enumtype and 26 < node.get('start', 0) + len(node):
                # fallback to arabic if alphabet counter overflows
                enumtype = 'arabic'

            return enumtype

        def get_nested_level(node: Element) -> int:
            if node is None:
                return 0
            elif isinstance(node, nodes.enumerated_list):
                return get_nested_level(node.parent) + 1
            else:
                return get_nested_level(node.parent)

        enum = "enum%s" % toRoman(get_nested_level(node)).lower()
        enumnext = "enum%s" % toRoman(get_nested_level(node) + 1).lower()
        style = ENUMERATE_LIST_STYLE.get(get_enumtype(node))
        prefix = node.get('prefix', '')
        suffix = node.get('suffix', '.')

        self.body.append(r'\begin{enumerate}' + CR)
        self.body.append(r'\sphinxsetlistlabels{%s}{%s}{%s}{%s}{%s}%%' %
                         (style, enum, enumnext, prefix, suffix) + CR)
        if 'start' in node:
            self.body.append(r'\setcounter{%s}{%d}' % (enum, node['start'] - 1) + CR)
        if self.table:
            self.table.has_problematic = True
```
### 38 - sphinx/writers/latex.py:

Start line: 1012, End line: 1030

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_acks(self, node: Element) -> None:
        # this is a list in the source, but should be rendered as a
        # comma-separated list here
        bullet_list = cast(nodes.bullet_list, node[0])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        self.body.append(BLANKLINE)
        self.body.append(', '.join(n.astext() for n in list_items) + '.')
        self.body.append(BLANKLINE)
        raise nodes.SkipNode

    def visit_bullet_list(self, node: Element) -> None:
        if not self.compact_list:
            self.body.append(r'\begin{itemize}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_bullet_list(self, node: Element) -> None:
        if not self.compact_list:
            self.body.append(r'\end{itemize}' + CR)
```
### 43 - sphinx/writers/latex.py:

Start line: 1977, End line: 2002

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_Text(self, node: Text) -> None:
        text = self.encode(node.astext())
        self.body.append(text)

    def depart_Text(self, node: Text) -> None:
        pass

    def visit_comment(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_meta(self, node: Element) -> None:
        # only valid for HTML
        raise nodes.SkipNode

    def visit_system_message(self, node: Element) -> None:
        pass

    def depart_system_message(self, node: Element) -> None:
        self.body.append(CR)

    def visit_math(self, node: Element) -> None:
        if self.in_title:
            self.body.append(r'\protect\(%s\protect\)' % node.astext())
        else:
            self.body.append(r'\(%s\)' % node.astext())
        raise nodes.SkipNode
```
### 46 - sphinx/writers/latex.py:

Start line: 73, End line: 94

```python
class LaTeXWriter(writers.Writer):

    supported = ('sphinxlatex',)

    settings_spec = ('LaTeX writer options', '', (
        ('Document name', ['--docname'], {'default': ''}),
        ('Document class', ['--docclass'], {'default': 'manual'}),
        ('Author', ['--author'], {'default': ''}),
    ))
    settings_defaults: Dict = {}

    output = None

    def __init__(self, builder: "LaTeXBuilder") -> None:
        super().__init__()
        self.builder = builder
        self.theme: Theme = None

    def translate(self) -> None:
        visitor = self.builder.create_translator(self.document, self.builder, self.theme)
        self.document.walkabout(visitor)
        self.output = cast(LaTeXTranslator, visitor).astext()
```
### 47 - sphinx/writers/latex.py:

Start line: 1391, End line: 1445

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_target(self, node: Element) -> None:
        def add_target(id: str) -> None:
            # indexing uses standard LaTeX index markup, so the targets
            # will be generated differently
            if id.startswith('index-'):
                return

            # equations also need no extra blank line nor hypertarget
            # TODO: fix this dependency on mathbase extension internals
            if id.startswith('equation-'):
                return

            # insert blank line, if the target follows a paragraph node
            index = node.parent.index(node)
            if index > 0 and isinstance(node.parent[index - 1], nodes.paragraph):
                self.body.append(CR)

            # do not generate \phantomsection in \section{}
            anchor = not self.in_title
            self.body.append(self.hypertarget(id, anchor=anchor))

        # skip if visitor for next node supports hyperlink
        next_node: Node = node
        while isinstance(next_node, nodes.target):
            next_node = next_node.next_node(ascend=True)

        domain = cast(StandardDomain, self.builder.env.get_domain('std'))
        if isinstance(next_node, HYPERLINK_SUPPORT_NODES):
            return
        elif domain.get_enumerable_node_type(next_node) and domain.get_numfig_title(next_node):
            return

        if 'refuri' in node:
            return
        if 'anonymous' in node:
            return
        if node.get('refid'):
            prev_node = get_prev_node(node)
            if isinstance(prev_node, nodes.reference) and node['refid'] == prev_node['refid']:
                # a target for a hyperlink reference having alias
                pass
            else:
                add_target(node['refid'])
        for id in node['ids']:
            add_target(id)

    def depart_target(self, node: Element) -> None:
        pass

    def visit_attribution(self, node: Element) -> None:
        self.body.append(CR + r'\begin{flushright}' + CR)
        self.body.append('---')

    def depart_attribution(self, node: Element) -> None:
        self.body.append(CR + r'\end{flushright}' + CR)
```
### 49 - sphinx/writers/latex.py:

Start line: 1578, End line: 1601

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_reference(self, node: Element) -> None:
        self.body.append(self.context.pop())
        if not self.is_inline(node):
            self.body.append(CR)

    def visit_number_reference(self, node: Element) -> None:
        if node.get('refid'):
            id = self.curfilestack[-1] + ':' + node['refid']
        else:
            id = node.get('refuri', '')[1:].replace('#', ':')

        title = self.escape(node.get('title', '%s')).replace(r'\%s', '%s')
        if r'\{name\}' in title or r'\{number\}' in title:
            # new style format (cf. "Fig.%{number}")
            title = title.replace(r'\{name\}', '{name}').replace(r'\{number\}', '{number}')
            text = escape_abbr(title).format(name=r'\nameref{%s}' % self.idescape(id),
                                             number=r'\ref{%s}' % self.idescape(id))
        else:
            # old style format (cf. "Fig.%{number}")
            text = escape_abbr(title) % (r'\ref{%s}' % self.idescape(id))
        hyperref = r'\hyperref[%s]{%s}' % (self.idescape(id), text)
        self.body.append(hyperref)

        raise nodes.SkipNode
```
### 54 - sphinx/writers/latex.py:

Start line: 1279, End line: 1319

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_image(self, node: Element) -> None:
        pass

    def visit_figure(self, node: Element) -> None:
        align = self.elements['figure_align']
        if self.no_latex_floats:
            align = "H"
        if self.table:
            # TODO: support align option
            if 'width' in node:
                length = self.latex_image_length(node['width'])
                if length:
                    self.body.append(r'\begin{sphinxfigure-in-table}[%s]' % length + CR)
                    self.body.append(r'\centering' + CR)
            else:
                self.body.append(r'\begin{sphinxfigure-in-table}' + CR)
                self.body.append(r'\centering' + CR)
            if any(isinstance(child, nodes.caption) for child in node):
                self.body.append(r'\capstart')
            self.context.append(r'\end{sphinxfigure-in-table}\relax' + CR)
        elif node.get('align', '') in ('left', 'right'):
            length = None
            if 'width' in node:
                length = self.latex_image_length(node['width'])
            elif isinstance(node[0], nodes.image) and 'width' in node[0]:
                length = self.latex_image_length(node[0]['width'])
            self.body.append(BLANKLINE)     # Insert a blank line to prevent infinite loop
                                            # https://github.com/sphinx-doc/sphinx/issues/7059
            self.body.append(r'\begin{wrapfigure}{%s}{%s}' %
                             ('r' if node['align'] == 'right' else 'l', length or '0pt') + CR)
            self.body.append(r'\centering')
            self.context.append(r'\end{wrapfigure}' + CR)
        elif self.in_minipage:
            self.body.append(CR + r'\begin{center}')
            self.context.append(r'\end{center}' + CR)
        else:
            self.body.append(CR + r'\begin{figure}[%s]' % align + CR)
            self.body.append(r'\centering' + CR)
            if any(isinstance(child, nodes.caption) for child in node):
                self.body.append(r'\capstart' + CR)
            self.context.append(r'\end{figure}' + CR)
```
### 55 - sphinx/writers/latex.py:

Start line: 1916, End line: 1966

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_inline(self, node: Element) -> None:
        self.body.append(self.context.pop())

    def visit_generated(self, node: Element) -> None:
        pass

    def depart_generated(self, node: Element) -> None:
        pass

    def visit_compound(self, node: Element) -> None:
        pass

    def depart_compound(self, node: Element) -> None:
        pass

    def visit_container(self, node: Element) -> None:
        classes = node.get('classes', [])
        for c in classes:
            self.body.append('\n\\begin{sphinxuseclass}{%s}' % c)

    def depart_container(self, node: Element) -> None:
        classes = node.get('classes', [])
        for c in classes:
            self.body.append('\n\\end{sphinxuseclass}')

    def visit_decoration(self, node: Element) -> None:
        pass

    def depart_decoration(self, node: Element) -> None:
        pass

    # docutils-generated elements that we don't support

    def visit_header(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_footer(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_docinfo(self, node: Element) -> None:
        raise nodes.SkipNode

    # text handling

    def encode(self, text: str) -> str:
        text = self.escape(text)
        if self.literal_whitespace:
            # Insert a blank before the newline, to avoid
            # ! LaTeX Error: There's no line here to end.
            text = text.replace(CR, r'~\\' + CR).replace(' ', '~')
        return text
```
### 57 - sphinx/writers/latex.py:

Start line: 943, End line: 978

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_entry(self, node: Element) -> None:
        if self.table.col > 0:
            self.body.append('&')
        self.table.add_cell(node.get('morerows', 0) + 1, node.get('morecols', 0) + 1)
        cell = self.table.cell()
        context = ''
        if cell.width > 1:
            if self.config.latex_use_latex_multicolumn:
                if self.table.col == 0:
                    self.body.append(r'\multicolumn{%d}{|l|}{%%' % cell.width + CR)
                else:
                    self.body.append(r'\multicolumn{%d}{l|}{%%' % cell.width + CR)
                context = '}%' + CR
            else:
                self.body.append(r'\sphinxstartmulticolumn{%d}%%' % cell.width + CR)
                context = r'\sphinxstopmulticolumn' + CR
        if cell.height > 1:
            # \sphinxmultirow 2nd arg "cell_id" will serve as id for LaTeX macros as well
            self.body.append(r'\sphinxmultirow{%d}{%d}{%%' % (cell.height, cell.cell_id) + CR)
            context = '}%' + CR + context
        if cell.width > 1 or cell.height > 1:
            self.body.append(r'\begin{varwidth}[t]{\sphinxcolwidth{%d}{%d}}'
                             % (cell.width, self.table.colcount) + CR)
            context = (r'\par' + CR + r'\vskip-\baselineskip'
                       r'\vbox{\hbox{\strut}}\end{varwidth}%' + CR + context)
            self.needs_linetrimming = 1
        if len(list(node.traverse(nodes.paragraph))) >= 2:
            self.table.has_oldproblematic = True
        if isinstance(node.parent.parent, nodes.thead) or (cell.col in self.table.stubs):
            if len(node) == 1 and isinstance(node[0], nodes.paragraph) and node.astext() == '':
                pass
            else:
                self.body.append(r'\sphinxstyletheadfamily ')
        if self.needs_linetrimming:
            self.pushbody([])
        self.context.append(context)
```
### 60 - sphinx/writers/latex.py:

Start line: 1786, End line: 1808

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_literal_block(self, node: Element) -> None:
        self.body.append(CR + r'\end{sphinxalltt}' + CR)
        self.in_parsed_literal -= 1
    visit_doctest_block = visit_literal_block
    depart_doctest_block = depart_literal_block

    def visit_line(self, node: Element) -> None:
        self.body.append(r'\item[] ')

    def depart_line(self, node: Element) -> None:
        self.body.append(CR)

    def visit_line_block(self, node: Element) -> None:
        if isinstance(node.parent, nodes.line_block):
            self.body.append(r'\item[]' + CR)
            self.body.append(r'\begin{DUlineblock}{\DUlineblockindent}' + CR)
        else:
            self.body.append(CR + r'\begin{DUlineblock}{0em}' + CR)
        if self.table:
            self.table.has_problematic = True

    def depart_line_block(self, node: Element) -> None:
        self.body.append(r'\end{DUlineblock}' + CR)
```
### 61 - sphinx/writers/latex.py:

Start line: 14, End line: 70

```python
import re
import warnings
from collections import defaultdict
from os import path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Set, Tuple, cast

from docutils import nodes, writers
from docutils.nodes import Element, Node, Text

from sphinx import addnodes, highlighting
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.domains import IndexEntry
from sphinx.domains.std import StandardDomain
from sphinx.errors import SphinxError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging, split_into, texescape
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import clean_astext, get_prev_node
from sphinx.util.template import LaTeXRenderer
from sphinx.util.texescape import tex_replace_map

try:
    from docutils.utils.roman import toRoman
except ImportError:
    # In Debian/Ubuntu, roman package is provided as roman, not as docutils.utils.roman
    from roman import toRoman  # type: ignore

if TYPE_CHECKING:
    from sphinx.builders.latex import LaTeXBuilder
    from sphinx.builders.latex.theming import Theme


logger = logging.getLogger(__name__)

MAX_CITATION_LABEL_LENGTH = 8
LATEXSECTIONNAMES = ["part", "chapter", "section", "subsection",
                     "subsubsection", "paragraph", "subparagraph"]
ENUMERATE_LIST_STYLE = defaultdict(lambda: r'\arabic',
                                   {
                                       'arabic': r'\arabic',
                                       'loweralpha': r'\alph',
                                       'upperalpha': r'\Alph',
                                       'lowerroman': r'\roman',
                                       'upperroman': r'\Roman',
                                   })

CR = '\n'
BLANKLINE = '\n\n'
EXTRA_RE = re.compile(r'^(.*\S)\s+\(([^()]*)\)\s*$')


class collected_footnote(nodes.footnote):
    """Footnotes that are collected are assigned this class."""


class UnsupportedError(SphinxError):
    category = 'Markup is unsupported in LaTeX'
```
### 63 - sphinx/writers/latex.py:

Start line: 2020, End line: 2053

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_math_reference(self, node: Element) -> None:
        label = "equation:%s:%s" % (node['docname'], node['target'])
        eqref_format = self.config.math_eqref_format
        if eqref_format:
            try:
                ref = r'\ref{%s}' % label
                self.body.append(eqref_format.format(number=ref))
            except KeyError as exc:
                logger.warning(__('Invalid math_eqref_format: %r'), exc,
                               location=node)
                self.body.append(r'\eqref{%s}' % label)
        else:
            self.body.append(r'\eqref{%s}' % label)

    def depart_math_reference(self, node: Element) -> None:
        pass

    def unknown_visit(self, node: Node) -> None:
        raise NotImplementedError('Unknown node: ' + node.__class__.__name__)

    @property
    def docclasses(self) -> Tuple[str, str]:
        """Prepends prefix to sphinx document classes"""
        warnings.warn('LaTeXWriter.docclasses() is deprecated.',
                      RemovedInSphinx70Warning, stacklevel=2)
        return ('howto', 'manual')


# FIXME: Workaround to avoid circular import
# refs: https://github.com/sphinx-doc/sphinx/issues/5433
from sphinx.builders.latex.nodes import ( # NOQA isort:skip
    HYPERLINK_SUPPORT_NODES, captioned_literal_block, footnotetext,
)
```
### 66 - sphinx/writers/latex.py:

Start line: 592, End line: 638

```python
class LaTeXTranslator(SphinxTranslator):

    def visit_title(self, node: Element) -> None:
        parent = node.parent
        if isinstance(parent, addnodes.seealso):
            # the environment already handles this
            raise nodes.SkipNode
        elif isinstance(parent, nodes.section):
            if self.this_is_the_title:
                if len(node.children) != 1 and not isinstance(node.children[0],
                                                              nodes.Text):
                    logger.warning(__('document title is not a single Text node'),
                                   location=node)
                if not self.elements['title']:
                    # text needs to be escaped since it is inserted into
                    # the output literally
                    self.elements['title'] = self.escape(node.astext())
                self.this_is_the_title = 0
                raise nodes.SkipNode
            else:
                short = ''
                if list(node.traverse(nodes.image)):
                    short = ('[%s]' % self.escape(' '.join(clean_astext(node).split())))

                try:
                    self.body.append(r'\%s%s{' % (self.sectionnames[self.sectionlevel], short))
                except IndexError:
                    # just use "subparagraph", it's not numbered anyway
                    self.body.append(r'\%s%s{' % (self.sectionnames[-1], short))
                self.context.append('}' + CR + self.hypertarget_to(node.parent))
        elif isinstance(parent, nodes.topic):
            self.body.append(r'\sphinxstyletopictitle{')
            self.context.append('}' + CR)
        elif isinstance(parent, nodes.sidebar):
            self.body.append(r'\sphinxstylesidebartitle{')
            self.context.append('}' + CR)
        elif isinstance(parent, nodes.Admonition):
            self.body.append('{')
            self.context.append('}' + CR)
        elif isinstance(parent, nodes.table):
            # Redirect body output until title is finished.
            self.pushbody([])
        else:
            logger.warning(__('encountered title node not in section, topic, table, '
                              'admonition or sidebar'),
                           location=node)
            self.body.append(r'\sphinxstyleothertitle{')
            self.context.append('}' + CR)
        self.in_title = 1
```
