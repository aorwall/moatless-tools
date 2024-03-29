# sphinx-doc__sphinx-9464

| **sphinx-doc/sphinx** | `810a1e2988b14f4d139b5ef328a91967f5ed7a08` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1983 |
| **Any found context length** | 1983 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -310,7 +310,10 @@ def stringify(annotation: Any) -> str:
         return INVALID_BUILTIN_CLASSES[annotation]
     elif (getattr(annotation, '__module__', None) == 'builtins' and
           hasattr(annotation, '__qualname__')):
-        return annotation.__qualname__
+        if hasattr(annotation, '__args__'):  # PEP 585 generic
+            return repr(annotation)
+        else:
+            return annotation.__qualname__
     elif annotation is Ellipsis:
         return '...'
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/util/typing.py | 313 | 313 | 5 | 3 | 1983


## Problem Statement

```
PEP 585 type hints not rendered correctly
### Describe the bug

If you use a PEP 585 generic as an annotation i.e. `list[str]`, autodoc renders the annotation as `list` rather than `list[str]`, this behaviour differs from using `typing.List[str]` which renders as expected.

Fixing this is quite simple as far as I can tell, https://github.com/sphinx-doc/sphinx/blob/810a1e2988b14f4d139b5ef328a91967f5ed7a08/sphinx/util/typing.py#L311-L313 just needs to check if the annotation has `__args__` and if it does, return `repr(annotation)`

### How to Reproduce

\`\`\`py
def foo() -> list[str]:
	...
\`\`\`

\`\`\`rst
.. autofunction:: foo
\`\`\`

### Expected behavior

An annotation of `list[str]` to be rendered as `list[str]`

### Your project

https://github.com/Gobot1234/sphinx-test

### Screenshots

![image](https://user-images.githubusercontent.com/50501825/126038116-252eee01-228a-42bb-b6ab-23bdf72968e3.png)


### OS

Mac

### Python version

Python 3.9.3

### Sphinx version

4.1.1

### Sphinx extensions

autodoc

### Extra tools

_No response_

### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/typehints.py | 83 | 127| 389 | 389 | 1448 | 
| 2 | 1 sphinx/ext/autodoc/typehints.py | 130 | 185| 454 | 843 | 1448 | 
| 3 | 2 sphinx/ext/autodoc/__init__.py | 2534 | 2554| 231 | 1074 | 24922 | 
| 4 | **3 sphinx/util/typing.py** | 386 | 449| 638 | 1712 | 29238 | 
| **-> 5 <-** | **3 sphinx/util/typing.py** | 287 | 320| 271 | 1983 | 29238 | 
| 6 | 4 sphinx/domains/python.py | 110 | 186| 670 | 2653 | 41324 | 
| 7 | **4 sphinx/util/typing.py** | 323 | 383| 677 | 3330 | 41324 | 
| 8 | 5 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 3569 | 42548 | 
| 9 | 5 sphinx/ext/autodoc/typehints.py | 11 | 37| 210 | 3779 | 42548 | 
| 10 | 5 sphinx/ext/autodoc/__init__.py | 2623 | 2648| 316 | 4095 | 42548 | 
| 11 | 5 sphinx/domains/python.py | 11 | 80| 518 | 4613 | 42548 | 
| 12 | 6 sphinx/ext/napoleon/docstring.py | 13 | 67| 578 | 5191 | 53548 | 
| 13 | 7 sphinx/util/inspect.py | 11 | 47| 290 | 5481 | 60118 | 
| 14 | 7 sphinx/ext/autodoc/__init__.py | 1976 | 2000| 259 | 5740 | 60118 | 
| 15 | 7 sphinx/ext/autodoc/__init__.py | 2606 | 2621| 182 | 5922 | 60118 | 
| 16 | 7 sphinx/ext/autodoc/__init__.py | 2556 | 2579| 229 | 6151 | 60118 | 
| 17 | 7 sphinx/ext/autodoc/__init__.py | 2581 | 2604| 286 | 6437 | 60118 | 
| 18 | 7 sphinx/ext/autodoc/type_comment.py | 115 | 140| 257 | 6694 | 60118 | 
| 19 | 8 sphinx/highlighting.py | 11 | 68| 620 | 7314 | 61664 | 
| 20 | 8 sphinx/ext/autodoc/__init__.py | 2742 | 2785| 528 | 7842 | 61664 | 
| 21 | 9 doc/conf.py | 1 | 82| 731 | 8573 | 63128 | 
| 22 | 10 sphinx/pygments_styles.py | 38 | 96| 506 | 9079 | 63820 | 
| 23 | 11 doc/usage/extensions/example_google.py | 38 | 75| 245 | 9324 | 65930 | 
| 24 | 12 sphinx/roles.py | 174 | 206| 337 | 9661 | 69133 | 
| 25 | 13 doc/usage/extensions/example_numpy.py | 48 | 98| 276 | 9937 | 71241 | 
| 26 | 13 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 10725 | 71241 | 
| 27 | 13 sphinx/ext/autodoc/__init__.py | 1236 | 1252| 184 | 10909 | 71241 | 
| 28 | 14 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 11120 | 73703 | 
| 29 | 14 sphinx/ext/autodoc/typehints.py | 40 | 80| 333 | 11453 | 73703 | 
| 30 | 14 sphinx/ext/autodoc/__init__.py | 296 | 372| 751 | 12204 | 73703 | 
| 31 | 14 sphinx/ext/autodoc/__init__.py | 1273 | 1392| 995 | 13199 | 73703 | 
| 32 | **14 sphinx/util/typing.py** | 213 | 284| 794 | 13993 | 73703 | 
| 33 | 14 sphinx/ext/autodoc/__init__.py | 2002 | 2039| 320 | 14313 | 73703 | 
| 34 | 15 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 14453 | 74663 | 
| 35 | 15 sphinx/ext/autodoc/__init__.py | 1927 | 1974| 365 | 14818 | 74663 | 
| 36 | 15 sphinx/ext/autodoc/__init__.py | 1426 | 1766| 3118 | 17936 | 74663 | 
| 37 | 15 doc/usage/extensions/example_google.py | 298 | 314| 125 | 18061 | 74663 | 
| 38 | 16 sphinx/ext/autodoc/mock.py | 11 | 69| 452 | 18513 | 76011 | 
| 39 | 17 sphinx/transforms/references.py | 11 | 54| 266 | 18779 | 76329 | 
| 40 | 17 sphinx/ext/autodoc/mock.py | 72 | 96| 210 | 18989 | 76329 | 
| 41 | 18 sphinx/pycode/ast.py | 11 | 44| 214 | 19203 | 78414 | 
| 42 | 18 sphinx/highlighting.py | 71 | 180| 873 | 20076 | 78414 | 
| 43 | 18 sphinx/domains/python.py | 287 | 321| 382 | 20458 | 78414 | 
| 44 | 18 sphinx/ext/autodoc/__init__.py | 2060 | 2261| 1807 | 22265 | 78414 | 
| 45 | 18 sphinx/ext/autodoc/__init__.py | 495 | 528| 286 | 22551 | 78414 | 
| 46 | 18 sphinx/pycode/ast.py | 148 | 193| 506 | 23057 | 78414 | 
| 47 | 19 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 23314 | 83713 | 
| 48 | **19 sphinx/util/typing.py** | 195 | 211| 174 | 23488 | 83713 | 
| 49 | 20 sphinx/writers/texinfo.py | 858 | 962| 788 | 24276 | 96027 | 
| 50 | 21 sphinx/writers/html5.py | 155 | 211| 521 | 24797 | 103149 | 
| 51 | 22 sphinx/domains/c.py | 2600 | 3404| 6591 | 31388 | 135447 | 
| 52 | 22 sphinx/ext/autodoc/__init__.py | 710 | 818| 879 | 32267 | 135447 | 
| 53 | 22 sphinx/ext/autodoc/__init__.py | 1395 | 1766| 189 | 32456 | 135447 | 
| 54 | 23 sphinx/writers/latex.py | 785 | 857| 605 | 33061 | 154836 | 
| 55 | 24 sphinx/cmd/quickstart.py | 11 | 119| 756 | 33817 | 160405 | 
| 56 | 24 sphinx/ext/autodoc/__init__.py | 1103 | 1120| 182 | 33999 | 160405 | 
| 57 | 24 sphinx/ext/autodoc/__init__.py | 890 | 976| 806 | 34805 | 160405 | 
| 58 | 24 sphinx/ext/autodoc/__init__.py | 584 | 596| 133 | 34938 | 160405 | 
| 59 | 24 sphinx/ext/autodoc/__init__.py | 2712 | 2742| 363 | 35301 | 160405 | 
| 60 | 25 sphinx/util/cfamily.py | 153 | 170| 136 | 35437 | 163862 | 
| 61 | 26 sphinx/util/docfields.py | 169 | 195| 259 | 35696 | 167353 | 
| 62 | 27 sphinx/writers/text.py | 1061 | 1173| 813 | 36509 | 176354 | 
| 63 | 28 sphinx/ext/todo.py | 14 | 42| 193 | 36702 | 178194 | 
| 64 | 28 sphinx/ext/autodoc/__init__.py | 1023 | 1035| 124 | 36826 | 178194 | 
| 65 | 29 sphinx/application.py | 1108 | 1121| 173 | 36999 | 189808 | 
| 66 | 29 sphinx/ext/napoleon/docstring.py | 931 | 949| 115 | 37114 | 189808 | 
| 67 | 30 sphinx/writers/html.py | 184 | 240| 516 | 37630 | 197398 | 
| 68 | 30 sphinx/writers/text.py | 664 | 781| 839 | 38469 | 197398 | 
| 69 | 30 sphinx/domains/c.py | 785 | 908| 952 | 39421 | 197398 | 
| 70 | 30 sphinx/ext/autodoc/deprecated.py | 114 | 127| 114 | 39535 | 197398 | 
| 71 | **30 sphinx/util/typing.py** | 11 | 73| 452 | 39987 | 197398 | 


### Hint

```
Hi @Gobot1234 , would you please upload your project to GitHub instead? Google Drive is not very convenient to read code.
https://github.com/Gobot1234/sphinx-test
@astrojuanlu Should I open a PR to fix this?
@Gobot1234 Yes, please!
```

## Patch

```diff
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -310,7 +310,10 @@ def stringify(annotation: Any) -> str:
         return INVALID_BUILTIN_CLASSES[annotation]
     elif (getattr(annotation, '__module__', None) == 'builtins' and
           hasattr(annotation, '__qualname__')):
-        return annotation.__qualname__
+        if hasattr(annotation, '__args__'):  # PEP 585 generic
+            return repr(annotation)
+        else:
+            return annotation.__qualname__
     elif annotation is Ellipsis:
         return '...'
 

```

## Test Patch

```diff
diff --git a/tests/test_util_typing.py b/tests/test_util_typing.py
--- a/tests/test_util_typing.py
+++ b/tests/test_util_typing.py
@@ -175,6 +175,18 @@ def test_stringify_type_hints_containers():
     assert stringify(Generator[None, None, None]) == "Generator[None, None, None]"
 
 
+@pytest.mark.skipif(sys.version_info < (3, 9), reason='python 3.9+ is required.')
+def test_stringify_type_hints_pep_585():
+    assert stringify(list[int]) == "list[int]"
+    assert stringify(list[str]) == "list[str]"
+    assert stringify(dict[str, float]) == "dict[str, float]"
+    assert stringify(tuple[str, str, str]) == "tuple[str, str, str]"
+    assert stringify(tuple[str, ...]) == "tuple[str, ...]"
+    assert stringify(tuple[()]) == "tuple[()]"
+    assert stringify(list[dict[str, tuple]]) == "list[dict[str, tuple]]"
+    assert stringify(type[int]) == "type[int]"
+
+
 @pytest.mark.skipif(sys.version_info < (3, 9), reason='python 3.9+ is required.')
 def test_stringify_Annotated():
     from typing import Annotated  # type: ignore

```


## Code snippets

### 1 - sphinx/ext/autodoc/typehints.py:

Start line: 83, End line: 127

```python
def modify_field_list(node: nodes.field_list, annotations: Dict[str, str]) -> None:
    arguments: Dict[str, Dict[str, bool]] = {}
    fields = cast(Iterable[nodes.field], node)
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                arg = arguments.setdefault(parts[1], {})
                arg['param'] = True
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                arg = arguments.setdefault(name, {})
                arg['param'] = True
                arg['type'] = True
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            arg = arguments.setdefault(name, {})
            arg['type'] = True
        elif parts[0] == 'rtype':
            arguments['return'] = {'type': True}

    for name, annotation in annotations.items():
        if name == 'return':
            continue

        arg = arguments.get(name, {})
        if not arg.get('type'):
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotation))
            node += field
        if not arg.get('param'):
            field = nodes.field()
            field += nodes.field_name('', 'param ' + name)
            field += nodes.field_body('', nodes.paragraph('', ''))
            node += field

    if 'return' in annotations and 'return' not in arguments:
        field = nodes.field()
        field += nodes.field_name('', 'rtype')
        field += nodes.field_body('', nodes.paragraph('', annotation))
        node += field
```
### 2 - sphinx/ext/autodoc/typehints.py:

Start line: 130, End line: 185

```python
def augment_descriptions_with_types(
    node: nodes.field_list,
    annotations: Dict[str, str],
) -> None:
    fields = cast(Iterable[nodes.field], node)
    has_description = set()  # type: Set[str]
    has_type = set()  # type: Set[str]
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                has_description.add(parts[1])
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                has_description.add(name)
                has_type.add(name)
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            has_type.add(name)
        elif parts[0] == 'return':
            has_description.add('return')
        elif parts[0] == 'rtype':
            has_type.add('return')

    # Add 'type' for parameters with a description but no declared type.
    for name in annotations:
        if name == 'return':
            continue
        if name in has_description and name not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
            node += field

    # Add 'rtype' if 'return' is present and 'rtype' isn't.
    if 'return' in annotations:
        if 'return' in has_description and 'return' not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'rtype')
            field += nodes.field_body('', nodes.paragraph('', annotations['return']))
            node += field


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('autodoc-process-signature', record_typehints)
    app.connect('object-description-transform', merge_typehints)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 2534, End line: 2554

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def update_annotations(self, parent: Any) -> None:
        """Update __annotations__ to support type_comment and so on."""
        try:
            annotations = dict(inspect.getannotations(parent))
            parent.__annotations__ = annotations

            for cls in inspect.getmro(parent):
                try:
                    module = safe_getattr(cls, '__module__')
                    qualname = safe_getattr(cls, '__qualname__')

                    analyzer = ModuleAnalyzer.for_module(module)
                    analyzer.analyze()
                    for (classname, attrname), annotation in analyzer.annotations.items():
                        if classname == qualname and attrname not in annotations:
                            annotations[attrname] = annotation
                except (AttributeError, PycodeError):
                    pass
        except (AttributeError, TypeError):
            # Failed to set __annotations__ (built-in, extensions, etc.)
            pass
```
### 4 - sphinx/util/typing.py:

Start line: 386, End line: 449

```python
def _stringify_py36(annotation: Any) -> str:
    """stringify() for py36."""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        elif getattr(annotation, '__origin__', None):
            qualname = stringify(annotation.__origin__)  # ex. Union
        else:
            qualname = repr(annotation).replace('typing.', '')
    elif hasattr(annotation, '__qualname__'):
        qualname = '%s.%s' % (module, annotation.__qualname__)
    else:
        qualname = repr(annotation)

    if (isinstance(annotation, typing.TupleMeta) and  # type: ignore
            not hasattr(annotation, '__tuple_params__')):  # for Python 3.6
        params = annotation.__args__
        if params:
            param_str = ', '.join(stringify(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
        else:
            return qualname
    elif isinstance(annotation, typing.GenericMeta):
        params = None
        if annotation.__args__ is None or len(annotation.__args__) <= 2:  # type: ignore  # NOQA
            params = annotation.__args__  # type: ignore
        elif annotation.__origin__ == Generator:  # type: ignore
            params = annotation.__args__  # type: ignore
        else:  # typing.Callable
            args = ', '.join(stringify(arg) for arg
                             in annotation.__args__[:-1])  # type: ignore
            result = stringify(annotation.__args__[-1])  # type: ignore
            return '%s[[%s], %s]' % (qualname, args, result)
        if params is not None:
            param_str = ', '.join(stringify(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
    elif (hasattr(annotation, '__origin__') and
          annotation.__origin__ is typing.Union):
        params = annotation.__args__
        if params is not None:
            if len(params) > 1 and params[-1] is NoneType:
                if len(params) > 2:
                    param_str = ", ".join(stringify(p) for p in params[:-1])
                    return 'Optional[Union[%s]]' % param_str
                else:
                    return 'Optional[%s]' % stringify(params[0])
            else:
                param_str = ', '.join(stringify(p) for p in params)
                return 'Union[%s]' % param_str

    return qualname


deprecated_alias('sphinx.util.typing',
                 {
                     'DirectiveOption': Callable[[str], Any],
                 },
                 RemovedInSphinx60Warning)
```
### 5 - sphinx/util/typing.py:

Start line: 287, End line: 320

```python
def stringify(annotation: Any) -> str:
    """Stringify type annotation object."""
    from sphinx.util import inspect  # lazy loading

    if isinstance(annotation, str):
        if annotation.startswith("'") and annotation.endswith("'"):
            # might be a double Forward-ref'ed type.  Go unquoting.
            return annotation[1:-1]
        else:
            return annotation
    elif isinstance(annotation, TypeVar):
        if annotation.__module__ == 'typing':
            return annotation.__name__
        else:
            return '.'.join([annotation.__module__, annotation.__name__])
    elif inspect.isNewType(annotation):
        # Could not get the module where it defined
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif annotation in INVALID_BUILTIN_CLASSES:
        return INVALID_BUILTIN_CLASSES[annotation]
    elif (getattr(annotation, '__module__', None) == 'builtins' and
          hasattr(annotation, '__qualname__')):
        return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'

    if sys.version_info >= (3, 7):  # py37+
        return _stringify_py37(annotation)
    else:
        return _stringify_py36(annotation)
```
### 6 - sphinx/domains/python.py:

Start line: 110, End line: 186

```python
def _parse_annotation(annotation: str, env: BuildEnvironment = None) -> List[Node]:
    """Parse type annotation."""
    def unparse(node: ast.AST) -> List[Node]:
        if isinstance(node, ast.Attribute):
            return [nodes.Text("%s.%s" % (unparse(node.value)[0], node.attr))]
        elif isinstance(node, ast.BinOp):
            result: List[Node] = unparse(node.left)
            result.extend(unparse(node.op))
            result.extend(unparse(node.right))
            return result
        elif isinstance(node, ast.BitOr):
            return [nodes.Text(' '), addnodes.desc_sig_punctuation('', '|'), nodes.Text(' ')]
        elif isinstance(node, ast.Constant):  # type: ignore
            if node.value is Ellipsis:
                return [addnodes.desc_sig_punctuation('', "...")]
            else:
                return [nodes.Text(node.value)]
        elif isinstance(node, ast.Expr):
            return unparse(node.value)
        elif isinstance(node, ast.Index):
            return unparse(node.value)
        elif isinstance(node, ast.List):
            result = [addnodes.desc_sig_punctuation('', '[')]
            if node.elts:
                # check if there are elements in node.elts to only pop the
                # last element of result if the for-loop was run at least
                # once
                for elem in node.elts:
                    result.extend(unparse(elem))
                    result.append(addnodes.desc_sig_punctuation('', ', '))
                result.pop()
            result.append(addnodes.desc_sig_punctuation('', ']'))
            return result
        elif isinstance(node, ast.Module):
            return sum((unparse(e) for e in node.body), [])
        elif isinstance(node, ast.Name):
            return [nodes.Text(node.id)]
        elif isinstance(node, ast.Subscript):
            result = unparse(node.value)
            result.append(addnodes.desc_sig_punctuation('', '['))
            result.extend(unparse(node.slice))
            result.append(addnodes.desc_sig_punctuation('', ']'))
            return result
        elif isinstance(node, ast.Tuple):
            if node.elts:
                result = []
                for elem in node.elts:
                    result.extend(unparse(elem))
                    result.append(addnodes.desc_sig_punctuation('', ', '))
                result.pop()
            else:
                result = [addnodes.desc_sig_punctuation('', '('),
                          addnodes.desc_sig_punctuation('', ')')]

            return result
        else:
            if sys.version_info < (3, 8):
                if isinstance(node, ast.Ellipsis):
                    return [addnodes.desc_sig_punctuation('', "...")]
                elif isinstance(node, ast.NameConstant):
                    return [nodes.Text(node.value)]

            raise SyntaxError  # unsupported syntax

    if env is None:
        warnings.warn("The env parameter for _parse_annotation becomes required now.",
                      RemovedInSphinx50Warning, stacklevel=2)

    try:
        tree = ast_parse(annotation)
        result = unparse(tree)
        for i, node in enumerate(result):
            if isinstance(node, nodes.Text) and node.strip():
                result[i] = type_to_xref(str(node), env)
        return result
    except SyntaxError:
        return [type_to_xref(annotation, env)]
```
### 7 - sphinx/util/typing.py:

Start line: 323, End line: 383

```python
def _stringify_py37(annotation: Any) -> str:
    """stringify() for py37+."""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        else:
            qualname = stringify(annotation.__origin__)  # ex. Union
    elif hasattr(annotation, '__qualname__'):
        qualname = '%s.%s' % (module, annotation.__qualname__)
    elif hasattr(annotation, '__origin__'):
        # instantiated generic provided by a user
        qualname = stringify(annotation.__origin__)
    elif types_Union and isinstance(annotation, types_Union):  # types.Union (for py3.10+)
        qualname = 'types.Union'
    else:
        # we weren't able to extract the base type, appending arguments would
        # only make them appear twice
        return repr(annotation)

    if getattr(annotation, '__args__', None):
        if not isinstance(annotation.__args__, (list, tuple)):
            # broken __args__ found
            pass
        elif qualname == 'Union':
            if len(annotation.__args__) > 1 and annotation.__args__[-1] is NoneType:
                if len(annotation.__args__) > 2:
                    args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
                    return 'Optional[Union[%s]]' % args
                else:
                    return 'Optional[%s]' % stringify(annotation.__args__[0])
            else:
                args = ', '.join(stringify(a) for a in annotation.__args__)
                return 'Union[%s]' % args
        elif qualname == 'types.Union':
            if len(annotation.__args__) > 1 and None in annotation.__args__:
                args = ' | '.join(stringify(a) for a in annotation.__args__ if a)
                return 'Optional[%s]' % args
            else:
                return ' | '.join(stringify(a) for a in annotation.__args__)
        elif qualname == 'Callable':
            args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
            returns = stringify(annotation.__args__[-1])
            return '%s[[%s], %s]' % (qualname, args, returns)
        elif qualname == 'Literal':
            args = ', '.join(repr(a) for a in annotation.__args__)
            return '%s[%s]' % (qualname, args)
        elif str(annotation).startswith('typing.Annotated'):  # for py39+
            return stringify(annotation.__args__[0])
        elif all(is_system_TypeVar(a) for a in annotation.__args__):
            # Suppress arguments if all system defined TypeVars (ex. Dict[KT, VT])
            return qualname
        else:
            args = ', '.join(stringify(a) for a in annotation.__args__)
            return '%s[%s]' % (qualname, args)

    return qualname
```
### 8 - sphinx/ext/autodoc/type_comment.py:

Start line: 11, End line: 35

```python
from inspect import Parameter, Signature, getsource
from typing import Any, Dict, List, cast

import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import inspect, logging

logger = logging.getLogger(__name__)


def not_suppressed(argtypes: List[ast.AST] = []) -> bool:
    """Check given *argtypes* is suppressed type_comment or not."""
    if len(argtypes) == 0:  # no argtypees
        return False
    elif len(argtypes) == 1 and ast_unparse(argtypes[0]) == "...":  # suppressed
        # Note: To support multiple versions of python, this uses ``ast_unparse()`` for
        # comparison with Ellipsis.  Since 3.8, ast.Constant has been used to represent
        # Ellipsis node instead of ast.Ellipsis.
        return False
    else:  # not suppressed
        return True
```
### 9 - sphinx/ext/autodoc/typehints.py:

Start line: 11, End line: 37

```python
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast

from docutils import nodes
from docutils.nodes import Element

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing


def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
                     options: Dict, args: str, retann: str) -> None:
    """Record type hints to env object."""
    try:
        if callable(obj):
            annotations = app.env.temp_data.setdefault('annotations', {})
            annotation = annotations.setdefault(name, OrderedDict())
            sig = inspect.signature(obj, type_aliases=app.config.autodoc_type_aliases)
            for param in sig.parameters.values():
                if param.annotation is not param.empty:
                    annotation[param.name] = typing.stringify(param.annotation)
            if sig.return_annotation is not sig.empty:
                annotation['return'] = typing.stringify(sig.return_annotation)
    except (TypeError, ValueError):
        pass
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 2623, End line: 2648

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        # Check the attribute has a docstring-comment
        comment = self.get_attribute_comment(self.parent, self.objpath[-1])
        if comment:
            return [comment]

        try:
            # Disable `autodoc_inherit_docstring` temporarily to avoid to obtain
            # a docstring from the value which descriptor returns unexpectedly.
            # ref: https://github.com/sphinx-doc/sphinx/issues/7805
            orig = self.config.autodoc_inherit_docstrings
            self.config.autodoc_inherit_docstrings = False  # type: ignore
            return super().get_doc(ignore)
        finally:
            self.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        # Disable analyzing attribute comment on Documenter.add_content() to control it on
        # AttributeDocumenter.add_content()
        self.analyzer = None

        if more_content is None:
            more_content = StringList()
        self.update_content(more_content)
        super().add_content(more_content, no_docstring)
```
### 32 - sphinx/util/typing.py:

Start line: 213, End line: 284

```python
def _restify_py36(cls: Optional[Type]) -> str:
    # ... other code

    if (isinstance(cls, typing.TupleMeta) and  # type: ignore
            not hasattr(cls, '__tuple_params__')):
        if module == 'typing':
            reftext = ':class:`~typing.%s`' % qualname
        else:
            reftext = ':class:`%s`' % qualname

        params = cls.__args__
        if params:
            param_str = ', '.join(restify(p) for p in params)
            return reftext + '\\ [%s]' % param_str
        else:
            return reftext
    elif isinstance(cls, typing.GenericMeta):
        if module == 'typing':
            reftext = ':class:`~typing.%s`' % qualname
        else:
            reftext = ':class:`%s`' % qualname

        if cls.__args__ is None or len(cls.__args__) <= 2:
            params = cls.__args__
        elif cls.__origin__ == Generator:
            params = cls.__args__
        else:  # typing.Callable
            args = ', '.join(restify(arg) for arg in cls.__args__[:-1])
            result = restify(cls.__args__[-1])
            return reftext + '\\ [[%s], %s]' % (args, result)

        if params:
            param_str = ', '.join(restify(p) for p in params)
            return reftext + '\\ [%s]' % (param_str)
        else:
            return reftext
    elif (hasattr(cls, '__origin__') and
          cls.__origin__ is typing.Union):
        params = cls.__args__
        if params is not None:
            if len(params) > 1 and params[-1] is NoneType:
                if len(params) > 2:
                    param_str = ", ".join(restify(p) for p in params[:-1])
                    return (':obj:`~typing.Optional`\\ '
                            '[:obj:`~typing.Union`\\ [%s]]' % param_str)
                else:
                    return ':obj:`~typing.Optional`\\ [%s]' % restify(params[0])
            else:
                param_str = ', '.join(restify(p) for p in params)
                return ':obj:`~typing.Union`\\ [%s]' % param_str
        else:
            return ':obj:`Union`'
    elif hasattr(cls, '__qualname__'):
        if cls.__module__ == 'typing':
            return ':class:`~%s.%s`' % (cls.__module__, cls.__qualname__)
        else:
            return ':class:`%s.%s`' % (cls.__module__, cls.__qualname__)
    elif hasattr(cls, '_name'):
        # SpecialForm
        if cls.__module__ == 'typing':
            return ':obj:`~%s.%s`' % (cls.__module__, cls._name)
        else:
            return ':obj:`%s.%s`' % (cls.__module__, cls._name)
    elif hasattr(cls, '__name__'):
        # not a class (ex. TypeVar)
        if cls.__module__ == 'typing':
            return ':obj:`~%s.%s`' % (cls.__module__, cls.__name__)
        else:
            return ':obj:`%s.%s`' % (cls.__module__, cls.__name__)
    else:
        # others (ex. Any)
        if cls.__module__ == 'typing':
            return ':obj:`~%s.%s`' % (cls.__module__, qualname)
        else:
            return ':obj:`%s.%s`' % (cls.__module__, qualname)
```
### 48 - sphinx/util/typing.py:

Start line: 195, End line: 211

```python
def _restify_py36(cls: Optional[Type]) -> str:
    module = getattr(cls, '__module__', None)
    if module == 'typing':
        if getattr(cls, '_name', None):
            qualname = cls._name
        elif getattr(cls, '__qualname__', None):
            qualname = cls.__qualname__
        elif getattr(cls, '__forward_arg__', None):
            qualname = cls.__forward_arg__
        elif getattr(cls, '__origin__', None):
            qualname = stringify(cls.__origin__)  # ex. Union
        else:
            qualname = repr(cls).replace('typing.', '')
    elif hasattr(cls, '__qualname__'):
        qualname = '%s.%s' % (module, cls.__qualname__)
    else:
        qualname = repr(cls)
    # ... other code
```
### 71 - sphinx/util/typing.py:

Start line: 11, End line: 73

```python
import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union

from docutils import nodes
from docutils.parsers.rst.states import Inliner

from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias

if sys.version_info > (3, 7):
    from typing import ForwardRef
else:
    from typing import _ForwardRef  # type: ignore

    class ForwardRef:
        """A pseudo ForwardRef class for py36."""
        def __init__(self, arg: Any, is_argument: bool = True) -> None:
            self.arg = arg

        def _evaluate(self, globalns: Dict, localns: Dict) -> Any:
            ref = _ForwardRef(self.arg)
            return ref._eval_type(globalns, localns)

if sys.version_info > (3, 10):
    from types import Union as types_Union
else:
    types_Union = None

if False:
    # For type annotation
    from typing import Type  # NOQA # for python3.5.1


# builtin classes that have incorrect __module__
INVALID_BUILTIN_CLASSES = {
    Struct: 'struct.Struct',  # Before Python 3.9
    TracebackType: 'types.TracebackType',
}


# Text like nodes which are initialized with text and rawsource
TextlikeNode = Union[nodes.Text, nodes.TextElement]

# type of None
NoneType = type(None)

# path matcher
PathMatcher = Callable[[str], bool]

# common role functions
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]],
                        Tuple[List[nodes.Node], List[nodes.system_message]]]

# A option spec for directive
OptionSpec = Dict[str, Callable[[str], Any]]

# title getter functions for enumerable nodes (see sphinx.domains.std)
TitleGetter = Callable[[nodes.Node], str]

# inventory data on memory
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]
```
