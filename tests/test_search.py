from moatless.retriever import CodeSnippet
from moatless.search import Search


def test_create_response_only_snippet_signatures():
    search = Search(path="data/python", code_index=None)

    snippets = [
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=3,
            end_line=5,
            tokens=10,
        ),
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=66,
            end_line=69,
            tokens=10,
        ),
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=381,
            end_line=392,
            tokens=10,
        ),
    ]

    response = search._create_response(snippets)

    print(response)

    assert response == "expected_response"


def test_create_response_only_signatures():
    search = Search(path="data/python", code_index=None)

    snippets = [
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=3,
            end_line=5,
            tokens=10,
        ),
    ]

    response = search._create_response(snippets, file_names=["schema.py"])

    print(response)

    assert response == "expected_response"


def test_create_response_with_class_signatures():
    search = Search(path="data/python", code_index=None)

    snippets = [
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=3,
            end_line=5,
            tokens=10,
        ),
    ]

    response = search._create_response(snippets, class_names=["Meta"])

    assert (
        response
        == """
print_block/schema.py
```python


# ...


class SchemaMeta(type):

    # ...

    def __new__(mcs, name, bases, attrs):

        # ...

    @classmethod
    def get_declared_fields(mcs, klass, cls_fields, inherited_fields, dict_cls):

        # ...

    # ...

    def _resolve_processors(self):

        # ...

# ...


class BaseSchema(base.SchemaABC):

    # ...

    class Meta(object):

        # ...

    # ...

# ...
```
"""
    )


def test_create_response_with_function():
    search = Search(path="data/python", code_index=None)

    snippets = [
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=3,
            end_line=5,
            tokens=10,
        ),
    ]

    response = search._create_response(
        snippets, file_names=["schema.py"], function_names=["handle_error"]
    )

    print(response)

    assert (
        response
        == """
print_block/schema.py
```python


# ...


class BaseSchema(base.SchemaABC):

    # ...

    def handle_error(self, error, data):
        \"""Custom error handler function for the schema.

        :param ValidationError error: The `ValidationError` raised during (de)serialization.
        :param data: The original input data.

        .. versionadded:: 2.0.0
        \"""
        pass

    # ...

# ...
```
"""
    )


def test_create_response_with_function_signature():
    search = Search(path="data/python", code_index=None)

    snippets = [
        CodeSnippet(
            id=None,
            file_path="print_block/schema.py",
            start_line=3,
            end_line=5,
            tokens=10,
        ),
    ]

    response = search._create_response(snippets, function_names=["handle_error"])

    print(response)

    assert (
        response
        == """
print_block/schema.py
```python


# ...


class BaseSchema(base.SchemaABC):

    # ...

    def handle_error(self, error, data):

        # ...

    # ...

# ...
```
"""
    )
