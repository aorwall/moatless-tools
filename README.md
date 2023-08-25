# Code Blocks

Code Blocks is a simplified variant of [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) implemented to simplify the
handling of code read and written by an LLM. The two main use cases are merging of incomplete code written by the LLM and 
splitting up code for embedding and indexing in a vector store.

## Merging
If you send a complete code file along with an instruction on what should be 
changed to an LLM like GPT3.5 and 4, it often responds with only the parts 
of the code that have been updated. Even if you explicitly instruct the LLM to
return the entire file, it doesn't always do so. Code Blocks' strategy is to 
handle the returned code instead of instructing GPT on how to return the code.
This is done by applying a number of not entirely scientific methods. The
merging process involves comparing the original and updated code blocks, 
identifying the differences, and merging them into a single, updated code block.

Examples and explanations of merge strategies:
- [Add new field to Java bean](docs/add_new_field_to_java_bean.md)
- [Add a new feature to a React component](docs/new_feature_in_react_component.md)

## Splitting
In order to index code in a vector store, it must first be split into blocks. 
Code Blocks attempts to do this by dividing the code into code blocks based 
on the structure of the code. This involves identifying the different components 
of the code (such as functions, classes, and statements), and splitting the code
at these boundaries. The resulting code blocks can then be individually indexed
in the vector store, allowing for more efficient storage and retrieval of code.

For examples of how Code Blocks splits code into chunks, see the following examples:
- [Java Example](tests/java/Example.md)
- [Python Example](tests/python/example.md)
- [TypeScript and React Example](tests/typescript/todo.md)

## Supported Languages
- [x] Python
- [x] Java
- [x] TypeScript
- [x] JavaScript
- [ ] ???

## Installation
To install Code Blocks, you can use pip:

```sh
pip install codeblocks-gpt
```

## Usage
Here is a basic example of how to use Code Blocks to split and merge code:

```python
from codeblocks import create_parser, CodeSplitter

# Create a parser for the language of your code (e.g., Python)
parser = create_parser("python")

# Parse your code into a CodeBlock
code_block = parser.parse(your_code)

# Create a CodeSplitter to split your code into blocks
splitter = CodeSplitter("python")

# Split your code into blocks
blocks = splitter.split_text(your_code)
```
