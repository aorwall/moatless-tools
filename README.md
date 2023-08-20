# Code Blocks

Code Blocks is a simplified variant of Tree-sitter implemented to simplify the
handling of code read and written by a LLM. It is designed to assist
in the process of code generation and manipulation, providing a structured way
to handle code blocks.

The two main use cases are merging of incomplete code written by the LLM and 
splitting up code for embedding and indexing in a vector store.

## Merging
If you send a complete code file along with an instruction on what should be 
changed to an LLM like GPT3.5 and 4 (ChatGPT), it often returns only the parts 
of the code that have been updated. Even if you explicitly instruct the LLM to
return the entire file, it doesn't always do so. Code Blocks' strategy is to 
handle the returned code instead of instructing GPT on how to return the code.
This is done by applying a number of not entirely scientific methods. The
merging process involves comparing the original and updated code blocks, 
identifying the differences, and merging them into a single, updated code block.

## Splitting
In order to index code in a vector store, it must first be split into blocks. 
Code Blocks attempts to do this by dividing the code into code blocks based 
on the structure of the code. This involves identifying the different components 
of the code (such as functions, classes, and statements), and splitting the code
at these boundaries. The resulting code blocks can then be individually indexed
in the vector store, allowing for more efficient storage and retrieval of code.

For examples of how Code Blocks splits code into chunks, see the following examples:
- [Java Example](tests/java/example.md)
- [Python Example](tests/python/example.md)

## Supported Languages
- [x] Python
- [x] Java
- [ ] JavaScript
- [ ] Typescript
- [ ] ???

## Installation
To install Code Blocks, you can use pip:
