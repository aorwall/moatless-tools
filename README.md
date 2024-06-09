# Moatless Tools
*We Have No Moat, And Neither Does Devin*

Moatless Tools is a hobby project where I experiment with some ideas I have 
about how LLMs can be used to edit code in large existing codebases. I believe
that rather than relying on an agent to reason its way to a solution, it is 
crucial to build good tools to insert the right context into the prompt and 
handle the response.

I use the SWE-bench benchmark as a way to verify these ideas. Currently, Moatless Tools is third on the SWE Bench Lite
leaderboard with a solution rate of 24%, with each benchmark instance costing an average of $0.1 to solve with GPT-4o. 
Running the SWE Bench Lite dataset with 300 instances costs approx 30 dollars.

## Try it out
I have focused on testing my ideas, and the project is currently a bit messy. My plan is to organize it in the coming
period. However, feel free to clone and try running any of these notebooks.

1. [Run Moatless Tools on any repository](notebooks/00_index_and_run.ipynb)
3. [Run the full SWE-bench Lite evaluation](notebooks/01_run_swebench_evaluation.ipynb)


## How it works
The solution is based on two loops (or agents if you will): one for searching code and one for modifying code. These
loops receive an instruction which they then iterate over by interacting with an LLM until they reach a result.

### Search Loop
Search Loop uses function calling to find relevant code using the following parameters:

 * `query` - query using natural language to describe the desired code.
 * `code_snippet` - a specific code snippet that should be exactly matched.
 * `class_name` - a specific class name to include in the search.
 * `function_name` - a specific function name to include in the search.
 * `file_pattern` - A glob pattern to filter search results to specific file types or directories.

The search loop will continue to do search requests until it finds a relevant code.

For semantic search, a vector index is used, which is based on the llama index. This is a classic RAG solution where 
all code in the repository is chunked into relevant parts, such as at the method level, embedded, and indexed in a
vector store. For class and function name search, a simple index is used where all function and class names are indexed.
See this notebook for an example of the ingestion process.

### Code Loop
The code loop functions as a finite state machine that transitions between the states of receiving requests, seeking
clarifications, and making code changes. In the request, the "span" must be specified, and if this is too large, 
line numbers must also be specified. The code change is inspired by the edit block concept in 
[Aider](https://aider.chat/docs/benchmarks.html), where the LLM specifies the code to be changed in a *search* block and
the code it will be changed to in a *replace* block. However, since the code to be changed is already known to the Code 
Loop, the search section is pre-filled, and the LLM only needs to respond with the replace section. The idea is that 
this reduces the risk of changing the wrong code by having an agreement on what to change before doing the change.




### Installation

```bash
pip install moatless
```

### Setup API Keys



### 1. Setting Up the Workspace

We create a workspace by specifying the directories for the git repository and the index. The `from_dirs` method initializes the workspace with the given paths.

```python
from moatless.workspace import Workspace

workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
```

 * `repo_dir`: Path to your git repository.
 * `persist_dir`: Path where the index will be stored.

### 3. Defining Instructions

Define the instructions for both the search and code loops. 

```python
instructions = "Convert the 'handle_request' function in 'server.py' to use async/await."
```

### 3. Executing the Search Loop
We initialize a SearchLoop with the workspace and the instructions, then execute it to search through the code.

``` python
from moatless.search import SearchLoop

search = SearchLoop(workspace, instructions=instructions)
response = search.execute()
```

 * `instructions`: The instructions that the search loop will follow.

The SearchLoop will return `response.files`, which is a list of `FileWithSpans` objects. Each `FileWithSpans` object contains:

 * `file_path`: The path to the file containing the relevant code.
 * `span_ids`: A list of identifiers that specify the sections of the code that were found by the search. These can be classes, functions, or segments of larger code sections that need to be modified.

### 4. Executing the Code Loop

Using the response from the search loop, we initialize a CodeLoop and execute it to apply the necessary code changes.

```python
from moatless.code import CodeLoop

coder = CodeLoop(workspace, instructions=instructions, files=response.files)
coder.execute()
```
