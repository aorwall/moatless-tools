# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have 
about how LLMs can be used to edit code in large existing codebases. I believe
that rather than relying on an agent to reason its way to a solution, it is 
crucial to build good tools to insert the right context into the prompt and 
handle the response.

I use the SWE-bench benchmark as a way to verify these ideas. Currently, Moatless Tools has a solution rate of 23%, with each benchmark instance costing an average of $0.1 to solve with GPT-4o. 
Running the SWE Bench Lite dataset with 300 instances costs approx 30 dollars.

## Try it out
I have focused on testing my ideas, and the project is currently a bit messy. My plan is to organize it in the coming
period. However, feel free to clone the repo and try running this notebook:

1. [Run Moatless Tools on any repository](notebooks/00_index_and_run.ipynb)

### Google Colab
You can also run the notebooks in Google Colab:

1. [Run the full SWE-bench Lite evaluation](https://colab.research.google.com/drive/15RpSjdprf9lcaP0oqKsuYfZl1c3kVB_t?usp=sharing)

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

