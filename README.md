# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

## SWE-Bench
I use the SWE-bench benchmark as a way to verify my ideas. 

### GPT-4o
Moatless Tools 0.0.1 has a solve rate of 24%, with each benchmark instance costing an average of $0.13 to solve with GPT-4o. Running the SWE Bench Lite dataset with 300 instances costs approx 40 dollars. 

[Try it out in Google Colab](https://colab.research.google.com/drive/15RpSjdprf9lcaP0oqKsuYfZl1c3kVB_t?usp=sharing)

### Claude 3.5 Sonnet
With version 0.0.2 I get 26.7% solve rate with Claude 3.5 Sonnet, with a bit higher cost of $0.15 per instance. 

[Try the Claude 3.5 evaluation set up on Google Colab](https://colab.research.google.com/drive/1pKecc3pumsrOGzTOOCEqjRKzeCWLWQpj?usp=sharing)

## Try it out
I have focused on testing my ideas, and the project is currently a bit messy. My plan is to organize it in the coming period. However, feel free to clone the repo and try running this notebook:

1. [Run Moatless Tools on any repository](notebooks/00_index_and_run.ipynb)


## How it works
The solution is based on an agentic loop that functions as a finite state machine, transitioning between states. Each state can have its own prompts and response handling.

The following states are used in the usual workflow and code flow.

### Search
The Search Loop uses function calling to find relevant code using the following parameters:

 * `query` - A query using natural language to describe the desired code.
 * `code_snippet` - A specific code snippet that should be exactly matched.
 * `class_name` - A specific class name to include in the search.
 * `function_name` - A specific function name to include in the search.
 * `file_pattern` - A glob pattern to filter search results to specific file types or directories.

For semantic search, a vector index is used, which is based on the llama index. This is a classic RAG solution where all code in the repository is chunked into relevant parts, such as at the method level, embedded, and indexed in a  vector store. For class and function name search, a simple index is used where all function and class names are indexed.

### Identify
Identifies the code relevant to the task. If not all relevant code is found, it transitions back to Search. Once all relevant code is found, it transitions to PlanToCode.

### PlanToCode
Breaks down the request for code changes into smaller changes to specific parts (code spans) of the codebase.

### ClarifyChange
If the proposed changes affect too large a portion of the code, the change needs to be clarified to affect a smaller number of code lines.

### EditCode
Code is edited in search/replace blocks inspired by the edit block concept in [Aider](https://aider.chat/docs/benchmarks.html). In this concept, the LLM specifies the code to be changed in a search block and the code it will be changed to in a replace block. However, since the code to be changed is already known to the Code Loop, the search section is pre-filled, and the LLM only needs to respond with the replace section. The idea is that this reduces the risk of changing the wrong code by having an agreement on what to change before making the change.
