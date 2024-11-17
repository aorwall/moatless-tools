# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_Right now I'm focusing on [moatless-tree-search](https://github.com/aorwall/moatless-tree-search), an extended version of moatless-tools that builds a tree structure of nodes with parallel solutions and uses tree search to find the optimal trajectory. The code in moatless-tools has been simplified and is now a streamlined version of this expanded codebase.

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas and am currently sharing the sixth place on the SWE-Bench Lite Leaderboard. 

### Version 0.0.3: Claude 3.5 Sonnet v20241022
With version 0.0.3 I get 38.3% solve rate with Claude 3.5 Sonnet v20241022. Average cost per instance is $0.30.

The three main reasons I’ve been able to go from 27% to 38% solved instances in this version:

- **Claude 3.5 Sonnet and Computer Use**  
  The solution has been adjusted to use the `text_editor_20241022` tool introduced in the new version of Claude 3.5 Sonnet. This provides more stable results when editing existing code.  

- **[moatless-testbeds](https://github.com/aorwall/moatless-testbeds)**  
  I set up a Kubernetes-based solution to run tests and provide feedback on test results to the agent. It’s worth noting that the agent has to independently identify the tests and can’t rely on the `PASS_TO_PASS` or `FAIL_TO_PASS` data for each instance.  

- **More flexible model**  
  In the earlier version of Moatless Tools, the agent followed a rigid flow where it first retrieved content and then edited the code. Now, it can dynamically choose between actions for code retrieval or editing, depending on the situation.

[Try the Claude 3.5 Sonnet v20241022 evaluation set up on Google Colab](https://colab.research.google.com/drive/1yOCXhTujvX4QIGJuO73UIVVqAqgwlhmC?usp=sharing)


### Version 0.0.2: Claude 3.5 Sonnet
With version 0.0.2 I get 26.7% solve rate with Claude 3.5 Sonnet, with a bit higher cost of $0.17 per instance. 

[Try the Claude 3.5 evaluation set up on Google Colab](https://colab.research.google.com/drive/1pKecc3pumsrOGzTOOCEqjRKzeCWLWQpj?usp=sharing)

### Version 0.0.1: GPT-4o
Moatless Tools 0.0.1 has a solve rate of 24%, with each benchmark instance costing an average of $0.13 to solve with GPT-4o. Running the SWE Bench Lite dataset with 300 instances costs approx 40 dollars. 

[Try it out in Google Colab](https://colab.research.google.com/drive/15RpSjdprf9lcaP0oqKsuYfZl1c3kVB_t?usp=sharing)


# Try it out
I have focused on testing my ideas, and the project is currently a bit messy. My plan is to organize it in the coming period. However, feel free to clone the repo and try running this notebook:

1. [Run Moatless Tools on any repository](notebooks/00_index_and_run.ipynb)

## Environment Setup

Before running the evaluation, you'll need:
1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.
3. (Optional) Access to a testbed environment - see [moatless-testbeds](https://github.com/aorwall/moatless-testbeds) for setup instructions

You can configure these settings by either:

1. Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env with your values
```

2. Or export the variables directly:
   
```bash
# Directory for storing vector index store files  
export INDEX_STORE_DIR="/tmp/index_store"    

# Directory for storing clonedrepositories 
export REPO_DIR="/tmp/repos"

# Required: At least one LLM provider API key
export OPENAI_API_KEY="<your-key>"
export ANTHROPIC_API_KEY="<your-key>"

# ...or Base URL for custom LLM API service (optional)
export CUSTOM_LLM_API_BASE="<your-base-url>"
export CUSTOM_LLM_API_KEY="<your-key>"

# Required: API Key for Voyage Embeddings
export VOYAGE_API_KEY="<your-key>"

# Optional: Configuration for testbed environment (https://github.com/aorwall/moatless-testbeds)
export TESTBED_API_KEY="<your-key>"
export TESTBED_BASE_URL="<your-base-url>"
```

## Example

Basic setup using the `AgenticLoop` to solve a SWE-Bench instance.

```python
from moatless.agent import ActionAgent
from moatless.agent.code_prompts import SIMPLE_CODE_PROMPT
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, RequestMoreContext, RequestCodeChange, Finish, Reject

index_store_dir = "/tmp/index_store"
repo_base_dir = "/tmp/repos"
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")

completion_model = CompletionModel(model="gpt-4o", temperature=0.0)

repository = create_repository(instance)

code_index = CodeIndex.from_index_name(
    instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
)

actions = [
    FindClass(code_index=code_index, repository=repository),
    FindFunction(code_index=code_index, repository=repository),
    FindCodeSnippet(code_index=code_index, repository=repository),
    SemanticSearch(code_index=code_index, repository=repository),
    RequestMoreContext(repository=repository),
    RequestCodeChange(repository=repository, completion_model=completion_model),
    Finish(),
    Reject()
]

file_context = FileContext(repo=repository)
agent = ActionAgent(actions=actions, completion=completion_model, system_prompt=SIMPLE_CODE_PROMPT)

loop = AgenticLoop.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    repository=repository,
    persist_path=persist_path,
    max_iterations=50,
    max_cost=2.0  # Optional: Set maximum cost in dollars
)

final_node = loop.run()
if final_node:
    print(final_node.observation.message)
```
