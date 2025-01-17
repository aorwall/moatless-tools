# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_For the implementation used in the paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285), please see [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)._

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas. 

### Version 0.0.4: Deepseek V3
With version 0.0.4 I get 30.7% solve rate (92 instances) using the open-source Deepseek V3 model. The most notable aspect of this is the extremely low cost - the entire evaluation run costs less than $4 ($0.0127 per instance), achieving **24 resolved instances per dollar spent**.

* [Deepseek V3 evaluation results](https://experiments.moatless.ai/evaluations/moatless_tools_v4_deepseek_chat_3_temp_0_iter_20_fmt_react)  
* [Claude 3.5 Sonnet v20241022 evaluation results](https://experiments.moatless.ai/evaluations/moatless_tools_v4_claude_3_5_sonnet_20241022_temp_0_iter_20_fmt_tool_call)

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

Install dependencies:
```bash
poetry install
```

## Environment Variables

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

## Verified Models

Default model configurations are provided for verified models. Note that other models may work but have not been extensively tested. When specifying just the `--model` argument, the following configurations are used:

| Model | Response Format | Message History | Thoughts in Action |
|-------|----------------|-----------------|-------------------|
| claude-3-5-sonnet-20241022 | tool_call | messages | no |
| claude-3-5-haiku-20241022 | tool_call | messages | no |
| gpt-4o-2024-11-20 | tool_call | messages | yes |
| gpt-4o-mini-2024-07-18 | tool_call | messages | yes |
| deepseek/deepseek-chat | react | react | yes |
| gemini/gemini-2.0-flash-exp | tool_call | messages | yes |
| openrouter/meta-llama/llama-3.1-70b-instruct | react | react | no |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | react | react | no |

## Verify Setup

Before running the full evaluation, you can verify your setup using the integration test script:

```bash
# Run a single model test
poetry run -m scripts.run_integration_tests --model claude-3-5-sonnet-20241022
```

The script will run the model against a sample SWE-Bench instance

Results are saved in `test_results/integration_test_<timestamp>/` .


## Run evaluation

The evaluation script supports various configuration options through command line arguments:

```bash
poetry run python -m scripts.run_evaluation [OPTIONS]
```

Required arguments:
- `--model MODEL`: Model to use for evaluation (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4o')

Optional arguments:
- Model settings:
  - `--model MODEL`: Model identifier. Can be a supported model from the table below or any custom model identifier. 
  - `--api-key KEY`: API key for the model
  - `--base-url URL`: Base URL for the model API
  - `--response-format FORMAT`: Response format ('tool_call' or 'react'). Defaults to 'tool_call' for custom models
  - `--message-history TYPE`: Message history type ('messages', 'summary', 'react', 'messages_compact', 'instruct'). Defaults to 'messages' for custom models
  - `--thoughts-in-action`: Enable thoughts in action
  - `--temperature FLOAT`: Temperature for model sampling. Defaults to 0.0

- Dataset settings:
  - `--split SPLIT`: Dataset split to use. Defaults to 'lite'
  - `--instance-ids ID [ID ...]`: Specific instance IDs to evaluate

- Loop settings:
  - `--max-iterations INT`: Maximum number of iterations
  - `--max-cost FLOAT`: Maximum cost in dollars

- Runner settings:
  - `--num-workers INT`: Number of parallel workers. Defaults to 10
  - `--evaluation-name NAME`: Custom name for the evaluation run
  - `--rerun-errors`: Rerun instances that previously errored

Available dataset splits that can be specified with the `--split` argument:

| Split Name | Description | Instance Count |
|------------|-------------|----------------|
| easy | Instances solved by at least 85% of all submissions to SWE-Bench. Can be used to verify the evaluation setup. | 4 |
| lite | All instances from the lite dataset | 300 | 
| lite_and_verified_solvable | Instances that exist in both lite and verified datasets and have at least one solved submission to SWE-Bench | 84 |
| verified | All instances from the verified dataset | 500 | 
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini), a subset of SWE-Bench Verified  | 50 |

Example usage:
```bash
# Run evaluation with Claude 3.5 Sonnet using the ReACT format
poetry run python -m scripts.run_evaluation \
  --model claude-3-5-sonnet-20241022 \
  --response-format react \
  --message-history react \
  --num-workers 10

# Run specific instances with GPT-4
poetry run python -m scripts.run_evaluation \
  --model gpt-4o \
  --instance-ids "django__django-16379"
```

# Code Example

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
