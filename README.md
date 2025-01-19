# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_For the implementation used in the paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285), please see [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)._

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas. 

### Version 0.0.4: Deepseek V3
With version 0.0.4 I get 30.7% solve rate (92 instances) using the open-source Deepseek V3 model. The most notable aspect of this is the extremely low cost - the entire evaluation run costs less than $4 ($0.0127 per instance), achieving **24 resolved instances per dollar spent**.

* [Deepseek V3 evaluation results](https://experiments.moatless.ai/evaluations/20250111_deepseek_chat_v3_temp_0_0_iter_20_fmt_react_hist_react)  
* [Claude 3.5 Sonnet v20241022 evaluation results](https://experiments.moatless.ai/evaluations/20250113_claude_3_5_sonnet_20241022_temp_0_0_iter_20_fmt_tool_call_hist_messages_lite)

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

Default model configurations are provided for verified models. Note that other models may work but have not been extensively tested. 
Verified models are models that have been tested and found to work with the [Verified Mini subset](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) of the SWE-Bench dataset.

When specifying just the `--model` argument, the following configurations are used:

| Model | Response Format | Message History | Thoughts in Action | Verified Mini |
|-------|----------------|-----------------|-------------------|---------------|
| claude-3-5-sonnet-20241022 | tool_call | messages | no | [46%](https://experiments.moatless.ai/evaluations/20250119_claude_3_5_sonnet_20241022_0_0_n_20_fmt_tool_call_verified_mini) | 
| claude-3-5-haiku-20241022 | tool_call | messages | no | [28%](https://experiments.moatless.ai/evaluations/20250118_claude_3_5_haiku_20241022_0_0_n_20_fmt_tool_call_verified_mini) |
| gpt-4o-2024-11-20 | tool_call | messages | yes | [32%](https://experiments.moatless.ai/evaluations/20250119_azure_gpt_4o_0_0_n_20_fmt_tool_call_thoughts-in-action_1_verified_mini) |
| gpt-4o-mini-2024-07-18 | tool_call | messages | yes | [16%](https://experiments.moatless.ai/evaluations/20250118_gpt_4o_mini_2024_07_18_0_0_n_20_fmt_tool_call_thoughts-in-action_6_verified_mini) |
| o1-mini-2024-09-12 | react | react | no (disabled thoughts) | [28%](https://experiments.moatless.ai/evaluations/20250114_o1_mini_2024_09_12_0_0_n_20_fmt_react_hist_react_verified_mini) |
| deepseek/deepseek-chat | react | react | no | [36%](https://experiments.moatless.ai/evaluations/20250118_deepseek_deepseek_chat_0_0_n_20_fmt_react_verified_mini) |
| gemini/gemini-2.0-flash-exp | react | react | no | [38%](https://experiments.moatless.ai/evaluations/20250119_gemini_gemini_2.0_flash_exp_0_0_n_20_fmt_react_verified_mini) |
| openrouter/meta-llama/llama-3.1-70b-instruct | react | react | no | - |
| openrouter/meta-llama/llama-3.1-405b-instruct | react | react | no | [28%](https://experiments.moatless.ai/evaluations/20250119_openai_meta_llama_Meta_Llama_3.1_405B_Instruct_FP8_0_0_n_20_fmt_react_verified_mini) | - |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | react | react | no | [32%](https://experiments.moatless.ai/evaluations/20250119_openai_Qwen_Qwen2.5_Coder_32B_Instruct_0_0_n_20_fmt_react_verified_mini) | - |

## Verify Setup

Before running the full evaluation, you can verify your setup using the integration test script:

```bash
# Run a single model test
poetry run python -m moatless.validation.validate_simple_code_flow --model claude-3-5-sonnet-20241022
```

The script will run the model against a sample SWE-Bench instance

Results are saved in `test_results/integration_test_<timestamp>/` .


## Run evaluation

The evaluation script supports various configuration options through command line arguments:

```bash
poetry run python -m moatless.benchmark.run_evaluation [OPTIONS]
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
| lite | All instances from the lite dataset | 300 | 
| verified | All instances from the verified dataset | 500 | 
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini), a subset of SWE-Bench Verified  | 50 |
| lite_and_verified_solvable | Instances that exist in both lite and verified datasets and have at least one solved submission to SWE-Bench | 84 |

Example usage:
```bash
# Run evaluation with Claude 3.5 Sonnet using the ReACT format
poetry run python -m moatless.benchmark.run_evaluation \
  --model claude-3-5-sonnet-20241022 \
  --response-format react \
  --message-history react \
  --num-workers 10

# Run specific instances with GPT-4
poetry run python -m moatless.benchmark.run_evaluation \
  --model gpt-4o-2024-11-20 \
  --instance-ids "django__django-16527"
```

# Code Example

Basic setup using the `AgenticLoop` to solve a SWE-Bench instance.

```python
from moatless.actions.string_replace import StringReplace
from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.base import BaseCompletionModel, LLMResponseFormat
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.schema import MessageHistoryType

index_store_dir = "/tmp/index_store"
repo_base_dir = "/tmp/repos"
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")

completion_model = BaseCompletionModel.create(response_format=LLMResponseFormat.TOOLS, model="claude-3-5-sonnet-20240620", temperature=0.0)

repository = create_repository(instance)

code_index = CodeIndex.from_index_name(
    instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
)

file_context = FileContext(repo=repository)
agent = CodingAgent.create(completion_model=completion_model, code_index=code_index, repository=repository, message_history_type=MessageHistoryType.MESSAGES)

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
