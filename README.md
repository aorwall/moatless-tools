# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_For the implementation used in the paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285), please see [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)._

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas. 

* [Claude 3.5 Sonnet v20241022 evaluation results](https://experiments.moatless.ai/evaluations/20250113_claude_3_5_sonnet_20241022_temp_0_0_iter_20_fmt_tool_call_hist_messages_lite) - 39% solve rate, 2.7 resolved instances per dollar
* [Deepseek V3](https://experiments.moatless.ai/evaluations/20250111_deepseek_chat_v3_temp_0_0_iter_20_fmt_react_hist_react) - 30.7% solve rate, 24 resolved instances per dollar

# Try it out

## Environment Setup

You can install Moatless Tools either from PyPI or from source:

### Install from PyPI

```bash
# Install base package only
pip install moatless

# Install with streamlit visualization tools
pip install "moatless[streamlit]"

# Install with API server
pip install "moatless[api]"

# Install everything (including dev dependencies)
pip install "moatless[all]"
```

### Install from source

Clone the repository and install using Poetry:

```bash
# Clone the repository
git clone https://github.com/aorwall/moatless-tools.git
cd moatless-tools

# Using Poetry:

# Install base package only
poetry install

# Install with streamlit visualization tools
poetry install --with streamlit

# Install with API server
poetry install --with api

# Alternative: Install all optional components at once
poetry install --all-extras
```

## Environment Variables

Before running the evaluation, you'll need:
1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.
3. (Optional) Access to a testbed environment - see [moatless-testbeds](https://github.com/aorwall/moatless-testbeds) for setup instructions

You can configure these settings by either:

1. Create a `.env` file in the project root (copy from `.env.example`):

```bash
# Using Poetry:
cp .env.example .env
# Edit .env with your values

# Using pip:
curl -O https://raw.githubusercontent.com/aorwall/moatless-tools/main/.env.example
mv .env.example .env
# Edit .env with your values
```

2. Or export the variables directly:
   
```bash
# Directory for storing vector index store files  
export INDEX_STORE_DIR="/tmp/index_store"    

# Directory for storing cloned repositories 
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

> **Note**: The current version of litellm lacks support for computer use tools required by Claude 3.5 Sonnet. You need to use a specific dependency:
> ```toml
> litellm = { git = "https://github.com/aorwall/litellm.git", branch = "anthropic-computer-use" }
> ```

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
| deepseek/deepseek-reasoner | react | react | no (disabled thoughts) | [50%](https://experiments.moatless.ai/evaluations/20250120_deepseek_deepseek_reasoner_None_n_20_fmt_react_verified_mini) |
| gemini/gemini-2.0-flash-exp | react | react | no | [38%](https://experiments.moatless.ai/evaluations/20250119_gemini_gemini_2.0_flash_exp_0_0_n_20_fmt_react_verified_mini) |
| openrouter/meta-llama/llama-3.1-70b-instruct | react | react | no | - |
| openrouter/meta-llama/llama-3.1-405b-instruct | react | react | no | [28%](https://experiments.moatless.ai/evaluations/20250119_openai_meta_llama_Meta_Llama_3.1_405B_Instruct_FP8_0_0_n_20_fmt_react_verified_mini) | - |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | react | react | no | [32%](https://experiments.moatless.ai/evaluations/20250119_openai_Qwen_Qwen2.5_Coder_32B_Instruct_0_0_n_20_fmt_react_verified_mini) | - |

## Verify Setup

Before running the full evaluation, you can verify your setup using the integration test script:

```bash
# Run a single model test
python -m moatless.validation.validate_simple_code_flow --model claude-3-5-sonnet-20241022
```

The script will run the model against a sample SWE-Bench instance

Results are saved in `test_results/integration_test_<timestamp>/` .


## Run evaluation

The evaluation script supports various configuration options through command line arguments:

```bash
python -m moatless.benchmark.run_evaluation [OPTIONS]
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
python -m moatless.benchmark.run_evaluation \
  --model claude-3-5-sonnet-20241022 \
  --response-format react \
  --message-history react \
  --num-workers 10

# Run specific instances with GPT-4
python -m moatless.benchmark.run_evaluation \
  --model gpt-4o-2024-11-20 \
  --instance-ids "django__django-16527"
```

# Running the UI and API

The project includes a web UI for visualizing saved trajectory files, built with SvelteKit. The UI is packaged with the Python package and will be served by the API server.

First, make sure you have the required components installed:
```bash
pip install "moatless[api]"
```

### Start the API Server
```bash
moatless-api
```

This will start the FastAPI server on http://localhost:8000 and serve the UI at the same address.

### Development Mode

If you want to develop the UI, you can run it in development mode:

```bash
# From the ui directory
cd ui
pnpm install
pnpm run dev
```
The UI development server will be available at http://localhost:5173.

# Code Examples

Basic setup using the `AgenticLoop` to solve a SWE-Bench instance.

## Example 1: Using Claude 3.5 Sonnet
```python
from moatless.benchmark.swebench import create_repository
from moatless.evaluation.utils import get_moatless_instance
from moatless.agent.code_agent import CodingAgent
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.file_context import FileContext
from moatless.completion.base import LLMResponseFormat
from moatless.schema import MessageHistoryType

index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")
repository = create_repository(instance)
code_index = CodeIndex.from_index_name(
    instance["instance_id"], 
    index_store_dir=index_store_dir, 
    file_repo=repository
)
file_context = FileContext(repo=repository)

# Create agent using Claude 3.5 Sonnet with explicit config
agent = CodingAgent.create(
    repository=repository, # Repository instance with codebase
    code_index=code_index, # Code index for semantic search
    
    model="claude-3-5-sonnet-20241022",
    temperature=0.0, 
    max_tokens=4000,
    few_shot_examples=False, # We don't need few-shot examples for this model
)

loop = AgenticLoop.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    repository=repository,
    persist_path=persist_path,
    max_iterations=50,
    max_cost=2.0
)

final_node = loop.run()
if final_node:
    print(final_node.observation.message)
```

## Example 2: Using Deepseek V3
```python
from moatless.benchmark.swebench import create_repository
from moatless.evaluation.utils import get_moatless_instance
from moatless.agent.code_agent import CodingAgent
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.file_context import FileContext
from moatless.completion.base import LLMResponseFormat
from moatless.schema import MessageHistoryType

index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")
repository = create_repository(instance)
code_index = CodeIndex.from_index_name(
    instance["instance_id"], 
    index_store_dir=index_store_dir, 
    file_repo=repository
)
file_context = FileContext(repo=repository)

# Create agent using Deepseek Chat with explicit config
agent = CodingAgent.create(
    repository=repository,
    code_index=code_index,
    
    model="deepseek/deepseek-chat",
    temperature=0.0,
    max_tokens=4000,
    few_shot_examples=True,
)

loop = AgenticLoop.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    repository=repository,
    persist_path=persist_path,
    max_iterations=50,
    max_cost=2.0
)

final_node = loop.run()
if final_node:
    print(final_node.observation.message)
```

## CodingAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier from supported models table (e.g., "claude-3-5-sonnet-20241022") |
| `repository` | Repository | Required | Repository instance containing the codebase |
| `code_index` | CodeIndex | None | Code index for semantic search functionality |
| `runtime` | RuntimeEnvironment | None | Environment for running tests |
| `thoughts_in_action` | bool | From config | Whether to include thoughts in action responses, used when the LLM can't provide the reasoning in the message content |
| `disable_thoughts` | bool | From config | Whether to completely disable thought generation, used for reasoning models like o1 and Deepseek R1 |
| `few_shot_examples` | bool | From config | Whether to use few-shot examples in prompts |
| `temperature` | float | From config | Temperature for model sampling (0.0 = deterministic) |
| `max_tokens` | int | From config | Maximum tokens per model completion |

The default values for optional parameters are taken from the model's configuration in `model_config.py`. See the Verified Models table above for model-specific defaults.
