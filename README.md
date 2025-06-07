# Moatless Tools
Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_For the implementation used in the paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285), please see [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)._

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas. 

* [Claude 3.5 Sonnet v20241022 evaluation results](https://experiments.moatless.ai/evaluations/20250113_claude_3_5_sonnet_20241022_temp_0_0_iter_20_fmt_tool_call_hist_messages_lite) - 39% solve rate, 2.7 resolved instances per dollar
* [Deepseek V3](https://experiments.moatless.ai/evaluations/20250111_deepseek_chat_v3_temp_0_0_iter_20_fmt_react_hist_react) - 30.7% solve rate, 24 resolved instances per dollar

# Try it out

## Run in Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/aorwall/moatless-tools.git
   cd moatless-tools
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to set your API keys and other configuration options, including the required `MOATLESS_DIR` variable:
   ```
   MOATLESS_DIR=/path/to/your/moatless/data
   ```
   
   **Note**: `MOATLESS_DIR` specifies the directory where Moatless will store configuration files and trajectory data. This directory will be mounted as a volume in the Docker containers.

3. Start the services:
   ```bash
   make run
   ```

4. Access the UI at http://localhost

## Install from PyPI

```bash
# Install base package only
pip install moatless

# Install with Kubernetes runner support
pip install "moatless[kubernetes]"
```

## Install from source

Clone the repository and install using Poetry

```bash
# Clone the repository
git clone https://github.com/aorwall/moatless-tools.git
cd moatless-tools

# Install using uv
uv sync
```

# Code Examples

## Basic agent flow

```python
from moatless.actions import Respond
from moatless.agent import ActionAgent
from moatless.completion.tool_call import ToolCallCompletionModel

completion_model = ToolCallCompletionModel(
    model="gpt-4.1-mini",
    temperature=0.0,
    model_api_key=""
)

agent = ActionAgent(
    completion_model=completion_model,
    system_prompt="You are a helpful assistant that can answer questions.",
    actions=[
        Respond()
    ]
)

observation = await agent.run_simple("Hello")

print(observation.message)
```

## Code inspector agent

[notebooks/code_inspector_agent.ipynb](See notebook)

# Run SWE-Bench evaluations 

Before running the evaluation, you'll need:
1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.

## Verify Setup

Before running the full evaluation, you can verify your setup running a simple SWE-Bench instance.

```bash
uv run python scripts/docker_run.py  --flow swebench_tools --model-id gpt-4o-mini-2024-07-18 --instance-id django__django-11099 --evaluation-name testing_setup
```

The script will run the model against a sample SWE-Bench instance

Results are saved in `.moatless/projects/testing_setup`.

## Run evaluation

Evaluation script to run evaluation in a docker containers. 

```bash
python3 scripts/run_evaluation.py  --model gpt-4o-mini-2024-07-18 --dataset-split [dataset_split] --evaluation-name [evaluation_name]
```

Required arguments:
- `--dataset-split`: Dataset split to use
- `--evaluation-name`: Name of the evaluation

Optional arguments:
- `--model`: Model to use for evaluation (default: gpt-4o-mini-2024-07-18, equivalent to --model-id)
- `--model-id`: Model configuration ID to use (replaces entire model configuration)
- `--litellm-model-name`: LiteLLM model name to override (keeps other model settings)
- `--flow`: Flow to use for evaluation (defaults to "simple_coding")
- `--num-parallel-jobs`: Number of parallel jobs (default: 1)

## Flows 
Available flows that can be specified with the `--flow` argument:

| Flow ID | Format | Best Suited For | Default Model |
|---------|--------|-----------------|---------------|
| swebench_tools | Function calling | Models with native function calling support | gpt-4o-mini-2024-07-18 |
| swebench_tools_and_reasoning | Function calling | Reasoning models with native function calling support | claude-sonnet-4-20250514 |
| swebench_react | ReACT | Open source models without native function calling support | openrouter/mistralai/devstral-small |
| swebench_react_reasoning | ReACT | Reasoning models without function calling support | openrouter/deepseek/deepseek-r1-0528 |

## Model Configuration

Both evaluation scripts support flexible model configuration through the following options:

### Model Options

- `--model-id`: Specify a complete model configuration ID. This replaces the entire completion model configuration including temperature, max tokens, and all other settings.
- `--litellm-model-name`: Override only the [LiteLLM model name](https://docs.litellm.ai/docs/providers) while keeping all other completion model settings (temperature, max tokens, etc.) from the flow or model configuration.

### Verified Models

Default model configurations are provided for verified models. Note that other models may work but have not been extensively tested. 
Verified models are models that have been tested and found to work with the [Verified Mini subset](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) of the SWE-Bench dataset.

When specifying just the `--model-id` argument, the following configurations are used:

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

## Dataset splits

Available dataset splits that can be specified with the `--dataset-split` argument:

| Split Name | Description | Instance Count |
|------------|-------------|----------------|
| lite | All instances from the lite dataset | 300 | 
| verified | All instances from the verified dataset | 500 | 
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini), a subset of SWE-Bench Verified  | 50 |
| lite_and_verified_solvable | Instances that exist in both lite and verified datasets and have at least one solved submission to SWE-Bench | 84 |

## Example usage
```bash
# Run evaluation with Claude 3.5 Sonnet using complete model configuration
python3 scripts/run_evaluation.py \
  --model-id claude-3-5-sonnet-20241022 \
  --flow swebench_tools_and_reasoning \
  --dataset-split verified_mini \
  --num-parallel-jobs 5

# Run evaluation overriding just the model name while keeping flow's model settings
python3 scripts/run_evaluation.py \
  --litellm-model-name openrouter/qwen/qwen-2.5-coder-32b-instruct \
  --flow swebench_react \
  --dataset-split verified_mini \
  --num-parallel-jobs 5
