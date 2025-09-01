# Local Models

This is to document the steps to run open sourced models locally.

## Qwen3-Coder 30b

My first attempt was Qwen3-Coder on Ollama. That stack cannot work with moatless tools.

So I did Qwen3-Coder on LM Studio.

It works for the:

- [verify step in README.md](README.md#verify-setup)

### Preparing LM Studio

1. Install LM Studio
2. The model tested is`qwen/qwen3-coder-30b` - `mlx` - `quantized at 4-bit`. Your mileage may vary. Choose the version that best works on your setup.
3. Turn on Developer mode in LM Studio. Turn on the server. If `http://localhost:1234/v1/models` displays `qwen/qwen3-coder-30b` as running, then you have done it correctly.
4. Make sure you create .env out of .env.example and set `VOYAGE_API_KEY=<YOUR KEY>`

### How to run Verify Setup

1. run `docker run -d -p 6379:6379 redis:alpine` in one terminal because the run_docker.py script + instance requires redis
2. Please use the commit `8d671b6733f76e3ea12f3cbb9b0b7ac1b6249c27` or later.It makes changes to `docker_run.py`, `moatless/evaluation/manager.py`, `moatless/flow/manager.py`.
3. install uv if you haven't already. I used `brew install uv` on macOS.
4. The original example in [verify step in README.md](README.md#verify-setup) is `uv run python scripts/docker_run.py  --flow swebench_tools --model-id gpt-4o-mini-2024-07-18 --instance-id django__django-11099 --evaluation-name testing_setup`
5. The general form of the command is `uv run python scripts/docker_run.py --flow swebench_react --litellm-model-name openai/qwen3-coder-30b --model-base-url http://host.docker.internal:1234/v1 --model-api-key faux_key --instance-id <INSTANCE_ID> --evaluation-name <EVAL_NAME>` I will explain each param and why.
   1. we follow the README which uses uv
   2. we use swebench_react because open-sourced models tend to have issues with tool calling. See the [Flows section](README.md#flows).
   3. moatless-tools uses litellm to access models in general. Since we are using lm studio which is openai compatible, therefore, we use `openai` as the prefix. This is backed up by [this comment in litellm](https://github.com/BerriAI/litellm/issues/3755#issuecomment-2498859187). Which I will quote :
        > for posterity (took me a day to figure out), running in a dev container, leveraging LMstudio with CrewAI/ LiteLLM:
        > `llm_lmstudio = LLM( model="openai/llama-3.2-1b-instruct", base_url="http://host.docker.internal:1234/v1", api_key="faux_key", )`
    4. This is also why the moatless-tools code needed to be changed.
    5. `--model-base-url http://host.docker.internal:1234/v1 --model-api-key faux_key` we are using docker. I am assuming the lm studio is running on your host OS. so the base url has to be `http://host.docker.internal` instead of `http://localhost`. Also, the two params here are new to moatless tools. The `faux_key` follows the comment from earlier.
    6. `--instance-id` in the verify setup uses `django__django-11099` which you can just follow.
    7. `--evaluation-name` in the verify setup uses `testing_setup` which you is a bad idea. Because if you encounter any error and you rerun with the same value, you will get an error. I recommend adding a timestamp and everytime you run, make sure you change the value or you delete the previous folder in `.moatless/projects/testing_setup_<TIMESTAMP>`.
 6. What I use is `uv run python scripts/docker_run.py --flow swebench_react --litellm-model-name openai/qwen3-coder-30b --model-base-url http://host.docker.internal:1234/v1 --model-api-key faux_key --instance-id django__django-11099 --evaluation-name testing_setup_YYYYMMDD_HHii`

Once the script is done depending on the instance, for `django__django-11099`,

this is expected in `.moatless/projects/<YOUR_EVAL>/trajs/django__django-11099/logs/node_<node>/20250831_092503_802506_._tests_runtests.py_rc0.log`

```bash
Ran 22 tests in 0.061s

OK
Destroying test database for alias 'default' ('file:memorydb_default?mode=memory&cache=shared')...
Testing against Django installed in '/testbed/django'
Importing application auth_tests
Skipping setup of unused database(s): other.
Operations to perform:
  Synchronize unmigrated apps: auth, auth_tests, contenttypes, messages, sessions, staticfiles
  Apply all migrations: admin, sites
Synchronizing apps without migrations:
  Creating tables...
    Creating table django_content_type
    Creating table auth_permission
    Creating table auth_group
    Creating table auth_user
    Creating table django_session
    Creating table auth_tests_customuser
    Creating table auth_tests_customuserwithoutisactivefield
    Creating table auth_tests_extensionuser
    Creating table auth_tests_custompermissionsuser
    Creating table auth_tests_customusernonuniqueusername
    Creating table auth_tests_isactivetestuser1
    Creating table auth_tests_minimaluser
    Creating table auth_tests_nopassworduser
    Creating table auth_tests_concrete
    Creating table auth_tests_uuiduser
    Creating table auth_tests_email
    Creating table auth_tests_customuserwithfk
    Creating table auth_tests_integerusernameuser
    Creating table auth_tests_userwithdisabledlastloginfield
    Running deferred SQL...
Running migrations:
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying sites.0001_initial... OK
  Applying sites.0002_alter_domain_unique... OK
System check identified no issues (0 silenced).
```

### How to run Evaluation

Coming soon