from moatless.actions.find_function import FindFunction, FindFunctionArgs
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.utils import get_moatless_instance
from moatless.file_context import FileContext


def test_find_function_init_method():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)

    action = FindFunction(
        repository=repository, code_index=code_index
    )

    action_args = FindFunctionArgs(
        scratch_pad="",
        class_name="ManagementUtility",
        function_name="__init__",
    )

    message = action.execute(action_args, file_context)
    print(message)
    assert len(file_context.files) == 1
    assert "ManagementUtility.__init__" in file_context.files[0].span_ids


def test_find_function():
    instance_id = "django__django-14855"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)

    action = FindFunction(
        repository=repository, code_index=code_index
    )

    action_args = FindFunctionArgs(
        scratch_pad="",
        function_name="cached_eval",
        file_pattern="**/*.py",
    )

    message = action.execute(action_args, file_context)
