from moatless.actions.view_code import (
    ViewCode,
    ViewCodeArgs,
    CodeSpan,
)
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.file_context import FileContext


def test_request_non_existing_method():
    instance_id = "django__django-11039"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    file_context = FileContext(repo=repository)

    action = ViewCode(
        repository=repository
    )

    args = ViewCodeArgs(
        scratch_pad="def non_existing_method():",
        files=[
            CodeSpan(
                file_path="tests/migrations/test_commands.py",
                span_ids=["test_sqlmigrate"],
            )
        ],
    )

    output = action.execute(args, file_context)
    print(output.message)


def test_request_many_spans():
    instance_id = "django__django-11039"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    file_context = FileContext(repo=repository)

    action = ViewCode(
        repository=repository
    )

    args = ViewCodeArgs(
        scratch_pad="Adding relevant code spans for the sqlmigrate logic to modify the output_transaction assignment.",
        files=[
            CodeSpan(
                file_path="django/core/management/commands/sqlmigrate.py",
                start_line=0,
                end_line=50,
                span_ids=[
                    "Command",
                    "Command.add_arguments",
                    "Command.handle",
                    "Command.execute",
                    "MigrateTests.test_sqlmigrate_for_non_atomic_migration",
                    "MigrateTests.test_sqlmigrate_forwards",
                    "MigrateTests.test_sqlmigrate_backwards",
                    "BaseDatabaseSchemaEditor.__init__",
                    "BaseDatabaseSchemaEditor.execute",
                    "BaseDatabaseOperations.start_transaction_sql",
                    "BaseDatabaseOperations.end_transaction_sql",
                    "BaseDatabaseOperations.execute_sql_flush",
                    "BaseDatabaseOperations.tablespace_sql",
                ],
            )
        ],
    )

    output = action.execute(args, file_context)

    print(file_context.model_dump())
    print(output.message)
