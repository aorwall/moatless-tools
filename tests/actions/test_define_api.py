import pytest
from moatless.actions.define_api import DefineAPI, DefineAPIArgs, APIEndpoint
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


@pytest.fixture
def repository():
    return InMemRepository()


@pytest.fixture
def workspace(repository):
    return Workspace(repository=repository)


@pytest.fixture
def file_context(repository):
    return FileContext(repo=repository)


@pytest.fixture
def api_args():
    return DefineAPIArgs(
        title="Test API",
        description="API for testing the DefineAPI action",
        version="1.0.0",
        base_path="/api/v1",
        endpoints=[
            APIEndpoint(
                path="/items",
                method="GET",
                summary="List all items",
                description="Returns a list of all items",
                response="Array of item objects",
                status_codes=["200: Success", "401: Unauthorized"]
            ),
            APIEndpoint(
                path="/items/{id}",
                method="GET",
                summary="Get a single item",
                description="Returns a single item by its ID",
                response="Item object",
                status_codes=["200: Success", "404: Not found"]
            )
        ]
    )


@pytest.mark.asyncio
async def test_define_api_basic(repository, workspace, file_context, api_args):
    action = DefineAPI()
    await action.initialize(workspace)

    observation = await action.execute(api_args, file_context)

    # Check the observation message (returned API spec)
    content = observation.message
    
    # Verify key elements are present
    assert "# Test API" in content
    assert "Version: 1.0.0" in content
    assert "Base Path: /api/v1" in content
    assert "GET /items" in content
    assert "GET /items/{id}" in content
    assert "Array of item objects" in content
    assert "- 200: Success" in content


@pytest.mark.asyncio
async def test_define_api_no_base_path(repository, workspace, file_context):
    # Create args without base path
    args = DefineAPIArgs(
        title="API Without Base Path",
        description="API specification without a base path",
        version="1.0.0",
        endpoints=[
            APIEndpoint(
                path="/users",
                method="GET",
                summary="List users",
                response="Array of user objects"
            )
        ]
    )
    
    action = DefineAPI()
    await action.initialize(workspace)
    observation = await action.execute(args, file_context)
    
    content = observation.message
    
    # Base path should not appear in content
    assert "Base Path:" not in content
    assert "API Without Base Path" in content


@pytest.mark.asyncio
async def test_define_api_with_request_body(repository, workspace, file_context):
    args = DefineAPIArgs(
        title="API With Request Body",
        description="API with endpoints that have request bodies",
        version="1.0.0",
        base_path="/api/v2",
        endpoints=[
            APIEndpoint(
                path="/users",
                method="POST",
                summary="Create user",
                request_body="User object with name and email fields",
                response="Created user object with ID",
                status_codes=["201: Created", "400: Bad request"]
            )
        ]
    )
    
    action = DefineAPI()
    await action.initialize(workspace)
    observation = await action.execute(args, file_context)
    
    content = observation.message
    
    # Check for request body section
    assert "Request:" in content
    assert "User object with name and email fields" in content 