import logging
from typing import List, Optional

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.schema import ActionArguments, Observation
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class APIEndpoint(ActionArguments):
    """Define a single API endpoint in the specification"""

    path: str = Field(..., description="The URL path for the endpoint (e.g., /users/{id})")
    method: str = Field(..., description="HTTP method (GET, POST, PUT, DELETE, etc.)")
    summary: str = Field(..., description="Brief description of the endpoint")
    description: Optional[str] = Field(None, description="Detailed description of the endpoint")
    request_body: Optional[str] = Field(None, description="Description of the request body if applicable")
    response: str = Field(..., description="Description of the expected response")
    status_codes: List[str] = Field(
        default_factory=list, description="List of possible status codes with descriptions"
    )


class DefineAPIArgs(ActionArguments):
    """
    Define a lightweight API specification that is easy to read and understand.
    This provides a simplified version of OpenAPI specifications with focus on 
    human readability and reduced cognitive load.
    """

    title: str = Field(..., description="Title of the API")
    description: str = Field(..., description="Description of the API's purpose and functionality")
    version: str = Field(..., description="API version (e.g., '1.0.0')")
    base_path: Optional[str] = Field(None, description="Base path for all endpoints (e.g., /api/v1)")
    endpoints: List[APIEndpoint] = Field(..., description="List of API endpoints to define")

    model_config = ConfigDict(title="DefineAPI")

    def format_args_for_llm(self) -> str:
        return f"Title: {self.title}\nDescription: {self.description}\nVersion: {self.version}\nBase Path: {self.base_path or '/'}\nEndpoints: {len(self.endpoints)}"

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create an API specification for a todo list application",
                action=DefineAPIArgs(
                    thoughts="Creating a simple API spec for a todo list app with basic CRUD operations",
                    title="Todo List API",
                    description="API for managing todo items in a simple task management application",
                    version="1.0.0",
                    base_path="/api/v1",
                    endpoints=[
                        APIEndpoint(
                            path="/todos",
                            method="GET",
                            summary="List all todos",
                            description="Returns a list of all todo items",
                            response="Array of todo objects",
                            status_codes=["200: Success", "401: Unauthorized"]
                        ),
                        APIEndpoint(
                            path="/todos/{id}",
                            method="GET",
                            summary="Get a single todo",
                            description="Returns a single todo item by its ID",
                            response="Todo object",
                            status_codes=["200: Success", "404: Not found", "401: Unauthorized"]
                        ),
                        APIEndpoint(
                            path="/todos",
                            method="POST",
                            summary="Create a new todo",
                            description="Creates a new todo item",
                            request_body="Todo object without ID",
                            response="Created todo object with ID",
                            status_codes=["201: Created", "400: Bad request", "401: Unauthorized"]
                        )
                    ]
                ),
            ),
            FewShotExample.create(
                user_input="Define an API for user authentication",
                action=DefineAPIArgs(
                    thoughts="Creating an authentication API with login, register, and profile endpoints",
                    title="User Authentication API",
                    description="API for user registration, authentication, and profile management",
                    version="1.0.0",
                    base_path="/auth",
                    endpoints=[
                        APIEndpoint(
                            path="/register",
                            method="POST",
                            summary="Register a new user",
                            request_body="User registration details (username, email, password)",
                            response="User object with token",
                            status_codes=["201: User created", "400: Invalid input"]
                        ),
                        APIEndpoint(
                            path="/login",
                            method="POST",
                            summary="Login existing user",
                            request_body="Login credentials (username/email, password)",
                            response="Authentication token and user details",
                            status_codes=["200: Success", "401: Invalid credentials"]
                        )
                    ]
                ),
            )
        ]


class DefineAPI(Action):
    """
    Action to define a lightweight API specification that is easy to read and understand.
    This creates a human-friendly API definition.
    """

    args_schema = DefineAPIArgs

    async def execute(
        self,
        args: DefineAPIArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        # Build the formatted API specification
        api_spec = self._build_api_spec(args)
        
        # Return the formatted API specification as an observation
        return Observation.create(
            message=api_spec
        )
    
    def _build_api_spec(self, args: DefineAPIArgs) -> str:
        """Build a readable text representation of the API specification"""
        lines = [
            f"# {args.title}",
            "",
            f"{args.description}",
            "",
            f"Version: {args.version}",
            ""
        ]
        
        if args.base_path:
            lines.extend([f"Base Path: {args.base_path}", ""])
        
        lines.append("ENDPOINTS")
        lines.append("=========")
        lines.append("")
        
        # Add each endpoint
        for endpoint in args.endpoints:
            # Endpoint header with method and path
            lines.append(f"{endpoint.method} {endpoint.path}")
            lines.append("-" * len(f"{endpoint.method} {endpoint.path}"))
            lines.append("")
            
            # Summary and description
            lines.append(f"Summary: {endpoint.summary}")
            lines.append("")
            
            if endpoint.description:
                lines.append(f"Description: {endpoint.description}")
                lines.append("")
            
            # Request body if applicable
            if endpoint.request_body:
                lines.append("Request:")
                lines.append(f"  {endpoint.request_body}")
                lines.append("")
            
            # Response
            lines.append("Response:")
            lines.append(f"  {endpoint.response}")
            lines.append("")
            
            # Status codes
            if endpoint.status_codes:
                lines.append("Status Codes:")
                for status_code in endpoint.status_codes:
                    lines.append(f"  - {status_code}")
                lines.append("")
            
            # Add separator between endpoints
            lines.append("")
        
        return "\n".join(lines) 