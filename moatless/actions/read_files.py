import logging
from typing import Optional, List, Dict

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext
from moatless.environment.local import LocalBashEnvironment

logger = logging.getLogger(__name__)


class ReadFilesArgs(ActionArguments):
    """Read multiple files that match a glob pattern.
    
    This action allows you to find and read the contents of multiple files matching a pattern.
    It combines glob-style pattern matching with file reading in a single operation.
    
    The action will find files using the provided glob pattern and then read the contents of each file.
    It's useful for examining multiple related files in one go, such as all JavaScript files in a directory 
    or all configuration files in a project.
    
    Example usage:
    - Read all Python files: {"glob_pattern": "**/*.py"}
    - Read all JavaScript files in src directory: {"glob_pattern": "src/**/*.js"}
    - Read specific config files: {"glob_pattern": "config/*.{json,yaml}"}
    """

    glob_pattern: str = Field(
        ...,
        description="The glob pattern to match files (e.g. '**/*.js', 'src/**/*.ts', 'config/*.{json,yaml}')"
    )
    max_files: int = Field(
        5,
        description="Maximum number of files to read (default: 5). Set to 0 to read all matched files."
    )
    max_lines_per_file: int = Field(
        100,
        description="Maximum number of lines to read from each file (default: 100). Set to 0 to read entire files."
    )
    
    
    model_config = ConfigDict(title="ReadFiles")

    @property
    def log_name(self):
        return f"ReadFiles({self.glob_pattern}, max_files={self.max_files})"

    def to_prompt(self):
        return f"Read files matching glob pattern '{self.glob_pattern}'"

    def short_summary(self) -> str:
        param_str = f"glob_pattern='{self.glob_pattern}'"
        if self.max_files != 5:
            param_str += f", max_files={self.max_files}"
        if self.max_lines_per_file != 100:
            param_str += f", max_lines_per_file={self.max_lines_per_file}"
        return f"{self.name}({param_str})"


class ReadFiles(Action):
    """An action that finds files matching a glob pattern and reads their contents.
    
    This action combines the functionality of GlobTool and ReadFile:
    1. First, it finds all files matching the specified glob pattern
    2. Then, it reads the content of each matched file
    
    It uses the LocalBashEnvironment to execute the file search and reading operations.
    """
    
    args_schema = ReadFilesArgs
    
    async def execute(
        self,
        args: ActionArguments,
        file_context: FileContext | None = None,
    ) -> Observation:
        # Cast the args to the correct type
        if not isinstance(args, ReadFilesArgs):
            raise ValueError(f"Expected ReadFilesArgs, got {type(args)}")
        read_files_args = args
        
        if file_context is None:
            raise ValueError("File context must be provided to execute the read files action.")

        if not self._workspace:
            raise RuntimeError("No workspace set")

        # Use the environment from the workspace if available, or create a new one
        if self.workspace.environment and isinstance(self.workspace.environment, LocalBashEnvironment):
            local_env = self.workspace.environment
        else:
            # Create a new LocalBashEnvironment with the repository path (if available)
            repo_path = getattr(self._repository, "repo_path", None)
            local_env = LocalBashEnvironment(cwd=repo_path)
        
        try:
            # First, use find command to get matching files
            find_command = f"find . -type f -path './{read_files_args.glob_pattern}' | sort"
            
            try:
                output = await local_env.execute(find_command)
                matching_files = [f.strip()[2:] for f in output.strip().split('\n') if f.strip()]
                
                # Special handling for **/* pattern to include files at all levels
                if "**" in read_files_args.glob_pattern:
                    # Get root path
                    base_path = read_files_args.glob_pattern.split("**/")[0]
                    # Get extension if any
                    extension = ""
                    if "." in read_files_args.glob_pattern.split("**/")[1]:
                        extension = read_files_args.glob_pattern.split("**/")[1].split("*")[1]
                    
                    # Create a pattern for root files only - explicitly exclude subdirectories
                    root_pattern = f"{base_path}*{extension}"
                    
                    # Fix pattern to only find files directly in the root directory
                    # We need to exclude any paths that contain additional directories after the base_path
                    # This can be done by using -not -path with a pattern that matches files in subdirectories
                    directory_parts = base_path.rstrip('/').split('/')
                    parent_dir = directory_parts[-1] if directory_parts else ""
                    
                    # Using maxdepth to only look at direct files in the specified directory
                    root_find_command = f"find . -maxdepth {len(directory_parts) + 1} -type f -path './{root_pattern}' | sort"
                    
                    try:
                        root_output = await local_env.execute(root_find_command)
                        root_matching_files = [f.strip()[2:] for f in root_output.strip().split('\n') if f.strip()]
                        
                        # Add files not already included
                        root_only_files = []
                        for file in root_matching_files:
                            # Only add files that are directly in the target directory (not in subdirectories)
                            file_parts = file.split('/')
                            if len(file_parts) == len(directory_parts) + 1:  # +1 for the filename itself
                                if file not in matching_files:
                                    matching_files.append(file)
                                    root_only_files.append(file)
                                    
                        # Log what we're doing for debugging
                        if root_only_files:
                            logger.info(f"Added {len(root_only_files)} root files to match pattern '{read_files_args.glob_pattern}': {root_only_files}")
                    except Exception as e:
                        logger.warning(f"Error finding root files: {str(e)}")
                
            except Exception as e:
                # If find fails, try using the repository's matching_files method if available
                if hasattr(self._repository, "matching_files"):
                    matching_files = await getattr(self._repository, "matching_files")(read_files_args.glob_pattern)
                else:
                    return Observation.create(
                        message=f"Error finding files: {str(e)}",
                        summary=f"Error: {str(e)}"
                    )
            
            if not matching_files:
                return Observation.create(
                    message=f"No files found matching glob pattern '{read_files_args.glob_pattern}'",
                    summary=f"No files found matching '{read_files_args.glob_pattern}'"
                )
            
            # Generate a list of files for the summary
            file_list_summary = f"Found {len(matching_files)} files matching '{read_files_args.glob_pattern}':"
            for file_path in matching_files[:min(len(matching_files), read_files_args.max_files if read_files_args.max_files > 0 else len(matching_files))]:
                file_list_summary += f"\n- {file_path}"
            
            if read_files_args.max_files > 0 and len(matching_files) > read_files_args.max_files:
                truncated = len(matching_files) - read_files_args.max_files
                file_list_summary += f"\n(+ {truncated} more files not shown)"
            
            # Apply max_files limit if set
            if read_files_args.max_files > 0 and len(matching_files) > read_files_args.max_files:
                truncated = len(matching_files) - read_files_args.max_files
                matching_files = matching_files[:read_files_args.max_files]
                truncation_note = f"\n\nNote: Found {truncated} additional files matching the pattern, but limited to {read_files_args.max_files} files."
            else:
                truncation_note = ""
            
            # Read each file's content
            result = []
            for file_path in matching_files:
                try:
                    # Try to read file content using environment
                    content = await local_env.read_file(file_path)
                    
                    # Apply line limit if set
                    if read_files_args.max_lines_per_file > 0:
                        lines = content.split('\n')
                        if len(lines) > read_files_args.max_lines_per_file:
                            content = '\n'.join(lines[:read_files_args.max_lines_per_file])
                            content += f"\n\n... (truncated at {read_files_args.max_lines_per_file} lines)"
                    
                    # Add to file context
                    #file_context.add_file(file_path)
                    #context_file = file_context.get_file(file_path)
                    #if context_file:
                        # Add the line span to include the whole file in context
                    #    file_context.add_line_span_to_context(
                    #        file_path,
                    #        1,  # Start from line 1
                    #        len(content.split('\n'))  # End at the last line
                    #    )
                    
                    # Add to results
                    result.append(f"### File: {file_path}\n```\n{content}\n```\n")
                except Exception as e:
                    result.append(f"### File: {file_path}\nError reading file: {str(e)}\n")
            
            # Combine results
            message = f"Found {len(matching_files)} files matching glob pattern '{read_files_args.glob_pattern}':{truncation_note}\n\n"
            message += "\n".join(result)
            
            return Observation.create(
                message=message,
                summary=file_list_summary
            )
            
        except Exception as e:
            return Observation.create(
                message=f"Error executing read files action: {str(e)}",
                summary=f"Error: {str(e)}"
            ) 