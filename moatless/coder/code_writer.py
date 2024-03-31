import difflib
import logging
import re
from typing import List
from typing import Optional

from pydantic import BaseModel, Field

from moatless.codeblocks import create_parser, CodeBlockType

logger = logging.getLogger(__name__)


def do_diff(file_path: str, original_content: str, updated_content: str) -> Optional[str]:
    return "".join(difflib.unified_diff(
        original_content.strip().splitlines(True),
        updated_content.strip().splitlines(True),
        fromfile=file_path, tofile=file_path, lineterm="\n"))


comment_marker_pattern = r"^[ \t]*(//|#|--|<!--)\s*"

incomplete_code_marker_patterns = [
    r"\(?(rest of|existing code|other code)",
    r"\.\.\.\s*\(?rest of code|existing code|other code\)?\s*",
    r"^\.\.\.\s*"
]


def is_complete(content: str):
    lines = content.split('\n')

    for i, line in enumerate(lines, start=1):
        if re.search(comment_marker_pattern, line):
            rest_of_line = re.sub(comment_marker_pattern, '', line)

            if any(re.search(pattern, rest_of_line, re.DOTALL | re.IGNORECASE) for pattern in
                   incomplete_code_marker_patterns):
                print("Not complete: Matched marker on line {}: {}".format(i, line))
                return False
    return True


class FileItem(BaseModel):
    file_path: str = Field(description="Path to the file")
    language: str = Field(description="Language of the file", default="python")
    content: str = Field(description="Contents of the file")
    readonly: bool = Field(default=False, description="If the file is readonly")


class WriteCodeRequest(BaseModel):
    updated_files: List[FileItem] = Field(description="Files to update")
    file_context: list[FileItem] = Field(default=[], description="Files in context")


class UpdatedFileItem(BaseModel):
    file_path: Optional[str] = Field(default=None, description="file to update or create")
    new_updates: Optional[str] = Field(default=None, description="provided changes")
    diff: Optional[str] = Field(default=None, description="diff of the file")
    content: Optional[str] = Field(default=None)
    invalid: Optional[str] = Field(default=None, description="file is invalid")
    created: bool = Field(default=False, description="file is created")


class WriteCodeResponse(BaseModel):
    updated_files: List[UpdatedFileItem] = Field(description="Updated files")


class CodeWriter:

    def __init__(self,
                 allow_hallucinated_files: bool = False,
                 expect_complete_functions: bool = True,
                 guess_similar_files: bool = True,
                 remove_new_comments: bool = True,
                 debug_mode: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.guess_similar_files = guess_similar_files
        self.expect_complete_functions = expect_complete_functions
        self.allow_hallucinated_files = allow_hallucinated_files
        self.remove_new_comments = remove_new_comments
        self.debug_mode = debug_mode

    def write_code(self, request: WriteCodeRequest) -> WriteCodeResponse:
        updated_files = [self._update_file(file_item, request.file_context) for file_item in request.updated_files]
        return WriteCodeResponse(updated_files=updated_files)

    def _update_file(self, update_file: FileItem, file_context: List[FileItem] = [], new_file: bool = False) -> UpdatedFileItem:
        existing_file = self._find_existing_file(update_file.file_path, file_context)

        if new_file:
            if existing_file:
                return UpdatedFileItem(file_path=update_file.file_path, invalid=f"An existing file found.")

            updated_file = UpdatedFileItem(file_path=update_file.file_path, content=update_file.content)
        else:
            if not existing_file:
                return UpdatedFileItem(file_path=update_file.file_path, invalid=f"No existing file found.")

            updated_file = self.update_file(new_updates=update_file.content,
                                            file_path=update_file.file_path,
                                            existing_file_item=existing_file)

        logger.info(f"Updated code diff:\n{updated_file.diff}")

        if self.debug_mode:
            debug_text_after = f"\n\n> _Updated file:_\n> {updated_file.file_path}\n```diff\n{updated_file.diff}\n```"
            logger.debug(debug_text_after)

        return updated_file

    def _find_existing_file(self, file_path: str, context_files: List[FileItem]) -> Optional[FileItem]:
        if file_path.startswith("/"):
            file_path = file_path[1:]

        ends_with_file_path = []
        for file_item in context_files:
            if file_item.file_path == file_path:
                return file_item
            elif file_item.file_path.endswith(file_path):
                ends_with_file_path.append(file_item)

        if len(ends_with_file_path) == 1:
            return ends_with_file_path[0]

        return None

    def find_matching_file_item(self,
                                code: str,
                                file_path: str = None,
                                language: str = None,
                                context_files: List[FileItem] = []):
        """Tries to find a file item that matches the code block."""

        if file_path:
            equal_match = [item for item in context_files if item.file_path == file_path]
            if equal_match:
                logger.debug(f"Found equal match for {file_path}: {equal_match[0].file_path}")
                return equal_match[0]

            similar_matches = [item for item in context_files if "file" in item.type and file_path in item.file_path]
            if len(similar_matches) == 1:
                logger.debug(f"Found similar match for {file_path}: {similar_matches[0].file_path}")
                return similar_matches[0]
            elif len(similar_matches) > 1:
                logger.debug(
                    f"Found {len(similar_matches)} similar matches for {file_path}: {similar_matches}, expected one")
                context_files = similar_matches

        if not self.guess_similar_files:
            file_names = ", ".join([item.file_path for item in context_files])
            logger.debug(f"Could not find a matching file item for {file_path} in file context with files {file_names}")
            return None

        for file_item in context_files:
            if not file_item.language:
                continue

            if language and language != file_item.language:
                logger.debug(
                    f"Language in block [{language}] does not match language [{file_item.language}] file path {file_item.file_path}")
                continue

            try:
                parser = create_parser(file_item.language)
                original_block = parser.parse(file_item.content)
                updated_block = parser.parse(code)

                # TODO: Sort by matching blocks first and pick the one with the most matching blocks
                matching_blocks = original_block.get_matching_blocks(updated_block)
                if any(matching_block for matching_block in matching_blocks
                       if matching_block.type in [CodeBlockType.FUNCTION, CodeBlockType.CLASS]):
                    matching_content_str = "\n\n".join([str(block) for block in matching_blocks])
                    self.debug_log(
                        f"Code block has similarities to {file_item.file_path}, will assume it's the "
                        f"updated file.\nSimilarities:\n```\n{matching_content_str}\n```")

                    return file_item
            except Exception as e:
                logger.warning(
                    f"Could not parse file with {file_item.language} or code block with language {language}: {e}")

        if (len(context_files) == 1
                and context_files[0].language == language):
            self.debug_log(f"Found only the file {context_files[0].file_path} with language {language}, "
                           f"will assume it's the updated file.")
            return context_files[0]

    def update_file(self,
                    new_updates: str,
                    file_path: str = None,
                    language: str = None,
                    existing_file_item: Optional[FileItem] = None) -> Optional[UpdatedFileItem]:
        logger.debug(f"update_file: file_path={file_path}, language={language}, "
                     f"existing_file_item={existing_file_item is not None}")

        language = language or existing_file_item.language

        updated_content = new_updates

        if not file_path and existing_file_item:
            file_path = existing_file_item.file_path

        def invalid_update(invalid: Optional[str] = None, updated_content: str = None):
            return UpdatedFileItem(
                file_path=file_path,
                language=language,
                content=updated_content,
                new_updates=new_updates,
                invalid=invalid
            )

        # Don't merge and mark as invalid if the file is readonly
        if existing_file_item and existing_file_item.readonly:
            return invalid_update("readonly")

        # Don't merge and mark as invalid if hallucinated files isn't allowed
        if not existing_file_item and not self.allow_hallucinated_files:
            return invalid_update("hallucination")

        parser = None
        try:
            parser = create_parser(language, apply_gpt_tweaks=True)
        except Exception as e:
            logger.warning(f"Could not create parser for language {language}: {e}")

        if parser:
            try:
                updated_block = parser.parse(updated_content)
            except Exception as e:
                logger.warning(f"Could not parse updated in {file_path}: {e}")
                return invalid_update("Couldn't parse the content, please provide valid code.")

            error_blocks = updated_block.find_errors()
            if error_blocks:
                logger.info("The updated file {} has errors. ".format(file_path))
                error_block_report = "\n\n".join([f"```{block.content}```" for block in error_blocks])
                return invalid_update(f"There are syntax errors in the updated file:\n\n{error_block_report}")

            if self.expect_complete_functions:
                incomplete_blocks = updated_block.find_incomplete_blocks_with_types(
                    [CodeBlockType.CONSTRUCTOR, CodeBlockType.FUNCTION, CodeBlockType.TEST_CASE])
                if incomplete_blocks:
                    incomplete_blocks_str = ", ".join([f"`{block.identifier}`" for block in incomplete_blocks])
                    return invalid_update(f"All code must be provided in the functions. "
                                          f"The content in the functions {incomplete_blocks_str} is not complete.")

            if existing_file_item and not existing_file_item.readonly:
                original_block = parser.parse(existing_file_item.content)

                try:
                    original_block.merge(updated_block)

                    updated_content = original_block.to_string()

                    merged_block = parser.parse(updated_content)
                    error_blocks = merged_block.find_errors()
                    if error_blocks:
                        logger.info("The merged file {} has errors.".format(file_path))
                        return invalid_update("Couldn't merge the updated content to the existing file. Please provide the complete contents of the existing file when updating it.", updated_content=updated_content)

                    if not original_block.is_complete():
                        logger.info(f"The merged content for [{file_path}] is not complete")
                        return invalid_update("Couldn't merge the updated content to the existing file. Please provide the complete contents of the existing file when updating it.",  updated_content=updated_content)

                except Exception as e:
                    logger.info(f"Could not merge {file_path}: {e}")
                    import traceback
                    traceback.print_exception(e)
                    return invalid_update("Couldn't merge the updated content to the existing file. Please provide the complete contents of the existing file when updating it.")
        elif not is_complete(updated_content):
            return invalid_update("No code isn't complete, provide the missing code.")

        diff = None
        if existing_file_item:
            diff = do_diff(file_path=file_path,
                           original_content=existing_file_item.content,
                           updated_content=updated_content)
            if not diff:
                return invalid_update("no_change")

        return UpdatedFileItem(
            file_path=file_path,
            language=language,
            new_updates=new_updates,
            content=updated_content,
            diff=diff)

    def debug_log(self, message):
        if self.debug_mode:
            logger.debug(message)
