from typing import List, Optional, Tuple

from code_blocks.codeblocks import CodeBlock, CodeBlockType
from code_blocks.parser import create_parser


def find_next_matching_block(original_blocks: List[CodeBlock],
                             updated_blocks: List[CodeBlock],
                             start_original: int,
                             start_updated: int):
    i = start_original

    next_updated_incomplete = None
    j = start_updated
    while j < len(updated_blocks):
        if not updated_blocks[j].is_complete() and updated_blocks[j].type != CodeBlockType.COMMENTED_OUT_CODE:
            next_updated_incomplete = j
            break
        j += 1

    max_j = len(updated_blocks) if next_updated_incomplete is None else next_updated_incomplete
    while i < len(original_blocks):
        j = start_updated
        while j < max_j:
            if original_blocks[i].content == updated_blocks[j].content:
                return i, j
            j += 1
        i += 1

    # try to find similar block if there are incomplete update blocks
    if next_updated_incomplete:
        similar_block = most_similar_block(original_blocks, updated_blocks[next_updated_incomplete], start_original)
        if similar_block:
            print(f"Will return index for similar block `{updated_blocks[similar_block].content}`")
            return similar_block, next_updated_incomplete

    return len(original_blocks), len(updated_blocks)


def most_similar_block(original_blocks: List[CodeBlock],
                       updated_block: CodeBlock,
                       start_original: int):
    """Naive solution for finding similar blocks."""

    max_similarity = 0
    max_i = None

    i = start_original
    while i < len(original_blocks):
        if original_blocks[i].type == updated_block.type:
            common_chars = sum(
                c1 == c2 for c1, c2 in zip(original_blocks[i].content, updated_block.content))
            if common_chars > max_similarity:
                max_similarity = common_chars
                max_i = i
        i += 1
    return max_i


class CodeMerger:

    def __init__(self, language: str):
        try:
            self.parser = create_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e

    def merge_blocks(self,
                     original_block: CodeBlock,
                     updated_block: CodeBlock,
                     first_level: bool = False) -> Tuple[CodeBlock, List[str]]:
        print(f"Merging block `{original_block.type.value}: {original_block.content}` with "
              f"`{updated_block.type.value}: {updated_block.content}`")

        if first_level and not original_block.has_any_matching_child(updated_block.children, 0):
            matching_block = original_block.find_nested_matching_block(updated_block)
            if matching_block:
                print(
                    f"Found matching children in original block `{original_block.type.value}: {original_block.content}`, "
                    f"will merge with updated children")
                indentation = matching_block.indentation
                updated_block.add_indentation(indentation)

                _, child_tweaks = self.merge_blocks(matching_block.parent, updated_block, first_level=True)
                return original_block, child_tweaks + ["find_nested"]
            else:
                print(
                    f"No matching children in original block `{original_block.type.value}: {original_block.content}`, "
                    f"will replace contents")
                original_block.children = updated_block.children
                return original_block, ["replace"]

        if original_block.has_all_matching_children(updated_block.children, 0) and updated_block.is_complete():
            print(f"All children match, and updated content is complete, will return updated block")
            original_block.children = updated_block.children
            return original_block, []

        gpt_tweaks = []

        i = 0
        j = 0
        while j < len(updated_block.children):
            if i >= len(original_block.children):
                original_block.children.extend(updated_block.children[j:])
                return original_block, []

            original_block_child = original_block.children[i]
            updated_block_child = updated_block.children[j]

            if original_block_child == updated_block_child:
                i += 1
                j += 1
            elif updated_block_child.type == CodeBlockType.COMMENTED_OUT_CODE:
                j += 1
                orig_next, update_next = find_next_matching_block(original_block.children, updated_block.children, i, j)

                i = orig_next
                if update_next > j:
                    original_block.children[i:i] = updated_block.children[j:update_next]
                    i += update_next - j

                j = update_next
                gpt_tweaks.append("commented_out")
            elif (original_block_child.content == updated_block_child.content and
                  original_block_child.children and updated_block_child.children):
                original_block.children[i], child_tweaks = self.merge_blocks(original_block_child, updated_block_child)
                gpt_tweaks.extend(child_tweaks)
                i += 1
                j += 1
            elif original_block_child.content == updated_block_child.content:
                original_block.children[i] = updated_block_child
                i += 1
                j += 1
            elif updated_block_child and original_block:
                # we expect to update a block when the updated block is incomplete and will try the find the most similar block.
                if not updated_block_child.is_complete():
                    similar_original_block = most_similar_block(original_block.children, updated_block_child, i)
                    print(f"Updated block with definition `{updated_block_child.content}` is not complete")
                    if similar_original_block == i:
                        gpt_tweaks.append("replace_similar")

                        print(f"Will replace similar original block definition: `{original_block_child.content}`")
                        original_block.children[i], child_tweaks = self.merge_blocks(original_block_child, updated_block_child)
                        gpt_tweaks.extend(child_tweaks)
                        i += 1
                        j += 1

                        continue
                    else:
                        print(f"Expected most similar original block to be `{original_block_child.content}, "
                              f"but was {original_block.children[similar_original_block].content}`")

                next_original_match = original_block.find_next_matching_child_block(i, updated_block_child)
                next_updated_match = updated_block.find_next_matching_child_block(j, original_block_child)
                next_commented_out = updated_block.find_next_commented_out(j)

                if next_original_match:
                    if first_level:
                        gpt_tweaks.append("next_original_match_keep")
                        # if the there is a match on the first level, we will keep the original blocks until that line
                        i = next_original_match
                    else:
                        gpt_tweaks.append("next_original_match_replace")
                        # if it's not on the first level we expect the blocks to be replaced
                        original_block.children = original_block.children[:i] + original_block.children[next_original_match:]
                elif next_commented_out is not None and (not next_updated_match or next_commented_out < next_updated_match):
                    # if there is commented out code after the updated block, we will insert the lines before in the original block
                    gpt_tweaks.append("next_commented_out_insert")
                    original_block.insert_children(i, updated_block.children[j:next_commented_out])
                    i += next_commented_out - j + 2
                    j = next_commented_out + 1
                elif next_updated_match:
                    # if there is a match in the updated block, we expect this to be an addition and insert the lines before in the original block
                    gpt_tweaks.append("next_original_match_insert")

                    original_block.insert_children(i, updated_block.children[j:next_updated_match])
                    diff = next_updated_match - j
                    i += diff
                    j = next_updated_match
                elif first_level:
                    original_block.insert_child(i, updated_block_child)
                    i += 1
                    j += 1
                else:
                    original_block.children.pop(i)
            else:
                original_block.insert_child(i, updated_block_child)
                j += 1
                i += 1

        return original_block, gpt_tweaks


    def merge(self, original_content: str, updated_content: str):
        original_block = self.parser.parse(original_content)
        updated_block = self.parser.parse(updated_content)
        merged_block, gpt_tweaks = self.merge_blocks(original_block, updated_block, first_level=True)
        return merged_block.to_string(), gpt_tweaks
