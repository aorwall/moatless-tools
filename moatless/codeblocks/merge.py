import logging

from moatless.codeblocks import CodeBlock
from moatless.codeblocks.codeblocks import MergeAction, NON_CODE_BLOCKS, CodeBlockType

logger = logging.getLogger(__name__)

class CodeMerger:

    def merge(self, original_block: CodeBlock, updated_block: CodeBlock):
        logger.debug(f"Merging block `{original_block.type.value}: {original_block.identifier}` ({len(original_block.children)} children) with "
                     f"`{updated_block.type.value}: {updated_block.identifier}` ({len(updated_block.children)} children)")

        # If there are no matching child blocks on root level expect separate blocks to update on other levels
        has_identifier = any(child.identifier for child in original_block.children)
        no_matching_identifiers = has_identifier and not original_block.has_any_matching_child(updated_block.children)
        if no_matching_identifiers:
            update_pairs = original_block.find_nested_matching_pairs(updated_block)
            if update_pairs:
                for original_child, updated_child in update_pairs:
                    original_indentation_length = len(original_child.indentation) + len(original_block.indentation)
                    updated_indentation_length = len(updated_child.indentation) + len(updated_block.indentation)
                    if original_indentation_length > updated_indentation_length:
                        additional_indentation = ' ' * (original_indentation_length - updated_indentation_length)
                        updated_child.add_indentation(additional_indentation)

                    original_block.merge_history.append(
                        MergeAction(action="find_nested_block", original_block=original_child, updated_block=updated_child))
                    original_child._merge(updated_child)
                return

            raise ValueError(f"Didn't find matching blocks in `{original_block.identifier}``")
        else:
            original_block._merge(updated_block)


    def _merge(original_block, updated_block: "CodeBlock"):
        logging.debug(f"Merging block `{original_block.type.value}: {original_block.identifier}` ({len(original_block.children)} children) with "
                      f"`{updated_block.type.value}: {updated_block.identifier}` ({len(updated_block.children)} children)")

        # Just replace if there are no code blocks in original block
        if len(original_block.children) == 0 or all(child.type in NON_CODE_BLOCKS for child in original_block.children):
            original_block.children = updated_block.children
            original_block.merge_history.append(MergeAction(action="replace_non_code_blocks"))

        # Find and replace if all children are matching
        update_pairs = original_block.find_matching_pairs(updated_block)
        if update_pairs:
            original_block.merge_history.append(
                MergeAction(action="all_children_match", original_block=original_block, updated_block=updated_block))

            for original_child, updated_child in update_pairs:
                original_child._merge(updated_child)

            return

        # Replace if block is complete
        if updated_block.is_complete():
            original_block.children = updated_block.children
            original_block.merge_history.append(
                MergeAction(action="replace_complete", original_block=self, updated_block=updated_block))

        original_block._merge_block_by_block(updated_block)


    def _merge_block_by_block(original_block, updated_block: "CodeBlock"):
        i = 0
        j = 0
        while j < len(updated_block.children):
            if i >= len(original_block.children):
                original_block.children.extend(updated_block.children[j:])
                return

            original_block_child = original_block.children[i]
            updated_block_child = updated_block.children[j]

            if original_block_child == updated_block_child:
                original_block_child.merge_history.append(MergeAction(action="is_same"))
                i += 1
                j += 1
            elif updated_block_child.type == CodeBlockType.COMMENTED_OUT_CODE:
                j += 1
                orig_next, update_next = original_block.find_next_matching_block(updated_block, i, j)

                for commented_out_child in original_block.children[i:orig_next]:
                    commented_out_child.merge_history.append(
                        MergeAction(action="commented_out", original_block=commented_out_child, updated_block=None))

                i = orig_next
                if update_next > j:
                    #  Clean up commented out code at the end
                    last_updated_child = updated_block.children[update_next - 1]
                    if last_updated_child.type == CodeBlockType.COMMENTED_OUT_CODE:
                        update_next -= 1

                    original_block.children[i:i] = updated_block.children[j:update_next]
                    i += update_next - j

                j = update_next
            elif (original_block_child.content == updated_block_child.content and
                  original_block_child.children and updated_block_child.children):
                original_block_child._merge(updated_block_child)
                i += 1
                j += 1
            elif original_block_child.content == updated_block_child.content:
                original_block.children[i] = updated_block_child
                i += 1
                j += 1
            elif updated_block_child:
                # we expect to update a block when the updated block is incomplete
                # and will try the find the most similar block.
                if not updated_block_child.is_complete():
                    similar_original_block = original_block.most_similar_block(updated_block_child, i)
                    logging.debug(f"Updated block with definition `{updated_block_child.content}` is not complete")
                    if similar_original_block == i:
                        original_block.merge_history.append(
                            MergeAction(action="replace_similar", original_block=original_block_child,
                                        updated_block=updated_block_child))

                        original_block_child = CodeBlock(
                            content=updated_block_child.content,
                            identifier=updated_block_child.identifier,
                            pre_code=updated_block_child.pre_code,
                            type=updated_block_child.type,
                            parent=original_block.parent,
                            children=original_block_child.children
                        )

                        original_block.children[i] = original_block_child

                        logging.debug(
                            f"Will replace similar original block definition: `{original_block_child.content}`")
                        original_block_child._merge(updated_block_child)
                        i += 1
                        j += 1

                        continue
                    elif not similar_original_block:
                        logging.debug(f"No most similar original block found to `{original_block_child.content}")
                    else:
                        logging.debug(f"Expected most similar original block to be `{original_block_child.content}, "
                                      f"but was {original_block.children[similar_original_block].content}`")

                next_original_match = original_block.find_next_matching_child_block(i, updated_block_child)
                next_updated_match = updated_block.find_next_matching_child_block(j, original_block_child)
                next_commented_out = updated_block.find_next_commented_out(j)

                if next_original_match:
                    original_block.merge_history.append(
                        MergeAction(action="next_original_match_replace", original_block=original_block.children[next_original_match],
                                    updated_block=updated_block_child))

                    # if it's not on the first level we expect the blocks to be replaced
                    original_block.children = original_block.children[:i] + original_block.children[next_original_match:]
                elif next_commented_out is not None and (
                        not next_updated_match or next_commented_out < next_updated_match):
                    # if there is commented out code after the updated block,
                    # we will insert the lines before the commented out block in the original block
                    original_block.merge_history.append(
                        MergeAction(action="next_commented_out_insert",
                                    original_block=original_block_child,
                                    updated_block=updated_block.children[next_commented_out]))

                    original_block.insert_children(i, updated_block.children[j:next_commented_out])
                    i += next_commented_out - j
                    j = next_commented_out
                elif next_updated_match:
                    # if there is a match in the updated block, we expect this to be an addition
                    # and insert the lines before in the original block
                    original_block.merge_history.append(
                        MergeAction(action="next_original_match_insert",
                                    original_block=original_block_child,
                                    updated_block=updated_block.children[next_updated_match]))

                    original_block.insert_children(i, updated_block.children[j:next_updated_match])
                    diff = next_updated_match - j
                    i += diff
                    j = next_updated_match
                else:
                    original_block.children.pop(i)
            else:
                original_block.insert_child(i, updated_block_child)
                j += 1
                i += 1
