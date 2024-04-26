from moatless.codeblocks.module import Module


def find_relevant_spans(original_block: Module, updated_block: Module):
    """Find relevant spans in test content. Used for finding the "perfect" context in benchmark instances."""

    relevant_spans = set()

    for span in updated_block.spans_by_id.values():
        if span.span_id in relevant_spans:
            continue

        if original_block.has_span(span.span_id):
            updated_content = updated_block.to_prompt(
                span_ids=set(span.span_id), show_outcommented_code=False
            ).strip()
            original_content = original_block.to_prompt(
                span_ids=set(span.span_id), show_outcommented_code=False
            ).strip()
            if original_content != updated_content:
                relevant_spans.add(span.span_id)

            # TODO: Second prio after token count
            related_span_ids = original_block.find_related_span_ids(span.span_id)
            relevant_spans.update(related_span_ids)
        else:
            parent_block = updated_block.find_first_by_span_id(span.span_id).parent
            original_parent_block = original_block.find_by_path(
                parent_block.full_path()
            )
            span_ids = list(original_parent_block.belongs_to_span.span_id)

            related_span_ids = updated_block.find_related_span_ids(span.span_id)
            for related_span_id in related_span_ids:
                if original_block.has_span(related_span_id):
                    span_ids.append(related_span_id)

    return relevant_spans
