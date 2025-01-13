from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup


def test_find_by_line_numbers():
    instance = get_moatless_instance("scikit-learn__scikit-learn-25570", split="lite")
    repository = create_repository(instance)

    file = repository.get_file(
        "sklearn/compose/tests/test_column_transformer.py"
    )

    empty_space = file.module.find_first_by_start_line(475)
    block_before = file.module.find_first_by_start_line(474)
    block_after = file.module.find_first_by_start_line(477)

    assert not empty_space
    assert block_before
    assert (
        block_before.path_string()
        == "test_column_transformer_sparse_stacking.assert_array_equal_X_tran_3"
    )
    assert block_after
    assert block_after.identifier == "test_column_transformer_mixed_cols_sparse"

    parent_structure_block = block_before.find_type_group_in_parents(
        CodeBlockTypeGroup.STRUCTURE
    )
    assert parent_structure_block
    assert (
        parent_structure_block.path_string()
        == "test_column_transformer_sparse_stacking"
    )

    blocks = file.module.find_blocks_by_line_numbers(475, 475)
    assert len(blocks) == 0

    blocks = file.module.find_blocks_by_line_numbers(475, 477)
    assert len(blocks) == 1
    assert blocks[0].path_string() == "test_column_transformer_mixed_cols_sparse"

    blocks = file.module.find_blocks_by_line_numbers(474, 478)
    assert len(blocks) == 4
