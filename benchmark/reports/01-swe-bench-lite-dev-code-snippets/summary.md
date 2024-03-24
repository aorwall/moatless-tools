# Benchmark summary

* Dataset: princeton-nlp/SWE-bench_Lite:dev
* Embedding model: text-embedding-3-small
* Splitter: `EpicSplitter(chunk_size=750, min_chunk_size=100, comment_strategy=CommentStrategy.ASSOCIATE)`
* Retrieve strategy: `RetrieveStrategy.CODE_SNIPPETS`

## Instances

| instance_id | no_of_patches | any_found_context_length | all_found_context_length | avg_pos | min_pos | max_pos | top_file_pos | missing_snippets | missing_patch_files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [marshmallow-code__marshmallow-1343](marshmallow-code__marshmallow-1343/report.md) | 1 | 571 | 571 | 4.0 | 2 | 2 | 1 | 2 | 0 |
| [marshmallow-code__marshmallow-1359](marshmallow-code__marshmallow-1359/report.md) | 1 | 17236 | 17236 | 61.0 | 61 | 61 | 3 | 1 | 0 |
| [pvlib__pvlib-python-1072](pvlib__pvlib-python-1072/report.md) | 1 | 1493 | 1493 | 2.0 | 2 | 2 | 1 | 1 | 0 |
| [pvlib__pvlib-python-1154](pvlib__pvlib-python-1154/report.md) | 1 | 302 | 302 | 1.0 | 1 | 1 | 1 | 1 | 0 |
| [pvlib__pvlib-python-1606](pvlib__pvlib-python-1606/report.md) | 1 |  |  | 0.0 |  |  |  | 3 | 1 |
| [pvlib__pvlib-python-1707](pvlib__pvlib-python-1707/report.md) | 1 | 1563 | 1563 | 4.0 | 2 | 2 | 1 | 2 | 0 |
| [pvlib__pvlib-python-1854](pvlib__pvlib-python-1854/report.md) | 1 | 425 | 2234 | 15.0 | 1 | 8 | 1 | 2 | 0 |
| [pydicom__pydicom-1139](pydicom__pydicom-1139/report.md) | 1 | 457 | 20728 | 122.0 | 2 | 54 | 2 | 2 | 0 |
| [pydicom__pydicom-1256](pydicom__pydicom-1256/report.md) | 1 |  |  | 0.0 |  |  | 2 | 1 | 0 |
| [pydicom__pydicom-1413](pydicom__pydicom-1413/report.md) | 1 |  |  | 0.0 |  |  | 1 | 1 | 0 |
| [pydicom__pydicom-1694](pydicom__pydicom-1694/report.md) | 1 | 13855 | 13855 | 29.0 | 29 | 29 | 2 | 1 | 0 |
| [pydicom__pydicom-901](pydicom__pydicom-901/report.md) | 1 | 296 |  | 3.0 | 1 | 2 | 1 | 3 | 0 |
| [pylint-dev__astroid-1196](pylint-dev__astroid-1196/report.md) | 1 | 468 | 468 | 2.0 | 2 | 2 | 2 | 1 | 0 |
| [pylint-dev__astroid-1268](pylint-dev__astroid-1268/report.md) | 1 | 640 |  | 171.0 | 1 | 33 | 1 | 2 | 0 |
| [pylint-dev__astroid-1333](pylint-dev__astroid-1333/report.md) | 1 | 833 | 21818 | 101.0 | 2 | 76 | 2 | 2 | 0 |
| [pylint-dev__astroid-1866](pylint-dev__astroid-1866/report.md) | 1 | 17428 | 17428 | 62.0 | 62 | 62 | 3 | 1 | 0 |
| [pylint-dev__astroid-1978](pylint-dev__astroid-1978/report.md) | 1 | 3295 | 11519 | 58.0 | 10 | 38 | 9 | 3 | 0 |
| [pyvista__pyvista-4315](pyvista__pyvista-4315/report.md) | 1 | 7205 | 7205 | 15.0 | 15 | 15 | 1 | 1 | 0 |
| [sqlfluff__sqlfluff-1517](sqlfluff__sqlfluff-1517/report.md) | 1 |  |  | 0.0 |  |  |  | 2 | 1 |
| [sqlfluff__sqlfluff-1625](sqlfluff__sqlfluff-1625/report.md) | 1 | 948 | 948 | 2.0 | 2 | 2 | 1 | 1 | 0 |
| [sqlfluff__sqlfluff-1733](sqlfluff__sqlfluff-1733/report.md) | 1 |  |  | 0.0 |  |  |  | 1 | 1 |
| [sqlfluff__sqlfluff-1763](sqlfluff__sqlfluff-1763/report.md) | 1 |  |  | 0.0 |  |  |  | 2 | 1 |
| [sqlfluff__sqlfluff-2419](sqlfluff__sqlfluff-2419/report.md) | 1 |  |  | 0.0 |  |  | 1 | 1 | 0 |
# Recall

|     | 13k | 27k | 50k |
| --- | --- | --- | --- |
| All | 56.52% | 69.57% | 69.57% |
